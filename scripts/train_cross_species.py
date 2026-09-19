"""大鼠→人受保护跨物种软先验级联训练。

每个目标外层折 f 使用不含 f 的大鼠训练数据。目标外层模型的训练行 g
还使用排除 f 与 g 的大鼠模型预测，避免同分子/同骨架的跨物种泄露。
默认只训练有匹配大鼠体系的 fu、CL、VDss、F、Thalf；不混合物种标签。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from pipeline_common import (ROOT, base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check, verify_stage)
from dmpk_toolkit import (TargetSpace, featurize, fit_model, load_task, metrics, molecule_weights,
                          prediction_table)

SYSTEMS = {'fu': 'plasma', 'CL': 'systemic_iv', 'VDss': 'steady_state_iv',
           'F': 'absolute_oral', 'Thalf': 'terminal_iv'}


def source_config(run_dir: Path, datasets: Path):
    meta = verify_stage(run_dir, 'stl_training')
    if meta.get('test_evaluated'):
        raise ValueError(f'跨物种先验不得使用已评估 test 的上游模型: {run_dir}')
    if meta['inputs']['datasets_complete_sha256'] != sha256(datasets / 'complete.json'):
        raise ValueError('大鼠上游模型与当前数据版本不一致')
    selected = json.loads((run_dir / 'selection.json').read_text())['algorithm']
    bundle = joblib.load(run_dir / selected / 'model.joblib')
    return selected, bundle.parameters, meta


def protected_prior(source, source_x, source_spec, source_names, target, target_x,
                    algorithm, parameters, excluded, args, cache):
    key = tuple(sorted(excluded))
    if key not in cache:
        take = np.flatnonzero(source.split.eq('train') & ~source.fold_id.isin(excluded))
        if source.iloc[take].molecule_id.nunique() < 2:
            raise ValueError(f'排除折 {key} 后大鼠训练分子不足')
        cache[key] = fit_model(source, source_x, take, algorithm, parameters,
                               source_spec, source_names, args)
    return cache[key].predict_features(target_x, target.smiles.to_numpy())


def feature_matrix(target, target_x, source, source_x, source_spec, source_names,
                   source_algorithm, source_parameters, outer_fold, args, cache):
    """每个目标行获得未见其所属折的大鼠预测，随后与结构特征拼接。"""
    prior = np.empty(len(target), dtype=float)
    is_train = target.split.eq('train').to_numpy()
    fold_id = target.fold_id.to_numpy()
    for fold in sorted(set(fold_id[is_train])):
        rows = np.flatnonzero(is_train & (fold_id == fold))
        excluded = {fold} if outer_fold is None else {outer_fold, fold}
        prior[rows] = protected_prior(source, source_x, source_spec, source_names,
                                      target.iloc[rows], target_x[rows], source_algorithm,
                                      source_parameters, excluded, args, cache)
    # val 分子不属于 source train；最终模型可用全部 source train，外层中则排除 f。
    heldout = ~is_train
    if outer_fold is not None:
        heldout |= is_train & (fold_id == outer_fold)
    rows = np.flatnonzero(heldout)
    if len(rows):
        excluded = set() if outer_fold is None else {outer_fold}
        prior[rows] = protected_prior(source, source_x, source_spec, source_names,
                                      target.iloc[rows], target_x[rows], source_algorithm,
                                      source_parameters, excluded, args, cache)
    transform = source_spec['transform']
    if transform == 'log10':
        prior_feature = np.log10(prior)
    elif transform == 'logit':
        clipped = np.clip(prior, np.nextafter(0., 1.), np.nextafter(1., 0.))
        prior_feature = np.log(clipped / (1. - clipped))
    else:
        prior_feature = prior
    return np.hstack([target_x, prior_feature[:, None]]), prior


def fit_cascade(frame, x, indices, algorithm, spec, args):
    weights = molecule_weights(frame.iloc[indices])
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if algorithm == 'ridge':
        steps.append(('scale', StandardScaler()))
    prep = Pipeline(steps).fit(x[indices])
    target = TargetSpace(spec['transform']).fit(frame.iloc[indices].raw_value, weights)
    if algorithm == 'dummy':
        estimator = DummyRegressor(strategy='mean')
    elif algorithm == 'ridge':
        estimator = Ridge(alpha=10., solver='lsqr', tol=1e-6)
    else:
        estimator = RandomForestRegressor(n_estimators=400, max_features=.7, min_samples_leaf=3,
                                          random_state=args.seed, n_jobs=args.threads)
    estimator.fit(prep.transform(x[indices]), target.forward(frame.iloc[indices].raw_value), sample_weight=weights)
    return prep, target, estimator


def predict(prep, target, estimator, x, endpoint):
    return target.inverse(estimator.predict(prep.transform(x)), endpoint)


def run():
    p = base_parser('受保护的大鼠→人跨物种软先验级联训练；默认 test 冻结')
    p.add_argument('--endpoint', required=True, choices=sorted(SYSTEMS))
    p.add_argument('--system', help='人源体系；默认使用端点的预设匹配体系')
    p.add_argument('--rat-system', help='大鼠体系；默认与人源预设体系相同')
    p.add_argument('--datasets-dir', type=Path)
    p.add_argument('--splits-dir', type=Path)
    p.add_argument('--source-run-dir', type=Path, help='已完成的大鼠先验目录')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--run-name', default='cross_species_diagnostic_v1')
    # 供 fit_model 的统一接口使用；本程序的目标层固定为 sklearn 基线。
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    configure_logging(args.root, 'train_cross_species')
    if args.threads < 1 or not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError('threads 必须为正数，run-name 必须是单个目录名')
    system = args.system or SYSTEMS[args.endpoint]
    rat_system = args.rat_system or SYSTEMS[args.endpoint]
    datasets = args.datasets_dir or args.root / 'data/processed_v2/datasets'
    splits = args.splits_dir or args.root / 'data/processed_v2/splits'
    task = f'{args.endpoint}__human__{system}'
    rat_task = f'{args.endpoint}__rat__{rat_system}'
    source_run = args.source_run_dir or args.root / 'models/stl' / rat_task / 'prior_v1'
    output = args.output or args.root / 'models/cascade/cross_species' / task / args.run_name
    startup_self_check(output=output)
    target, target_spec, split_meta = load_task(args.root, datasets, splits, task)
    source, source_spec, _ = load_task(args.root, datasets, splits, rat_task)
    source_algorithm, source_parameters, source_meta = source_config(source_run, datasets)
    if source_meta['task_id'] != rat_task:
        raise ValueError(f'上游任务不匹配: 需要 {rat_task}，实际为 {source_meta["task_id"]}')
    train_idx = np.flatnonzero(target.split.eq('train'))
    folds = sorted(target.iloc[train_idx].fold_id.unique())
    if len(folds) < 3:
        raise ValueError('目标任务固定训练折不足 3，不能建立受保护跨物种级联')
    logging.info('target=%s；训练记录=%d，分子=%d；source=%s (%s)', task, len(train_idx),
                 target.iloc[train_idx].molecule_id.nunique(), rat_task, source_algorithm)
    if args.check_only:
        logging.info('目标/大鼠任务、固定划分、上游模型和输出路径检查通过；尚未训练。')
        return
    target_x, names = featurize(target.smiles.tolist())
    source_x, source_names = featurize(source.smiles.tolist())
    cache = {}
    algorithms = ('dummy', 'ridge', 'rf')
    oof = {name: np.full(len(target), np.nan) for name in algorithms}
    with threadpool_limits(limits=args.threads), stage_output(output) as out:
        for fold in tqdm(folds, desc=f'{task} protected outer folds'):
            x, _ = feature_matrix(target, target_x, source, source_x, source_spec, source_names,
                                  source_algorithm, source_parameters, fold, args, cache)
            fit_idx = train_idx[target.iloc[train_idx].fold_id.to_numpy() != fold]
            valid_idx = train_idx[target.iloc[train_idx].fold_id.to_numpy() == fold]
            for algorithm in algorithms:
                prep, target_space, estimator = fit_cascade(target, x, fit_idx, algorithm, target_spec, args)
                oof[algorithm][valid_idx] = predict(prep, target_space, estimator, x[valid_idx], args.endpoint)
        final_x, final_prior = feature_matrix(target, target_x, source, source_x, source_spec, source_names,
                                              source_algorithm, source_parameters, None, args, cache)
        summary = {}
        val_idx = np.flatnonzero(target.split.eq('val'))
        for algorithm in algorithms:
            folder = out / algorithm
            folder.mkdir()
            prep, target_space, estimator = fit_cascade(target, final_x, train_idx, algorithm, target_spec, args)
            val_prediction = (predict(prep, target_space, estimator, final_x[val_idx], args.endpoint)
                              if len(val_idx) else np.array([], dtype=float))
            summary[algorithm] = {'oof': metrics(target.iloc[train_idx], oof[algorithm][train_idx], target_spec),
                                  'val': metrics(target.iloc[val_idx], val_prediction, target_spec) if len(val_idx) else None}
            prediction_table(target.iloc[train_idx], oof[algorithm][train_idx], target_spec,
                             'protected_cross_species_oof', []).to_csv(folder / 'oof_predictions.csv', index=False)
            if len(val_idx):
                prediction_table(target.iloc[val_idx], val_prediction, target_spec,
                                 'validation', []).to_csv(folder / 'val_predictions.csv', index=False)
            joblib.dump({'preprocessor': prep, 'target': target_space, 'estimator': estimator,
                         'feature_names': names + [f'{rat_task}_protected_prior'],
                         'source_run_dir': str(source_run)}, folder / 'model.joblib')
        selected = min(algorithms, key=lambda name: summary[name]['oof']['primary'])
        dump_json(out / 'metrics.json', summary)
        dump_json(out / 'selection.json', {'algorithm': selected, 'criterion': 'protected_outer_oof_primary',
                  'caveat': 'cross-species prior is a soft feature; test remains frozen'})
        dump_json(out / 'lineage.json', {'target_task': task, 'source_task': rat_task,
                  'source_run_dir': str(source_run), 'source_algorithm': source_algorithm,
                  'protection': 'outer fold and each target training row fold excluded from rat source fitting'})
        finish_stage(out, 'protected_cross_species_cascade',
                     inputs={'datasets_complete_sha256': sha256(datasets / 'complete.json'),
                             'splits_complete_sha256': sha256(splits / 'complete.json'),
                             'source_complete_sha256': sha256(source_run / 'complete.json')},
                     task_id=task, source_task_id=rat_task, test_evaluated=False, partial=False)
    logging.info('跨物种级联产物: %s；受保护 OOF 选定 %s；测试评估=False', output, selected)


if __name__ == '__main__':
    run_cli(run)

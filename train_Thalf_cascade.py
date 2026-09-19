"""受保护的 CL/VDss → 人 Thalf 级联诊断训练。

每个 Thalf 外层折 f 的上游 CL/VDss 模型均排除 f；用于该外层模型
训练行 g 的上游模型还排除 g。因而任何 Thalf 行都不会使用一个见过
同折分子的上游模型预测。val 预测只使用全部上游 train 折拟合的模型。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from pipeline_common import (ROOT, base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check, verify_stage)
from dmpk_toolkit import (ModelBundle, TargetSpace, featurize, load_task, metrics,
                          molecule_weights, primary_score, fit_model)


def selected_source(run_dir: Path):
    """读取已发布上游模型的固定算法/参数，而不读取其预测文件。"""
    meta = verify_stage(run_dir, 'stl_training')
    if meta['test_evaluated']:
        raise ValueError(f'级联构建前不得使用已测试的上游运行: {run_dir}')
    selection = json.loads((run_dir / 'selection.json').read_text())
    algorithm = selection['algorithm']
    bundle = joblib.load(run_dir / algorithm / 'model.joblib')
    return algorithm, bundle.parameters, meta


def source_prediction(source_frame, source_x, source_spec, source_names, target_frame, target_x,
                      algorithm, parameters, excluded_folds, args, cache, label):
    """拟合不含 excluded_folds 的上游模型，向 target_frame 生成物理空间预测。"""
    key = (label, tuple(sorted(excluded_folds)))
    if key not in cache:
        take = np.flatnonzero(source_frame.split.eq('train') & ~source_frame.fold_id.isin(excluded_folds))
        if source_frame.iloc[take].molecule_id.nunique() < 2:
            raise ValueError(f'{label} 排除折 {sorted(excluded_folds)} 后训练分子不足')
        cache[key] = fit_model(source_frame, source_x, take, algorithm, parameters,
                               source_spec, source_names, args)
    model = cache[key]
    return model.predict_features(target_x, target_frame.smiles.to_numpy())


def make_features(thalf_frame, thalf_x, cl, vdss, source_frames, source_x, source_specs,
                  source_names, source_configs, outer_fold, args, cache):
    """为一个 Thalf 外层范围建立结构特征及两项严格受保护先验。"""
    result = np.empty((len(thalf_frame), 2), dtype=np.float64)
    train = thalf_frame.split.eq('train').to_numpy()
    # 外层验证行排除 outer_fold；训练行额外排除各自所属折。
    fold_values = thalf_frame.fold_id.to_numpy()
    groups = {int(f) for f in fold_values[train]}
    for fold in groups:
        rows = np.flatnonzero(train & (fold_values == fold))
        excluded = {fold} if outer_fold is None else {outer_fold, fold}
        result[rows, 0] = source_prediction(*source_frames['CL'], thalf_frame.iloc[rows], thalf_x[rows],
                                             *source_configs['CL'], excluded, args, cache, 'CL')
        result[rows, 1] = source_prediction(*source_frames['VDss'], thalf_frame.iloc[rows], thalf_x[rows],
                                             *source_configs['VDss'], excluded, args, cache, 'VDss')
    # val 或外层验证行不会出现在上游 train 中，使用仅排除 outer_fold 的模型。
    heldout = ~train
    if outer_fold is not None:
        heldout |= train & (fold_values == outer_fold)
    rows = np.flatnonzero(heldout)
    if len(rows):
        excluded = set() if outer_fold is None else {outer_fold}
        result[rows, 0] = source_prediction(*source_frames['CL'], thalf_frame.iloc[rows], thalf_x[rows],
                                             *source_configs['CL'], excluded, args, cache, 'CL')
        result[rows, 1] = source_prediction(*source_frames['VDss'], thalf_frame.iloc[rows], thalf_x[rows],
                                             *source_configs['VDss'], excluded, args, cache, 'VDss')
    # log10(CL) 与 log10(VDss) 已蕴含 0.693*VDss/CL 的守恒关系；保留原预测，
    # 让模型学习残差而不硬编码单位不完全确定的半衰期常数。
    return np.hstack([thalf_x, np.log10(result)])


def fit_cascade(frame, x, indices, algorithm, alpha, args):
    weights = molecule_weights(frame.iloc[indices])
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if algorithm == 'ridge':
        steps.append(('scale', StandardScaler()))
    prep = Pipeline(steps)
    xx = prep.fit_transform(x[indices])
    target = TargetSpace('log10').fit(frame.iloc[indices].raw_value, weights)
    if algorithm == 'ridge':
        estimator = Ridge(alpha=alpha, solver='lsqr', tol=1e-6)
    else:
        estimator = RandomForestRegressor(n_estimators=400, max_features=.7, min_samples_leaf=3,
                                          random_state=args.seed, n_jobs=args.threads)
    estimator.fit(xx, target.forward(frame.iloc[indices].raw_value), sample_weight=weights)
    return prep, target, estimator


def predict(prep, target, estimator, x):
    return target.inverse(estimator.predict(prep.transform(x)), 'Thalf')


def run():
    p = base_parser('CL/VDss 受保护先验的 Thalf 级联诊断；默认不评估 test')
    p.add_argument('--datasets-dir', type=Path)
    p.add_argument('--splits-dir', type=Path)
    p.add_argument('--cl-run-dir', type=Path, default=ROOT / 'models/stl/CL__human__systemic_iv/baseline_v1')
    p.add_argument('--vdss-run-dir', type=Path, default=ROOT / 'models/stl/VDss__human__steady_state_iv/baseline_v1')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    # 上游 fit_model 复用统一估计器接口；此级联当前只使用已发布的
    # sklearn 基线，因此固定 CPU 参数，不开放神经网络训练。
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--run-name', default='cascade_diagnostic_v1')
    args = p.parse_args()
    configure_logging(args.root, 'train_Thalf_cascade')
    if args.threads < 1 or not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError('threads 必须为正数，run-name 必须是单个目录名')
    datasets = args.datasets_dir or args.root / 'data/processed_v2/datasets'
    splits = args.splits_dir or args.root / 'data/processed_v2/splits'
    output = args.output or args.root / 'models/cascade/Thalf__human__terminal_iv' / args.run_name
    startup_self_check(output=output)
    thalf, spec, split_meta = load_task(args.root, datasets, splits, 'Thalf__human__terminal_iv')
    cl, cl_spec, _ = load_task(args.root, datasets, splits, 'CL__human__systemic_iv')
    vdss, vdss_spec, _ = load_task(args.root, datasets, splits, 'VDss__human__steady_state_iv')
    cl_algorithm, cl_params, cl_meta = selected_source(args.cl_run_dir)
    vdss_algorithm, vdss_params, vdss_meta = selected_source(args.vdss_run_dir)
    if cl_meta['inputs']['datasets_complete_sha256'] != sha256(datasets / 'complete.json') or vdss_meta['inputs']['datasets_complete_sha256'] != sha256(datasets / 'complete.json'):
        raise ValueError('上游模型与当前数据版本不一致')
    idx = np.flatnonzero(thalf.split.eq('train'))
    folds = sorted(thalf.iloc[idx].fold_id.unique())
    if len(folds) < 3:
        raise ValueError('Thalf 训练折不足，不能建立受保护级联')
    logging.info('Thalf 训练记录=%d，分子=%d，folds=%s；CL=%s，VDss=%s',
                 len(idx), thalf.iloc[idx].molecule_id.nunique(), folds, cl_algorithm, vdss_algorithm)
    if args.check_only:
        logging.info('数据、统一划分及上游模型契约检查通过；尚未训练。')
        return
    thalf_x, names = featurize(thalf.smiles.tolist())
    cl_x, cl_names = featurize(cl.smiles.tolist())
    vdss_x, vdss_names = featurize(vdss.smiles.tolist())
    cache = {}
    source_frames = {'CL': (cl, cl_x, cl_spec, cl_names), 'VDss': (vdss, vdss_x, vdss_spec, vdss_names)}
    source_configs = {'CL': (cl_algorithm, cl_params), 'VDss': (vdss_algorithm, vdss_params)}
    algorithms = {'ridge': 10.0, 'rf': None}
    oof = {name: np.full(len(thalf), np.nan) for name in algorithms}
    with threadpool_limits(limits=args.threads), stage_output(output) as out:
        for fold in tqdm(folds, desc='Thalf 受保护外层折'):
            features = make_features(thalf, thalf_x, cl, vdss, source_frames, source_x=None,
                                     source_specs=None, source_names=None, source_configs=source_configs,
                                     outer_fold=fold, args=args, cache=cache)
            train_idx = idx[thalf.iloc[idx].fold_id.to_numpy() != fold]
            valid_idx = idx[thalf.iloc[idx].fold_id.to_numpy() == fold]
            for algorithm, alpha in algorithms.items():
                prep, target, estimator = fit_cascade(thalf, features, train_idx, algorithm, alpha, args)
                oof[algorithm][valid_idx] = predict(prep, target, estimator, features[valid_idx])
        # 最终训练特征使用每行自身折外的上游模型；val 上游模型使用完整 source train。
        final_features = make_features(thalf, thalf_x, cl, vdss, source_frames, source_x=None,
                                       source_specs=None, source_names=None, source_configs=source_configs,
                                       outer_fold=None, args=args, cache=cache)
        summary = {}
        for algorithm, alpha in algorithms.items():
            folder = out / algorithm
            folder.mkdir()
            prep, target, estimator = fit_cascade(thalf, final_features, idx, algorithm, alpha, args)
            val_idx = np.flatnonzero(thalf.split.eq('val'))
            val_pred = predict(prep, target, estimator, final_features[val_idx])
            summary[algorithm] = {'oof': metrics(thalf.iloc[idx], oof[algorithm][idx], spec),
                                  'val': metrics(thalf.iloc[val_idx], val_pred, spec)}
            pd.DataFrame({'row_id': thalf.iloc[idx].row_id.to_numpy(),
                          'predicted_physical': oof[algorithm][idx]}).to_csv(folder / 'oof_predictions.csv', index=False)
            pd.DataFrame({'row_id': thalf.iloc[val_idx].row_id.to_numpy(),
                          'predicted_physical': val_pred}).to_csv(folder / 'val_predictions.csv', index=False)
            joblib.dump({'preprocessor': prep, 'target': target, 'estimator': estimator,
                         'feature_names': names + ['log10_CL_prior', 'log10_VDss_prior'],
                         'source_run_dirs': {'CL': str(args.cl_run_dir), 'VDss': str(args.vdss_run_dir)}}, folder / 'model.joblib')
        selected = min(algorithms, key=lambda name: summary[name]['oof']['primary'])
        dump_json(out / 'metrics.json', summary)
        dump_json(out / 'selection.json', {'algorithm': selected, 'criterion': 'protected_outer_oof_primary',
                  'caveat': 'diagnostic only: human Thalf has few independent molecules; test remains frozen'})
        dump_json(out / 'lineage.json', {'CL': str(args.cl_run_dir), 'VDss': str(args.vdss_run_dir),
                  'protection': 'outer fold and each training row fold are excluded from its source models'})
        finish_stage(out, 'thalf_protected_cascade', inputs={'datasets_complete_sha256': sha256(datasets / 'complete.json'),
                     'splits_complete_sha256': sha256(splits / 'complete.json'),
                     'cl_complete_sha256': sha256(args.cl_run_dir / 'complete.json'),
                     'vdss_complete_sha256': sha256(args.vdss_run_dir / 'complete.json')},
                     task_id='Thalf__human__terminal_iv', test_evaluated=False, partial=False)
    logging.info('级联诊断产物: %s；受保护 OOF 选定 %s；测试评估=False', output, selected)


if __name__ == '__main__':
    run_cli(run)

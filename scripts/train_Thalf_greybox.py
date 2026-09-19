"""Protected-OFF mechanistic Thalf model: structure -> CL/VDss -> half-life residual."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dmpk_toolkit import (featurize, fit_model, load_task, metrics, molecule_weights,
                          prediction_table)
from pipeline_common import (base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check)


THALF_TASK = 'Thalf__human__terminal_iv'
UPSTREAM_TASKS = {
    'CL': 'CL__human__systemic_iv',
    'VDss': 'VDss__human__steady_state_iv',
}
UPSTREAM_PARAMS = {
    'n_estimators': 200,
    'min_samples_leaf': 3,
    'max_features': .3,
    'max_depth': 24,
}
RESIDUAL_SPECS = {
    'mechanistic_only': None,
    'residual_ridge': {'algorithm': 'ridge', 'alpha': 100.},
    'residual_extratrees': {
        'algorithm': 'extratrees', 'n_estimators': 400, 'min_samples_leaf': 3,
        'max_features': .7, 'max_depth': 24,
    },
}


def make_residual_estimator(spec, seed, threads):
    if spec['algorithm'] == 'ridge':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
            ('model', Ridge(solver='lsqr', tol=1e-6, alpha=spec['alpha'])),
        ])
    if spec['algorithm'] == 'extratrees':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('model', ExtraTreesRegressor(
                random_state=seed, n_jobs=threads, n_estimators=spec['n_estimators'],
                min_samples_leaf=spec['min_samples_leaf'], max_features=spec['max_features'],
                max_depth=spec['max_depth'])),
        ])
    raise ValueError(spec['algorithm'])


def fit_residual(frame, features, indices, spec, seed, threads):
    model = make_residual_estimator(spec, seed, threads)
    residual = np.log10(frame.iloc[indices].raw_value.to_numpy(float)) - features[indices, -1]
    model.fit(features[indices], residual, model__sample_weight=molecule_weights(frame.iloc[indices]))
    return model


def physical_prediction(mechanistic_log, residual_prediction):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.power(10., mechanistic_log + residual_prediction)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError('灰盒反变换出现非正或非有限预测')
    return result


def fit_upstream_models(thalf, upstream, upstream_features, thalf_features, spec, args, out):
    train_idx = np.flatnonzero(thalf.split.eq('train'))
    val_idx = np.flatnonzero(thalf.split.eq('val'))
    folds = sorted(thalf.iloc[train_idx].fold_id.unique())
    crossfit = {name: np.full(len(thalf), np.nan) for name in UPSTREAM_TASKS}
    models = {name: {} for name in UPSTREAM_TASKS}

    for name, task in UPSTREAM_TASKS.items():
        frame, task_spec = upstream[name]
        x = upstream_features[name]
        folder = out / 'upstream' / name
        folder.mkdir(parents=True)
        for fold in folds:
            heldout = train_idx[thalf.iloc[train_idx].fold_id.to_numpy() == fold]
            fit_idx = np.flatnonzero(frame.split.eq('train') & frame.fold_id.ne(fold))
            overlap = set(frame.iloc[fit_idx].molecule_id) & set(thalf.iloc[heldout].molecule_id)
            if overlap:
                raise ValueError(f'上游 {name} fold {fold} 与 Thalf heldout 分子泄漏: {next(iter(overlap))}')
            bundle = fit_model(frame, x, fit_idx, 'rf', UPSTREAM_PARAMS, task_spec,
                               args.upstream_descriptor_names, args)
            values = bundle.predict_features(thalf_features[heldout],
                                             thalf.iloc[heldout].smiles.to_numpy())
            if len(values) != len(heldout):
                raise ValueError(f'上游 {name} fold {fold} 预测长度不匹配')
            crossfit[name][heldout] = values
            models[name][f'fold_{fold}'] = bundle
            fold_dir = folder / f'fold_{fold}'
            fold_dir.mkdir()
            joblib.dump(bundle, fold_dir / 'model.joblib', compress=3)
            dump_json(fold_dir / 'lineage.json', {
                'task_id': task, 'algorithm': 'rf', 'parameters': UPSTREAM_PARAMS,
                'heldout_thalf_fold': int(fold),
                'upstream_train_molecules': len(bundle.train_molecule_ids),
                'thalf_heldout_molecules': int(thalf.iloc[heldout].molecule_id.nunique()),
                'overlap_with_thalf_heldout': 0,
            })

        full_idx = np.flatnonzero(frame.split.eq('train'))
        overlap = set(frame.iloc[full_idx].molecule_id) & set(thalf.iloc[val_idx].molecule_id)
        if overlap:
            raise ValueError(f'上游 {name} 与 Thalf validation 分子泄漏: {next(iter(overlap))}')
        full = fit_model(frame, x, full_idx, 'rf', UPSTREAM_PARAMS, task_spec,
                         args.upstream_descriptor_names, args)
        if len(val_idx):
            values = full.predict_features(
                thalf_features[val_idx], thalf.iloc[val_idx].smiles.to_numpy())
            if len(values) != len(val_idx):
                raise ValueError(f'上游 {name} validation 预测长度不匹配')
            crossfit[name][val_idx] = values
        models[name]['full_train'] = full
        joblib.dump(full, folder / 'model.joblib', compress=3)
        dump_json(folder / 'lineage.json', {
            'task_id': task, 'algorithm': 'rf', 'parameters': UPSTREAM_PARAMS,
            'fit_scope': 'all_upstream_train_only',
            'upstream_train_molecules': len(full.train_molecule_ids),
            'overlap_with_thalf_validation': 0,
        })

    used = np.concatenate([train_idx, val_idx])
    if not all(np.isfinite(crossfit[name][used]).all() and (crossfit[name][used] > 0).all()
               for name in UPSTREAM_TASKS):
        raise ValueError('上游交叉拟合预测不完整或非正')
    return crossfit, models


def write_prediction(path, frame, prediction, spec, kind, groups):
    prediction_table(frame, prediction, spec, kind, groups).to_csv(path, index=False)


def run():
    parser = base_parser('Thalf protected-OFF CL/VDss mechanistic residual greybox; no test evaluation')
    parser.add_argument('--datasets-dir', type=Path)
    parser.add_argument('--splits-dir', type=Path)
    parser.add_argument('--upstream-feature-set', default='ecfp4_rdkit2d',
                        choices=['ecfp4_rdkit2d', 'rdkit2d'])
    parser.add_argument('--residual-feature-set', default='rdkit2d', choices=['rdkit2d', 'mechanism2d'])
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--run-name', default='v15_greybox_preregistered_v1')
    args = parser.parse_args()
    configure_logging(args.root, 'train_Thalf_greybox')
    if args.threads < 1:
        raise ValueError('--threads 必须为正')
    if not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError('--run-name 必须为单个目录名')
    # fit_model is shared with the STL trainer; RF ignores these neural/GPU fields.
    args.device = 'cpu'
    args.epochs = 80
    args.batch_size = 64
    args.feature_set = args.upstream_feature_set
    datasets = args.datasets_dir or args.root / 'data/processed_v15/datasets'
    splits = args.splits_dir or args.root / 'data/processed_v15/splits'
    output = args.output or args.root / 'models/greybox' / THALF_TASK / args.run_name
    startup_self_check(required=[datasets / 'complete.json', splits / 'complete.json'], output=output)

    thalf, spec, split_meta = load_task(args.root, datasets, splits, THALF_TASK)
    upstream = {}
    for name, task in UPSTREAM_TASKS.items():
        frame, task_spec, upstream_split_meta = load_task(args.root, datasets, splits, task)
        if upstream_split_meta != split_meta:
            raise ValueError(f'{name} 划分谱系不一致')
        upstream[name] = (frame, task_spec)
    train_idx = np.flatnonzero(thalf.split.eq('train'))
    val_idx = np.flatnonzero(thalf.split.eq('val'))
    folds = sorted(thalf.iloc[train_idx].fold_id.unique())
    if len(folds) < 3 or len(val_idx) == 0:
        raise ValueError('Thalf 需要至少三个训练折和非空 validation')

    # Descriptor names are deterministic and are recorded with all saved bundles.
    _, args.upstream_descriptor_names = featurize(['CC'], progress=False,
                                                   feature_set=args.upstream_feature_set)
    if args.check_only:
        print('Validated v15 tasks, fixed folds, upstream split alignment, and protected-fit plan; no models trained.')
        return

    with stage_output(output) as out:
        thalf_upstream_features, _ = featurize(thalf.smiles.tolist(), progress=True,
                                                feature_set=args.upstream_feature_set)
        upstream_features = {
            name: featurize(frame.smiles.tolist(), progress=True,
                            feature_set=args.upstream_feature_set)[0]
            for name, (frame, _) in upstream.items()
        }
        crossfit, _ = fit_upstream_models(
            thalf, upstream, upstream_features, thalf_upstream_features, spec, args, out)
        mechanistic_log = (np.log10(.693) + np.log10(crossfit['VDss']) - np.log10(crossfit['CL']))
        residual_features, residual_names = featurize(
            thalf.smiles.tolist(), progress=True, feature_set=args.residual_feature_set)
        feature_matrix = np.column_stack([
            residual_features, np.log10(crossfit['CL']), np.log10(crossfit['VDss']), mechanistic_log])
        feature_names = residual_names + ['log10_CL_hat', 'log10_VDss_hat', 'log10_Thalf_mechanistic']
        dump_json(out / 'feature_schema.json', {
            'upstream_feature_set': args.upstream_feature_set,
            'residual_feature_set': args.residual_feature_set,
            'residual_feature_names': feature_names,
            'mechanistic_formula': 'Thalf_h = 0.693 * VDss_L_per_kg / CL_L_per_h_per_kg',
            'upstream_algorithm': 'rf', 'upstream_parameters': UPSTREAM_PARAMS,
            'residual_specs': RESIDUAL_SPECS,
        })
        pd.DataFrame({
            'row_id': thalf.row_id, 'molecule_id': thalf.molecule_id, 'split': thalf.split,
            'fold_id': thalf.fold_id, 'CL_hat': crossfit['CL'], 'VDss_hat': crossfit['VDss'],
            'Thalf_mechanistic_hat': np.power(10., mechanistic_log),
        }).loc[thalf.split.isin(['train', 'val'])].to_csv(
            out / 'protected_upstream_predictions.csv', index=False)

        summary = {}
        for label, residual_spec in RESIDUAL_SPECS.items():
            folder = out / label
            folder.mkdir()
            oof = np.full(len(thalf), np.nan)
            if residual_spec is None:
                oof[train_idx] = physical_prediction(mechanistic_log[train_idx], 0.)
                val_prediction = physical_prediction(mechanistic_log[val_idx], 0.)
                groups = []
            else:
                for fold in folds:
                    heldout = train_idx[thalf.iloc[train_idx].fold_id.to_numpy() == fold]
                    fit_idx = train_idx[thalf.iloc[train_idx].fold_id.to_numpy() != fold]
                    model = fit_residual(thalf, feature_matrix, fit_idx, residual_spec,
                                         args.seed + int(fold), args.threads)
                    oof[heldout] = physical_prediction(
                        mechanistic_log[heldout], model.predict(feature_matrix[heldout]))
                    fold_dir = folder / f'fold_{fold}'
                    fold_dir.mkdir()
                    joblib.dump(model, fold_dir / 'model.joblib', compress=3)
                    dump_json(fold_dir / 'lineage.json', {
                        'heldout_fold': int(fold), 'residual_spec': residual_spec,
                        'train_molecules': int(thalf.iloc[fit_idx].molecule_id.nunique()),
                        'upstream_input': 'cross_fitted_predictions_only',
                    })
                final = fit_residual(thalf, feature_matrix, train_idx, residual_spec,
                                     args.seed, args.threads)
                val_prediction = physical_prediction(
                    mechanistic_log[val_idx], final.predict(feature_matrix[val_idx]))
                joblib.dump(final, folder / 'model.joblib', compress=3)
                dump_json(folder / 'lineage.json', {
                    'fit_scope': 'all_thalf_train', 'residual_spec': residual_spec,
                    'upstream_input': 'cross_fitted_train_predictions; full-train upstream validation predictions',
                })
                groups = sorted(thalf.iloc[train_idx].scaffold_group.unique())
            if not np.isfinite(oof[train_idx]).all():
                raise ValueError(f'{label} OOF 不完整')
            write_prediction(folder / 'oof_predictions.csv', thalf.iloc[train_idx], oof[train_idx],
                             spec, 'protected_outer_oof', groups)
            write_prediction(folder / 'val_predictions.csv', thalf.iloc[val_idx], val_prediction,
                             spec, 'validation', groups)
            summary[label] = {
                'oof': metrics(thalf.iloc[train_idx], oof[train_idx], spec),
                'val': metrics(thalf.iloc[val_idx], val_prediction, spec),
            }
        selected = min(summary, key=lambda item: summary[item]['oof']['primary'])
        dump_json(out / 'metrics.json', summary)
        dump_json(out / 'selection.json', {
            'algorithm': selected, 'criterion': 'protected_outer_oof_primary',
            'caveat': 'one validation comparison only; no test predictions or labels evaluated',
        })
        finish_stage(out, 'thalf_greybox_training', inputs={
            'datasets_complete_sha256': sha256(datasets / 'complete.json'),
            'splits_complete_sha256': sha256(splits / 'complete.json'),
        }, task_id=THALF_TASK, upstream_tasks=UPSTREAM_TASKS,
        upstream_algorithm='rf', upstream_parameters=UPSTREAM_PARAMS,
        residual_specs=RESIDUAL_SPECS, seed=args.seed, test_evaluated=False,
        partial=False)
    print(f'灰盒训练产物: {output}; 训练 OOF 选定 {selected}; 测试评估=False')


if __name__ == '__main__':
    run_cli(run)

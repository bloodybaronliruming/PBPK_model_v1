"""预注册的 Thalf v15 Elastic Net 与加性样条 GAM；固定折且不评估 test。"""
from __future__ import annotations

import itertools
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from dmpk_toolkit import (MECHANISM_2D_DESCRIPTORS, TargetSpace, featurize,
                          load_task, metrics, molecule_weights, prediction_table,
                          primary_score)
from pipeline_common import (ROOT, base_parser, configure_logging, dump_json,
                             finish_stage, run_cli, sha256, stage_output,
                             startup_self_check, verify_stage)


TASK = 'Thalf__human__terminal_iv'
CONFIGS = {
    'elasticnet_mechanism2d': {'algorithm': 'elasticnet', 'feature_set': 'mechanism2d'},
    'elasticnet_rdkit2d': {'algorithm': 'elasticnet', 'feature_set': 'rdkit2d'},
    'splinegam_mechanism2d': {'algorithm': 'splinegam', 'feature_set': 'mechanism2d'},
}
L1_RATIOS = (0.05, 0.2, 0.5, 0.8, 0.95, 1.0)
ELASTICNET_ALPHAS = tuple(float(x) for x in np.logspace(-4, 0, 9))
GAM_KNOTS = (4, 6, 8)
GAM_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
BENCHMARK_ENSEMBLE = 'extratrees_3seed'
NONINFERIORITY_MARGIN = 0.030
ELASTICNET_MAX_ITER = 100000
ELASTICNET_TOL = 1e-5


class ModelConvergenceError(RuntimeError):
    """A candidate is numerically ineligible for model selection."""


def parameter_grid(algorithm):
    if algorithm == 'elasticnet':
        return [{'alpha': alpha, 'l1_ratio': ratio}
                for alpha, ratio in itertools.product(ELASTICNET_ALPHAS, L1_RATIOS)]
    if algorithm == 'splinegam':
        return [{'n_knots': knots, 'alpha': alpha}
                for knots, alpha in itertools.product(GAM_KNOTS, GAM_ALPHAS)]
    raise ValueError(f'未知白盒算法: {algorithm}')


def build_preprocessor(algorithm, parameters):
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if algorithm == 'elasticnet':
        steps.append(('scale', StandardScaler()))
    elif algorithm == 'splinegam':
        steps.extend([
            ('spline', SplineTransformer(n_knots=parameters['n_knots'], degree=3,
                                         include_bias=False, extrapolation='linear')),
            ('scale', StandardScaler()),
        ])
    else:
        raise ValueError(f'未知白盒算法: {algorithm}')
    return Pipeline(steps)


def build_estimator(algorithm, parameters):
    if algorithm == 'elasticnet':
        return ElasticNet(alpha=parameters['alpha'], l1_ratio=parameters['l1_ratio'],
                          selection='cyclic', max_iter=ELASTICNET_MAX_ITER,
                          tol=ELASTICNET_TOL)
    if algorithm == 'splinegam':
        return Ridge(alpha=parameters['alpha'], solver='lsqr', tol=1e-7)
    raise ValueError(f'未知白盒算法: {algorithm}')


def fit_model(frame, X, indices, config, parameters):
    training = frame.iloc[indices]
    weights = molecule_weights(training)
    preprocessor = build_preprocessor(config['algorithm'], parameters)
    transformed = preprocessor.fit_transform(X[indices])
    if not np.isfinite(transformed).all():
        raise ValueError('白盒特征预处理后出现非有限值')
    target = TargetSpace('log10').fit(training.raw_value, weights)
    estimator = build_estimator(config['algorithm'], parameters)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(transformed, target.forward(training.raw_value), sample_weight=weights)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ModelConvergenceError(
            f'{config["algorithm"]} 未收敛: parameters={parameters}, '
            f'n_iter={getattr(estimator, "n_iter_", None)}, '
            f'dual_gap={getattr(estimator, "dual_gap_", None)}')
    return {
        'algorithm': config['algorithm'],
        'feature_set': config['feature_set'],
        'preprocessor': preprocessor,
        'estimator': estimator,
        'target': target,
        'parameters': parameters,
        'convergence': {
            'status': 'converged',
            'n_iter': (int(getattr(estimator, 'n_iter_'))
                       if hasattr(estimator, 'n_iter_')
                       and np.ndim(getattr(estimator, 'n_iter_')) == 0 else None),
            'dual_gap': (float(getattr(estimator, 'dual_gap_'))
                         if hasattr(estimator, 'dual_gap_')
                         and np.ndim(getattr(estimator, 'dual_gap_')) == 0 else None),
            'max_iter': ELASTICNET_MAX_ITER if config['algorithm'] == 'elasticnet' else None,
            'tol': ELASTICNET_TOL if config['algorithm'] == 'elasticnet' else None,
        },
        'descriptor_names': list(MECHANISM_2D_DESCRIPTORS) if config['feature_set'] == 'mechanism2d' else None,
        'train_molecule_ids': sorted(training.molecule_id.unique()),
        'train_groups': sorted(training.scaffold_group.unique()),
    }


def predict(model, X):
    standardized = model['estimator'].predict(model['preprocessor'].transform(X))
    return model['target'].inverse(standardized, 'Thalf')


def tune(frame, X, indices, config, options, records, scope):
    folds = sorted(int(x) for x in frame.iloc[indices].fold_id.unique())
    if len(folds) < 2:
        raise ValueError(f'{scope} 的内层固定折不足')
    scores = []
    for trial, parameters in enumerate(tqdm(options, desc=f'{scope} 调参', leave=False)):
        trial_prediction = np.full(len(frame), np.nan)
        failure = None
        for fold in folds:
            in_scope = frame.iloc[indices]
            fit_idx = indices[in_scope.fold_id.to_numpy() != fold]
            valid_idx = indices[in_scope.fold_id.to_numpy() == fold]
            try:
                model = fit_model(frame, X, fit_idx, config, parameters)
            except ModelConvergenceError as exc:
                failure = f'inner_fold_{fold}: {exc}'
                break
            trial_prediction[valid_idx] = predict(model, X[valid_idx])
        if failure is None:
            try:
                # Also require convergence on the exact scope used by the
                # subsequently saved model.
                fit_model(frame, X, indices, config, parameters)
            except ModelConvergenceError as exc:
                failure = f'full_scope: {exc}'
        if failure is None:
            value = primary_score(frame.iloc[indices], trial_prediction[indices],
                                  {'endpoint': 'Thalf', 'transform': 'log10'})
            status = 'converged'
        else:
            value = float('inf')
            status = 'invalid_nonconverged'
        records.append({'scope': scope, 'trial': trial, 'parameters': parameters,
                        'status': status, 'failure': failure,
                        'primary_log10_rmse': value if np.isfinite(value) else None})
        scores.append(value)
    if not np.isfinite(scores).any():
        raise ModelConvergenceError(f'{scope} 的全部候选均未收敛')
    return options[int(np.argmin(scores))]


def feature_names(feature_set, descriptor_names):
    if feature_set == 'mechanism2d':
        return list(MECHANISM_2D_DESCRIPTORS)
    return list(descriptor_names)


def elasticnet_stability(config_name, models, names):
    rows = []
    coefficients = []
    for fold, model in models:
        coefficient = np.asarray(model['estimator'].coef_, dtype=float) * model['target'].std
        coefficients.append(coefficient)
        for name, value in zip(names, coefficient):
            rows.append({'config': config_name, 'fold': fold, 'feature': name,
                         'standardized_log10_coefficient': value,
                         'nonzero': abs(value) > 1e-10, 'sign': int(np.sign(value))})
    detail = pd.DataFrame(rows)
    array = np.stack(coefficients)
    nonzero = np.abs(array) > 1e-10
    positive = array > 1e-10
    negative = array < -1e-10
    summary = pd.DataFrame({
        'config': config_name,
        'feature': names,
        'nonzero_fold_fraction': nonzero.mean(axis=0),
        'positive_fold_fraction': positive.mean(axis=0),
        'negative_fold_fraction': negative.mean(axis=0),
        'mean_standardized_log10_coefficient': array.mean(axis=0),
        'sd_standardized_log10_coefficient': array.std(axis=0, ddof=1),
    })
    return detail, summary


def final_coefficients(config_name, model, names):
    coefficient = np.asarray(model['estimator'].coef_, dtype=float) * model['target'].std
    return pd.DataFrame({'config': config_name, 'feature': names,
                         'standardized_log10_coefficient': coefficient,
                         'absolute_coefficient': np.abs(coefficient),
                         'nonzero': np.abs(coefficient) > 1e-10}).sort_values(
                             'absolute_coefficient', ascending=False)


def gam_shapes(config_name, model, X_train, names):
    imputer = model['preprocessor'].named_steps['imputer']
    imputed = imputer.transform(X_train)
    baseline = np.nanmedian(imputed, axis=0)
    baseline_prediction = float(model['estimator'].predict(
        model['preprocessor'].transform(baseline.reshape(1, -1)))[0] * model['target'].std)
    rows = []
    ranges = []
    for column, name in enumerate(names):
        low, high = np.quantile(imputed[:, column], [0.01, 0.99])
        grid = np.linspace(low, high, 101) if high > low else np.repeat(low, 101)
        probe = np.repeat(baseline.reshape(1, -1), len(grid), axis=0)
        probe[:, column] = grid
        prediction = model['estimator'].predict(model['preprocessor'].transform(probe))
        effect = prediction * model['target'].std - baseline_prediction
        rows.extend({'config': config_name, 'feature': name, 'feature_value': value,
                     'centered_log10_effect': response, 'train_quantile_low': low,
                     'train_quantile_high': high}
                    for value, response in zip(grid, effect))
        ranges.append({'config': config_name, 'feature': name,
                       'train_quantile_01': low, 'train_quantile_99': high})
    return pd.DataFrame(rows), pd.DataFrame(ranges)


def validation_extrapolation(config_name, model, X_train, X_val, names):
    imputer = model['preprocessor'].named_steps['imputer']
    train = imputer.transform(X_train)
    val = imputer.transform(X_val)
    rows = []
    for column, name in enumerate(names):
        low, high = np.quantile(train[:, column], [0.01, 0.99])
        rows.append({'config': config_name, 'feature': name, 'train_quantile_01': low,
                     'train_quantile_99': high,
                     'validation_below_fraction': float((val[:, column] < low).mean()),
                     'validation_above_fraction': float((val[:, column] > high).mean()),
                     'validation_outside_fraction': float(((val[:, column] < low) | (val[:, column] > high)).mean())})
    return pd.DataFrame(rows)


def high_correlations(config_name, X_train, names):
    frame = pd.DataFrame(X_train, columns=names).replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(frame.median())
    correlation = frame.corr().to_numpy()
    rows = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            value = correlation[left, right]
            if np.isfinite(value) and abs(value) >= 0.90:
                rows.append({'config': config_name, 'feature_a': names[left],
                             'feature_b': names[right], 'pearson_r': value})
    return pd.DataFrame(rows, columns=['config', 'feature_a', 'feature_b', 'pearson_r'])


def benchmark_scores(path):
    meta = verify_stage(path, 'thalf_descriptor_three_seed_analysis')
    if meta.get('test_evaluated'):
        raise ValueError('基准分析不得包含 test')
    table = pd.read_csv(path / 'table_3_ensemble_metrics.csv')
    part = table.loc[(table.family == 'rdkit2d') & (table.ensemble == BENCHMARK_ENSEMBLE)]
    scores = dict(zip(part.scope, part.primary_log10_rmse))
    if set(scores) != {'oof', 'val'}:
        raise ValueError('缺少预注册 RDKit2D ExtraTrees 集成基准')
    return {key: float(value) for key, value in scores.items()}, meta


def run():
    parser = base_parser('Preregistered Thalf white-box training without test evaluation')
    parser.add_argument('--datasets-dir', type=Path, default=ROOT / 'data/processed_v15/datasets')
    parser.add_argument('--splits-dir', type=Path, default=ROOT / 'data/processed_v15/splits')
    parser.add_argument('--benchmark-dir', type=Path,
                        default=ROOT / 'results/analysis/thalf_v15_rdkit2d_three_seed_v1')
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--run-name', default='v15_whitebox_preregistered_v1')
    args = parser.parse_args()
    configure_logging(args.root, 'train_Thalf_whitebox')
    if args.threads < 1 or not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError('threads 必须为正数，run-name 必须是单个目录名')
    output = args.output or args.root / 'models/whitebox' / TASK / args.run_name
    startup_self_check([args.datasets_dir / 'complete.json', args.splits_dir / 'complete.json',
                        args.benchmark_dir / 'complete.json'], output=output)
    frame, spec, split_meta = load_task(args.root, args.datasets_dir, args.splits_dir, TASK)
    if spec.get('transform') != 'log10':
        raise ValueError('Thalf 白盒预注册要求 log10 目标')
    benchmark, benchmark_meta = benchmark_scores(args.benchmark_dir)
    train_idx = np.flatnonzero(frame.split.eq('train'))
    val_idx = np.flatnonzero(frame.split.eq('val'))
    folds = sorted(int(x) for x in frame.iloc[train_idx].fold_id.unique())
    if folds != list(range(split_meta['folds'])) or len(val_idx) == 0:
        raise ValueError('训练固定折或 validation 不完整')
    logging.info('task=%s；训练记录=%d，分子=%d；validation记录=%d，分子=%d；固定折=%s；CPU线程=%d',
                 TASK, len(train_idx), frame.iloc[train_idx].molecule_id.nunique(), len(val_idx),
                 frame.iloc[val_idx].molecule_id.nunique(), folds, args.threads)
    if args.check_only:
        logging.info('数据、划分、基准、白盒参数空间及 test 隔离自检通过；尚未训练。')
        return

    features = {}
    schemas = {}
    for feature_set in sorted({item['feature_set'] for item in CONFIGS.values()}):
        X, descriptor_names = featurize(frame.smiles.tolist(), feature_set=feature_set)
        names = feature_names(feature_set, descriptor_names)
        if X.shape[1] != len(names):
            raise ValueError(f'{feature_set} 特征名称与矩阵不一致')
        features[feature_set] = X
        schemas[feature_set] = names

    all_metrics = {}
    tuning_records = []
    final_models = {}
    fold_models = {}
    oof_tables = {}
    with threadpool_limits(limits=args.threads), stage_output(output) as out:
        # Model ranking is completed from OOF before validation predictions are created.
        for config_name, config in CONFIGS.items():
            logging.info('开始白盒配置: %s', config_name)
            folder = out / config_name
            folder.mkdir()
            X = features[config['feature_set']]
            options = parameter_grid(config['algorithm'])
            oof = np.full(len(frame), np.nan)
            fold_models[config_name] = []
            tables = []
            for fold in tqdm(folds, desc=f'{config_name} 外层 OOF'):
                outer_train = train_idx[frame.iloc[train_idx].fold_id.to_numpy() != fold]
                outer_valid = train_idx[frame.iloc[train_idx].fold_id.to_numpy() == fold]
                records = []
                parameters = tune(frame, X, outer_train, config, options, records, f'outer_{fold}')
                tuning_records.extend({'config': config_name, **item} for item in records)
                model = fit_model(frame, X, outer_train, config, parameters)
                if set(model['train_groups']) & set(frame.iloc[outer_valid].scaffold_group):
                    raise ValueError('白盒外层 OOF 骨架泄漏')
                oof[outer_valid] = predict(model, X[outer_valid])
                fold_models[config_name].append((fold, model))
                fold_dir = folder / f'fold_{fold}'
                fold_dir.mkdir()
                joblib.dump(model, fold_dir / 'model.joblib', compress=3)
                dump_json(fold_dir / 'lineage.json', {
                    'heldout_fold': fold, 'parameters': parameters,
                    'train_molecule_ids': model['train_molecule_ids'],
                    'train_groups': model['train_groups'],
                })
                tables.append(prediction_table(frame.iloc[outer_valid], oof[outer_valid], spec,
                                               'outer_oof', model['train_groups']))
            if not np.isfinite(oof[train_idx]).all():
                raise ValueError(f'{config_name} OOF 不完整')
            oof_tables[config_name] = pd.concat(tables, ignore_index=True)
            oof_tables[config_name].to_csv(folder / 'oof_predictions.csv', index=False)
            all_metrics[config_name] = {'oof': metrics(frame.iloc[train_idx], oof[train_idx], spec)}
            records = []
            parameters = tune(frame, X, train_idx, config, options, records, 'final_training_cv')
            tuning_records.extend({'config': config_name, **item} for item in records)
            model = fit_model(frame, X, train_idx, config, parameters)
            final_models[config_name] = model
            joblib.dump(model, folder / 'model.joblib', compress=3)
            dump_json(folder / 'lineage.json', {
                'parameters': parameters, 'train_molecule_ids': model['train_molecule_ids'],
                'train_groups': model['train_groups'], 'feature_set': config['feature_set'],
            })
            dump_json(folder / 'feature_schema.json', {
                'feature_set': config['feature_set'], 'feature_count': len(schemas[config['feature_set']]),
                'feature_order': schemas[config['feature_set']], 'fit_scope': 'deterministic_structure_only',
            })

        selected = min(CONFIGS, key=lambda name: all_metrics[name]['oof']['primary'])
        ordered = sorted(CONFIGS, key=lambda name: all_metrics[name]['oof']['primary'])
        if ({ordered[0], ordered[1]} == {'elasticnet_mechanism2d', 'splinegam_mechanism2d'}
                and abs(all_metrics[ordered[0]]['oof']['primary']
                        - all_metrics[ordered[1]]['oof']['primary']) < 0.005):
            selected = 'elasticnet_mechanism2d'

        # Validation is evaluated once, only after OOF ranking is fixed.
        for config_name, config in CONFIGS.items():
            X = features[config['feature_set']]
            model = final_models[config_name]
            validation = predict(model, X[val_idx])
            prediction_table(frame.iloc[val_idx], validation, spec, 'val_heldout',
                             model['train_groups']).to_csv(out / config_name / 'val_predictions.csv', index=False)
            all_metrics[config_name]['val'] = metrics(frame.iloc[val_idx], validation, spec)

        decision = {
            'selected_by_nested_oof': selected,
            'tie_rule': 'mechanism Elastic Net wins when its OOF is within 0.005 of mechanism spline GAM',
            'benchmark': 'RDKit2D ExtraTrees three-seed transformed-space ensemble',
            'benchmark_oof_log10_rmse': benchmark['oof'],
            'benchmark_validation_log10_rmse': benchmark['val'],
            'noninferiority_margin': NONINFERIORITY_MARGIN,
            'predictive_candidate': bool(
                all_metrics[selected]['oof']['primary'] <= benchmark['oof'] + NONINFERIORITY_MARGIN
                and all_metrics[selected]['val']['primary'] <= benchmark['val'] + NONINFERIORITY_MARGIN),
            'test_evaluated': False,
        }
        dump_json(out / 'metrics.json', all_metrics)
        dump_json(out / 'selection.json', decision)
        pd.DataFrame(tuning_records).to_csv(out / 'tuning_history.csv', index=False)

        detail_tables, stability_tables, coefficient_tables = [], [], []
        shape_tables, range_tables, extrapolation_tables, correlation_tables = [], [], [], []
        for config_name, config in CONFIGS.items():
            X = features[config['feature_set']]
            names = schemas[config['feature_set']]
            model = final_models[config_name]
            extrapolation_tables.append(validation_extrapolation(
                config_name, model, X[train_idx], X[val_idx], names))
            correlation_tables.append(high_correlations(config_name, X[train_idx], names))
            if config['algorithm'] == 'elasticnet':
                detail, stability = elasticnet_stability(config_name, fold_models[config_name], names)
                detail_tables.append(detail)
                stability_tables.append(stability)
                coefficient_tables.append(final_coefficients(config_name, model, names))
            else:
                shapes, ranges = gam_shapes(config_name, model, X[train_idx], names)
                shape_tables.append(shapes)
                range_tables.append(ranges)
        pd.concat(detail_tables, ignore_index=True).to_csv(out / 'elasticnet_fold_coefficients.csv', index=False)
        pd.concat(stability_tables, ignore_index=True).to_csv(out / 'elasticnet_feature_stability.csv', index=False)
        pd.concat(coefficient_tables, ignore_index=True).to_csv(out / 'elasticnet_final_coefficients.csv', index=False)
        pd.concat(shape_tables, ignore_index=True).to_csv(out / 'splinegam_shape_functions.csv', index=False)
        pd.concat(range_tables, ignore_index=True).to_csv(out / 'splinegam_training_ranges.csv', index=False)
        pd.concat(extrapolation_tables, ignore_index=True).to_csv(out / 'validation_descriptor_extrapolation.csv', index=False)
        pd.concat(correlation_tables, ignore_index=True).to_csv(out / 'descriptor_correlations_abs_ge_0.90.csv', index=False)
        dump_json(out / 'protocol.json', {
            'configs': CONFIGS, 'elasticnet_alphas': ELASTICNET_ALPHAS, 'l1_ratios': L1_RATIOS,
            'gam_knots': GAM_KNOTS, 'gam_alphas': GAM_ALPHAS,
            'elasticnet_solver': {'selection': 'cyclic', 'max_iter': ELASTICNET_MAX_ITER,
                                  'tol': ELASTICNET_TOL,
                                  'nonconverged_candidate_policy': 'invalid_and_ineligible'},
            'selection_scope': 'nested_oof_only', 'validation_access': 'once_after_oof_ranking',
            'test_evaluated': False,
        })
        finish_stage(out, 'thalf_whitebox_training', inputs={
            'datasets_complete_sha256': sha256(args.datasets_dir / 'complete.json'),
            'splits_complete_sha256': sha256(args.splits_dir / 'complete.json'),
            'benchmark_complete_sha256': sha256(args.benchmark_dir / 'complete.json'),
        }, task_id=TASK, configs=CONFIGS, selected_by_nested_oof=selected,
                     test_evaluated=False, validation_access='once_after_oof_ranking', partial=False)
    logging.info('白盒训练产物: %s；OOF选择=%s；测试评估=False', output, selected)


if __name__ == '__main__':
    run_cli(run)

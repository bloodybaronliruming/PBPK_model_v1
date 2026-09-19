"""汇总 Thalf v15 RDKit2D 三种子及组合特征对照；不读取 test。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (base_parser, configure_logging, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)


TASK = 'Thalf__human__terminal_iv'
SEEDS = (2026, 2027, 2028)
ALGORITHMS = ('ridge', 'rf', 'extratrees')
FAMILIES = {
    'rdkit2d': ('v15_rdkit2d_seed', 'rdkit2d', 210),
    'ecfp4_rdkit2d': ('v15_obach_seed', 'ecfp4_rdkit2d', 2258),
}
ENSEMBLES = {
    'ridge_3seed': ('ridge',),
    'rf_3seed': ('rf',),
    'extratrees_3seed': ('extratrees',),
    'rf_extratrees_6model': ('rf', 'extratrees'),
}


def run_paths(root, family):
    prefix = FAMILIES[family][0]
    base = Path(root) / 'models/stl' / TASK
    return [base / f'{prefix}{seed}' for seed in SEEDS]


def validate_runs(paths, expected_feature_set):
    runs = []
    for expected_seed, path in zip(SEEDS, paths):
        meta = verify_stage(path, 'stl_training')
        if meta.get('task_id') != TASK or int(meta.get('seed', -1)) != expected_seed:
            raise ValueError(f'任务或种子不匹配: {path}')
        # Runs created before feature-set ablation used this exact combined
        # representation but did not yet persist its explicit name.
        feature_set = meta.get('feature_set', 'ecfp4_rdkit2d')
        if feature_set != expected_feature_set:
            raise ValueError(f'特征集合不匹配: {path}')
        if meta.get('test_evaluated'):
            raise ValueError(f'只接受未评估 test 的训练产物: {path}')
        test_files = [p for p in path.rglob('*') if p.is_file() and 'test' in p.name.lower()]
        if test_files:
            raise ValueError(f'训练目录包含 test 文件: {test_files[0]}')
        metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        if not set(ALGORITHMS) <= set(metrics):
            raise ValueError(f'缺少算法指标: {path}')
        schema = json.loads((path / 'feature_schema.json').read_text(encoding='utf-8'))
        schema = dict(schema)
        schema.setdefault('feature_set', 'ecfp4_rdkit2d')
        schema.setdefault('feature_count',
                          (2048 if schema.get('fingerprint') else 0)
                          + len(schema.get('descriptor_names', [])))
        if schema.get('feature_set') != expected_feature_set:
            raise ValueError(f'特征 schema 不匹配: {path}')
        runs.append((path, meta, metrics, schema))
    reference = runs[0][1]
    for _, meta, _, schema in runs[1:]:
        for key in ('inputs', 'task_id', 'excluded_folds', 'algorithms'):
            if meta.get(key) != reference.get(key):
                raise ValueError(f'三种子训练契约不一致: {key}')
        if schema != runs[0][3]:
            raise ValueError('三种子特征 schema 不一致')
    return runs


def seed_metrics(all_runs):
    rows = []
    for family, runs in all_runs.items():
        feature_count = int(runs[0][3]['feature_count'])
        for _, meta, metrics, _ in runs:
            for algorithm in ALGORITHMS:
                for scope in ('oof', 'val'):
                    item = metrics[algorithm][scope]
                    rows.append({
                        'family': family,
                        'feature_count': feature_count,
                        'seed': int(meta['seed']),
                        'algorithm': algorithm,
                        'scope': scope,
                        'primary_log10_rmse': float(item['primary']),
                        'gmfe': float(item['gmfe_positive_subset']),
                        'within_2fold': float(item['within_2fold_positive_subset']),
                        'within_3fold': float(item['within_3fold_positive_subset']),
                        'records': int(item['records']),
                        'molecules': int(item['molecules']),
                    })
    return pd.DataFrame(rows)


def summarize_seed_metrics(seed):
    rows = []
    keys = ['family', 'feature_count', 'algorithm', 'scope']
    for values, group in seed.groupby(keys, sort=False):
        scores = group.primary_log10_rmse.to_numpy(float)
        rows.append(dict(zip(keys, values)) | {
            'mean_primary_log10_rmse': scores.mean(),
            'sd_primary_log10_rmse': scores.std(ddof=1),
            'min_primary_log10_rmse': scores.min(),
            'max_primary_log10_rmse': scores.max(),
            'mean_gmfe': group.gmfe.mean(),
            'mean_within_2fold': group.within_2fold.mean(),
            'mean_within_3fold': group.within_3fold.mean(),
            'seeds': len(group),
        })
    return pd.DataFrame(rows)


def read_ensemble_predictions(runs, algorithms, scope):
    filename = 'oof_predictions.csv' if scope == 'oof' else 'val_predictions.csv'
    merged = None
    prediction_columns = []
    for path, meta, _, _ in runs:
        for algorithm in algorithms:
            column = f'pred_{algorithm}_{meta["seed"]}'
            frame = pd.read_csv(path / algorithm / filename)
            required = {'row_id', 'molecule_id', 'observed_physical', 'predicted_transformed'}
            if not required <= set(frame):
                raise ValueError(f'预测表缺少列: {path / algorithm / filename}')
            part = frame[list(required)].rename(columns={'predicted_transformed': column})
            if merged is None:
                merged = part
            else:
                merged = merged.merge(part[['row_id', column]], on='row_id', validate='one_to_one')
            prediction_columns.append(column)
    if merged is None or merged.row_id.duplicated().any():
        raise ValueError('无法构建唯一预测集成')
    merged['observed_transformed'] = np.log10(merged.observed_physical.to_numpy(float))
    merged['predicted_transformed'] = merged[prediction_columns].mean(axis=1)
    merged['predicted_physical'] = np.power(10., merged.predicted_transformed)
    return merged


def molecule_loss(frame):
    work = frame[['molecule_id', 'observed_transformed', 'predicted_transformed']].copy()
    work['squared_error'] = (work.predicted_transformed - work.observed_transformed) ** 2
    return work.groupby('molecule_id', sort=True).squared_error.mean()


def ensemble_metric(frame):
    error = frame.predicted_transformed.to_numpy() - frame.observed_transformed.to_numpy()
    counts = frame.groupby('molecule_id').molecule_id.transform('size').to_numpy()
    weights = 1. / counts
    weights /= weights.sum()
    absolute = np.abs(error)
    return {
        'primary_log10_rmse': float(np.sqrt(np.sum(weights * error ** 2))),
        'log10_mae': float(np.sum(weights * absolute)),
        'gmfe': float(10 ** np.sum(weights * absolute)),
        'within_2fold': float(np.sum(weights * (absolute <= np.log10(2)))),
        'within_3fold': float(np.sum(weights * (absolute <= np.log10(3)))),
        'records': len(frame),
        'molecules': int(frame.molecule_id.nunique()),
    }


def ensemble_tables(all_runs):
    rows = []
    predictions = {}
    for family, runs in all_runs.items():
        for label, algorithms in ENSEMBLES.items():
            for scope in ('oof', 'val'):
                frame = read_ensemble_predictions(runs, algorithms, scope)
                predictions[(family, label, scope)] = frame
                rows.append({'family': family, 'ensemble': label, 'scope': scope,
                             'members': len(SEEDS) * len(algorithms)} | ensemble_metric(frame))
    return pd.DataFrame(rows), predictions


def paired_bootstrap(predictions, repeats, seed=20260913):
    rng = np.random.default_rng(seed)
    comparisons = [
        ('rdkit2d_et3_vs_combined_et3', 'extratrees_3seed', 'extratrees_3seed'),
        ('rdkit2d_six_vs_combined_six', 'rf_extratrees_6model', 'rf_extratrees_6model'),
    ]
    rows = []
    for label, candidate, reference in comparisons:
        for scope in ('oof', 'val'):
            c = molecule_loss(predictions[('rdkit2d', candidate, scope)])
            r = molecule_loss(predictions[('ecfp4_rdkit2d', reference, scope)])
            paired = pd.concat([c.rename('candidate'), r.rename('reference')], axis=1, join='inner')
            if len(paired) != len(c) or len(paired) != len(r) or paired.isna().any().any():
                raise ValueError(f'配对分子不一致: {label} {scope}')
            values = paired.to_numpy(float)
            take = rng.integers(0, len(values), size=(repeats, len(values)))
            sampled = values[take]
            delta = np.sqrt(sampled[:, :, 0].mean(axis=1)) - np.sqrt(sampled[:, :, 1].mean(axis=1))
            candidate_rmse = float(np.sqrt(values[:, 0].mean()))
            reference_rmse = float(np.sqrt(values[:, 1].mean()))
            rows.append({
                'comparison': label,
                'scope': scope,
                'candidate': f'rdkit2d:{candidate}',
                'reference': f'ecfp4_rdkit2d:{reference}',
                'molecules': len(paired),
                'candidate_log10_rmse': candidate_rmse,
                'reference_log10_rmse': reference_rmse,
                'delta_rmse_candidate_minus_reference': candidate_rmse - reference_rmse,
                'bootstrap_95_ci_lower': float(np.quantile(delta, .025)),
                'bootstrap_95_ci_upper': float(np.quantile(delta, .975)),
                'candidate_lower_molecule_loss_fraction': float((values[:, 0] < values[:, 1]).mean()),
            })
    return pd.DataFrame(rows)


def report_text(summary, ensemble, paired):
    def score(family, algorithm, scope):
        row = summary.loc[(summary.family == family) & (summary.algorithm == algorithm)
                          & (summary.scope == scope)].iloc[0]
        return row.mean_primary_log10_rmse, row.sd_primary_log10_rmse

    def ensemble_score(family, label, scope):
        return float(ensemble.loc[(ensemble.family == family) & (ensemble.ensemble == label)
                                  & (ensemble.scope == scope), 'primary_log10_rmse'].iloc[0])

    r_oof, r_oof_sd = score('rdkit2d', 'extratrees', 'oof')
    r_val, r_val_sd = score('rdkit2d', 'extratrees', 'val')
    et_oof = ensemble_score('rdkit2d', 'extratrees_3seed', 'oof')
    et_val = ensemble_score('rdkit2d', 'extratrees_3seed', 'val')
    c_oof = ensemble_score('ecfp4_rdkit2d', 'extratrees_3seed', 'oof')
    c_val = ensemble_score('ecfp4_rdkit2d', 'extratrees_3seed', 'val')
    p = paired.loc[paired.comparison.eq('rdkit2d_et3_vs_combined_et3')].set_index('scope')
    return f"""# Thalf v15 RDKit2D 三种子正式汇总

生成日期：2026-09-13

## 完整性

RDKit2D 与 ECFP4+RDKit2D 各三个训练目录均通过产物哈希、任务、数据、划分、特征 schema 和种子契约核验。所有输入均为 `test_evaluated=false`，目录中不存在 test 预测文件。本报告只使用训练 OOF 与 validation。

## 主要结果

RDKit2D ExtraTrees 三种子 OOF 为 `{r_oof:.4f} ± {r_oof_sd:.4f}`，validation 为 `{r_val:.4f} ± {r_val_sd:.4f}`。三种子 log10 预测平均后的 OOF/validation 为 `{et_oof:.4f}/{et_val:.4f}`。

与 2258 维组合特征 ExtraTrees 三种子集成 `{c_oof:.4f}/{c_val:.4f}` 相比，210 维 RDKit2D 将维度减少 90.7%，OOF RMSE 增加 `{et_oof-c_oof:.4f}`，validation RMSE 减少 `{c_val-et_val:.4f}`。

逐分子配对 bootstrap 中，RDKit2D 减去组合特征的 OOF RMSE 差为 `{p.loc['oof','delta_rmse_candidate_minus_reference']:.4f}`，95% CI `[{p.loc['oof','bootstrap_95_ci_lower']:.4f}, {p.loc['oof','bootstrap_95_ci_upper']:.4f}]`；validation 差为 `{p.loc['val','delta_rmse_candidate_minus_reference']:.4f}`，95% CI `[{p.loc['val','bootstrap_95_ci_lower']:.4f}, {p.loc['val','bootstrap_95_ci_upper']:.4f}]`。两个区间均跨 0，不能宣称统计显著优越。

## 决定

1. RDKit2D ExtraTrees 三种子集成进入预候选，定位为紧凑、较可解释的性能候选。
2. 完整特征 RF+ExtraTrees 六模型集成继续作为强性能预候选；当前不冻结唯一方案。
3. 不继续根据 validation 增加描述符或扩大树模型调参。
4. 下一阶段执行已预注册的白盒 Elastic Net 与样条 GAM；仍不读取 test。
5. v15 的 Obach 来源集中问题在白盒之后作为独立来源敏感性分析，不以继续调参替代。

## 产物

- `table_1_seed_level_metrics.csv`：全部种子、算法和范围的指标。
- `table_2_three_seed_summary.csv`：三种子均值与标准差。
- `table_3_ensemble_metrics.csv`：三种子及六模型集成指标。
- `table_4_paired_molecule_bootstrap.csv`：RDKit2D 与组合特征逐分子配对区间。
"""


def run():
    parser = base_parser('Thalf v15 descriptor three-seed analysis without test access')
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    configure_logging(args.root, 'analyze_thalf_descriptor_seeds')
    output = args.output or args.root / 'results/analysis/thalf_v15_rdkit2d_three_seed_v1'
    startup_self_check(output=output)
    if args.bootstrap < 200:
        raise ValueError('--bootstrap 必须至少为 200')
    all_runs = {}
    for family, (_, expected_feature_set, _) in FAMILIES.items():
        all_runs[family] = validate_runs(run_paths(args.root, family), expected_feature_set)
    seed = seed_metrics(all_runs)
    summary = summarize_seed_metrics(seed)
    ensemble, predictions = ensemble_tables(all_runs)
    paired = paired_bootstrap(predictions, args.bootstrap)
    if args.check_only:
        print('Validated six training runs, prediction alignment, test isolation, ensembles, and paired bootstrap inputs.')
        return
    inputs = {}
    for runs in all_runs.values():
        inputs.update({str(path.resolve()): sha256(path / 'complete.json') for path, *_ in runs})
    with stage_output(output) as out:
        seed.to_csv(out / 'table_1_seed_level_metrics.csv', index=False, float_format='%.6f')
        summary.to_csv(out / 'table_2_three_seed_summary.csv', index=False, float_format='%.6f')
        ensemble.to_csv(out / 'table_3_ensemble_metrics.csv', index=False, float_format='%.6f')
        paired.to_csv(out / 'table_4_paired_molecule_bootstrap.csv', index=False, float_format='%.6f')
        (out / 'analysis_report.md').write_text(report_text(summary, ensemble, paired), encoding='utf-8')
        finish_stage(out, 'thalf_descriptor_three_seed_analysis', inputs=inputs,
                     task_id=TASK, seeds=list(SEEDS), families=list(FAMILIES),
                     bootstrap_replicates=args.bootstrap, bootstrap_seed=20260913,
                     test_evaluated=False, partial=False)
    print(f'分析产物: {output}')


if __name__ == '__main__':
    run_cli(run)

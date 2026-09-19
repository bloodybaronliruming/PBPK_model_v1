"""Analyze Thalf v15 source sensitivity using OOF and validation predictions only."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_thalf_descriptor_seeds import (FAMILIES, SEEDS, TASK,
                                            read_ensemble_predictions,
                                            run_paths, validate_runs)
from pipeline_common import (base_parser, configure_logging, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)


OBACH_DOI = '10.1124/dmd.108.020479'
WHITEBOX_RUN = Path('models/whitebox') / TASK / 'v15_whitebox_preregistered_v2'
MODEL_SPECS = {
    'rdkit2d_extratrees_3seed': ('rdkit2d', ('extratrees',)),
    'combined_rf_extratrees_6model': ('ecfp4_rdkit2d', ('rf', 'extratrees')),
}
WHITEBOX_SPECS = {
    'whitebox_elasticnet_rdkit2d': 'elasticnet_rdkit2d',
    'whitebox_elasticnet_mechanism2d': 'elasticnet_mechanism2d',
}


def normalize_doi(value):
    if pd.isna(value):
        return ''
    text = str(value).strip().lower()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'doi:'):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def parse_activity_ids(value):
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f'无效 source_activity_ids: {value}')
    return [int(item) for item in parsed]


def build_source_map(root):
    task_path = root / 'data/processed_v15/datasets/tasks' / f'{TASK}.csv'
    source_path = root / 'data/processed_v15/datasets/source_long.csv'
    task = pd.read_csv(task_path, usecols=[
        'row_id', 'molecule_id', 'split', 'doc_id', 'source_activity_ids'])
    source = pd.read_csv(source_path, usecols=[
        'activity_id', 'doc_id', 'doi', 'status', 'task_id'], low_memory=False)
    source = source.loc[(source.task_id == TASK) & (source.status == 'accepted')].copy()
    source['doi_normalized'] = source.doi.map(normalize_doi)
    activity = source.set_index('activity_id')[['doc_id', 'doi_normalized']]
    if activity.index.duplicated().any():
        raise ValueError('accepted source_long 中 activity_id 不唯一')

    rows = []
    for record in task.itertuples(index=False):
        ids = parse_activity_ids(record.source_activity_ids)
        missing = sorted(set(ids) - set(activity.index))
        if missing:
            raise ValueError(f'任务记录引用未接受或不存在的 activity: {missing[:5]}')
        matched = activity.loc[ids]
        doc_ids = set(matched.doc_id.astype(str))
        dois = {item for item in matched.doi_normalized if item}
        if len(doc_ids) != 1 or str(record.doc_id) not in doc_ids:
            raise ValueError(f'任务记录文献映射不一致: {record.row_id}')
        if len(dois) > 1:
            raise ValueError(f'同一任务记录映射多个 DOI: {record.row_id}')
        doi = next(iter(dois), '')
        rows.append({
            'row_id': record.row_id,
            'molecule_id': record.molecule_id,
            'split': record.split,
            'doc_id': str(record.doc_id),
            'doi': doi,
            'source_key': f'doi:{doi}' if doi else f'doc_id:{record.doc_id}',
            'source_activity_count': len(ids),
            'source_group': 'obach' if doi == OBACH_DOI else 'non_obach',
        })
    result = pd.DataFrame(rows)
    if result.row_id.duplicated().any():
        raise ValueError('任务 row_id 不唯一')
    return result, task_path, source_path


def read_whitebox_predictions(path, member, scope):
    name = 'oof_predictions.csv' if scope == 'oof' else 'val_predictions.csv'
    frame = pd.read_csv(path / member / name, usecols=[
        'row_id', 'molecule_id', 'observed_physical', 'predicted_transformed'])
    frame['observed_transformed'] = np.log10(frame.observed_physical.to_numpy(float))
    return frame


def read_models(root):
    all_runs = {
        family: validate_runs(run_paths(root, family), expected_feature_set)
        for family, (_, expected_feature_set, _) in FAMILIES.items()
    }
    whitebox = root / WHITEBOX_RUN
    meta = verify_stage(whitebox, 'thalf_whitebox_training')
    if meta.get('test_evaluated'):
        raise ValueError('白盒产物已评估 test，不适用于本分析')
    predictions = {}
    for label, (family, algorithms) in MODEL_SPECS.items():
        for scope in ('oof', 'val'):
            predictions[(label, scope)] = read_ensemble_predictions(
                all_runs[family], algorithms, scope)
    for label, member in WHITEBOX_SPECS.items():
        for scope in ('oof', 'val'):
            predictions[(label, scope)] = read_whitebox_predictions(whitebox, member, scope)
    return predictions, all_runs, whitebox


def attach_sources(predictions, source_map):
    attached = {}
    for key, frame in predictions.items():
        merged = frame.merge(source_map, on=['row_id', 'molecule_id'], validate='one_to_one')
        if len(merged) != len(frame) or merged.source_key.isna().any():
            raise ValueError(f'预测来源映射不完整: {key}')
        expected_split = 'train' if key[1] == 'oof' else 'val'
        if set(merged.split) != {expected_split}:
            raise ValueError(f'预测 split 不符合预期: {key}')
        merged['squared_error'] = (
            merged.predicted_transformed - merged.observed_transformed) ** 2
        merged['absolute_error'] = np.abs(
            merged.predicted_transformed - merged.observed_transformed)
        attached[key] = merged
    return attached


def metric(frame, weighting='molecule'):
    if frame.empty:
        return None
    work = frame.copy()
    if weighting == 'molecule':
        weights = 1. / work.groupby('molecule_id').molecule_id.transform('size').to_numpy()
    elif weighting == 'source':
        molecule_size = work.groupby(['source_key', 'molecule_id']).molecule_id.transform('size').to_numpy()
        source_molecules = work.groupby('source_key').molecule_id.transform('nunique').to_numpy()
        weights = 1. / (work.source_key.nunique() * source_molecules * molecule_size)
    else:
        raise ValueError(weighting)
    weights = weights / weights.sum()
    absolute = work.absolute_error.to_numpy(float)
    squared = work.squared_error.to_numpy(float)
    return {
        'log10_rmse': float(np.sqrt(np.sum(weights * squared))),
        'log10_mae': float(np.sum(weights * absolute)),
        'gmfe': float(10 ** np.sum(weights * absolute)),
        'within_2fold': float(np.sum(weights * (absolute <= np.log10(2)))),
        'within_3fold': float(np.sum(weights * (absolute <= np.log10(3)))),
        'records': len(work),
        'molecules': int(work.molecule_id.nunique()),
        'sources': int(work.source_key.nunique()),
    }


def stratified_metrics(attached):
    rows = []
    for (model, scope), frame in attached.items():
        for group in ('all', 'obach', 'non_obach'):
            subset = frame if group == 'all' else frame.loc[frame.source_group == group]
            for weighting in ('molecule', 'source'):
                values = metric(subset, weighting)
                if values is not None:
                    rows.append({'model': model, 'scope': scope, 'source_group': group,
                                 'weighting': weighting} | values)
    return pd.DataFrame(rows)


def bootstrap_rmse(frame, source_counts, sources):
    source_index = pd.Series(np.arange(len(sources)), index=sources)
    row_sources = source_index.loc[frame.source_key].to_numpy(int)
    multiplicity = source_counts[:, row_sources]
    squared = frame.squared_error.to_numpy(float)
    mse_sum = np.zeros(len(source_counts), dtype=float)
    molecule_count = np.zeros(len(source_counts), dtype=int)
    for indices in frame.groupby('molecule_id', sort=False).indices.values():
        selected = multiplicity[:, indices]
        denominator = selected.sum(axis=1)
        present = denominator > 0
        numerator = (selected * squared[indices]).sum(axis=1)
        mse_sum[present] += numerator[present] / denominator[present]
        molecule_count[present] += 1
    if (molecule_count == 0).any():
        raise ValueError('来源 bootstrap 产生空重采样')
    return np.sqrt(mse_sum / molecule_count)


def cluster_bootstrap(attached, repeats, seed):
    rng = np.random.default_rng(seed)
    reference = 'rdkit2d_extratrees_3seed'
    rows = []
    for scope in ('oof', 'val'):
        source_reference = attached[(reference, scope)]
        for group in ('all', 'non_obach'):
            base = source_reference if group == 'all' else source_reference.loc[
                source_reference.source_group == group]
            sources = np.array(sorted(base.source_key.unique()))
            if len(sources) < 2:
                continue
            source_counts = rng.multinomial(
                len(sources), np.full(len(sources), 1. / len(sources)), size=repeats)
            bootstraps = {}
            models = sorted({item[0] for item in attached})
            models = [reference] + [model for model in models if model != reference]
            for model in models:
                frame = attached[(model, scope)]
                if group != 'all':
                    frame = frame.loc[frame.source_group == group]
                by_source = {key: part for key, part in frame.groupby('source_key')}
                if set(by_source) != set(sources):
                    raise ValueError(f'模型间来源集合不一致: {model} {scope} {group}')
                observed = metric(frame, 'molecule')['log10_rmse']
                boot = bootstrap_rmse(frame, source_counts, sources)
                bootstraps[model] = boot
                rows.append({
                    'model': model, 'reference': '', 'scope': scope,
                    'source_group': group, 'sources': len(sources),
                    'estimate_log10_rmse': observed,
                    'delta_candidate_minus_reference': np.nan,
                    'bootstrap_95_ci_lower': float(np.quantile(boot, .025)),
                    'bootstrap_95_ci_upper': float(np.quantile(boot, .975)),
                    'bootstrap_replicates': repeats,
                })
                if model == reference:
                    continue
                delta = boot - bootstraps[reference]
                rows.append({
                    'model': model, 'reference': reference, 'scope': scope,
                    'source_group': group, 'sources': len(sources),
                    'estimate_log10_rmse': observed,
                    'delta_candidate_minus_reference': (
                        observed - metric(base, 'molecule')['log10_rmse']),
                    'bootstrap_95_ci_lower': float(np.quantile(delta, .025)),
                    'bootstrap_95_ci_upper': float(np.quantile(delta, .975)),
                    'bootstrap_replicates': repeats,
                })
    return pd.DataFrame(rows)


def source_composition(source_map, attached):
    used = pd.concat([
        frame[['row_id']] for frame in attached.values()
    ], ignore_index=True).drop_duplicates()
    frame = source_map.merge(used, on='row_id', validate='one_to_one')
    return (frame.groupby(['split', 'source_group'], as_index=False)
            .agg(records=('row_id', 'size'), molecules=('molecule_id', 'nunique'),
                 sources=('source_key', 'nunique')))


def report_text(metrics, composition, bootstrap):
    primary = metrics.loc[(metrics.model == 'rdkit2d_extratrees_3seed')
                          & (metrics.weighting == 'molecule')]
    def value(scope, group):
        row = primary.loc[(primary.scope == scope) & (primary.source_group == group)].iloc[0]
        return row.log10_rmse, int(row.molecules), int(row.sources)
    oof_all = value('oof', 'all')
    oof_non = value('oof', 'non_obach')
    val_all = value('val', 'all')
    val_non = value('val', 'non_obach')
    return f"""# Thalf v15 来源敏感性分析

## 数据隔离

本分析只读取训练 OOF 与 validation 预测。任务表只读取来源标识列，不读取 `target_value`；未读取或生成 test 预测与指标。

## 主要结果

RDKit2D ExtraTrees 三种子集成的整体 OOF/validation log10 RMSE 为 `{oof_all[0]:.4f}/{val_all[0]:.4f}`。非 Obach 子集分别为 `{oof_non[0]:.4f}`（{oof_non[1]} 个分子、{oof_non[2]} 个来源）和 `{val_non[0]:.4f}`（{val_non[1]} 个分子、{val_non[2]} 个来源）。

`table_2_stratified_metrics.csv` 同时报告普通分子等权和文献等权结果；后者是评估敏感性，不等同于来源均衡重训。`table_3_source_cluster_bootstrap.csv` 以文献为簇给出区间及相对主候选的配对差异。Obach 本身只有一个来源，因此不为 Obach-only 伪造来源簇区间。

## 解释边界

非 Obach 子集样本和来源数量有限，尤其 validation，点估计只能用于发现来源依赖，不能充当充分的独立外部验证。若来源等权指标、非 Obach 指标或文献簇区间显示明显退化，下一步应优先进行来源均衡重训和 publication-held-out 评估；否则进入灰盒机制残差模型，但仍保留来源集中限制。
"""


def run():
    parser = base_parser('Thalf v15 source sensitivity analysis without test access')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260913)
    args = parser.parse_args()
    configure_logging(args.root, 'analyze_Thalf_source_sensitivity')
    output = args.output or args.root / 'results/analysis/thalf_v15_source_sensitivity_v1'
    startup_self_check(output=output)
    if args.bootstrap < 200:
        raise ValueError('--bootstrap 必须至少为 200')
    source_map, task_path, source_path = build_source_map(args.root)
    predictions, all_runs, whitebox = read_models(args.root)
    attached = attach_sources(predictions, source_map)
    composition = source_composition(source_map, attached)
    metrics = stratified_metrics(attached)
    if args.check_only:
        print('Validated source mapping, model predictions, split isolation, and bootstrap inputs.')
        return
    bootstrap = cluster_bootstrap(attached, args.bootstrap, args.bootstrap_seed)
    inputs = {
        str(task_path.resolve()): sha256(task_path),
        str(source_path.resolve()): sha256(source_path),
        str((whitebox / 'complete.json').resolve()): sha256(whitebox / 'complete.json'),
    }
    for runs in all_runs.values():
        for path, *_ in runs:
            inputs[str((path / 'complete.json').resolve())] = sha256(path / 'complete.json')
    with stage_output(output) as out:
        source_map.loc[source_map.split.isin(['train', 'val'])].to_csv(
            out / 'table_1_prediction_source_map.csv', index=False)
        composition.to_csv(out / 'table_1b_source_composition.csv', index=False)
        metrics.to_csv(out / 'table_2_stratified_metrics.csv', index=False, float_format='%.6f')
        bootstrap.to_csv(out / 'table_3_source_cluster_bootstrap.csv', index=False,
                         float_format='%.6f')
        (out / 'analysis_report.md').write_text(
            report_text(metrics, composition, bootstrap), encoding='utf-8')
        finish_stage(out, 'thalf_source_sensitivity_analysis', inputs=inputs,
                     task_id=TASK, bootstrap_replicates=args.bootstrap,
                     bootstrap_seed=args.bootstrap_seed, test_evaluated=False, partial=False)
    print(f'分析产物: {output}')


if __name__ == '__main__':
    run_cli(run)

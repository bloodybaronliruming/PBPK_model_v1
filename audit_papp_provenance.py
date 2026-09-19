"""只读审计人源 Caco-2 A→B Papp 的高值记录、单位和来源链路。

默认只审计 train/val，避免在模型选择前查看测试标签。该程序不会修改数据集、固定划分或模型。
"""
from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (base_parser, configure_logging, dump_json, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)


TASK_ID = 'Papp__human__caco2_ab'
SOURCE_COLUMNS = [
    'activity_id', 'assay_id', 'doc_id', 'molecule_chembl_id', 'assay_chembl_id', 'doi', 'pubmed_id',
    'standard_type', 'standard_value', 'standard_units', 'standard_relation', 'canonical_value',
    'canonical_unit', 'conversion_rule', 'description', 'assay_cell_type', 'assay_organism',
    'assay_parameters_json', 'pH', 'route', 'status', 'reason', 'relation',
]


def ids(value):
    parsed = ast.literal_eval(value) if isinstance(value, str) and value else []
    if not isinstance(parsed, list):
        raise ValueError(f'无效 source_activity_ids: {value!r}')
    return [int(item) for item in parsed]


def summary(frame, by, name):
    grouped = frame.groupby(by, dropna=False).agg(
        source_activity_records=('row_id', 'size'), molecules=('molecule_id', 'nunique'),
        high_source_activity_records=('is_high_value', 'sum'),
        value_min=('raw_value', 'min'), value_median=('raw_value', 'median'), value_max=('raw_value', 'max'),
    ).reset_index().sort_values(['high_source_activity_records', 'source_activity_records'], ascending=False)
    grouped.to_csv(name, index=False)
    return grouped


def audit(datasets, output, threshold, splits):
    meta = verify_stage(datasets, 'datasets')
    startup_self_check([datasets / 'tasks' / f'{TASK_ID}.csv', datasets / 'source_long.csv'], output=output)
    task = pd.read_csv(datasets / 'tasks' / f'{TASK_ID}.csv', keep_default_na=False)
    source = pd.read_csv(datasets / 'source_long.csv', keep_default_na=False, low_memory=False)
    chosen = task.loc[task.split.isin(splits)].copy()
    if chosen.empty:
        raise ValueError(f'没有属于 splits={splits} 的 Papp 记录')
    chosen['raw_value'] = pd.to_numeric(chosen.raw_value, errors='raise')
    if (chosen.raw_value <= 0).any() or not np.isfinite(chosen.raw_value).all():
        raise ValueError('Papp 精确记录必须是有限正值')
    chosen['is_high_value'] = chosen.raw_value.ge(threshold)
    chosen['source_activity_id_list'] = chosen.source_activity_ids.map(ids)
    chosen = chosen.explode('source_activity_id_list').rename(columns={'source_activity_id_list': 'activity_id'})
    chosen['activity_id'] = chosen.activity_id.astype(str)
    relevant = source.loc[(source.task_id == TASK_ID) & source.activity_id.astype(str).isin(chosen.activity_id)].copy()
    relevant['activity_id'] = relevant.activity_id.astype(str)
    if relevant.activity_id.duplicated().any():
        raise ValueError('同一 Papp source activity 出现多次，拒绝产生模糊谱系')
    lineage_columns = [c for c in SOURCE_COLUMNS if c in relevant]
    source_detail = relevant[lineage_columns].rename(columns={c: f'source_{c}' for c in lineage_columns if c != 'activity_id'})
    lineage = chosen.merge(source_detail, on='activity_id', how='left', validate='many_to_one')
    missing = lineage.source_canonical_value.isna()
    if missing.any():
        raise ValueError(f'{int(missing.sum())} 条任务记录缺少 source_long 来源；请先修复数据版本')
    lineage['source_canonical_value'] = pd.to_numeric(lineage.source_canonical_value, errors='raise')
    # 一个任务记录可以是同一 molecule/task/assay/document 的重复测定中位数，
    # 因而应与其完整 activity 谱系的中位数比较，而不是逐行强制相等。
    source_median = lineage.groupby('row_id', sort=False).source_canonical_value.transform('median')
    if not np.allclose(lineage.raw_value, source_median, rtol=1e-12, atol=1e-12):
        raise ValueError('任务标签与其完整来源 activity 的中位数不一致，拒绝把该数据版本用于结论')
    if not lineage.source_status.eq('accepted').all() or not lineage.source_relation.eq('=').all():
        raise ValueError('Papp 任务出现非 accepted 或非精确来源记录')
    with stage_output(output) as out:
        lineage.sort_values(['is_high_value', 'raw_value'], ascending=[False, False]).to_csv(out / 'papp_record_lineage.csv', index=False)
        lineage.loc[lineage.is_high_value].sort_values('raw_value', ascending=False).to_csv(out / 'papp_high_value_records.csv', index=False)
        summary(lineage, ['source_standard_type', 'source_standard_units', 'canonical_unit', 'source_conversion_rule'], out / 'unit_conversion_summary.csv')
        documents = summary(lineage, ['source_doc_id', 'source_doi', 'source_pubmed_id'], out / 'document_summary.csv')
        decisions = documents.loc[documents.high_source_activity_records.gt(0)].copy()
        decisions['reviewer_decision'] = ''
        decisions['source_table_or_page'] = ''
        decisions['evidence_note'] = ''
        decisions['reviewer'] = ''
        decisions['review_date'] = ''
        decisions.to_csv(out / 'high_value_document_decision_template.csv', index=False)
        summary(lineage, ['source_assay_id', 'source_assay_chembl_id', 'source_assay_cell_type', 'pH'], out / 'assay_summary.csv')
        direction = lineage[['row_id', 'activity_id', 'split', 'raw_value', 'is_high_value', 'source_description', 'source_standard_type',
                             'source_assay_cell_type', 'source_assay_parameters_json', 'source_reason']].copy()
        direction['ab_classification'] = 'accepted_caco2_ab_by_endpoint_rules'
        direction.to_csv(out / 'direction_evidence.csv', index=False)
        source_unit = lineage.source_standard_units.fillna('').str.strip().replace('', '<missing>')
        review = {
            'task_id': TASK_ID,
            'included_splits': splits,
            'test_labels_included': 'test' in splits,
            'threshold_canonical_1e-6_cm_s': threshold,
            'task_records': int(lineage.row_id.nunique()),
            'source_activity_records': int(len(lineage)),
            'molecules': int(lineage.molecule_id.nunique()),
            'high_task_records': int(lineage.loc[lineage.is_high_value, 'row_id'].nunique()),
            'high_source_activity_records': int(lineage.is_high_value.sum()),
            'high_molecules': int(lineage.loc[lineage.is_high_value, 'molecule_id'].nunique()),
            'raw_value_min': float(lineage.raw_value.min()),
            'raw_value_median': float(lineage.raw_value.median()),
            'raw_value_p99': float(lineage.raw_value.quantile(.99)),
            'raw_value_max': float(lineage.raw_value.max()),
            'source_units': source_unit.value_counts(dropna=False).to_dict(),
            'all_exact_accepted': True,
            'lineage_complete': True,
            'conclusion': ('Review high-value source records before changing labels. This audit itself does not classify values as errors '
                           'and does not modify the dataset.'),
        }
        dump_json(out / 'audit_summary.json', review)
        finish_stage(out, 'papp_provenance_audit', inputs={'datasets_complete_sha256': sha256(datasets / 'complete.json')},
                     partial=False, test_evaluated=False, task_id=TASK_ID, included_splits=splits,
                     high_value_threshold=threshold)
    logging.info('Papp 来源审计完成: %s；%d 条任务记录/%d 条来源 activity，%d 条高值任务记录（>= %g）。',
                 output, lineage.row_id.nunique(), len(lineage), lineage.loc[lineage.is_high_value, 'row_id'].nunique(), threshold)


def main():
    p = base_parser(__doc__)
    p.add_argument('--datasets-dir', type=Path, help='已完成的 datasets 阶段目录')
    p.add_argument('--threshold', type=float, default=1e4, help='高值阈值，规范单位 1e-6 cm/s')
    p.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'],
                   help='默认不读取测试标签；如需质量控制全量审计，显式指定 train val test')
    args = p.parse_args()
    configure_logging(args.root, 'audit_papp_provenance')
    if not np.isfinite(args.threshold) or args.threshold <= 0:
        raise ValueError('--threshold 必须为有限正数')
    datasets = args.datasets_dir or args.root / 'data/processed_v2/datasets'
    output = args.output or args.root / 'results/analysis/papp_provenance_audit_v1'
    verify_stage(datasets, 'datasets')
    startup_self_check(output=output)
    if args.check_only:
        logging.info('Papp 数据集阶段、任务文件和来源文件自检通过；未读取标签和未生成审计产物。')
        return
    audit(datasets, output, args.threshold, args.splits)


if __name__ == '__main__':
    run_cli(main)

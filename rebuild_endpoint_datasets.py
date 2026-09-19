"""第二步：来源审计通过的精确测定形成新任务视图，不覆盖旧标签。"""
from __future__ import annotations
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pipeline_common import (base_parser, configure_logging, verify_stage, verify_fixed_inputs, startup_self_check,
                             stage_output, finish_stage, dump_json, stable_id, sha256, run_cli)


def transform_values(y, transform):
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('非有限标签')
    if transform == 'log10':
        if (y <= 0).any():
            raise ValueError('log10 要求正值')
        return np.log10(y)
    if transform == 'logit':
        if ((y <= 0) | (y >= 1)).any():
            raise ValueError('fu 内部值必须严格处于 (0,1)，边界需单独审核')
        return np.log(y) - np.log1p(-y)
    if transform == 'identity':
        return y.copy()
    raise ValueError(transform)


def rebuild(root, audit_dir, output):
    meta = verify_stage(audit_dir, 'audit')
    verify_fixed_inputs(meta, root)
    startup_self_check(output=output)
    source = pd.read_csv(audit_dir / 'source_records.csv', keep_default_na=False)
    accepted = source.loc[source.status.eq('accepted')].copy()
    if accepted.empty:
        raise ValueError('无已确认精确测定；先查看 task_audit_counts.csv，不得放宽为默认猜测。')
    if accepted.activity_id.duplicated().any():
        raise ValueError('同一 activity 被分配到多个来源/任务')
    accepted['canonical_value'] = pd.to_numeric(accepted.canonical_value, errors='raise')
    if not np.isfinite(accepted.canonical_value).all() or not accepted.relation.eq('=').all():
        raise ValueError('accepted 表中出现非精确/非有限标签')
    registry, rows = {}, []
    # 同一 assay/doc 内去重复测量；不同实验条件保留独立行，不按分子跨实验直接平均。
    groups = accepted.groupby(['molecule_id', 'task_id', 'assay_id', 'doc_id'], sort=True, dropna=False)
    for key, group in tqdm(groups, total=len(groups), desc='重建测定级任务视图'):
        first = group.iloc[0]
        transform = 'logit' if first.endpoint == 'fu' else 'identity' if first.endpoint == 'F' else 'log10'
        for col in ['canonical_unit', 'split', 'system', 'species', 'smiles']:
            if group[col].nunique(dropna=False) != 1:
                raise ValueError(f'同一聚合组 {key} 的 {col} 不一致')
        y = float(group.canonical_value.median())
        row = dict(row_id=stable_id(json.dumps([str(k) for k in key])), molecule_id=first.molecule_id,
                   smiles=first.smiles, split=first.split, task_id=first.task_id, endpoint=first.endpoint,
                   species=first.species, system=first.system, canonical_unit=first.canonical_unit,
                   raw_value=y, target_value=float(transform_values([y], transform)[0]), mask=1,
                   assay_id=first.assay_id, doc_id=first.doc_id, pH=first.pH, route=first.route,
                   source_activity_ids=json.dumps(sorted(group.activity_id.astype(int).unique().tolist())),
                   source_n=len(group), source_min=float(group.canonical_value.min()), source_max=float(group.canonical_value.max()))
        rows.append(row)
        registry[first.task_id] = dict(endpoint=first.endpoint, species=first.species, system=first.system,
                                      unit=first.canonical_unit, transform=transform,
                                      target_space='transformed_unstandardized', standardization='fit_only_in_each_training_fold',
                                      exact_only=True, aggregation='median_within_molecule_task_assay_document')
    rebuilt = pd.DataFrame(rows)
    with stage_output(output) as out:
        source.to_csv(out / 'source_long.csv', index=False)
        source.loc[source.status.eq('review')].to_csv(out / 'review_records.csv', index=False)
        source.loc[source.status.eq('censored')].to_csv(out / 'censored_records.csv', index=False)
        rebuilt.to_csv(out / 'task_records.csv', index=False)
        (out / 'tasks').mkdir()
        for task, data in rebuilt.groupby('task_id', sort=True):
            data.to_csv(out / 'tasks' / f'{task}.csv', index=False)
        rebuilt.groupby(['task_id', 'split']).agg(records=('row_id', 'size'), molecules=('molecule_id', 'nunique')).reset_index().to_csv(out / 'task_counts.csv', index=False)
        dump_json(out / 'task_registry.json', registry)
        finish_stage(out, 'datasets', inputs={'audit_complete_sha256': sha256(audit_dir / 'complete.json')},
                     fixed_split_hashes=meta['fixed_split_hashes'], source_audit_dir=str(audit_dir.resolve()),
                     rule_version=meta['rule_version'], partial=False)
    logging.info('新数据视图完成: %s；%d 个任务，%d 条测定聚合记录。', output, len(registry), len(rebuilt))


def main():
    p = base_parser(__doc__)
    p.add_argument('--audit-dir', type=Path)
    args = p.parse_args()
    configure_logging(args.root, 'rebuild_endpoint_datasets')
    audit_dir = args.audit_dir or args.root / 'data/processed_v2/audit'
    output = args.output or args.root / 'data/processed_v2/datasets'
    meta = verify_stage(audit_dir, 'audit')
    verify_fixed_inputs(meta, args.root)
    startup_self_check(output=output)
    if args.check_only:
        logging.info('前序审计完整性自检通过。')
        return
    rebuild(args.root, audit_dir, output)


if __name__ == '__main__':
    run_cli(main)

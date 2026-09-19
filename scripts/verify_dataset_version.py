"""校验两个已发布数据版本的阶段完整性、固定划分和任务容量变化。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pipeline_common import ROOT, configure_logging, run_cli, sha256, verify_stage


DEFAULT_TASKS = ('F__human__absolute_oral', 'Thalf__human__terminal_iv')


def parse_expected_delta(value: str) -> tuple[str, str, int, int]:
    """Parse TASK:SPLIT:RECORDS:MOLECULES without constraining task-id syntax."""
    parts = value.rsplit(':', 3)
    if len(parts) != 4 or not parts[0] or parts[1] not in {'train', 'val', 'test'}:
        raise ValueError(f'无效 --expect-delta: {value}; 格式为 TASK:SPLIT:RECORDS:MOLECULES')
    try:
        return parts[0], parts[1], int(parts[2]), int(parts[3])
    except ValueError as exc:
        raise ValueError(f'--expect-delta 的 records/molecules 必须为整数: {value}') from exc


def task_counts(version_dir: Path) -> pd.DataFrame:
    counts = pd.read_csv(version_dir / 'datasets' / 'task_counts.csv')
    required = {'task_id', 'split', 'records', 'molecules'}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f'{version_dir} task_counts.csv 缺少列: {sorted(missing)}')
    if counts.duplicated(['task_id', 'split']).any():
        raise ValueError(f'{version_dir} task_counts.csv 的 task/split 不唯一')
    return counts[['task_id', 'split', 'records', 'molecules']].copy()


def compare_task_counts(baseline: pd.DataFrame, candidate: pd.DataFrame, tasks: tuple[str, ...]) -> pd.DataFrame:
    key = pd.MultiIndex.from_product([tasks, ('train', 'val', 'test')], names=['task_id', 'split'])
    left = baseline.set_index(['task_id', 'split']).reindex(key, fill_value=0)
    right = candidate.set_index(['task_id', 'split']).reindex(key, fill_value=0)
    result = pd.DataFrame({
        'baseline_records': left.records.astype(int),
        'candidate_records': right.records.astype(int),
        'delta_records': (right.records - left.records).astype(int),
        'baseline_molecules': left.molecules.astype(int),
        'candidate_molecules': right.molecules.astype(int),
        'delta_molecules': (right.molecules - left.molecules).astype(int),
    }).reset_index()
    return result


def verify_versions(baseline_dir: Path, candidate_dir: Path, tasks: tuple[str, ...], expected: tuple[tuple[str, str, int, int], ...]) -> pd.DataFrame:
    for version in (baseline_dir, candidate_dir):
        verify_stage(version / 'audit', 'audit')
        verify_stage(version / 'datasets', 'datasets')
        verify_stage(version / 'splits', 'splits')

    baseline_manifest = sha256(baseline_dir / 'splits' / 'split_manifest.csv')
    candidate_manifest = sha256(candidate_dir / 'splits' / 'split_manifest.csv')
    if baseline_manifest != candidate_manifest:
        raise ValueError('固定划分清单哈希不一致；禁止在不同划分上比较容量或模型。')

    report = compare_task_counts(task_counts(baseline_dir), task_counts(candidate_dir), tasks)
    for task_id, split, records, molecules in expected:
        row = report.loc[report.task_id.eq(task_id) & report.split.eq(split)]
        if len(row) != 1:
            raise ValueError(f'期望增量对应任务不存在: {task_id}/{split}')
        actual = row.iloc[0]
        if (actual.delta_records, actual.delta_molecules) != (records, molecules):
            raise ValueError(
                f'{task_id}/{split} 增量为 records={actual.delta_records}, molecules={actual.delta_molecules}；'
                f'期望 records={records}, molecules={molecules}'
            )
    logging.info('固定划分清单 SHA-256 一致: %s', baseline_manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--baseline-dir', type=Path, required=True, help='已核验的基准 processed_vN 目录')
    parser.add_argument('--candidate-dir', type=Path, required=True, help='待核验的 processed_vN 目录')
    parser.add_argument('--tasks', default=','.join(DEFAULT_TASKS), help='逗号分隔的 task_id；默认 F 与 Thalf 人源任务')
    parser.add_argument('--expect-delta', action='append', default=[], metavar='TASK:SPLIT:RECORDS:MOLECULES',
                        help='可重复；断言某任务/集合的记录和分子增量')
    args = parser.parse_args()
    configure_logging(args.root, 'verify_dataset_version')
    tasks = tuple(task.strip() for task in args.tasks.split(',') if task.strip())
    if not tasks:
        raise ValueError('--tasks 至少指定一个任务')
    expected = tuple(parse_expected_delta(item) for item in args.expect_delta)
    report = verify_versions(args.baseline_dir, args.candidate_dir, tasks, expected)
    print(report.to_string(index=False))
    logging.info('版本核验通过: %s -> %s', args.baseline_dir, args.candidate_dir)


if __name__ == '__main__':
    run_cli(main)

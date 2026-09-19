"""Freeze the pre-selected Thalf v15 RDKit2D ExtraTrees three-seed ensemble."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

from pipeline_common import (ROOT, base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check, verify_stage)


TASK = 'Thalf__human__terminal_iv'
SEEDS = (2026, 2027, 2028)
RUN_NAMES = tuple(f'v15_rdkit2d_seed{seed}' for seed in SEEDS)
SELECTION_STAGES = {
    'descriptor_analysis': ('results/analysis/thalf_v15_rdkit2d_three_seed_v1',
                            'thalf_descriptor_three_seed_analysis'),
    'whitebox': ('models/whitebox/Thalf__human__terminal_iv/v15_whitebox_preregistered_v2',
                 'thalf_whitebox_training'),
    'source_sensitivity': ('results/analysis/thalf_v15_source_sensitivity_v1',
                           'thalf_source_sensitivity_analysis'),
    'gpu_boosting': ('models/stl/Thalf__human__terminal_iv/v15_gpu_boosting_seed2026',
                     'stl_training'),
    'greybox': ('models/greybox/Thalf__human__terminal_iv/v15_greybox_preregistered_v1',
                'thalf_greybox_training'),
}
TDC_BENCHMARK = ('results/benchmarks/tdc_half_life_obach_rdkit2d_et_v2',
                 'tdc_half_life_obach_benchmark')


def timestamp(value):
    return datetime.fromisoformat(value)


def canonical_smiles(value):
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f'无法规范化结构: {value}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def tdc_overlap(root):
    records = pd.read_csv(root / 'data/processed_v15/datasets/task_records.csv',
                          usecols=['molecule_id', 'smiles', 'split', 'task_id'])
    manifest = pd.read_csv(root / 'data/processed_v15/splits/split_manifest.csv',
                           usecols=['molecule_id', 'eligible'])
    if manifest.molecule_id.duplicated().any():
        raise ValueError('split manifest 的 molecule_id 不唯一')
    test = records.loc[records.task_id.eq(TASK) & records.split.eq('test'),
                       ['molecule_id', 'smiles']].drop_duplicates('molecule_id')
    test = test.merge(manifest, on='molecule_id', how='left', validate='one_to_one')
    test = test.loc[test.eligible.eq(True), ['molecule_id', 'smiles']].copy()
    if len(test) != 53:
        raise ValueError(f'预期 53 个 v15 test 分子，实际 {len(test)}')
    base = root / 'data/external/tdc_benchmark/admet_group/half_life_obach'
    train_val = pd.read_csv(base / 'train_val.csv', usecols=['Drug'])
    tdc_test = pd.read_csv(base / 'test.csv', usecols=['Drug'])
    train_structures = set(train_val.Drug.map(canonical_smiles))
    test_structures = set(tdc_test.Drug.map(canonical_smiles))
    if train_structures & test_structures:
        raise ValueError('TDC train_val 与 test 存在规范化结构重叠')
    test['canonical_smiles'] = test.smiles.map(canonical_smiles)
    test['tdc_overlap'] = np.select(
        [test.canonical_smiles.isin(train_structures), test.canonical_smiles.isin(test_structures)],
        ['tdc_train_val', 'tdc_test'], default='not_in_tdc')
    counts = test.tdc_overlap.value_counts().reindex(
        ['tdc_train_val', 'tdc_test', 'not_in_tdc'], fill_value=0).astype(int).to_dict()
    return test[['molecule_id', 'canonical_smiles', 'tdc_overlap']], counts


def validate_member(root, run_name, seed):
    run = root / 'models/stl' / TASK / run_name
    meta = verify_stage(run, 'stl_training')
    if (meta.get('task_id') != TASK or meta.get('seed') != seed
            or meta.get('feature_set') != 'rdkit2d' or meta.get('test_evaluated')
            or meta.get('excluded_folds')):
        raise ValueError(f'不符合冻结条件的训练产物: {run}')
    selection = json.loads((run / 'selection.json').read_text(encoding='utf-8'))
    if selection.get('algorithm') != 'extratrees':
        raise ValueError(f'训练 OOF 未选择 ExtraTrees: {run}')
    if list(run.rglob('*test*')):
        raise ValueError(f'冻结前训练目录含 test 文件: {run}')
    bundle = joblib.load(run / 'extratrees/model.joblib')
    if bundle.algorithm != 'extratrees' or bundle.feature_set != 'rdkit2d':
        raise ValueError(f'模型 bundle 契约不匹配: {run}')
    saved = pd.read_csv(run / 'extratrees/val_predictions.csv')
    actual = bundle.predict_smiles(saved.smiles.tolist())
    if not np.allclose(actual, saved.predicted_physical.to_numpy(float), rtol=2e-6, atol=2e-8):
        raise ValueError(f'模型重载预测不一致: {run}')
    return run, meta, bundle


def run():
    parser = base_parser(__doc__)
    args = parser.parse_args()
    configure_logging(args.root, 'freeze_Thalf_v15_candidate')
    output = args.output or args.root / 'models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1'
    required = [args.root / path / 'complete.json' for path, _ in SELECTION_STAGES.values()]
    required += [args.root / TDC_BENCHMARK[0] / 'complete.json']
    required += [args.root / 'data/processed_v15/datasets/complete.json',
                 args.root / 'data/processed_v15/splits/complete.json']
    startup_self_check(required, output=output)

    selection_meta = {}
    for name, (relative, stage) in SELECTION_STAGES.items():
        selection_meta[name] = verify_stage(args.root / relative, stage)
        if selection_meta[name].get('test_evaluated'):
            raise ValueError(f'候选选择阶段读取过 v15 test: {relative}')
    cutoff = max(timestamp(meta['created_utc']) for meta in selection_meta.values())
    tdc_path = args.root / TDC_BENCHMARK[0]
    tdc_meta = verify_stage(tdc_path, TDC_BENCHMARK[1])
    if not tdc_meta.get('benchmark_test_evaluated') or cutoff >= timestamp(tdc_meta['created_utc']):
        raise ValueError('无法证明 v15 候选选择早于 TDC test 评估')

    members = [validate_member(args.root, name, seed)
               for name, seed in zip(RUN_NAMES, SEEDS)]
    expected_inputs = {
        'datasets_complete_sha256': sha256(args.root / 'data/processed_v15/datasets/complete.json'),
        'splits_complete_sha256': sha256(args.root / 'data/processed_v15/splits/complete.json'),
    }
    if any(meta['inputs'] != expected_inputs for _, meta, _ in members):
        raise ValueError('候选模型与 processed_v15 不匹配')
    if any(bundle.train_groups != members[0][2].train_groups for _, _, bundle in members[1:]):
        raise ValueError('三个候选成员的训练骨架不一致')

    ensemble = pd.read_csv(args.root / SELECTION_STAGES['descriptor_analysis'][0]
                           / 'table_3_ensemble_metrics.csv')
    selected = ensemble.loc[(ensemble.family == 'rdkit2d')
                            & (ensemble.ensemble == 'extratrees_3seed')].set_index('scope')
    if set(selected.index) != {'oof', 'val'}:
        raise ValueError('缺少预选择集成指标')
    overlap, overlap_counts = tdc_overlap(args.root)
    registry = {
        'registry_version': 1,
        'task_id': TASK,
        'data_version': 'processed_v15',
        'model_kind': 'stl_seed_ensemble',
        'algorithm': 'extratrees',
        'feature_set': 'rdkit2d',
        'aggregation': 'arithmetic_mean_log10_prediction',
        'seeds': list(SEEDS),
        'members': [{
            'seed': seed,
            'run_dir': str(run.resolve()),
            'run_complete_sha256': sha256(run / 'complete.json'),
            'model_sha256': sha256(run / 'extratrees/model.joblib'),
        } for seed, (run, _, _) in zip(SEEDS, members)],
        'pretest_oof_log10_rmse': float(selected.loc['oof', 'primary_log10_rmse']),
        'pretest_validation_log10_rmse': float(selected.loc['val', 'primary_log10_rmse']),
        'selection_cutoff_utc': cutoff.isoformat(),
        'tdc_benchmark_evaluated_utc': tdc_meta['created_utc'],
        'selection_finalized_before_tdc_benchmark': True,
        'v15_test_labels_evaluated': False,
        'v15_test_structure_overlap_audited': True,
        'tdc_overlap_counts': overlap_counts,
        'selection_reason': (
            'RDKit2D ExtraTrees three-seed ensemble remained the compact main candidate after '
            'pre-TDC descriptor, whitebox, source-sensitivity, boosting, and greybox analyses.'),
        'policy': 'No member, feature, hyperparameter, aggregation, or subset may change after publication.',
    }
    if args.check_only:
        print(f'Freeze contract valid; v15 test labels not evaluated; TDC overlap={overlap_counts}')
        return
    with stage_output(output) as out:
        dump_json(out / 'candidate_registry.json', registry)
        overlap.to_csv(out / 'test_structure_overlap.csv', index=False)
        (out / 'selection_report.md').write_text(
            '# Frozen Thalf v15 candidate\n\n'
            'The candidate is the geometric mean on the physical-hour scale of three RDKit2D '
            'ExtraTrees models (seeds 2026-2028), equivalent to the arithmetic mean in log10 space. '
            'Selection was completed before the TDC benchmark test evaluation. The v15 test labels '
            'have not been scored. Structure-only overlap auditing found 40 v15 test molecules in '
            'TDC train_val, 10 in TDC test, and 3 outside TDC; this limits independence claims but '
            'did not alter candidate selection.\n', encoding='utf-8')
        inputs = {str(run.resolve()): sha256(run / 'complete.json') for run, _, _ in members}
        inputs.update({str((args.root / relative).resolve()): sha256(args.root / relative / 'complete.json')
                       for relative, _ in SELECTION_STAGES.values()})
        inputs[str(tdc_path.resolve())] = sha256(tdc_path / 'complete.json')
        finish_stage(out, 'thalf_v15_frozen_candidate', inputs=inputs, task_id=TASK,
                     seeds=list(SEEDS), feature_set='rdkit2d', algorithm='extratrees',
                     v15_test_labels_evaluated=False, v15_test_structure_overlap_audited=True,
                     selection_finalized_before_tdc_benchmark=True, partial=False)
    print(f'Frozen Thalf v15 candidate: {output}; v15 test labels not evaluated.')


if __name__ == '__main__':
    run_cli(run)

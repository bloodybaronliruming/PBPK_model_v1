"""Evaluate the frozen Thalf v15 ensemble once, without refitting or reselection."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dmpk_toolkit import load_task, metrics, prediction_table
from freeze_Thalf_v15_candidate import TASK, tdc_overlap
from pipeline_common import (ROOT, base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check, verify_stage)


def geometric_ensemble(member_predictions):
    values = np.asarray(member_predictions, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('集成成员预测必须是至少两组正有限值')
    return np.power(10., np.log10(values).mean(axis=0))


def bootstrap_metrics(frame, prediction, repeats=10000, seed=20260913):
    work = pd.DataFrame({
        'molecule_id': frame.molecule_id.to_numpy(),
        'observed': np.log10(frame.raw_value.to_numpy(float)),
        'predicted': np.log10(np.asarray(prediction, dtype=float)),
    }).groupby('molecule_id', sort=True).mean()
    loss = (work.predicted - work.observed).to_numpy() ** 2
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(work), size=(repeats, len(work)))
    rmse = np.sqrt(loss[take].mean(axis=1))
    return {
        'replicates': repeats,
        'seed': seed,
        'log10_rmse_95_ci': [float(np.quantile(rmse, .025)), float(np.quantile(rmse, .975))],
    }


def run():
    parser = base_parser(__doc__)
    parser.add_argument('--registry', type=Path,
                        default=ROOT / 'models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1')
    parser.add_argument('--confirm-frozen-test', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    configure_logging(args.root, 'evaluate_Thalf_v15_frozen')
    output = args.output or args.root / 'results/final/Thalf_v15_rdkit2d_et3_test_v1'
    startup_self_check([args.registry / 'complete.json', args.registry / 'candidate_registry.json'],
                       output=output)
    meta = verify_stage(args.registry, 'thalf_v15_frozen_candidate')
    registry = json.loads((args.registry / 'candidate_registry.json').read_text(encoding='utf-8'))
    if meta.get('v15_test_labels_evaluated') or registry.get('v15_test_labels_evaluated'):
        raise ValueError('冻结注册表必须在 test 标签评估前发布')
    if args.bootstrap < 1000:
        raise ValueError('--bootstrap 必须至少为 1000')
    for member in registry['members']:
        run_dir = Path(member['run_dir'])
        if (sha256(run_dir / 'complete.json') != member['run_complete_sha256']
                or sha256(run_dir / 'extratrees/model.joblib') != member['model_sha256']):
            raise ValueError(f'冻结成员发生改变: {run_dir}')
    if args.check_only:
        print('Frozen registry and member hashes verified; v15 test labels not loaded.')
        return
    if not args.confirm_frozen_test:
        raise ValueError('一次性 test 评估必须显式提供 --confirm-frozen-test')

    frame, spec, _ = load_task(args.root, args.root / 'data/processed_v15/datasets',
                               args.root / 'data/processed_v15/splits', TASK)
    test = frame.loc[frame.split.eq('test')].reset_index(drop=True)
    member_predictions = []
    train_groups = None
    for member in registry['members']:
        bundle = joblib.load(Path(member['run_dir']) / 'extratrees/model.joblib')
        if set(bundle.train_groups) & set(test.scaffold_group):
            raise ValueError('冻结模型与 test 存在训练骨架重叠')
        train_groups = bundle.train_groups if train_groups is None else train_groups
        if bundle.train_groups != train_groups:
            raise ValueError('冻结成员的训练骨架不一致')
        member_predictions.append(bundle.predict_smiles(test.smiles.tolist()))
    prediction = geometric_ensemble(member_predictions)
    result = metrics(test, prediction, spec)
    interval = bootstrap_metrics(test, prediction, args.bootstrap)
    overlap, counts = tdc_overlap(args.root)
    if counts != registry['tdc_overlap_counts']:
        raise ValueError('TDC 结构重叠状态在冻结后发生变化')
    table = prediction_table(test, prediction, spec, 'frozen_test_ensemble', train_groups)
    table = table.merge(overlap[['molecule_id', 'tdc_overlap']], on='molecule_id',
                        how='left', validate='many_to_one')
    subgroup = {}
    for label, group in table.groupby('tdc_overlap', sort=True):
        rows = test.loc[test.molecule_id.isin(group.molecule_id)].copy()
        values = table.set_index('row_id').loc[rows.row_id, 'predicted_physical'].to_numpy(float)
        subgroup[label] = metrics(rows, values, spec)
    payload = {
        'task_id': TASK,
        'model': 'RDKit2D ExtraTrees seeds 2026-2028 log10 ensemble',
        'full_test': result,
        'bootstrap': interval,
        'tdc_overlap_subgroups': subgroup,
        'interpretation': ('Confirmatory for the frozen v15 scaffold split, but not independent of '
                           'the Obach/TDC source universe; subgroup estimates are descriptive.'),
    }
    with stage_output(output) as out:
        table.to_csv(out / 'test_predictions.csv', index=False)
        dump_json(out / 'metrics.json', payload)
        pd.DataFrame([{
            'task_id': TASK,
            'records': result['records'],
            'molecules': result['molecules'],
            'log10_rmse': result['primary'],
            'log10_rmse_ci_lower': interval['log10_rmse_95_ci'][0],
            'log10_rmse_ci_upper': interval['log10_rmse_95_ci'][1],
            'spearman_molecule_mean': result['spearman_molecule_mean'],
            'gmfe': result['gmfe_positive_subset'],
            'within_2fold': result['within_2fold_positive_subset'],
            'within_3fold': result['within_3fold_positive_subset'],
        }]).to_csv(out / 'test_summary.csv', index=False, float_format='%.6f')
        (out / 'analysis_report.md').write_text(
            '# Thalf v15 frozen test evaluation\n\n'
            f"The frozen RDKit2D ExtraTrees three-seed ensemble was evaluated once on "
            f"{result['molecules']} test molecules without refitting or reselection. The primary "
            f"log10 RMSE was {result['primary']:.4f} with molecule-bootstrap 95% CI "
            f"[{interval['log10_rmse_95_ci'][0]:.4f}, {interval['log10_rmse_95_ci'][1]:.4f}]. "
            f"Molecule-level Spearman was {result['spearman_molecule_mean']:.4f}, GMFE was "
            f"{result['gmfe_positive_subset']:.3f}, and the within-twofold fraction was "
            f"{result['within_2fold_positive_subset']:.3f}.\n\n"
            'This is the confirmatory result for the fixed v15 scaffold split. It is not an '
            'independent external validation relative to TDC/Obach: 40 test structures occur in '
            'TDC train_val, 10 in TDC test, and 3 are outside TDC. No model choice or parameter was '
            'changed after test evaluation.\n', encoding='utf-8')
        finish_stage(out, 'thalf_v15_frozen_test_evaluation',
                     inputs={'registry_complete_sha256': sha256(args.registry / 'complete.json')},
                     task_id=TASK, test_evaluated=True, refit=False, selection_changed=False,
                     tdc_overlap_counts=counts, bootstrap_replicates=args.bootstrap, partial=False)
    print(f'Thalf v15 frozen test artifact: {output}')


if __name__ == '__main__':
    run_cli(run)

"""Evaluate a pre-published candidate registry once and create final test tables and figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dmpk_toolkit import load_task, metrics, prediction_table
from freeze_final_candidates import predict_task
from pipeline_common import (ROOT, base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check, verify_stage)
from rebuild_endpoint_datasets import transform_values


LABELS = {
    'fu__human__plasma': 'Plasma fu',
    'Papp__human__caco2_ab': 'Caco-2 Papp',
    'CL__human__systemic_iv': 'Systemic CL',
    'VDss__human__steady_state_iv': 'Steady-state VDss',
    'Thalf__human__terminal_iv': 'Terminal half-life',
}


def selected_validation_primary(candidate):
    if candidate['model_kind'] == 'multitask_seed_ensemble':
        return candidate['pretest_ensemble_validation_primary']
    run = Path(candidate['run_dir'])
    values = json.loads((run / 'metrics.json').read_text())
    return values[candidate['algorithm']]['val']['primary']


def evaluate_candidate(root, candidate):
    version = candidate['data_version']
    datasets = root / f'data/{version}/datasets'
    splits = root / f'data/{version}/splits'
    frame, spec, _ = load_task(root, datasets, splits, candidate['task_id'])
    test = frame.loc[frame.split.eq('test')].reset_index(drop=True)
    if test.empty:
        raise ValueError(f"No eligible test records for {candidate['task_id']}")
    train_groups = sorted(frame.loc[frame.split.eq('train'), 'scaffold_group'].unique())
    if set(train_groups) & set(test.scaffold_group):
        raise ValueError(f"Train/test scaffold leakage for {candidate['task_id']}")
    if candidate['model_kind'] == 'multitask_seed_ensemble':
        members = [predict_task(Path(run), candidate['architecture'], candidate['task_id'], test.smiles)
                   for run in candidate['run_dirs']]
        prediction = np.mean(np.stack(members), axis=0)
        model_label = 'Shared MLP 3-seed ensemble'
    else:
        run = Path(candidate['run_dir'])
        meta = verify_stage(run, 'stl_training')
        expected = {'datasets_complete_sha256': sha256(datasets / 'complete.json'),
                    'splits_complete_sha256': sha256(splits / 'complete.json')}
        if meta['inputs'] != expected:
            raise ValueError(f"Model/data version mismatch for {candidate['task_id']}")
        bundle = joblib.load(run / candidate['algorithm'] / 'model.joblib')
        prediction = bundle.predict_smiles(test.smiles.tolist())
        model_label = f"STL {candidate['algorithm'].upper()}"
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"Non-finite test prediction for {candidate['task_id']}")
    table = prediction_table(test, prediction, spec, 'frozen_test_evaluation', train_groups)
    result = metrics(test, prediction, spec)
    return table, result, spec, model_label


def metric_row(candidate, result, spec, model_label):
    validation = selected_validation_primary(candidate)
    return {
        'Task': candidate['task_id'], 'Task label': LABELS[candidate['task_id']],
        'Frozen model': model_label, 'Status': candidate['inferential_status'],
        'Records': result['records'], 'Molecules': result['molecules'],
        'Primary metric': 'MAE (physical)' if spec['endpoint'] in {'fu', 'F'} else 'RMSE (log10)',
        'Validation primary': validation, 'Test primary': result['primary'],
        'Test vs validation (%)': 100 * (result['primary'] / validation - 1),
        'Physical MAE': result['physical']['mae'], 'Physical RMSE': result['physical']['rmse'],
        'Physical R2': result['physical']['r2'], 'Transformed RMSE': result['transformed']['rmse'],
        'Transformed R2': result['transformed']['r2'],
        'GMFE': result.get('gmfe_positive_subset'),
        'Within 2-fold': result.get('within_2fold_positive_subset'),
        'Within 3-fold': result.get('within_3fold_positive_subset'),
        'Within absolute 0.10': result.get('absolute_within_0.10'),
        'Canonical unit': spec['unit'],
    }


def plot_observed_predicted(predictions, specs, summary, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8), constrained_layout=True)
    for letter, (task, ax) in enumerate(zip(LABELS, axes.flat)):
        frame = predictions.loc[predictions.task_id.eq(task)]
        spec = specs[task]
        observed = frame.observed_physical.to_numpy()
        predicted = frame.predicted_physical.to_numpy()
        if spec['transform'] == 'log10':
            x = transform_values(observed, 'log10'); y = transform_values(predicted, 'log10')
            axis_label = r'$\log_{10}$ physical value'
        else:
            x, y = observed, predicted
            axis_label = 'Physical value'
        low = float(min(x.min(), y.min())); high = float(max(x.max(), y.max()))
        margin = max((high - low) * .08, .02)
        limits = (low - margin, high + margin)
        ax.scatter(x, y, s=18, alpha=.65, color='#2468a2', edgecolors='white', linewidths=.25)
        ax.plot(limits, limits, color='#222222', linewidth=1, label='1:1')
        if spec['transform'] == 'log10':
            grid = np.asarray(limits)
            ax.plot(grid, grid + np.log10(2), '--', color='#df8f2d', linewidth=.8)
            ax.plot(grid, grid - np.log10(2), '--', color='#df8f2d', linewidth=.8, label='2-fold')
            ax.plot(grid, grid + np.log10(3), ':', color='#b64343', linewidth=.8)
            ax.plot(grid, grid - np.log10(3), ':', color='#b64343', linewidth=.8, label='3-fold')
        ax.set(xlim=limits, ylim=limits, xlabel=f'Observed {axis_label}', ylabel=f'Predicted {axis_label}')
        row = summary.loc[summary.Task.eq(task)].iloc[0]
        suffix = ' (exploratory, n=2)' if row.Status.startswith('exploratory') else ''
        ax.set_title(f"{chr(97 + letter)}) {LABELS[task]}{suffix}\nPrimary={row['Test primary']:.3f}; molecules={int(row.Molecules)}", loc='left')
        ax.grid(alpha=.18, linewidth=.5)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[1].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False, title='Reference')
    fig.suptitle('Frozen Test-Set Performance of Pre-Selected PK/ADME Models', fontsize=13, fontweight='bold')
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_table(summary, output):
    shown = summary[['Task label', 'Frozen model', 'Molecules', 'Primary metric', 'Validation primary', 'Test primary', 'Test vs validation (%)', 'Status']].copy()
    for col in ['Validation primary', 'Test primary']:
        shown[col] = shown[col].map(lambda value: f'{value:.3f}')
    shown['Test vs validation (%)'] = shown['Test vs validation (%)'].map(lambda value: f'{value:+.1f}%')
    shown.columns = ['Endpoint', 'Model', 'N', 'Primary metric', 'Validation', 'Test', 'Change', 'Status']
    fig, ax = plt.subplots(figsize=(12.0, 2.8)); ax.axis('off')
    table = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc='center', colLoc='center', loc='center',
                     colWidths=[.13, .19, .06, .13, .09, .08, .08, .15])
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d8dee6'); cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor('#244a73'); cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#edf3f8')
    ax.set_title('Frozen Test Evaluation: Pre-Selected Models', fontsize=12, fontweight='bold', pad=12)
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white'); plt.close(fig)


def report(summary):
    lines = ['# Frozen test evaluation', '',
             'Models were fixed in the candidate registry before test metrics were calculated. Lower primary scores are better.', '',
             '| Endpoint | Model | Test molecules | Validation primary | Test primary | Change | Status |',
             '|---|---|---:|---:|---:|---:|---|']
    for _, row in summary.iterrows():
        lines.append(f"| {row['Task label']} | {row['Frozen model']} | {int(row.Molecules)} | {row['Validation primary']:.3f} | {row['Test primary']:.3f} | {row['Test vs validation (%)']:+.1f}% | {row.Status} |")
    lines += ['', 'The test-versus-validation change describes transport across fixed scaffold-separated splits; it was not used to revise any model.',
              'The terminal half-life result contains only two test molecules and cannot support inferential or generalization claims.']
    return '\n'.join(lines) + '\n'


def run():
    parser = base_parser(__doc__)
    parser.add_argument('--registry', type=Path, default=ROOT / 'models/frozen/final_candidates_v1')
    parser.add_argument('--confirm-frozen-test', action='store_true',
                        help='Required explicit guard: evaluate the already frozen candidates without refitting')
    args = parser.parse_args()
    configure_logging(args.root, 'evaluate_frozen_candidates')
    output = args.output or args.root / 'results/final/frozen_test_v1'
    startup_self_check([args.registry / 'complete.json', args.registry / 'candidate_registry.json'], output=output)
    meta = verify_stage(args.registry, 'frozen_candidate_registry')
    registry = json.loads((args.registry / 'candidate_registry.json').read_text())
    if meta.get('test_accessed') or registry.get('test_accessed'):
        raise ValueError('Registry must have been published before test access')
    if args.check_only:
        logging.info('Frozen registry and all hashes verified. No dataset or test label was read.')
        return
    if not args.confirm_frozen_test:
        raise ValueError('Final test requires --confirm-frozen-test after the candidate registry is published')
    all_predictions, rows, all_metrics, specs = [], [], {}, {}
    for candidate in registry['candidates']:
        table, result, spec, model_label = evaluate_candidate(args.root, candidate)
        all_predictions.append(table); specs[candidate['task_id']] = spec
        all_metrics[candidate['task_id']] = {'model': model_label, 'status': candidate['inferential_status'], 'test': result}
        rows.append(metric_row(candidate, result, spec, model_label))
        logging.info('%s frozen test complete: primary=%.6g, molecules=%d', candidate['task_id'], result['primary'], result['molecules'])
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = pd.DataFrame(rows)
    with stage_output(output) as out:
        predictions.to_csv(out / 'test_predictions.csv', index=False)
        summary.to_csv(out / 'table_1_test_metrics.csv', index=False)
        dump_json(out / 'metrics.json', all_metrics)
        plot_observed_predicted(predictions, specs, summary, out / 'figure_1_observed_vs_predicted.png')
        plot_table(summary, out / 'table_1_test_metrics.png')
        (out / 'analysis_report.md').write_text(report(summary), encoding='utf-8')
        finish_stage(out, 'frozen_test_evaluation',
                     inputs={'candidate_registry_complete_sha256': sha256(args.registry / 'complete.json')},
                     test_evaluated=True, refit=False, selection_changed=False,
                     figure_language='English', dpi=600, partial=False)
    logging.info('Frozen test evaluation and publication figures published: %s', output)


if __name__ == '__main__':
    run_cli(run)

"""汇总三个任务均衡 MTL 种子，生成英文出版级图、表和分析报告；不读取 test 标签。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (ROOT, base_parser, configure_logging, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)


TASKS = [
    'fu__human__plasma',
    'Papp__human__caco2_ab',
    'CL__human__systemic_iv',
    'VDss__human__steady_state_iv',
    'Thalf__human__terminal_iv',
]
DISPLAY = {
    'fu__human__plasma': 'Plasma fu',
    'Papp__human__caco2_ab': 'Caco-2 Papp',
    'CL__human__systemic_iv': 'Systemic CL',
    'VDss__human__steady_state_iv': 'Steady-state VDss',
    'Thalf__human__terminal_iv': 'Terminal half-life',
}
ARCHITECTURES = ['shared', 'mmoe']
ARCH_LABEL = {'shared': 'Shared MLP', 'mmoe': 'MMoE'}
COLORS = {'shared': '#2878B5', 'mmoe': '#D9534F'}


def default_mtl(root):
    return [Path(root) / f'models/multitask/balanced_pappfixed_v3_cuda_seed{seed}'
            for seed in (2026, 2027, 2028)]


def default_stl(root):
    root = Path(root)
    return {
        'fu__human__plasma': root / 'models/stl/fu__human__plasma/baseline_v1',
        'Papp__human__caco2_ab': root / 'models/stl/Papp__human__caco2_ab/papp_unitfixed_v3_seed2026',
        'CL__human__systemic_iv': root / 'models/stl/CL__human__systemic_iv/baseline_v1',
        'VDss__human__steady_state_iv': root / 'models/stl/VDss__human__steady_state_iv/baseline_v1',
        'Thalf__human__terminal_iv': root / 'models/stl/Thalf__human__terminal_iv/diagnostic_v1',
    }


def primary_metric(task):
    return 'MAE' if task.startswith('fu__') else 'log10 RMSE'


def read_mtl(path):
    path = Path(path)
    meta = verify_stage(path, 'multitask_training')
    if meta.get('test_evaluated'):
        raise ValueError(f'Test-evaluated MTL run is not allowed: {path}')
    metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
    if not set(ARCHITECTURES) <= set(metrics):
        raise ValueError(f'Missing Shared MLP or MMoE metrics: {path}')
    return path, meta, metrics


def read_stl(path, expected_task):
    path = Path(path)
    meta = verify_stage(path, 'stl_training')
    if meta.get('test_evaluated') or meta.get('task_id') != expected_task:
        raise ValueError(f'Invalid STL run for {expected_task}: {path}')
    algorithm = json.loads((path / 'selection.json').read_text(encoding='utf-8'))['algorithm']
    metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))[algorithm]
    if not {'oof', 'val'} <= set(metrics):
        raise ValueError(f'STL run lacks OOF or validation metrics: {path}')
    return path, algorithm, metrics, meta


def check_mtl_contract(runs):
    metas = [item[1] for item in runs]
    if len(runs) != 3 or len({int(meta['seed']) for meta in metas}) != 3:
        raise ValueError('Exactly three unique MTL seeds are required')
    reference = metas[0]
    for meta in metas[1:]:
        for key in ('inputs', 'tasks', 'architectures', 'batch_sampler', 'code_hashes'):
            if meta.get(key) != reference.get(key):
                raise ValueError(f'MTL runs differ in {key}; seed comparison is invalid')
    if reference.get('batch_sampler') != 'task_balanced':
        raise ValueError('Only task-balanced MTL runs are accepted')
    missing = set(TASKS) - set(reference.get('tasks', []))
    if missing:
        raise ValueError(f'MTL runs lack required human tasks: {sorted(missing)}')


def seed_table(runs, stl):
    rows = []
    for _, meta, metrics in sorted(runs, key=lambda item: int(item[1]['seed'])):
        seed = int(meta['seed'])
        for task in TASKS:
            _, algorithm, direct, _ = stl[task]
            for architecture in ARCHITECTURES:
                for scope in ('oof', 'val'):
                    score = float(metrics[architecture][scope][task]['primary'])
                    baseline = float(direct[scope]['primary'])
                    rows.append({
                        'Task': task,
                        'Task label': DISPLAY[task],
                        'Primary metric': primary_metric(task),
                        'Architecture': ARCH_LABEL[architecture],
                        'Architecture key': architecture,
                        'Seed': seed,
                        'Scope': 'OOF' if scope == 'oof' else 'Validation',
                        'Records': int(metrics[architecture][scope][task]['records']),
                        'STL model': algorithm.upper(),
                        'STL score': baseline,
                        'MTL score': score,
                        'Relative change vs STL (%)': 100 * (score / baseline - 1),
                        'MTL wins': score < baseline,
                    })
    return pd.DataFrame(rows)


def summary_table(seed):
    rows = []
    for (task, architecture, scope), group in seed.groupby(['Task', 'Architecture key', 'Scope'], sort=False):
        values = group['MTL score'].to_numpy(float)
        changes = group['Relative change vs STL (%)'].to_numpy(float)
        rows.append({
            'Task': task,
            'Task label': DISPLAY[task],
            'Primary metric': primary_metric(task),
            'Architecture': ARCH_LABEL[architecture],
            'Architecture key': architecture,
            'Scope': scope,
            'STL model': group['STL model'].iloc[0],
            'STL score': group['STL score'].iloc[0],
            'MTL mean': values.mean(),
            'MTL SD': values.std(ddof=1),
            'MTL min': values.min(),
            'MTL max': values.max(),
            'Mean relative change vs STL (%)': changes.mean(),
            'SD relative change (%)': changes.std(ddof=1),
            'Seeds better than STL': int(group['MTL wins'].sum()),
            'Seeds evaluated': len(group),
        })
    return pd.DataFrame(rows)


def decisions(summary):
    rows = []
    for task in TASKS:
        candidates = []
        task_rows = summary.loc[summary.Task.eq(task)]
        for architecture in ARCHITECTURES:
            part = task_rows.loc[task_rows['Architecture key'].eq(architecture)].set_index('Scope')
            oof, val = part.loc['OOF'], part.loc['Validation']
            eligible = (oof['Seeds better than STL'] == 3 and val['Seeds better than STL'] == 3
                        and oof['MTL mean'] <= oof['STL score'] and val['MTL mean'] <= val['STL score'])
            rows.append({
                'Task': task,
                'Task label': DISPLAY[task],
                'Architecture': ARCH_LABEL[architecture],
                'OOF mean ± SD': f"{oof['MTL mean']:.3f} ± {oof['MTL SD']:.3f}",
                'OOF change (%)': oof['Mean relative change vs STL (%)'],
                'OOF wins': f"{int(oof['Seeds better than STL'])}/3",
                'Validation mean ± SD': f"{val['MTL mean']:.3f} ± {val['MTL SD']:.3f}",
                'Validation change (%)': val['Mean relative change vs STL (%)'],
                'Validation wins': f"{int(val['Seeds better than STL'])}/3",
                'Eligible by preregistered rule': bool(eligible),
            })
            if eligible:
                candidates.append((float(val['MTL mean']), 0 if architecture == 'shared' else 1, architecture))
        stl_model = task_rows['STL model'].iloc[0]
        selected = ARCH_LABEL[min(candidates)[2]] if candidates else f'STL {stl_model}'
        reason = ('Eligible in all three seeds for both OOF and validation; lowest mean validation score among eligible MTL models.'
                  if candidates else 'No MTL architecture improved both OOF and validation in all three seeds.')
        for row in rows[-2:]:
            row['Selected model'] = selected
            row['Selection rationale'] = reason
    return pd.DataFrame(rows)


def molecule_losses(frame, endpoint, prefix):
    observed = frame[f'observed_physical_{prefix}'].to_numpy(float)
    predicted = frame[f'predicted_physical_{prefix}'].to_numpy(float)
    if endpoint == 'fu':
        loss = np.abs(predicted - observed)
        label = 'absolute physical error'
    else:
        if np.any(observed <= 0) or np.any(predicted <= 0):
            raise ValueError(f'Nonpositive value in paired log-space comparison for {endpoint}')
        loss = (np.log10(predicted) - np.log10(observed)) ** 2
        label = 'squared log10 error'
    return pd.DataFrame({'molecule_id': frame.molecule_id, f'{prefix}_loss': loss}).groupby('molecule_id').mean(), label


def paired_oof(runs, stl, bootstrap, seed=20260910):
    rng = np.random.default_rng(seed)
    rows = []
    for run_path, meta, _ in sorted(runs, key=lambda item: int(item[1]['seed'])):
        for task in TASKS:
            stl_path, algorithm, _, _ = stl[task]
            direct = pd.read_csv(stl_path / algorithm / 'oof_predictions.csv')
            endpoint = task.split('__')[0]
            for architecture in ARCHITECTURES:
                mtl = pd.read_csv(run_path / architecture / 'train_predictions.csv')
                mtl = mtl.loc[mtl.task_id.eq(task)]
                merged = direct[['row_id', 'molecule_id', 'observed_physical', 'predicted_physical']].merge(
                    mtl[['row_id', 'observed_physical', 'predicted_physical']], on='row_id', suffixes=('_stl', '_mtl'),
                    how='inner', validate='one_to_one')
                if len(merged) != len(direct) or len(merged) != len(mtl):
                    raise ValueError(f'Paired OOF rows do not match for {task}, seed {meta["seed"]}, {architecture}')
                if not np.allclose(merged.observed_physical_stl, merged.observed_physical_mtl, rtol=1e-10, atol=1e-12):
                    raise ValueError(f'Observed values differ in paired OOF comparison: {task}')
                stl_loss, label = molecule_losses(merged, endpoint, 'stl')
                mtl_loss, _ = molecule_losses(merged, endpoint, 'mtl')
                paired = stl_loss.join(mtl_loss, how='inner')
                denominator = paired.stl_loss.mean()
                change = 100 * (paired.mtl_loss.mean() / denominator - 1)
                sample_changes = np.empty(bootstrap)
                values = paired[['stl_loss', 'mtl_loss']].to_numpy()
                for iteration in range(bootstrap):
                    take = rng.integers(0, len(values), len(values))
                    sampled = values[take]
                    sample_changes[iteration] = 100 * (sampled[:, 1].mean() / sampled[:, 0].mean() - 1)
                rows.append({
                    'Task': task,
                    'Task label': DISPLAY[task],
                    'Architecture': ARCH_LABEL[architecture],
                    'Architecture key': architecture,
                    'Seed': int(meta['seed']),
                    'Molecules': len(paired),
                    'Paired loss': label,
                    'Mean paired loss change vs STL (%)': change,
                    'Bootstrap 95% CI lower (%)': np.quantile(sample_changes, .025),
                    'Bootstrap 95% CI upper (%)': np.quantile(sample_changes, .975),
                    'Molecules with lower MTL loss (%)': 100 * (paired.mtl_loss < paired.stl_loss).mean(),
                })
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.labelsize': 9.5, 'legend.fontsize': 8.5,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.linewidth': .8,
        'xtick.direction': 'out', 'ytick.direction': 'out',
    })


def relative_change_figure(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), constrained_layout=True, sharey=True)
    x = np.arange(len(TASKS)); width = .34
    for ax, scope in zip(axes, ('OOF', 'Validation')):
        part = summary.loc[summary.Scope.eq(scope)].set_index(['Task', 'Architecture key'])
        for shift, architecture in [(-width / 2, 'shared'), (width / 2, 'mmoe')]:
            means = np.array([part.loc[(task, architecture), 'Mean relative change vs STL (%)'] for task in TASKS])
            errors = np.array([part.loc[(task, architecture), 'SD relative change (%)'] for task in TASKS])
            ax.bar(x + shift, means, width, yerr=errors, capsize=3, label=ARCH_LABEL[architecture],
                   color=COLORS[architecture], edgecolor='white', linewidth=.5)
        ax.axhline(0, color='#222222', linewidth=1)
        ax.set_title(f'{scope} performance across three seeds')
        ax.set_xticks(x, [DISPLAY[t] for t in TASKS], rotation=27, ha='right')
        ax.set_ylabel('Relative change in primary error vs STL (%)\nNegative values indicate improvement')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E6E6E6', linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('Three-Seed Task-Balanced MTL versus Single-Task Baselines', fontsize=14, fontweight='bold')
    fig.savefig(output / 'figure_1_three_seed_relative_change.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def consistency_figure(seed, output):
    row_keys = [(task, architecture) for task in TASKS for architecture in ARCHITECTURES]
    labels = [f'{DISPLAY[t]} — {ARCH_LABEL[a]}' for t, a in row_keys]
    scopes = ['OOF', 'Validation']
    all_values = seed['Relative change vs STL (%)'].to_numpy(float)
    limit = max(10., float(np.nanpercentile(np.abs(all_values), 95)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.0), constrained_layout=True, sharey=True)
    image = None
    for ax, scope in zip(axes, scopes):
        part = seed.loc[seed.Scope.eq(scope)].set_index(['Task', 'Architecture key', 'Seed'])
        seeds = sorted(seed.Seed.unique())
        matrix = np.array([[part.loc[(task, architecture, value), 'Relative change vs STL (%)']
                            for value in seeds] for task, architecture in row_keys])
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                color = 'white' if abs(matrix[row, col]) > .58 * limit else '#202020'
                ax.text(col, row, f'{matrix[row, col]:+.1f}', ha='center', va='center', fontsize=7.5, color=color)
        ax.set_title(f'{scope}: error change vs STL')
        ax.set_xticks(range(len(seeds)), [f'Seed {value}' for value in seeds])
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=axes, shrink=.76, pad=.02)
    bar.set_label('Relative error change (%)')
    fig.suptitle('Seed-Level Direction and Stability', fontsize=14, fontweight='bold')
    fig.savefig(output / 'figure_2_seed_consistency_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def decision_table_figure(decision, output):
    shown = decision[['Task label', 'Architecture', 'OOF mean ± SD', 'OOF change (%)', 'OOF wins',
                      'Validation mean ± SD', 'Validation change (%)', 'Validation wins',
                      'Eligible by preregistered rule', 'Selected model']].copy()
    for column in ['OOF change (%)', 'Validation change (%)']:
        shown[column] = shown[column].map(lambda value: f'{value:+.1f}%')
    shown['Eligible by preregistered rule'] = shown['Eligible by preregistered rule'].map({True: 'Yes', False: 'No'})
    fig, ax = plt.subplots(figsize=(16.5, 5.2))
    ax.axis('off')
    artist = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc='center', loc='center')
    artist.auto_set_font_size(False); artist.set_fontsize(7.2); artist.scale(1, 1.55)
    for (row, _), cell in artist.get_celld().items():
        cell.set_edgecolor('#D0D0D0'); cell.set_linewidth(.55)
        if row == 0:
            cell.set_facecolor('#155A7A'); cell.set_text_props(color='white', weight='bold')
        elif row % 2:
            cell.set_facecolor('#EDF4F7')
    ax.set_title('Table 1. Three-seed model selection under the preregistered rule', pad=13,
                 fontsize=12, fontweight='bold')
    fig.savefig(output / 'table_1_model_selection.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def paired_figure(paired, output):
    labels = [f'{DISPLAY[t]} — {ARCH_LABEL[a]}' for t in TASKS for a in ARCHITECTURES]
    keys = [(t, a) for t in TASKS for a in ARCHITECTURES]
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    offsets = [-.18, 0, .18]
    markers = ['o', 's', '^']
    for offset, marker, seed in zip(offsets, markers, sorted(paired.Seed.unique())):
        part = paired.loc[paired.Seed.eq(seed)].set_index(['Task', 'Architecture key'])
        x = np.array([part.loc[key, 'Mean paired loss change vs STL (%)'] for key in keys])
        lo = np.array([part.loc[key, 'Bootstrap 95% CI lower (%)'] for key in keys])
        hi = np.array([part.loc[key, 'Bootstrap 95% CI upper (%)'] for key in keys])
        y = np.arange(len(keys)) + offset
        ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt=marker, markersize=4.4,
                    capsize=2.5, linewidth=.8, label=f'Seed {seed}', alpha=.9)
    ax.axvline(0, color='#202020', linewidth=1)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Paired molecule-level loss change vs STL (%)\nNegative values indicate improvement')
    ax.set_title('Paired OOF Comparison with Molecule Bootstrap 95% CIs')
    ax.grid(axis='x', color='#E5E5E5', linewidth=.7); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, ncol=3, loc='lower right')
    fig.savefig(output / 'figure_3_paired_oof_bootstrap.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def report_text(summary, decision, paired, seeds):
    lines = [
        '# Three-Seed Task-Balanced Multitask Learning Analysis', '',
        f'Runs: seeds {", ".join(map(str, seeds))}. All results use structure-disjoint protected OOF and the frozen validation set. The test set was not evaluated.', '',
        '## Model selection results', '',
        '| Task | Selected model | Shared OOF change | Shared validation change | MMoE OOF change | MMoE validation change |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for task in TASKS:
        part = summary.loc[summary.Task.eq(task)].set_index(['Architecture key', 'Scope'])
        selected = decision.loc[decision.Task.eq(task), 'Selected model'].iloc[0]
        values = [part.loc[(architecture, scope), 'Mean relative change vs STL (%)']
                  for architecture in ARCHITECTURES for scope in ('OOF', 'Validation')]
        lines.append(f'| {DISPLAY[task]} | {selected} | {values[0]:+.1f}% | {values[1]:+.1f}% | {values[2]:+.1f}% | {values[3]:+.1f}% |')
    lines += [
        '', 'The preregistered eligibility rule requires an architecture to outperform STL in both OOF and validation for all three seeds. Only plasma fu satisfies this rule. Shared MLP is selected because it has the lower mean validation error among eligible MTL architectures and is the simpler model.', '',
        'Papp Shared MLP improves validation error in all three seeds but worsens OOF error in all three seeds; Papp therefore remains assigned to the corrected v3 STL RF model. CL, VDss, and terminal half-life do not show consistent two-scope gains and remain assigned to their STL models.', '',
        'Terminal half-life MMoE has a reproducible OOF improvement, but its validation estimate is based on only nine records and is unstable across seeds. This result is hypothesis-generating rather than sufficient for model replacement.', '',
        '## Paired OOF analysis', '',
        'The paired analysis compares MTL and STL predictions for the same molecules. Loss is absolute physical error for fu and squared log10 error for the other endpoints. Confidence intervals are percentile intervals from molecule-level bootstrap resampling. These intervals describe paired sampling uncertainty and are not adjusted for multiple comparisons.', '',
    ]
    for task in TASKS:
        for architecture in ARCHITECTURES:
            part = paired.loc[paired.Task.eq(task) & paired['Architecture key'].eq(architecture)]
            lines.append(f'- {DISPLAY[task]}, {ARCH_LABEL[architecture]}: mean paired loss change across seeds {part["Mean paired loss change vs STL (%)"].mean():+.1f}%; seed range {part["Mean paired loss change vs STL (%)"].min():+.1f}% to {part["Mean paired loss change vs STL (%)"].max():+.1f}%.')
    lines += [
        '', '## Interpretation and limitations', '',
        'The three runs use identical data hashes, split hashes, task lists, code hashes, architectures, and task-balanced sampling. Seed is the intended varying factor. The completed model metadata does not record the CUDA device, batch size, epoch count, or thread count; those settings should be added to future run manifests. The current analysis does not read test labels and should be used to freeze the candidate list before one-time test evaluation.', '',
        '## Files', '',
        '- `table_1_three_seed_summary.csv`: mean, SD, range, relative change, and win counts.',
        '- `table_2_seed_level_metrics.csv`: every seed-level score.',
        '- `table_3_model_selection.csv`: preregistered-rule decisions.',
        '- `table_4_paired_oof_bootstrap.csv`: paired molecule-level loss changes and bootstrap intervals.',
        '- `figure_1_three_seed_relative_change.png`: mean relative error changes with seed SD.',
        '- `figure_2_seed_consistency_heatmap.png`: direction and magnitude for every seed.',
        '- `figure_3_paired_oof_bootstrap.png`: paired OOF effects with confidence intervals.',
        '- `table_1_model_selection.png`: publication-ready model selection table.',
    ]
    return '\n'.join(lines) + '\n'


def run():
    parser = base_parser('Three-seed task-balanced MTL publication figures, tables, and report')
    parser.add_argument('--mtl-run', action='append', type=Path,
                        help='Completed MTL run; provide exactly three times. Defaults to v3 seeds 2026-2028.')
    parser.add_argument('--bootstrap', type=int, default=2000, help='Molecule bootstrap replicates for paired OOF CIs')
    args = parser.parse_args()
    configure_logging(args.root, 'multitask_three_seed_analysis')
    output = args.output or args.root / 'results/analysis/multitask_pappfixed_v3_three_seed_v1'
    startup_self_check(output=output)
    if args.bootstrap < 200:
        raise ValueError('--bootstrap must be at least 200')
    mtl_paths = args.mtl_run or default_mtl(args.root)
    runs = [read_mtl(path) for path in mtl_paths]
    check_mtl_contract(runs)
    stl = {task: read_stl(path, task) for task, path in default_stl(args.root).items()}
    seed = seed_table(runs, stl)
    summary = summary_table(seed)
    decision = decisions(summary)
    paired = paired_oof(runs, stl, args.bootstrap)
    seeds = sorted(int(item[1]['seed']) for item in runs)
    if args.check_only:
        print(f'Validated three MTL seeds {seeds}, five STL baselines, paired OOF rows, and frozen-test status.')
        return
    inputs = {str(path.resolve()): sha256(path / 'complete.json') for path, _, _ in runs}
    inputs.update({str(path.resolve()): sha256(path / 'complete.json') for path, *_ in stl.values()})
    style()
    with stage_output(output) as out:
        summary.to_csv(out / 'table_1_three_seed_summary.csv', index=False, float_format='%.6f')
        seed.to_csv(out / 'table_2_seed_level_metrics.csv', index=False, float_format='%.6f')
        decision.to_csv(out / 'table_3_model_selection.csv', index=False, float_format='%.6f')
        paired.to_csv(out / 'table_4_paired_oof_bootstrap.csv', index=False, float_format='%.6f')
        relative_change_figure(summary, out)
        consistency_figure(seed, out)
        paired_figure(paired, out)
        decision_table_figure(decision, out)
        (out / 'analysis_report.md').write_text(report_text(summary, decision, paired, seeds), encoding='utf-8')
        finish_stage(out, 'multitask_three_seed_analysis', inputs=inputs, seeds=seeds,
                     tasks=TASKS, architectures=ARCHITECTURES, test_evaluated=False,
                     selection_rule='MTL must beat STL in OOF and validation for all three seeds',
                     bootstrap_replicates=args.bootstrap, figure_language='English', dpi=600, partial=False)
    print(f'Publication analysis written to {output}')


if __name__ == '__main__':
    run_cli(run)

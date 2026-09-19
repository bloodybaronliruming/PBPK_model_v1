"""Post-test diagnostics for the frozen Thalf v15 predictions; never refit or reselect."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

from dmpk_toolkit import load_task
from pipeline_common import (ROOT, base_parser, configure_logging, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)


TASK = 'Thalf__human__terminal_iv'
OVERLAP_ORDER = ('tdc_train_val', 'tdc_test', 'not_in_tdc')
OVERLAP_LABELS = {
    'tdc_train_val': 'TDC train/valid',
    'tdc_test': 'TDC test',
    'not_in_tdc': 'Outside TDC',
}
COLORS = {'tdc_train_val': '#2673a6', 'tdc_test': '#d9822b', 'not_in_tdc': '#4b8f63'}


def fingerprints(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f'Invalid SMILES: {value}')
        result.append(generator.GetFingerprint(mol))
    return result


def diagnostics(predictions, train):
    frame = predictions.copy()
    frame['observed_log10'] = np.log10(frame.observed_physical.to_numpy(float))
    frame['predicted_log10'] = np.log10(frame.predicted_physical.to_numpy(float))
    frame['residual_log10'] = frame.predicted_log10 - frame.observed_log10
    frame['absolute_log10_error'] = frame.residual_log10.abs()
    train_unique = train[['molecule_id', 'smiles']].drop_duplicates('molecule_id')
    test_unique = frame[['molecule_id', 'smiles']].drop_duplicates('molecule_id')
    train_fp = fingerprints(train_unique.smiles)
    similarity = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fp))
                  for fp in fingerprints(test_unique.smiles)]
    mapping = dict(zip(test_unique.molecule_id, similarity))
    frame['nearest_train_tanimoto'] = frame.molecule_id.map(mapping)
    return frame


def summary_table(frame):
    slope, intercept = np.polyfit(frame.observed_log10, frame.predicted_log10, 1)
    rho, pvalue = spearmanr(frame.nearest_train_tanimoto, frame.absolute_log10_error)
    error = frame.residual_log10.to_numpy(float)
    return pd.DataFrame([{
        'records': len(frame),
        'molecules': frame.molecule_id.nunique(),
        'log10_rmse': np.sqrt(np.mean(error ** 2)),
        'log10_mae': np.mean(np.abs(error)),
        'mean_log10_bias': np.mean(error),
        'median_log10_bias': np.median(error),
        'calibration_slope': slope,
        'calibration_intercept': intercept,
        'spearman': spearmanr(frame.observed_log10, frame.predicted_log10).statistic,
        'median_nearest_train_tanimoto': frame.nearest_train_tanimoto.median(),
        'q1_nearest_train_tanimoto': frame.nearest_train_tanimoto.quantile(.25),
        'q3_nearest_train_tanimoto': frame.nearest_train_tanimoto.quantile(.75),
        'molecules_below_tanimoto_0_30': int((frame.nearest_train_tanimoto < .30).sum()),
        'similarity_error_spearman': rho,
        'similarity_error_pvalue': pvalue,
    }])


def overlap_table(frame):
    rows = []
    for category in OVERLAP_ORDER:
        group = frame.loc[frame.tdc_overlap.eq(category)]
        error = group.residual_log10.to_numpy(float)
        rows.append({
            'tdc_overlap': category,
            'label': OVERLAP_LABELS[category],
            'molecules': group.molecule_id.nunique(),
            'log10_rmse': np.sqrt(np.mean(error ** 2)),
            'log10_mae': np.mean(np.abs(error)),
            'spearman': (spearmanr(group.observed_log10, group.predicted_log10).statistic
                         if len(group) > 2 else np.nan),
            'gmfe': 10 ** np.mean(np.abs(error)),
            'within_2fold': np.mean(np.abs(error) <= np.log10(2)),
            'within_3fold': np.mean(np.abs(error) <= np.log10(3)),
            'median_nearest_train_tanimoto': group.nearest_train_tanimoto.median(),
        })
    return pd.DataFrame(rows)


def plot_observed_predicted(frame, output):
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    for category in OVERLAP_ORDER:
        group = frame.loc[frame.tdc_overlap.eq(category)]
        ax.scatter(group.observed_log10, group.predicted_log10, s=34, alpha=.78,
                   color=COLORS[category], edgecolors='white', linewidths=.35,
                   label=f'{OVERLAP_LABELS[category]} (n={len(group)})')
    low = min(frame.observed_log10.min(), frame.predicted_log10.min()) - .08
    high = max(frame.observed_log10.max(), frame.predicted_log10.max()) + .08
    grid = np.array([low, high])
    ax.plot(grid, grid, color='#202020', linewidth=1.2, label='Identity')
    ax.plot(grid, grid + np.log10(2), '--', color='#777777', linewidth=.9)
    ax.plot(grid, grid - np.log10(2), '--', color='#777777', linewidth=.9, label='Twofold bounds')
    slope, intercept = np.polyfit(frame.observed_log10, frame.predicted_log10, 1)
    ax.plot(grid, intercept + slope * grid, color='#b33f3f', linewidth=1.3,
            label=f'Calibration slope = {slope:.2f}')
    ax.set(xlim=(low, high), ylim=(low, high), xlabel='Observed half-life (log10 h)',
           ylabel='Predicted half-life (log10 h)', title='Frozen Thalf v15 Test Performance')
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_residuals(frame, output):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for category in OVERLAP_ORDER:
        group = frame.loc[frame.tdc_overlap.eq(category)]
        ax.scatter(group.observed_log10, group.residual_log10, s=34, alpha=.78,
                   color=COLORS[category], edgecolors='white', linewidths=.35,
                   label=OVERLAP_LABELS[category])
    coefficients = np.polyfit(frame.observed_log10, frame.residual_log10, 1)
    grid = np.linspace(frame.observed_log10.min(), frame.observed_log10.max(), 100)
    ax.plot(grid, np.polyval(coefficients, grid), color='#b33f3f', linewidth=1.3,
            label='Linear trend')
    ax.axhline(0, color='#202020', linewidth=1)
    ax.axhline(np.log10(2), linestyle='--', color='#888888', linewidth=.8)
    ax.axhline(-np.log10(2), linestyle='--', color='#888888', linewidth=.8)
    ax.set(xlabel='Observed half-life (log10 h)', ylabel='Residual: predicted - observed (log10 h)',
           title='Residual Compression Across the Observed Range')
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_applicability(frame, output):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for category in OVERLAP_ORDER:
        group = frame.loc[frame.tdc_overlap.eq(category)]
        ax.scatter(group.nearest_train_tanimoto, group.absolute_log10_error, s=34, alpha=.78,
                   color=COLORS[category], edgecolors='white', linewidths=.35,
                   label=OVERLAP_LABELS[category])
    coefficients = np.polyfit(frame.nearest_train_tanimoto, frame.absolute_log10_error, 1)
    grid = np.linspace(frame.nearest_train_tanimoto.min(), frame.nearest_train_tanimoto.max(), 100)
    ax.plot(grid, np.polyval(coefficients, grid), color='#b33f3f', linewidth=1.3)
    rho = spearmanr(frame.nearest_train_tanimoto, frame.absolute_log10_error).statistic
    ax.set(xlabel='Nearest training ECFP4 Tanimoto', ylabel='Absolute log10 error',
           title=f'Chemical Applicability Domain (Spearman rho = {rho:.2f})')
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_overlap(frame, output):
    groups = [frame.loc[frame.tdc_overlap.eq(category), 'absolute_log10_error'].to_numpy()
              for category in OVERLAP_ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    boxes = ax.boxplot(groups, patch_artist=True, widths=.55, showfliers=False)
    for patch, category in zip(boxes['boxes'], OVERLAP_ORDER):
        patch.set_facecolor(COLORS[category]); patch.set_alpha(.52)
    rng = np.random.default_rng(20260913)
    for index, (values, category) in enumerate(zip(groups, OVERLAP_ORDER), start=1):
        jitter = rng.uniform(-.10, .10, size=len(values))
        ax.scatter(index + jitter, values, s=24, alpha=.72, color=COLORS[category],
                   edgecolors='white', linewidths=.3)
    ax.set_xticks(range(1, 4), [f'{OVERLAP_LABELS[c]}\n(n={len(v)})'
                               for c, v in zip(OVERLAP_ORDER, groups)])
    ax.set(ylabel='Absolute log10 error', title='Error by TDC Structure-Overlap Category')
    ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def report(summary, overlap):
    row = summary.iloc[0]
    lines = [
        '# Thalf v15 frozen test diagnostics', '',
        'These diagnostics use the already frozen test predictions. No model was refitted, calibrated, or reselected.', '',
        '## Calibration and applicability domain', '',
        f"The calibration slope is {row.calibration_slope:.3f}, with mean log10 bias "
        f"{row.mean_log10_bias:+.3f}. The median nearest-training ECFP4 Tanimoto is "
        f"{row.median_nearest_train_tanimoto:.3f}; {int(row.molecules_below_tanimoto_0_30)} "
        f"test molecules are below 0.30. Similarity versus absolute error has Spearman rho "
        f"{row.similarity_error_spearman:.3f} (p={row.similarity_error_pvalue:.3g}).", '',
        '## TDC structure-overlap strata', '',
        '| Category | Molecules | Log10 RMSE | Spearman | GMFE | Within twofold |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for _, item in overlap.iterrows():
        lines.append(f"| {item.label} | {int(item.molecules)} | {item.log10_rmse:.3f} | "
                     f"{item.spearman:.3f} | {item.gmfe:.3f} | {item.within_2fold:.3f} |")
    lines += ['', 'The outside-TDC stratum contains only three molecules and is not inferential. '
              'All subgroup analyses are descriptive and cannot alter the frozen primary result.']
    return '\n'.join(lines) + '\n'


def run():
    parser = base_parser(__doc__)
    parser.add_argument('--frozen-test-dir', type=Path,
                        default=ROOT / 'results/final/Thalf_v15_rdkit2d_et3_test_v1')
    args = parser.parse_args()
    configure_logging(args.root, 'analyze_Thalf_v15_frozen_test')
    output = args.output or args.root / 'results/final/Thalf_v15_rdkit2d_et3_diagnostics_v1'
    startup_self_check([args.frozen_test_dir / 'complete.json',
                        args.frozen_test_dir / 'test_predictions.csv'], output=output)
    meta = verify_stage(args.frozen_test_dir, 'thalf_v15_frozen_test_evaluation')
    if not meta.get('test_evaluated') or meta.get('refit') or meta.get('selection_changed'):
        raise ValueError('输入不是有效的冻结 test 评估')
    predictions = pd.read_csv(args.frozen_test_dir / 'test_predictions.csv')
    required = {'molecule_id', 'smiles', 'observed_physical', 'predicted_physical', 'tdc_overlap'}
    if not required <= set(predictions) or len(predictions) != 53:
        raise ValueError('冻结预测表契约不匹配')
    frame, _, _ = load_task(args.root, args.root / 'data/processed_v15/datasets',
                            args.root / 'data/processed_v15/splits', TASK)
    train = frame.loc[frame.split.eq('train'), ['molecule_id', 'smiles']]
    points = diagnostics(predictions, train)
    summary = summary_table(points)
    overlap = overlap_table(points)
    largest = points.nlargest(10, 'absolute_log10_error')[
        ['molecule_id', 'smiles', 'observed_physical', 'predicted_physical', 'residual_log10',
         'absolute_log10_error', 'nearest_train_tanimoto', 'tdc_overlap']]
    if args.check_only:
        print(f'Validated 53 frozen predictions and {train.molecule_id.nunique()} training molecules; no refit.')
        return
    with stage_output(output) as out:
        summary.to_csv(out / 'table_1_diagnostic_summary.csv', index=False, float_format='%.6f')
        overlap.to_csv(out / 'table_2_tdc_overlap_diagnostics.csv', index=False, float_format='%.6f')
        points.to_csv(out / 'table_3_applicability_domain_molecules.csv', index=False)
        largest.to_csv(out / 'table_4_largest_errors.csv', index=False)
        plot_observed_predicted(points, out / 'figure_1_observed_vs_predicted.png')
        plot_residuals(points, out / 'figure_2_residuals.png')
        plot_applicability(points, out / 'figure_3_applicability_domain.png')
        plot_overlap(points, out / 'figure_4_tdc_overlap_errors.png')
        (out / 'analysis_report.md').write_text(report(summary, overlap), encoding='utf-8')
        finish_stage(out, 'thalf_v15_frozen_test_diagnostics',
                     inputs={'frozen_test_complete_sha256': sha256(args.frozen_test_dir / 'complete.json')},
                     task_id=TASK, test_evaluated=True, refit=False, selection_changed=False,
                     figure_language='English', dpi=600, partial=False)
    print(f'Thalf v15 frozen test diagnostics: {output}')


if __name__ == '__main__':
    run_cli(run)

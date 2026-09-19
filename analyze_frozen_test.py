"""Publication diagnostics for frozen test predictions without model refitting or reselection."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from tqdm import tqdm

from dmpk_toolkit import load_task, metrics
from pipeline_common import (ROOT, base_parser, configure_logging, finish_stage, run_cli, sha256,
                             stage_output, startup_self_check, verify_stage)
from rebuild_endpoint_datasets import transform_values


TASKS = [
    ('fu__human__plasma', 'Plasma fu', 'processed_v3'),
    ('Papp__human__caco2_ab', 'Caco-2 Papp', 'processed_v3'),
    ('CL__human__systemic_iv', 'Systemic CL', 'processed_v2'),
    ('VDss__human__steady_state_iv', 'Steady-state VDss', 'processed_v2'),
    ('Thalf__human__terminal_iv', 'Terminal half-life', 'processed_v2'),
]
PAPP_DOCUMENT = 117572


def primary_losses(frame, endpoint, transform):
    values = frame.copy()
    if endpoint in {'fu', 'F'}:
        values['loss'] = np.abs(values.observed_physical - values.predicted_physical)
        molecule = values.groupby('molecule_id', as_index=False).loss.mean()
        return molecule, lambda x: float(np.mean(x))
    observed = transform_values(values.observed_physical, transform)
    predicted = transform_values(values.predicted_physical, transform)
    values['loss'] = (observed - predicted) ** 2
    molecule = values.groupby('molecule_id', as_index=False).loss.mean()
    return molecule, lambda x: float(np.sqrt(np.mean(x)))


def bootstrap_primary(frame, spec, repeats, rng):
    molecule, aggregate = primary_losses(frame, spec['endpoint'], spec['transform'])
    losses = molecule.loss.to_numpy()
    estimates = np.empty(repeats)
    for index in range(repeats):
        estimates[index] = aggregate(rng.choice(losses, size=len(losses), replace=True))
    estimate = aggregate(losses)
    low, high = np.quantile(estimates, [.025, .975])
    return estimate, float(low), float(high)


def papp_sensitivity(predictions, task_frame, spec, source_records, repeats, seed):
    frame = predictions.loc[predictions.task_id.eq('Papp__human__caco2_ab')].copy()
    affected = set(source_records.loc[(source_records.doc_id.eq(PAPP_DOCUMENT)) &
                                      source_records.task_id.eq('Papp__human__caco2_ab'), 'molecule_id'])
    if not affected:
        raise ValueError(f'Papp document {PAPP_DOCUMENT} has no mapped task molecule')
    if not set(frame.loc[frame.molecule_id.isin(affected), 'molecule_id']):
        raise ValueError('Papp source molecule is absent from frozen test predictions')
    variants = []
    for order, (name, action) in enumerate([
        ('As frozen: literal cm/s conversion', 'unchanged'),
        ('Sensitivity: implicit 10^-6 cm/s', 'divide affected observed values by 1e6'),
        ('Sensitivity: exclude unresolved source', 'exclude affected molecule'),
    ]):
        data = frame.copy()
        if order == 1:
            data.loc[data.molecule_id.isin(affected), 'observed_physical'] /= 1e6
        elif order == 2:
            data = data.loc[~data.molecule_id.isin(affected)].copy()
        metric_frame = data.rename(columns={'observed_physical': 'raw_value'})
        result = metrics(metric_frame, data.predicted_physical.to_numpy(), spec)
        estimate, low, high = bootstrap_primary(data, spec, repeats, np.random.default_rng(seed + order))
        variants.append({'Scenario': name, 'Action': action, 'Records': len(data),
                         'Molecules': data.molecule_id.nunique(), 'Affected molecules': len(affected),
                         'Log10 RMSE': result['primary'], 'Bootstrap 95% CI low': low,
                         'Bootstrap 95% CI high': high, 'Log10 R2': result['transformed']['r2'],
                         'GMFE': result.get('gmfe_positive_subset'),
                         'Within 2-fold': result.get('within_2fold_positive_subset'),
                         'Within 3-fold': result.get('within_3fold_positive_subset')})
    return pd.DataFrame(variants), sorted(affected)


def fingerprints(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f'Invalid SMILES in applicability-domain analysis: {value}')
        result.append(generator.GetFingerprint(mol))
    return result


def applicability(task, label, version, predictions, root, spec):
    datasets = root / f'data/{version}/datasets'; splits = root / f'data/{version}/splits'
    frame, _, _ = load_task(root, datasets, splits, task)
    train = frame.loc[frame.split.eq('train'), ['molecule_id', 'smiles']].drop_duplicates('molecule_id')
    test = predictions.loc[predictions.task_id.eq(task),
                           ['molecule_id', 'smiles', 'observed_physical', 'predicted_physical']].copy()
    test_unique = test[['molecule_id', 'smiles']].drop_duplicates('molecule_id')
    train_fp = fingerprints(train.smiles)
    test_fp = fingerprints(test_unique.smiles)
    similarities = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fp)) for fp in tqdm(test_fp, desc=f'{label} AD')]
    similarity = dict(zip(test_unique.molecule_id, similarities))
    test['nearest_train_similarity'] = test.molecule_id.map(similarity)
    if spec['endpoint'] in {'fu', 'F'}:
        test['absolute_error'] = np.abs(test.observed_physical - test.predicted_physical)
    else:
        test['absolute_error'] = np.abs(transform_values(test.observed_physical, spec['transform']) -
                                        transform_values(test.predicted_physical, spec['transform']))
    molecule = test.groupby('molecule_id', as_index=False).agg(nearest_train_similarity=('nearest_train_similarity', 'first'),
                                                               absolute_error=('absolute_error', 'mean'))
    rho, pvalue = spearmanr(molecule.nearest_train_similarity, molecule.absolute_error) if len(molecule) >= 4 else (np.nan, np.nan)
    summary = {'Task': task, 'Task label': label, 'Test molecules': len(molecule),
               'Median nearest-train Tanimoto': molecule.nearest_train_similarity.median(),
               'Q1 nearest-train Tanimoto': molecule.nearest_train_similarity.quantile(.25),
               'Q3 nearest-train Tanimoto': molecule.nearest_train_similarity.quantile(.75),
               'Molecules below 0.30': int((molecule.nearest_train_similarity < .30).sum()),
               'Spearman similarity-error rho': rho, 'Spearman p-value': pvalue}
    molecule['Task'] = task; molecule['Task label'] = label
    return summary, molecule


def residual_summary(task, label, prediction, spec):
    observed = prediction.observed_physical.to_numpy(); predicted = prediction.predicted_physical.to_numpy()
    if spec['transform'] != 'identity':
        observed = transform_values(observed, spec['transform']); predicted = transform_values(predicted, spec['transform'])
    residual = predicted - observed
    if len(observed) > 1 and np.var(observed) > 0:
        slope, intercept = np.polyfit(observed, predicted, 1)
    else:
        slope, intercept = np.nan, np.nan
    rho, pvalue = spearmanr(predicted, np.abs(residual)) if len(observed) >= 4 else (np.nan, np.nan)
    return {'Task': task, 'Task label': label, 'Scale': spec['transform'], 'Records': len(observed),
            'Calibration slope': slope, 'Calibration intercept': intercept,
            'Mean residual': float(np.mean(residual)), 'Residual SD': float(np.std(residual, ddof=1)) if len(residual)>1 else np.nan,
            'Spearman predicted-absolute-residual rho': rho, 'Spearman p-value': pvalue}


def plot_bootstrap(table, output):
    plot = table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot)); center = plot['Primary estimate'].to_numpy()
    lower = center - plot['95% CI low'].to_numpy(); upper = plot['95% CI high'].to_numpy() - center
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.errorbar(center, y, xerr=np.vstack([lower, upper]), fmt='o', color='#245b85', ecolor='#79a6c9', capsize=4)
    ax.set_yticks(y, plot['Task label']); ax.set_xlabel('Frozen test primary error (95% molecule-bootstrap CI)')
    ax.set_title('Uncertainty of Pre-Selected Models', fontweight='bold'); ax.grid(axis='x', alpha=.25)
    for yi, status in zip(y, plot.Status):
        if status.startswith('exploratory'): ax.text(center[yi], yi + .18, 'n=2 exploratory', fontsize=7, ha='center', color='#a33')
    fig.tight_layout(); fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white'); plt.close(fig)


def plot_papp(table, output):
    labels = ['Frozen\nliteral cm/s', 'Implicit\n$10^{-6}$ cm/s', 'Exclude\nunresolved source']
    values = table['Log10 RMSE'].to_numpy(); low = table['Bootstrap 95% CI low'].to_numpy(); high = table['Bootstrap 95% CI high'].to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(labels, values, color=['#315f86', '#d08a36', '#779e55'], width=.62)
    ax.errorbar(np.arange(3), values, yerr=np.vstack([values-low, high-values]), fmt='none', color='#222', capsize=4)
    ax.set_ylabel(r'Papp test RMSE ($\log_{10}$)'); ax.set_title('Post-Test Papp Unit Sensitivity', fontweight='bold')
    ax.grid(axis='y', alpha=.22)
    for index, value in enumerate(values): ax.text(index, value + .025, f'{value:.3f}', ha='center', fontsize=9)
    fig.tight_layout(); fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white'); plt.close(fig)


def plot_ad(points, output):
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.3), constrained_layout=True)
    for index, ((task, label, _), ax) in enumerate(zip(TASKS, axes.flat)):
        frame = points.loc[points.Task.eq(task)]
        ax.scatter(frame.nearest_train_similarity, frame.absolute_error, s=16, alpha=.62,
                   color='#2f78a8', edgecolors='white', linewidths=.2)
        if len(frame) >= 4:
            coefficients = np.polyfit(frame.nearest_train_similarity, frame.absolute_error, 1)
            x = np.linspace(frame.nearest_train_similarity.min(), frame.nearest_train_similarity.max(), 100)
            ax.plot(x, np.polyval(coefficients, x), color='#c14f45', linewidth=1)
        ax.set_title(f'{chr(97+index)}) {label}', loc='left', fontsize=9)
        ax.set_xlabel('Nearest training Tanimoto'); ax.set_ylabel('Absolute error' if task.startswith('fu') else r'Absolute $\log_{10}$ error')
        ax.grid(alpha=.18)
    axes.flat[-1].axis('off')
    fig.suptitle('Chemical Applicability Domain on Frozen Test Molecules', fontsize=13, fontweight='bold')
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white'); plt.close(fig)


def plot_residuals(predictions, specs, output):
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.3), constrained_layout=True)
    for index, ((task, label, _), ax) in enumerate(zip(TASKS, axes.flat)):
        frame = predictions.loc[predictions.task_id.eq(task)]
        observed = frame.observed_physical.to_numpy(); predicted = frame.predicted_physical.to_numpy()
        if specs[task]['transform'] != 'identity':
            observed = transform_values(observed, specs[task]['transform']); predicted = transform_values(predicted, specs[task]['transform'])
        residual = predicted - observed
        ax.scatter(predicted, residual, s=16, alpha=.62, color='#4b83ad', edgecolors='white', linewidths=.2)
        ax.axhline(0, color='#222', linewidth=1)
        if len(predicted) >= 4:
            coefficients = np.polyfit(predicted, residual, 1); x = np.linspace(predicted.min(), predicted.max(), 100)
            ax.plot(x, np.polyval(coefficients, x), color='#c14f45', linewidth=1)
        ax.set_title(f'{chr(97+index)}) {label}', loc='left', fontsize=9); ax.set_xlabel('Predicted value'); ax.set_ylabel('Residual (predicted - observed)')
        ax.grid(alpha=.18)
    axes.flat[-1].axis('off')
    fig.suptitle('Residual Diagnostics of Frozen Test Predictions', fontsize=13, fontweight='bold')
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white'); plt.close(fig)


def write_report(ci, papp, ad, residual, affected):
    lines = ['# Frozen test diagnostics', '',
             'All analyses use the already frozen predictions. No model was refitted or reselected.', '',
             '## Bootstrap uncertainty', '',
             '| Endpoint | Primary estimate | Molecule-bootstrap 95% CI | Status |', '|---|---:|---:|---|']
    for _, row in ci.iterrows():
        lines.append(f"| {row['Task label']} | {row['Primary estimate']:.3f} | {row['95% CI low']:.3f}–{row['95% CI high']:.3f} | {row.Status} |")
    lines += ['', '## Papp unit sensitivity', '',
              f'Document {PAPP_DOCUMENT} affects {len(affected)} test molecule. The frozen result remains the primary record.', '',
              '| Scenario | Log10 RMSE | 95% CI | Log10 R2 |', '|---|---:|---:|---:|']
    for _, row in papp.iterrows():
        lines.append(f"| {row.Scenario} | {row['Log10 RMSE']:.3f} | {row['Bootstrap 95% CI low']:.3f}–{row['Bootstrap 95% CI high']:.3f} | {row['Log10 R2']:.3f} |")
    lines += ['', 'The sensitivity variants are descriptive and cannot be used to alter model selection.', '',
              '## Applicability domain', '']
    for _, row in ad.iterrows():
        lines.append(f"- {row['Task label']}: median nearest-training Tanimoto {row['Median nearest-train Tanimoto']:.3f}; similarity-error Spearman rho {row['Spearman similarity-error rho']:.3f}.")
    lines += ['', 'Calibration slopes and heteroscedasticity statistics are available in `table_4_residual_diagnostics.csv`. Terminal half-life remains exploratory because n=2.']
    return '\n'.join(lines) + '\n'


def run():
    parser = base_parser(__doc__)
    parser.add_argument('--frozen-test-dir', type=Path, default=ROOT/'results/final/frozen_test_v1')
    parser.add_argument('--papp-review-dir', type=Path, default=ROOT/'results/analysis/papp_posttest_provenance_v1')
    parser.add_argument('--bootstrap-repeats', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    configure_logging(args.root, 'analyze_frozen_test')
    output = args.output or args.root/'results/final/frozen_test_diagnostics_v1'
    startup_self_check([args.frozen_test_dir/'complete.json', args.frozen_test_dir/'test_predictions.csv',
                        args.papp_review_dir/'complete.json'], output=output)
    test_meta = verify_stage(args.frozen_test_dir, 'frozen_test_evaluation')
    verify_stage(args.papp_review_dir, 'papp_posttest_provenance_review')
    if not test_meta.get('test_evaluated') or test_meta.get('refit') or test_meta.get('selection_changed'):
        raise ValueError('Input is not a valid frozen, no-refit test evaluation')
    if args.bootstrap_repeats < 200:
        raise ValueError('bootstrap-repeats must be at least 200')
    predictions = pd.read_csv(args.frozen_test_dir/'test_predictions.csv')
    source = pd.read_csv(args.root/'data/processed_v3/audit/source_records.csv', low_memory=False)
    specs = {}; ci_rows = []; ad_rows = []; ad_points = []; residual_rows = []
    for offset, (task, label, version) in enumerate(TASKS):
        _, spec, _ = load_task(args.root, args.root/f'data/{version}/datasets', args.root/f'data/{version}/splits', task)
        specs[task] = spec
        frame = predictions.loc[predictions.task_id.eq(task)].copy()
        estimate, low, high = bootstrap_primary(frame, spec, args.bootstrap_repeats, np.random.default_rng(args.seed + offset))
        status = 'exploratory_n2' if frame.molecule_id.nunique() < 10 else 'confirmatory'
        ci_rows.append({'Task': task, 'Task label': label, 'Molecules': frame.molecule_id.nunique(),
                        'Primary metric': 'MAE (physical)' if spec['endpoint'] in {'fu','F'} else 'RMSE (log10)',
                        'Primary estimate': estimate, '95% CI low': low, '95% CI high': high,
                        'Bootstrap repeats': args.bootstrap_repeats, 'Status': status})
        summary, points = applicability(task, label, version, predictions, args.root, spec)
        ad_rows.append(summary); ad_points.append(points)
        residual_rows.append(residual_summary(task, label, frame, spec))
    ci = pd.DataFrame(ci_rows); ad = pd.DataFrame(ad_rows); points = pd.concat(ad_points, ignore_index=True)
    residual = pd.DataFrame(residual_rows)
    papp, affected = papp_sensitivity(predictions, None, specs['Papp__human__caco2_ab'], source,
                                      args.bootstrap_repeats, args.seed + 100)
    logging.info('Diagnostics calculated; Papp document %d affects %d frozen test molecule(s).', PAPP_DOCUMENT, len(affected))
    if args.check_only:
        logging.info('Inputs, test lineage, source mapping, bootstrap settings, and AD molecules verified; no output published.')
        return
    with stage_output(output) as out:
        ci.to_csv(out/'table_1_bootstrap_primary_ci.csv', index=False)
        papp.to_csv(out/'table_2_papp_unit_sensitivity.csv', index=False)
        ad.to_csv(out/'table_3_applicability_domain.csv', index=False)
        points.to_csv(out/'table_3_applicability_domain_molecules.csv', index=False)
        residual.to_csv(out/'table_4_residual_diagnostics.csv', index=False)
        plot_bootstrap(ci, out/'figure_1_bootstrap_primary_ci.png')
        plot_papp(papp, out/'figure_2_papp_unit_sensitivity.png')
        plot_ad(points, out/'figure_3_applicability_domain.png')
        plot_residuals(predictions, specs, out/'figure_4_residual_diagnostics.png')
        (out/'analysis_report.md').write_text(write_report(ci, papp, ad, residual, affected), encoding='utf-8')
        finish_stage(out, 'frozen_test_diagnostics',
                     inputs={'frozen_test_complete_sha256': sha256(args.frozen_test_dir/'complete.json'),
                             'papp_review_complete_sha256': sha256(args.papp_review_dir/'complete.json')},
                     bootstrap_repeats=args.bootstrap_repeats, seed=args.seed, refit=False,
                     selection_changed=False, papp_sensitivity_post_test=True,
                     figure_language='English', dpi=600, partial=False)
    logging.info('Frozen test diagnostics published: %s', output)


if __name__ == '__main__':
    run_cli(run)

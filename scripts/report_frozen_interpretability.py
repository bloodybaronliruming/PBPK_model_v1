"""Summarise global Random Forest feature importance for frozen STL candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from dmpk_toolkit import feature_layout
from pipeline_common import ROOT, run_cli, startup_self_check


def report(bundle, task):
    importance = bundle.estimator.feature_importances_
    feature_set = getattr(bundle, 'feature_set', 'ecfp4_rdkit2d')
    fingerprint, _ = feature_layout(feature_set, bundle.descriptor_names)
    names = (['ECFP4 bit ' + str(i) for i in range(2048)] if fingerprint else []) + list(bundle.descriptor_names)
    values = pd.DataFrame({'task_id': task, 'feature': names, 'importance': importance,
                           'feature_family': (['ECFP4'] * 2048 if fingerprint else [])
                                             + ['RDKit_2D'] * len(bundle.descriptor_names)})
    return values.sort_values('importance', ascending=False, ignore_index=True)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--registry', type=Path, default=ROOT / 'models/frozen/final_candidates_v1/candidate_registry.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/final/frozen_model_interpretability_v1')
    args = parser.parse_args()
    startup_self_check([args.registry], output=args.output)
    registry = json.loads(args.registry.read_text(encoding='utf-8'))
    tables, summaries = [], []
    for candidate in registry['candidates']:
        if candidate['model_kind'] != 'stl' or candidate['algorithm'] != 'rf':
            continue
        bundle = joblib.load(Path(candidate['run_dir']) / 'rf/model.joblib')
        table = report(bundle, candidate['task_id'])
        tables.append(table)
        grouped = table.groupby('feature_family', as_index=False).importance.sum()
        grouped['task_id'] = candidate['task_id']
        summaries.append(grouped)
    args.output.mkdir(parents=True)
    pd.concat(tables, ignore_index=True).to_csv(args.output / 'rf_feature_importance_all.csv', index=False)
    pd.concat(summaries, ignore_index=True).to_csv(args.output / 'rf_feature_importance_by_family.csv', index=False)
    descriptor_rows = pd.concat([table.loc[table.feature_family.eq('RDKit_2D')].head(15) for table in tables])
    descriptor_rows.to_csv(args.output / 'top_rdkit_descriptor_importance.csv', index=False)
    lines = ['# Frozen-model global interpretability', '',
             'This report is a read-only summary of impurity-based Random Forest split importance. It is descriptive, not causal, and correlated descriptors can share or mask importance.', '',
             'The frozen fu candidate is a Shared MLP ensemble; it is intentionally not represented by RF split importance.', '',
             '## Top RDKit 2D descriptors', '']
    for task, frame in descriptor_rows.groupby('task_id', sort=True):
        lines += [f'### {task}', '', '| Feature | Importance |', '|---|---:|']
        lines += [f'| {row.feature} | {row.importance:.6f} |' for row in frame.itertuples(index=False)]
        lines.append('')
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run_cli(run)

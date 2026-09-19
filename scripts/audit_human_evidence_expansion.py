"""Prioritise human F, CLint, and Thalf review records for source-level curation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import ROOT, finish_stage, run_cli, sha256, stage_output, startup_self_check, verify_stage


TASKS = ('F__human__absolute_oral', 'CLint__human__microsome', 'Thalf__human__terminal_iv')
COLUMNS = ['molecule_id', 'smiles', 'activity_id', 'assay_id', 'doc_id', 'standard_type',
           'standard_value', 'standard_units', 'standard_relation', 'data_validity_comment',
           'potential_duplicate', 'assay_cell_type', 'assay_tissue', 'assay_subcellular_fraction',
           'description', 'doi', 'pubmed_id', 'endpoint', 'species', 'system', 'route', 'status',
           'reason', 'task_id']


def review_rows(path):
    rows = []
    for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=100_000, keep_default_na=False, dtype=str):
        rows.append(chunk.loc[chunk.task_id.isin(TASKS) & chunk.status.eq('review')])
    return pd.concat(rows, ignore_index=True)


def text(frame):
    return frame.description.astype(str).str.lower()


def prioritise(frame):
    values = pd.to_numeric(frame.standard_value, errors='coerce')
    relation = frame.standard_relation.astype(str).str.strip().eq('=')
    clean = values.gt(0) & relation & frame.data_validity_comment.eq('') & frame.potential_duplicate.astype(str).isin(['0', '0.0', 'False', 'false'])
    detail = text(frame)
    score = np.zeros(len(frame), dtype=int)
    evidence = np.full(len(frame), '', dtype=object)

    f = frame.task_id.eq('F__human__absolute_oral')
    absolute = detail.str.contains('absolute bioavailability', regex=False)
    oral = detail.str.contains('oral', regex=False)
    score += np.where(f & absolute, 8, 0) + np.where(f & oral, 2, 0)
    evidence = np.where(f & absolute, 'Description claims absolute bioavailability; verify IV reference and study design.', evidence)
    evidence = np.where(f & ~absolute & oral, 'Description claims oral bioavailability only; verify whether absolute F was measured.', evidence)

    thalf = frame.task_id.eq('Thalf__human__terminal_iv')
    iv = detail.str.contains(r'\bintravenous\b|\biv\b|\bi\.v\.\b', regex=True)
    terminal = detail.str.contains(r'\bterminal\b|\belimination\b', regex=True)
    score += np.where(thalf & iv, 6, 0) + np.where(thalf & terminal, 6, 0)
    evidence = np.where(thalf & iv & terminal, 'Description signals IV terminal/elimination half-life; verify table and phase definition.', evidence)
    evidence = np.where(thalf & iv & ~terminal, 'Description signals IV but not terminal phase; verify pharmacokinetic parameter definition.', evidence)

    clint = frame.task_id.eq('CLint__human__microsome')
    microsome = detail.str.contains('microsom', regex=False)
    intrinsic = detail.str.contains('intrinsic clearance', regex=False)
    protein_unit = frame.standard_units.astype(str).str.lower().str.contains(r'(?:ul|µl|micro).*min.*mg|ml.*min.*mg', regex=True)
    score += np.where(clint & microsome, 5, 0) + np.where(clint & intrinsic, 5, 0) + np.where(clint & protein_unit, 8, 0)
    evidence = np.where(clint & microsome & intrinsic & protein_unit,
                        'Description and unit signal microsomal CLint per mg protein; verify assay table and relation.', evidence)
    evidence = np.where(clint & microsome & intrinsic & ~protein_unit,
                        'Microsomal CLint signal but normalization is not per mg protein; verify original material basis.', evidence)

    doi = frame.doi.astype(str).str.strip().ne('')
    score += np.where(clean, 3, -10) + np.where(doi, 1, 0)
    output = frame.assign(priority_score=score, curation_rationale=evidence, numeric_relation_quality=clean)
    return output.loc[clean & output.curation_rationale.astype(str).ne('')].sort_values(
        ['task_id', 'priority_score', 'doc_id', 'assay_id'], ascending=[True, False, True, True], ignore_index=True)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'data/processed_v3/audit')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis/human_evidence_expansion_v1')
    args = parser.parse_args()
    startup_self_check([args.audit_dir / 'complete.json', args.audit_dir / 'source_records.csv'], output=args.output)
    verify_stage(args.audit_dir, 'audit')
    candidates = prioritise(review_rows(args.audit_dir / 'source_records.csv'))
    documents = (candidates.groupby(['task_id', 'doc_id', 'doi', 'pubmed_id', 'priority_score', 'curation_rationale'], dropna=False)
                 .agg(records=('activity_id', 'size'), molecules=('molecule_id', 'nunique'), assays=('assay_id', 'nunique'))
                 .reset_index().sort_values(['task_id', 'priority_score', 'records'], ascending=[True, False, False]))
    with stage_output(args.output) as out:
        candidates.to_csv(out / 'candidate_records.csv', index=False)
        documents.to_csv(out / 'candidate_documents.csv', index=False)
        summary = candidates.groupby('task_id').agg(records=('activity_id', 'size'), molecules=('molecule_id', 'nunique'), documents=('doc_id', 'nunique')).reset_index()
        summary.to_csv(out / 'summary.csv', index=False)
        lines = ['# Human evidence expansion queue', '',
                 'These records remain **review** records. This queue does not modify datasets, split manifests, models, or frozen evaluations.', '',
                 'Rows require a finite positive value, exact relation, no source validity flag, and no duplicate flag. Priority is based only on text/unit signals that identify which source tables merit manual verification.', '',
                 'A record can be accepted only after original-source evidence establishes: absolute oral F with an IV reference; terminal IV half-life; or microsomal CLint normalized per mg protein.', '',
                 '## Queue size', '', '| Task | Records | Molecules | Documents |', '|---|---:|---:|---:|']
        lines += [f'| {row.task_id} | {row.records} | {row.molecules} | {row.documents} |' for row in summary.itertuples(index=False)]
        (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        finish_stage(out, 'human_evidence_expansion', inputs={'audit_complete_sha256': sha256(args.audit_dir / 'complete.json')}, partial=False)


if __name__ == '__main__':
    run_cli(run)

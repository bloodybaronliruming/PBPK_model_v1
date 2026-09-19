"""Build a label-blinded, non-Obach queue for independent Thalf validation curation."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem

from audit_thalf_accepted_sources import _document_risk_flags, load_document_metadata
from pipeline_common import (ROOT, configure_logging, finish_stage, run_cli, sha256,
                             stable_id, stage_output, startup_self_check, verify_stage)


TASK = 'Thalf__human__terminal_iv'
OBACH_DOI = '10.1124/dr.108.000117'
FORBIDDEN_LABEL_COLUMNS = {
    'standard_value', 'canonical_value', 'raw_value', 'target_value', 'tdc_value', 'y',
}
BLINDED_COLUMNS = [
    'candidate_id', 'activity_id', 'assay_id', 'doc_id', 'molecule_id', 'smiles',
    'doi', 'pubmed_id', 'standard_type', 'standard_units', 'standard_relation',
    'description', 'curation_rationale', 'title', 'doc_type', 'journal', 'year',
    'title_risk_flags',
    'document_candidate_records', 'document_candidate_molecules', 'screening_rank',
]
SECONDARY_TITLE_PATTERN = re.compile(
    r'\breview\b|\boverview\b|\badvances?\b|\bupdate\b|\binsight\b|\bevolution\b|'
    r'synthetic approaches|\bstrategy\b|essential medicinal chemistry|from concept to clinic|'
    r'recent progress|pharmacology of', re.IGNORECASE)
INCOMPATIBLE_DESCRIPTION_PATTERN = re.compile(
    r'\b(?:rat|rats|mouse|mice|dog|dogs|monkey|monkeys)\b|microsom|metabolic stability',
    re.IGNORECASE)


def normalize_doi(value):
    text = str(value or '').strip().lower()
    text = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', text)
    return text.rstrip(' .;,')


def canonical_smiles(value):
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f'无法规范化候选结构: {value}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def title_risk_flags(value):
    flags = [item for item in _document_risk_flags(value).split(';') if item]
    if SECONDARY_TITLE_PATTERN.search(str(value or '')):
        flags.append('likely_secondary_source')
    return ';'.join(dict.fromkeys(flags))


def build_queue(candidates, source, tdc_structures):
    """Apply leakage exclusions without inspecting candidate endpoint values."""
    frame = candidates.loc[candidates.task_id.eq(TASK)].copy()
    counts = [('input_thalf_review_queue', len(frame), frame.molecule_id.nunique(), frame.doc_id.nunique())]

    exact = frame.standard_relation.fillna('').astype(str).str.strip().eq('=')
    clean = frame.data_validity_comment.fillna('').astype(str).str.strip().eq('')
    duplicate = frame.potential_duplicate.fillna('').astype(str).str.lower().isin(
        {'0', '0.0', 'false', ''})
    doi = frame.doi.map(normalize_doi)
    eligible = (frame.status.eq('review') & exact & clean & duplicate & doi.ne(''))
    frame = frame.loc[eligible].copy()
    frame['normalized_doi'] = frame.doi.map(normalize_doi)
    counts.append(('clean_exact_doi_records', len(frame), frame.molecule_id.nunique(), frame.doc_id.nunique()))

    frame = frame.loc[~frame.description.fillna('').astype(str).str.contains(
        INCOMPATIBLE_DESCRIPTION_PATTERN)].copy()
    counts.append(('exclude_explicit_nonhuman_or_in_vitro', len(frame), frame.molecule_id.nunique(),
                   frame.doc_id.nunique()))

    accepted = source.loc[source.task_id.eq(TASK) & source.status.eq('accepted')].copy()
    used_molecules = set(accepted.molecule_id.astype(str))
    used_documents = set(accepted.doc_id.astype(str))
    used_dois = set(accepted.doi.map(normalize_doi)) - {''}
    keep = (~frame.molecule_id.astype(str).isin(used_molecules)
            & ~frame.doc_id.astype(str).isin(used_documents)
            & ~frame.normalized_doi.isin(used_dois))
    frame = frame.loc[keep].copy()
    counts.append(('exclude_v15_molecules_and_sources', len(frame), frame.molecule_id.nunique(), frame.doc_id.nunique()))

    frame['canonical_smiles'] = frame.smiles.map(canonical_smiles)
    frame = frame.loc[~frame.canonical_smiles.isin(set(tdc_structures))
                      & frame.normalized_doi.ne(OBACH_DOI)].copy()
    counts.append(('exclude_tdc_structures_and_obach', len(frame), frame.molecule_id.nunique(), frame.doc_id.nunique()))

    frame = frame.drop_duplicates(['activity_id']).copy()
    frame['candidate_id'] = frame.apply(
        lambda row: stable_id(f"{row.doc_id}|{row.activity_id}|{row.molecule_id}"), axis=1)
    document_counts = (frame.groupby('doc_id', dropna=False)
                       .agg(document_candidate_records=('activity_id', 'size'),
                            document_candidate_molecules=('molecule_id', 'nunique'))
                       .reset_index())
    frame = frame.merge(document_counts, on='doc_id', how='left', validate='many_to_one')
    frame = frame.sort_values(
        ['document_candidate_molecules', 'document_candidate_records', 'normalized_doi',
         'molecule_id', 'activity_id'],
        ascending=[False, False, True, True, True], ignore_index=True)
    frame['screening_rank'] = range(1, len(frame) + 1)
    counts.append(('final_blinded_queue', len(frame), frame.molecule_id.nunique(), frame.doc_id.nunique()))
    return frame, pd.DataFrame(counts, columns=['step', 'records', 'molecules', 'documents'])


def assert_label_blinded(frame):
    forbidden = {column for column in frame.columns if column.lower() in FORBIDDEN_LABEL_COLUMNS}
    if forbidden:
        raise ValueError(f'盲态清单含禁止标签列: {sorted(forbidden)}')


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--candidate-dir', type=Path,
                        default=ROOT / 'results/analysis/human_evidence_expansion_v15')
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'data/processed_v15/audit')
    parser.add_argument('--tdc-dir', type=Path,
                        default=ROOT / 'data/external/tdc_benchmark/admet_group/half_life_obach')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/analysis/thalf_external_validation_queue_v4')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    configure_logging(args.root, 'build_Thalf_external_validation_queue')

    candidate_path = args.candidate_dir / 'candidate_records.csv'
    source_path = args.audit_dir / 'source_records.csv'
    tdc_paths = [args.tdc_dir / 'train_val.csv', args.tdc_dir / 'test.csv']
    frozen_path = args.root / 'models/frozen/Thalf__human__terminal_iv/v15_rdkit2d_et3_v1/complete.json'
    test_path = args.root / 'results/final/Thalf_v15_rdkit2d_et3_test_v1/complete.json'
    required = [args.candidate_dir / 'complete.json', args.audit_dir / 'complete.json',
                candidate_path, source_path, frozen_path, test_path, *tdc_paths]
    startup_self_check(required, output=None if args.check_only else args.output)
    verify_stage(args.candidate_dir, 'human_evidence_expansion')
    audit_meta = verify_stage(args.audit_dir, 'audit')
    verify_stage(frozen_path.parent, 'thalf_v15_frozen_candidate')
    verify_stage(test_path.parent, 'thalf_v15_frozen_test_evaluation')

    candidate_columns = set(pd.read_csv(candidate_path, nrows=0).columns)
    required_candidate = {
        'task_id', 'status', 'standard_relation', 'data_validity_comment',
        'potential_duplicate', 'molecule_id', 'smiles', 'activity_id', 'assay_id',
        'doc_id', 'doi', 'pubmed_id', 'standard_type', 'standard_units', 'description',
        'curation_rationale',
    }
    if not required_candidate <= candidate_columns:
        raise ValueError(f'候选记录缺少列: {sorted(required_candidate - candidate_columns)}')
    source_columns = ['task_id', 'status', 'molecule_id', 'doc_id', 'doi']
    candidates = pd.read_csv(candidate_path, keep_default_na=False, low_memory=False)
    source = pd.read_csv(source_path, usecols=source_columns, keep_default_na=False,
                         low_memory=False, dtype=str)
    tdc_structures = set()
    for path in tdc_paths:
        tdc = pd.read_csv(path, usecols=['Drug'])
        tdc_structures.update(tdc.Drug.map(canonical_smiles))
    queue, exclusions = build_queue(candidates, source, tdc_structures)

    database = audit_meta.get('source_database', {}).get('path')
    if not database:
        raise ValueError('audit complete.json 未记录 ChEMBL 来源数据库')
    metadata = load_document_metadata(database, queue.doc_id)
    queue = queue.merge(metadata, on='doc_id', how='left', validate='many_to_one')
    queue['title_risk_flags'] = queue.title.map(title_risk_flags)
    queue = queue.sort_values(
        ['title_risk_flags', 'document_candidate_molecules', 'document_candidate_records',
         'normalized_doi', 'molecule_id', 'activity_id'],
        ascending=[True, False, False, True, True, True], ignore_index=True)
    queue['screening_rank'] = range(1, len(queue) + 1)
    blinded = queue[BLINDED_COLUMNS].copy()
    assert_label_blinded(blinded)

    documents = (blinded.sort_values('screening_rank')
                 .groupby(['doc_id', 'doi', 'pubmed_id', 'title', 'doc_type', 'journal', 'year',
                           'title_risk_flags'],
                          dropna=False, sort=False)
                 .agg(records=('activity_id', 'size'), molecules=('molecule_id', 'nunique'),
                      first_screening_rank=('screening_rank', 'min'))
                 .reset_index().sort_values('first_screening_rank', ignore_index=True))
    assert_label_blinded(documents)
    template = blinded[['candidate_id', 'activity_id', 'doc_id', 'molecule_id', 'doi']].copy()
    for column in ['fulltext_path', 'eligibility_decision', 'route_evidence',
                   'terminal_phase_evidence', 'parent_systemic_evidence', 'population',
                   'source_table_or_page', 'extracted_value', 'extracted_unit',
                   'exclusion_reason', 'reviewer', 'review_date']:
        template[column] = ''

    if args.check_only:
        logging.info('外部验证候选自检通过: %d 条、%d 分子、%d 文献；输出保持标签盲态。',
                     len(blinded), blinded.molecule_id.nunique(), blinded.doc_id.nunique())
        return

    with stage_output(args.output) as out:
        blinded.to_csv(out / 'candidate_records_blinded.csv', index=False)
        documents.to_csv(out / 'candidate_documents.csv', index=False)
        exclusions.to_csv(out / 'exclusion_summary.csv', index=False)
        template.to_csv(out / 'curation_template.csv', index=False)
        protocol = f"""# Independent human IV terminal half-life validation protocol

This queue was created only after the v15 candidate and one-time test evaluation were frozen. It contains no endpoint-value column and must not be joined back to ChEMBL labels during eligibility screening.

## Independence rule

- Exclude every molecule and source document already accepted into v15 for `{TASK}`.
- Exclude every exact canonical structure in TDC Half_Life_Obach train/validation/test and the Obach source itself.
- Make eligibility decisions from the original article before entering the numerical value.

## Inclusion criteria

- Original human pharmacokinetic study with intravenous administration.
- Parent drug measured in systemic plasma or serum.
- Explicit terminal/elimination-phase half-life point estimate with a recoverable unit.
- Sufficient source location, population, route and phase evidence for independent audit.

## Exclusion criteria

- Review, compilation, historical comparator or value copied from another paper.
- Oral-only, absorption/distribution/effective half-life, metabolite, tissue-fluid or total-radioactivity result.
- Unclear terminal phase, mixed route, non-human study, censored/range-only value, or unresolved unit.

## Sample-size gate

The screening queue contains {blinded.molecule_id.nunique()} molecules from {blinded.doc_id.nunique()} documents. Do not evaluate the frozen model until the curation registry is locked. Aim for at least 50 eligible unique molecules from at least 10 independent original documents, preferably 100 molecules from at least 20 documents; no single document should contribute more than 20% of the final set. If fewer than 50 molecules survive, report a pilot external validation with wide bootstrap intervals rather than a definitive benchmark.
"""
        (out / 'protocol.md').write_text(protocol, encoding='utf-8')
        finish_stage(out, 'thalf_external_validation_queue', inputs={
            'candidate_complete_sha256': sha256(args.candidate_dir / 'complete.json'),
            'audit_complete_sha256': sha256(args.audit_dir / 'complete.json'),
            'frozen_candidate_complete_sha256': sha256(frozen_path),
            'frozen_test_complete_sha256': sha256(test_path),
            'tdc_train_val_sha256': sha256(tdc_paths[0]),
            'tdc_test_sha256': sha256(tdc_paths[1]),
        }, task_id=TASK, label_blinded=True, records=len(blinded),
                     molecules=blinded.molecule_id.nunique(), documents=blinded.doc_id.nunique(),
                     partial=False)
    logging.info('盲态外部验证候选队列完成: %s；%d 条、%d 分子、%d 文献。',
                 args.output, len(blinded), blinded.molecule_id.nunique(), blinded.doc_id.nunique())


if __name__ == '__main__':
    run_cli(run)

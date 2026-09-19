"""Audit automatically accepted human terminal-IV half-life sources without using test records."""
from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

from pipeline_common import (ROOT, base_parser, configure_logging, finish_stage, run_cli, sha256,
                             stage_output, startup_self_check, verify_stage)


TASK = 'Thalf__human__terminal_iv'
SPLITS = ('train', 'val')
REQUIRED_COLUMNS = {
    'molecule_id', 'split', 'activity_id', 'assay_id', 'doc_id', 'standard_type',
    'standard_value', 'standard_units', 'standard_relation', 'description', 'doi',
    'pubmed_id', 'status', 'task_id', 'evidence_override_decision',
}
TRIGGER_PATTERN = re.compile(r'terminal|elimination|beta|lambda', re.IGNORECASE)
RISK_PATTERNS = (
    ('distribution_phase', re.compile(r'distribution|alpha(?:\s|-)?phase', re.IGNORECASE)),
    ('absorption_phase', re.compile(r'absorption|absorption-phase', re.IGNORECASE)),
    ('primary_phase', re.compile(r'primary(?:\s|-)?phase', re.IGNORECASE)),
    ('effective_half_life', re.compile(r'effective(?:\s|-)?half(?:\s|-)?life', re.IGNORECASE)),
    ('non_parent_analyte', re.compile(r'metabolite|metabolic product|alkylating activity|total radioactivity', re.IGNORECASE)),
    ('secondary_or_review', re.compile(r'\breview\b|literature value|previous(?:ly)? reported', re.IGNORECASE)),
)
DOCUMENT_RISK_PATTERNS = (
    ('nonhuman_model', re.compile(r'\b(?:mice|mouse|rats?|dogs?|monkeys?)\b|humanized liver', re.IGNORECASE)),
    ('review_or_overview', re.compile(r'\breview\b|\boverview\b|recent progress|perspective', re.IGNORECASE)),
    ('medicinal_chemistry_secondary', re.compile(r'\bdesign\b.*\bsynthesis\b|drug discovery|biological evaluation', re.IGNORECASE)),
    ('metabolite_or_activity', re.compile(r'metabolite|6-mercaptopurine|alkylating activity|monohydroxy derivative', re.IGNORECASE)),
)


def _joined_matches(pattern, value):
    return ';'.join(dict.fromkeys(match.group(0).lower() for match in pattern.finditer(str(value or ''))))


def _risk_flags(value):
    return ';'.join(name for name, pattern in RISK_PATTERNS if pattern.search(str(value or '')))


def _document_risk_flags(value):
    return ';'.join(name for name, pattern in DOCUMENT_RISK_PATTERNS if pattern.search(str(value or '')))


def load_document_metadata(database, doc_ids):
    database = Path(database)
    startup_self_check([database])
    ids = sorted({int(value) for value in doc_ids})
    rows = []
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            placeholders = ','.join('?' for _ in batch)
            rows.extend(connection.execute(
                f'SELECT doc_id,title,doc_type,journal,year FROM docs WHERE doc_id IN ({placeholders})', batch))
    return pd.DataFrame(rows, columns=['doc_id', 'title', 'doc_type', 'journal', 'year'])


def build_tables(source):
    """Return provenance summaries after excluding test rows before evidence analysis."""
    scoped = source.loc[source.split.isin(SPLITS) & source.task_id.eq(TASK)].copy()
    accepted = scoped.loc[scoped.status.eq('accepted')].copy()
    decisions = accepted.evidence_override_decision.fillna('').astype(str).str.strip().str.lower()
    unknown = sorted(set(decisions) - {'', 'accepted'})
    if unknown:
        raise ValueError(f'Accepted records contain incompatible evidence decisions: {unknown}')

    accepted['provenance'] = decisions.map({'': 'automatic_rule', 'accepted': 'manual_source_decision'})
    summary = (accepted.groupby(['split', 'provenance'], sort=True, dropna=False)
               .agg(records=('activity_id', 'size'), molecules=('molecule_id', 'nunique'),
                    documents=('doc_id', 'nunique')).reset_index())

    automatic = accepted.loc[accepted.provenance.eq('automatic_rule')].copy()
    source_text = automatic.standard_type.fillna('').astype(str) + ' ' + automatic.description.fillna('').astype(str)
    automatic['automatic_trigger_terms'] = source_text.map(lambda value: _joined_matches(TRIGGER_PATTERN, value))
    automatic['risk_flags'] = source_text.map(_risk_flags)
    automatic['risk_flagged'] = automatic.risk_flags.ne('')
    automatic = automatic.sort_values(['risk_flagged', 'doi', 'doc_id', 'assay_id', 'activity_id'],
                                      ascending=[False, True, True, True, True], ignore_index=True)

    document_key = automatic.doi.fillna('').astype(str).str.strip()
    automatic['document_key'] = document_key.where(document_key.ne(''), 'doc_id:' + automatic.doc_id.astype(str))
    documents = (automatic.groupby(['document_key', 'doc_id', 'doi', 'pubmed_id'], sort=True, dropna=False)
                 .agg(records=('activity_id', 'size'), molecules=('molecule_id', 'nunique'),
                      assays=('assay_id', 'nunique'), train_records=('split', lambda x: int(x.eq('train').sum())),
                      val_records=('split', lambda x: int(x.eq('val').sum())),
                      risk_flagged_records=('risk_flagged', 'sum'),
                      trigger_terms=('automatic_trigger_terms', lambda x: ';'.join(sorted({v for s in x for v in str(s).split(';') if v}))),
                      risk_flags=('risk_flags', lambda x: ';'.join(sorted({v for s in x for v in str(s).split(';') if v}))))
                 .reset_index())
    documents['review_priority'] = (documents.risk_flagged_records * 100
                                    + documents.molecules * 10 + documents.records)
    documents = documents.sort_values(['review_priority', 'molecules', 'records', 'document_key'],
                                      ascending=[False, False, False, True], ignore_index=True)
    return automatic, documents, summary


def audit(audit_dir, output):
    meta = verify_stage(audit_dir, 'audit')
    source_path = audit_dir / 'source_records.csv'
    startup_self_check([source_path], output=output, columns={source_path: REQUIRED_COLUMNS})
    source = pd.read_csv(source_path, keep_default_na=False, low_memory=False)
    automatic, documents, summary = build_tables(source)
    database = meta.get('source_database', {}).get('path')
    if not database:
        raise ValueError('Audit metadata does not identify the read-only source database')
    metadata = load_document_metadata(database, automatic.doc_id)
    automatic = automatic.merge(metadata, on='doc_id', how='left', validate='many_to_one')
    documents = documents.merge(metadata, on='doc_id', how='left', validate='one_to_one')
    documents['document_risk_flags'] = documents.title.map(_document_risk_flags)
    documents['review_priority'] += documents.document_risk_flags.ne('') * 1000
    documents = documents.sort_values(['review_priority', 'molecules', 'records', 'document_key'],
                                      ascending=[False, False, False, True], ignore_index=True)
    with stage_output(output) as out:
        automatic.to_csv(out / 'automatic_accepted_records.csv', index=False)
        documents.to_csv(out / 'automatic_accepted_documents.csv', index=False)
        summary.to_csv(out / 'summary.csv', index=False)
        lines = [
            '# Human terminal-IV half-life accepted-source audit', '',
            'This report covers train and validation source records only. Test records and test metrics are not reported.', '',
            'Automatic acceptance means that the endpoint rule accepted the source text without an activity-level original-source decision.', '',
            f'- Automatically accepted source records: {len(automatic)}',
            f'- Automatically accepted documents: {automatic.document_key.nunique()}',
            f'- Description-risk-flagged source records: {int(automatic.risk_flagged.sum())}',
            f'- Title-risk-flagged documents: {int(documents.document_risk_flags.ne("").sum())}', '',
            'Risk flags are triage signals, not final rejection decisions. Every exclusion still requires an activity-level registry entry backed by original-source evidence.',
        ]
        (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        finish_stage(out, 'thalf_accepted_source_audit',
                     inputs={'audit_complete_sha256': sha256(audit_dir / 'complete.json')},
                     source_rule_version=meta.get('rule_version'), task_id=TASK, splits=list(SPLITS), partial=False)
    logging.info('Thalf 已接受来源审计完成: %s；自动记录 %d，文献 %d，描述风险 %d，标题风险 %d。',
                 output, len(automatic), automatic.document_key.nunique(), automatic.risk_flagged.sum(),
                 documents.document_risk_flags.ne('').sum())


def main():
    parser = base_parser(__doc__)
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'data/processed_v11/audit')
    args = parser.parse_args()
    configure_logging(args.root, 'audit_thalf_accepted_sources')
    output = args.output or args.root / 'results/analysis/thalf_accepted_sources_v1'
    source_path = args.audit_dir / 'source_records.csv'
    startup_self_check([args.audit_dir / 'complete.json', source_path], output=output,
                       columns={source_path: REQUIRED_COLUMNS})
    verify_stage(args.audit_dir, 'audit')
    if args.check_only:
        logging.info('Thalf 已接受来源审计输入与字段自检通过；未生成报告。')
        return
    audit(args.audit_dir, output)


if __name__ == '__main__':
    run_cli(main)

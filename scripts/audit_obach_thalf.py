"""Batch-audit Obach human IV terminal half-life records and build a decision registry."""
from __future__ import annotations

import argparse
import logging
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from human_evidence_overrides import load_human_evidence_decisions
from pipeline_common import (ROOT, configure_logging, finish_stage, run_cli, sha256,
                             stage_output, startup_self_check, verify_stage)
from papp_provenance_overrides import normalized_doi


DOI = '10.1124/dmd.108.020479'
TASK = 'Thalf__human__terminal_iv'
SOURCE_REFERENCE = 'Abstract p. 1385; Table 1 pp. 1386-1395; Materials and Methods p. 1395'
SOURCE_NOTE = ('Obach et al. define the compiled t1/2 values as terminal-phase half-life obtained or derived '
               'from human studies using intravenous administration exclusively.')
REGISTRY_COLUMNS = ['activity_id', 'source_doi', 'task_id', 'reviewer_decision',
                    'source_table_or_page', 'evidence_note', 'reviewer', 'review_date']
SOURCE_COLUMNS = ['activity_id', 'molecule_id', 'molecule_chembl_id', 'smiles', 'split',
                  'assay_id', 'assay_chembl_id', 'doi', 'task_id', 'status', 'reason',
                  'standard_type', 'standard_value', 'standard_units', 'standard_relation',
                  'canonical_value', 'canonical_unit', 'relation', 'assay_organism',
                  'assay_tax_id', 'description', 'data_validity_comment', 'potential_duplicate']


def extract_pdf_text(path: Path) -> str:
    with tempfile.TemporaryDirectory() as directory:
        text_path = Path(directory) / 'source.txt'
        try:
            subprocess.run(['pdftotext', str(path), str(text_path)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError('Obach 原文核验需要系统命令 pdftotext') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Obach PDF 文本提取失败: {exc.stderr.strip()}') from exc
        return text_path.read_text(encoding='utf-8', errors='replace')


def validate_pdf_evidence(text: str) -> None:
    normalized = re.sub(r'\s+', ' ', text).lower()
    required = (
        'terminal phase half-life were obtained or derived from original references exclusively from studies utilizing i.v. administration',
        'pharmacokinetic data included in this database are strictly from i.v. administration',
        'no data included from oral, i.m., or any other dosing routes',
        'summary of i.v. pharmacokinetic parameters and plasma protein binding values for 670 compounds in humans',
    )
    missing = [phrase for phrase in required if phrase not in normalized]
    if missing:
        raise ValueError(f'Obach PDF 缺少预期证据文本: {missing}')


def read_tdc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep='\t')
    aliases = {'ID': 'tdc_chembl_id', 'X': 'tdc_smiles', 'Y': 'tdc_value',
               'Drug_ID': 'tdc_chembl_id', 'Drug': 'tdc_smiles'}
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    required = {'tdc_chembl_id', 'tdc_smiles', 'tdc_value'}
    if not required <= set(frame.columns):
        raise ValueError(f'TDC Obach 表缺少列: {sorted(required - set(frame.columns))}')
    frame = frame[list(required)].copy()
    frame['tdc_value'] = pd.to_numeric(frame.tdc_value, errors='coerce')
    if (frame.tdc_chembl_id.astype(str).str.strip().eq('').any()
            or frame.tdc_smiles.astype(str).str.strip().eq('').any()
            or not np.isfinite(frame.tdc_value).all() or frame.tdc_value.le(0).any()):
        raise ValueError('TDC Obach 表存在空标识、空结构或无效半衰期')
    return frame


def evaluate_records(source: pd.DataFrame, tdc: pd.DataFrame) -> pd.DataFrame:
    """Return one decision row per activity without exposing test labels in reports."""
    missing = set(SOURCE_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f'审计 source_records.csv 缺少列: {sorted(missing)}')
    selected = source.loc[
        source.doi.map(normalized_doi).eq(DOI) & source.task_id.eq(TASK), SOURCE_COLUMNS
    ].copy()
    if selected.empty:
        raise ValueError('没有找到 Obach Thalf 审计记录')
    if selected.activity_id.duplicated().any():
        raise ValueError('Obach Thalf activity_id 不唯一')

    numeric = pd.to_numeric(selected.canonical_value, errors='coerce')
    tax_id = pd.to_numeric(selected.assay_tax_id, errors='coerce')
    duplicate = selected.potential_duplicate.astype(str).str.strip().str.lower()
    checks = {
        'single_assay': pd.Series(selected.assay_id.nunique() == 1, index=selected.index),
        'expected_review_state': selected.status.eq('review') & selected.reason.eq('not_confirmed_IV_terminal_half_life'),
        'human': selected.assay_organism.eq('Homo sapiens') & tax_id.eq(9606),
        'iv_description': selected.description.astype(str).str.lower().str.contains(r'\biv\b|intravenous', regex=True),
        'half_life_type': selected.standard_type.eq('T1/2'),
        'exact_relation': selected.standard_relation.eq('=') & selected.relation.eq('='),
        'hours': selected.standard_units.astype(str).str.lower().isin(['h', 'hr', 'hrs', 'hour', 'hours']) & selected.canonical_unit.eq('h'),
        'positive_finite_value': numeric.gt(0) & np.isfinite(numeric),
        'no_validity_issue': selected.data_validity_comment.fillna('').astype(str).str.strip().eq(''),
        'not_potential_duplicate': duplicate.isin(['0', '0.0', 'false']),
    }
    failed = pd.DataFrame(checks).apply(lambda row: ';'.join(row.index[~row]), axis=1)

    value_lookup = {
        chembl_id: group.tdc_value.to_numpy(dtype=float)
        for chembl_id, group in tdc.groupby('tdc_chembl_id', sort=False)
    }
    selected['tdc_crosscheck'] = [
        ('tdc_id_and_value_confirmed'
         if chembl_id in value_lookup and np.isclose(float(value), value_lookup[chembl_id], rtol=1e-7, atol=1e-9).any()
         else 'tdc_value_conflict' if chembl_id in value_lookup
         else 'primary_article_scope_only')
        for chembl_id, value in zip(selected.molecule_chembl_id, numeric)
    ]
    selected['quarantine_reason'] = failed
    conflict = selected.tdc_crosscheck.eq('tdc_value_conflict')
    selected.loc[conflict, 'quarantine_reason'] = selected.loc[conflict, 'quarantine_reason'].map(
        lambda value: ';'.join(part for part in (value, 'tdc_value_conflict') if part))
    selected['reviewer_decision'] = np.where(selected.quarantine_reason.eq(''), 'accepted', 'quarantined')
    return selected


def build_registry(base: pd.DataFrame, decisions: pd.DataFrame, review_date: str) -> pd.DataFrame:
    missing = set(REGISTRY_COLUMNS) - set(base.columns)
    if missing:
        raise ValueError(f'基础人源证据表缺少列: {sorted(missing)}')
    accepted = decisions.loc[decisions.reviewer_decision.eq('accepted')].copy()
    additions = pd.DataFrame({
        'activity_id': accepted.activity_id.astype(int),
        'source_doi': DOI,
        'task_id': TASK,
        'reviewer_decision': 'accepted',
        'source_table_or_page': SOURCE_REFERENCE,
        'evidence_note': [f'{SOURCE_NOTE} Cross-check: {status}.' for status in accepted.tdc_crosscheck],
        'reviewer': 'project_curation_obach_batch',
        'review_date': review_date,
    })
    overlap = set(pd.to_numeric(base.activity_id, errors='raise').astype(int)) & set(additions.activity_id)
    if overlap:
        raise ValueError(f'基础证据表已含 Obach activity_id，禁止重复追加: {sorted(overlap)[:5]}')
    result = pd.concat([base[REGISTRY_COLUMNS], additions[REGISTRY_COLUMNS]], ignore_index=True)
    if result.activity_id.duplicated().any():
        raise ValueError('合并后人源证据表 activity_id 不唯一')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'data/processed_v14/audit')
    parser.add_argument('--source-pdf', type=Path,
                        default=ROOT / '参考文献/补充文献/Thalf文献/obach2008.pdf')
    parser.add_argument('--tdc-table', type=Path, default=ROOT / 'data/external/tdc/half_life_obach.tab')
    parser.add_argument('--base-registry', type=Path, default=ROOT / 'results/analysis/human_evidence_decisions_v12.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis/obach_thalf_audit_v1')
    parser.add_argument('--review-date', default='2026-09-12')
    args = parser.parse_args()
    configure_logging(args.root, 'audit_obach_thalf')
    required = [args.audit_dir / 'complete.json', args.audit_dir / 'source_records.csv',
                args.source_pdf, args.tdc_table, args.base_registry]
    startup_self_check(required, output=args.output)
    verify_stage(args.audit_dir, 'audit')
    validate_pdf_evidence(extract_pdf_text(args.source_pdf))
    source = pd.read_csv(args.audit_dir / 'source_records.csv', low_memory=False)
    tdc = read_tdc(args.tdc_table)
    evaluated = evaluate_records(source, tdc)
    base = pd.read_csv(args.base_registry, dtype=str, keep_default_na=False)
    registry = build_registry(base, evaluated, args.review_date)

    accepted = evaluated.reviewer_decision.eq('accepted')
    quarantined = ~accepted
    with stage_output(args.output) as out:
        train_val = evaluated.loc[evaluated.split.isin(['train', 'val'])].copy()
        train_val.to_csv(out / 'train_val_crosscheck.csv', index=False)
        blind_columns = ['activity_id', 'molecule_id', 'molecule_chembl_id', 'split', 'assay_id',
                         'assay_chembl_id', 'doi', 'tdc_crosscheck', 'reviewer_decision', 'quarantine_reason']
        evaluated.loc[evaluated.split.eq('test'), blind_columns].to_csv(out / 'blind_test_manifest.csv', index=False)
        quarantine_columns = blind_columns + ['task_id', 'status', 'reason']
        evaluated.loc[quarantined, quarantine_columns].to_csv(out / 'quarantined_records.csv', index=False)
        duplicates = (evaluated.groupby(['split', 'molecule_id', 'molecule_chembl_id'], dropna=False)
                      .agg(activity_records=('activity_id', 'size')).reset_index())
        duplicates.loc[duplicates.activity_records.gt(1)].to_csv(out / 'duplicate_source_molecules.csv', index=False)
        summary = (evaluated.assign(accepted=accepted, quarantined=quarantined)
                   .groupby('split').agg(records=('activity_id', 'size'), molecules=('molecule_id', 'nunique'),
                                         accepted=('accepted', 'sum'), quarantined=('quarantined', 'sum'),
                                         tdc_id_value_confirmed=('tdc_crosscheck', lambda x: x.eq('tdc_id_and_value_confirmed').sum()),
                                         primary_article_scope_only=('tdc_crosscheck', lambda x: x.eq('primary_article_scope_only').sum()))
                   .reset_index())
        summary.to_csv(out / 'summary.csv', index=False)
        registry.to_csv(out / 'human_evidence_decisions_v13.csv', index=False)
        readme = (
            '# Obach human IV terminal half-life batch audit\n\n'
            'The primary article defines Table 1 half-life values as terminal-phase human pharmacokinetic '
            'parameters obtained or derived exclusively from intravenous studies. TDC is used only as a '
            'structured mirror cross-check; absence from TDC is not treated as contradictory evidence.\n\n'
            'Test labels are intentionally absent from `blind_test_manifest.csv`, `quarantined_records.csv`, '
            'and `summary.csv`. Only the train/validation cross-check artifact contains values.\n'
        )
        (out / 'README.md').write_text(readme, encoding='utf-8')
        finish_stage(out, 'obach_thalf_audit', inputs={
            'audit_complete_sha256': sha256(args.audit_dir / 'complete.json'),
            'source_pdf_sha256': sha256(args.source_pdf),
            'tdc_table_sha256': sha256(args.tdc_table),
            'base_registry_sha256': sha256(args.base_registry),
        }, partial=False, source_doi=DOI, source_records=int(len(evaluated)),
                     accepted_records=int(accepted.sum()), quarantined_records=int(quarantined.sum()),
                     tdc_records=int(len(tdc)), test_labels_exported=False)

    load_human_evidence_decisions(args.output / 'human_evidence_decisions_v13.csv')
    logging.info('Obach 批量核验完成: %d 条接受，%d 条隔离；测试标签未导出。', accepted.sum(), quarantined.sum())


if __name__ == '__main__':
    run_cli(main)

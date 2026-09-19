"""将人工原文单位裁定作为 Papp 来源审计的显式、可追溯输入。"""
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

from endpoint_rules import normalize_unit


MICRO_UNIT_DECISION = 'confirmed 10^-6 cm/s'
PRINTED_CM_DECISION = 'confirmed cm/s (as printed)'
UNRESOLVED_DECISION = 'unresolved'
ALLOWED_DECISIONS = {MICRO_UNIT_DECISION, PRINTED_CM_DECISION, UNRESOLVED_DECISION}
REQUIRED_COLUMNS = {'source_doi', 'reviewer_decision', 'source_table_or_page', 'evidence_note', 'reviewer', 'review_date'}
OVERRIDE_VERSION = '1.0.0'
PRINTED_CM_UNITS = {'cm/s', 'cm/sec'}
MICRO_CM_UNITS = {'10-6cm/s', "10'-6cm/s", '10^-6cm/s', '1e-6cm/s',
                  '10-6cm/sec', "10'-6cm/sec"}


def normalized_doi(value):
    return (str(value or '').strip().lower().removeprefix('doi:')
            .removeprefix('https://doi.org/').removeprefix('http://doi.org/'))


def load_papp_decisions(path):
    """读取人工裁定；拒绝不完整、未知或重复 DOI，避免默默采用错误表。"""
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f'Papp 裁定表缺少列: {sorted(REQUIRED_COLUMNS-set(reader.fieldnames or []))}')
        result = {}
        for line, row in enumerate(reader, start=2):
            doi = normalized_doi(row['source_doi'])
            decision = str(row['reviewer_decision'] or '').strip()
            if not doi:
                raise ValueError(f'Papp 裁定表第 {line} 行 DOI 为空')
            if decision not in ALLOWED_DECISIONS:
                raise ValueError(f'Papp 裁定表第 {line} 行有未知裁定: {decision!r}')
            if decision != UNRESOLVED_DECISION and (not row['source_table_or_page'].strip() or not row['evidence_note'].strip()):
                raise ValueError(f'Papp 裁定表第 {line} 行确认单位时必须给出表格/页码和证据')
            if doi in result:
                raise ValueError(f'Papp 裁定表 DOI 重复: {doi}')
            result[doi] = dict(decision=decision, source_table_or_page=row['source_table_or_page'].strip(),
                               evidence_note=row['evidence_note'].strip(), reviewer=row['reviewer'].strip(),
                               review_date=row['review_date'].strip())
    if not result:
        raise ValueError('Papp 裁定表为空')
    return result


def apply_papp_decision(record, result, decisions, unresolved_policy):
    """仅对已分类为 Papp 的来源记录覆盖单位；返回更新结果与谱系列。"""
    if unresolved_policy not in {'exclude', 'keep'}:
        raise ValueError(f'未知 unresolved policy: {unresolved_policy}')
    lineage = dict(literature_unit_decision='', literature_unit_policy='', literature_unit_source='')
    if result.get('endpoint') != 'Papp':
        return result, lineage
    doi = normalized_doi(record.get('doi'))
    entry = decisions.get(doi)
    if entry is None:
        lineage['literature_unit_policy'] = 'not_listed'
        return result, lineage
    decision = entry['decision']
    lineage.update(literature_unit_decision=decision, literature_unit_policy=unresolved_policy,
                   literature_unit_source=entry['source_table_or_page'])
    if decision == MICRO_UNIT_DECISION:
        unit = normalize_unit(record.get('standard_units'))
        if unit not in PRINTED_CM_UNITS | MICRO_CM_UNITS:
            raise ValueError(f'{doi} 已裁定为 10^-6 cm/s，但来源单位无法解释: {unit!r}')
        try:
            value = float(record['standard_value'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{doi} 缺少可覆盖的 Papp 数值') from exc
        if not math.isfinite(value):
            raise ValueError(f'{doi} 的 Papp 数值不是有限数')
        # 同一原文在 ChEMBL 中既有错误标为 cm/s 的行，也有已标为微单位的行。
        # DOI 裁定让两类记录均映射到同一物理量；只消除单位拼写造成的拒绝，保留其他审计拒绝理由。
        reasons = {item for item in str(result.get('reason') or '').split(';') if item}
        reasons.discard('unknown_unit_or_normalization_basis')
        result['reason'] = ';'.join(sorted(reasons))
        result['status'] = 'review' if reasons else ('accepted' if result['relation'] == '=' else 'censored')
        if result.get('canonical_value') is not None or not reasons:
            result['canonical_value'] = value
            result['lower_bound'] = value if result['relation'] in {'>', '>='} else None
            result['upper_bound'] = value if result['relation'] in {'<', '<='} else None
        result['conversion_rule'] = 'literature_doi_override:printed_1e-6_cm_s'
    elif decision == UNRESOLVED_DECISION and unresolved_policy == 'exclude':
        reasons = {item for item in str(result.get('reason') or '').split(';') if item}
        reasons.add('literature_unit_unresolved_excluded')
        result.update(status='review', reason=';'.join(sorted(reasons)), canonical_value=None,
                      lower_bound=None, upper_bound=None, conversion_rule='literature_unit_unresolved_excluded')
    elif decision == UNRESOLVED_DECISION:
        result['conversion_rule'] = result['conversion_rule'] + ';literature_unit_unresolved_kept'
    return result, lineage


def decision_counts(decisions):
    return dict(Counter(entry['decision'] for entry in decisions.values()))

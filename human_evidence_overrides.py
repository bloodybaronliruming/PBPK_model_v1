"""Apply narrowly scoped, activity-level source decisions for human PK evidence."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from papp_provenance_overrides import normalized_doi


ACCEPTED = 'accepted'
REJECTED = 'rejected'
ALLOWED_DECISIONS = {ACCEPTED, REJECTED}
REQUIRED_COLUMNS = {'activity_id', 'source_doi', 'task_id', 'reviewer_decision',
                    'source_table_or_page', 'evidence_note', 'reviewer', 'review_date'}
OVERRIDE_VERSION = '1.1.0'


def load_human_evidence_decisions(path):
    """Load decisions keyed by activity ID; reject incomplete or ambiguous evidence."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            raise ValueError(f'Human evidence registry is missing columns: {missing}')
        result = {}
        for line, row in enumerate(reader, start=2):
            try:
                activity_id = int(row['activity_id'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Human evidence registry line {line} has invalid activity_id') from exc
            doi = normalized_doi(row['source_doi'])
            task = str(row['task_id'] or '').strip()
            decision = str(row['reviewer_decision'] or '').strip()
            if activity_id <= 0 or not doi or not task or decision not in ALLOWED_DECISIONS:
                raise ValueError(f'Human evidence registry line {line} is incomplete or has an unknown decision')
            if not row['source_table_or_page'].strip() or not row['evidence_note'].strip():
                raise ValueError(f'Human evidence registry line {line} has no source evidence')
            if activity_id in result:
                raise ValueError(f'Human evidence registry repeats activity_id {activity_id}')
            result[activity_id] = dict(doi=doi, task_id=task, decision=decision,
                                       source=row['source_table_or_page'].strip(),
                                       note=row['evidence_note'].strip(),
                                       reviewer=row['reviewer'].strip(), review_date=row['review_date'].strip())
    if not result:
        raise ValueError('Human evidence registry is empty')
    return result


def _accepted_clint(record, result):
    if result['reason'] != 'unknown_unit_or_normalization_basis' or result['relation'] != '=':
        raise ValueError(f"CLint activity {record['activity_id']} has unexpected pre-override state")
    try:
        value = float(record['standard_value'])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CLint activity {record['activity_id']} has no numeric value") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"CLint activity {record['activity_id']} has invalid value")
    result.update(canonical_value=value, canonical_unit='uL/min/mg_protein',
                  conversion_rule='evidence_activity_override:ml_per_min_per_g_protein_to_ul_per_min_per_mg_protein',
                  status='accepted', reason='', lower_bound=None, upper_bound=None)


def _accepted_thalf(record, result):
    review_candidate = (result['status'] == 'review'
                        and result['reason'] == 'not_confirmed_IV_terminal_half_life')
    automatic_candidate = (result['status'] == 'accepted' and result['reason'] == ''
                           and result['route'] == 'IV')
    if (not (review_candidate or automatic_candidate) or result['relation'] != '='
            or result['canonical_value'] is None):
        raise ValueError(f"Thalf activity {record['activity_id']} has unexpected pre-override state")
    result.update(route='IV', status='accepted', reason='', lower_bound=None, upper_bound=None,
                  conversion_rule='evidence_activity_override:verified_IV_terminal_elimination_half_life')


def _accepted_f(record, result):
    if result['reason'] != 'not_confirmed_absolute_oral_F' or result['relation'] != '=' or result['canonical_value'] is None:
        raise ValueError(f"F activity {record['activity_id']} has unexpected pre-override state")
    try:
        value = float(result['canonical_value'])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"F activity {record['activity_id']} has no numeric fraction") from exc
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"F activity {record['activity_id']} has an invalid fraction")
    result.update(status='accepted', reason='', lower_bound=None, upper_bound=None,
                  conversion_rule='evidence_activity_override:verified_absolute_oral_bioavailability')


def apply_human_evidence_decision(record, result, decisions):
    """Apply only exact, pre-registered overrides; all other records pass through unchanged."""
    lineage = {'evidence_override_decision': '', 'evidence_override_source': '', 'evidence_override_note': ''}
    activity_id = record.get('activity_id')
    try:
        entry = decisions.get(int(activity_id))
    except (TypeError, ValueError):
        entry = None
    if entry is None:
        return result, lineage
    if normalized_doi(record.get('doi')) != entry['doi'] or result.get('task_id') != entry['task_id']:
        raise ValueError(f'Human evidence registry does not match source DOI/task for activity {activity_id}')
    lineage.update(evidence_override_decision=entry['decision'], evidence_override_source=entry['source'],
                   evidence_override_note=entry['note'])
    if entry['decision'] == REJECTED:
        result.update(canonical_value=None, conversion_rule='evidence_activity_override:rejected',
                      status='excluded', reason='evidence_activity_override_rejected',
                      lower_bound=None, upper_bound=None)
        return result, lineage
    if entry['task_id'] == 'CLint__human__microsome':
        _accepted_clint(record, result)
    elif entry['task_id'] == 'Thalf__human__terminal_iv':
        _accepted_thalf(record, result)
    elif entry['task_id'] == 'F__human__absolute_oral':
        _accepted_f(record, result)
    else:
        raise ValueError(f"Unsupported accepted human evidence task: {entry['task_id']}")
    return result, lineage

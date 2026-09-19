"""第一步：只读回查 ChEMBL，输出来源、保守端点分类和旧标签问题。"""
from __future__ import annotations
import csv
import json
import logging
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from pipeline_common import (base_parser, configure_logging, read_fixed_splits, startup_self_check,
                             stage_output, finish_stage, dump_json, run_cli, sha256)
from endpoint_rules import family, classify, RULE_VERSION
from papp_provenance_overrides import (OVERRIDE_VERSION, apply_papp_decision, decision_counts,
                                       load_papp_decisions)
from human_evidence_overrides import (OVERRIDE_VERSION as HUMAN_EVIDENCE_OVERRIDE_VERSION,
                                      apply_human_evidence_decision, load_human_evidence_decisions)

RAW_COLUMNS = ['activity_id', 'molregno', 'assay_id', 'doc_id', 'standard_type', 'standard_value',
               'standard_units', 'standard_relation', 'data_validity_comment', 'potential_duplicate',
               'assay_organism', 'assay_tax_id', 'assay_cell_type', 'assay_tissue',
               'assay_subcellular_fraction', 'description', 'assay_chembl_id', 'molecule_chembl_id', 'doi', 'pubmed_id']


def check_database(conn):
    schemas = {
        'activities': RAW_COLUMNS[:10],
        'assays': ['assay_id','assay_organism','assay_tax_id','assay_cell_type','assay_tissue','assay_subcellular_fraction','description','chembl_id'],
        'compound_structures': ['molregno','standard_inchi_key'],
        'molecule_dictionary': ['molregno','chembl_id'], 'docs': ['doc_id','doi','pubmed_id'],
        'assay_parameters': ['assay_id','standard_type','standard_value','standard_units','standard_text_value'],
    }
    for table, columns in schemas.items():
        conn.execute(f'SELECT {",".join(columns)} FROM {table} LIMIT 0')


def legacy_contracts(root):
    """核查旧值，不把旧值作为新训练标签。"""
    output = []
    for path in sorted((Path(root) / 'data/processed').glob('chembl_*.csv')):
        x = pd.read_csv(path)
        for col in x.columns:
            if not col.startswith('target_') or col[7:] not in x:
                continue
            ep = col[7:]
            raw, target = pd.to_numeric(x[ep], errors='coerce'), pd.to_numeric(x[col], errors='coerce')
            if ep in {'fu_unbound', 'Bioavailability_F'}:
                y = raw.clip(.01, .99)
                expected = np.log(y / (1-y))
            elif ep in {'pKa', 'LogP_LogD'}:
                expected = raw
            else:
                expected = np.log10(raw.where(raw > 0))
            mask = x.get('mask_'+ep, pd.Series(0, index=x.index))
            mismatch = expected.notna() & target.notna() & ~np.isclose(expected, target, atol=1e-7, rtol=1e-7)
            output.append(dict(file=path.name, endpoint=ep, raw_n=int(raw.notna().sum()),
                               present_but_mask_zero=int((raw.notna() & mask.eq(0)).sum()),
                               missing_but_mask_one=int((raw.isna() & mask.eq(1)).sum()),
                               target_mismatch=int(mismatch.sum())))
    return pd.DataFrame(output, columns=['file', 'endpoint', 'raw_n', 'present_but_mask_zero', 'missing_but_mask_one', 'target_mismatch'])


def audit(root, output, limit=None, papp_decision_registry=None, papp_unresolved_policy='exclude',
          human_evidence_registry=None):
    fixed, split_hashes = read_fixed_splits(root)
    raw_db = Path(root) / 'data/raw/chembl_37/chembl_37_sqlite/chembl_37.db'
    registry = load_papp_decisions(papp_decision_registry) if papp_decision_registry else {}
    human_decisions = load_human_evidence_decisions(human_evidence_registry) if human_evidence_registry else {}
    required = [raw_db] + ([Path(papp_decision_registry)] if papp_decision_registry else []) + ([Path(human_evidence_registry)] if human_evidence_registry else [])
    startup_self_check(required, output=output)
    RDLogger.DisableLog('rdApp.warning')
    with sqlite3.connect(raw_db.resolve().as_uri()+'?mode=ro', uri=True) as conn:
        check_database(conn)
        initial_stat = raw_db.stat()
        conn.execute('BEGIN')
        types = [r[0] for r in conn.execute('SELECT DISTINCT standard_type FROM activities') if family(r[0])]
        if not types:
            raise ValueError('没有可识别端点类型，请检查数据库版本。')
        chosen = fixed.head(limit).copy() if limit else fixed.copy()
        maps, source_owners = [], defaultdict(set)
        for r in tqdm(chosen.to_dict('records'), desc='分子来源映射'):
            mol = Chem.MolFromSmiles(r['smiles'])
            key = Chem.MolToInchiKey(mol) if mol is not None else ''
            ids = conn.execute('SELECT molregno FROM compound_structures WHERE standard_inchi_key=?', (key,)).fetchall() if key else []
            item = dict(r, inchi_key=key, mapping_status='matched' if ids else 'unmapped',
                        molregnos=json.dumps([i[0] for i in ids]))
            maps.append(item)
            for (idx,) in ids:
                source_owners[idx].add(r['molecule_id'])
        ambiguous = {owner for owners in source_owners.values() if len(owners)>1 for owner in owners}
        for m in maps:
            if m['molecule_id'] in ambiguous:
                m['mapping_status'] = 'ambiguous_source_owner'
        by_id = {m['molecule_id']: m for m in maps}
        source_map = {idx: by_id[next(iter(owners))] for idx, owners in source_owners.items()
                      if len(owners)==1 and next(iter(owners)) not in ambiguous}
        with stage_output(output) as out:
            pd.DataFrame(maps).to_csv(out / 'molecule_mapping.csv', index=False)
            legacy_contracts(root).to_csv(out / 'legacy_contract_issues.csv', index=False)
            counts = Counter()
            first_result = classify({})
            columns = ['molecule_id', 'smiles', 'split'] + RAW_COLUMNS + ['assay_parameters_json'] + list(first_result) + [
                'literature_unit_decision', 'literature_unit_policy', 'literature_unit_source']
            columns += ['evidence_override_decision', 'evidence_override_source', 'evidence_override_note']
            with (out / 'source_records.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                ids = sorted(source_map)
                for offset in tqdm(range(0, len(ids), 300), desc='回查测定与单位'):
                    batch = ids[offset:offset+300]
                    params = ','.join('?' for _ in batch)
                    types_sql = ','.join('?' for _ in types)
                    query = f'''SELECT a.activity_id,a.molregno,a.assay_id,a.doc_id,a.standard_type,a.standard_value,
                        a.standard_units,a.standard_relation,a.data_validity_comment,a.potential_duplicate,
                        s.assay_organism,s.assay_tax_id,s.assay_cell_type,s.assay_tissue,s.assay_subcellular_fraction,
                        s.description,s.chembl_id,m.chembl_id,d.doi,d.pubmed_id
                        FROM activities a JOIN assays s ON s.assay_id=a.assay_id
                        JOIN molecule_dictionary m ON m.molregno=a.molregno LEFT JOIN docs d ON d.doc_id=a.doc_id
                        WHERE a.molregno IN ({params}) AND a.standard_type IN ({types_sql})'''
                    rows = [dict(zip(RAW_COLUMNS, r)) for r in conn.execute(query, batch+types)]
                    assay_ids = sorted({r['assay_id'] for r in rows})
                    conditions = defaultdict(list)
                    for start in range(0, len(assay_ids), 500):
                        sub = assay_ids[start:start+500]
                        q = ','.join('?' for _ in sub)
                        for assay, typ, val, unit, txt in conn.execute(f'''SELECT assay_id,standard_type,standard_value,
                                standard_units,standard_text_value FROM assay_parameters WHERE assay_id IN ({q})''', sub):
                            conditions[assay].append(dict(type=typ, value=val, unit=unit, text=txt))
                    for r in rows:
                        molecule = source_map[r['molregno']]
                        decision = classify(r)
                        decision, literature = apply_papp_decision(r, decision, registry, papp_unresolved_policy)
                        decision, evidence = apply_human_evidence_decision(r, decision, human_decisions)
                        counts[(decision['task_id'], decision['status'], decision['reason'])] += 1
                        writer.writerow(dict(molecule_id=molecule['molecule_id'], smiles=molecule['smiles'], split=molecule['split'],
                                             **r, assay_parameters_json=json.dumps(conditions[r['assay_id']], ensure_ascii=False),
                                             **decision, **literature, **evidence))
            pd.DataFrame([dict(task_id=t, status=s, reason=r, records=n) for (t,s,r),n in counts.items()],
                         columns=['task_id', 'status', 'reason', 'records']).to_csv(out / 'task_audit_counts.csv', index=False)
            # 数据库很大：登记只读路径、大小、mtime；不将其误称为内容 SHA-256。
            stat = raw_db.stat()
            if (stat.st_size,stat.st_mtime_ns) != (initial_stat.st_size,initial_stat.st_mtime_ns):
                raise ValueError('审计期间源数据库发生变化，拒绝发布混合版本。')
            if read_fixed_splits(root)[1] != split_hashes:
                raise ValueError('审计期间固定 split 文件发生变化。')
            finish_stage(out, 'audit', fixed_split_hashes=split_hashes, partial=limit is not None,
                         rule_version=RULE_VERSION, source_database=dict(path=str(raw_db.resolve()), size=stat.st_size,
                         mtime_ns=stat.st_mtime_ns), mapped_molecules=len({r['molecule_id'] for r in source_map.values()}),
                         mapping_method='exact_standard_inchikey; ambiguous ownership excluded', records=sum(counts.values()),
                         papp_literature_overrides=(dict(path=str(Path(papp_decision_registry).resolve()),
                         sha256=sha256(papp_decision_registry), version=OVERRIDE_VERSION,
                         unresolved_policy=papp_unresolved_policy, decisions=decision_counts(registry)) if papp_decision_registry else None),
                         human_evidence_overrides=(dict(path=str(Path(human_evidence_registry).resolve()),
                         sha256=sha256(human_evidence_registry), version=HUMAN_EVIDENCE_OVERRIDE_VERSION,
                         decisions=len(human_decisions)) if human_evidence_registry else None))
    logging.info('审计完成: %s；候选测定 %d；未修改源数据库/旧 CSV。', output, sum(counts.values()))


def main():
    p = base_parser(__doc__)
    p.add_argument('--limit', type=int, help='仅试跑前 N 个分子；产物标记 partial，后续程序拒绝使用')
    p.add_argument('--papp-decision-registry', type=Path,
                   help='原文单位裁定 CSV；已确认微单位 DOI 选择性覆盖，unresolved 默认进入 review')
    p.add_argument('--papp-unresolved-policy', choices=['exclude', 'keep'], default='exclude')
    p.add_argument('--human-evidence-registry', type=Path,
                   help='按 activity_id 的人源 F/CLint/Thalf 原文裁定 CSV；仅精确匹配的记录可覆盖保守 review 规则')
    args = p.parse_args()
    configure_logging(args.root, 'audit_endpoint_sources')
    output = args.output or args.root / 'data/processed_v2/audit'
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit 必须为正数')
    read_fixed_splits(args.root)
    required = [args.root / 'data/raw/chembl_37/chembl_37_sqlite/chembl_37.db'] + ([args.papp_decision_registry] if args.papp_decision_registry else []) + ([args.human_evidence_registry] if args.human_evidence_registry else [])
    startup_self_check(required, output=output)
    if args.check_only:
        registry = load_papp_decisions(args.papp_decision_registry) if args.papp_decision_registry else {}
        human_decisions = load_human_evidence_decisions(args.human_evidence_registry) if args.human_evidence_registry else {}
        db = args.root / 'data/raw/chembl_37/chembl_37_sqlite/chembl_37.db'
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as conn:
            check_database(conn)
        logging.info('启动文件、数据库结构和原文裁定表自检通过；Papp DOI %d 个，人源 activity %d 个；未执行测定审计。', len(registry), len(human_decisions))
        return
    audit(args.root, output, args.limit, args.papp_decision_registry, args.papp_unresolved_policy,
          args.human_evidence_registry)


if __name__ == '__main__':
    run_cli(main)

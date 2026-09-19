"""第三步：保留既有成员关系，统一生成骨架分组/训练折并隔离跨集合冲突。"""
from __future__ import annotations
import logging
import inspect
import json
import sqlite3
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
from pathlib import Path
from pipeline_common import (base_parser, configure_logging, read_fixed_splits, verify_stage, verify_fixed_inputs,
                             startup_self_check, stage_output, finish_stage, stable_id, sha256, run_cli)


def structure_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'无效 SMILES: {smiles}')
    parent = rdMolStandardize.FragmentParent(mol)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    parent = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
    Chem.RemoveStereochemistry(parent)
    identity = Chem.MolToSmiles(parent, isomericSmiles=False)
    scaffold = MurckoScaffold.GetScaffoldForMol(parent)
    scaf = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    return stable_id(identity), 'ring:'+scaf if scaf else 'acyclic:'+identity


def cached_structure_groups(fixed, cache_path, retry_failed=False):
    """逐分子检查点独立于最终产物。化学转换失败只隔离，不换用宽松分组。"""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    version = stable_id('structure_groups_v1\n'+rdBase.rdkitVersion+'\n'+inspect.getsource(structure_groups))
    rows, reused = [], 0
    conn = sqlite3.connect(cache_path, timeout=60)
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS structure_cache (
            version TEXT NOT NULL, smiles TEXT NOT NULL, result TEXT NOT NULL,
            PRIMARY KEY(version, smiles))''')
        conn.commit()
        for position, smi in enumerate(tqdm(fixed.smiles, desc='标准化结构与骨架分组')):
            found = conn.execute('SELECT result FROM structure_cache WHERE version=? AND smiles=?', (version,smi)).fetchone()
            result = json.loads(found[0]) if found else None
            if result is not None and not (retry_failed and result['standardization_status']=='failed'):
                reused += 1
            else:
                try:
                    parent, scaffold = structure_groups(smi)
                    result = dict(parent_id=parent, scaffold_group=scaffold, standardization_status='ok', standardization_error='')
                except (RuntimeError, ValueError) as exc:
                    # 独立占位仅用于记录；eligible=False，绝不参与折分配或训练。
                    result = dict(parent_id='failed:'+stable_id(smi), scaffold_group='failed:'+stable_id(smi),
                                  standardization_status='failed', standardization_error=f'{type(exc).__name__}: {exc}')
                    logging.warning('结构规范化失败，隔离分子 index=%d smiles=%s error=%s',position,smi,exc)
                conn.execute('INSERT OR REPLACE INTO structure_cache VALUES (?,?,?)', (version,smi,json.dumps(result,ensure_ascii=False)))
            rows.append(result)
            if (position+1)%25==0:
                conn.commit()
    finally:
        # 普通异常或 Ctrl+C 也保存此前结果；硬终止最多损失最近 24 条。
        try:
            conn.commit()
        finally:
            conn.close()
    logging.info('结构检查点：复用 %d/%d 条，路径 %s',reused,len(rows),cache_path)
    return pd.DataFrame(rows,index=fixed.index), version


def build(root, datasets, output, n_folds, seed, cache_path=None, retry_failed=False):
    meta = verify_stage(datasets, 'datasets')
    verify_fixed_inputs(meta, root)
    fixed, hashes = read_fixed_splits(root)
    startup_self_check(output=output)
    if n_folds < 2:
        raise ValueError('folds 必须至少 2')
    cache_path = cache_path or Path(output).parent/'_cache/structure_groups.sqlite'
    groups, cache_version = cached_structure_groups(fixed,cache_path,retry_failed)
    fixed = pd.concat([fixed,groups],axis=1)
    # 同母体/骨架跨集合：全部标记排除，既不移动分子，也不根据标签挑选保留方。
    conflicts = set(fixed.groupby('scaffold_group').split.nunique().loc[lambda x: x>1].index)
    parent_conflicts = set(fixed.groupby('parent_id').split.nunique().loc[lambda x: x>1].index)
    fixed['eligible'] = fixed.standardization_status.eq('ok') & ~fixed.scaffold_group.isin(conflicts) & ~fixed.parent_id.isin(parent_conflicts)
    fixed['exclusion_reason'] = np.where(fixed.eligible, '', 'cross_split_parent_or_scaffold')
    fixed.loc[fixed.standardization_status.eq('failed'),'exclusion_reason'] = 'structure_standardization_failed'
    fixed['fold_id'] = -1
    train = fixed.loc[fixed.split.eq('train') & fixed.eligible]
    size = train.groupby('scaffold_group').size()
    if len(size) < n_folds:
        raise ValueError(f'独立训练骨架仅 {len(size)}，不足 {n_folds} 折')
    rng = np.random.default_rng(seed)
    order = pd.DataFrame({'group': size.index, 'n': size.values, 'tie': rng.random(len(size))}).sort_values(['n', 'tie'], ascending=[False, True])
    loads = np.zeros(n_folds, dtype=int)
    assignments = {}
    for r in order.itertuples(index=False):
        fold = int(np.argmin(loads))
        assignments[r.group] = fold
        loads[fold] += r.n
    fixed.loc[train.index, 'fold_id'] = train.scaffold_group.map(assignments).astype(int)
    tasks = pd.read_csv(datasets / 'task_records.csv')
    joined = tasks.merge(fixed[['molecule_id', 'eligible', 'fold_id', 'scaffold_group']], on='molecule_id', how='left', validate='many_to_one')
    if joined.eligible.isna().any():
        raise ValueError('任务分子不属于固定集合')
    with stage_output(output) as out:
        fixed.to_csv(out / 'split_manifest.csv', index=False)
        fixed.loc[~fixed.eligible].to_csv(out / 'excluded_structures.csv', index=False)
        fixed.loc[fixed.standardization_status.eq('failed')].to_csv(out / 'structure_failures.csv',index=False)
        joined.loc[joined.eligible].groupby(['task_id', 'split', 'fold_id']).agg(records=('row_id', 'size'), molecules=('molecule_id', 'nunique'),
            scaffolds=('scaffold_group', 'nunique')).reset_index().to_csv(out / 'task_fold_counts.csv', index=False)
        finish_stage(out, 'splits', inputs={'datasets_complete_sha256': sha256(datasets / 'complete.json')},
                     fixed_split_hashes=hashes, source_datasets_dir=str(datasets.resolve()), folds=n_folds, seed=seed,
                     exclusions=int((~fixed.eligible).sum()), partial=False,
                     standardization_failures=int(fixed.standardization_status.eq('failed').sum()),
                     structure_cache_version=cache_version, rdkit_version=rdBase.rdkitVersion,
                     grouping='RDKit FragmentParent + Uncharger + canonical tautomer + nonstereo Murcko; acyclic exact parent')
    logging.info('固定划分清单完成: %s；总排除 %d 个分子，其中规范化失败 %d 个。', output, (~fixed.eligible).sum(),fixed.standardization_status.eq('failed').sum())


def main():
    p = base_parser(__doc__)
    p.add_argument('--datasets-dir', type=Path)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--cache-path', type=Path, help='结构检查点 SQLite；默认 data/processed_v2/_cache/structure_groups.sqlite')
    p.add_argument('--retry-failed', action='store_true', help='重新计算已缓存的失败分子；成功结果仍复用')
    args = p.parse_args()
    configure_logging(args.root, 'build_split_manifest')
    data = args.datasets_dir or args.root / 'data/processed_v2/datasets'
    output = args.output or args.root / 'data/processed_v2/splits'
    meta = verify_stage(data, 'datasets')
    verify_fixed_inputs(meta, args.root)
    startup_self_check(output=output)
    if args.check_only:
        logging.info('数据版本和固定划分来源检查通过。')
        return
    build(args.root, data, output, args.folds, args.seed, args.cache_path, args.retry_failed)


if __name__ == '__main__':
    run_cli(main)

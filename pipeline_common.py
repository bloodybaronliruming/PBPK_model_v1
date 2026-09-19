"""公共启动自检、日志和不可覆盖的阶段产物协议。"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_id(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:24]


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def startup_self_check(required=(), output=None, columns=None):
    """所有入口共用；显式异常不会被 python -O 禁用。"""
    for p in required:
        if not Path(p).is_file():
            raise FileNotFoundError(f'缺少前序文件: {p}')
    if output is not None and Path(output).exists():
        raise FileExistsError(f'产物目录已存在，禁止覆盖: {output}；请使用新的 --output/--run-name。')
    if columns:
        import pandas as pd
        for p, names in columns.items():
            actual = set(pd.read_csv(p, nrows=0).columns)
            if not set(names) <= actual:
                raise ValueError(f'{p} 缺少列: {sorted(set(names)-actual)}')


def configure_logging(root, name):
    folder = Path(root) / 'results/pipeline_logs'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(folder / f'{name}_{stamp}.log', encoding='utf-8')],
                        force=True)


def run_cli(main):
    try:
        main()
    except (Exception, KeyboardInterrupt):
        logging.exception('执行失败；未发布完成标记。修复错误后使用新输出路径重试。')
        sys.exit(1)


@contextlib.contextmanager
def stage_output(output):
    """先写临时目录，成功后一次发布，失败不留下伪完成产物。"""
    output = Path(output)
    startup_self_check(output=output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{output.name}.partial_', dir=output.parent))
    try:
        yield temp
        if not (temp / 'complete.json').is_file():
            raise RuntimeError('阶段没有生成 complete.json')
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def finish_stage(folder, stage, inputs=None, **extra):
    folder = Path(folder)
    artifacts = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'complete.json'}
    code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    dump_json(folder / 'complete.json', dict(schema_version=SCHEMA_VERSION, stage=stage,
              created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs or {}, artifacts=artifacts,
              code_hashes=code, **extra))


def verify_stage(folder, expected_stage):
    folder = Path(folder)
    startup_self_check([folder / 'complete.json'])
    meta = json.loads((folder / 'complete.json').read_text())
    if meta.get('stage') != expected_stage or meta.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'阶段/版本错误: {folder}')
    if meta.get('partial', False):
        raise ValueError(f'只接受全量阶段，当前为 --limit 试跑产物: {folder}')
    for name, digest in meta['artifacts'].items():
        p = folder / name
        if not p.is_file() or sha256(p) != digest:
            raise ValueError(f'产物丢失或被修改: {p}')
    return meta


def base_parser(description):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--root', type=Path, default=ROOT, help='项目根目录；可使用独立测试项目')
    p.add_argument('--output', type=Path, help='新的输出目录，不覆盖既有产物')
    p.add_argument('--check-only', action='store_true', help='仅做前序文件和契约自检')
    return p


def read_fixed_splits(root):
    import pandas as pd
    files = [Path(root) / 'data/processed' / f'chembl_{s}.csv' for s in ['train', 'val', 'test']]
    startup_self_check(files, columns={p: ['Canonical_SMILES', 'Split'] for p in files})
    frames = []
    for split, p in zip(['train', 'val', 'test'], files):
        x = pd.read_csv(p, usecols=['Canonical_SMILES', 'Split'])
        if x.Canonical_SMILES.isna().any() or not x.Split.eq(split).all():
            raise ValueError(f'无效 SMILES 或 Split 列: {p}')
        frames.append(x)
    x = pd.concat(frames, ignore_index=True).rename(columns={'Canonical_SMILES': 'smiles', 'Split': 'split'})
    if x.smiles.duplicated().any():
        raise ValueError('固定集合内或集合间有重复 SMILES；禁止静默重新划分。')
    x['molecule_id'] = x.smiles.map(stable_id)
    return x, {str(p.resolve()): sha256(p) for p in files}


def verify_fixed_inputs(meta, root):
    _, current = read_fixed_splits(root)
    if meta['fixed_split_hashes'] != current:
        raise ValueError('现有固定 split 文件与前序审计不一致，请重新生成独立版本。')

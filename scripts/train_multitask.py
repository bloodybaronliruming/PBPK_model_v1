"""固定骨架、掩码损失的共享 MLP / dense MMoE 多任务训练。

任务定义为 endpoint × species × system。训练时按 molecule × task 聚合重复测定，
缺失值仅为张量占位且 mask=0；每个任务的目标空间在每个外层训练范围单独拟合。
默认排除容量少于 40 个训练分子的任务，避免极小任务以等任务权重主导共享表征。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm

from pipeline_common import (ROOT, base_parser, configure_logging, dump_json, finish_stage,
                             run_cli, sha256, stage_output, startup_self_check)
from dmpk_toolkit import TargetSpace, featurize, load_task, metrics, prediction_table


class Residual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return torch.relu(self.norm(x + self.layers(x)))


class MaskedMultiTaskNet(nn.Module):
    """共享编码器；MMoE 为每个任务提供独立、输入相关的专家门控和私有头。"""
    def __init__(self, n_features, n_tasks, architecture, width, experts, dropout):
        super().__init__()
        self.architecture = architecture
        self.encoder = nn.Sequential(nn.Linear(n_features, width), nn.ReLU(), Residual(width, dropout), nn.Dropout(dropout))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.ReLU(), Residual(width, dropout))
                                      for _ in range(experts)]) if architecture == 'mmoe' else None
        self.gates = nn.ModuleList([nn.Linear(width, experts) for _ in range(n_tasks)]) if architecture == 'mmoe' else None
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
                                    for _ in range(n_tasks)])

    def forward(self, x):
        h = self.encoder(x)
        expert_values = torch.stack([expert(h) for expert in self.experts], dim=1) if self.experts else None
        values = []
        for task, head in enumerate(self.heads):
            if expert_values is None:
                task_h = h
            else:
                weights = torch.softmax(self.gates[task](h), dim=1).unsqueeze(-1)
                task_h = (expert_values * weights).sum(dim=1)
            values.append(head(task_h).squeeze(1))
        return torch.stack(values, dim=1)


def task_ids(root, datasets, splits, minimum):
    registry = json.loads((datasets / 'task_registry.json').read_text())
    counts = pd.read_csv(splits / 'task_fold_counts.csv')
    train = counts.loc[(counts.split == 'train') & (counts.fold_id >= 0)]
    summary = train.groupby('task_id').agg(molecules=('molecules', 'sum'), folds=('fold_id', 'nunique'))
    return [task for task in sorted(registry) if task in summary.index
            and summary.loc[task, 'molecules'] >= minimum and summary.loc[task, 'folds'] >= 3]


def build_matrix(root, datasets, splits, tasks):
    frames, specs = {}, {}
    molecules = []
    for task in tasks:
        frame, spec, _ = load_task(root, datasets, splits, task)
        frames[task], specs[task] = frame, spec
        molecules.append(frame[['molecule_id', 'smiles', 'split', 'fold_id', 'scaffold_group']])
    molecule = pd.concat(molecules).drop_duplicates('molecule_id')
    if molecule.molecule_id.duplicated().any():
        raise ValueError('多任务分子 ID 不唯一')
    # 同分子跨任务必须继承完全相同的固定成员关系和折。
    for task, frame in frames.items():
        check = frame.merge(molecule[['molecule_id', 'smiles', 'split', 'fold_id']], on='molecule_id',
                            suffixes=('', '_matrix'), validate='many_to_one')
        if not check.smiles.eq(check.smiles_matrix).all() or not check.split.eq(check.split_matrix).all() or not check.fold_id.eq(check.fold_id_matrix).all():
            raise ValueError(f'{task} 与统一分子矩阵不一致')
    molecule = molecule.sort_values('molecule_id').reset_index(drop=True)
    lookup = {mid: i for i, mid in enumerate(molecule.molecule_id)}
    raw = np.full((len(molecule), len(tasks)), np.nan, dtype=float)
    mask = np.zeros_like(raw, dtype=bool)
    for col, task in enumerate(tasks):
        values = frames[task].groupby('molecule_id', as_index=False).raw_value.median()
        rows = np.fromiter((lookup[mid] for mid in values.molecule_id), dtype=int)
        raw[rows, col] = values.raw_value.to_numpy()
        mask[rows, col] = True
    return molecule, raw, mask, frames, specs


def fit_targets(raw, mask, indices, specs, tasks):
    targets = []
    y = np.zeros_like(raw, dtype=np.float32)
    for col, task in enumerate(tasks):
        take = indices[mask[indices, col]]
        if not len(take):
            raise ValueError(f'{task} 在当前训练范围无标签')
        target = TargetSpace(specs[task]['transform']).fit(raw[take, col], np.ones(len(take)))
        y[take, col] = target.forward(raw[take, col])
        targets.append(target)
    return targets, y


def epoch_batches(indices, mask, n_tasks, batch_size, sampler, rng):
    """生成一个 epoch 的 batch；task_balanced 只改变训练抽样，不触及划分或标签。"""
    if sampler == 'molecule_uniform':
        order = rng.permutation(indices)
        return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    if sampler != 'task_balanced':
        raise ValueError(f'未知 batch sampler: {sampler}')
    labelled = [indices[mask[indices, task]] for task in range(n_tasks)]
    missing = [task for task, rows in enumerate(labelled) if not len(rows)]
    if missing:
        raise ValueError(f'任务均衡采样发现训练范围内无标签任务: {missing}')
    n_batches = int(np.ceil(len(indices) / batch_size))
    batches = []
    # 每批为任务轮换抽取锚点；batch_size 足够时每个任务至少出现一次。
    for batch_number in range(n_batches):
        anchor_tasks = np.arange(n_tasks)
        rng.shuffle(anchor_tasks)
        if batch_size < n_tasks:
            offset = (batch_number * batch_size) % n_tasks
            anchor_tasks = np.roll(anchor_tasks, -offset)[:batch_size]
        chosen = []
        for task in anchor_tasks:
            candidates = labelled[int(task)]
            # 同一分子可合法承担多个任务；不做昂贵的跨任务去重，避免在大矩阵训练中
            # 让采样器的开销随任务数和样本数相乘。
            row = int(rng.choice(candidates))
            chosen.append(row)
        remaining = batch_size - len(chosen)
        if remaining:
            chosen.extend(rng.choice(indices, size=remaining, replace=len(indices) < remaining).astype(int).tolist())
        batches.append(np.asarray(chosen, dtype=int))
    return batches


def train_network(x, y, mask, indices, n_tasks, architecture, args, seed):
    torch.set_num_threads(args.threads)
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = MaskedMultiTaskNet(x.shape[1], n_tasks, architecture, args.width, args.experts, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in trange(args.epochs, desc=f'{architecture} epochs', leave=False):
        model.train()
        for batch in epoch_batches(indices, mask, n_tasks, args.batch_size, args.batch_sampler, rng):
            xx = torch.as_tensor(x[batch], dtype=torch.float32, device=args.device)
            yy = torch.as_tensor(y[batch], dtype=torch.float32, device=args.device)
            mm = torch.as_tensor(mask[batch], dtype=torch.bool, device=args.device)
            prediction = model(xx)
            task_losses = []
            for task in range(n_tasks):
                present = mm[:, task]
                if present.any():
                    task_losses.append(torch.nn.functional.huber_loss(prediction[present, task], yy[present, task], delta=1.))
            if not task_losses:
                continue
            loss = torch.stack(task_losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('多任务 masked loss 非有限')
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
    return model.cpu()


def raw_prediction(model, x, targets, specs, tasks):
    model.eval()
    with torch.no_grad():
        standardized = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    result = np.empty_like(standardized, dtype=float)
    for col, task in enumerate(tasks):
        result[:, col] = targets[col].inverse(standardized[:, col], specs[task]['endpoint'])
    return result


def report_predictions(folder, architecture, molecules, predictions, frames, specs, tasks, kind, train_groups):
    rows = []
    mapping = {mid: i for i, mid in enumerate(molecules.molecule_id)}
    summary = {}
    for col, task in enumerate(tasks):
        frame = frames[task]
        take = frame.split.eq(kind)
        if not take.any():
            summary[task] = None
            continue
        values = np.asarray([predictions[mapping[mid], col] for mid in frame.loc[take, 'molecule_id']])
        summary[task] = metrics(frame.loc[take], values, specs[task])
        table = prediction_table(frame.loc[take], values, specs[task], f'multitask_{kind}', train_groups)
        rows.append(table)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(folder / f'{kind}_predictions.csv', index=False)
    return summary


def run():
    p = base_parser('共享 MLP / dense MMoE；固定骨架 OOF，默认不使用 test 标签')
    p.add_argument('--datasets-dir', type=Path)
    p.add_argument('--splits-dir', type=Path)
    p.add_argument('--tasks', nargs='*', help='明确任务 ID；省略时按最小训练分子数选择')
    p.add_argument('--min-train-molecules', type=int, default=40)
    p.add_argument('--architectures', default='shared,mmoe', help='shared,mmoe 的逗号列表')
    p.add_argument('--width', type=int, default=192)
    p.add_argument('--experts', type=int, default=4)
    p.add_argument('--dropout', type=float, default=.15)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--batch-sampler', choices=['molecule_uniform', 'task_balanced'], default='molecule_uniform',
                   help='molecule_uniform 为原始基线；task_balanced 每批为任务提供均衡的有标签锚点')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--run-name', default='mmoe_diagnostic_v1')
    args = p.parse_args()
    configure_logging(args.root, 'train_multitask')
    if min(args.min_train_molecules, args.width, args.experts, args.epochs, args.batch_size, args.threads) < 1:
        raise ValueError('训练参数必须为正数')
    architectures = list(dict.fromkeys(args.architectures.split(',')))
    if not architectures or not set(architectures) <= {'shared', 'mmoe'}:
        raise ValueError('architectures 仅支持 shared,mmoe')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求 CUDA，但 PyTorch 没有可用 GPU')
    datasets = args.datasets_dir or args.root / 'data/processed_v2/datasets'
    splits = args.splits_dir or args.root / 'data/processed_v2/splits'
    output = args.output or args.root / 'models/multitask' / args.run_name
    startup_self_check(output=output)
    tasks = args.tasks or task_ids(args.root, datasets, splits, args.min_train_molecules)
    if len(tasks) < 2:
        raise ValueError('多任务训练至少需要两个合格任务')
    molecule, raw, mask, frames, specs = build_matrix(args.root, datasets, splits, tasks)
    train = np.flatnonzero(molecule.split.eq('train'))
    folds = sorted(molecule.iloc[train].fold_id.unique())
    if len(folds) < 3:
        raise ValueError('统一训练折不足 3')
    logging.info('任务=%d；分子=%d；训练分子=%d；folds=%s；架构=%s；batch_sampler=%s',
                 len(tasks), len(molecule), len(train), folds, architectures, args.batch_sampler)
    if args.check_only:
        logging.info('任务矩阵、raw/target/mask、统一划分、设备与输出路径检查通过；尚未训练。')
        return
    x, names = featurize(molecule.smiles.tolist())
    oof = {architecture: np.full(raw.shape, np.nan) for architecture in architectures}
    with threadpool_limits(limits=args.threads), stage_output(output) as out:
        for fold in tqdm(folds, desc='多任务受保护外层折'):
            fit = train[molecule.iloc[train].fold_id.to_numpy() != fold]
            valid = train[molecule.iloc[train].fold_id.to_numpy() == fold]
            targets, y = fit_targets(raw, mask, fit, specs, tasks)
            prep = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())])
            fit_x = prep.fit_transform(x[fit])
            valid_x = prep.transform(x[valid])
            x_fold = np.empty_like(x, dtype=np.float32); x_fold[fit] = fit_x; x_fold[valid] = valid_x
            for offset, architecture in enumerate(architectures):
                model = train_network(x_fold, y, mask, fit, len(tasks), architecture, args, args.seed + fold * 100 + offset)
                oof[architecture][valid] = raw_prediction(model, x_fold[valid], targets, specs, tasks)
        final_targets, final_y = fit_targets(raw, mask, train, specs, tasks)
        final_prep = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())])
        final_x = final_prep.fit_transform(x[train])
        all_x = final_prep.transform(x)
        final_train_groups = sorted(molecule.iloc[train].scaffold_group.unique())
        summaries = {}
        for offset, architecture in enumerate(architectures):
            folder = out / architecture; folder.mkdir()
            model = train_network(all_x, final_y, mask, train, len(tasks), architecture, args, args.seed + 10000 + offset)
            final_prediction = raw_prediction(model, all_x, final_targets, specs, tasks)
            summaries[architecture] = {'oof': report_predictions(folder, architecture, molecule, oof[architecture], frames, specs, tasks, 'train', final_train_groups),
                                       'val': report_predictions(folder, architecture, molecule, final_prediction, frames, specs, tasks, 'val', final_train_groups)}
            torch.save({'state_dict': model.state_dict(), 'architecture': architecture, 'width': args.width,
                        'experts': args.experts, 'dropout': args.dropout, 'tasks': tasks, 'feature_names': names,
                        'training_config': {'learning_rate': args.learning_rate, 'epochs': args.epochs,
                                            'batch_size': args.batch_size, 'batch_sampler': args.batch_sampler,
                                            'threads': args.threads, 'device': args.device, 'seed': args.seed}}, folder / 'model.pt')
            joblib.dump({'preprocessor': final_prep, 'targets': final_targets, 'task_specs': specs}, folder / 'preprocessing.joblib')
        dump_json(out / 'metrics.json', summaries)
        dump_json(out / 'task_registry.json', {task: specs[task] for task in tasks})
        dump_json(out / 'selection.json', {'criterion': 'per-task protected OOF compared with STL',
                  'batch_sampler': args.batch_sampler,
                  'caveat': 'no single global winner across differently scaled tasks; test remains frozen'})
        dump_json(out / 'training_config.json', {'min_train_molecules': args.min_train_molecules,
                  'width': args.width, 'experts': args.experts, 'dropout': args.dropout,
                  'learning_rate': args.learning_rate, 'epochs': args.epochs,
                  'batch_size': args.batch_size, 'batch_sampler': args.batch_sampler,
                  'threads': args.threads, 'device': args.device, 'seed': args.seed})
        finish_stage(out, 'multitask_training', inputs={'datasets_complete_sha256': sha256(datasets / 'complete.json'),
                     'splits_complete_sha256': sha256(splits / 'complete.json')}, tasks=tasks, architectures=architectures,
                     test_evaluated=False, partial=False, seed=args.seed, batch_sampler=args.batch_sampler,
                     training_config={'min_train_molecules': args.min_train_molecules, 'width': args.width,
                     'experts': args.experts, 'dropout': args.dropout, 'learning_rate': args.learning_rate,
                     'epochs': args.epochs, 'batch_size': args.batch_size, 'threads': args.threads,
                     'device': args.device})
    logging.info('多任务训练产物: %s；test 评估=False', output)


if __name__ == '__main__':
    run_cli(run)

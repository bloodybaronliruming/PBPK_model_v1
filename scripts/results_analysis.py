"""生成冻结验证集上的英文性能图与表；不读取或报告 test 标签。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (ROOT, base_parser, configure_logging, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)


def selected_run(path: Path):
    meta = verify_stage(path, 'stl_training')
    if meta.get('test_evaluated'):
        raise ValueError(f'只能生成未解冻 test 的验证报告: {path}')
    selected = json.loads((path / 'selection.json').read_text())['algorithm']
    metrics = json.loads((path / 'metrics.json').read_text())[selected]
    task = meta['task_id']
    for scope in ('oof', 'val'):
        if scope not in metrics:
            raise ValueError(f'{path} 缺少 {scope} 指标')
    return task, selected, metrics, meta


def display_name(task: str):
    endpoint = task.split('__')[0]
    return {'CL': 'Rat CL', 'VDss': 'Rat VDss', 'fu': 'Rat plasma fu',
            'F': 'Rat absolute oral F', 'Thalf': 'Rat terminal IV half-life'}.get(endpoint, task)


def panel(ax, data, result, title, endpoint):
    if endpoint == 'F':
        x = data.observed_physical.to_numpy()
        y = data.predicted_physical.to_numpy()
        lo, hi, delta = -.03, 1.03, .10
        xlabel, ylabel, band_label = 'Observed fraction', 'Predicted fraction', '±0.10'
    else:
        x = np.log10(data.observed_physical.to_numpy())
        y = np.log10(data.predicted_physical.to_numpy())
        lo, hi = np.nanmin(np.r_[x, y]), np.nanmax(np.r_[x, y])
        margin = max(.08, .04 * (hi - lo))
        lo, hi = lo - margin, hi + margin
        delta = np.log10(2)
        xlabel, ylabel, band_label = 'Observed log10 value', 'Predicted log10 value', 'Two-fold'
    ax.scatter(x, y, s=16, color='#1874A5', alpha=.52, linewidths=0, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color='#202020', linewidth=1.2, label='Identity')
    ax.plot([lo, hi], [lo + delta, hi + delta], color='#8C8C8C', linestyle='--', linewidth=.9, label='Two-fold')
    ax.plot([lo, hi], [lo - delta, hi - delta], color='#8C8C8C', linestyle='--', linewidth=.9)
    ax.lines[-2].set_label(band_label)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal', title=title, xlabel=xlabel, ylabel=ylabel)
    primary_name = 'MAE' if endpoint in {'fu', 'F'} else 'RMSE'
    coverage = (f"Within ±0.10 = {result['absolute_within_0.10']:.1%}"
                if endpoint in {'fu', 'F'} else f"Within 2-fold = {result['within_2fold_positive_subset']:.1%}")
    text = (f"n = {result['molecules']} molecules\n"
            f"{primary_name} = {result['primary']:.3f}\n"
            f"GMFE = {result['gmfe_positive_subset']:.2f}\n"
            f"{coverage}")
    ax.text(.04, .96, text, transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#C9C9C9', 'alpha': .94})
    ax.spines[['top', 'right']].set_visible(False)


def run():
    p = base_parser('验证集英文图表；输入模型必须保持 test 未评估')
    p.add_argument('--run-dir', type=Path, action='append', required=True,
                   help='可重复提供已完成 STL 运行目录')
    args = p.parse_args()
    configure_logging(args.root, 'results_analysis')
    output = args.output or args.root / 'results/analysis/rat_priors_v1'
    startup_self_check(output=output)
    entries = []
    for path in args.run_dir:
        task, algorithm, metrics, meta = selected_run(path)
        entries.append((path, task, algorithm, metrics, meta))
    if not 1 <= len(entries) <= 2:
        raise ValueError('当前出版图版接受一个任务，或一对 CL/VDss 任务')
    endpoints = {task.split('__')[0] for _, task, *_ in entries}
    valid_pairs = ({'CL', 'VDss'}, {'F', 'Thalf'})
    if len(entries) == 2 and endpoints not in valid_pairs:
        raise ValueError('两个任务的出版图版仅接受 CL/VDss 或 F/Thalf 组合')
    entries.sort(key=lambda item: item[1])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'bold',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    rows = []
    with stage_output(output) as out:
        fig, axes = plt.subplots(len(entries), 2, figsize=(10.4, 4.95 * len(entries)),
                                 constrained_layout=True, squeeze=False)
        for row, (path, task, algorithm, metrics, _) in enumerate(entries):
            endpoint = task.split('__')[0]
            for col, scope in enumerate(('oof', 'val')):
                data = pd.read_csv(path / algorithm / f'{scope}_predictions.csv')
                panel(axes[row, col], data, metrics[scope], f"{display_name(task)} — {'Cross-validated OOF' if scope == 'oof' else 'Frozen validation'}", endpoint)
            primary_name = 'MAE' if endpoint in {'fu', 'F'} else 'log10 RMSE'
            rows.append({'Task': task, 'Selected model': algorithm.upper(), 'Primary metric': primary_name,
                         'OOF molecules': metrics['oof']['molecules'], 'OOF score': metrics['oof']['primary'],
                         'OOF GMFE': metrics['oof']['gmfe_positive_subset'], 'OOF within 2-fold': metrics['oof']['within_2fold_positive_subset'],
                         'Validation molecules': metrics['val']['molecules'], 'Validation score': metrics['val']['primary'],
                         'Validation GMFE': metrics['val']['gmfe_positive_subset'], 'Validation within 2-fold': metrics['val']['within_2fold_positive_subset']})
        axes[0, 0].legend(loc='lower right', frameon=False, fontsize=8)
        title = 'Rat Pharmacokinetic Prior Models: Structure-Disjoint Evaluation'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        if len(entries) == 2:
            stem = 'rat_cl_vdss_performance' if endpoints == {'CL', 'VDss'} else 'rat_f_thalf_performance'
        else:
            stem = f"{entries[0][1].replace('__', '_')}_performance"
        fig.savefig(out / f'{stem}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        table = pd.DataFrame(rows)
        table.to_csv(out / f'{stem}.csv', index=False, float_format='%.4f')
        shown = table.copy()
        for col in shown.columns:
            if 'within 2-fold' in col:
                shown[col] = shown[col].map(lambda value: f'{value:.1%}')
            elif 'RMSE' in col or 'GMFE' in col:
                shown[col] = shown[col].map(lambda value: f'{value:.3f}')
        fig, ax = plt.subplots(figsize=(15.5, 1.75 + .7 * len(entries)))
        ax.axis('off')
        artist = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc='center', loc='center')
        artist.auto_set_font_size(False)
        artist.set_fontsize(8)
        artist.scale(1, 1.7)
        for (r, c), cell in artist.get_celld().items():
            cell.set_edgecolor('#D0D0D0')
            if r == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#145A7A')
            elif r % 2:
                cell.set_facecolor('#EDF4F7')
        ax.set_title('Table 1. Structure-disjoint performance of selected rat PK prior models', fontweight='bold', pad=12)
        fig.savefig(out / f'{stem}_table.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        finish_stage(out, 'results_analysis', inputs={str(path): sha256(path / 'complete.json') for path, *_ in entries},
                     test_evaluated=False, figure_language='English', dpi=600, partial=False)


if __name__ == '__main__':
    run_cli(run)

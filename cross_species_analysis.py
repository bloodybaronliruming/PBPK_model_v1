"""生成受保护大鼠→人跨物种级联的英文性能图及直接模型对照表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (ROOT, base_parser, configure_logging, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)
from results_analysis import panel

TASKS = ('fu__human__plasma', 'CL__human__systemic_iv', 'VDss__human__steady_state_iv',
         'Thalf__human__terminal_iv', 'F__human__absolute_oral')
BASELINES = {
    'fu__human__plasma': ROOT / 'models/stl/fu__human__plasma/baseline_v1',
    'CL__human__systemic_iv': ROOT / 'models/stl/CL__human__systemic_iv/baseline_v1',
    'VDss__human__steady_state_iv': ROOT / 'models/stl/VDss__human__steady_state_iv/baseline_v1',
    'Thalf__human__terminal_iv': ROOT / 'models/stl/Thalf__human__terminal_iv/diagnostic_v1',
}


def model_result(path, stage):
    meta = verify_stage(path, stage)
    if meta.get('test_evaluated'):
        raise ValueError(f'分析仅接受 test 未评估的运行: {path}')
    selected = json.loads((path / 'selection.json').read_text())['algorithm']
    return selected, json.loads((path / 'metrics.json').read_text())[selected], meta


def score_name(task):
    return 'MAE' if task.split('__')[0] in {'fu', 'F'} else 'log10 RMSE'


def human_display_name(task):
    endpoint = task.split('__')[0]
    return {'fu': 'Human plasma fu', 'CL': 'Human systemic CL', 'VDss': 'Human VDss',
            'Thalf': 'Human terminal IV half-life', 'F': 'Human absolute oral F'}[endpoint]


def format_score(value):
    return '—' if value is None else f'{value:.4f}'


def run():
    p = base_parser('跨物种级联英文图表；所有输入必须保持 test 冻结')
    p.add_argument('--run-dir', type=Path, action='append', help='可选；默认读取五项标准跨物种运行')
    args = p.parse_args()
    configure_logging(args.root, 'cross_species_analysis')
    output = args.output or args.root / 'results/analysis/cross_species_human_v1'
    startup_self_check(output=output)
    paths = args.run_dir or [args.root / 'models/cascade/cross_species' / task / 'cross_species_diagnostic_v1'
                             for task in TASKS]
    entries = []
    for path in paths:
        algorithm, result, meta = model_result(path, 'protected_cross_species_cascade')
        if meta['task_id'] not in TASKS:
            raise ValueError(f'不支持的跨物种任务: {meta["task_id"]}')
        entries.append((meta['task_id'], path, algorithm, result, meta))
    if {item[0] for item in entries} != set(TASKS):
        raise ValueError(f'需要且仅需要五项任务: {TASKS}')
    entries.sort(key=lambda item: TASKS.index(item[0]))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titleweight': 'bold',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    rows = []
    with stage_output(output) as out:
        fig, axes = plt.subplots(len(entries), 2, figsize=(10.6, 4.8 * len(entries)),
                                 constrained_layout=True, squeeze=False)
        inputs = {}
        for row, (task, path, algorithm, result, _) in enumerate(entries):
            endpoint = task.split('__')[0]
            inputs[str(path)] = sha256(path / 'complete.json')
            for col, scope in enumerate(('oof', 'val')):
                ax = axes[row, col]
                prediction = path / algorithm / f'{scope}_predictions.csv'
                if result.get(scope) is None or not prediction.exists():
                    ax.axis('off')
                    ax.text(.5, .58, 'No eligible validation molecules', ha='center', va='center',
                            fontsize=13, fontweight='bold')
                    ax.text(.5, .43, 'OOF diagnostic only; test remains frozen', ha='center', va='center', fontsize=10)
                    ax.set_title(f'{human_display_name(task)} — Frozen validation')
                else:
                    panel(ax, pd.read_csv(prediction), result[scope],
                          f"{human_display_name(task)} — {'Cross-validated OOF' if scope == 'oof' else 'Frozen validation'}", endpoint)
            baseline = BASELINES.get(task)
            base_oof = base_val = None
            baseline_model = '—'
            if baseline and baseline.exists():
                baseline_model, baseline_metrics, baseline_meta = model_result(baseline, 'stl_training')
                inputs[str(baseline)] = sha256(baseline / 'complete.json')
                base_oof = baseline_metrics['oof']['primary']
                base_val = baseline_metrics['val']['primary'] if baseline_metrics.get('val') else None
            cross_oof = result['oof']['primary']
            cross_val = result['val']['primary'] if result.get('val') else None
            rows.append({'Task': task, 'Primary metric': score_name(task),
                         'Direct selected model': baseline_model.upper() if baseline_model != '—' else '—',
                         'Cross-species selected model': algorithm.upper(),
                         'Direct OOF score': base_oof, 'Cross-species OOF score': cross_oof,
                         'OOF change (cross − direct)': None if base_oof is None else cross_oof - base_oof,
                         'Direct validation score': base_val, 'Cross-species validation score': cross_val,
                         'Validation change (cross − direct)': None if base_val is None or cross_val is None else cross_val - base_val})
        axes[0, 0].legend(loc='lower right', frameon=False, fontsize=8)
        fig.suptitle('Protected Rat-to-Human Cross-Species Prior Models', fontsize=15, fontweight='bold')
        fig.savefig(out / 'cross_species_performance.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        table = pd.DataFrame(rows)
        table.to_csv(out / 'cross_species_comparison.csv', index=False, float_format='%.4f')
        shown = table.copy()
        for col in shown.columns:
            if 'score' in col.lower() or 'change' in col.lower():
                shown[col] = shown[col].map(format_score)
        fig, ax = plt.subplots(figsize=(17.5, 3.7))
        ax.axis('off')
        artist = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc='center', loc='center')
        artist.auto_set_font_size(False)
        artist.set_fontsize(7.4)
        artist.scale(1, 1.65)
        for (r, _), cell in artist.get_celld().items():
            cell.set_edgecolor('#D0D0D0')
            if r == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#145A7A')
            elif r % 2:
                cell.set_facecolor('#EDF4F7')
        ax.set_title('Table 1. Direct versus protected cross-species models on structure-disjoint evaluation',
                     fontweight='bold', pad=12)
        fig.savefig(out / 'cross_species_comparison_table.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        finish_stage(out, 'cross_species_analysis', inputs=inputs, test_evaluated=False,
                     figure_language='English', dpi=600, partial=False)


if __name__ == '__main__':
    run_cli(run)

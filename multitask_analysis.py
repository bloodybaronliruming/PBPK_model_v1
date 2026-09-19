"""比较人源 STL、共享 MLP 与 MMoE 的 OOF/冻结验证表现，生成英文图表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (ROOT, base_parser, configure_logging, finish_stage, run_cli,
                             sha256, stage_output, startup_self_check, verify_stage)

HUMAN_STL = {
    'fu__human__plasma': ROOT / 'models/stl/fu__human__plasma/baseline_v1',
    'CL__human__systemic_iv': ROOT / 'models/stl/CL__human__systemic_iv/baseline_v1',
    'VDss__human__steady_state_iv': ROOT / 'models/stl/VDss__human__steady_state_iv/baseline_v1',
    'Thalf__human__terminal_iv': ROOT / 'models/stl/Thalf__human__terminal_iv/diagnostic_v1',
    'Papp__human__caco2_ab': ROOT / 'models/stl/Papp__human__caco2_ab/baseline_v1',
}


def selected(path, stage):
    meta = verify_stage(path, stage)
    if meta.get('test_evaluated'):
        raise ValueError(f'只接受 test 未评估的运行: {path}')
    algorithm = json.loads((path / 'selection.json').read_text()).get('algorithm')
    data = json.loads((path / 'metrics.json').read_text())
    return algorithm, data[algorithm] if stage == 'stl_training' else data, meta


def metric(task):
    return 'MAE' if task.split('__')[0] == 'fu' else 'log10 RMSE'


def run():
    p = base_parser('人源 STL、共享 MLP 与 MMoE 的英文比较图表')
    p.add_argument('--shared-run', type=Path, default=ROOT / 'models/multitask/shared_diagnostic_v1')
    p.add_argument('--mmoe-run', type=Path, default=ROOT / 'models/multitask/mmoe_diagnostic_v1')
    args = p.parse_args()
    configure_logging(args.root, 'multitask_analysis')
    output = args.output or args.root / 'results/analysis/multitask_human_v1'
    startup_self_check(output=output)
    _, shared, shared_meta = selected(args.shared_run, 'multitask_training')
    _, mmoe, mmoe_meta = selected(args.mmoe_run, 'multitask_training')
    if 'shared' not in shared or 'mmoe' not in mmoe:
        raise ValueError('运行目录必须分别包含 shared 和 mmoe 架构')
    rows, inputs = [], {str(args.shared_run): sha256(args.shared_run / 'complete.json'),
                        str(args.mmoe_run): sha256(args.mmoe_run / 'complete.json')}
    for task, path in HUMAN_STL.items():
        algorithm, direct, _ = selected(path, 'stl_training')
        inputs[str(path)] = sha256(path / 'complete.json')
        if task not in shared['shared']['oof'] or task not in mmoe['mmoe']['oof']:
            continue
        row = {'Task': task, 'Primary metric': metric(task), 'STL model': algorithm.upper()}
        row['STL OOF'] = direct['oof']['primary']
        row['STL validation'] = direct['val']['primary']
        for label, values in [('Shared MLP', shared['shared']), ('MMoE', mmoe['mmoe'])]:
            row[f'{label} OOF'] = values['oof'][task]['primary']
            row[f'{label} validation'] = values['val'][task]['primary']
        row['Shared OOF change (%)'] = 100 * (row['Shared MLP OOF'] - row['STL OOF']) / row['STL OOF']
        row['MMoE OOF change (%)'] = 100 * (row['MMoE OOF'] - row['STL OOF']) / row['STL OOF']
        row['Shared validation change (%)'] = 100 * (row['Shared MLP validation'] - row['STL validation']) / row['STL validation']
        row['MMoE validation change (%)'] = 100 * (row['MMoE validation'] - row['STL validation']) / row['STL validation']
        viable = []
        if row['Shared OOF change (%)'] <= 0 and row['Shared validation change (%)'] <= 0:
            viable.append(('Shared candidate', row['Shared MLP OOF']))
        if row['MMoE OOF change (%)'] <= 0 and row['MMoE validation change (%)'] <= 0:
            viable.append(('MMoE candidate', row['MMoE OOF']))
        row['Candidate decision'] = min(viable, key=lambda item: item[1])[0] if viable else 'Keep STL'
        rows.append(row)
    table = pd.DataFrame(rows)
    with stage_output(output) as out:
        table.to_csv(out / 'human_stl_mtl_comparison.csv', index=False, float_format='%.4f')
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
        labels = [task.replace('__human__', '\n') for task in table.Task]
        pos = np.arange(len(table)); width = .34
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True, sharey=True)
        for ax, scope in zip(axes, ('OOF', 'validation')):
            shared_col = f'Shared {scope} change (%)'
            mmoe_col = f'MMoE {scope} change (%)'
            ax.axhline(0, color='#202020', linewidth=1)
            ax.bar(pos - width / 2, table[shared_col], width, label='Shared MLP', color='#4C78A8')
            ax.bar(pos + width / 2, table[mmoe_col], width, label='MMoE', color='#E45756')
            ax.set_xticks(pos, labels, rotation=25, ha='right')
            ax.set_title(f'{scope}: relative change from STL')
            ax.set_ylabel('Change in primary score (%)\nnegative = improvement')
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(frameon=False)
        fig.suptitle('Human Tasks: Shared MLP and MMoE versus Single-Task Baselines', fontweight='bold', fontsize=14)
        fig.savefig(out / 'human_stl_mtl_relative_change.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        shown = table.copy()
        for col in shown.columns:
            if 'change' in col.lower() or col.endswith('OOF') or col.endswith('validation'):
                shown[col] = shown[col].map(lambda x: f'{x:.3f}' if isinstance(x, float) else x)
        fig, ax = plt.subplots(figsize=(18, 3.6)); ax.axis('off')
        artist = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc='center', loc='center')
        artist.auto_set_font_size(False); artist.set_fontsize(6.8); artist.scale(1, 1.65)
        for (r, _), cell in artist.get_celld().items():
            cell.set_edgecolor('#D0D0D0')
            if r == 0:
                cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#145A7A')
            elif r % 2:
                cell.set_facecolor('#EDF4F7')
        ax.set_title('Table 1. Human-task comparison of STL and masked MTL candidates', fontweight='bold', pad=12)
        fig.savefig(out / 'human_stl_mtl_comparison_table.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        finish_stage(out, 'multitask_analysis', inputs=inputs, test_evaluated=False, figure_language='English', dpi=600, partial=False)


if __name__ == '__main__':
    run_cli(run)

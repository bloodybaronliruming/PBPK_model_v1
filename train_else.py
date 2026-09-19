"""动物或非默认任务的单任务先验训练入口。

必须显式提供 endpoint、species 与 system，避免把不同物种或实验体系
混入一个模型。其余训练参数完全沿用 dmpk_toolkit 的固定骨架折流程。
"""
from __future__ import annotations

import argparse
import sys

from pipeline_common import run_cli
from dmpk_toolkit import train_endpoint


ENDPOINTS = ('fu', 'CL', 'CLint', 'VDss', 'F', 'Thalf', 'Papp')


def run():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--endpoint', required=True, choices=ENDPOINTS)
    selector.add_argument('--species', required=True, choices=('human', 'rat', 'mouse', 'dog', 'monkey'))
    selector.add_argument('--system', required=True)
    selected, remaining = selector.parse_known_args()
    # train_endpoint 是统一的参数、自检和不可覆盖产物入口。仅剥离本入口
    # 的任务选择参数，其他参数原样交给它解析。
    sys.argv = [sys.argv[0], *remaining, '--species', selected.species, '--system', selected.system]
    train_endpoint(selected.endpoint, selected.system)


if __name__ == '__main__':
    run_cli(run)

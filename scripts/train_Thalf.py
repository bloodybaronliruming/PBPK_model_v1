"""第九步：人终末静脉消除半衰期 Thalf 的直接诊断基线。

该入口不把 CL 或 VDss 的全训练预测直接拼入训练特征，避免产生
跨端点的同分子信息泄露。级联版将在有受保护外层预测时另行实现。
"""
from pipeline_common import run_cli
from dmpk_toolkit import train_endpoint


if __name__ == '__main__':
    run_cli(lambda: train_endpoint('Thalf', 'terminal_iv'))

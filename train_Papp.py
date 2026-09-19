"""第六步：经审计的人源 Caco-2 A→B Papp 单任务。"""
from pipeline_common import run_cli
from dmpk_toolkit import train_endpoint

if __name__ == '__main__':
    run_cli(lambda: train_endpoint('Papp', 'caco2_ab'))

"""口服绝对生物利用度 F 单任务训练；默认人源，物种需显式选择。"""
from pipeline_common import run_cli
from dmpk_toolkit import train_endpoint


if __name__ == '__main__':
    run_cli(lambda: train_endpoint('F', 'absolute_oral'))

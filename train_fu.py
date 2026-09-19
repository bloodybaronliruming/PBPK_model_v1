"""第五步：fu 单任务；默认人血浆，其他物种需显式 --species。"""
from pipeline_common import run_cli
from dmpk_toolkit import train_endpoint

if __name__ == '__main__':
    run_cli(lambda: train_endpoint('fu', 'plasma'))

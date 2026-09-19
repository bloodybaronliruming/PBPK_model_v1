"""第八步：人稳态静脉 VDss 单任务直接基线。"""
from pipeline_common import run_cli
from dmpk_toolkit import train_endpoint


if __name__ == '__main__':
    run_cli(lambda: train_endpoint('VDss', 'steady_state_iv'))

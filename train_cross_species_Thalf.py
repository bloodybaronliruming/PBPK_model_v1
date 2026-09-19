"""人 Thalf 的受保护大鼠先验级联入口。"""
import sys
from pipeline_common import run_cli
from train_cross_species import run

if __name__ == '__main__':
    sys.argv.insert(1, '--endpoint=Thalf')
    run_cli(run)

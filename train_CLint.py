"""第六步：人肝微粒体 apparent CLint；unbound 任务显式 --system microsome_unbound。"""
from pipeline_common import run_cli
from dmpk_toolkit import train_endpoint

if __name__ == '__main__':
    run_cli(lambda: train_endpoint('CLint', 'microsome'))

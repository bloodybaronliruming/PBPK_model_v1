# Model V1 archive

第一代模型代码与结果归档：

```text
model_v1/scripts/   V1 构建、训练、评价、发布与历史工具
model_v1/results/   V1 smoke、benchmark、analysis、final 与日志产物
```

根目录 `results` 是指向 `model_v1/results/` 的兼容链接。根目录 `scripts/` 是兼容层：普通历史入口链接到 `model_v1/scripts/`；依赖真实路径语义的 `pipeline_common.py`、`package_research_process_archive.py` 和 `evaluate_jia2025_author_like_r3b.py` 则在旧位置保留字节不变的实体，归档目录反向链接它们。这样旧脚本计算出的项目 `ROOT`、协议绑定的 scorer 路径、旧命令和已登记代码哈希保持不变。V1 保持冻结；新的 V2 入口与结果分别写入 `model_v2/scripts/` 和 `model_v2/results/`。

2026-09-19 搬迁后验收：使用 `oneadmet` 环境从旧路径运行 `python -m unittest discover -s scripts/tests -p 'test_*.py' -v`，118 项测试全部通过；`processed_v15` 的 audit/datasets/splits 与 Gate 7 release-readiness 阶段通过 `verify_stage()`；兼容目录无断链。环境未提供独立 `pytest` 可执行文件，未为此安装或升级依赖。

# 本轮程序验证记录

日期：2026-09-10。

- `python -m unittest discover -s scripts/tests -v`：20 项通过，最后一次完整测试耗时约 9.9 秒。
- 新增 Papp 原文单位规则测试：对 DOI 特异性 `cm/s → 10^-6 cm/s` 覆盖、ChEMBL `10'-6 cm/s` 拼写变体和 unresolved 默认排除进行验证。
- 真实 ChEMBL 的 11 个已裁定 DOI 只读演练匹配 66 条 Papp 记录：27 条微单位记录 accepted、14 条因原有语义条件保留 review；未解析 DOI 的 3 条记录均进入 review。演练不写入正式数据。
- 合成测试夹具包含 15 个分子、3 个端点、45 条测定；其 SQLite、模型和预测均在独立临时目录中生成并清理，不进入正式数据。
- 覆盖：微粒体/体重单位分离、g/mg 同材料换算、PPB 删失关系反转、低 fu 保真、结构分组、完整产物检查、固定折嵌套调参、保护外层范围、3 个端点入口、模型重载、普通推理和冻结测试评估。
- 实际拟合并重载的算法：常数基线、Ridge、RF、ExtraTrees、XGBoost、LightGBM、CatBoost、Res-MLP、D-MPNN。神经测试使用 CPU 和 1 epoch，仅验证工程功能。
- 7 个主入口的命令帮助和 Python 语法检查通过。
- 真实 ChEMBL 只读试跑：前 24 个固定训练分子，52 条候选测定。该产物位于 `/tmp/mtl_real_source_smoke_20260909`，标记 partial，后续重建明确拒收。
- 最后修改后的真实数据库字段自检通过；完整审计流程在合成数据库中复测通过。
- 原 `data/processed` 的 16 份已登记分子 CSV 的 SHA-256 未改变。
- 没有执行真实数据的全量审计/重建/骨架重算，也没有创建正式 `data/processed_v2` 或正式模型。

验证环境：Python 3.11、RDKit 2023.09.6、NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.9.0、SciPy 1.17.1、XGBoost 3.2.0、LightGBM 4.7.0、CatBoost 1.2.10、PyTorch 2.5.1+cu121。

未验证事项：全量真实数据的最终任务容量、模型科研性能、完整实际训练耗时、CUDA 训练与显存峰值。当前保守审计规则可能将大量语义不明记录留在 review；需要依据原始记录完善具体规则，不能据此宣称全部标签问题已解决。

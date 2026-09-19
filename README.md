# 数据审计、底座与微观端点训练：运行说明

所有入口放在 `scripts`。这批程序实现：来源审计 → 新数据视图 → 固定骨架折 → 单任务与受保护级联 → 共享 MLP / MMoE 多任务训练。融合/PBPK 程序属于后续批次，本次没有以空壳文件代替它们。

程序供人工按顺序运行，不会自动启动下一阶段。旧 `data/processed` 和原始 SQLite 只读；新版本默认写入 `data/processed_v2`。输出目录已经存在时会立即报错，不覆盖模型或数据。若需重跑，请用新的 `--output` 路径，并在后续步骤用 `--audit-dir`、`--datasets-dir`、`--splits-dir` 指向对应版本。训练重跑使用新的 `--run-name`。

## Jia 2025 N0 只读审计

`audit_jia2025_n0.py` 对本地 Jia 2025 正文、SI PDF 与 workbook 做不训练的可复现性审计。它仅发布源哈希、字段覆盖、哈希化 author split 成员、parent/scaffold 重叠及 106-compound 表的身份关系；不导出标签、不拟合模型、不产生预测指标，也不读取本项目已消费测试集。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_jia2025_n0.py
```

默认产物为 `results/analysis/jia2025_n0_audit_v1`；已发布时禁止覆盖。其结果将 Jia fu/CL/VDss 定为 B 级 author-like 复现轨，后续必须先冻结单独 protocol，且 author-test 只可一次性读取。

`freeze_jia2025_author_like_protocol.py` 将 N0 的 B 级结论落实为不可覆盖的 author-like baseline protocol：提取 Jia SI 的 15 个固定超参数单元，锁定 54 个预注册配置、cohort/重复规则、train-only 特征规则、指标和 author-test 生命周期；不拟合模型或评分。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_jia2025_author_like_protocol.py
```

默认产物是 `data/public_development/jia2025_author_like_protocol_v1`。后续必须先通过 train-only feature/pipeline smoke，才可进行一次 author-test 描述性评分。

`smoke_jia2025_author_train_only.py` 对每端点 64 条确定性 author-train 成员完成 10-fold 工程 smoke，覆盖全部 54 个冻结配置和所有特征家族。Mordred/merged 的缺失、方差和相关性裁剪严格折内拟合；树数临时限制为 8，只用于接口验证。v2 是有效放行版本：

```bash
conda run --no-capture-output -n oneadmet python scripts/smoke_jia2025_author_train_only.py
```

它不导出标签、预测或性能，且不选择或使用 author-test 行；完整工作簿的技术载入状态会被如实登记。全量 R3a 应使用新的独立 runner 与输出路径。

`train_jia2025_author_like_r3a.py` 是 R3a 正式 CPU runner：对 54 个冻结配置运行完整 author-train 10-fold CV，并将全训练模型冻结到 `models/jia2025_author_like_r3a_v1`；它明确禁止 author-test 预测/指标。该任务可能耗时数小时，先进行以下自检，再在后台运行：

```bash
conda run --no-capture-output -n oneadmet python scripts/train_jia2025_author_like_r3a.py --check-only
bash scripts/run_jia2025_long_task_background.sh scripts/run_jia2025_author_like_r3a_v1.sh
```

`check_jia2025_long_task.sh <实际Log路径>` 会显示进程状态、最后日志和 `complete.json` 标记。不要把 `--max-author-train-rows` 产生的 `partial=true` 工程 smoke 当作正式结果。

`register_jia2025_author_like_r3b_protocol.py` 和 `evaluate_jia2025_author_like_r3b.py` 分别登记、评分 Jia 的 B 级 author-like holdout。有效 protocol 为 `jia2025_author_like_r3b_protocol_v5`，绑定 R3a 模型文件、N0 canonical-parent 成员和 evaluator code hash。v3 的预测前失败源于 raw-SMILES/canonical-parent hash 混用；只可使用 v5。先只运行下面 smoke；只有用户明确确认后才执行第二条一次性评分命令：

```bash
conda run --no-capture-output -n oneadmet python scripts/evaluate_jia2025_author_like_r3b.py --check-only
conda run --no-capture-output -n oneadmet python scripts/evaluate_jia2025_author_like_r3b.py --membership-smoke
conda run --no-capture-output -n oneadmet python scripts/evaluate_jia2025_author_like_r3b.py --confirm-one-time-author-test-score
```

评分后任何 evaluator 重跑、模型选择、校准或候选替换均被禁止。

## MMPK / NCA N0 只读审计

`audit_mmpk_n0.py` 对本地 MMPK v3 论文、SI、代码、raw/model dataset 和 10 个 checkpoint 做字段、剂量、目标变换、结构重叠、split 实现和许可证审计。不会训练、读取项目受保护 test 或导出原始标签。推荐产物为 `results/analysis/mmpk_n0_audit_v5`：MIT 已确认适用于代码及随附文档；数据的 Zenodo 记录显示版权而未给出单独数据复用许可证，故不得静默假定数据也受 MIT 覆盖；后续 N1 仅限内部 train-only protocol，外部标签必须封存。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_mmpk_n0.py
```

`audit_mmpk_n1a_lineage_feasibility.py` 在 N0 哈希约束下，只对 approved raw/model data 进行 canonical parent + dose 来源映射。其 v2 输出 hash lineage、target label-origin proxy 及 parent/source/scaffold connected-component 可行性；不导出 raw 标签、不训练、也不读取 external 标签。N1b 必须以其组件与两类 label tier 为输入。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_mmpk_n1a_lineage_feasibility.py
```

`freeze_mmpk_n1b_cohort_protocol.py` 冻结 MMPK approved development 的 R1 direct-observation / R2 author-augmented label tier、作者 exact-SMILES 10-fold 与项目 strict parent+source+scaffold 5-fold。唯一有效输出为 `data/nca_internal_development/mmpk_n1b_cohort_protocol_v3`；v1 被发现有空 fold，v2 曾将一条无 finite transformed `Log F` 的 `F=130%` 边界原始记录误列为监督可用，二者均未用于训练。v3 保持所有 strict fold membership 不变，仅透明排除该不可训练 mask。程序不导出标签、不训练、不读取 external。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_mmpk_n1b_cohort_protocol.py
```

`mmpk_nca_common.py` 提供后续 NCA 模型唯一允许使用的 strict outer-train loader；它只识别 approved source table，输出 ECFP4(2,048)+作者 `Log Dose [mg/kg]` 和 R1/R1+R2 masked transformed targets。每个 dose scaler state 都必须在对应 outer-train 拟合，接口不返回 held-out labels。`smoke_mmpk_n1c_train_fold_loader.py` 对全部五折和两个 tier view 验证这一契约；不拟合模型、不创建预测或性能指标。

```bash
conda run --no-capture-output -n oneadmet python scripts/smoke_mmpk_n1c_train_fold_loader.py
```

默认产物为 `results/analysis/mmpk_n1c_train_fold_loader_smoke_v1`。它已发布，禁止覆盖或将其计数表当作性能结果；后续训练必须使用新的 protocol/输出路径并继续封存 external。

`audit_mmpk_n1d_context_derivation.py` 将 approved model records 以 N1a 的 parent+dose 规则映射回 approved raw metadata，审计 formulation、AUC type、aggregate subject metadata、route/time-window/analyte/matrix 及 R1/R2 derivation 边界。它只输出哈希化 context 状态和覆盖计数，不导出 target values、拟合模型或读取 external。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_mmpk_n1d_context_derivation.py
```

正式产物为 `results/analysis/mmpk_n1d_context_derivation_audit_v1`。当前只放行 N2a baseline protocol freeze；不得将固定 oral cohort scope 误称为逐行 route-conditioned 通用 NCA 模型。

`freeze_mmpk_n2a_baseline_protocol.py` 在不读取 target values 的前提下，冻结第一批 MMPK 条件基线的 strict outer/inner hash memberships、R1 主开发与 R1+R2/formulation sensitivity、有限算法候选、指标和作者 B 级参考栏。它不训练、不预测、不读取 external。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_mmpk_n2a_baseline_protocol.py
```

N2a v1 因未明确固定 seed 与 candidate–feature-view 对应关系而仅保留为历史，未训练；唯一有效协议为 `data/nca_internal_development/mmpk_n2a_baseline_protocol_v2`。N2b 必须严格使用其中的 nested membership，先通过 runner smoke 与 XGBoost GPU smoke，才可执行正式 R1 screen；禁止使用 outer evaluation 或 R2/formulation sensitivity 重新选择候选。

`smoke_mmpk_n2b_strict_baseline_runner.py` 完成无指标的 train-only nested 工程检查；`run_mmpk_n2b_strict_baseline_screen.py` 是需明确确认的 1,700-fit 正式 R1 screen。它在每个 outer-train 内选择候选后才读取对应 outer R1 labels；R2、formulation 与 author B-track 均不会参与选择。先运行 `check_gpu_boosters.py --algorithms xgboost`，再用下列 wrapper 交由后台执行：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_mmpk_n2b_strict_r1_baseline_screen_v1.sh
```

`analyze_mmpk_n2b_strict_baseline_screen.py` 只读验证已完成 N2b screen 的 prediction membership 与保存指标，生成 pooled/fold-wise performance、候选稳定性和两张英文 600-dpi PNG；禁止重训、校准或重新选择。

```bash
conda run --no-capture-output -n oneadmet python scripts/analyze_mmpk_n2b_strict_baseline_screen.py
```

正式分析包为 `results/analysis/mmpk_n2b_strict_r1_baseline_analysis_v1`。N2b 后不得因 outer 性能重新选算法；R1+R2 与 formulation 只能在 N2c 预注册的 paired sensitivity 中使用。

`freeze_mmpk_n2c_sensitivity_protocol.py` 已发布唯一有效的 N2c 登记：它逐 outer-fold/endpoint 绑定 N2b 的 20 个 R1-selected configurations，并登记分离的 `R1_plus_R2_train` 与 `formulation_matched` 配对敏感性臂。它不训练、不读取 outer labels、不输出预测或性能。比较只能以 canonical-parent clustered paired bootstrap（10,000 次、seed `20260919`）报告；禁止重选、将两个敏感性臂合并、或将 formulation 的子集与全量 baseline 非配对比较。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_mmpk_n2c_sensitivity_protocol.py
```

`smoke_mmpk_n2c_paired_sensitivity_runner.py` 已用 outer-0 完成 no-label 工程检查：它重建 N1d formulation mapping、在 outer-train 内拟合 one-hot 词表，并确认 R1/R1+R2 masks 与所有 4 个已选配置的输入维度兼容。该 smoke 严禁打开 outer target columns、拟合模型、预测或评分；正式 paired runner 必须另建新输出路径。

`run_mmpk_n2c_paired_sensitivity.py --smoke-no-label` 进一步对 outer-0/AUC 的两个 frozen arms 完成真实 fit 与 held-input inference，但不写 prediction、不读 held target、不计算 metric。CPU smoke 已完成；在 RTX 工作站先运行 CUDA smoke：

```bash
conda run --no-capture-output -n oneadmet python scripts/run_mmpk_n2c_paired_sensitivity.py \
  --smoke-no-label --device cuda --threads 8 \
  --output results/analysis/mmpk_n2c_paired_sensitivity_fit_smoke_cuda_v1
```

成功后，正式 N2c 是 40 个预注册 refits，必须在后台运行：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_mmpk_n2c_paired_sensitivity_v1.sh
```

它只在每一 outer fold 的两个 sensitivity arms 全部预测后才读取该 fold 的 R1 labels 评分，并持久化同记录 paired predictions；bootstrap 仍由后续只读脚本完成。

`analyze_mmpk_n2c_paired_sensitivity.py` 是 N2c 的唯一 closeout 分析器：它仅附加已经消费的 direct-R1 outer labels，逐项重算 40 个正式指标，并按 canonical-parent 进行预注册的 10,000 次 paired bootstrap（seed `20260919`）。它不拟合、预测、校准、选择或访问 external/test；正式产物为 `results/analysis/mmpk_n2c_paired_sensitivity_analysis_v1`，其中包括机器可读 CI/决策表和两张全英文 600-dpi PNG。

```bash
conda run --no-capture-output -n oneadmet python scripts/analyze_mmpk_n2c_paired_sensitivity.py --confirm-read-only-analysis
```

正式 closeout 已完成且 8 个比较均未达 sensitivity-signal 三重门槛；因此不得基于 R2/formulation 启动 N3 条件 MLP/MTL 或替换 N2b strict R1 基线。

`audit_gate7_release_readiness.py` 是 Gate 7 的 R0 只读资产审计：它核对 Gate-4 七端点决定、冻结/历史模型的 `complete.json` 与全部 artifact hash、模型文件、单位和测试生命周期。它不拟合、不预测、不读取任何标签，也不修改候选。特别地，它会把后续 Gate-4 train-CV 研究参照与已消费测试的历史资产分开，禁止统一接口静默混用二者。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_gate7_release_readiness.py
```

正式有效审计为 `results/analysis/gate7_release_readiness_audit_v3`。v1 因 `complete.json` 缺少显式 `partial=false` 而只保留作工程谱系；v2 虽完整但继承 Gate-4 中 CLint 的过期 `test_unread` 记录，均不再作为当前接口真相。v3 使用同一只读资产核验，并由已完成的 Gate-5 evaluation/closeout 将 CLint 生命周期校正为 `test_consumed_no_reselection`。R0 确认 1 个当前 final 资产、5 个历史 test-locked 资产和 1 个 F 状态接口，但**不**授权单一数值 release；下一步只能冻结双轨 input/output contract 与无标签 batch-inference smoke。

`freeze_gate7_release_contract.py` 固定 Gate 7 R1 的双轨输入/输出 contract 及后续 no-label batch-inference smoke protocol。输入只接受 `smiles` 和可选 `input_id`；任何标签类列名都必须拒绝。输出必须保留 asset lane、模型版本、单位、AD、证据状态、test lifecycle 与限制。F 永远只返回 `not_available_exploratory` 状态。此步骤为纯声明性冻结，不加载模型、不预测、不读取标签。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_gate7_release_contract.py
```

正式契约为 `data/public_development/gate7_release_contract_v2`；v1 保留其工程谱系但因继承 R0 v2 的过期 CLint lifecycle 而不可用于当前接口。v2 预注册了一个后续显式确认的、2–8 条无标签输入的技术 smoke：它可验证模型装载、输出 schema、AD 与重复确定性，但禁止读取标签、计算性能、校准、选择或把历史资产与 Gate-4 研究参照直接比较。

`smoke_gate7_release_interface.py` 依照 v2 contract 运行唯一的 Gate 7 R2 默认二探针技术 smoke。它实际加载六个数值资产、计算技术性预测/nearest-train Tanimoto 并检查重复确定性；F 仅写 status-only 行。它拒绝标签类输入列，且不读取标签或计算任何性能指标。

```bash
conda run --no-capture-output -n oneadmet python scripts/smoke_gate7_release_interface.py --confirm-no-label-smoke
```

正式 smoke 为 `results/analysis/gate7_release_no_label_smoke_v2`：2 条无标签探针产生 14 个任务行，12 个数值技术预测均为有限值、2 个 F 行均为 status-only，最大重复差为 `1.78e-15`（阈值 `1e-12`）。这是接口通过，不是性能评价，也不授权统一数值 release。

`audit_gate7_model_data_cards.py` 是 Gate 7 R3 的只读卡片生成器：它从 Gate-4、R1 contract 和 R2 smoke 汇总端点定义、绑定资产、数据版本、证据/lifecycle、AD 与不确定性可用性。它不加载模型、不预测、不读标签或生成新性能。当前唯一正式输出为 `results/analysis/gate7_model_data_card_audit_v1`。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_gate7_model_data_cards.py
```

R3 结论：仅 CLint 已有绑定资产内预注册的 train-only AD 阈值；fu/t½ 可技术性取得 ensemble spread，但所有数值端点均未打包经校准的预测区间。该缺口应在后续模型卡/接口中明确显示，不得由技术 smoke 推断 coverage 或区间校准。

`infer_gate7_dual_lane.py` 是 Gate 7 R4 的可复用双轨、无标签批量推理入口。它复用 R2 已验证的模型加载和技术 AD 路径，并绑定 R1 v2 contract 与 R3 卡片；输入只允许 `smiles` 和可选 `input_id`，输出逐行保留 model lane/version、单位、Tanimoto/AD、未校准不确定性可用性、证据状态、test lifecycle 和限制。六个数值端点仅返回 `technical_prediction_available_not_release_authorized`；F 始终是无数值的 `not_available_exploratory`。

```bash
conda run --no-capture-output -n oneadmet python scripts/infer_gate7_dual_lane.py \
  --input data/public_development/gate7_release_cli_smoke_input_v1.csv \
  --output results/analysis/gate7_release_cli_smoke_v1 \
  --technical-smoke --confirm-no-label-inference
```

`--output` 必须是新目录。常规批量推理只去掉 `--technical-smoke`，但仍必须显式提供 `--confirm-no-label-inference`；它不读取标签、计算性能、训练、校准或改变候选，也不授权统一数值 release。正式 R4 smoke 使用 acetone/cyclohexane 两条新探针，14 行 schema 完整、12 个数值输出有限、F 的两行均为 status-only，重复最大差为 `0`。输入边界由 `scripts/tests/test_gate7_release_cli.py` 的标准库测试覆盖，无需 `pytest` 依赖。

`analyze_gate7_endpoint_evidence_matrix.py` 是 Gate 7 R5 的只读证据整合器。它只读取 Gate 4、R0/R3/R4 和既有冻结 test/CLint/t½ **汇总表**；拒绝读取模型、逐行 prediction 或原始标签，也不重算指标。输出七端点证据矩阵、来源登记、主图规划表和英文 600-dpi 状态图，用于避免把 historical test-locked 资产、Gate-4 train-CV 参照和 technical interface 混写为同一性能证据。

```bash
conda run --no-capture-output -n oneadmet python scripts/analyze_gate7_endpoint_evidence_matrix.py \
  --confirm-read-only-evidence-synthesis
```

当前有效产物为 `results/analysis/gate7_endpoint_evidence_matrix_v3`。v1/v2 均保留：v1 曾把 t½ 来源组成审计中的 `519/552` 误称为模型训练分子，v2 已澄清其发生在结构可用性过滤之前；v3 进一步将 CLint 的 AD-outside 数量与其他端点的 `Tanimoto <0.30` 数量分列标识。R5 不产生新的模型性能结论：只有 CLint 有绑定资产的预注册 AD；所有数值端点仍没有经校准预测区间；t½ 的 `50/53` 冻结 test 分子与 TDC ecosystem 重叠，不能声称独立外部验证。

`freeze_gate7_reproducibility_delivery_manifest.py` 与 `verify_gate7_clean_environment_reload.py` 构成 Gate 7 R6 的无标签可复现性交付链。前者冻结 R0/R1/R3/R4/R5 前序哈希、关键代码/输入/技术输出哈希、Conda explicit 与 Pip 全量环境快照、运行时版本，以及独立环境重载协议；它不加载模型或标签，也不创建环境。后者只核对新的 label-free technical smoke：输入与 contract/card provenance 严格哈希一致、输出 schema/非预测字段严格一致、12 个数值技术输出的跨进程最大绝对差不超过冻结的 `1e-12`。CSV 字节 SHA-256 仍记录为审计信息，但不作为跨进程 pass/fail，因为末位浮点序列化可改变字节而不改变数值结果。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_gate7_reproducibility_delivery_manifest.py
conda run --no-capture-output -n oneadmet python scripts/verify_gate7_clean_environment_reload.py \
  --manifest data/public_development/gate7_reproducibility_delivery_manifest_v4 \
  --candidate-smoke results/analysis/gate7_reproducibility_same_env_replay_smoke_v1 \
  --output results/analysis/gate7_reproducibility_same_env_verifier_smoke_v3 \
  --confirm-clean-reload-verification
```

当前有效清单是 `data/public_development/gate7_reproducibility_delivery_manifest_v8`；独立环境技术结果为 `results/analysis/gate7_clean_environment_reload_smoke_v2` 和 `results/analysis/gate7_clean_environment_reload_verification_v3`。新环境 `oneadmet-gate7-reload-v2` 的 14 行输出 schema/非预测字段与 R4 完全一致，12 个数值技术输出最大绝对差为 `0.0`、CSV SHA-256 相同、in-process repeat 差为 `1.78e-14`（阈值 `1e-12`）。这仅是无标签技术重载验证，不等同于性能评价或统一数值 release。v1–v7 均保留为不可覆盖工程谱系：v4 explicit-file、v5 PyG wheel source、v6 Conda local build path 依次停止；v7 解决路径问题。此次 verifier v2 曾将数值诊断 `max_abs_cross_process_prediction_difference=0.0` 误纳入布尔 `all()`，产生无完成标记的失败；v8 只对显式布尔检查项作通过判断，未改任何模型、输入、标签或预测。

`freeze_gate8_manuscript_evidence_registry.py` 是 Gate 8 R0 的只读稿件证据登记器。它只验证允许引用的上游 `complete.json`，并哈希指定的汇总表或发表图；不读取原始标签、逐行预测或模型，且不重算指标、拟合、校准或改变候选。当前正式登记为 `data/public_development/gate8_manuscript_evidence_registry_v2`：15 个来源、36 个资产，覆盖 Table 1–4、Figure 1–6 和 Supplement。Jia author-like B 级结果与原文 C 级背景值必须在后续表中分行；MMPK 仅内部使用，禁止公开比较行或逐行数据/预测发布。v1 因 B/C 混合来源标记不够明确而仅保留工程谱系。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_gate8_manuscript_evidence_registry.py --check-only
```

正式目录不可覆盖；如协议将来需要修订，必须先说明理由并使用新的明确版本路径。

`compile_gate8_table1_schema.py` 是 Gate 8 R1 的主表 1 schema/汇总入口。它必须先验证 R0 v2，再只读取 R0 manifest 明示并哈希绑定的 `endpoint_candidate_freeze_registry.csv` 与 `endpoint_diagnostic_evidence_matrix.csv`。输出 7 行、12 字段的英文上下文表、字段级来源/主张约束和两项资产 contract；不读取标签、逐行预测、模型或 fold 成员，也不重算指标、拟合、校准或重选。精确 parent/scaffold/source/study/fold 成员只能继续在冻结 registry/protocol 与 Supplement 报告，不能由该摘要表反推。

```bash
conda run --no-capture-output -n oneadmet python scripts/compile_gate8_table1_schema.py --check-only
```

当前正式输出为 `data/public_development/gate8_table1_schema_v1`；不可覆盖。若需要变更字段或证据边界，须先说明理由、影响和测试生命周期，并使用新的版本目录。

`compile_gate8_table3_strict_internal.py` 是 Gate 8 R2 的主表 3 严格内部证据装配入口。它必须先验证 R0 v2，再只读取其中已哈希的 Gate 4 endpoint registry、P1 matched summary、B6 endpoint decision 表与 B6 aggregate decision。输出 18 行、15 字段的全英文长表：6 个冻结端点参考、6 个 P1 matched comparisons、6 个 B6 negative-ablation decisions。所有行固定为 B 级；F 因 C 级探索性状态被排除。它不读取标签、逐行预测、模型、fold/test 成员，也不重算指标、bootstrap、排名、拟合、校准或选模；不同 `comparison_id` 的数值不得相互排名。

```bash
conda run --no-capture-output -n oneadmet python scripts/compile_gate8_table3_strict_internal.py --check-only
```

当前正式输出为 `data/public_development/gate8_table3_strict_internal_v1`；不可覆盖。

`compile_gate8_table4_frozen_evidence.py` 是 Gate 8 R3 的主表 4 装配入口。它先验证 R0 v2，并只读取其中已哈希的 Gate 7 evidence matrix、CLint closeout 的 test/calibration 摘要、legacy frozen primary/AD 摘要、Gate 7 delivery manifest 和 clean-reload verification。输出七端点 frozen diagnostic panel、一个 engineering panel、字段 schema、来源 contract 和 terminal t½ 排除谱系。legacy t½ `n=2` exploratory 行不得与 Gate 7 v15 `n=53` source-overlap-limited summary 合并；工程重载不等于性能验证。脚本不读取标签、预测、模型、fold/test 成员，且不重算指标/bootstrap、拟合、校准或选模。

```bash
conda run --no-capture-output -n oneadmet python scripts/compile_gate8_table4_frozen_evidence.py --check-only
```

当前正式输出为 `data/public_development/gate8_table4_frozen_evidence_v1`；不可覆盖。

`compile_gate8_table2_literature_boundaries.py` 是 Gate 8 R4 的主表 2 文献/author-like 边界装配入口。它先验证 R0 v2，读取两项 Jia 汇总资产并将原来同一行的 published Jia 2025 与 project author-like 数值分别写成 C 级和 B 级行；差值列不输出，且 project strict context 保持为独立 B 级比较域。MMPK 的两项 R0 资产只被哈希验证，数值不解析、不写出；输出只有无数值的 `internal_only` 许可边界。脚本不读取标签、预测、模型、fold/test 成员，且不重算指标、拟合、校准、排名或选模。

```bash
conda run --no-capture-output -n oneadmet python scripts/compile_gate8_table2_literature_boundaries.py --check-only
```

当前正式输出为 `data/public_development/gate8_table2_literature_boundaries_v1`；不可覆盖。

`freeze_gate8_supplement_manifest.py` 是 Gate 8 R5 的 Supplement 装配/权限登记器。它验证 Gate 8 R0–R4 与 R0-approved P1、B6、Gate 7 reproducibility 资产的完成标记和哈希，输出公开资产、MMPK internal-only 资产、装配缺口及 S1–S9 计划。它不解析标签、预测、模型、foldwise 或 MMPK 数值；未登记的逐折/详细 bootstrap 资产被标记为 S8 的未装配缺口，MMPK 数值材料被标记为 S9 的公开排除项。不得由该入口补入任何未登记或受限内容。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_gate8_supplement_manifest.py --check-only
```

当前正式输出为 `data/public_development/gate8_supplement_manifest_v1`；不可覆盖。

`freeze_gate8_figure_contract.py` 是 Gate 8 R6 的主图资产/图注冻结入口。它只验证 R0 v2 批准的 PNG hash、PNG header 与像素尺寸，输出 Figure 1–6 的面板 asset manifest、英文 caption/claim contract、Figure 6 nonvisual caption evidence、source contract 及排除登记。它不重绘或重算任何图，也不读标签、预测、模型或指标表。Figure 4 固定为 CL/fu/VDss/t½ 四面板，Jia 不替代主图，MMPK 图因 internal-only 权利限制排除。该 contract 是后续纯排版合成的唯一输入边界。

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_gate8_figure_contract.py --check-only
```

当前正式输出为 `data/public_development/gate8_figure_contract_v1`；不可覆盖。

`compose_gate8_figures.py` 是 Gate 8 R7 的投稿图 layout-only renderer。它先验证 R6 contract 和每个源 PNG 的 SHA-256/像素尺寸，再以保持纵横比的方式将 13 个面板排入六张白色画布，添加外置英文 Figure title 与面板字母，并以 600 dpi 输出 PNG。它不读取任何标签、预测、模型或指标表，也不修改源图的数据层、数值、颜色、图例或 R6 claim boundary。当前正式输出为 `results/analysis/gate8_figure_composition_v1`；不可覆盖。

```bash
conda run --no-capture-output -n oneadmet python scripts/compose_gate8_figures.py --check-only
```

`build_gate8_manuscript_outline.py` 是 Gate 8 R8 的英文稿件 scaffold 入口。它仅验证 R0–R7 的 `complete.json`，生成题目、结构化摘要、Introduction、Methods、Results、Discussion、Limitations 和 Declarations 的简洁英文提纲，以及 18 条逐段 claim–evidence map 与已验证阶段清单；不读取标签、预测、模型、指标表或 MMPK 数值，不生成性能数值、不拟合、校准、选模或重开已关闭 test。当前正式输出为 `进度/投稿草稿/gate8_manuscript_outline_v1`；不可覆盖。该简洁 scaffold 是当前充分的写作交付，不再扩写完整正文。

```bash
conda run --no-capture-output -n oneadmet python scripts/build_gate8_manuscript_outline.py --check-only
```

`audit_gate9_second_iteration_readiness.py` 是 Gate 9 R0 的只读二轮重构决策审计入口。它只读取 Gate 4/7/8 的已完成摘要资产，核验 7 个端点、34 条已止损/未晋级的复杂架构记录及生命周期边界，输出逐端点的新独立证据要求、匹配基线、首步预算和止损条件；不读取标签、预测、模型、test 数值或 MMPK 数值，不重算性能、不训练或重开 test。Gate 7 是当前生命周期权威；Gate 4 的历史状态仅保留用于时间线协调。当前正式输出为 `data/public_development/gate9_second_iteration_readiness_v2`；v1 因未显式提升 CLint 的较新 Gate 7 生命周期而只保留工程谱系，不得用于当前决策。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_gate9_second_iteration_readiness.py --check-only
```

`build_gate9_papp_independent_cohort_protocol.py` 是 Gate 9 R1 的 metadata-only Papp 二轮 cohort 协议入口。它验证 Gate 9 R0 与 processed v15 dataset/split stages，只读取 Papp 的 row/molecule/split/assay/document/source-count 元数据以及 eligible split registry 的 parent/scaffold 标识；不读取 SMILES、raw/target value、mask、预测、模型或历史 test 值。它登记三类独立来源候选、两级独立性定义及匿名 exclusion fingerprints。当前正式输出为 `data/public_development/gate9_papp_independent_cohort_protocol_v1`。其初始 strict cohort admission 因 24 个历史 molecule identities（27 records）缺失 eligible-registry parent/scaffold 指纹而为 false；该谱系缺口已由 R2 的保守 exclusion-ledger audit 关闭，但任何新来源仍须经过 R3 资格筛选。

```bash
conda run --no-capture-output -n oneadmet python scripts/build_gate9_papp_independent_cohort_protocol.py --check-only
```

`audit_gate9_papp_unmapped_lineage.py` 是 Gate 9 R2 的无标签 Papp 谱系协调入口。它仅对 R1 登记的 27 条 records/24 个 molecule identities 读取 SMILES 与完整 split registry 的结构/资格元数据，以冻结 `structure_groups` 规则重算 parent/scaffold 并核验一致性；不读取 Papp raw/target value、mask、预测、模型或 test 值。正式输出为 `data/public_development/gate9_papp_unmapped_lineage_audit_v1`：27 条均成功匹配、均因 `cross_split_parent_or_scaffold` 保持 `eligible=False`，零条恢复到任何历史集合；仅输出未来外部候选需排除的匿名补充指纹。

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_gate9_papp_unmapped_lineage.py --check-only
```

`evaluate_gate8_performance_benchmark_scorecard.py` 是只读性能评价与领域对标汇总入口。它验证 Gate 4 candidate freeze、Gate 8 Table 2–4 和 TDC Half-Life_Obach benchmark，只读取已冻结摘要：输出 Jia published C-level 与 project author-like B-level 的 6 个 context cells、TDC A-level 的 3 行 direct benchmark、七端点 strict-internal/frozen-diagnostic 分栏及 12 条 P1/B6 architecture decision evidence。它不读取标签、预测、模型或 test 值，不重算指标、不训练、校准或选模，且不能改变任何 test 生命周期。当前正式输出为 `results/analysis/gate8_performance_benchmark_scorecard_v2`；v1 保留，v2 仅改善 Figure 2 单元格可读性。两幅图均为英文 600-dpi PNG，所有跨协议排名均被明确禁止。

```bash
conda run --no-capture-output -n oneadmet python scripts/evaluate_gate8_performance_benchmark_scorecard.py --check-only
```

`analyze_jia2025_r3c_closeout.py` 只读汇总已关闭的 R3b 评分、R3a 模型冻结和 Gate 4 证据边界；正式报告为 `results/analysis/jia2025_author_like_r3c_closeout_v2`。`analyze_jia2025_r3d_bootstrap.py` 进一步仅对已保存的 R3b 预测做 canonical-parent cluster bootstrap；它先重验保存预测可逐项重算既有指标，随后输出描述性 95% CI，不会重跑 scorer、拟合模型、校准或选择模型。其 10,000 次正式运行应在后台启动：

```bash
bash scripts/run_jia2025_r3d_bootstrap_background_v1.sh
```

将输出的实际 `Log:` 路径（不含尖括号）传给：

```bash
bash scripts/check_jia2025_r3d_bootstrap_task.sh 实际Log路径
```

## 1. 文件与职责

| 程序 | 用途 | 默认输出 |
|---|---|---|
| `freeze_cross_gate_endpoint_candidates.py` | 汇总 Stage-A、P1、多模态、跨物种、多任务与 Gate 3 正式证据，发布 Gate 4 端点候选、测试生命周期和文献对标登记；不拟合模型、不读取 CLint test，也不直接修改研究路线 | `data/public_development/cross_gate_candidate_freeze_v1` |
| `freeze_gate5_clint_extratrees.py` | 仅使用已发布 train-only 表和无标签特征缓存，按 Gate 4 冻结的 ECFP4+RDKit2D ExtraTrees c01 在全部 CLint train 上拟合、哈希、重载并预注册 AD/评价规则；不读取 validation/test | `models/frozen/CLint__human__microsome/gate5_stageA_et_c01_v2` |
| `evaluate_gate5_clint_frozen.py` | 原 Gate 5 evaluator；首次确认运行在 test-feature cache 缺口处、预测/指标/产物前停止，保留供失败审计，**不得重跑** | `results/final/CLint_gate5_stageA_et_c01_test_v1`（未创建） |
| `register_gate5_clint_technical_recovery.py` | 历史 recovery v1 登记；其 evaluator 后因混用来源 `molecule_id` 与 canonical `parent_id` 在预测前停止，保留供审计 | `data/public_development/gate5_clint_technical_recovery_v1` |
| `evaluate_gate5_clint_technical_recovery.py` | 历史 recovery v1 evaluator；未产生预测/指标/产物，**不得重跑** | `results/final/CLint_gate5_stageA_et_c01_test_recovery_v1`（未创建） |
| `register_gate5_clint_technical_recovery_v2.py` | 登记最终恢复修正：明确 source `molecule_id` 仅用于追踪、split `parent_id` 用于 canonical-feature 身份验证；绑定原模型、v1、split 与最终 evaluator 哈希 | `data/public_development/gate5_clint_technical_recovery_v2` |
| `evaluate_gate5_clint_technical_recovery_v2.py` | 使用未变模型与评价规则执行最终完成性评分；运行前重验 302 个训练 parent 的逐元素特征等价性，无第三次恢复 | `results/final/CLint_gate5_stageA_et_c01_test_recovery_v2` |
| `analyze_gate5_clint_final_closeout.py` | 只读核验最终评分哈希，生成 AD/source/CV-test 诊断、英文 600-dpi PNG 和不可再评分声明；禁止重训、校准或重选 | `results/analysis/gate5_clint_final_diagnostic_closeout_v1` |
| `package_research_process_archive.py` | 将已发布的关键研究图复制到长期研究档案，并生成原始路径、SHA-256 与字节数清单；不修改源产物 | `进度/研究过程/figures` |
| `audit_endpoint_sources.py` | 匹配原始 ChEMBL，审计物种/端点/单位/删失及旧标签契约 | `data/processed_v2/audit` |
| `papp_provenance_overrides.py` | 读取原文 DOI 单位裁定，并在审计时选择性修正 Papp 的单位解释 | 由 `audit_endpoint_sources.py` 写入审计谱系 |
| `rebuild_endpoint_datasets.py` | 重建来源长表、精确测定任务视图及变换标签 | `data/processed_v2/datasets` |
| `build_split_manifest.py` | 固定 train/val/test 成员、结构排除清单、共享训练折 | `data/processed_v2/splits` |
| `audit_multitask_endpoint_readiness.py` | 只读汇总 7 个投稿主端点的任务定义、split 规模、开发来源数、跨端点成员重叠及 test 生命周期；不输出端点数值 | `results/analysis/multitask_endpoint_readiness_v1` |
| `freeze_CLint_v6_candidate.py` | 以既有 nested-OOF 规则冻结 CLint v6 的 RF 三种子集成；验证成员哈希和重载，禁止读取 test 标签 | `models/frozen/CLint__human__microsome/v6_rf3_v1` |
| `evaluate_CLint_v6_frozen.py` | 仅在显式确认后对冻结 CLint 集成进行一次 test 评分；禁止重拟合、校准或重选 | `results/final/CLint_v6_rf3_test_v1` |
| `build_multitask_physics_explainability_contract.py` | 发布七端点物理语义、允许的条件性关系、禁止硬约束和解释性证据协议；不读取标签、预测或模型 | `results/analysis/multitask_physics_explainability_contract_v1` |
| `build_pkdb_absolute_f_candidate_queue.py` | 从标签盲态 PK-DB human+IV 快照发布“absolute bioavailability”一手来源索取队列；不是训练/验证集，也不输出 F 数值 | `results/analysis/pkdb_absolute_f_candidate_queue_v1` |
| `lock_pkdb_absolute_f_eligibility.py` | 锁定 PK-DB absolute-F 队列的一手来源资格（oral+IV、血浆母体、绝对 F 与定位证据）；明确禁止写入 F 数值 | `results/analysis/pkdb_absolute_f_eligibility_lock_v1` |
| `lock_pkdb_absolute_f_extraction.py` | 在 F 资格锁后锁定 Table 1 的几何均值 Fp.o. 数值、90% CI、剂量和 microdose 溯源；不改变既有数据集或模型 | `results/analysis/pkdb_absolute_f_extraction_lock_v1` |
| `build_absolute_f_increment_view.py` | 对已锁定的 F 文献记录核验现有人 F 父体、骨架和来源隔离，并发布保留的单来源增量视图；不修改 `processed_v15` | `data/literature/absolute_f_increment_view_v1` |
| `register_public_f_development_candidates.py` | 批量登记 ChEMBL review 与 F 专用 PK-DB 题名候选，隔离严格 F/保留增量且不读取数值；为 public-F cohort 做准备 | `results/analysis/public_f_development_candidate_registry_v1` |
| `audit_public_f_definition.py` | 对 ChEMBL “absolute bioavailability”强信号按文献/assay 审计端点定义；无资格者不进入 public-F cohort，且不读取数值 | `results/analysis/public_f_definition_audit_v1` |
| `build_public_f_primary_source_request_queue.py` | 将 F 专用 PK-DB 候选按原始研究聚合为全文索取/资格审核队列；不读取端点数值 | `results/analysis/public_f_primary_source_request_queue_v1` |
| `lock_public_f_pkdb_eligibility.py` | 锁定 F 专用 PK-DB 原文的 oral+IV、母体和 absolute-F 资格；分层来源及未直接报告 F 的研究不进入数值提取 | `results/analysis/public_f_pkdb_eligibility_lock_v1` |
| `audit_obach_thalf.py` | 以 Obach 原文和 TDC 镜像批量核验人体 IV terminal Thalf，并生成 activity 级证据表 | `results/analysis/obach_thalf_audit_v1` |
| `run_v15_build.sh` | 使用 Obach v13 证据表顺序构建并严格核验 processed_v15 | `data/processed_v15` |
| `dmpk_toolkit.py` | 公共训练模块；另提供 `check`、`predict`、`evaluate-test` 命令 | 由调用入口决定 |
| `build_stl_benchmark_protocol.py` | 从 interface v5 固定六个人体 head、五折、canonical-parent 特征缓存、算法与分阶段选择契约；不训练或读取 test | `data/public_development/stl_benchmark_protocol_v1` |
| `stl_benchmark_common.py` | 强 STL benchmark 的统一 classical/boosting/MLP 算法适配器和 feature-view 加载 | 由筛选入口调用 |
| `run_stl_benchmark_screen.py` | 在协议约束下运行 train-CV 广筛或显式确认后的 validation；limited run 永久不可选 | `results/benchmarks/stl_broad_screen_v1` |
| `run_stl_first_batch_stageA_v1.sh` | 执行首批八算法、每算法最多两个候选的 RDKit2D frozen train-CV 长任务；不读取 validation/test | `results/benchmarks/stl_stageA_first_batch_rdkit2d_v1` |
| `audit_stl_stageA_results.py` | 对既有 Stage A OOF 做覆盖、折波动、来源簇外推、数值稳定性审计并生成算法多样临时短名单；不重拟合、不读取 validation/test | 推荐 `results/analysis/stl_stageA_first_batch_audit_v3` |
| `run_stl_linear_stabilization_stageA_v1.sh` | 只补做协议内强 Ridge 与两种 ElasticNet，建立数值稳定线性参照 | `results/benchmarks/stl_stageA_linear_stabilization_rdkit2d_v1` |
| `analyze_stl_linear_stabilization.py` | 比较线性补充与首批最佳稳定候选，显式审计极端标准化预测；不读取 validation/test | `results/analysis/stl_stageA_linear_stabilization_audit_v1` |
| `run_stl_ecfp4_stageA_v1.sh` | 在相同冻结五折运行 ECFP4 Stage A；长任务顺序执行 | `results/benchmarks/stl_stageA_ecfp4_v1` |
| `run_stl_combined_stageA_v1.sh` | 在相同冻结五折运行 ECFP4+RDKit2D 双视图 Stage A；不是完整多模态结论 | `results/benchmarks/stl_stageA_ecfp4_rdkit2d_v1` |
| `run_stl_long_task_background.sh` | 以 nohup 后台启动已登记的 ECFP4 或组合特征长任务，保存独立日志/PID；拒绝未登记脚本 | `results/pipeline_logs/stl_long_tasks/` |
| `check_stl_long_task.sh` | 检查后台任务 PID、显示末尾日志并提示 complete.json | 只读检查 |
| `audit_stl_stageA_three_views.py` | 统一审计 RDKit2D、ECFP4 和组合视图，排除数值/收敛异常并冻结 train-CV-only 短名单；不读取 validation/test | `results/analysis/stl_stageA_three_view_audit_v2` |
| `plot_stl_stageA_feature_views.py` | 根据统一审计表生成全英文 600 dpi 三视图性能图；不重新拟合 | `results/figures/stl_stageA_feature_view_comparison_v2` |
| `build_endpoint_expansion_feasibility.py` | 固化 F/logP/logD/pKa 的本地数据规模、角色、pKa 数据契约和 logP 派生描述符防泄漏规则；AUC/Cmax 明确延期 | `results/analysis/endpoint_expansion_feasibility_v1` |
| `build_gate1b_multimodal_protocol.py` | 固化八类模态、逐任务权限、B0–B8 增量消融、OOF 祖先、融合/mask、泄漏和候选晋级规则；不计算表征或训练模型 | `data/public_development/gate1b_multimodal_protocol_v2` |
| `build_gate1b_graph_manifest.py` | 从冻结 canonical-parent 成员生成去显式氢、化学等价的确定性图视图，并审计 23 维原子/6 维键 schema；不读取标签 | `data/public_development/gate1b_graph_manifest_v1` |
| `run_gate1b_dmpnn_screen.py` | 在相同冻结五折运行 graph+RDKit2D D-MPNN；CUDA 不可用时拒绝静默回退，保存 OOF、隔离和重载审计 | `results/benchmarks/gate1b_dmpnn_stage1_v1` |
| `run_gate1b_dmpnn_stage1_v1.sh` | 顺序运行六任务、两个预注册 D-MPNN 配置的正式 Gate 1B GPU 长任务；不读取 validation/test | `results/benchmarks/gate1b_dmpnn_stage1_v1` |
| `audit_gate1b_dmpnn_stage1.py` | 将正式 D-MPNN 与各端点冻结 Stage-A leader 作逐折 train-CV 配对，并按 ≥2% 改善且 ≥3/5 折不劣规则决定是否允许 validation | `results/analysis/gate1b_dmpnn_stage1_audit_v1` |
| `run_v15_thalf_feature_ablation.sh` | 顺序运行 Thalf v15 的 ECFP4、RDKit 2D 与机制 2D 特征消融 | `models/stl/Thalf__human__terminal_iv/v15_*_seed2026` |
| `train_fu.py` | 默认人血浆 fu | `models/stl/fu__human__plasma/<run-name>` |
| `train_CLint.py` | 默认人肝微粒体 apparent CLint | `models/stl/CLint__human__microsome/<run-name>` |
| `train_Papp.py` | 默认人源 Caco-2 A→B Papp | `models/stl/Papp__human__caco2_ab/<run-name>` |
| `train_F.py` | 默认人绝对口服生物利用度 F | `models/stl/F__human__absolute_oral/<run-name>` |
| `train_CL.py` | 默认人全身静脉 CL | `models/stl/CL__human__systemic_iv/<run-name>` |
| `train_VDss.py` | 默认人稳态静脉 VDss | `models/stl/VDss__human__steady_state_iv/<run-name>` |
| `train_Thalf.py` | 默认人终末静脉 Thalf 直接诊断基线 | `models/stl/Thalf__human__terminal_iv/<run-name>` |
| `train_else.py` | 显式选择物种、端点和体系的动物先验训练 | `models/stl/<task_id>/<run-name>` |
| `results_analysis.py` | 在 test 冻结时生成英文出版图表 | `results/analysis/<name>` |
| `cross_species_analysis.py` | 对照直接与跨物种模型的英文图表 | `results/analysis/cross_species_human_v1` |
| `train_cross_species.py` | 受保护大鼠→人软先验级联公共程序 | `models/cascade/cross_species/<task>/<run-name>` |
| `train_cross_species_{fu,CL,VDss,F,Thalf}.py` | 各端点跨物种级联入口 | 同上 |
| `train_multitask.py` | 共享 MLP / dense MMoE 掩码多任务训练 | `models/multitask/<run-name>` |
| `multitask_three_seed_analysis.py` | 三种子 MTL/STL 汇总、配对 OOF bootstrap、英文出版图表和模型选择报告 | `results/analysis/multitask_pappfixed_v3_three_seed_v1` |
| `report_multitask_gradient_affinity.py` | 对锁定的 train-only 梯度亲和性结果生成非对角线英文 600-dpi 图、稳定方向计数、端点对汇总和唯一有限架构候选；不重训、不读取 validation/test | `results/analysis/multitask_gradient_affinity_report_v2` |
| `build_multitask_two_group_protocol.py` | 将锁定梯度冲突结果固化为容量匹配的 `fu-family/non-fu` 双 encoder 唯一候选，并绑定既有 full-shared/private/Stage-A 对照与两级晋级 Gate | `data/public_development/multitask_two_group_protocol_v2` |
| `run_multitask_two_group_smoke.py` | 对双分支执行单折工程 smoke：审计跨分支零梯度、全任务覆盖、隔离、模块更新、有界输出和重载；不读取评价/validation/test targets | `results/benchmarks/multitask_two_group_smoke_v2` |
| `run_multitask_two_group_traincv.py` | 正式训练唯一容量匹配双分支候选；五折三种子，全任务联合清除，逐任务 train-only 处理，每折完成全部模型后才读取评价标签 | `results/benchmarks/multitask_two_group_traincv_v1` |
| `analyze_multitask_two_group_traincv.py` | 在相同 outer-fold parents 上复用锁定 full-shared、fully-private 与 corrected Stage-A，执行 paired bootstrap、机制性 Gate 和待来源敏感性的候选 Gate | `results/analysis/multitask_two_group_traincv_analysis_v1` |
| `freeze_gate2b_decisions_and_audit_gate3.py` | 绑定Stage-A、跨物种、共享/双分支、图与冻结SMILES证据，冻结Gate 2B止损状态，并仅授权B6实验logP/logD嵌套OOF进入Gate 3协议审计 | `data/public_development/gate2b_decision_freeze_gate3_audit_v1` |
| `build_gate3_b6_physchem_protocol.py` | 冻结logP/logD有限producer、C0/C1/B6三路线、双层nested预测祖先、容量匹配控制和端点止损门槛；不拟合模型或读取评价标签 | `data/public_development/gate3_b6_physchem_protocol_v2` |
| `build_gate3_b6_physchem_cohort.py` | 从审计clean pool发布记录lineage、parent等权目标、泄漏安全ECFP4/RDKit2D、辅助scaffold folds和60个端点特异purge memberships；不拟合模型 | `data/public_development/gate3_b6_physchem_cohort_v1` |
| `analyze_thalf_descriptor_seeds.py` | Thalf v15 RDKit2D/组合特征三种子汇总、集成和逐分子 bootstrap；不读取 test | `results/analysis/thalf_v15_rdkit2d_three_seed_v1` |
| `train_Thalf_whitebox.py` | 预注册 Thalf Elastic Net 与加性样条 GAM；固定嵌套折、收敛候选筛选、解释稳定性、一次 validation、无 test | `models/whitebox/Thalf__human__terminal_iv/v15_whitebox_preregistered_v2` |
| `analyze_Thalf_source_sensitivity.py` | Thalf v15 Obach/非 Obach 分层、来源等权指标及文献簇 bootstrap；不读取 test | `results/analysis/thalf_v15_source_sensitivity_v1` |
| `check_gpu_boosters.py` | 微型拟合并核验 XGBoost/LightGBM/CatBoost 确实使用 GPU，禁止静默 CPU 回退 | 无持久产物 |
| `train_Thalf_greybox.py` | 按 Thalf 固定外层折交叉拟合结构→CL/VDss，再以 `0.693×VDss/CL` 加结构残差预测 Thalf；无 test | `models/greybox/Thalf__human__terminal_iv/v15_greybox_preregistered_v1` |
| `benchmark_tdc_halflife.py` | 五个 TDC 官方 scaffold seeds 上复现锁定 RDKit2D ExtraTrees，以 Spearman 对齐公开 Half_Life_Obach 文献榜单；完全独立于 v15 test | `results/benchmarks/tdc_half_life_obach_rdkit2d_et_v1` |
| `build_Thalf_external_validation_queue.py` | 在标签盲态下排除 v15 与 TDC/Obach 重叠，生成独立人体 IV terminal Thalf 原文筛选队列和裁定模板 | `results/analysis/thalf_external_validation_queue_v4` |
| `collect_pkdb_iv_candidates.py` | 从 PK-DB 官方 API 保存可追溯 IV 快照，并生成不含端点数值的人体研究候选表 | `data/external/pkdb_iv_candidates_v1` |
| `prepare_oneadmet_auxiliary.py` | 核验 OneADMET ZIP/CSV、统计全部任务，并发布结构隔离的 PK 辅助长表与盲态候选表 | `data/external/oneadmet_auxiliary_v2` |
| `build_oneadmet_pk_splits.py` | 将 OneADMET PK 辅助表按母体/骨架生成安全 train/validation/folds，保留作者 test，并显式审计作者集合的母体/骨架交叉 | `data/external/oneadmet_pk_splits_v2` |
| `preprocess_pkdb_candidates.py` | 获取 PK-DB 物质节点，以 InChIKey/ChEMBL 精确映射结构并排除既有来源和结构重叠 | `data/external/pkdb_preprocessed_v1` |
| `build_pkdb_external_review_batch.py` | 应用审计过的母体结构映射，重新核验 v15/TDC/外部队列重叠，并发布标签盲态的 PK-DB 原文审核批次 | `results/analysis/pkdb_external_review_batch_v2` |
| `lock_pkdb_eligibility.py` | 校验全部 PK-DB 资格裁定、禁止端点数值提前进入，并发布带 SHA-256 的不可覆盖资格锁；锁后才允许填写数值提取表 | `results/analysis/pkdb_external_eligibility_lock_v1` |
| `lock_pkdb_extraction.py` | 核验提取表只引用当前资格锁、仅对 accepted 候选提取正数小时标签，并发布不可覆盖的数值提取锁 | `results/analysis/pkdb_external_extraction_lock_v1` |
| `collect_pubmed_iv_terminal_discovery.py` | 以 human+IV+terminal/elimination 文献线索建立仅含 DOI/PMID/题录的盲态 PubMed 发现队列，并排除已用来源；不下载摘要、全文、结构或端点数值 | `results/analysis/pubmed_iv_terminal_discovery_v1` |
| `collect_pubmed_iv_terminal_subqueues.py` | 以 Phase-I/健康受试者、单/多剂量、口服-IV 交叉三个设计信号，建立小型可去重的 PubMed 来源发现子队列；仍不下载摘要/全文/结构/数值 | `results/analysis/pubmed_iv_terminal_subqueues_v1` |
| `build_pubmed_fulltext_request_list.py` | 对盲态题录作题名级优先级分流，生成原始全文索取清单；动物/二次分析/大分子仅降优先，绝不构成正式资格裁定 | `results/analysis/pubmed_fulltext_request_list_v1` |
| `build_pubmed_external_review_batch.py` | 将全文索取短名单变为不含数值的 PubMed 资格审核模板；资格锁定前禁止数值提取或外部评分 | `results/analysis/pubmed_external_review_batch_v1` |
| `build_pubmed_independent_candidate_index.py` | 将全文已到件的 PubMed 候选映射至本地 ChEMBL 37，复核 v15/TDC/ChEMBL 队列的来源、母体和骨架独立性；索引仍不含端点数值 | `results/analysis/pubmed_external_independent_batch_v3` |
| `lock_pubmed_eligibility.py` | 将完整 PubMed 全文资格证据锁定为不可覆盖注册表；对相对全文路径按项目根目录解析，锁定前禁止数值 | `results/analysis/pubmed_external_eligibility_lock_v1` |
| `lock_pubmed_extraction.py` | 仅允许引用当前 PubMed 资格锁的小时标签提取，并记录聚合规则与页/表定位 | `results/analysis/pubmed_external_extraction_lock_v1` |
| `audit_external_validation_registry.py` | 对已锁定外部验证注册表发布来源/分子/裁定和小时范围审计；用于训练资料卡与文献对标，不产生训练标签 | `results/analysis/thalf_external_validation_composition_v22` |
| `build_pksmart_external_candidate_queue.py` | 将 PKSmart 公开 external CSV 变为标签盲态、结构隔离的原始来源追溯队列；因 CSV 无逐行原始来源定位，绝不直接产生正式外部标签 | `results/analysis/pksmart_external_candidate_queue_v1` |
| `map_pksmart_candidates_to_chembl.py` | 对 PKSmart 非重叠候选作本地 ChEMBL 37 精确映射，并将去电荷二级映射明确标为仅供名称检索；不读取或输出端点数值 | `results/analysis/pksmart_external_chembl_mapping_v1` |
| `audit_pksmart_candidate_provenance.py` | 以逐条可审计的官方标签作来源分流；口服标签只能证明其不能支撑未定位行的直接 IV 资格，绝不将其当作端点或“无 IV 研究”的结论 | `results/analysis/pksmart_external_provenance_audit_v1` |
| `render_pksmart_candidate_funnel.py` | 从锁定候选摘要、结构映射和 IV 产品线索清单生成可视化 HTML；只读取计数与来源状态，绝不读取端点数值 | 用户指定 HTML 路径 |
| `plot_pksmart_candidate_funnel.py` | 从同一锁定输入生成全英文发表级候选分流图（600 dpi PNG）；直接标注，避免图例/标签遮挡数据 | `results/analysis/pksmart_candidate_funnel_v1` |
| `build_pksmart_primary_source_request_queue.py` | 将已审核 IV 产品线索变为标签盲态的一手全文索取/来源定位队列；不产生正式资格或端点标签 | `results/analysis/pksmart_primary_source_request_queue_v1` |
| `register_pksmart_public_benchmark.py` | 在读取 PKSmart 数值前固定公开数据库对标的数据哈希、结构隔离 cohort、冻结 v15 模型、指标与禁止事项；该轨与正式外部集完全隔离 | `results/benchmarks/pksmart_public_thalf_v1` |
| `evaluate_pksmart_public_benchmark.py` | 校验 PKSmart 标签盲态登记、原始文件和冻结模型哈希后，一次性对公开标签评分；结果只能写入新的二级对标目录 | `results/benchmarks/pksmart_public_thalf_v1_evaluation_v1` |
| `build_oneadmet_public_thalf_cohort.py` | 从 OneADMET human-plasma 标签发布宽口径公开建模 cohort；隔离正式外部与已评分 PKSmart 结构，并保留标签盲态作者 source test | `data/public_benchmarks/oneadmet_human_plasma_thalf_v2` |
| `train_oneadmet_public_thalf.py` | 仅用已标注的公开开发集作嵌套 scaffold OOF 和一次 validation；训练入口不能读取 source-test 标签 | `models/public_thalf/oneadmet_human_plasma_v1` |
| `freeze_oneadmet_public_thalf_candidate.py` | 核验已选 public ExtraTrees 可重载且 source-test 仍盲态后，冻结候选；之后才允许独立 source-test evaluator 开标签 | `models/frozen/Thalf__human__plasma_public/oneadmet_v1_et_v1` |
| `evaluate_oneadmet_public_thalf_source_test.py` | 核验冻结候选、盲态成员和原始 OneADMET 哈希后，一次性评分作者 source-test；不允许重新拟合、选择或校准 | `results/benchmarks/oneadmet_human_plasma_source_test_v1` |
| `build_pksmart_primary_source_review_batch.py` | 将已定位原文的 PKSmart 线索转为标签盲态资格审核批次；二级去电荷映射只用于检索，不能单独成为正式身份结论 | `results/analysis/pksmart_primary_source_review_batch_v1` |
| `lock_pksmart_primary_eligibility.py` | 只锁定原文支持的人体 direct-IV、母体血浆和 terminal-phase 资格证据；数值必须在后续独立提取锁才可出现 | `results/analysis/pksmart_primary_source_eligibility_lock_v1` |
| `lock_pksmart_primary_extraction.py` | 锁定原文人群级数值和预先规定的单分子正式聚合；同文多个人群不得重复放大正式外部评分分母 | `results/analysis/pksmart_primary_source_extraction_lock_v1` |
| `build_pksmart_primary_formal_addition.py` | 将已锁定的 PKSmart 一手来源资格与聚合数值转成正式注册表增量及身份索引；不使用 PKSmart CSV 标签 | `results/analysis/pksmart_primary_source_formal_addition_v1` |
| `download_pmc_scanned_pages.py` | 从已保存的 PMC 正文 HTML 提取并校验官方 CDN 扫描页，供旧文献逐页审核 | 用户指定证据目录 |
| `merge_external_validation_decisions.py` | 校验 accepted/rejected/pending 证据字段并生成不可覆盖的新注册表版本；默认追加新裁定，`--replace-existing` 可在候选索引复核下修订已有裁定并保持行数、顺序及身份字段不变 | `results/analysis/thalf_external_validation_decisions_v22.csv` |

配套模块：`pipeline_common.py`（所有入口共享自检和产物协议）、`endpoint_rules.py`（版本化分类/换算规则）、`neural_models.py`（Res-MLP/D-MPNN）。`tests/test_pipeline.py` 是合成夹具功能测试，独立于正式训练数据。

### 外部数据库辅助层

```bash
python scripts/collect_pkdb_iv_candidates.py --check-only
python scripts/collect_pkdb_iv_candidates.py \
  --output data/external/pkdb_iv_candidates_v2

python scripts/prepare_oneadmet_auxiliary.py --check-only
python scripts/prepare_oneadmet_auxiliary.py \
  --output data/external/oneadmet_auxiliary_v3

python scripts/build_oneadmet_pk_splits.py \
  --source data/external/oneadmet_auxiliary_v2 \
  --output data/external/oneadmet_pk_splits_v2

python scripts/preprocess_pkdb_candidates.py --check-only
python scripts/preprocess_pkdb_candidates.py \
  --snapshot data/external/pkdb_iv_candidates_v1 \
  --output data/external/pkdb_preprocessed_v2

python scripts/build_pkdb_external_review_batch.py --check-only
python scripts/build_pkdb_external_review_batch.py \
  --source data/external/pkdb_preprocessed_v1 \
  --mapping-registry results/analysis/pkdb_structure_mapping_decisions_v1.csv \
  --document-registry results/analysis/pkdb_document_mapping_decisions_v1.csv \
  --output results/analysis/pkdb_external_review_batch_v2

python scripts/merge_external_validation_decisions.py \
  --base results/analysis/thalf_external_validation_decisions_v8.csv \
  --additions results/analysis/thalf_external_validation_decisions_additions_v9.csv \
  --candidate-index results/analysis/thalf_external_validation_queue_v4/candidate_records_blinded.csv \
  --output results/analysis/thalf_external_validation_decisions_v9.csv

# 修订已有裁定时必须提供候选索引；原注册表仍不可覆盖
python scripts/merge_external_validation_decisions.py \
  --base results/analysis/thalf_external_validation_decisions_v11.csv \
  --additions results/analysis/thalf_external_validation_decisions_revisions_v12.csv \
  --candidate-index results/analysis/thalf_external_validation_queue_v4/candidate_records_blinded.csv \
  --replace-existing \
  --output results/analysis/thalf_external_validation_decisions_v12.csv
```

PK-DB 抓取保留官方响应和哈希。若生产端 `outputs.csv` 为空，脚本只发布 human+IV 文献候选，禁止据此声称已获得 terminal half-life 标签。OneADMET 的全部 1,533 个任务都进入任务注册表；仅 PK 相关任务展开为辅助长表。`human_plasma_half_life_candidates_blinded.csv` 刻意不含数值，必须追溯逐条来源并确认人体、IV、系统血浆和终末相后，才能进入独立验证注册表。OneADMET 原始 `Set` 只作为作者划分保存，不自动替代本项目的母体/骨架隔离划分。历史外部注册表的 `source_doi`/`primary_source_doi` 列兼容保存“规范来源标识”：优先为 DOI；原文和 PubMed 都未提供 DOI 的旧文献才使用明确的 `pmid:<id>`，绝不伪造 DOI。

## 2. 执行前准备

```bash
cd /home/shahab/MTL-model-4-1
conda activate oneadmet
```

使用项目已有环境。核心依赖为 RDKit、NumPy、pandas、SciPy、scikit-learn、joblib、tqdm、threadpoolctl；选用 XGBoost/LightGBM/CatBoost/PyTorch 时才检查对应依赖。默认基线为常数模型、Ridge 和 RF，默认 CPU、4 线程；GPU 算法须显式传入 `--device cuda`，并先运行 `check_gpu_boosters.py`。

单任务训练入口支持 `--feature-set ecfp4_rdkit2d|ecfp4|rdkit2d|mechanism2d`。默认值 `ecfp4_rdkit2d` 保持既有模型行为；每个模型保存自己的特征集合、描述符顺序和特征维数，推理时按同一模式重建特征。`mechanism2d` 是预先固定的 16 项 MW、疏水性、极性、氢键、电荷、柔性和环/杂原子描述符，不在 validation/test 上筛选。

每个入口调用同一 `startup_self_check`，以显式异常检查缺失文件、已有输出、契约/版本和前序完成标记；日志保存在 `results/pipeline_logs`。长操作显示进度条。阶段先写临时目录，全部成功后发布 `complete.json`，下游会校验产物 SHA-256。

输入必须已存在：

```text
data/processed/chembl_train.csv
data/processed/chembl_val.csv
data/processed/chembl_test.csv
data/raw/chembl_37/chembl_37_sqlite/chembl_37.db
```

旧人源子表可用于一致性诊断，但不是新标签真相。原始数据库必须含标准 ChEMBL 的 activities、assays、compound_structures、molecule_dictionary、docs、assay_parameters 表及所用字段。

## 3. 批次 A：按顺序执行数据程序

### A1. 来源审计

先检查路径和数据库结构，再正式执行：

```bash
python scripts/audit_endpoint_sources.py --check-only
python scripts/audit_endpoint_sources.py
```

输出重点：

- `molecule_mapping.csv`：按完整 standard InChIKey 的映射；未匹配/多归属记录明确标记。
- `source_records.csv`：activity/assay/doc、原物种、单位、关系符、assay 原文和参数，以及分类/换算结果。
- `task_audit_counts.csv`：accepted、censored、review 的数量与原因。
- `legacy_contract_issues.csv`：旧 raw/target/mask 不一致计数。

审计不会把所有带 CL 字样的测定都当成同一端点。`mL/min/kg` 不换成微粒体 `µL/min/mg`；`mL/min/g` 的 g 未说明蛋白材料时留在 review。PPB 到 fu 的补数转换同步反转不等式方向。未知物种、未知关系符、fu=0/1、来源质量标记和不明确的实验体系不进入默认精确回归。

若只想试跑：

```bash
python scripts/audit_endpoint_sources.py --limit 100 --output /tmp/dmpk_audit_trial
```

`--limit` 产物明确标记 partial，数据重建程序会拒绝消费，不能把试跑当完整数据。映射使用保守的完整 InChIKey，未做模糊结构匹配；未匹配记录不会自动强行关联。当前规则不从原文猜测体重、生理缩放、酸碱常数或检测限。

### A2. 重建任务数据

```bash
python scripts/rebuild_endpoint_datasets.py --check-only
python scripts/rebuild_endpoint_datasets.py
```

输出重点：

- `source_long.csv` 保留全部审计候选来源；`review_records.csv` 与 `censored_records.csv` 分开保存。
- `task_records.csv` 与 `tasks/<task_id>.csv` 只包含 accepted 精确测定。
- `task_registry.json` 固定任务物种、体系、单位和数学变换。
- `task_counts.csv` 报告每任务/每集合的记录数和独立分子数。

重复测量只在相同“分子 × 任务 × assay × doc”内用中位数聚合，保留活动 ID、源数量、最小/最大值。不同研究/体系保留独立记录，不将旧全局聚合值伪装成人体测量。训练时同一分子的多条记录合计权重相等。

raw 值是规范单位下的观测，`target_value` 是未标准化的变换值，`mask=1` 表示已确认精确标签。fu 内部值使用真实 logit，保留小于 0.01 的差异；CLint/Papp 使用 log10；F（审计支持，训练入口后置）使用原空间。z-score 均值/方差不在此阶段全局拟合，而在每个训练范围内单独学习。

review/censored 表不是被删除的数据。当前训练实现是精确标签基线，没有实现删失似然或把 `<LLOQ` 当成精确零。若任务容量不足，应根据来源文献改进明确的规则并重跑新版本，不能直接把 review 全部改为 accepted。

### A3. 固定骨架与交叉验证折

```bash
python scripts/build_split_manifest.py --check-only
python scripts/build_split_manifest.py --folds 5 --seed 2026
```

原 train/val/test 成员不移动。结构分组采用 RDKit FragmentParent、去电荷、规范互变异构体、去立体信息后的 Murcko 骨架；无环分子按规范母体分组。若相同母体/骨架跨集合，相关成员全部在新视图中排除，理由写入 `excluded_structures.csv`，不根据标签选择保留哪一侧。

RDKit 对个别结构抛出 `RuntimeError`/`ValueError` 时，记录完整 SMILES 与错误到 `structure_failures.csv`，并设置 `eligible=False`、`fold_id=-1`。不退回未经规范化的结构参与分组，因此这些记录不会进入训练。

结构计算每 25 条保存一次检查点，默认位于 `data/processed_v2/_cache/structure_groups.sqlite`，与最终 `splits` 目录分离。中断后直接重跑同一命令，会复用已保存的成功/失败结果；缓存按 RDKit 版本及分组函数代码隔离。`--retry-failed` 可重试失败记录，`--cache-path` 可指定另一缓存。旧版在内存中计算、未保存的进度无法追溯恢复。只有已发布的最终输出目录才会阻止同路径重跑。

训练折统一生成一次并存入 `split_manifest.csv`。所有端点与物种按同一分子/骨架读取 fold；端点脚本不会调用随机 `train_test_split`。查看 `task_fold_counts.csv`，确认每个任务有足够有效训练骨架和验证/测试覆盖；环骨架分组不保证隔离所有更宽泛的类似物系列。

### A4. 版本化证据增量核验

在新 `processed_vN` 的三个阶段都完成后，使用 `verify_dataset_version.py` 对已发布基准和候选版本进行只读核验。该命令逐一验证 audit/datasets/splits 的完成标记与文件哈希，要求两个 `split_manifest.csv` 的 SHA-256 一致，并按任务与固定集合报告记录/分子增量。`--expect-delta` 可将预期增量变成显式断言；任何不一致都会以非零退出，不应继续训练。

```bash
python scripts/verify_dataset_version.py \
  --baseline-dir data/processed_v7 --candidate-dir data/processed_v8 \
  --expect-delta Thalf__human__terminal_iv:train:1:1
```

## 4. 批次 B：底座自检与第一个 fu 基线

```bash
python scripts/dmpk_toolkit.py check --task fu__human__plasma
python scripts/train_fu.py --check-only
python scripts/train_CLint.py --check-only
python scripts/train_Papp.py --check-only
python scripts/train_CL.py --check-only
python scripts/train_VDss.py --check-only
python scripts/train_Thalf.py --check-only
```

如果某任务不存在或覆盖的训练折不足，自检会给出错误与可用任务，不会启动长时间训练。`dmpk_toolkit.py` 是公共模块，**不需要作为独立训练任务先运行**；这里的 `check` 用于验证接口。

首先运行 fu，检查真实数据规模、耗时和产物：

```bash
python scripts/train_fu.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
```

默认额外运行常数基线。`--trials 3` 表示每算法最多 3 套候选参数；在外层固定折内用其余固定折进行内层调参。每个外层验证分子的标签不参与其模型的参数选择、特征插补/标准化或目标标准化。最后在全部合格 train 中重新调参/拟合最终模型。

使用的结构特征为 2,048 位 ECFP4 + 当前 RDKit 有名称清单的 2D 描述符；确定性结构计算可覆盖各集合，所有需要拟合的操作只能看训练范围。未使用旧来源不明的 joblib/NPZ 特征缓存。

Ridge 固定使用 LSQR 求解器，适配 ECFP 与描述符组成的高维共线矩阵。分数端点在反 logit 前提升为 float64，并裁剪到浮点可表示的开区间 `(0,1)`，避免 float32 饱和为精确 0/1 后导致预测文件转换失败。

正值端点的反 log10 变换也带有数据派生的数值保护：预测超出训练变换范围 8 个训练标准差时裁剪，并限制在 `[-300, 300]` 的可计算区间；模型对象记录 `inverse_clip_count`。这只防止 `10**z` 溢出，不代表该模型在极端化学空间中具有可靠外推能力，验证报告应检查该计数。

## 5. 批次 C：CLint 与 Papp，可受控并行

fu 的运行流程与产物验证后，执行：

```bash
python scripts/train_CLint.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
python scripts/train_Papp.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
```

这些微观单任务之间没有模型前置依赖，可以在两个终端同时执行；以上连续写法则是顺序运行。建议首次使用顺序运行，确认峰值内存后再开两个 CPU 作业，每作业 4 线程。不要同时让多个 GPU 作业争用 12 GB 显存。

其他物种或 fu 基质需显式选择，例如：

```bash
python scripts/train_fu.py --species rat --system plasma --check-only
python scripts/train_CLint.py --species human --system microsome_unbound --check-only
```

只有审计产生了相应任务，才能训练；不自动混合动物和人、血浆和血清、apparent 与 unbound CLint。

## 6. 批次 D：人体宏观 CL 与 VDss 的直接基线

当前严格来源视图中，CLint 只有 20 个训练分子，因此它的 `diagnostic_v1` 只能用于检查流程，不能作为 CL 的级联先验。先训练不依赖 CLint 的直接 CL 与 VDss 基线：

```bash
python scripts/train_CL.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
python scripts/train_VDss.py --algorithms ridge,rf --trials 3 --threads 4 --run-name baseline_v1
```

保持 test 集冻结。完成后检查 `metrics.json` 和 `selection.json`；只有扩充并重新审计 CLint、F 与半衰期数据后，才开始 CLint→CL、CL/VDss→F 或 PBPK 级联。

## 7. 批次 E：Thalf 直接诊断基线

人体 Thalf 只有 43 个有效训练分子。先运行直接基线以确认标签、固定折和可达到的误差尺度；它不是可报告的最终模型，也不启用 test 评估：

```bash
python scripts/train_Thalf.py --algorithms ridge,rf --trials 3 --threads 4 --run-name diagnostic_v1
```

随后运行 `train_Thalf_cascade.py`。它为每个 Thalf 外层折重新拟合 CL、VDss 上游模型，排除该外层折；用于外层训练行的先验还排除该行自身折。不得把 CL/VDss 全训练模型对 Thalf 训练分子的预测直接作为特征。

```bash
python scripts/train_Thalf_cascade.py --check-only
python scripts/train_Thalf_cascade.py --threads 4 --run-name cascade_diagnostic_v1
```

## 8. 批次 F：大鼠跨物种先验

人体 F 只有 15 个训练分子，人体 Thalf 也只有 43 个；在它们上面继续调参不会产生稳健结论。先在样本量充足、体系明确的大鼠任务上训练单任务先验。每条命令读取同一固定骨架划分，任务之间可并行，但首次建议先跑 CL 与 VDss：

```bash
python scripts/train_else.py --endpoint CL --species rat --system systemic_iv --check-only
python scripts/train_else.py --endpoint VDss --species rat --system steady_state_iv --check-only

python scripts/train_else.py --endpoint CL --species rat --system systemic_iv --algorithms ridge,rf --trials 3 --threads 4 --run-name prior_v1
python scripts/train_else.py --endpoint VDss --species rat --system steady_state_iv --algorithms ridge,rf --trials 3 --threads 4 --run-name prior_v1
```

两项成功后，再依次建立 `fu__rat__plasma`、`F__rat__absolute_oral` 和 `Thalf__rat__terminal_iv` 的先验。人源模型不会和动物标签混合训练；后续跨物种程序只消费动物模型的受保护预测。

大鼠 F 与 Thalf 的专用入口如下：

```bash
python scripts/train_F.py --species rat --system absolute_oral --check-only
python scripts/train_Thalf.py --species rat --system terminal_iv --check-only
```

## 9. 批次 G：受保护的大鼠→人跨物种级联

每个人源外层验证折都会重新拟合大鼠模型，并排除该折；供人源外层训练行使用的大鼠预测还排除该行自身折。因此大鼠模型不可能见过目标行所在的统一结构折。大鼠预测作为单独的软特征拼接，物种标签和物种测定值不在同一监督目标中混合。

支持的同体系任务为 `fu/plasma`、`CL/systemic_iv`、`VDss/steady_state_iv`、`F/absolute_oral`、`Thalf/terminal_iv`。Papp 缺少匹配大鼠 Caco-2 任务；CLint 的人/大鼠严格数据均不足，两个端点不会被自动迁移。

先执行容量充分的 fu、CL、VDss；三项可在独立终端并行，每项最多 4 CPU 线程：

```bash
python scripts/train_cross_species_fu.py --check-only
python scripts/train_cross_species_CL.py --check-only
python scripts/train_cross_species_VDss.py --check-only

python scripts/train_cross_species_fu.py --threads 4 --run-name cross_species_diagnostic_v1
python scripts/train_cross_species_CL.py --threads 4 --run-name cross_species_diagnostic_v1
python scripts/train_cross_species_VDss.py --threads 4 --run-name cross_species_diagnostic_v1
```

再顺序执行 Thalf，并最后运行 F。人体 Thalf（43 个训练分子）与 F（13 个）只生成诊断结果，不解冻 test；严格划分后 F 没有合格 val 行，因此仅报告受保护 OOF，不伪造验证指标：

```bash
python scripts/train_cross_species_Thalf.py --check-only
python scripts/train_cross_species_Thalf.py --threads 4 --run-name cross_species_diagnostic_v1

python scripts/train_cross_species_F.py --check-only
python scripts/train_cross_species_F.py --threads 4 --run-name cross_species_diagnostic_v1
```

## 10. 批次 H：多任务共享表征与 MMoE

`train_multitask.py` 从测定级长表建立 molecule × task 标签矩阵。任务是 `endpoint × species × system`，不使用旧来源不明宽表。重复测定先按分子/任务取中位数；缺失标签只作为张量占位，mask=0，不参与监督损失、目标标准化或指标。

默认选择有效训练分子不少于 40、覆盖至少 3 个统一 fold 的任务；当前自检选择 29 个任务、22,306 个分子和 17,903 个训练分子。每个外层折中，特征插补/缩放及每个任务的目标变换统计量均只由其余训练折拟合。任务损失在 batch 内按“有标签任务”平均，避免大任务吞没小任务。

先运行共享 MLP 基线，再独立运行 dense MMoE；二者均只报告逐任务 OOF 和冻结验证指标，不能以不同量纲任务的全局平均选择赢家：

```bash
python scripts/train_multitask.py --check-only

python scripts/train_multitask.py --architectures shared --width 192 --epochs 80 --batch-size 64 --threads 4 --run-name shared_diagnostic_v1
python scripts/train_multitask.py --architectures mmoe --width 192 --experts 4 --epochs 80 --batch-size 64 --threads 4 --run-name mmoe_diagnostic_v1
```

如果 CUDA 自检通过，可将最后两条命令中的 `--device cuda` 加入；同一 GPU 上只能运行一个多任务作业。对每个人源任务，只有当 MTL 在 OOF 和冻结验证均不劣于对应 STL 时，才进入最终候选；否则保留 STL。当前首版不实现 Top-2 稀疏门控、Kendall 权重或在线物理闭环，避免在未验证共享收益前增加不可分辨的复杂度。

## 11. 扩展候选算法

已实现的候选为：`dummy,ridge,rf,extratrees,xgboost,lightgbm,catboost,resmlp,dmpnn`。

树模型扩展可单独运行，避免第一轮直接开启大预算：

```bash
python scripts/train_fu.py --algorithms extratrees,xgboost,lightgbm,catboost --trials 10 --threads 4 --run-name trees_v1
```

神经候选示例（确认 CUDA 可用后）：

```bash
python scripts/train_fu.py --algorithms resmlp,dmpnn --trials 3 --epochs 80 --batch-size 32 --threads 4 --device cuda --run-name neural_v1
```

Res-MLP 与 D-MPNN 为此底座的小型 PyTorch 实现。D-MPNN 使用有向键消息传递、反向边排除、分子聚合及数值特征支路；不是对 Chemprop 训练结果的复现。模型保存 CPU state_dict，可在无 GPU 的机器推理。

当前神经训练使用指定的固定 epoch 和 Huber 损失，不用外层验证/测试做 early stopping。`epochs` 属于需预先选择/内层验证的训练配置。当前没有预训练 SMILES/BRICS、多任务、机理损失、smearing 或融合模块。嵌套调参会显著放大运行量，先基线后扩大预算。

## 12. 输出检查与冻结测试

每个端点运行目录包含：

```text
complete.json                 # 产物 hash、代码 hash、数据版本、算法与 seed
feature_schema.json           # 指纹参数、描述符名字/顺序、RDKit 版本
tuning_history.json           # 外层范围内及最终训练范围的候选得分
metrics.json                  # 各算法 OOF 和 val；默认无 test 指标
selection.json                # 依据训练嵌套 OOF 选择的算法及注意事项
environment.json              # 实际包版本
<algorithm>/
    model.joblib              # 最终 train 模型、变换器和前处理
    lineage.json              # 训练分子/骨架、超参、任务定义
    oof_predictions.csv       # 带 row_id、fold、物理/变换空间的外层预测
    val_predictions.csv       # 有合格 val 标签时生成
    fold_<k>/model.joblib
    fold_<k>/lineage.json
```

程序会重载每个最终模型并核对预测。指标同时报告原空间和变换空间误差；fu 优化分子等权原空间 MAE，CLint/Papp 优化分子等权 log10 RMSE；另报低 fu 误差、倍数误差和相应计数。没有样本的 val 不伪造结果，常数标签下 R² 记为 null。

算法选择基于 train OOF，所选算法的同一 OOF 得分存在选择偏差；val/test 用于评估泛化。默认不计算测试性能。方案冻结后可直接评估保存模型，无需重训：

```bash
python scripts/dmpk_toolkit.py evaluate-test --run-dir models/stl/fu__human__plasma/baseline_v1 --output results/test_evaluation/fu_baseline_v1
python scripts/dmpk_toolkit.py evaluate-test --run-dir models/stl/CLint__human__microsome/baseline_v1 --output results/test_evaluation/clint_baseline_v1
python scripts/dmpk_toolkit.py evaluate-test --run-dir models/stl/Papp__human__caco2_ab/baseline_v1 --output results/test_evaluation/papp_baseline_v1
```

该命令使用运行目录中按训练 OOF 选定的模型，验证数据版本后输出独立测试报告，不更新权重和选择结果。端点训练入口另有 `--evaluate-test`，仅适用于方案已冻结时的新运行，不建议探索阶段开启。

任意新分子推理（输入需要 `smiles` 或 `Canonical_SMILES` 列）：

```bash
python scripts/dmpk_toolkit.py predict --model models/stl/fu__human__plasma/baseline_v1/rf/model.joblib --input new_molecules.csv --output results/new_fu_predictions.csv
```

普通推理结果标为 `inference_not_oof`，不能当作训练级联先验。输出含规范单位及物理/变换空间预测。

## 8. 为下一阶段级联预留的保护范围

例如，人体宏观模型的外层保护折是 0，微观模型可执行：

```bash
python scripts/train_fu.py --algorithms rf --trials 3 --exclude-fold 0 --run-name cascade_outer0
```

该运行所有拟合均排除统一 fold=0，内部 OOF 又在其余折中交叉拟合，保存 `excluded_fold_predictions.csv` 及祖先训练组。这是正确嵌套级联所需的子范围。实际宏观脚本仍需对所有端点/物种一致使用该范围，并为缺失微观标签的分子生成对应范围的预测；本次尚未自动构建完整宏观先验矩阵。

不能把 `baseline_v1` 的全 train 模型回代 train 当成先验；也不能将按全局 OOF 选出的算法误称为对每个外层折都独立选择。后续级联需要在保护范围内选择算法/超参或预先固定算法，再审计完整祖先链。

## 9. 本轮验证

2026-09-09：13 项功能测试全部通过，7 个入口的 `--help` 均通过，真实数据库启动结构自检通过。原 `data/processed` 中 16 份被登记 CSV 的 SHA-256 与第一阶段核查一致。详见 [验证记录](/home/shahab/MTL-model-4-1/scripts/VALIDATION.md)。

```bash
python -m unittest discover -s scripts/tests -v
```

测试夹具在独立临时目录中创建合成 SMILES/SQLite，验证：单位量纲、PPB 删失方向、低 fu 保真、结构分组、旧/新产物协议、固定折嵌套训练、所有候选算法的拟合与重载、三个训练入口、保护外层折、普通推理和冻结测试评估。

本轮另用真实数据库前 24 个分子进行了只读审计试跑，读取 52 条候选测定；不把这些非随机抽样的通过率当作全库质量比例。未执行真实数据全量重建、全量骨架重算或正式模型训练，测试指标不代表科研性能。

**实际执行顺序：A1 审计 → A2 重建 → A3 清单 → B 自检与 fu 基线 → C CLint/Papp → D CL/VDss 直接基线 → E Thalf 直接诊断 → 受保护级联 → 候选扩展 → 方案冻结后的测试。**

## VDss 来源泛化压力测试（2026-09-17）

`build_vdss_source_generalization_protocol.py` 审计人源 VDss 的 parent–document 图并生成来源可行性图。由于最大连通分量覆盖 554/562 parents，平衡的连通来源簇五折不可行；脚本注册五组 purged document-group 替代方案。

`run_vdss_source_document_holdout.py` 在每个来源压力折从五物种训练输入同时删除 evaluation documents、parents 与 scaffolds，固定比较 corrected Stage-A、human-only neural 和 species-conditioned joint。它只产生来源敏感性结果，不授权模型选择、validation 或 test。

正式 GPU 后台入口：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_vdss_source_document_holdout_v1.sh
```

## 配置化跨物种迁移（2026-09-17）

`build_cross_species_transfer_protocol.py` 按端点冻结 human/animal tasks、单位、标签变换、Stage-A 对照、逐折资源、训练预算和来源图可行性。`run_cross_species_transfer_traincv.py` 从该 protocol 读取任务配置，复用经过 VDss 验证的 train-fold-only 标准化、parent/scaffold 清除、task-balanced 训练和评价标签生命周期。

当前 CL 与 terminal t½ 正式 GPU 入口：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_cl_cross_species_transfer_traincv_v1.sh
bash scripts/run_stl_long_task_background.sh scripts/run_thalf_cross_species_transfer_traincv_v1.sh
```

两项必须分别运行，不能同时写入同一输出目录。terminal t½ protocol 只接受 `terminal_iv / h / log10` 的 human、dog、monkey、mouse、rat 五任务；smoke 为工程检查，正式比较仍需完整 45-model GPU 结果。

## 10. Papp 来源审计与 MTL 任务均衡采样

Papp 审计只读 `datasets` 中的任务记录及其 `source_long.csv` 谱系。默认审计 train/val，避免在模型选择前读取测试标签；输出每条高值记录的原始单位、规范单位、换算规则、测定、文献和方向证据：

```bash
python scripts/audit_papp_provenance.py --check-only
python scripts/audit_papp_provenance.py --threshold 10000 --output results/analysis/papp_provenance_audit_v2
```

审计不自动删除或修正记录。若核查确认单位、方向或来源错误，必须建立新的 audit/datasets/splits 版本，再重跑受影响的 Papp 模型；若未确认错误，保留当前版本及审计报告。

`train_multitask.py` 的默认 `molecule_uniform` 是已完成 MTL 的可复现基线。`task_balanced` 在每个 batch 为各任务抽取有标签锚点并以均匀分子补足，缓解稀疏任务在 molecule-uniform batch 中被低频看见的问题；它不会更改 split、标签或测试集。

```bash
python scripts/train_multitask.py --check-only --batch-sampler task_balanced
python scripts/train_multitask.py --architectures shared,mmoe --batch-sampler task_balanced \
  --seed 2026 --width 192 --experts 4 --epochs 80 --batch-size 64 --threads 4 \
  --run-name balanced_seed2026_v1
```

随后使用 `--seed 2027` 和 `--seed 2028`，每次使用不同 `--run-name`。最终比较须按相同架构、相同 seed 汇总 protected OOF 的均值和标准差；测试集在模型锁定前保持冻结。

## 11. Papp 原文单位校正：新版本审计、数据集与划分

原文核查确认 Papp 的单位错误只发生在部分 DOI，不能全局修改 `cm/s` 转换。`audit_endpoint_sources.py` 可读取裁定表：确认 `10^-6 cm/s` 的 DOI 会选择性保留原数值；也识别 ChEMBL 的 `10'-6 cm/s` 拼写变体。`unresolved` 默认进入 review，因而不会训练。旧的 `processed_v2` 和已完成模型不可修改。

```bash
python scripts/audit_endpoint_sources.py --check-only \
  --papp-decision-registry results/analysis/papp_provenance_decisions_v2.csv

python scripts/audit_endpoint_sources.py \
  --papp-decision-registry results/analysis/papp_provenance_decisions_v2.csv \
  --papp-unresolved-policy exclude \
  --output data/processed_v3/audit

python scripts/rebuild_endpoint_datasets.py \
  --audit-dir data/processed_v3/audit \
  --output data/processed_v3/datasets

python scripts/build_split_manifest.py \
  --datasets-dir data/processed_v3/datasets \
  --cache-path data/processed_v3/_cache/structure_groups.sqlite \
  --output data/processed_v3/splits
```

每个命令都必须成功发布新的 `complete.json` 才能进行下一个。之后重跑 Papp STL 和所有 MTL 种子，并显式传入 `--datasets-dir data/processed_v3/datasets --splits-dir data/processed_v3/splits`；测试集继续冻结。

## 12. 三种子 MTL 正式分析

三个 v3 任务均衡种子完成后运行：

```bash
python scripts/multitask_three_seed_analysis.py --check-only
python scripts/multitask_three_seed_analysis.py \
  --output results/analysis/multitask_pappfixed_v3_three_seed_v1
```

程序验证三个 MTL 运行的数据、划分、代码、任务与采样协议一致，并确认所有输入均未评估 test。输出包括种子级和汇总 CSV、逐分子配对 OOF bootstrap、600 dpi 英文 PNG、模型选择表及同目录英文分析报告。

Thalf v15 描述符三种子复验使用独立汇总入口；它核验六个训练目录及 test 隔离，并输出种子级、集成和逐分子配对 bootstrap 表：

```bash
python scripts/analyze_thalf_descriptor_seeds.py --check-only

python scripts/analyze_thalf_descriptor_seeds.py \
  --bootstrap 10000 \
  --output results/analysis/thalf_v15_rdkit2d_three_seed_v1
```

预注册 Thalf 白盒训练使用 CPU。程序先完成三个配置的 nested OOF 排序，再一次性计算 validation，并输出系数稳定性、GAM 形状函数、外推比例和高相关描述符对：

```bash
python scripts/train_Thalf_whitebox.py --check-only \
  --threads 12 \
  --run-name v15_whitebox_preregistered_v2

python scripts/train_Thalf_whitebox.py \
  --threads 12 \
  --run-name v15_whitebox_preregistered_v2
```

## 13. 冻结候选与一次性测试评估

先发布候选登记。该程序只读取 OOF/validation 结果，并逐个重载三个 Shared MLP 检查预测一致性；不读取测试标签：

```bash
python scripts/freeze_final_candidates.py --check-only
python scripts/freeze_final_candidates.py
```

当前登记将人血浆 fu 固定为 seeds 2026/2027/2028 的 Shared MLP 物理尺度均值集成；Papp、CL、VDss 和 Thalf 固定为预先选定的 STL RF。登记发布后运行一次统一测试：

```bash
python scripts/evaluate_frozen_candidates.py --check-only
MPLCONFIGDIR=/tmp/mtl_matplotlib python scripts/evaluate_frozen_candidates.py \
  --confirm-frozen-test
```

第二条命令不重新拟合，不允许更新选择，输出逐记录预测、完整指标 JSON/CSV、英文 600 dpi observed-vs-predicted 图和指标表。Thalf 仅有 2 个测试分子，程序将其明确标为探索性结果。

## 14. 冻结测试诊断与 Papp 单位敏感性

测试候选发布后，使用同一份不可修改的预测执行分子级 bootstrap、残差校准、ECFP4 最近训练分子适用域和 Papp 三情景单位敏感性分析：

```bash
MPLCONFIGDIR=/tmp/mtl_matplotlib python scripts/analyze_frozen_test.py --check-only
MPLCONFIGDIR=/tmp/mtl_matplotlib python scripts/analyze_frozen_test.py
```

Papp 三种情景为：保留冻结测试原值、将 DOI `10.1016/j.bmcl.2020.127669` 解释为隐含 `10^-6 cm/s`、排除该未解决来源。后两者是测试后敏感性分析，不能用于改变模型选择。输出目录为 `results/final/frozen_test_diagnostics_v1`，包括英文 600 dpi PNG、分子级明细、CSV 和分析报告。

## 15. 冻结候选推理与全局解释

面向新结构的推理必须使用冻结候选登记，而非任意单模型。输入 CSV 需要 `smiles` 或 `Canonical_SMILES` 列；输出为长表，每个输入结构对应每个登记端点一行：

```bash
python scripts/infer_frozen_candidates.py \
  --input new_molecules.csv \
  --output results/new_frozen_predictions.csv
```

结果包含物理/变换空间预测、规范单位、证据状态、最近训练分子的 ECFP4 Tanimoto 和适用域标志。`outside` 表示相似度低于预先固定的 0.30 阈值，预测仍会返回，但不应脱离该警告解读。该入口只加载候选登记的模型；fu 自动使用三个 Shared MLP 的物理尺度平均。

冻结 STL RF 的全局分裂重要性可重新生成，且不读取测试标签：

```bash
python scripts/report_frozen_interpretability.py
```

输出 `results/final/frozen_model_interpretability_v1`。重要性是描述性而非因果归因；相关描述符和 ECFP 位会相互分摊重要性。fu 的冻结模型是 Shared MLP 集成，故不会被误报为 RF 特征重要性。

## 16. 人源 F、CLint 与 Thalf 的证据扩充队列

下列程序将审计中保留的 review 记录组织为按 DOI、文献和 assay 聚合的人工核查队列。它不会更改数据、划分或模型：

```bash
python scripts/audit_human_evidence_expansion.py
```

输出 `results/analysis/human_evidence_expansion_v1`。只有在原始来源明确证明绝对口服 F 的 IV 参照、终末 IV 半衰期，或以 mg 蛋白为归一化基准的微粒体 CLint 后，才可通过版本化规则将记录写入新的数据集。

## 17. Thalf 已接受来源专项审计

人工 `rejected` 裁定会把精确匹配的 activity 标记为 `excluded`，清空规范标签，并保留原始值和裁定来源供追溯。`accepted` 既可提升保守 review，也可把已有 automatic accepted 转为原文确认。接受或拒绝都必须填写原文页码/表格和证据说明。

以下程序只审计 train/validation 中已接受的人源 terminal-IV Thalf 来源，不报告测试记录或测试指标：

```bash
python scripts/audit_thalf_accepted_sources.py --check-only \
  --audit-dir data/processed_v11/audit \
  --output results/analysis/thalf_accepted_sources_v11_r2

python scripts/audit_thalf_accepted_sources.py \
  --audit-dir data/processed_v11/audit \
  --output results/analysis/thalf_accepted_sources_v11_r2
```

输出记录级 `automatic_accepted_records.csv`、文献级 `automatic_accepted_documents.csv` 和人工/自动来源汇总 `summary.csv`。风险标志仅用于安排原文审核顺序，不能自动生成拒绝裁定。

## 18. Gate 1B 冻结 SMILES embedding

算法/provider 协议登记全部候选家族，但将“保留证据”与“授权计算/验证”分离：

```bash
conda run --no-capture-output -n oneadmet \
  python scripts/build_algorithm_landscape_and_smiles_provider_protocol.py --check-only

conda run --no-capture-output -n oneadmet \
  python scripts/build_molformer_parent_embedding_cache.py --check-only
```

MoLFormer 的固定 revision 已通过 32-parent GPU/CPU smoke。全量 7,939-parent 缓存使用专用 allowlist 后台入口：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_molformer_parent_cache_v1.sh
```

它生成 label-free、parent-keyed、768 维 float32 cache；不读取 validation/test，不拟合模型。检查状态时必须使用启动输出中的实际日志路径：

```bash
bash scripts/check_stl_long_task.sh results/pipeline_logs/stl_long_tasks/run_molformer_parent_cache_v1_YYYYMMDD_HHMMSS.log
```

缓存发布后，冻结表征使用 Ridge、ExtraTrees、LightGBM 各两个预注册配置执行相同六端点、相同五折的 train-CV 探针。正式任务使用后台入口：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_gate1b_frozen_smiles_probe_v1.sh
```

限定 parent 的运行仅为工程 smoke，`partial=true` 且禁止模型选择。正式结果也必须先经过与 Stage A leader 的逐折增量审计，不能直接打开 fixed validation。

首轮 frozen-SMILES 审计后，后续 train-CV 使用不含 fixed-validation target 的模型入口，并在每折训练 parent 上单独拟合目标标准化：

```bash
conda run --no-capture-output -n oneadmet python scripts/build_stl_train_only_protocol.py --check-only
conda run --no-capture-output -n oneadmet python scripts/run_gate1b_frozen_smiles_adaptation.py \
  --max-parents-per-task-fold 8 --check-only
```

正式 PCA‑Ridge/MLP 适配与 matched structure control 使用：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_gate1b_frozen_smiles_adaptation_v1.sh
```

该任务仍为 train-CV-only；fixed validation/test 保持关闭。

### fu 物理标签与有界输出诊断

`fu` 的原始记录在 logit 空间聚合时，`sigmoid(mean(logit(fu)))` 不一定等于物理 `fu` 的算术均值。因此该独立诊断保留原 Stage-A logit ExtraTrees 对照，但以 record-level inverse-logit 后、parent-level arithmetic mean 的物理标签统一评分；两个神经候选均用 sigmoid 有界输出和 whole-scaffold 内部 early stopping 选择训练 epoch，再以该 epoch 在完整外层训练折重训。

```bash
conda run --no-capture-output -n oneadmet python scripts/run_gate1b_fu_physical_alignment.py \
  --max-parents-per-task-fold 16 --max-epochs 8 --patience 3 \
  --output results/benchmarks/gate1b_fu_physical_alignment_smoke_v3

bash scripts/run_stl_long_task_background.sh scripts/run_gate1b_fu_physical_alignment_v1.sh
```

正式任务只读取 `stl_train_only_protocol_v1`；它是新的 train-CV 诊断，不能覆盖已经锁定的 Stage-A 结果，亦不打开 fixed validation/test。

### fu 受控跨物种迁移

正式协议固定 human/dog/monkey/mouse/rat plasma `fu`，先逐记录 inverse-logit，再按 parent 聚合；模型采用 sigmoid 有界输出，训练损失与主指标均为 physical MAE。工程 smoke 和完整配置预检通过后，正式 CUDA 任务使用：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_fu_cross_species_transfer_traincv_v2.sh
```

任务包含 5 folds × 3 seeds × 3 routes；fixed validation/test 始终关闭。完成后应先检查 `results/benchmarks/fu_cross_species_transfer_traincv_v2/complete.json`，再做统一配对分析，不得根据训练中间输出选择路线。protocol/smoke v1 仅保留为元数据文字不一致的历史工程记录；推荐入口为 v2。

统一分析器读取 protocol 中登记的主指标：CL/Thalf 使用 transformed RMSE，fu 使用 physical MAE，并把 corrected Stage-A 转为相同的逐记录 inverse-logit、parent 聚合物理尺度后再做配对 bootstrap。fu 推荐正式分析为 `results/analysis/fu_cross_species_transfer_traincv_analysis_v2`；v1 仅因来源图说明沿用旧文字而保留为历史。

### 多端点 shared/private 有限消融

推荐 protocol v2 与 smoke v2 使用 45 个任务、六个人体 outer-CV 评价头，并对全部任务应用六个评价集 parent/scaffold 的联合清除。fully-private 与 shared/private 两路线使用相同的逐任务 train-fold-only 特征 scaler；每 epoch 先保证所有 active tasks 覆盖，再按冻结 tier 权重追加采样。fu/F 使用有界物理输出，human-F 禁止选择。

正式 CUDA 任务：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_multitask_shared_private_traincv_v2.sh
```

完成标记为 `results/benchmarks/multitask_shared_private_traincv_v2/complete.json`。正式结果必须逐端点同时比较 shared、fully-private 与 corrected Stage-A；不得使用 smoke 或 human-F 选择架构。

正式 30-model 任务已完成，推荐分析为 `results/analysis/multitask_shared_private_traincv_analysis_v2`。分析使用六端点相同 outer-fold parents、2,000 次按折分层 paired bootstrap，并将 corrected Stage-A 作为同评价集强参考。单一全共享 encoder 为非晋级阴性消融；后续先做 train-only 任务亲和性/梯度冲突诊断，不直接启动无边界 MMoE 搜索。

### Gate 2A 多任务梯度亲和性

推荐 protocol v1 与 smoke v2 使用冻结的15个 shared checkpoints和 retained outer-train labels，按任务计算共享 encoder 梯度 cosine 与范数；不更新模型，不读取评价、fixed validation或test标签。正式任务使用：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_multitask_gradient_affinity_v1.sh
```

输出 `results/analysis/multitask_gradient_affinity_v1`。正式15-context分类规则为至少10/15同方向；smoke只有一个context，不得用于架构选择。

### Gate 3 B6 实验理化辅助数据可行性审计

在任何logP/logD cohort或模型构建前，先执行train侧来源、结构、重复、受保护集合和逐outer-fold purge审计：

```bash
conda run --no-capture-output -n oneadmet python scripts/audit_physchem_auxiliary_feasibility.py --check-only
conda run --no-capture-output -n oneadmet python scripts/audit_physchem_auxiliary_feasibility.py
```

推荐发布目录为 `data/public_development/physchem_auxiliary_feasibility_audit_v3`。输出仅含OneADMET source-train logP/logD标签；source-test值不用于计算或发布，项目fixed validation/test target文件不读取。该阶段即使为`CONDITIONAL_PASS`也只允许冻结`G3_B6_nested_physchem_oof_late_fusion`协议，不允许直接把审计表当作训练cohort。理化辅助head必须排除计算logP派生描述符，PK行只能接收严格nested-crossfit预测及missingness mask，不能使用实测理化值。

### Gate 3 B6 nested-crossfit 协议冻结

```bash
conda run --no-capture-output -n oneadmet python scripts/build_gate3_b6_physchem_protocol.py --check-only
conda run --no-capture-output -n oneadmet python scripts/build_gate3_b6_physchem_protocol.py
```

推荐发布目录为 `data/public_development/gate3_b6_physchem_protocol_v2`。它冻结每个logP/logD辅助任务的两个有限ExtraTrees producer、C0 corrected Stage-A、C1容量匹配结构晚融合控制、B6 nested experimental-physchem晚融合，以及五层OOF祖先和A1–A7晋级/止损规则。v1的C0行继承了不生效的ridge元字段，仅作历史记录；v2已将其明确为`none`。本阶段只授权后续构建versioned train-only cohort和一个outer-fold bounded smoke，不授权正式训练或fixed validation/test。

### Gate 3 B6 train-only实验理化cohort

```bash
conda run --no-capture-output -n oneadmet python scripts/build_gate3_b6_physchem_cohort.py --check-only
conda run --no-capture-output -n oneadmet python scripts/build_gate3_b6_physchem_cohort.py
```

推荐发布目录为 `data/public_development/gate3_b6_physchem_cohort_v1`。它只接收可行性审计v3的train-side clean pool，按parent算术均值聚合并等权，排除15个计算logP派生描述符，发布8,003-parent ECFP4/RDKit2D缓存、五折辅助scaffold memberships和60个端点特异purge视图。RDKit无法为一个季鏻盐parent计算的10个描述符单元显式保留为NaN，后续只能在当前producer训练parents内拟合中位数插补；缺失状态不作为预测特征。本阶段只授权一个outer-fold工程smoke。

### Gate 3 B6 bounded工程smoke

```bash
conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_smoke.py --check-only
conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_smoke.py
```

推荐产物为 `results/benchmarks/gate3_b6_physchem_smoke_v2`。入口固定 CL outer fold 0，以完整scaffold为单位限制规模，贯通 corrected Stage-A、容量匹配结构控制和B6 nested logP/logD三路线。v2补齐两个meta-model ancestry与全部模型SHA-256；v1只作历史记录。outer-evaluation只发布盲态预测，不计算指标；smoke禁止用于路线选择或性能结论。

### Gate 3 B6 正式train-CV准备与首cell资源pilot

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_gate3_b6_formal_run.py --check-only
conda run --no-capture-output -n oneadmet python scripts/run_gate3_b6_physchem_outer_fold.py \
  --task CL__human__systemic_iv --outer-fold 0 --check-only

bash scripts/run_stl_long_task_background.sh scripts/run_gate3_b6_first_formal_cell_v2.sh
```

推荐授权与资源清单为 `data/public_development/gate3_b6_formal_run_v4`。正式worker固定P1 corrected Stage-A的三种子集成、每个outer-fold独立producer选择和crossfit、逐模型SHA-256、fold-local插补及拟合后评分。每个cell必须发布47条插补审计，其中20条覆盖producer candidate-CV；缺行、全缺失训练列或非当前训练parents拟合均会失败。完整范围是30个独立cells、2,190次拟合和约342,000棵树。worker逐fit输出五阶段进度、73-fit总进度、elapsed与ETA；批量入口输出30-cell总进度和按实测单元耗时更新的ETA。任何单cell不得用于路线选择，fixed validation/test继续关闭。

v4全部30 cells使用可恢复的串行后台入口：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_gate3_b6_all_cells_v1.sh
```

批量运行器会校验并跳过已经完成的cells；遇到失败立即停止，修复后使用同一命令续跑。只有全部30 cells通过后才生成`results/benchmarks/gate3_b6_physchem_formal_cells_v3/batch_complete.json`，该标记仍不授权架构选择，必须继续运行预注册汇总分析。v3目录保留先前完整pilot作为资源与数值证据，不与v4授权批次混合。

### Gate 3 B6预注册正式汇总与发表图

```bash
conda run --no-capture-output -n oneadmet python scripts/freeze_gate3_b6_aggregate_analysis.py --check-only
conda run --no-capture-output -n oneadmet python scripts/analyze_gate3_b6_physchem_formal.py --check-only
conda run --no-capture-output -n oneadmet python scripts/analyze_gate3_b6_physchem_formal.py
```

冻结协议为`data/public_development/gate3_b6_aggregate_analysis_protocol_v1`，推荐结果为`results/analysis/gate3_b6_physchem_formal_analysis_v2`。分析重新验证30个cell，使用10,000次按outer fold分层的paired-parent bootstrap，计算A1–A7、低相似度和重复parent敏感性，并发布9张机器可读表、英文报告和5张600-dpi PNG。v1仅因图例布局保留为历史；v2统计数值未改变。0/6端点晋级，B6停止扩展并作为阴性消融保留；fixed validation/test仍关闭。

投稿/归档整合包使用：

```bash
conda run --no-capture-output -n oneadmet python scripts/package_gate3_b6_publication_bundle.py --check-only
conda run --no-capture-output -n oneadmet python scripts/package_gate3_b6_publication_bundle.py
```

推荐整合目录为`results/analysis/gate3_b6_publication_bundle_v1`。它不移动或改写原始v2产物，而是把中英文报告、9张表、5张600-dpi PNG、统计合同、Gate表和批次来源复制到同一平铺目录，并用`source_manifest.csv`与独立`complete.json`验证所有副本哈希。

为检验旧 sklearn MLP 的随机内部验证是否影响了其余端点的 representation 结论，另有 5 个非-fu 端点的严格 nested-scaffold MLP 诊断。它固定在 float64 拟合特征标准化，并将标准化值裁剪为 `[-20,20]` 后进入网络；这防止极端 descriptor 主导梯度，而不从任何 evaluation 数据估计参数。它应在 fu 正式任务完成后串行运行：

```bash
bash scripts/run_stl_long_task_background.sh scripts/run_gate1b_nested_scaffold_mlp_v1.sh
```

# transformers v5 适配分支：改动概览与待完成列表

> 分支 `feat/transformers5-hews`，基于 upstream `4f9953b`。**这是供查看的独立开发分支，不用于合并**；正式提交会按主题拆成 PR 走 CI，届时本文件会删除。
> 负责范围：Llama / Mixtral / DeepSeek-v3 / BERT。验证环境：torch 2.11.0+cu128，transformers 5.17.0 / 5.16.1 / 4.51.3，H20。

## 一、改动概览（按主题）

| 主题 | commit | 说明 |
|---|---|---|
| 基础兼容 | `2746d1ae` `39a36b20` `1fbbd2fc` | torch 2.11 下 `CpuAccelerator.get_rng_state`；lazy init 路径的 v5 兼容垫片（`colossalai/_compat.py`）；`StaticCache` 导入位置 |
| 四模型 v5 适配 | `31b9c8e5` `b33997ba` `df66bfc7` `0014174d` `5e750d00` `df42e2cc` `a919c544` | Llama / Mixtral / DeepSeek-v3 / BERT 的 modeling 与 policy 按 v5 契约重写；Mixtral SP 与 PP router logits；DeepSeek-v3 原生 v5 类的 policy 注册 |
| 注释清理 | `6dbc991f` `79410f39` `8c575517` `26459ca5` | 无行为变化；提 PR 前折叠进对应 commit |
| MoE 专家并行（EP） | `bbc357fc` `5493ccf3` `89c3be7a` | Mixtral / DeepSeek-v3 在 v5 融合专家参数上的 EP；Mixtral EP over TP |
| MoE checkpoint | `6b2afa0b` `a119f934` `9b30ee6e` `743fdbb7` | 从完整 checkpoint 加载融合专家；保存时沿 TP / EP 收集专家；MoE 异步分片保存修复 |
| pipeline | `e2508587` `6857632c` | 嵌套输出的 PP 梯度传递；interleaved 调度按 chunk 缓存 P2P 元数据 |
| checkpoint 通用 | `dbe022d3` `71a2beba` | 填充参数别名的加载路由；v5 下分片保存写入 `config.json` |
| 测试 | `1f8649a1` | DeepSeek-v3 原生与 Hub 远程代码两条实现路线分别测试 |
| MoE 修复（10-12） | `d225d4bb` `aab79753` | 只对融合专家沿 EP 收集 checkpoint，修复 v4 下 MoE checkpoint 误拼专家（`9b30ee6e` 的 fixup，提 PR 前折叠）；EP 切分融合专家时保持参数对象不变，专家才能被优化器更新 |
| MoE 优化器 checkpoint（10-15） | `73da6aa8` `0bb440e7` | 非根 ZeRO rank 不再参与 TP 收集（upstream 既有缺陷，EP×TP 下报错或挂起）；融合专家的优化器状态按与权重相同的方式沿 TP / EP 收集和切分 |
| v4 / v5 双实现 | `324d5530` `07f29e58` | 四模型 `modeling/`、`policies/` 按 transformers 大版本分派：`_<模型>_tf4.py` 为 upstream 原样，`_<模型>_tf5.py` 为 v5 实现；公共入口导入路径不变 |
| bug 修复 | `0d5a88ab` `4cc3d220` | Llama 在 SP 且不开 flash attention 时的因果掩码；ring 序列并行的权重梯度（upstream #6086 引入，与 v5 无关，可单独 cherry-pick） |

测试现状：

- **最近一次完整测试战役**从 10-12 开始，只测分支代码（`4de1acf2`），范围是四模型相关的 61 个测试文件。4 卡及以下的节点已跑完，8 卡节点还在等卡。
  - transformers 5.17.0：93 PASS / 6 FAIL / 2 需联网 / 10 SKIPPED。6 个 FAIL：GPT2、BLIP-2 属四模型以外；2 个因服务器未装 apex；`test_gemini_ckpIO` 的键数不一致在 upstream 上同样失败；`test_moe_expert_param_identity` 的一个用例因测试端口冲突失败（同一代码单独运行时通过）。
  - transformers 4.51.3：84 PASS / 3 FAIL / 9 需联网 / 18 SKIPPED。3 个 FAIL 为 apex 1 个、Gemini 2 个（同上）。
- 之后的 `73da6aa8`、`0bb440e7` 做了针对性验证：`test_moe_optimizer_checkpoint` 的 5 个配置（Mixtral EP4、EP2×TP2、EP2，DeepSeek-v3 EP4、EP2）及 MoE 相关回归在 v5 下通过，v4 回归通过。
- **更正**：此前本文写的"83 PASS"来自 09-29 的战役，包含当时未提交的专家参数身份修复。在 `aab79753` 提交之前，分支本身在 v5 下的 `test_mixtral[4]`、`test_deepseek_v3[4]` 实际是失败的（10-12 在纯分支代码上复现）。

## 二、待完成

### 1. MoE

- [x] **专家参数身份**：v5 EP 切分专家时新建了参数对象，专家不被优化器更新。已修复（`aab79753`）。
- [x] **融合专家的优化器状态 checkpoint**：EP 下 `save_optimizer` 断言失败，EP×TP 下挂起。已修复（`73da6aa8`、`0bb440e7`）；EP2×TP2 且 moe_dp = 2 的配置需 8 卡，待测。
- [ ] **combine 改为 fp32 路由权重 + fp32 求和**：草稿已验证（低精度下与 HF 的差异归零），未包含在本分支，语义待与 EP / MoE 层重构方案对齐（方案将于 10-15 ~ 10-21 讨论确定）。
- [x] **v4 下 MoE checkpoint 误拼专家**：`9b30ee6e` 的收集逻辑在 v4 逐专家结构下出错（v4 有 3 个测试回归）。已修复（`d225d4bb`，显式标记融合专家）；`test_mixtral_3d[8]` 待 8 卡验证。
- [ ] DeepSeek-v3 Hub 远程代码路线的 EP 支持是否恢复（目前远程代码测试按能力探测跳过）。

### 2. transformers v4 兼容

- [ ] v4 只验证了 4.51.3；路线图计划升 4.57.6，需确认范围后重新验证 `_tf4`。
- [ ] v4 Llama 路径（upstream 原样）在 SP 且不开 flash attention 时：PP 后段掩码长度错；输入无 padding 时掩码为 None，attention 非因果（静默算错）。是否修复待定；对应测试配置目前在 v4 下跳过。

### 3. 四模型以外

- [ ] 8 个重点模型以外的 12 个 policy 已做运行与静态检查：v5 下 SAM 可用；OPT、GPT2、Bloom、T5、Falcon（调用 v5 已删除的接口）与 BLIP-2、Command（模块结构或返回值变化）不可用；Mistral 测试挂起待诊断；GPT-J 测试被 upstream 跳过，ChatGLM2、DeepSeek v2 需联网无法判定。v4 下可测的全部通过。修复方式待定。
- [ ] 其他已知 v5 失败：推理模块（diffusers 0.29）。

### 4. 提交与验证

- [ ] 在 CI 上运行相关批次（CI 已就绪）；在 Docker 镜像中复测；8 核心模型组合回归。
- [ ] 提交结构整理：折叠注释清理 commit；代码注释与提交说明中引用的个人文档（如 `docs/30`、`scripts/N11_...`）改为自洽说明；按主题拆 PR 并关联 Issue。
- [ ] PR 中说明的测试调整：v4 下跳过的 Llama eager 配置；原生 DeepSeek-v3 loss 容差 rtol=atol=1e-3。
- [ ] 只针对分支代码的完整测试战役：4 卡及以下节点已完成（见上），8 卡节点待补。
- [ ] 用户迁移指南 / release notes。

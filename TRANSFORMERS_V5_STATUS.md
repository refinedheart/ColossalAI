# transformers v5 适配分支：改动概览与待完成列表

> 分支 `feat/transformers5-hews`，基于 upstream `4f9953b`。**这是供查看的独立开发分支，不用于合并**；正式提交会在 CI 接入后按主题拆成 PR，届时本文件会删除。
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
| v4 / v5 双实现 | `324d5530` `07f29e58` | 四模型 `modeling/`、`policies/` 按 transformers 大版本分派：`_<模型>_tf4.py` 为 upstream 原样，`_<模型>_tf5.py` 为 v5 实现；公共入口导入路径不变 |
| bug 修复 | `0d5a88ab` `4cc3d220` | Llama 在 SP 且不开 flash attention 时的因果掩码；ring 序列并行的权重梯度（upstream #6086 引入，与 v5 无关，可单独 cherry-pick） |

测试现状（四模型相关的 55 个测试文件，transformers 5.17.0 与 5.16.1 结果逐项一致）：83 PASS / 6 FAIL / 3 需联网 / 10 SKIPPED / 14 收集失败。失败均已定位：融合专家优化器 checkpoint（见下）、原生 DeepSeek-v3 容差（已修）、apex 未安装、Gemini 键数不一致（upstream 既有）、四模型以外的模型（GPT2、BLIP-2）。

## 二、待完成

### 1. MoE（等 EP / MoE 层重构方案的结论）

- [ ] **专家参数身份**：v5 EP 切分专家时新建了参数对象，专家不被优化器更新。修复已 review，**未包含在本分支**。
- [ ] **融合专家的优化器状态 checkpoint**：EP 下 `save_optimizer` 断言失败，EP×TP 下死锁；验收测试已写好，修复未实现。
- [ ] **combine 改为 fp32 路由权重 + fp32 求和**：草稿已验证（低精度下与 HF 的差异归零），未包含在本分支，语义待与重构方案对齐。
- [ ] **v4 下 MoE checkpoint 误拼专家**：`9b30ee6e` 的收集逻辑在 v4 逐专家结构下出错（v4 有 3 个测试回归）；修法已设计（显式标记融合专家），未实现。
- [ ] DeepSeek-v3 Hub 远程代码路线的 EP 支持是否恢复（目前远程代码测试按能力探测跳过）。

### 2. transformers v4 兼容

- [ ] v4 只验证了 4.51.3；路线图计划升 4.57.6，需确认范围后重新验证 `_tf4`。
- [ ] v4 Llama 路径（upstream 原样）在 SP 且不开 flash attention 时：PP 后段掩码长度错；输入无 padding 时掩码为 None，attention 非因果（静默算错）。是否修复待定；对应测试配置目前在 v4 下跳过。

### 3. 四模型以外

- [ ] "14 个长尾 policy"版本门禁：名单与负责人待确认（门禁设计：v5 下清晰报错 + 测试跳过）。
- [ ] 已知 v5 失败：ViT（policy 导入失败）、GPT2（`get_head_mask`）、BLIP-2（测试数据）、qwen2 / qwen3 / command（未适配）、推理模块（diffusers 0.29）。其余模型族只验证了导入。

### 4. 提交与验证

- [ ] 接入 CI；在 Docker 镜像中复测；8 核心模型组合回归。
- [ ] 提交结构整理：折叠注释清理 commit；代码注释与提交说明中引用的个人文档（如 `docs/30`、`scripts/N11_...`）改为自洽说明；按主题拆 PR 并关联 Issue。
- [ ] PR 中说明的测试调整：v4 下跳过的 Llama eager 配置；原生 DeepSeek-v3 loss 容差 rtol=atol=1e-3；优化器 checkpoint 测试的 `xfail`（随专家参数身份修复一起提交）。
- [ ] 用户迁移指南 / release notes。

# ADR-004: 首批 Open-R1 Codeforces 仅复用可精确输出验证的任务

## 状态

Accepted

## 日期

2026-08-03

## 背景

项目已经有一条经过千级真实采集验证的统一流水线：版本化任务记录进入
DeepSeek API collector，原始响应由单 writer 追加保存，随后流式规范化，并由多个隔离
verifier worker 执行官方测试。现有 stdin/stdout verifier 采用规范化换行和首尾空白后的
预期输出精确匹配。

Open-R1 Codeforces 提供 `verifiable` 子集，但其中仍包含三类当前 verifier 不能正确处理的
任务：需要 custom checker 的多解题、file I/O 题和交互题。数据集还提供约 110GB 的外置
generated tests；直接引入这些能力会同时改变执行协议、安全边界和存储要求，不再是简单的
数据源扩展。

## 决策

首批 Open-R1 Codeforces 使用 pinned revision
`fbe3f6e903ee854eec2e69e9d96d0306cde59baf` 的 `verifiable/train`，并只接受同时满足以下
条件的任务：

- `executable = true`；
- `official_tests_complete = true`；
- `input_mode = stdio`；
- `generated_checker` 为空；
- `interaction_format` 为空；
- 至少有一条结构完整的 `official_tests`。

适配器只把题面、公开样例和接口要求放入教师 prompt。官方测试、editorial 和 checker
都不进入 prompt。editorial 只保留 SHA-256 摘要；官方测试仅存于任务记录并交给隔离
verifier。

任务继续使用 `coding.task.multisource.v1`，后续完整复用已有 collector、raw queue、
normalizer、trace byte reconstruction、stdio verifier、聚合、clean/ALM/EOS 审计。ALM
trainer 和训练协议不变。

## 备选方案

### 立即支持 custom checker

- 优点：可利用更多多解题。
- 缺点：需要执行数据集提供的生成 checker，并重新定义 checker 的可信边界、超时、资源
  限制和评分语义。
- 结论：作为独立的第二阶段 verifier 功能开发，不混入本次采集。

### 下载全部 generated tests

- 优点：可验证更多 `official_tests_complete = false` 的题，并提高困难测试覆盖。
- 缺点：外置数据约 110GB，超过当前剩余数据盘空间，且需要新的按 contest 关联逻辑。
- 结论：当前不下载；扩盘并完成单独审计后再考虑。

### 只把题目送给教师，不执行完整验证

- 优点：实现最快。
- 缺点：会破坏训练候选集“真实 trace + 可解析代码 + 官方测试通过”的既有资格契约。
- 结论：拒绝。

## 后果

- 新数据源只增加 importer/过滤器和 source registry，长期维护面较小。
- 所有 accepted 记录仍保持与现有 ALM 数据兼容的实际 token bytes/logprobs。
- 首批样本会系统性排除多解题、交互题和 file I/O 题，不能声称代表完整 Codeforces。
- 数据集 metadata tag 标注 `cc-by-4.0`，README 正文却写 ODC-By 4.0；manifest 暂按机器可读
  tag 记录 `CC-BY-4.0`，并显式保留冲突说明。
- 初始 3 题冒烟用 `selection=first` 以缩短 schema 验证时间；正式数据集若使用 random，必须
  扫描整个 pinned split 并记录额外下载/扫描成本。

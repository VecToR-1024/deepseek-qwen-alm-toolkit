# TACO breadth-first v2：1,000 个新题单次盲采

> 日期：2026-07-28
> 状态：采集、验证、聚合与 ALM 审计完成
> 分支：`codex/alm-offline-kd`

## 1. 目标

本 campaign 用相同 API 调用预算优先覆盖更多独立编程问题，而不是对已失败的
任务进行低温、无反馈的重复生成。它只扩展离线教师数据，不修改 ALM trainer，
也不启动学生训练。

决策依据见 `docs/decisions/003-prefer-taco-breadth-over-retries.md`。

## 2. 数据来源与固定选择

| 字段 | 值 |
|---|---|
| 数据集 | `BAAI/TACO` |
| split | `train` |
| revision | `d593ed0a2becbbc952230bb89be09189bf1056dc` |
| 固定分片 | `train/data-00000-of-00009.arrow` |
| provenance | `https://github.com/FlagOpen/TACO` |
| mirror | `https://huggingface.co/datasets/BAAI/TACO` |
| license | dataset card/repository 为 Apache-2.0；上游题目许可各异，card 特别提示 HackerRank 权利未知 |
| selection | 排除后 random |
| seed | `20260728` |
| 新任务数 | 1,000 |
| 排除 | TACO pilot v1 的 100 个 task ID |
| 排除来源 | `geeksforgeeks` |
| selection scope | `single_pinned_train_shard_breadth_v2` |

本次固定使用已经下载并验证的单个 Arrow 分片，避免为了 breadth 实验额外下载
其余约 GB 级分片。由此产生 single-shard sampling bias：这 1,000 题不能表述
为完整 TACO train 的无偏样本。

固定任务文件：

```text
data/taco_breadth_v2_seed20260728/selected_tasks_1000.jsonl
```

校验值：

```text
file SHA-256:
b485c3bcd5ff7f1d99b38344b8bec3f9778b1a0039cb227b890bd10bab77349d

ordered task IDs SHA-256:
85cd2d0f7f5a1176b8ea494bd523d9805de10db80e13117847c1e5af256eb450
```

选择检查：

- 1,000 个 task ID 全部唯一；
- 与 pilot v1 的 task ID 交集为 0；
- `geeksforgeeks` 来源为 0；
- 1,000/1,000 个 teacher prompt 均不包含官方 tests；
- 保持固定任务顺序，不按回答长度、难度或验证结果重新排序。

来源分布为 Codeforces 464、HackerEarth 143、AIZU 132、CodeChef 110、
AtCoder 87、Kattis 64。

## 3. 教师生成协议

| 参数 | 值 |
|---|---|
| provider/model | DeepSeek / `deepseek-v4-pro` |
| attempts per task | 1 |
| temperature | 0.2 |
| top_p | 1.0 |
| logprobs | true |
| top_logprobs | 20 |
| max_tokens | 4096 |
| thinking | disabled |
| workers | 4 |

每题只生成一次。无论失败类别是什么，都不向教师返回官方测试、失败断言、
stderr、traceback 或 verifier 反馈，也不自动增加 `max_tokens`。

API key 只从 `DEEPSEEK_API_KEY` 环境变量读取，不写入 manifest、日志或仓库。

## 4. 数据流与恢复

```text
fixed 1,000-task manifest
-> one blind DeepSeek request per task
-> append-only raw_attempts.jsonl
-> strict trace normalization and byte reconstruction
-> conservative source extraction
-> isolated stdin/stdout verifier
-> streaming accepted/rejected aggregation
-> streaming audit and ALM preprocessing diagnostics
```

attempt ID 固定为 `{task_id}__attempt_1`。collector 先读取已完成 ID，再只请求
pending IDs，因此进程重启不会重复已落盘 attempt。原始 API JSON 和所有失败
结果永久保留。

运行中发现原 collector 和后处理对 GB 级 JSONL 存在内存放大：

- 并发 collector 曾保留全部已完成 response futures；
- resume planner 曾读取完整 raw/verifier records；
- append normalizer 和 verifier 曾把完整输入载入列表。

这些路径已分别改为有界 future 集合、轻量 attempt-history projection 和两遍
流式扫描。数据内容、schema、验证规则和训练器均未改变。

## 5. 运行目录与复现命令

运行目录：

```text
data/taco_breadth_v2_seed20260728/run1000
```

准备固定 manifest：

```powershell
python scripts/import_taco_breadth.py `
  --input-arrow <pinned-train-arrow> `
  --prior-tasks data/taco_pilot_v1/selected_tasks_100.jsonl `
  --output data/taco_breadth_v2_seed20260728/selected_tasks_1000.jsonl
```

采集、规范化和验证：

```powershell
python scripts/collect_taco_breadth.py `
  --tasks data/taco_breadth_v2_seed20260728/selected_tasks_1000.jsonl `
  --prior-tasks data/taco_pilot_v1/selected_tasks_100.jsonl `
  --run-dir data/taco_breadth_v2_seed20260728/run1000 `
  --expected-tasks 1000 `
  --collect-only
```

流式聚合：

```powershell
python scripts/collect_taco_breadth.py `
  --tasks data/taco_breadth_v2_seed20260728/selected_tasks_1000.jsonl `
  --prior-tasks data/taco_pilot_v1/selected_tasks_100.jsonl `
  --run-dir data/taco_breadth_v2_seed20260728/run1000 `
  --expected-tasks 1000 `
  --aggregate-only
```

审计和 ALM 预处理诊断：

```powershell
python scripts/audit_taco_breadth.py `
  --run-dir data/taco_breadth_v2_seed20260728/run1000
```

## 6. 最终结果

### 6.1 收集与验证

| 指标 | 结果 |
|---|---:|
| 选择任务 / API attempts | 1,000 / 1,000 |
| API 成功 | 997/1,000（99.70%） |
| trace 精确重建 | 997/997（100%） |
| source extraction | 812/997（81.44%） |
| 唯一 accepted / pass@1 | 412/1,000（41.20%） |
| 失败全部 attempt 的任务 | 588 |
| 重复 raw / normalized / verifier ID | 0 / 0 / 0 |
| resumability | 1,000/1,000 completed，resume safe |

失败分布：

| 类别 | 数量 |
|---|---:|
| API error | 3 |
| assertion failure | 343 |
| extraction error | 176 |
| runtime error | 41 |
| timeout | 16 |
| syntax error | 8 |
| forbidden operation | 1 |

finish reason 为 791 个 `stop`、206 个 `length`。本 campaign 不对 206 个
触顶 attempt 自动重试；其中是否存在可通过的截断前缀由 verifier 正常决定，
不会单凭 finish reason 删除或改写记录。

### 6.2 trace、token 与成本

| 指标 | 结果 |
|---|---:|
| completion tokens | 1,403,222 |
| prompt tokens | 693,140 |
| cache hit / miss input tokens | 138,368 / 554,772 |
| total tokens | 2,096,362 |
| response actual tokens | min 13 / median 549 / p95 4096 / max 4096 |
| prompt tokens | min 225 / median 656 / p95 1131 / max 2846 |
| API latency seconds | min 1.08 / median 8.57 / p95 83.86 / max 433.83 |
| actual-token logprob | 1,403,221/1,403,221 positions |
| 完整 top-20 | 1,403,221/1,403,221 positions |
| 无效 actual/top-candidate bytes | 0 / 0 |
| 估算 API 总成本 | CNY 10.0871072 |
| 每 attempt / 每 accepted | CNY 0.0100871 / 0.0244833 |

成本使用审计脚本当前固定单价：cache-hit input CNY 0.025/M、cache-miss
input CNY 3/M、output CNY 6/M。它是根据 token usage 的估算，不代替 provider
账单。

### 6.3 ALM 预处理

412 条 accepted 全部成功完成 Qwen tokenizer、teacher forcing 所需边界构造
和 byte-endpoint chunk 对齐，0 error、0 zero-chunk、0 prompt/completion
boundary drop。

| 指标 | 结果 |
|---|---:|
| Qwen sequence length | min 353 / median 994 / p95 2699 / max 4775 |
| ALM chunks/example | min 13 / median 282.5 / p95 1797 / max 3987 |
| 1:1 chunks | 211,771 |
| 1:N chunks | 1,471 |
| N:1 chunks | 2,590 |
| N:M chunks | 410 |
| zero valid chunks | 0 |
| 超过 4096 tokens | 5 |

超过 4096 的 attempt IDs：

```text
taco_train_002275__attempt_1
taco_train_000664__attempt_1
taco_train_001406__attempt_1
taco_train_000283__attempt_1
taco_train_001152__attempt_1
```

这些记录没有为改善统计而截断、删除或修改。正式训练数据版本必须显式决定是
排除、提高长度上限，还是采用可验证的 completion-aware 截断策略。

### 6.4 权威产物

```text
data/taco_breadth_v2_seed20260728/run1000/raw_attempts.jsonl
data/taco_breadth_v2_seed20260728/run1000/normalized_attempts.jsonl
data/taco_breadth_v2_seed20260728/run1000/verifier_attempts.jsonl
data/taco_breadth_v2_seed20260728/run1000/accepted_unique.jsonl
data/taco_breadth_v2_seed20260728/run1000/rejected_attempts.jsonl
data/taco_breadth_v2_seed20260728/run1000/rejected_tasks.jsonl
data/taco_breadth_v2_seed20260728/run1000/attempt_ledger.jsonl
data/taco_breadth_v2_seed20260728/run1000/audit_report.json
data/taco_breadth_v2_seed20260728/run1000/audit_report.md
```

SHA-256：

```text
accepted_unique.jsonl  d5ebda9d841e1f7437ab8b3c70b940723f188081fc7b9e9d1000867c4720337b
audit_report.json      56a92ce6c32dd60e0745afe9fd8ce34876ce260a020b89dd0e57815bf735a84e
audit_report.md        5f0dc0c57ee0cbda7371969784e24640c0d6b63114087e48ef8109cde1f5b25f
breadth_summary.json   881f5ad30afe243fa53caf628716482e9a660b2687484ed1848ca22acbfa2f93
campaign_manifest.json e26f5f336206610ee5d4dae896861f0b99d96311f8a1abb3a432ef5f3e4fe9b0
```

## 7. 已知限制

- 当前 Windows verifier 使用隔离子进程和临时工作目录，但不是完整安全沙箱；
  对不可信生成代码的正式大规模执行优先迁移到 Linux 容器/虚拟机并禁网。
- 单分片 selection bias 仍然存在。
- TACO 官方测试本身只判定给定样例；通过不等同于对隐藏分布的完全正确。
- breadth-first pass@1 与旧 campaign 的累计 pass@3 不是同一指标，不能直接
  比较而忽略 API 调用数。

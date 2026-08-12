# DeepSeek → Qwen Offline ALM Toolkit

这是一套从研究项目中拆出的、可独立安装和测试的代码蒸馏工具链。它把公开编程题送给 DeepSeek API，持久化实际生成 token 的 bytes/logprob，在隔离进程中运行官方测试，再用同一 completion 对 Qwen 做 teacher forcing，通过字节端点对齐实现 Approximate Likelihood Matching（ALM）。

主目标保持为：

```text
total_loss = hard_sft_loss + alpha_alm * alm_loss
```

仓库不包含 API key、原始 trace、训练数据、模型、LoRA adapter、checkpoint 或历史日志。`proven_assets/` 只保存已经跑通过的启动器快照，便于复用旧轮子；其中的旧云端绝对路径是历史运行契约，不是通用配置。

## 当前能力

- 数据导入：MBPP、APPS、CodeContests、TACO multi-shard、Open-R1 Codeforces、ODEX、xCodeEval。
- 教师采集：DeepSeek OpenAI-compatible chat API，支持 `actual_only` 和旧的 `top20` trace。
- 48-worker 流水线：32 API workers + durable raw single writer + streaming normalizer + bounded queue + 16 isolated verifier workers + verifier single writer。
- 可靠性：append-only JSONL、fsync、断点恢复、任务/attempt 去重、错误记录、磁盘水位保护、优雅停止。
- 验证：源码提取、AST、接口、禁用操作、compile/import/test 分阶段隔离执行。
- 数据冻结：通过率、格式、trace、近重复、ALM dry-run、EOS/label 和 benchmark overlap 审计。
- 蒸馏：OfflineTeacherTraceProvider、O(T+S) 字节端点 ALM、稳定 forward-KL、严格 top-20 基线。
- 训练：同一 Transformers/TRL 入口支持 BF16 LoRA 与 BF16 全参训练，支持 SFT-only 和 SFT+ALM 对照。
- Benchmark：保留已跑通的 EvalPlus / LiveCodeBench 资产及监控脚本作为可追溯参考。

## 2026-08-12 当前状态

- 48-worker `actual_only` 采集/验证链路属于已跑通资产。
- 最新 4,500 次采集得到 1,656 条 clean ALM 候选、按题面精确去重后 1,619 条；数据文件不进入本仓库，验收摘要见 `docs/data_acceptance_qwen3_0_6b_20260811.md`。
- `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` 的非思考模板数据契约已验证：1,656/1,656 EOS supervised，ALM error、zero chunk、boundary drop 和 over-4096 均为 0。
- Qwen3 全参 BF16 启动器是新整理的候选资产，尚未完成 GPU smoke，因此不得与 `proven_assets/` 中已经实际跑通的 Qwen2.5 LoRA 启动器混为一谈。

## 快速安装

推荐 Python 3.12：

```powershell
conda create -n deepseek-qwen-alm python=3.12 -y
conda activate deepseek-qwen-alm
python -m pip install -e ".[collect,archive,data,train,test]"
python -m pytest -q --basetemp=.pytest-tmp
```

仅做本地收集/验证时不需要 CUDA。训练环境的 PyTorch/CUDA 应按目标 GPU 镜像安装；不要因为驱动显示 CUDA 13.x 就强行安装同版本 toolkit，PyTorch wheel 自带的 CUDA runtime 与足够新的驱动兼容即可。

## 先检查 48-worker 命令

示例配置位于 `configs/collection.actual-only.48workers.example.json`。它默认用单个 TACO lane 吃满 32+16 worker，`actual_only` 不请求 top-20：

```powershell
python scripts/run_hard_collection_campaign.py `
  --config configs/collection.actual-only.48workers.example.json `
  --repo-root . `
  --python (Get-Command python).Source `
  --dry-run
```

实际开始前，复制配置并修改 `campaign_id`、`run_root`、数据源、limit、seed 和排除列表；不要直接复用示例 run ID。

```powershell
$env:DEEPSEEK_API_KEY = "..."
powershell -ExecutionPolicy Bypass -File examples/collection_48workers/start_local.ps1 `
  -Config configs/my_campaign.json
```

监控和优雅停止：

```powershell
powershell -ExecutionPolicy Bypass -File examples/collection_48workers/monitor_local.ps1 `
  -Config configs/my_campaign.json
powershell -ExecutionPolicy Bypass -File examples/collection_48workers/stop_local.ps1 `
  -Config configs/my_campaign.json
```

停止只写入 `STOP` 并终止子进程树，已经 fsync 的 raw/normalized/verifier 队列会保留。再次启动同一冻结配置会跳过已完成 ID。

## 数据流

```text
pinned benchmark tasks
        ↓
DeepSeek API workers (32)
        ↓
raw_attempts.jsonl ── single writer / append-only
        ↓
streaming normalization + byte reconstruction
        ↓
bounded verification queue
        ↓
isolated verifier workers (16)
        ↓
verifier_attempts.jsonl ── single writer / append-only
        ↓
accepted_alm / accepted_sft_only / rejected / audit / frozen dataset
```

官方测试永远不进入 teacher prompt。所有失败尝试都保留，主 ALM 数据只接受原始教师回答本身已满足格式、trace 和官方测试契约的样本。清洗过文本不能继续沿用原始 token bytes/logprob。

## 训练

权威入口是 `examples/train_offline_alm.py`。它不会 4-bit 量化基础模型，因此这里没有把 BF16 LoRA 错称为 QLoRA。已有两个 Qwen2.5 LoRA 对照模板：

- `configs/training.sft-only.example.json`：`ALPHA_ALM=0.0`
- `configs/training.sft-alm.example.json`：`ALPHA_ALM=10.0`

另有一个 Qwen3-0.6B 全参候选模板：

- `configs/training.qwen3-0.6b-full.example.json`：`USE_LORA=0`，并把 `{"enable_thinking": false}` 同时传给 prompt 与 completion 的 chat template。

Linux GPU 上可用同一个启动器依次跑两臂：

```bash
export TRAIN_DATASET=/path/to/frozen/training_records.jsonl
export STUDENT_MODEL=/path/to/pinned/qwen/snapshot
export OUTPUT_ROOT=/path/to/experiment
nohup bash examples/training/launch_pair.sh > "$OUTPUT_ROOT/launcher.log" 2>&1 &
bash examples/training/monitor_training.sh "$OUTPUT_ROOT"
```

Qwen3-0.6B 全参训练应先做短 smoke，具体命令和验收门禁见 `docs/qwen3_0_6b_full_finetune.md`。模板可依次运行 SFT-only 与 SFT+ALM：

```bash
export TRAIN_DATASET=/path/to/frozen/training_records.jsonl
export OUTPUT_ROOT=/path/to/qwen3_0_6b_full_pair
TRAIN_LIMIT=8 MAX_STEPS=2 bash examples/training/launch_qwen3_0_6b_full_pair.sh
bash examples/training/monitor_training.sh "$OUTPUT_ROOT"
```

训练前建议先运行：

```bash
python scripts/audit_frozen_training_dataset.py --help
python scripts/audit_training_data_contract.py --help
```

这些检查覆盖 trace 重建、Qwen tokenizer/chat template、EOS label、4096 长度、ALM chunk 和 benchmark overlap。Qwen3 审计必须显式传 `--chat-template-kwargs '{"enable_thinking": false}'`。训练不会在采集、审计或安装过程中自动开始。

## Top-20 与 ALM 的关系

主路径 `actual_only` 只需要完整 completion、实际 token bytes 和实际 token logprob。它显著减小 raw JSON。旧的 top-20 + tail-bucket loss 仍保留在 `topk_distill` 中作实验基线，但不是默认训练目标。不要把 GOLD 的 ULD loss 直接替换进来；这里只复用了跨 tokenizer 的字节/跨度对齐思想。

## Benchmark 资产

`proven_assets/` 包含两组未改动的、已实际跑通过的快照：

- Qwen2.5-7B-Instruct alpha=10 训练与 base/checkpoint 对比启动器；
- EvalPlus + LiveCodeBench 生成、隔离评分与结果汇总 harness。

这些脚本冻结了当时的模型 revision、benchmark commit、SHA-256 和云端目录。迁移到新机器时应复制成新的 run 目录并显式替换路径，而不是修改历史快照。Benchmark 会执行不可信代码，只应在 Linux 的非 root 用户、资源限制和 seccomp/容器边界内运行。

## Hugging Face 数据集发布

`scripts/release_hf_dataset.py` 可从权威冻结 JSONL 构建确定性、默认私有的
Hub 发布包。发布投影保留 ALM 所需的实际 token bytes/logprobs，并剥离
benchmark 测试、本机路径、provider 标识、疑似凭据及不用的 top-20 候选。
上传默认只打印 dry-run，只有显式加入 `--execute` 才会写入 Hub。完整命令、
2,041 条权威训练集身份和混合许可说明见
[`docs/huggingface_dataset_release.md`](docs/huggingface_dataset_release.md)。

## 目录

```text
src/deepseek_distill/   数据、API、持久化流水线、验证、审计、ALM 预处理
src/topk_distill/       ALM 数学/Trainer 与严格 top-20 实验基线
scripts/                可组合 CLI
examples/               训练入口和 48-worker/训练操作模板
configs/                无密钥示例配置
proven_assets/          已跑通并完成机器信息脱敏的历史启动器模板
tests/                  纯离线 fake-API/子进程测试
docs/                   架构、计划、ADR、来源与历史技术说明
```

详细架构见 `docs/architecture.md`，拆仓范围见 `docs/spec.md`，来源见 `ASSET_MANIFEST.md`。

## 安全与边界

- API key 只从 `DEEPSEEK_API_KEY` 读取；示例和测试不含密钥。
- pytest 不发起 live API 请求，也不下载模型或数据。
- 生成代码绝不在收集主进程执行。
- `data/`、`outputs/`、`runs/`、模型、checkpoint 和 `.env` 默认忽略。
- 密码 SSH 部署器只从通用的 `REMOTE_SSH_PASSWORD` 读取；主机、端口、用户和远端路径必须显式传参。
- 历史启动器从 `AUTODL_ROOT` 读取远端工作区根目录，不含实例专属连接信息。
- 生产使用建议换成 SSH key 和严格 host-key 校验。

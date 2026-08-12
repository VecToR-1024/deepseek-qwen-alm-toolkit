# Authoritative MBPP DeepSeek collection

## Scope

This run builds the first coding dataset for offline DeepSeek-to-Qwen
distillation. It does not train the student. The teacher is the
`deepseek-v4-pro` chat API in non-thinking mode; the future training path remains
`hard_sft_loss + alpha_alm * alm_loss`.

The pipeline reuses the repository's raw v1 collector, append-only resume logic,
normalized v1 trace schema, exact byte validation, `OfflineTeacherTraceProvider`,
and current ALM preprocessing. The ALM trainer and loss were not modified.

## Dataset provenance

- Original dataset: Google Research, `google-research/google-research/mbpp`.
- Original split definition: task IDs 601–974 are training; IDs 511–600 are
  validation; IDs 11–510 are test; IDs 1–10 are prompt examples.
- Mirror: Hugging Face `google-research-datasets/mbpp`.
- Mirror configuration and split: `full/train` only.
- Pinned mirror revision:
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Mirrored training parquet LFS SHA-256:
  `09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae`.
- License: CC BY 4.0 for the dataset. Repository code is separately Apache 2.0.
- Selection: original task IDs 601–620, sorted numerically; `selection=first`,
  `limit=20`, `seed=0`. No validation or test record is used.

The importer stores the original problem statement verbatim. Official tests,
challenge tests, test setup, and reference code are separate metadata and never
become part of either teacher message. Function names and signatures are
deterministically extracted from explicit definitions, test AST calls, and
licensed reference code in that order. For task 601, the nested constructor
interface `Pair(a, b)` is exposed without exposing any assertion; the original
problem remains unchanged.

## Pinned provider and tokenizer contracts

The collection request is:

- model: `deepseek-v4-pro`;
- thinking: disabled;
- temperature: 0.2;
- top-p: 1.0;
- logprobs: true;
- top-logprobs: 20;
- max completion tokens: 4096.

DeepSeek's current API contract permits `top_logprobs` up to 20 and returns the
actual token, actual-token logprob, optional UTF-8 bytes, and top-N candidates at
each output position. Pricing used for the audit, checked 2026-07-21, is CNY per
million tokens for V4 Pro: 0.025 cached-input, 3 uncached-input, and 6 output.

ALM diagnostics use only the tokenizer from
`Qwen/Qwen2.5-Coder-7B-Instruct`, pinned to revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`. The Qwen repository is Apache 2.0.
No Qwen model weights are loaded by the audit.

## Reproduction

Install the declared optional dependencies:

```powershell
conda activate topk-distill
python -m pip install -e ".[collect,data,test]"
```

Keep the API key outside code and datasets:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
```

Import the fixed tasks:

```powershell
python scripts/import_mbpp.py `
  --output data/mbpp_smoke_v1/tasks_20.jsonl `
  --summary-output data/mbpp_smoke_v1/import_summary_20.json `
  --limit 20 --selection first --seed 0 `
  --revision 4bb6404fdc6cacfda99d4ac4205087b89d32030c `
  --cache-dir data/mbpp_smoke_v1/.hf-cache
```

Collect and preserve complete raw responses:

```powershell
python scripts/collect_teacher.py `
  --tasks data/mbpp_smoke_v1/tasks_20.jsonl `
  --raw-output data/mbpp_smoke_v1/raw_20.jsonl `
  --model deepseek-v4-pro `
  --temperature 0.2 --top-p 1.0 --top-logprobs 20 `
  --max-tokens 4096 --workers 1 --requests-per-minute 20 `
  --timeout 180 --max-retries 2 `
  --summary-output data/mbpp_smoke_v1/collection_summary_20.json
```

Normalize, validate, and verify:

```powershell
python scripts/validate_dataset.py --input data/mbpp_smoke_v1/raw_20.jsonl
python scripts/normalize_responses.py `
  --input data/mbpp_smoke_v1/raw_20.jsonl `
  --output data/mbpp_smoke_v1/normalized_20.jsonl
python scripts/validate_dataset.py --input data/mbpp_smoke_v1/normalized_20.jsonl
python scripts/verify_code.py `
  --input data/mbpp_smoke_v1/normalized_20.jsonl `
  --output data/mbpp_smoke_v1/verifier_20.jsonl `
  --phase-timeout 5 --max-output-characters 65536
```

Build candidates and audit without training:

```powershell
python scripts/build_coding_dataset.py `
  --raw data/mbpp_smoke_v1/raw_20.jsonl `
  --normalized data/mbpp_smoke_v1/normalized_20.jsonl `
  --verifier data/mbpp_smoke_v1/verifier_20.jsonl `
  --accepted-output data/mbpp_smoke_v1/accepted_20.jsonl `
  --rejected-output data/mbpp_smoke_v1/rejected_20.jsonl

python scripts/audit_dataset.py `
  --tasks data/mbpp_smoke_v1/tasks_20.jsonl `
  --raw data/mbpp_smoke_v1/raw_20.jsonl `
  --normalized data/mbpp_smoke_v1/normalized_20.jsonl `
  --verifier data/mbpp_smoke_v1/verifier_20.jsonl `
  --accepted data/mbpp_smoke_v1/accepted_20.jsonl `
  --resume-summary data/mbpp_smoke_v1/resume_summary_20.json `
  --json-output data/mbpp_smoke_v1/audit_20_final.json `
  --markdown-output data/mbpp_smoke_v1/audit_20_final.md `
  --student-tokenizer Qwen/Qwen2.5-Coder-7B-Instruct `
  --student-revision c03e6d358207e414f1eca0bb1891e29f1db0e242 `
  --tokenizer-cache-dir data/mbpp_smoke_v1/.hf-cache `
  --max-length 4096
```

Rerunning the collection command against the same raw file produced
`total=20, skipped=20, succeeded=0, failed=0` and no duplicate record IDs.

## Observed 20-task result

- API success: 20/20.
- Exact response byte reconstruction: 20/20.
- Source extraction, syntax, and import success: 20/20 each.
- Official test pass: 11/20 (55%).
- Rejections: 9 `assertion_failure`; no API, malformed trace, extraction,
  syntax, import, forbidden-operation, timeout, or runtime failures.
- Actual logprobs: 1180/1180 positions.
- Exactly 20 candidates: 1180/1180 positions, 23,600 candidate rows.
- Missing/invalid actual or candidate byte arrays: 0.
- Finish reasons: 20 `stop`.
- Usage: 4,746 prompt tokens and 1,180 completion tokens; 5,926 total.
- Estimated cost: CNY 0.013702 total; CNY 0.00068510 per attempt;
  CNY 0.00124564 per accepted record.

Accepted IDs are 601, 604, 605, 606, 608, 610, 611, 614, 616, 618, and 619.
Rejected IDs are 602, 603, 607, 609, 612, 613, 615, 617, and 620.

ALM preprocessing succeeded for all 11 accepted records:

- Qwen sequence lengths: min 255, median 288, mean 296.45, max 374.
- Valid ALM chunks per example: min 16, median 36, mean 49.91, max 117.
- Chunk shapes: 537 `1:1`, 1 `1:N`, 10 `N:1`, and 1 `N:M`.
- Prompt/completion boundary drops: 0.
- Examples with zero valid ALM chunks: 0.
- Records exceeding 4096 tokens: 0.
- ALM preprocessing errors: 0.

The authoritative machine-readable values are in `audit_20_final.json`; the
human-readable companion is `audit_20_final.md`.

## Isolation and unresolved risks

Generated source never runs in the collection process. The verifier first
parses and statically rejects obvious file, network, subprocess, external
package, `eval`, and `exec` operations. It then runs separate compile, import,
and test child processes under `python -I`, a fresh temporary working directory,
a sanitized environment, captured output, and strict wall-clock timeouts. Linux
additionally applies CPU, address-space, file-size, file-descriptor, and process
resource limits where supported.

This is not a complete Windows security sandbox. Static AST filtering can be
bypassed by sufficiently adversarial Python, and Windows execution does not
provide seccomp or network namespaces. Before scaling on the Linux vGPU host,
run the verifier inside a network-disabled container or equivalent sandbox with
a read-only base filesystem and disposable writable directory.

MBPP problem statements can also be underspecified. The pipeline deliberately
does not inject expected outputs or paraphrase the task after observing a test
failure. The nine failures remain useful for later correction or preference
data but are not ALM/SFT training candidates.

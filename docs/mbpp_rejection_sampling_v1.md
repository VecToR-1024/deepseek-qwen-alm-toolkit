# MBPP rejection-sampling v1 run

## Status and scope

This run produced the requested 200-example training-candidate dataset without
starting student training or modifying the ALM trainer. It used blind
rejection sampling: each selected MBPP problem received at most three
independent generations, the teacher received the same system/user messages on
every attempt, and collection stopped for a problem immediately after its
first passing attempt.

The authoritative run directory is:

`data/mbpp_rejection_v1_seed20260721`

## Dataset and selection contract

- Dataset mirror: `google-research-datasets/mbpp`.
- Original provenance: Google Research MBPP.
- Configuration/split: `full/train` only.
- Pinned revision: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- License: CC BY 4.0.
- Selection: 300 unique tasks, `random`, seed `20260721`.
- Order: the exact output order of
  `random.Random(20260721).sample(range(601, 975), 300)` after the importer
  canonically sorts the source rows by task ID.
- First ten original IDs: `914, 695, 794, 961, 804, 807, 742, 840, 604, 707`.
- Selected-task SHA-256:
  `2f2ca2030fb2211f1b700b84fea0ff2d8574ef5ae9132443c1670f8fb03bb3f5`.
- Ordered-ID SHA-256:
  `53da0123976de0c3d29757a3ba0c2470f1649a62f65ad7241edaba37f7ec6f84`.

Official tests, challenge tests, setup code, and reference code remain in local
task metadata for the isolated verifier. Prompt construction reads only the
original problem and deterministic function interface. Attempt IDs are
`mbpp_<original_id>__attempt_<1..3>`, while the teacher-visible Task ID remains
`mbpp_<original_id>` so retries cannot infer prior failure.

## Provider contract

- Model: `deepseek-v4-pro`.
- Thinking: disabled.
- Temperature: `0.2`.
- Top-p: `1.0`.
- Logprobs: enabled.
- Top logprobs: `20`.
- Maximum completion tokens: `4096`.
- Collection concurrency: 4 workers, request starts limited to 60/minute.

The cost calculation uses the DeepSeek V4 Pro CNY prices checked on 2026-07-21:
0.025/million cached input tokens, 3/million uncached input tokens, and
6/million output tokens. The current source is the
[official DeepSeek pricing page](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/).

## Reproduction

The commands below assume the `topk-distill` conda environment and a
`DEEPSEEK_API_KEY` environment variable. Credentials are never command-line
arguments or persisted fields.

```powershell
$run = "data/mbpp_rejection_v1_seed20260721"
$revision = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
$qwenRevision = "c03e6d358207e414f1eca0bb1891e29f1db0e242"

python scripts/import_mbpp.py `
  --output "$run/selected_tasks_300.jsonl" `
  --summary-output "$run/import_summary.json" `
  --limit 300 --selection random --seed 20260721 `
  --revision $revision `
  --cache-dir data/mbpp_smoke_v1/.hf-cache

python scripts/collect_mbpp_rejection_sampling.py `
  --tasks "$run/selected_tasks_300.jsonl" `
  --run-dir $run `
  --workers 4 --requests-per-minute 60 `
  --model deepseek-v4-pro `
  --temperature 0.2 --top-p 1.0 `
  --top-logprobs 20 --max-tokens 4096 `
  --summary-output "$run/collection_summary_initial.json"

python scripts/build_rejection_sampling_dataset.py `
  --run-dir $run --target 200

# A second invocation must make no provider requests.
python scripts/collect_mbpp_rejection_sampling.py `
  --tasks "$run/selected_tasks_300.jsonl" `
  --run-dir $run `
  --workers 4 --requests-per-minute 60 `
  --model deepseek-v4-pro `
  --temperature 0.2 --top-p 1.0 `
  --top-logprobs 20 --max-tokens 4096 `
  --summary-output "$run/collection_summary_resume.json"

python scripts/audit_rejection_sampling.py `
  --run-dir $run --target 200 `
  --resume-summary "$run/collection_summary_resume.json" `
  --student-tokenizer Qwen/Qwen2.5-Coder-7B-Instruct `
  --student-revision $qwenRevision `
  --tokenizer-cache-dir data/mbpp_smoke_v1/.hf-cache `
  --max-length 4096
```

The output files are immutable: reruns compare wave inputs and aggregate
outputs with prior content rather than replacing them. Raw, normalized, and
verifier JSONL files are append-only and keyed by attempt ID.

## Collection result

| Metric | Result |
|---|---:|
| Selected unique tasks | 300 |
| Raw API attempts | 491 |
| API successes | 491/491 (100%) |
| Exact trace reconstruction | 491/491 (100%) |
| Source extraction | 491/491 (100%) |
| Syntax success | 491/491 (100%) |
| Import success | 491/491 (100%) |
| Pass@1 | 203/300 (67.67%) |
| Cumulative pass@2 | 206/300 (68.67%) |
| Cumulative pass@3 | 208/300 (69.33%) |
| Accepted on attempt 1 / 2 / 3 | 203 / 3 / 2 |
| Failed all three attempts | 92 |
| Mean earliest passing attempt | 1.0337 |
| Actual API attempts per accepted task | 2.3606 |

Attempt-level outcomes were:

- Attempt 1: 203 passed, 92 assertion failures, 5 runtime errors.
- Attempt 2: 3 passed, 89 assertion failures, 5 runtime errors.
- Attempt 3: 2 passed, 88 assertion failures, 4 runtime errors.

The 14 runtime-error attempts came from five problems and were generated-code
exceptions during the test phase (`TypeError`, `AttributeError`, or `KeyError`),
not compile/import or verifier infrastructure failures.

Blind retries at temperature 0.2 were highly correlated: 191 retry calls added
only five unique passes. This is an observed data-efficiency limitation, not a
reason to reorder or filter the accepted dataset after testing.

## Trace, usage, and cost

- Actual-token positions with logprobs: 29,744/29,744.
- Positions with exactly 20 candidates: 29,744/29,744.
- Top-20 candidate rows: 594,880.
- Missing actual byte arrays: 0.
- Missing/invalid top-candidate byte arrays: 0.
- Finish reasons: 491 `stop`.
- Prompt tokens: 116,218 total; median 236; p95 247.
- Completion tokens: 29,744 total; median response 44 tokens; p95 178.
- Total tokens: 145,962.
- Response size: median 142 UTF-8 bytes; p95 549; maximum 1,441.
- API latency: median 1.605 s; p95 3.206 s; maximum 7.458 s.
- Estimated total cost: CNY 0.3390028.
- Estimated cost per API attempt: CNY 0.00069043.
- Estimated cost per unique accepted task: CNY 0.00162982.

## Accepted and rejected outputs

- `accepted_unique.jsonl`: 208 records, original seeded task order, earliest
  passing attempt only.
- `accepted_first_200.jsonl`: first 200 records from that ordered accepted
  set, with no post-pass quality or length selection.
- `rejected_attempts.jsonl`: 283 actual attempts not selected for training.
- `rejected_tasks.jsonl`: all 92 tasks that failed all three attempts.
- `attempt_ledger.jsonl`: all 900 possible task/attempt slots.
- Pending ledger slots: 0.
- Attempts generated after a prior pass: 0.
- Duplicate selected, raw, normalized, verifier, or accepted IDs: 0.

Artifact hashes:

- `accepted_unique.jsonl`:
  `6412caaccec648c5cd5ee6053f7e5462ff44df7a939ce3d3e455a8888fcfb78e`.
- `accepted_first_200.jsonl`:
  `19014be66857e258925a20e211f853df6f5471e01c110a67364b2250ce5e5e95`.
- `audit_report.json`:
  `a09d5924afdbcf239622fa6e5fa31a3c111e30ce092e02547485a1cc0f908c81`.

## ALM preprocessing without training

The tokenizer is `Qwen/Qwen2.5-Coder-7B-Instruct` at revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`. No model weights were loaded.

| Diagnostic | All 208 accepted | First 200 |
|---|---:|---:|
| Preprocessing success | 208/208 | 200/200 |
| Sequence length min / median / p95 / max | 250 / 286 / 402 / 683 | 250 / 286 / 393 / 683 |
| ALM chunks min / median / p95 / max | 10 / 37 / 149 / 425 | 10 / 37 / 143 / 425 |
| 1:1 chunks | 10,833 | 10,380 |
| 1:N chunks | 60 | 60 |
| N:1 chunks | 307 | 295 |
| N:M chunks | 52 | 52 |
| Boundary drops | 0 | 0 |
| Zero-chunk records | 0 | 0 |
| Records over 4096 | 0 | 0 |
| Preprocessing errors | 0 | 0 |

## Resume verification

The complete collection command was run a second time against the same run
directory. It skipped 300/300 attempt-1 IDs, 97/97 attempt-2 IDs, and 94/94
attempt-3 IDs. Normalization and verification each skipped all 491 completed
IDs. No new API request, normalized row, verifier row, or duplicate ID was
created.

## Verification and unresolved risks

The automated test suite passed: 124 tests, with one pre-existing TRL
experimental API warning. The ALM trainer was not changed and student training
was not started.

Generated code is parsed and statically screened before execution, then run in
separate `python -I` compile/import/test child processes with a temporary
directory, sanitized environment, captured output, and wall-clock timeout.
Linux additionally receives process resource limits. Windows remains weaker
than a network-disabled Linux container: AST screening is not a complete
security boundary. Run future large verification jobs in an isolated,
network-disabled container on the Linux vGPU host.

MBPP's public tests are small and passing them does not prove broad semantic
correctness. Also, the 200 accepted records are enough for a pipeline/training
smoke experiment, not a broad coding corpus for a 7B model.

## Recommended next data-volume step

Do not spend the next budget on more low-temperature blind retries of these 92
failed problems: rounds two and three added only five tasks. For the next data
collection milestone, prioritize new unique, licensed training problems. First
collect the 74 MBPP `full/train` tasks excluded by this seeded sample, then run a
separate 1,000-accepted-example pilot from another established coding benchmark
training split under the same no-test-leakage verifier contract. Keep datasets
and provenance strata separate in metadata and audit them before any training.

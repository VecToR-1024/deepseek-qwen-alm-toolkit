# Multi-source clean-1500 offline KD data collection plan

> Date: 2026-07-31
> Status: three of five 50-task cloud calibrations complete; CodeContests and
> TACO are running sequentially in the background
> Branch: `codex/alm-offline-kd`
> Target student: `Qwen/Qwen2.5-Coder-7B-Instruct`
> Teacher: offline DeepSeek `deepseek-v4-pro` top-20 traces
> Training is explicitly out of scope for this plan.

## 1. Objective

Build a versioned dataset containing exactly 1,500 unique, executable,
format-clean teacher responses for the existing PyTorch/Transformers/PEFT ALM
training stack.

The primary future objective remains:

```text
total_loss = hard_sft_loss + alpha_alm * alm_loss
```

The collection must preserve:

- actual generated token bytes;
- actual generated-token logprobs;
- top-20 candidate bytes and logprobs;
- the complete unmodified teacher response;
- the official verifier result;
- ALM preprocessing diagnostics.

Existing raw attempts are immutable. Failed and format-ineligible records remain
available for audit and are never silently deleted.

## 1.1 Live calibration checkpoint

The authoritative run root is:

```text
${AUTODL_ROOT}/data/multisource_v4_clean1500_20260731/calibration_v1
```

Every completed official-test pass has also passed through the production Qwen
chat template, ALM example builder, and EOS-label audit. Current results:

| Source | API success | Official pass | Final clean | Clean/request |
|---|---:|---:|---:|---:|
| APPS | 50/50 | 19/50 | 13/50 | 26% |
| ODEX | 50/50 | 30/50 | 30/50 | 60% |
| xCodeEval | 50/50 | 33/50 | 20/50 | 40% |
| Completed subtotal | 150/150 | 82/150 | 63/150 | 42% |

Across the 82 official-test passes:

- EOS is present and supervised in 82/82 records;
- ALM preprocessing errors: 0;
- zero-valid-chunk records: 0;
- prompt/completion boundary drops: 0;
- one xCodeEval record exceeds 4,096 Qwen tokens;
- six APPS and thirteen xCodeEval records exceed the 20% comment-line limit;
  the over-length xCodeEval record is also in the latter set.

APPS has longer outputs and a lower clean yield than the preliminary quota
assumed. ODEX has excellent yield but only 207 eligible unique inputs.
xCodeEval is useful but its pool is capped at 106. Final quotas and request
counts remain deliberately unfrozen until the CodeContests and TACO
calibrations finish.

The remaining driver runs CodeContests first and TACO second, with 12 API
workers and 12 verifier workers. It automatically runs the same immutable
clean/ALM/EOS audit after each source. No student training is part of the
driver.

## 2. Why the earlier 845-request estimate was insufficient

The old 655-example training set contains 200 MBPP and 455 TACO examples.
However, 845 additional accepted examples cannot be obtained with only 845 API
requests because teacher correctness and output-format compliance are both
below 100%.

Observed historical results:

| Campaign | Result |
|---|---:|
| TACO breadth pass@1 | 412/1,000 = 41.20% |
| TACO pilot pass@1 | 39/100 = 39% |
| TACO pilot cumulative pass@3 | 47/100 = 47% |
| MBPP pass@1 | 203/300 = 67.67% |
| MBPP cumulative pass@3 | 208/300 = 69.33% |

Repeated attempts produced only modest gains. The default expansion strategy is
therefore breadth-first: request one answer for many unique tasks before
requesting another answer for a failed task.

## 3. Immutable trace rule

Do not remove Markdown fences, comments, prose, or trailing content from an
existing response and then reuse its ALM trace.

ALM token bytes and logprobs describe the exact original `response_text`.
Changing the response invalidates the teacher trace. Therefore:

- raw records remain unchanged;
- a clean-training eligibility layer selects or excludes whole records;
- fenced records may remain useful for analysis or a separate non-ALM
  experiment, but they are not eligible for the primary clean ALM dataset;
- SFT-only and SFT+ALM should use the same accepted dataset to avoid a dataset
  confound in the comparison.

## 4. Clean training eligibility contract

A record is eligible only if all of the following hold:

1. The API request succeeded.
2. Concatenated actual-token bytes exactly reproduce
   `response_text.encode("utf-8")`.
3. Actual-token logprobs and all requested top-20 candidates are present.
4. `finish_reason` is `stop`, not `length`.
5. The raw response contains no Markdown code fence.
6. The raw response itself parses as Python; validity after fence stripping is
   insufficient.
7. There is no prose before or after the program.
8. Top-level explanatory string expressions and function/class docstrings are
   absent.
9. Comment-only lines occupy at most 20% of source lines.
10. The response does not include benchmark tests, example calls, or a repeated
    problem statement.
11. The official benchmark tests pass in the isolated verifier.
12. The generated program uses no forbidden operation.
13. The Qwen chat-template sequence length is at most 4,096.
14. ALM preprocessing succeeds with at least one valid chunk.
15. Prompt/completion boundary drops are zero.
16. The Qwen assistant `<|im_end|>` is present and its hard-SFT label is not
    `-100`.

DeepSeek must not emit the literal Qwen `<|im_end|>` token inside Python source.
The raw DeepSeek response ends normally with `finish_reason=stop`; Qwen's chat
template adds `<|im_end|>`. Hard SFT supervises that template token, while ALM
continues to cover only actual teacher-response bytes.

## 5. Preliminary re-audit of the existing 655 records

The existing audit found:

| Source | Records | With comments | With fence |
|---|---:|---:|---:|
| MBPP | 200 | 7 | 0 |
| TACO | 455 | 369 | 268 |
| Total | 655 | 376 | 268 |

A preliminary local scan gives the following planning estimates:

| Rule | MBPP retained | TACO retained | Total retained |
|---|---:|---:|---:|
| Existing 4,096-token training set | 200 | 455 | 655 |
| Exclude every fenced response | 200 | 187 | 387 |
| Also require comment-line ratio <=20% | about 196 | about 147 | about 343 |
| Ban every comment | 193 | 62 | 255 |

The recommended contract permits sparse code-local comments but rejects
comment-heavy explanatory answers. Completely banning comments discards too
much otherwise valid code.

The versioned eligibility audit is now authoritative:

| Source | Audited | Retained | Excluded |
|---|---:|---:|---:|
| MBPP | 200 | 196 | 4 |
| TACO | 455 | 140 | 315 |
| Total | 655 | 336 | 319 |

The exclusions are non-exclusive by reason: 268 contain Markdown fences (and
therefore also fail raw-source parsing), 157 exceed the 20% comment-line
threshold, and two contain docstrings. The original 655 records and traces
remain unchanged.

## 6. Provisional final source composition

| Source | Existing clean | New clean target | Final target |
|---|---:|---:|---:|
| MBPP | 196 | 0 | 196 |
| TACO | 140 | 150 | 290 |
| APPS train | 0 | 450 | 450 |
| CodeContests train | 0 | 430 | 430 |
| ODEX English | 0 | 90 | 90 |
| xCodeEval compact | 0 | 44 | 44 |
| Total | 336 | 1,164 | 1,500 |

These are calibration placeholders, not frozen quotas. APPS and CodeContests
are the designated sources that absorb shortfalls because their candidate
pools are much larger. The local pinned-source audit found:

- ODEX has 439 English rows but only 333 unique task IDs; 207 unique tasks pass
  the deterministic standard-library and executable-test ingestion contract.
- xCodeEval compact has 106 unique executable tasks.
- Consequently, the earlier targets of 250 accepted ODEX examples and 100
  accepted xCodeEval examples were not operationally credible.
- ODEX exposes only an official `test` split. xCodeEval compact is backed by
  its raw validation file. Both uses are explicitly recorded as project
  exceptions in each task and import manifest.

The placeholder mix reduces TACO from 69.5% of the old dataset to approximately
19.3% of the new dataset. Final quotas are locked only after the per-source
calibration.

## 7. Provisional request volume

The initial bulk plan assumes a conservative source-dependent clean pass@1 of
roughly 20-35% and includes a buffer:

| Source | New clean target | Provisional unique first attempts |
|---|---:|---:|
| APPS | 450 | calibration-controlled |
| CodeContests | 430 | calibration-controlled |
| ODEX | 90 | at most 207 unique eligible inputs |
| TACO train shards 1-8 | 150 | calibration-controlled |
| xCodeEval compact | 44 | at most 106 unique inputs |
| Total | 1,164 | computed after calibration |

The earlier 4,930-6,500-attempt envelope remains only a budget ceiling for
planning. It is no longer treated as a source allocation. If the target is
still projected to be short after 6,500 attempts, pause and inspect source
quality, verifier compatibility, and the clean-output rate. Do not silently
reduce the target.

At the old TACO average of approximately CNY 0.010 per attempt, 4,930 attempts
would be about CNY 49.30. APPS and CodeContests answers may be longer, so a
conservative token-cost range is CNY 50-100. The API is charged to the
project's existing provider account; this remains a usage estimate rather than
a reimbursement request or provider billing statement.

## 8. Step-by-step implementation and collection

### Step 1: Freeze v3 inputs

Create a new versioned run root, for example:

```text
data/multisource_v4_clean1500_20260731/
```

Acceptance criteria:

- existing v3 files and hashes are unchanged;
- new commands never overwrite prior artifacts;
- all new outputs are append-only or written to a new versioned path.

### Step 2: Build the eligibility audit

Audit every existing record and emit:

```json
{
  "id": "example_attempt_id",
  "eligible": false,
  "reasons": [
    "markdown_fence",
    "comment_ratio_above_limit"
  ],
  "raw_trace_preserved": true
}
```

Required reason codes include:

- `markdown_fence`;
- `finish_reason_length`;
- `prose_outside_code`;
- `docstring`;
- `comment_ratio_above_limit`;
- `syntax_error`;
- `official_test_failure`;
- `sequence_over_4096`;
- `alm_preprocessing_failure`;
- `zero_alm_chunks`;
- `boundary_drop`.

Expected outputs:

```text
existing_v3_eligibility.jsonl
existing_v3_retained.jsonl
existing_v3_excluded.jsonl
existing_v3_clean_audit.json
existing_v3_clean_audit.md
```

Checkpoint:

- every fenced record is excluded;
- no `response_text`, token byte, or logprob is changed;
- the authoritative existing clean count is used to revise source quotas.

### Step 3: Version the teacher prompt

Create a new prompt contract such as `deepseek.python.clean.v2` with explicit
requirements:

```text
Your response must begin directly with Python source code.
Do not use Markdown fences.
Do not include prose before or after the code.
Do not include comments or docstrings unless required for correctness.
End immediately after the final Python statement.
```

The eligibility gate remains authoritative; the prompt is not trusted by
itself.

Required regression tests:

- a fenced response is ineligible even when extracted code passes;
- raw plain Python proceeds to verification;
- prose and top-level explanatory strings are rejected;
- sparse code-local comments can pass;
- template EOS remains supervised;
- ALM still uses only original teacher bytes.

### Step 4: Add source adapters

Implemented and tested:

1. APPS train;
2. CodeContests train;
3. ODEX English;
4. xCodeEval compact;
5. TACO train shards 1-8.

Every adapter must:

- use only the designated training split;
- preserve the original task identifier and source metadata;
- keep tests and reference answers out of the teacher prompt;
- emit the common versioned coding-task schema;
- provide deterministic seeded selection;
- avoid task-ID collisions across sources.

All five completed adapters use exact pinned revisions and the common
`coding.task.multisource.v1` schema. `scripts/import_multisource.py` publishes
immutable task JSONL plus an import summary containing provenance, split,
license, ordered IDs, and an ordered-ID SHA-256. The collector, one-attempt
breadth aggregator, rejection-sampling records, and candidate merge now
preserve source-specific names rather than treating every new source as TACO.
The automated suite currently passes 256 tests.

### Step 5: Run a 50-task calibration per source

Select 50 unique tasks from each completed source and make one generation
attempt per task: 250 API attempts total. The TACO adapter uses only train
shards 1-8, gives every task a shard-qualified ID, and excludes the previously
attempted shard 0 by construction.

Measure per source:

- API success;
- byte-exact trace reconstruction;
- raw plain-Python rate;
- fence and prose rates;
- clean pass@1;
- official-test pass rate;
- 4,096-token overflow rate;
- ALM preprocessing success;
- prompt and completion token distributions;
- API latency and cost per clean accepted example.

Checkpoint:

- pause a source if clean pass@1 is below 10%;
- distinguish verifier/environment failures from teacher failures;
- fix the prompt or request contract before scaling if fence rate exceeds 10%.

### Step 6: Lock request counts from calibration

For each source, compute a conservative request count using the 95% Wilson
lower confidence bound of clean pass@1:

```text
planned_unique_tasks =
ceil(clean_target / clean_pass_rate_lower_bound * 1.10)
```

Use the calibrated result to replace the provisional 4,900-task allocation
before the bulk campaign starts.

### Step 7: Run breadth-first bulk collection

Run one attempt for each selected unique task in waves:

```text
Wave 1: 1,000 unique tasks
Wave 2: 1,000 unique tasks
Wave 3: 1,000 unique tasks
Wave 4: 1,000 unique tasks
Wave 5: remaining planned tasks
```

Produce an incremental audit every 500 attempts:

- clean accepted count and source quota progress;
- clean pass@1;
- fence, length, assertion, runtime, and infrastructure failures;
- tokens, latency, estimated cost, and projected final count.

Attempt IDs must encode source, task ID, and attempt number. Resume must skip
completed attempt IDs without duplicating raw, normalized, verifier, or
eligibility records.

### Step 8: Handle any remaining shortfall

Use this order:

1. sample more unique tasks from the same source;
2. use a predeclared fallback source if the source pool is exhausted;
3. use attempt 2 only for a source with limited remaining unique tasks;
4. never send tests, stderr, tracebacks, or verifier feedback to the teacher;
5. do not use attempt 3 unless an attempt-2 audit demonstrates worthwhile
   marginal yield.

Eligible attempt-2 categories may include assertion failure, runtime error,
timeout, and transient API error. Fenced/prose responses, length termination,
forbidden operations, and over-4,096 responses are not automatically retried.

### Step 9: Build the deterministic clean dataset

For each unique task, retain only its earliest clean passing attempt.

Selection rules:

- follow the original seeded task order within each source;
- do not select by response length, ALM chunks, or perceived code quality after
  official tests pass;
- keep excess clean records in a reserve set;
- deterministically interleave sources in the final 1,500 rather than placing
  each source in one contiguous block.

Expected final outputs:

```text
accepted_clean_all.jsonl
accepted_train_1500.jsonl
accepted_reserve.jsonl
rejected_attempts.jsonl
dataset_manifest.json
audit_report.json
audit_report.md
```

### Step 10: Complete the pre-training audit

The dataset is ready only when:

- there are exactly 1,500 unique problem IDs;
- there are zero code fences and zero surrounding explanations;
- every accepted response has `finish_reason=stop`;
- 1,500/1,500 responses reconstruct from teacher token bytes;
- 1,500/1,500 pass official tests;
- 1,500/1,500 pass ALM preprocessing;
- there are zero zero-chunk examples and zero boundary drops;
- no sequence exceeds 4,096;
- 1,500/1,500 examples contain supervised Qwen `<|im_end|>`;
- exact and near-duplicate checks are reported;
- source, interface, difficulty, length, trace, and cost distributions are
  recorded;
- all final artifacts and their SHA-256 values are frozen in the manifest.

Stop for human review after this audit. Do not start student training
automatically.

## 9. Where to run collection

### 9.1 GPU does not accelerate teacher API collection

DeepSeek inference happens at the provider. The local process mainly:

- sends HTTP requests;
- waits for API responses;
- appends JSONL records;
- normalizes traces;
- executes generated Python in isolated child processes;
- tokenizes accepted records for diagnostics.

These are network-, CPU-, and disk-bound operations. Turning on a GPU instance
does not materially reduce DeepSeek response latency and does not increase the
provider's rate limit. GPU count must never be used as the API worker count.

### 9.2 Recommended deployment

Use the normal cloud instance with 20 CPU cores and the persistent data disk
for the authoritative pipeline:

| Work | Recommended location |
|---|---|
| Code changes and unit tests | Local Windows/Miniforge |
| 50-task source pilots | Normal cloud mode, 20 CPU cores |
| Bulk API collection | Normal cloud mode, persistent data disk |
| Isolated verification | Normal cloud mode, 20 CPU cores |
| Audit and ALM preprocessing diagnostics | Normal cloud mode |
| Student teacher-forcing/training | Same instance with GPU enabled |
| Final benchmark generation | Same instance with GPU enabled |
| Benchmark scoring | Same instance CPU, or no-GPU mode if practical |

The platform's no-GPU mode supplies only approximately half a CPU core. It is
appropriate for keeping the persistent disk mounted, transferring files,
checking logs, and other lightweight administration. It is not the execution
target for bulk JSON parsing, concurrent isolated verification, or ALM
preprocessing.

The normal 20-core cloud mode is preferable for the bulk campaign because:

- Linux isolation and resource limits are stronger than the current Windows
  child-process verifier;
- a 20-core host can verify several programs concurrently;
- raw top-20 JSONL files remain on the same persistent data disk later used for
  training;
- it avoids uploading several GB of traces from the local machine to the vGPU
  host;
- collection, verification, audit, training, and benchmark artifacts remain in
  one versioned environment.

The GPU will be mostly idle while waiting for DeepSeek API responses. This is
an accepted cost trade-off for the simpler, faster 20-core workflow; the speed
gain comes from CPU capacity and eliminating data transfer, not from GPU
inference.

### 9.3 Concurrency calibration

Do not immediately launch dozens of workers. Run a short concurrency ladder
with the same task mix:

```text
100 tasks at 4 API workers
100 tasks at 8 API workers
100 tasks at 12 API workers
100 tasks at 16 API workers, only if 12 remains stable
```

Compare:

- completed attempts per hour;
- p50 and p95 API latency;
- HTTP 429/5xx and timeout rates;
- duplicate/resume behavior;
- JSONL append integrity.

Select the lowest worker count within 5% of peak stable throughput. The
expected practical range is 8-12 API workers if the provider permits it.
Twenty CPU cores do not imply 20 API workers because provider rate limits and
large top-20 JSON responses can become the bottleneck first.

Use a separate verifier worker limit based on CPU capacity, initially:

```text
verifier_workers = 12
```

Test 16 verifier workers only after the 12-worker run shows no increase in
timeout rate, memory pressure, or disk contention. Keep at least four logical
cores available for the collector, normalizer, operating system, and audit
processes. A single collector process should own each append-only output file;
multiple independent processes must not concurrently append to the same JSONL.

## 10. Decision summary

```text
Re-audit old 655
-> retain 336 clean examples
-> implement and test five source adapters
-> deploy immutable import and breadth-collection CLIs to the 20-core host
-> calibrate 50 unique tasks per source
-> compute source request counts from clean pass@1 lower bounds
-> keep approximately 4,930-6,500 breadth-first attempts as a budget ceiling
-> build 1,500 deterministic clean accepted examples plus reserve
-> complete trace/EOS/ALM/duplicate audit
-> stop before training
```

Bulk collection, verification, and audit should run on the normal cloud
instance with 20 CPU cores and the persistent data disk. Half-core no-GPU mode
is reserved for lightweight administration rather than the data pipeline.

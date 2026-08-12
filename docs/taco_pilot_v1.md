# TACO stdin/stdout collection pilot v1

## Outcome

The local pilot is complete. It selected 100 tasks, made at most three blind
DeepSeek attempts per task, and produced 47 unique accepted training
candidates. No student training was started and the ALM trainer was not
modified.

Collection and verification ran from approximately 10:13 to 10:44 on
2026-07-28 (about 30 minutes). The final in-memory aggregation initially
duplicated large raw records and reached 5.7GB RAM; it was stopped after all
append-only attempt files were durable, then rerun in lightweight
`--aggregate-only` mode without another API request.

## Dataset provenance and selection

| Field | Value |
|---|---|
| Dataset | `BAAI/TACO` |
| Split | `train` only |
| Revision | `d593ed0a2becbbc952230bb89be09189bf1056dc` |
| Arrow shard | `train/data-00000-of-00009.arrow` |
| Selection | random from eligible rows in this one shard |
| Seed | `20260728` |
| Selected task count | 100 |
| Ordered ID SHA-256 | `708ba03d5a0e81c3d0ec7e5c0ae19914a0fde06ad93a0a0f7251136e722ab988` |

The importer downloads Arrow directly and does not execute the repository's
custom loader. JSON columns are parsed with `json.loads`. The dataset card and
repository identify Apache-2.0, while upstream problem licenses vary and the
card calls HackerRank rights unknown; HackerRank rows were excluded. The
historical loader's license metadata differs from the card, so per-source
redistribution rights remain an unresolved provenance risk.

This pilot is not representative of all 25,443 TACO train rows because it uses
one of nine train shards. It also excludes call-based, picture-bearing, and
non-string-test rows.

## Data flow

```text
pinned TACO Arrow shard
  -> versioned coding.task.taco.v1 records
  -> two-message stdin/stdout DeepSeek prompt (tests omitted)
  -> append-only raw top-20 trace
  -> exact UTF-8 byte reconstruction and normalization
  -> conservative Python extraction
  -> compile + fresh child process per official input/output case
  -> first passing attempt per task
  -> ALM preprocessing diagnostics
```

The teacher prompt never contains official inputs, outputs, reference
solutions, verifier failures, stderr, or tracebacks. Attempts two and three use
the same prompt contract and receive no feedback.

## Generation and verification contract

- Model: `deepseek-v4-pro`
- Temperature: `0.2`
- `top_p`: `1.0`
- `logprobs`: true
- `top_logprobs`: 20
- `max_tokens`: 4096
- Thinking: disabled
- Workers: 4
- Rate limit: 120 request starts/minute
- Test timeout: 8 seconds per case
- Output comparison: normalize CRLF/CR to LF, strip outer whitespace, then
  exact string equality

The Windows verifier uses `python -I`, a fresh child process and temporary
directory per test, a sanitized environment, static forbidden-operation
checks, captured output, and timeouts. It is not a security sandbox because the
machine has no WSL or Docker.

## Results

### Unique-task outcomes

| Metric | Result |
|---|---:|
| Pass@1 | 39/100 (39%) |
| Cumulative pass@2 | 41/100 (41%) |
| Cumulative pass@3 | 47/100 (47%) |
| Accepted at attempt 1 / 2 / 3 | 39 / 2 / 6 |
| Failed all three attempts | 53 |
| Mean earliest passing attempt | 1.298 |
| Actual calls per accepted task | 4.681 |

### Attempt funnel and trace

| Metric | Result |
|---|---:|
| Raw API attempts | 220 |
| API success | 220/220 |
| Exact trace reconstruction | 220/220 |
| Source extraction and syntax success | 177/220 |
| Passing attempts | 47/220 |
| Actual-token logprobs | 298,553/298,553 |
| Positions with all top-20 candidates | 298,553/298,553 |
| Missing actual-token byte arrays | 0 |
| Invalid top-candidate byte arrays | 0 |
| Duplicate task/attempt IDs | 0 |

Failure outcomes, excluding passes:

| Category | Count |
|---|---:|
| `assertion_failure` | 84 |
| `runtime_error` | 45 |
| `extraction_error` | 41 |
| `syntax_error` | 2 |
| `timeout` | 1 |

All 41 extraction failures and both syntax failures had
`finish_reason=length`. There were 48 length-terminated responses in total.
The other 172 stopped normally.

### Token usage and estimated API cost

| Metric | Result |
|---|---:|
| Prompt tokens | 146,426 |
| Completion tokens | 298,553 |
| Total tokens | 444,979 |
| Estimated total | ¥1.9800 |
| Estimated per API attempt | ¥0.00900 |
| Estimated per accepted task | ¥0.04213 |

The estimate uses ¥0.025/M cache-hit input tokens, ¥3/M cache-miss input
tokens, and ¥6/M output tokens.

### ALM preprocessing without training

All 47 accepted records preprocess successfully with
`Qwen/Qwen2.5-Coder-7B-Instruct` revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`.

| Metric | Result |
|---|---:|
| Success | 47/47 |
| Sequence length min / median / p95 / max | 405 / 1,151 / 3,804 / 4,765 |
| ALM chunks min / median / p95 / max | 57 / 359 / 3,096 / 4,032 |
| 1:1 / 1:N / N:1 / N:M groups | 29,068 / 266 / 324 / 52 |
| Prompt/completion boundary drops | 0 |
| Zero-chunk examples | 0 |
| Records above 4096 tokens | 1 |

The over-limit record remains unchanged in the accepted dataset. Training code
must apply the existing length policy rather than silently altering this
collection.

## Known v1 issues

1. `fn_name` absence was not sufficient to identify true executable stdin
   data. Thirteen selected GeeksForGeeks tasks used pseudo-input strings such
   as `L = 1, R = 10`; 35 runtime-error attempts came from that source. They
   were rejected, so accepted training candidates are not contaminated, but
   future v2 selection should exclude this source or add a separately tested
   adapter.
2. Forty-eight attempts reached 4096 generated tokens. Raising the limit may
   improve coverage but increases latency and cost; it must be a new versioned
   generation policy, not an in-place retry of v1.
3. The conservative output comparator can create false negatives relative to
   permissive historical TACO/APPS evaluation.
4. Local Windows execution is not a security boundary. Move verification to a
   network-disabled Linux container/VM before large-scale collection.
5. This is a single-shard pilot. Do not quote its 47% acceptance rate as a
   full-TACO estimate.

## Reproduction

The API key is read only from `DEEPSEEK_API_KEY`.

```powershell
conda run -n topk-distill python scripts/import_taco.py `
  --output data/taco_pilot_v1/selected_tasks_100.jsonl `
  --summary-output data/taco_pilot_v1/import_summary.json `
  --limit 100 --selection random --seed 20260728 `
  --cache-dir data/taco_pilot_v1/hf_cache

conda run -n topk-distill python scripts/collect_taco_pilot.py `
  --tasks data/taco_pilot_v1/selected_tasks_100.jsonl `
  --expected-tasks 100 --target 100 `
  --run-dir data/taco_pilot_v1/run100 `
  --workers 4 --requests-per-minute 120 --phase-timeout 8

conda run -n topk-distill python scripts/audit_taco_pilot.py `
  --run-dir data/taco_pilot_v1/run100 `
  --expected-tasks 100 --target 100 `
  --output-prefix audit_report_v2
```

Re-running collection is append-only and skips completed attempts. If only
aggregation needs to be rebuilt, add `--aggregate-only`; that mode does not
instantiate the API client.

## Artifact paths

- Selected tasks: `data/taco_pilot_v1/selected_tasks_100.jsonl`
- Raw attempts: `data/taco_pilot_v1/run100/raw_attempts.jsonl`
- Normalized attempts: `data/taco_pilot_v1/run100/normalized_attempts.jsonl`
- Verifier results: `data/taco_pilot_v1/run100/verifier_attempts.jsonl`
- Accepted candidates: `data/taco_pilot_v1/run100/accepted_unique.jsonl`
- Rejected task/attempt indexes:
  `data/taco_pilot_v1/run100/rejected_tasks.jsonl` and
  `data/taco_pilot_v1/run100/rejected_attempts.jsonl`
- Machine-readable audit:
  `data/taco_pilot_v1/run100/audit_report_v2.json`
- Markdown audit: `data/taco_pilot_v1/run100/audit_report_v2.md`

Large data artifacts remain outside Git. The code and documentation are on
branch `codex/alm-offline-kd`.

# TACO 8192-token blind length retry v2

## Outcome

The versioned length-retry experiment is complete. It retried the eligible
TACO v1 responses that had ended at the 4096-token limit, using the identical
teacher prompt and no verifier feedback, with `max_tokens=8192`.

All 42 API responses had complete, byte-exact traces. Three retry attempts
passed the official tests, representing two newly accepted tasks. The combined
TACO candidate set therefore increased from 47 to 49 unique tasks.

The retry cost was approximately CNY 1.612. Twenty-eight of the 42 responses
still ended at the new 8192-token limit. This experiment does not justify
another blanket output-limit increase: the marginal yield was 2/18 eligible
unaccepted tasks, or 11.11%, at approximately CNY 0.806 per newly accepted
task. No student training was started and the ALM trainer was not modified.

## Selection and invariants

The source is the immutable TACO v1 pilot described in
`docs/taco_pilot_v1.md`.

| Field | Value |
|---|---|
| v1 selected tasks | 100 |
| v1 length-terminated attempts | 48 across 21 tasks |
| Already accepted v1 tasks with a length attempt | 3 |
| Eligible unaccepted tasks | 18 |
| Canonical retry attempts | 42 |
| Selection order | Original v1 task order, then attempt number |
| Teacher feedback | None |
| New attempt ID suffix | `__length_retry_v2` |

The six length-terminated attempts belonging to three already accepted tasks
were not retried. Each retry refers to its original v1 attempt and preserves
the original task. The teacher receives exactly the same two-message prompt as
v1; official tests, prior outputs, `finish_reason`, verifier failures, stderr,
and tracebacks remain hidden.

The v2 policy changes only the attempt identity and output limit. Generation
remains:

```text
model = deepseek-v4-pro
temperature = 0.2
top_p = 1.0
logprobs = true
top_logprobs = 20
max_tokens = 8192
thinking = disabled
```

The collector, normalizer, source extractor, and isolated stdin/stdout verifier
are reused from v1. Raw, normalized, and verifier files remain append-only and
resumable.

## Smoke runs

Two separate smoke directories were used before the authoritative campaign:

- `smoke3` retried the first three canonical attempts. Those attempts happened
  to belong to one very hard problem; all three produced valid traces, reached
  8192 tokens, and failed conservative extraction.
- `smoke3_unique` retried one attempt from each of the first three distinct
  problems. All three traces were valid; two stopped normally but failed
  assertions, and one again reached 8192 tokens. None passed.

These smoke runs are independent artifacts and are not part of the
authoritative 42-attempt result.

An initial `run42` prepare invocation exposed that an absolute Windows path
containing the Chinese workspace name was not stable across process encoding
boundaries. It failed before API client creation and contains no raw attempts.
The manifest now stores a stable repository-relative POSIX path. The
authoritative campaign is `run42_authoritative`.

## Authoritative results

### Attempt funnel

| Metric | Result |
|---|---:|
| Planned / raw / normalized / verified | 42 / 42 / 42 / 42 |
| API success | 42/42 |
| Exact trace reconstruction | 42/42 |
| Passing retry attempts | 3/42 (7.14%) |
| Newly accepted unique tasks | 2/18 (11.11%) |
| Combined accepted TACO tasks | 49 |
| `finish_reason=stop` | 14 |
| `finish_reason=length` | 28 |

The new accepted records are:

- `taco_train_000364__attempt_2__length_retry_v2`
- `taco_train_001302__attempt_1__length_retry_v2`

The third passing attempt belongs to one of those same tasks, so only the
earliest passing retry is retained for the combined unique dataset.

Failure outcomes, excluding the three passes:

| Category | Count |
|---|---:|
| `extraction_error` | 23 |
| `assertion_failure` | 13 |
| `runtime_error` | 2 |
| `timeout` | 1 |

### Trace and cost

| Metric | Result |
|---|---:|
| Actual generated positions | 267,471 |
| Actual-token logprobs available | 267,471/267,471 |
| Positions with top-20 | 267,471/267,471 |
| Top candidates preserved | 5,349,420 |
| Prompt tokens | 34,749 |
| Completion tokens | 267,471 |
| Total tokens | 302,220 |
| Estimated total cost | CNY 1.6123498 |
| Cost per retry | CNY 0.0383893 |
| Cost per new unique task | CNY 0.8061749 |

The cost estimate uses CNY 0.025/M cache-hit input tokens, CNY 3/M cache-miss
input tokens, and CNY 6/M output tokens.

## ALM preprocessing without training

Both newly accepted records and all 49 combined records were processed with
`Qwen/Qwen2.5-Coder-7B-Instruct` revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`.

| Metric | New 2 | Combined 49 |
|---|---:|---:|
| Preprocessing success | 2/2 | 49/49 |
| Sequence min / median / p95 / max | 1217 / 2092 / 2967 / 2967 | 405 / 1162 / 3804 / 4765 |
| Chunks min / median / p95 / max | 471 / 1343 / 2215 / 2215 | 57 / 366 / 3096 / 4032 |
| 1:1 groups | 2,612 | 31,680 |
| 1:N groups | 1 | 267 |
| N:1 groups | 43 | 367 |
| N:M groups | 30 | 82 |
| Boundary drops | 0 | 0 |
| Zero-chunk examples | 0 | 0 |
| Above 4096 tokens | 0 | 1 |

The one over-limit combined record is the existing v1 record
`taco_train_000902__attempt_3`; v2 does not alter it.

## Decision

Do not run a blanket 16K/32K retry over these remaining attempts. Most of the
additional tokens were spent on responses that still did not terminate, and
only two unique tasks were recovered. Future TACO expansion should first:

1. improve the versioned task-eligibility policy and exclude pseudo-stdin
   sources;
2. use a network-disabled Linux container or VM for verification;
3. sample new, independently selected tasks rather than repeatedly extending
   already rambling generations;
4. retain the 49 accepted records as a small diagnostic set, not as evidence
   that TACO collection is ready to scale.

## Reproduction

The API key is read only from `DEEPSEEK_API_KEY`. The authoritative data already
exists; use `--aggregate-only` to rebuild indexes without another API request.

```powershell
conda run -n topk-distill python scripts/run_taco_length_retry.py `
  --v1-run-dir data/taco_pilot_v1/run100 `
  --run-dir data/taco_length_retry_v2/run42_authoritative `
  --aggregate-only

conda run -n topk-distill python scripts/audit_taco_length_retry.py `
  --run-dir data/taco_length_retry_v2/run42_authoritative
```

To prepare a separate smoke manifest without making an API request:

```powershell
conda run -n topk-distill python scripts/run_taco_length_retry.py `
  --v1-run-dir data/taco_pilot_v1/run100 `
  --run-dir data/taco_length_retry_v2/example_smoke `
  --smoke-problems 3 --prepare-only
```

## Artifact paths

- Campaign manifest:
  `data/taco_length_retry_v2/run42_authoritative/campaign_manifest.json`
- Retry task records:
  `data/taco_length_retry_v2/run42_authoritative/retry_tasks_42.jsonl`
- Raw attempts:
  `data/taco_length_retry_v2/run42_authoritative/raw_retries.jsonl`
- Normalized traces:
  `data/taco_length_retry_v2/run42_authoritative/normalized_retries.jsonl`
- Verifier results:
  `data/taco_length_retry_v2/run42_authoritative/verifier_retries.jsonl`
- Newly accepted:
  `data/taco_length_retry_v2/run42_authoritative/newly_accepted_unique.jsonl`
- Combined accepted:
  `data/taco_length_retry_v2/run42_authoritative/combined_accepted_unique.jsonl`
- Retry outcome ledger:
  `data/taco_length_retry_v2/run42_authoritative/retry_outcomes.jsonl`
- Machine-readable audit:
  `data/taco_length_retry_v2/run42_authoritative/audit_report.json`
- Markdown audit:
  `data/taco_length_retry_v2/run42_authoritative/audit_report.md`

Large data artifacts remain outside Git.

Key SHA-256 values:

```text
newly_accepted_unique.jsonl
8ce2a8d11044d1671032762635563a9c4b16ded814eda5554333e08c94df4987

combined_accepted_unique.jsonl
3607284de2c3e83b7bca99e4ec00c8699139a3167c52381f2134e132ed810603

audit_report.json
69997a9de26aba5dc2cedd8865b76ae244dd68a4ac66085777c0c472297a8b7c
```

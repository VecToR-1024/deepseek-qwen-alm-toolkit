# Qwen3-0.6B actual-only data acceptance — 2026-08-11

This document preserves the useful acceptance facts without copying private
training records or raw API traces into the toolkit.

## Outcome

| Gate | Result |
|---|---:|
| Selected / durable raw attempts | 4,500 / 4,500 |
| API success | 4,492 (99.82%) |
| Byte reconstruction | 4,492 / 4,492 |
| Official-test pass | 2,298 (51.07% per attempt) |
| Clean ALM candidates before deduplication | 1,656 |
| Exact unique clean problem texts | **1,619** |
| Exact duplicates | 37 |

The 4,492 structurally valid traces contain 4,876,789 actual teacher tokens.
Missing/invalid byte arrays, missing/non-finite actual-token logprobs, response
reconstruction failures, and duplicate record IDs were all zero. Top-k rows
were absent by design because this wave used the `actual_only` profile.

Eight durable API errors were retained: six connection errors and two timeout
errors. They were transport failures, not malformed traces, and no verifier
feedback was sent back to the teacher.

## Clean yield by source

| Source/lane group | Clean records |
|---|---:|
| TACO (all included waves) | 770 |
| CodeContests | 457 |
| APPS | 71 |
| Open-R1 Codeforces | 358 |
| Total | **1,656** |

Among 642 official-test passes excluded from the clean set, reasons could
overlap: 628 exceeded the 20% comment-line threshold, 22 contained Markdown
fences, 24 exceeded 4096 Qwen3 tokens, 22 had raw-response syntax failures,
three ended with `finish_reason=length`, and two contained docstrings.

Text-cleaning a response invalidates its original ALM bytes/logprobs. Such a
record may become SFT-only data, but must not be silently put back into the ALM
set.

## Qwen3 ALM contract

Tokenizer/model contract:

- model: `Qwen/Qwen3-0.6B`
- revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- chat-template kwargs: `{"enable_thinking": false}`
- maximum training length: 4096

| ALM/EOS metric | Result |
|---|---:|
| EOS supervised | 1,656 / 1,656 |
| Preprocessing errors | 0 |
| Zero-chunk records | 0 |
| Prompt/completion boundary drops | 0 |
| Records over 4096 | 0 |
| Sequence length min / median / p95 / max | 362 / 841 / 1,464 / 2,458 |
| ALM chunks min / median / p95 / max | 4 / 165 / 482 / 1,389 |
| 1:1 / 1:N / N:1 / N:M chunks | 334,254 / 1,517 / 4,181 / 738 |

## Remaining freeze gates

The 1,619 exact-unique records are candidates, not a globally frozen training
dataset. Before training, recover the cloud-only prior freeze, deduplicate all
sources together, run benchmark-overlap checks against pinned HumanEval+,
MBPP+, and LiveCodeBench inputs, record hashes, and rerun the Qwen3 contract
audit on the exact final JSONL.

No training was started as part of this acceptance audit.

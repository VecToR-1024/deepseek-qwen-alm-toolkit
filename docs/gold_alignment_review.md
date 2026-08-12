# GOLD byte-span alignment review

Reviewed against the installed TRL 1.7.1 implementation and TRL `main` on
2026-07-20. The implementation in this repository ports only the alignment
geometry. The trainer, strict `soft_positions`, top-20 + tail-bucket loss, and
offline DeepSeek trace format are unchanged.

## Where GOLD aligns tokenizers

The current code is split across two places:

- `trl/experimental/utils.py::encode_with_byte_offsets` makes one fast-tokenizer
  backend encoding call, then converts its character offsets to UTF-8 byte
  offsets. Its normalization helpers split repeated offsets produced by
  ByteLevel/byte-fallback pieces and reject offsets that remain overlapping.
- `trl/experimental/gold/gold_trainer.py::ULDLoss._align_by_byte_offsets`
  walks student and teacher offsets together. It advances whichever current
  token ends first and closes a group only when both streams reach the same
  byte end. Those common boundaries naturally create 1:1, 1:N, N:1, and N:M
  groups.

For offline examples, GOLD tokenizes the fully rendered chat once and slices
completion-relative offsets from that encoding. For sampled token IDs, GOLD
derives byte-piece lengths directly and explicitly avoids a decode/re-encode
round trip. These are the parts worth preserving.

Official references:

- [TRL GOLDTrainer documentation](https://huggingface.co/docs/trl/main/gold_trainer)
- [Current GOLD trainer source](https://github.com/huggingface/trl/blob/main/trl/experimental/gold/gold_trainer.py)
- [Current byte-offset utilities](https://github.com/huggingface/trl/blob/main/trl/experimental/utils.py)

## Strict baseline versus span grouping

| Property | Existing strict aligner | New span diagnostic |
|---|---|---|
| Teacher source | Offline DeepSeek top-20 rows | Same rows; actual provider bytes only |
| Student tokenization | Re-encode prefix + candidate | One full-text backend encoding with offsets |
| Primary coordinate | Exact token-prefix stability | Response-relative UTF-8 bytes |
| Accepted shapes | One teacher position to one student token | 1:1, 1:N, N:1, N:M actual-token groups |
| Candidate target | Loss-ready student token ID | Not produced |
| Training use | Current `soft_positions` and tail bucket | Diagnostics only |
| Failure behavior | Conservative omission/tail | Strict result remains the fallback |

`CrossTokenizerAligner` is in
`src/deepseek_distill/cross_tokenizer_aligner.py`. Its tokenizer boundary
requires token IDs and offsets from the same call. The included Hugging Face
adapter supports Qwen-style fast ByteLevel tokenizers. Zero-width BOS/special
tokens are ignored. A token fused across the prompt/response boundary is
clipped on the diagnostic byte axis and counted explicitly in
`boundary_clipped_student_positions`.

The module never calls `decode()` and does not use decoded token-string
equality. DeepSeek's returned byte arrays are authoritative on the teacher
side, including when one teacher token contains only part of a multibyte UTF-8
character.

Pass `--alignment-diagnostics` to `scripts/align_tokenizers.py` to run this
analysis during normal JSONL preprocessing. The option is disabled by default.
When enabled, each record receives an `alignment_diagnostics` object and the
CLI summary receives aggregate diagnostics. `training_alignment` is always
`strict_1_to_1`; a span failure is recorded as `strict_fallback` instead of
failing or changing the training record.

## Coverage and probability-mass comparison

`compare_strict_and_span` reports separate quantities so geometric coverage is
not confused with a valid training objective:

- `strict_position_coverage`: teacher positions emitted as current strict
  `soft_positions`.
- `span_position_coverage`: teacher positions contained in valid shared-byte
  groups.
- `strict_retained_topk_mass`: top-20 mass mapped to unique one-token student
  targets by the strict aligner.
- `span_covered_topk_mass`: top-20 mass located at byte-span-covered teacher
  positions. This is diagnostic coverage, not mapped candidate mass.
- `loss_ready_topk_mass`: deliberately equal to the strict retained mass.

The aggregate mass fields sum over teacher positions, so they can exceed one;
the corresponding ratio fields divide by total returned top-20 mass. Dataset
mass totals cover `comparable_records`; failed span diagnostics are counted
separately in `strict_fallback_records` and contribute zero span position
coverage.

## Why GOLD's ULD loss was not ported

GOLD assumes locally available teacher and student distributions over their
full vocabularies. After span grouping, its ULD path merges position
probabilities and compares sorted distributions with an L1 objective (or a
hybrid matched-vocabulary objective). Its observed merge also multiplies a
full distribution at one position by scalar probabilities of the actual later
tokens; GOLD documents that counterfactual branches are biased and the merged
distribution is unnormalized. Current TRL also exposes observed and Bayesian
merge strategies, but neither is the same objective as this project.

Our data contains an offline DeepSeek top-20 distribution at each teacher
prefix, not full teacher logits and not a locally queryable teacher. If a
teacher candidate maps to multiple student tokens, the exact student sequence
probability needs later logits conditioned on that candidate path. Those
counterfactual logits are absent from a single forward pass over the observed
response. For N:1 groups, intermediate teacher prefixes do not even coincide
with a student token boundary. Treating either case as a normal one-position
soft target would silently change the objective.

Therefore this change keeps the existing top-20 probabilities, folds all
unmapped and outside-top-20 mass into the same tail bucket, and returns the
strict alignment as `training_result`. `diagnose_with_strict_fallback`
records a span error without changing that training result.

## Test coverage

`tests/test_cross_tokenizer_aligner.py` covers:

- zero-width/different BOS behavior;
- leading spaces;
- newlines and Python indentation;
- Chinese and UTF-8 characters split across teacher tokens;
- one teacher token to multiple student tokens;
- multiple teacher tokens to one student token;
- a tokenizer for which decode/re-encode would not reproduce the original ID;
- prompt-boundary token fusion in a full chat-template encoding;
- strict fallback and preservation of the tail bucket;
- strict-versus-span coverage and retained-mass metrics.

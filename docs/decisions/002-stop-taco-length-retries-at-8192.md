# ADR-002: Stop blanket TACO length retries after the 8192-token experiment

## Status

Accepted.

## Date

2026-07-28

## Context

The TACO v1 pilot accepted 47 of 100 selected tasks. Forty-eight API attempts
across 21 tasks ended at `max_tokens=4096`; 18 of those tasks remained
unaccepted. It was unclear whether the truncations were otherwise-correct
programs that needed modestly more output space or long, unfocused generations.

Changing the original campaign in place would break provenance. A separate v2
campaign therefore retried the 42 eligible original attempts with
`max_tokens=8192`, the identical blind prompt, and no tests or failure feedback.

## Decision

- Preserve TACO v1 unchanged.
- Treat the 8192-token campaign as a separate, versioned, append-only dataset.
- Keep only the earliest passing retry per unique problem when building the
  combined accepted dataset.
- Do not run another blanket 16K/32K retry on the remaining truncated attempts.
- Prefer new task sampling and a stricter v2 eligibility policy over spending
  more tokens on the same generations.
- Do not start student training from this experiment automatically.

## Evidence

The authoritative v2 campaign made 42 calls:

- 42/42 API responses had exact, complete token-byte traces;
- only 14/42 stopped before 8192 tokens;
- 28/42 again ended with `finish_reason=length`;
- three attempts passed, representing two unique tasks;
- unique accepted tasks increased from 47 to 49;
- estimated cost was CNY 1.6123498, or CNY 0.8061749 per new unique task;
- ALM preprocessing succeeded for both new records and all 49 combined records.

The marginal gain of 2/18 eligible unaccepted tasks does not support another
uniform increase. A higher cap would also expand latency and raw trace size
without addressing pseudo-stdin rows, hard-problem selection, or the tendency
to generate analysis-like prose before code.

## Alternatives considered

### Increase all remaining attempts to 16K or 32K

Rejected. Two thirds of the 8192-token retries were still truncated, while the
unique-task yield was 11.11%. The evidence points to generation behavior and
task suitability, not merely a slightly undersized cap.

### Recover partial code from truncated output

Rejected. Conservative extraction deliberately avoids guessing around
incomplete Markdown or surrounding prose. Keeping false positives out of the
training candidates is more important than increasing nominal yield.

### Add verifier feedback to subsequent teacher calls

Rejected for this campaign. It would change blind rejection sampling into a
correction loop, expose test-derived information, and make the v1/v2 results
incomparable.

### Discard failed and superseded attempts

Rejected. Raw traces and verifier outcomes are required for audit,
resumability, and potential future correction or preference experiments.

## Consequences

- The authoritative TACO diagnostic candidate set is 49 unique accepted tasks.
- Remaining length-truncated attempts stay available as audit data but receive
  no automatic further API calls.
- The next data-volume experiment must use a versioned eligibility policy,
  stronger Linux isolation, and newly sampled tasks.
- TACO v2 evidence and exact artifacts are documented in
  `docs/taco_length_retry_v2.md`.

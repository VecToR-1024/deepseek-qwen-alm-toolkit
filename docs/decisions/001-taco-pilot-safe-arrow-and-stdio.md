# ADR-001: Pin TACO Arrow data and verify stdin/stdout programs locally

## Status

Accepted for the v1 pilot; the task-eligibility policy requires a v2 revision.

## Date

2026-07-28

## Context

The project needed a harder coding-data source after the 200-example MBPP
campaign. TACO provides official train data with hidden input/output pairs, but
its Hugging Face repository uses a custom Python loader and its rows mix
call-based functions, real stdin/stdout programs, pictures, and pseudo-input
examples copied from tutorial sites.

The local machine is Windows and has neither WSL nor Docker. Generated programs
therefore cannot be placed inside a real OS security sandbox.

## Decision

- Pin `BAAI/TACO` at commit
  `d593ed0a2becbbc952230bb89be09189bf1056dc`.
- Download `train/data-00000-of-00009.arrow` directly and never execute the
  remote dataset loader.
- Parse `input_output` and `solutions` with `json.loads`; never use `eval`.
- Use only `train` rows with no `fn_name`, no pictures, string-valued tests,
  and no HackerRank source.
- Keep tests and reference solutions outside the DeepSeek prompt.
- Execute each test in a fresh `python -I` child process with a temporary
  working directory, sanitized environment, wall-clock timeout, captured
  output, static forbidden-operation checks, and conservative exact output
  comparison after outer-whitespace normalization.
- Treat this Windows child-process boundary as risk reduction, not a security
  sandbox. Later Linux collection should use a container or isolated VM with
  networking disabled and resource limits.
- Preserve raw, normalized, and verifier JSONL append-only. Rejected summaries
  reference those files by attempt ID instead of embedding multi-gigabyte
  duplicate records.

## Alternatives considered

### Execute the official Hugging Face loader

Rejected because the loader is remote Python code and uses `eval` on dataset
fields. Direct Arrow ingestion is sufficient and has a smaller supply-chain
surface.

### Treat every row without `fn_name` as executable stdin/stdout

Used in v1, then shown to be incomplete. Thirteen selected GeeksForGeeks rows
contained pseudo-input such as `L = 1, R = 10`; 35 runtime-error attempts came
from that source. A v2 selection policy must explicitly exclude or separately
adapt these rows.

### Recover code aggressively from truncated Markdown responses

Rejected. Every v1 extraction failure had `finish_reason=length`. Guessing a
partial code block would create false-positive training records; the raw
attempt remains available for later correction data.

### Run generated code in the collection process

Rejected because it would let untrusted code corrupt collector state and
credentials.

## Consequences

- The v1 pilot is exactly reproducible, including its known single-shard and
  eligibility biases.
- Accepted records have complete DeepSeek traces and pass conservative tests.
- False negatives are possible because output comparison is stricter than the
  full historical TACO/APPS comparator.
- The v1 run selected 100 tasks, made 220 calls, and accepted 47 tasks.
- Before scaling, define a versioned v2 eligibility policy that excludes
  pseudo-input sources and decide whether `max_tokens` should exceed 4096.

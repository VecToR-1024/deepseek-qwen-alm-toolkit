# ADR-005: Prefer proven launchers and structured visual monitors

## Status

Accepted

## Date

2026-08-05

## Context

This project has accumulated several collection, training, and benchmark
pipelines that have already completed authoritative runs. Rebuilding similar
orchestration from scratch reintroduced avoidable incompatibilities during the
Qwen2.5-7B-Instruct phase:

- a three-task HumanEval smoke was sent to a scorer that requires all 164 tasks;
- a base-model codegen summary did not expose the adapter-shaped compatibility
  fields expected by the shared summarizer;
- two deployed Python files acquired CRLF line endings and were rejected by the
  frozen SHA-256 preflight;
- the new one-shot monitor defaulted to raw log tails and made stale shutdown
  markers look like current state.

Earlier monitors already provided a clearer operational view: named stages,
completed-task counts, GPU state, checkpoints, scores, and fatal-pattern counts.
The project should preserve that interface instead of treating every run as a
new orchestration design.

## Decision

For future data collection, training, and benchmark phases, use this order:

1. Reuse an asset that has completed the same class of run end to end.
2. Add a thin parameter or path adapter around that asset.
3. Generalize shared behavior only when at least two proven runs require it.
4. Create a new launcher, scorer, or monitor only when the existing contract is
   demonstrably incompatible; record the incompatibility in the run README.

Reuse includes the whole operational contract, not only the core Python call:

- input/output schemas and candidate ordering;
- append-only attempt records and completed markers;
- stage locks, resume behavior, and terminal exit codes;
- pinned revisions and SHA-256 checks;
- non-root/seccomp scoring boundaries;
- monitor field names and progress semantics.

Deployment must verify LF line endings for shell and Python orchestration files
before launch. A checksum mismatch is a stop-the-line event; preserve the
drifted bytes before restoring an attested copy.

The default monitor is a compact operational dashboard. It must show, where
applicable:

- overall run state and one row per candidate/stage;
- generated/scored task counts such as `83/164` or `210/339`;
- GPU utilization, memory, temperature, and active process;
- latest optimizer step, elapsed time, checkpoints, and selected loss fields;
- available benchmark scores and recovery mode;
- fatal-pattern count and a short error summary.

Raw log tails are opt-in through `--verbose` or a separate diagnostic command.
Stale markers must be labelled with their timestamps and must not override the
newest launcher attempt.

## Proven reference assets

- Training dashboard:
  `runs/merged_v3_lora_655_eosfix_v2_20260730/monitor_training.sh`.
- Checkpoint benchmark dashboard:
  `runs/multisource_hard_combined_alpha10_checkpoints_benchmarks_v1_20260804/monitor_benchmark.sh`.
- Per-candidate HumanEval/LiveCodeBench monitors:
  `runs/merged_v3_lora_655_eosfix_v2_humaneval_v1_20260730/monitor_benchmark.sh`
  and
  `runs/merged_v3_lora_655_eosfix_v2_livecodebench_v1_20260730/monitor_benchmark.sh`.
- Durable stage and resume patterns:
  the corresponding successful launchers in those versioned run directories.

These paths are reference implementations, not files to edit in place. New
runs copy or parameterize them into a new versioned directory and retain the
historical evidence unchanged.

## Alternatives considered

### Build each run from scratch

Rejected. It appears flexible but repeatedly loses small compatibility details
that are already encoded in successful assets.

### Use raw logs as the monitor

Rejected as the default. Logs remain the source of detailed evidence, but they
are poor at answering the common questions: what stage is active, how far it
has progressed, whether the GPU is healthy, and what has already passed.

### Remove integrity checks to avoid deployment failures

Rejected. The CRLF incident showed that the checksum gate worked correctly.
The remedy is deterministic deployment and clearer diagnostics, not weaker
attestation.

## Consequences

- New experiment setup begins with an inventory of proven assets.
- Diffs become smaller and are easier to test against earlier behavior.
- Monitors become useful status interfaces rather than log viewers.
- Some run-specific wrappers remain, but they are intentionally thin and
  versioned.
- A launcher may stop earlier on byte drift; this is accepted because it avoids
  running an unaudited benchmark implementation.

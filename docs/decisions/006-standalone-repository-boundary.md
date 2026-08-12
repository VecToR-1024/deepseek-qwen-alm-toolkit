# ADR-006: Extract a standalone toolkit without changing training semantics

## Status

Accepted

## Date

2026-08-06

## Context

The research workspace accumulated reusable source modules, generic CLIs,
run-specific launchers, raw traces, model artifacts, and experiment reports.
Copying the whole workspace would expose private runtime state and make it hard
to tell which scripts are reusable. Rewriting only the visible launchers would
lose the tested schema, recovery, verifier, and ALM dependencies behind them.

## Decision

Create a separate repository that preserves both Python package import
contracts as one coherent source closure, copies generic CLIs and applicable
offline tests, and provides new repository-relative operator examples.

Historical launchers that contain useful pinned hashes or proven orchestration
are retained under `proven_assets/` without modification. They are references,
not default entrypoints. Raw data, credentials, models, checkpoints, and logs
are excluded.

The extraction does not change ALM math, dataset schemas, or the trainer. The
authoritative training mode is documented accurately as BF16 LoRA; 4-bit QLoRA
is deferred instead of being inferred or reimplemented during packaging.

## Alternatives Considered

### Copy the complete research workspace

Rejected because run outputs dwarf the source, include private state, and make
the repository non-reproducible.

### Copy only the 48-worker supervisor and training shell script

Rejected because both depend on tested modules for durable queues, task
schemas, trace validation, isolated verification, ALM preprocessing, and loss.

### Rewrite a smaller toolkit from memory

Rejected because it would discard known-good behavior and risk changing the
mathematical objective or recovery semantics.

## Consequences

- The standalone repo is usable without the original workspace.
- Proven historical launchers remain auditable but require explicit path
  migration before reuse.
- The copied test suite is smaller than the research workspace suite because
  tests whose only purpose was auditing old run directories are not included.
- Future QLoRA support must be a separate tested feature with an explicit model
  loading and optimizer contract.

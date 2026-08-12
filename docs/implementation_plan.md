# Implementation Plan: Standalone Toolkit Extraction

## Architecture Decisions

- Preserve existing Python import contracts and copy source modules as a
  coherent dependency closure instead of forking trainer logic.
- Ship generic runnable examples separately from immutable provenance notes.
- Treat historical results as evidence documented by commit/hash, not as source
  assets to copy into the new repository.

## Task List

### Phase 1: Foundation

- [ ] Copy package source, generic CLIs, training entrypoints, and applicable
  tests from the pinned source commit.
  - Acceptance: imports resolve without the original workspace.
  - Verify: pytest collection succeeds in the staging repository.
- [ ] Replace distribution metadata and add safe ignore rules.
  - Acceptance: editable install metadata exposes all optional dependency groups.
  - Verify: `python -m pip install --dry-run -e .` resolves metadata.

### Checkpoint: Foundation

- [ ] Collection and ALM unit tests pass.

### Phase 2: Operator Assets

- [ ] Add a 32 API + 16 verifier actual-only campaign example with start,
  monitor, and graceful-stop scripts.
  - Acceptance: campaign dry-run contains 32 API and 16 verifier workers and no
    top-20 request.
  - Verify: static/example tests plus supervisor dry-run.
- [ ] Add BF16 LoRA SFT-only and SFT+ALM training templates and manifest notes.
  - Acceptance: both arms use one entrypoint and differ only by `ALPHA_ALM`.
  - Verify: training-entrypoint unit tests and shell syntax checks when available.
- [ ] Add benchmark adapter inventory and hardened execution runbook.
  - Acceptance: provenance and required external harness pins are explicit.
  - Verify: adapter unit tests run without executing generated code.

### Checkpoint: Operator Assets

- [ ] Example configs are path-relative and contain no credentials.

### Phase 3: Documentation and Release Gate

- [ ] Write README, architecture, provenance manifest, and ADR.
  - Acceptance: a new operator can install, dry-run, collect, audit, and train
    without consulting the original workspace.
  - Verify: documented commands match `--help` output and file layout.
- [ ] Run full offline tests and repository hygiene scans.
  - Acceptance: tests pass; no secret, private data, absolute user path, model,
    checkpoint, or large run artifact is tracked.
  - Verify: pytest, `rg` scans, and size inventory.
- [ ] Move staging tree to its sibling directory and initialize Git.
  - Acceptance: the destination is a standalone clean repository with one
    descriptive initial commit.
  - Verify: `git status --short`, `git log -1`, and source commit in provenance.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Run-specific scripts hide absolute paths | High | Parameterize runnable templates; retain only provenance descriptions |
| Copying a partial module closure | High | Copy both complete source packages and run import/test collection |
| Accidental secret/data inclusion | High | Allowlist assets, ignore data/runs, scan staged content before commit |
| Mislabeling BF16 LoRA as QLoRA | Medium | State the actual mode in README/spec/manifest and defer 4-bit support |
| Benchmark code executes untrusted output | High | Keep evaluation opt-in and document non-root/seccomp isolation requirements |

## Open Questions

None blocking. The user requested a local standalone repository and approved
starting the extraction; later publication, remote push, and QLoRA additions are
outside this task.

# Spec: DeepSeek-to-Qwen Offline ALM Toolkit

## Objective

Extract the reusable engineering assets from the research workspace into a
standalone, installable, testable repository. The repository must cover the
complete path from benchmark task import through offline DeepSeek trace
collection, isolated verification, dataset audit, ALM preprocessing, LoRA
training, and benchmark handoff.

Success means a new checkout can inspect and dry-run the 48-worker collection
campaign, install the Python packages, run the copied unit tests without live
API or GPU access, and use documented templates to launch collection or
training after supplying its own data, model cache, and credentials.

## Assumptions

1. The standalone repository is named `deepseek-qwen-alm-toolkit` and will live
   beside, not inside, the research workspace.
2. Source packages keep their existing import names (`deepseek_distill` and
   `topk_distill`) to avoid changing the trainer or dataset contract.
3. The authoritative 48-worker topology is 32 API workers plus 16 isolated
   verifier workers; writer/normalizer coordination threads are not counted as
   API or verifier workers.
4. The current authoritative training entrypoint is BF16 LoRA, not 4-bit
   QLoRA. The repository must not relabel it as QLoRA.
5. Historical data, API responses, model weights, checkpoints, logs, passwords,
   API keys, and machine-specific active-run configuration are out of scope.
6. Proven historical launchers may be retained only as clearly marked
   references; runnable examples must use parameters or repository-relative
   paths.

## Tech Stack

- Python 3.11 or 3.12
- PyTorch, Transformers, PEFT, TRL, Datasets
- OpenAI-compatible client for the DeepSeek API
- zstandard for sealed raw trace shards
- pytest for offline tests
- PowerShell for local Windows supervision and Bash for Linux GPU launchers

## Commands

```text
Install:  python -m pip install -e ".[collect,archive,data,train,test]"
Test:     python -m pytest -q
Dry run:  python scripts/run_hard_collection_campaign.py --config configs/collection.actual-only.48workers.example.json --repo-root . --python python --dry-run
Train:    python examples/train_offline_alm.py
```

## Project Structure

```text
src/deepseek_distill/   collection, normalization, verification, audit, ALM preprocessing
src/topk_distill/       ALM math/trainer and optional strict top-20 baseline
scripts/                import, collection, aggregation, freeze, audit, deployment helpers
examples/               training entrypoints and 48-worker operator scripts
configs/                non-secret example campaign/training configuration
benchmarks/             reusable benchmark adapters and provenance notes
tests/                  offline unit/integration tests with fake API responses
docs/                   architecture, runbooks, decisions, and provenance
```

## Code Style

Use typed Python, `pathlib.Path`, append-only JSONL artifacts, explicit schema
versions, and fail-closed validation. A representative public boundary is:

```python
def load_campaign_config(path: Path, *, repo_root: Path) -> CampaignConfig:
    """Load, validate, and resolve a non-secret campaign description."""
```

## Testing Strategy

- Unit tests cover alignment, ALM math, schemas, importers, prompt construction,
  byte reconstruction, source extraction, and audit calculations.
- Integration tests use fake API responses and isolated child processes; pytest
  never requires a live API key, network connection, model download, or GPU.
- Static asset tests validate the shipped 48-worker example and ensure training
  examples are path-relative and secret-free.
- A final repository scan rejects credentials, private keys, large data/model
  artifacts, and hard-coded workspace/cloud passwords.

## Boundaries

- Always: preserve append-only/resumable semantics, raw teacher traces, byte
  reconstruction checks, official tests outside teacher prompts, and the
  `hard_sft_loss + alpha_alm * alm_loss` objective.
- Ask first: change dataset schemas, ALM math, trainer behavior, or add a new
  runtime dependency.
- Never: copy secrets, raw research data, checkpoints, model weights, active
  run logs, or start a live API/training job as part of repository assembly.

## Success Criteria

- The repository contains the 32+16 durable streaming collector and an example
  configuration that dry-runs to commands with exactly those worker budgets.
- The OfflineTeacherTraceProvider, byte-level ALM alignment/loss, BF16 LoRA
  training entrypoint, strict top-20 baseline, dataset freeze/audit tools, and
  benchmark adapters are present.
- Installation metadata and documentation are standalone and do not refer to
  the original workspace as a runtime dependency.
- Focused tests and the applicable copied test suite pass offline.
- A secret/path/large-file audit passes before the first Git commit.
- The final repository records the source branch and commit used for extraction.

## Open Questions

- 4-bit QLoRA support is intentionally not invented during extraction. It can
  be added later as a separately tested training mode.
- Live benchmark harness installation remains an operator step because EvalPlus
  and LiveCodeBench execute untrusted generated code and need a hardened Linux
  environment.

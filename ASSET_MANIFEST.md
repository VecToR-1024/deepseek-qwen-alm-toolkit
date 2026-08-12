# Asset manifest

## Extraction provenance

- Source workspace: `蒸馏`
- Source branch: `codex/alm-offline-kd`
- Source commit: `ec1323bf27d1d78736be0705a04909578c158a6f`
- Extraction date: `2026-08-06`
- Standalone schema: `deepseek_qwen_alm.asset_manifest.v1`

## Included

- Complete `src/deepseek_distill` package.
- Complete `src/topk_distill` package.
- Generic import, collection, normalization, verification, aggregation, freeze,
  audit, deployment, archive, and recovery CLIs from `scripts/`.
- Offline ALM and pluggable training entrypoints from `examples/`.
- 355 applicable offline tests, including four standalone-asset contract tests.
- Curated design/runbook documents and ADRs 001–006.
- Repository-relative 32+16 actual-only collection example.
- Repository-relative SFT-only / SFT+ALM BF16 LoRA comparison example.
- Two proven launcher templates with machine roots and SSH endpoints removed.

## Intentionally excluded

- `data/` and every raw/normalized/verifier/accepted/rejected JSONL artifact.
- Model caches, weights, LoRA adapters, optimizers, and checkpoints.
- Live API keys, SSH passwords, `.env`, host keys, and personal credentials.
- `ms-swift/` and other third-party source checkouts.
- Historical launcher logs, benchmark generations/results, GPU samples, PIDs,
  completion markers, and machine-specific mutable run directories.
- Tests that only asserted the content of excluded historical run directories.

## Integrity model

The source commit above is the authoritative hash for copied implementation
files. The standalone Git commit records packaging-only additions and the
allowlisted extraction. Runtime datasets should always carry their own record
count, byte count, SHA-256, source revisions, and `training_started=false`
manifest before training.

## 2026-08-12 refresh

- Added an opt-in `chat_template_kwargs` contract to ALM preprocessing and the
  frozen-dataset audit. The default remains empty, preserving Qwen2.5 behavior.
- Added a pinned Qwen3-0.6B non-thinking, BF16 full-finetune candidate config
  and sequential SFT/SFT+ALM launcher.
- Added a data-only acceptance summary for the 2026-08-11 actual-only wave.
- The Qwen3 data contract is locally validated; the new optimizer/launcher
  settings are deliberately marked `data_contract_only` until a GPU smoke run
  succeeds.
- No dataset, model weight, checkpoint, trace, API credential, or SSH secret was
  added by this refresh.

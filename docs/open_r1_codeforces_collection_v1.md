# Open-R1 Codeforces collection v1

> Started: 2026-08-03
> Branch: `codex/alm-offline-kd`
> Adapter commit: `c2374f8`
> Training started: **false**

## Purpose

Extend the verified multi-source DeepSeek-to-Qwen dataset with Codeforces problems while
reusing the existing append-only collection and ALM preprocessing stack. This version is a
3-task authoritative smoke run, not a bulk collection and not a training run.

## Pinned source

- Dataset: `open-r1/codeforces`
- Config: `verifiable`
- Split: `train`
- Revision: `fbe3f6e903ee854eec2e69e9d96d0306cde59baf`
- Machine-readable license tag: `cc-by-4.0`
- Source: <https://huggingface.co/datasets/open-r1/codeforces>
- License caveat: the dataset tag says CC-BY-4.0 while the README body says ODC-By 4.0;
  both facts are retained in source notes.

The dataset card says the test split contains late-2024/early-2025 problems and should not be
used for training. This pipeline therefore imports train only.

## Eligibility contract

The current exact-output verifier is authoritative only for rows with all of the following:

- `executable=true`;
- `official_tests_complete=true`;
- `input_mode=stdio`;
- no `generated_checker`;
- no `interaction_format`;
- at least one complete official input/output pair.

Custom-checker, file-I/O and interactive tasks are excluded rather than weakly or incorrectly
verified. External generated tests are not downloaded in v1. See
[`ADR-004`](decisions/004-reuse-exact-output-pipeline-for-open-r1-codeforces.md).

## Reused architecture

```text
open-r1/codeforces train row
  -> coding.task.multisource.v1 adapter          [new]
  -> existing clean-v2 teacher prompt            [reused]
  -> existing DeepSeek collector                 [reused]
  -> append-only raw_attempts.jsonl single writer [reused]
  -> streaming normalized_attempts.jsonl          [reused]
  -> isolated stdio verifier workers              [reused]
  -> accepted_unique / rejected audit records     [reused]
  -> existing clean + EOS + ALM diagnostics       [reused]
```

No ALM trainer, loss, tokenizer alignment, generation contract or student-training code was
changed.

## Cloud paths

```text
Code snapshot:
${AUTODL_ROOT}/repo_openr1_codeforces_20260803_c2374f8

Run root:
${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v1

Import artifacts:
${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v1/import/tasks_3.jsonl
${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v1/import/import_3.json

Smoke collection:
${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v1/runs/smoke3
```

The importer runs in the background. A supervisor waits for both immutable import artifacts,
then automatically runs one blind attempt per task through API collection, byte reconstruction,
normalization, isolated official-test verification, aggregation and ALM/EOS/clean auditing. It
does not start training.

## Monitoring

One snapshot:

```bash
bash ${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v1/monitor_collection.sh --once
```

Refresh every 15 seconds:

```bash
bash ${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v1/monitor_collection.sh
```

Custom refresh interval:

```bash
bash ${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v1/monitor_collection.sh --interval 30
```

The monitor is read-only. It shows active import/API/verifier processes, disk usage, durable
raw/normalized/verifier queue lengths, official passes, accepted records and recent logs.

## Verification completed before launch

- New adapter and generic CLI/verifier tests: `24 passed`.
- Full local repository regression: `292 passed`, with one pre-existing TRL experimental warning.
- Same adapter tests in the cloud snapshot: `12 passed`.
- A fake normalized actual-token trace passed the unchanged isolated stdin/stdout verifier.

## Current milestone gate

Do not start a 200-task or larger run until the smoke artifacts prove at least one real response
passes all of the following:

1. raw actual-token byte reconstruction;
2. conservative Python extraction;
3. isolated execution against every retained official test;
4. EOS supervision audit;
5. ALM preprocessing with at least one valid chunk.

If the smoke passes, the next decision is whether to pay the full streaming scan cost for seeded
random selection or use a separately documented deterministic stratified selection. Do not label
the 3-task `selection=first` smoke as a representative Codeforces sample.

## Bulk v2 launch

The smoke completed with 3/3 API, trace, extraction and official-test passes. All three records
passed ALM preprocessing and EOS supervision; two passed the final clean gate. One was excluded
for both sequence length over 4096 and comment ratio over 20%.

An independent bulk run was therefore launched at:

```text
${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v2_bulk1000
```

The first full-split scan encountered an otherwise executable row with an empty `description`.
The importer now deterministically treats empty `title` or `description` rows as ineligible
instead of aborting the whole scan. The guarded bulk snapshot is commit `803cecf` at:

```text
${AUTODL_ROOT}/repo_openr1_codeforces_20260803_803cecf
```

Frozen bulk configuration:

- 1,000 unique `verifiable/train` tasks;
- `selection=random`, `seed=20260803`;
- the three smoke task IDs are excluded before publication;
- one blind DeepSeek attempt per task;
- 32 API workers, 120 requests/minute cap;
- streaming normalization and 16 verifier workers;
- `temperature=0.2`, `top_p=1.0`, top-20 logprobs and 4096 completion-token cap;
- automatic ALM/EOS/clean audit after collection;
- no student training.

Bulk monitor:

```bash
bash ${AUTODL_ROOT}/data/open_r1_codeforces_20260803/v2_bulk1000/monitor_bulk1000.sh
```

Use `--once` for a single snapshot. The optimized monitor reads the small durable state and
summary JSON files instead of repeatedly scanning multi-gigabyte JSONL queues.

### Bulk v2 result

The run completed without queue lag or pipeline errors:

- raw / normalized / verified: `1000 / 1000 / 1000`;
- official-test accepted: `652`;
- final clean eligible: `459`;
- ALM preprocessing errors: `0`;
- zero-chunk records: `0`;
- EOS supervised: `652 / 652`;
- clean exclusions: 193, dominated by 192 responses above the 20% comment-line limit;
- disk footprint: approximately 9.8GB.

Training remained disabled. Cross-source deduplication is still required before merging these 459
records into the next immutable training manifest.

# Spec: 20k Multi-source ALM Collection v2

## Objective

Build a deterministic pool of up to 20,000 previously unused Python coding
tasks, collect one blind DeepSeek generation per task, and produce roughly
3,000 clean ALM records without changing the trainer or starting training.

The immediate release gate is smaller: the repository must be able to run and
audit a 500-task `actual_only` pilot before a 20k campaign is permitted.

## Assumptions

1. `codex/alm-offline-kd` remains the authoritative implementation branch.
2. Existing raw and normalized schema versions remain readable.
3. ALM needs generated-token bytes and generated-token logprobs, not top-20
   alternatives.
4. Existing top-20 collection remains available as an experimental baseline.
5. A failed official test is terminal for the task; only transport failures
   and one malformed-trace recovery may make another provider request.
6. Training, trainer changes, and benchmark launches are outside this change.

## Tech Stack

- Python 3.12 (project supports Python `>=3.11,<3.13`)
- `openai>=2.45,<3` for the DeepSeek-compatible client
- pytest 9 for unit and integration tests
- Existing append-only JSONL durable pipeline and isolated verifier
- Zstandard for sealed raw shard compression only after the pilot proves the
  dependency and recovery behavior; it is not required by the trace-profile
  slice.

## Commands

Install/test:

```powershell
python -m pip install -e ".[collect,test,archive]"
python -m pytest tests/test_deepseek_api.py tests/test_deepseek_records.py tests/test_collect_multisource_breadth_cli.py
```

Prepare a campaign without API calls:

```powershell
python scripts/collect_multisource_breadth.py --source taco --tasks TASKS.jsonl --import-summary IMPORT.json --run-dir RUN --prepare-only --trace-profile actual_only --max-attempts-per-task 1
```

Run the 500-task pilot after its immutable task manifest is frozen:

```powershell
python scripts/collect_multisource_breadth.py --source taco --tasks TASKS_500.jsonl --import-summary IMPORT_500.json --run-dir RUN_500 --trace-profile actual_only --workers 32 --verifier-workers 20 --requests-per-minute 0 --max-attempts-per-task 1 --streaming-pipeline
```

Seal one fully drained batch without deleting its raw JSONL:

```powershell
python scripts/seal_raw_shard.py --run-dir RUN_500 --expected-records 500 --compression-level 6
```

The 20k command must not be run until the pilot gate in Success Criteria is
satisfied.

## Project Structure

- `src/deepseek_distill/api.py`: provider request and persisted generation
  contract.
- `src/deepseek_distill/records.py`: untrusted response normalization and trace
  validation.
- `src/deepseek_distill/durable_pipeline.py`: existing raw -> normalized ->
  verifier streaming pipeline; keep its queue semantics.
- `scripts/collect_multisource_breadth.py`: backward-compatible CLI surface.
- `tests/`: fake-provider contract and resumability tests; pytest must never
  require a live API call.
- `docs/`: policy, provenance, run commands, and audit decisions.

## Interface Contract

Two trace profiles are supported:

| Profile | Request | Required normalized data | Purpose |
| --- | --- | --- | --- |
| `top20` | `logprobs=true`, `top_logprobs=20` | actual trace plus candidates | legacy strict/top-20 baseline |
| `actual_only` | `logprobs=true`, omit `top_logprobs` | actual token, bytes, logprob | primary ALM collection |

`top_logprobs` remains accepted as a legacy CLI override. It must not be used
together with an incompatible explicit trace profile. Old manifests and JSONL
records without `trace_profile` continue to mean the existing top-k contract.

For `actual_only`, providers may omit a token row's `top_logprobs` field or
return an empty list. For `top20`, the field remains required and may contain
fewer candidates than requested, matching the provider contract.

## Collection Policy

- Deterministically freeze ordered task IDs and SHA256 before collection.
- Exclude prior training task IDs and all benchmark exclusion IDs before
  selection.
- Prefer approximately 8k-9k TACO, 8k-9k CodeContests, all eligible remaining
  APPS, and 2k-3k from another verified source. Do not invent tasks to fill a
  source shortfall.
- One semantic generation per task.
- SDK transport retries are allowed and remain the same semantic attempt.
- A malformed actual trace may receive one blind replacement request with no
  test, stderr, traceback, or verifier feedback.
- Failed official tests are retained and not retried.
- Preserve every raw attempt and verifier outcome.

## Shard Contract

Raw data is eventually written in 500-attempt shards:

1. append and fsync the open shard;
2. seal it with record count and SHA256;
3. let normalizer/verifier acknowledge every successful record or durable
   failure;
4. compress the sealed shard to a temporary zstd file;
5. verify decompressed byte count, record count, and SHA256;
6. atomically rename the compressed file;
7. only then may the sealed uncompressed shard be removed.

Crash recovery must never duplicate an attempt ID and must accept either the
last valid uncompressed sealed shard or its verified compressed replacement.

## Data Outputs

- `accepted_alm.jsonl`: raw teacher text is already code-only, trace-valid,
  test-passing, and ALM-preprocessable.
- `accepted_sft_only.jsonl`: a separate cleaned text record that is parsed and
  reverified after cleaning; it never claims the original ALM trace still
  matches.
- `rejected.jsonl`: all structural, format, extraction, or test failures.
- append-only raw, normalization-error, verifier, shard manifest, and campaign
  audit artifacts.

## Code Style

Use typed, additive Python contracts with explicit boundary validation:

```python
config = GenerationConfig(trace_profile="actual_only")
kwargs = config.as_api_kwargs(messages)
assert kwargs["logprobs"] is True
assert "top_logprobs" not in kwargs
```

Do not infer a trace profile from response size or candidate contents. Persist
the requested profile in request metadata.

## Testing Strategy

- Unit: profile validation, exact request kwargs, response normalization with
  missing/empty top candidates, top-20 backward compatibility.
- Integration: fake actual-only raw record through normalize, validate,
  OfflineTeacherTraceProvider, and durable resume.
- Live gate: 3-10 requests first, then 500 tasks. Live calls are manual and
  credentials come only from `DEEPSEEK_API_KEY`.
- Sharding: crash at every seal/compress/rename boundary, corrupt archive
  detection, duplicate prevention, and resume.

## Boundaries

- Always: preserve raw response JSON, validate byte reconstruction, fsync
  append-only records, write immutable manifests, and run tests before commits.
- Ask first: add a mandatory dependency, change an existing schema version,
  delete an uncompressed raw shard, or launch more than the approved pilot.
- Never: hard-code credentials, rewrite teacher text while retaining its ALM
  trace, expose tests to the teacher, modify the trainer, start training, or
  silently fill a source-capacity shortfall with benchmark data.

## Success Criteria

Trace-profile slice:

- Existing top-20 tests remain green.
- `actual_only` omits `top_logprobs` from the API request and persists the
  profile.
- Missing/empty candidate arrays normalize without weakening actual token byte
  and logprob checks.
- An actual-only fake record passes raw -> normalize -> validate -> offline ALM
  trace extraction.

500-task pilot gate:

- At least one real response passes exact byte reconstruction, code extraction,
  isolated official tests, clean-format audit, and ALM preprocessing.
- Actual-token logprob availability and exact byte reconstruction are both
  reported; any rate below 99.9% blocks 20k.
- Compressed and uncompressed size per attempt are measured.
- Resume makes no duplicate provider request.

20k completion:

- Immutable task order, provenance, exclusion manifest, and hashes exist.
- All queues are terminal and resumable, with no duplicate task or attempt IDs.
- Clean ALM, SFT-only, and rejected outputs are separated without trace/text
  mismatch.
- No student training starts automatically.

## Implementation Tasks

- [x] T1: Add backward-compatible trace profiles and fake-response tests.
  - Verify: focused API/record/offline-provider tests pass.
- [x] T2: Thread the profile through the existing breadth CLI and campaign
  manifest.
  - Verify: parser/prepare-only tests preserve old defaults and record the new
    profile.
- [x] T3: Run a 3-10 request live smoke and publish byte/logprob/size evidence.
  - Verify: at least one end-to-end accepted ALM record.
- [x] T4: Add sealed 500-record raw shards and verified zstd compaction.
  - Verify: crash/recovery and corruption tests pass.
- [ ] T5: Inventory and freeze a deduplicated 20k candidate manifest.
  - Verify: deterministic rerun, source counts, exclusions, and SHA256 match.
- [x] T6: Run the 500-task pilot, audit it, and make a go/no-go decision.
  - Verify: pilot gate above is satisfied before any 20k launch.
- [ ] T7: Run a bounded 2,000-task post-gate expansion without training.
  - Verify: 1,000 new TACO plus 1,000 new CodeContests tasks, source-level
    clean/ALM audits, verified raw archives, and no duplicate local task IDs.

T6 was prepared locally on 2026-08-06 with a deterministic 500-task TACO
mixed-difficulty manifest (seed `2026080601`). It excludes the 4,250 historical
non-zero-shard TACO task IDs currently available locally. The campaign assets
are in `runs/multisource_actual_only_pilot500_v1r1_20260806/`; final 20k
freezing remains blocked on recovering the missing cloud-only history
manifests.

The pilot entered the live `collecting` phase at 2026-08-06 10:53
(Asia/Shanghai). Startup revalidated the 500-record task hash and persisted an
`actual_only` campaign manifest with `top_logprobs=null` and one blind attempt
per task.

The pilot completed with 500/500 API, normalized, and verifier records; zero
missing token bytes/logprobs; zero byte-reconstruction failures; and zero
top-k candidate rows. Official tests passed for 217/500, and 151 records were
clean eligible. All 217 official passes had supervised EOS and valid non-zero
ALM chunks. A no-op resume skipped all 500 durable IDs. The verified zstd raw
archive is 35,841,897 bytes versus 144,066,984 bytes uncompressed.

T7 uses two mixed-difficulty lanes: 1,000 additional TACO tasks with seed
`2026080602` and 1,000 additional CodeContests tasks with seed `2026080603`.
It remains a bounded local-history expansion rather than the authoritative
20k freeze because some early cloud-only task manifests are still missing.

T7 entered live execution at 2026-08-06 11:48 (Asia/Shanghai). The TACO lane
entered `collecting` while the CodeContests lane continued its bounded pinned
train import. The shared supervisor was alive with no stderr and will keep the
workstation awake; no training stage exists in this campaign.

## Open Questions

- Exact remaining CodeContests capacity must be measured before fixing source
  quotas.
- The optional fourth source must have reliable runnable Python tests before it
  can enter the immutable pool.
- Zstandard will be selected only after confirming the available runtime and
  pilot compression ratio.

## Implementation Evidence (2026-08-06)

Actual-only provider probe:

- `logprobs=true` with `top_logprobs` omitted returned every actual token's
  token text, byte array, and logprob.
- The provider returned an empty `top_logprobs` array at every position.
- Probe 001: 16/16 actual logprobs, zero null byte arrays, exact response-byte
  reconstruction, `finish_reason=stop`, 2,659 raw bytes.

Ten-task MBPP train smoke (not part of the future 20k training pool):

- 10/10 API successes, 10/10 normalized traces, zero byte warnings.
- 5/10 official-test passes.
- 5/5 passing records were clean eligible and EOS-supervised.
- ALM: zero preprocessing errors, zero zero-chunk examples, zero boundary
  drops, and no sequence over 4096.
- Resume used a provider client that raises on any call; all 10 requests were
  skipped and all durable artifacts remained terminal.
- Artifacts: `data/actual_only_smoke_mbpp10_v1/run/`.

Zstd shard smoke:

- Raw 10-attempt shard: 86,640 bytes.
- Verified level-6 zstd archive: 8,373 bytes (10.35x smaller).
- A second seal returned `unchanged`; source deletion was not requested.
- Automated tests cover missing verifier acknowledgement, corrupted archive,
  post-deletion resume, inconsistent manifests, and changed compression policy
  after a crash boundary.

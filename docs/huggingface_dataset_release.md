# Hugging Face dataset release

## Release-record hygiene

Keep deployment-specific account names, repository IDs, Hub commit hashes,
manifest hashes, SSH endpoints, and key-management records in a private operator
manifest outside this toolkit. Public documentation should contain only the
reproducible packaging and verification procedure below.

After each release, verify privately that the remote manifest hash matches the
reviewed local manifest and that `load_dataset(..., split="train")` loads the
expected row count. Never copy the private release record back into this
repository.

## Decision

Publish the authoritative offline-ALM training data as a private Hugging Face
dataset first. The Hub package is a semantic projection of the frozen JSONL,
not a byte-for-byte copy: it preserves every actual generated-token byte array
and log probability consumed by ALM while removing fields that are unnecessary
or unsafe to redistribute.

The release tool supports two non-interchangeable profiles:

- `actual_only` (default): the compact primary ALM dataset, without alternative
  token distributions;
- `strict_top20`: a separate experimental baseline dataset with exactly 20
  alternative candidate byte arrays and log probabilities at every generated
  position, plus probability mass recomputed from those candidates.

Use a separate Hub repository for `strict_top20` so publishing its much larger
artifacts cannot replace the compact ALM dataset card or shards.

The first configuration should be named `trained_2041` because it is the exact
dataset used by the completed Qwen training runs. Do not label the unfinished
2,751-record local candidate pool as a trained dataset.

Authoritative `trained_2041` identity:

- records: `2041`
- source bytes: `900901387`
- source SHA256:
  `64b28fd6d6b090055684c7470da0bd6e6591d52dd72b8651e54c280b6b3b830f`
- source distribution: TACO 554, xCodeEval 44, APPS 157, CodeContests 666,
  ODEX 117, Open-R1 Codeforces 503

The exact JSONL currently exists only on the cloud host and must be recovered
before packaging. Two local frozen datasets (723 and 512 records) are inputs or
later candidates, not substitutes for the authoritative 2,041-record file.

## Published contract

Each published row keeps:

- normalized schema version and record ID;
- teacher request messages with local absolute paths replaced by
  `<LOCAL_PATH>`;
- unmodified accepted response text;
- actual token `bytes` and `logprob` values;
- task source provenance, revision, split, and recorded license label;
- compact generation, usage, and sampling metadata.

The release builder validates that concatenated token bytes exactly reconstruct
`response_text.encode("utf-8")`. The resulting rows remain directly compatible
with `OfflineTeacherTraceProvider` and the current ALM training loader.

This prompt-only path replacement changes the published context but never the
teacher completion or actual-token trace. A local path found in a completion is
a hard rejection because rewriting it would invalidate the stored logprobs.

Both profiles omit:

- official, private, or generated tests;
- reference solutions and task metadata blobs;
- verifier stdout, stderr, tracebacks, and local artifact paths;
- provider response IDs, base URLs, and system fingerprints;
- any detected credential-like value or local user path.

The `actual_only` profile additionally omits top-20 alternatives. The
`strict_top20` profile keeps each candidate's token string, bytes, and logprob
because the existing strict aligner consumes all three fields. It requires
`generation_config.top_logprobs == 20` and exactly 20 valid candidates at every
actual-token position; incomplete records fail packaging.
The stored `top_probability_mass` is recomputed from candidate logprobs, so its
complement can be used as the tail bucket by the existing strict baseline.

Never edit response text and reuse the old trace: byte and log-probability
alignment would no longer be authoritative.

## Package

Install the optional publishing dependency:

```powershell
python -m pip install -e ".[release]"
```

Build deterministic gzip-compressed JSONL shards. Packaging refuses an existing
output path, a wrong record count, or a wrong source hash.

```powershell
python scripts/release_hf_dataset.py package `
  --input C:\path\to\training_records.jsonl `
  --output-dir C:\path\to\hf_release_trained_2041 `
  --config-name trained_2041 `
  --repo-id NAMESPACE/DATASET_NAME `
  --expected-records 2041 `
  --expected-sha256 64b28fd6d6b090055684c7470da0bd6e6591d52dd72b8651e54c280b6b3b830f
```

For a separate strict top-20 release, add the explicit profile and use a
different destination directory and Hub repository:

```powershell
python scripts/release_hf_dataset.py package `
  --input C:\path\to\training_records.jsonl `
  --output-dir C:\path\to\hf_release_strict_top20_2041 `
  --config-name strict_top20_2041 `
  --repo-id NAMESPACE/STRICT_TOP20_DATASET_NAME `
  --trace-profile strict_top20 `
  --expected-records 2041 `
  --expected-sha256 64b28fd6d6b090055684c7470da0bd6e6591d52dd72b8651e54c280b6b3b830f
```

The package contains `README.md`, `release_manifest.json`, and deterministic
`data/trained_2041/*.jsonl.gz` shards. The manifest records input identity,
source/license counts, redactions, and every output shard hash.

Re-run the independent release audit before review or upload:

```powershell
python scripts/release_hf_dataset.py audit `
  --package-dir C:\path\to\hf_release_trained_2041
```

It rechecks the package file set, shard sizes and hashes, duplicate IDs,
forbidden fields, credential/path patterns, and every actual-token byte
reconstruction. For `strict_top20`, it also recounts all candidate positions,
requires exactly 20 candidates per position, and verifies each recomputed mass.
The upload path runs this audit again automatically.

## Upload

Authenticate outside the repository. Do not write a token into a script,
manifest, shell history, or dataset row. Use the Hub CLI's interactive prompt:

```powershell
hf auth login
hf auth whoami
```

Preview the operation (no network mutation):

```powershell
python scripts/release_hf_dataset.py upload `
  --package-dir C:\path\to\hf_release_trained_2041 `
  --repo-id NAMESPACE/DATASET_NAME
```

The preview prints the exact `release_manifest.json` SHA256. Review the dataset
card, manifest, destination repository, and visibility before continuing.

Create/update the private dataset repository, upload the package, and verify
that every manifest-listed file exists remotely:

```powershell
python scripts/release_hf_dataset.py upload `
  --package-dir C:\path\to\hf_release_trained_2041 `
  --repo-id NAMESPACE/DATASET_NAME `
  --confirm-manifest-sha256 <SHA256_FROM_REVIEWED_DRY_RUN> `
  --execute
```

Both `--execute` and a matching human-reviewed manifest hash are mandatory.

Public visibility requires the additional `--public` flag. Keep it private
until the mixed upstream licensing and TACO/Open-R1/xCodeEval license-label
caveats have been reviewed.

## Source terms

The dataset card deliberately uses `license: other` and lists license labels by
source. APPS, CodeContests, TACO, ODEX, xCodeEval, and Open-R1 Codeforces do not
form one uniform licensing regime. In particular, do not publish CodeContests
private tests, and do not flatten TACO's upstream caveats or the documented
Open-R1/xCodeEval metadata discrepancies into a single permissive claim.

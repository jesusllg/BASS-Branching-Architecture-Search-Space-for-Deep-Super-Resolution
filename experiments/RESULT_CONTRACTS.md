# Result contracts for hardware and training adapters

The repository does not prescribe a laboratory scheduler, dataset mount, or
GPU telemetry stack. It does prescribe what those adapters must return so a
gate cannot pass by silently losing inconvenient cases.

Every result uses the envelope defined in
[`result-contracts.json`](../src/bass/experiments/protocols/result-contracts.json)
and contains:

- the exact protocol ID/digest, work-order digest, gate ID, clean source
  revision, version-specific runtime-tree digest, timezone-aware start/finish
  timestamps, expanded command, and observed environment;
- `smoke`, `qualifying`, and one of `NOT_RUN`, `PASS`, `FAIL`, or `ERROR`;
- one disposition for every frozen criterion, in the original order;
- immutable artifact paths plus SHA-256 digests; and
- notes, deviations, exclusions, OOMs, and failed seeds.

`PASS` is rejected when the run is smoke/nonqualifying, its source, runtime
tree, protocol, gate, or hardware does not match the supplied work order, a
criterion is false, or a review gate lacks a reviewer identity.

The three frozen identities answer different questions:

| Identity | What it binds |
|---|---|
| `source_revision` | The complete Git checkout used for the run |
| `runtime_tree_sha256` | The version-specific executable BASS implementation |
| `protocol_digest` | The immutable gate design, thresholds, and expected evidence |

A documentation-only commit may change the first identity without changing the
runtime tree. A gate edit changes the protocol digest without pretending the
model implementation changed. Qualifying work orders must nevertheless be
created from a clean tree and match all frozen identities.

`runtime_contract.audit_base_revision` is provenance: it identifies the
revision whose audit caused the contract revision. It is deliberately not
treated as the executable runtime identity; `runtime_tree_sha256` provides the
enforced compatibility rule.

## Required adapter records

| Contract | Minimum per-case fields | Non-negotiable behavior |
|---|---|---|
| `architecture-cohort-v1` | index, canonical genome/hash, version, stratum, seed, selection rule | Freeze before measurements; retain boundaries and duplicates selected by rule |
| `model-profile-v1` | hash, Params, traced FLOPs, raw/warm latency, shape, batch, precision, device, status/error | Synchronize device; analytical FLOPs do not qualify |
| `memory-case-v1` | hash, mode, peak bytes, allocator/device counters, batch/input/precision, status/error | One architecture per fresh process; OOM remains a case |
| `proxy-value-v1` | hash, proxy name/version, seed, value, runtime, finite flag, error | Store every repeat; do not average away failures |
| `training-result-v1` | hash, recipe/dataset digests, seed, checkpoints, PSNR/SSIM, cost, status/error | Same frozen recipe within a comparison; retain all seeds |
| `search-history-v1` | run seed, generation, genomes/hashes, objectives, rejected duplicates, mutations, timing | Persist every generation, not only the final Pareto set |
| `ablation-result-v1` | pair/block ID, factor/level, match deltas, seeds, outcomes, costs, diagnostics | Mark unmatched pairs and forbidden/null combinations explicitly |

JSONL is the interchange format for per-case data. Summary JSON may be added
but never replaces raw records. Large tensors/checkpoints may live in object
storage if the manifest contains stable URIs and hashes.

## Failure semantics

- `FAIL` means the run was technically complete and at least one frozen
  criterion failed.
- `ERROR` means the planned evidence could not be completed; it never counts as
  a pass and must preserve completed partial records.
- `NOT_RUN` is the only honest status before hardware execution.
- `SKIP` is not a result status. A conditional gate can be omitted only in the
  final ledger with a signed justification (for example, no aggregate proxy was
  preregistered).

## Schema validation is not evidence verification

`bass-gates validate-result` checks the envelope, originating work order, and
frozen criterion mapping. It intentionally reports `verification: schema` and
does not claim that a URI exists or that its bytes match the manifest.

`bass-gates verify-result` additionally verifies ordered timezone-aware
timestamps, the observed hardware minima, local artifact containment and
existence, and SHA-256 digests computed from the actual bytes:

```bash
bass-gates verify-result v3 V3-G06 result.json \
  --work-order work-order.json --artifact-root runs/v3-g06
```

Remote object-store URIs require an explicit laboratory resolver; absence of a
resolver is a verification failure, not implicit trust. Scientific statistics
remain the responsibility of the versioned analysis scripts and human review.
Neither schema validation nor byte verification manufactures a scientific
conclusion.

## Final gate ledger

Execution results retain only `NOT_RUN`, `PASS`, `FAIL`, and `ERROR`.
`JUSTIFIED_SKIP` is a separate final-ledger disposition and is legal only for a
gate whose protocol decision mode is `conditional`. A `GO` ledger requires a
verified `PASS` digest for every mandatory gate, and either `PASS` or
`JUSTIFIED_SKIP` for each conditional gate. Final decisions require a reviewer
identity and a timezone-aware signature timestamp; a `PENDING` ledger remains
unsigned.

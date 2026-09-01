# Result contracts for hardware and training adapters

The repository does not prescribe a laboratory scheduler, dataset mount, or
GPU telemetry stack. It does prescribe what those adapters must return so a
gate cannot pass by silently losing inconvenient cases.

Every result uses the envelope defined in
[`result-contracts.json`](../src/bass/experiments/protocols/result-contracts.json)
and contains:

- the exact protocol ID/digest, work-order digest, gate ID, clean source
  revision, start/finish timestamps, expanded command, and observed environment;
- `smoke`, `qualifying`, and one of `NOT_RUN`, `PASS`, `FAIL`, or `ERROR`;
- one disposition for every frozen criterion, in the original order;
- immutable artifact paths plus SHA-256 digests; and
- notes, deviations, exclusions, OOMs, and failed seeds.

`PASS` is rejected when the run is smoke/nonqualifying, its source/protocol does
not match the supplied work order, a criterion is false, or a review gate lacks
a reviewer identity.

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

The validator checks the envelope and frozen criterion mapping. Scientific
statistics remain the responsibility of the versioned analysis scripts and
human review; schema validation is not a rubber stamp for a conclusion.

# Round-2 hybrid NAS audit: disposition after BASS V3

This document reconciles the independent **BASS V2 Hybrid NAS Search-Space
Audit — Round 2** with the repository after the implementation of BASS V3.
The audit examined revision `cbc473d7c3ad127081a704864a95a85b27fca012`,
before `bass.v3` existed. Its central warning remains valid: executable code is
not evidence that a full NAS study is scientifically ready.

The audit is treated as a falsification checklist, not as a specification to
copy blindly. In particular, its advice not to inflate V2 is preserved by
freezing the V2 phenotype and placing interaction search in an independent V3
namespace. It would not be sound to delete V3 merely because the audited
revision predated it.

## Issue register

| ID | Disposition after V3 | Evidence or remaining gate |
|---|---|---|
| R2-001 nonuniform canonical prior | **Resolved narrowly in search code** | `sample_canonical_genome()` exhaustively catalogs 68,923 canonical V2 branch states and samples unordered three-branch multisets directly. V3 rejection-weights valid exchange counts. This removes raw-preimage bias, but canonical-uniform is not complexity-uniform: the 10k V2 audit averaged 8.9234 active units out of 9. Complexity/cost stratification therefore remains an explicit experimental choice. |
| R2-002 configuration count is not cost balance | **Audit accepted; experiment open** | Documentation no longer equates the 7/7 catalog split with FLOP, latency, or memory fairness. Run the computational profiling gate before any family-fairness claim. |
| R2-003 branch rank affects crossover | **Resolved in V2 and V3** | Crossover samples three branch tokens without replacement from the complete six-parent branch multiset, then canonicalizes. Lexicographic rank is no longer used as biological correspondence. |
| R2-004 whole-state mutation is nonlocal | **Resolved in V2 and V3** | Unit moves are typed as `repeat`, `argument`, `operation`, `family_flip`, `insert`, or `delete`; V3 exchange moves are insert/delete/prototype changes. Probabilities are public constants and attempted transitions, including rejected duplicates, are recorded in search history. |
| R2-005 AZ-score/proxy pipeline absent | **Open scientific gate** | No unsupported “AZ-score” was invented. Candidate proxies must be implemented and calibrated against short-trained SR outcomes before ranking architectures. |
| R2-006 500-model executable gate incomplete | **Open large gate** | Unit and smoke validations pass, but the repository does not claim that the prescribed 500-model V2 or V3 run has completed. |
| R2-007 Params/FLOPs collinearity | **Open measurement/decision** | Profile the intended candidate distribution, report correlations, then retain, transform, or remove redundant objectives. |
| R2-008 peak memory boundary missing | **Open hardware gate** | Measure worst-case candidates at target patch sizes and scales on the target accelerator. Parameter count is not a memory proxy. |
| R2-009 family strata too coarse | **Partly resolved** | Audit/validation records now include active-unit fraction, repeat-weighted family depth, and total repeat depth. FLOP share still requires the real profiler. |
| R2-010 structural audit lacks compute distributions | **Partly resolved** | Structural depth descriptors were added. The separate 10k computational profiler remains required because TensorFlow FLOPs/latency/memory cannot be inferred honestly from labels. |
| R2-011 reference-direction EA cross-validation | **Open** | The implementation remains named `ReferenceDirectionEA`; `NSGA3` is only a compatibility alias. Compare against a maintained NSGA-III implementation before algorithmic equivalence claims. |
| R2-012 residual-head ablation | **Open experiment** | `head_mode="direct"` is available as the controlled alternative; the ablation result is not fabricated. |
| R2-013 taps hide unequal depth | **Resolved at interface level** | `feature_tap_metadata()` exposes operation, argument, repeat, cumulative repeat depth, internal attention-block count, and the following V3 exchange. Proxy-bias testing remains open. |
| R2-014 heterogeneous primitive depth | **Intentional, experiment-gated** | Search expressivity is retained; family/cost-balanced proxy and short-training analyses must test the confound. |
| R2-015 compatibility facades hide migration | **Policy enforced** | Research examples and CLI selectors use `bass.v1`, `bass.v2`, or `bass.v3` explicitly. Top-level facades remain compatibility dispatchers only. |
| R2-016 legacy SynFlow alias | **Resolved for V2/V3 execution** | `metric="synflow"` is rejected. Calling the retired V2/V3 alias now raises instead of silently evaluating the repository's different `gradient_flow` quantity. |

## Why V3 is compatible with the audit's restraint

V3 does not add another primitive to V2's already uncharacterized 14-entry
unit catalog. It creates a separate question: whether searched BASS branches
benefit from compact communication after stages 1 and 2. The macro identity is
unchanged—one stem, three branches, three stages, final element-wise addition,
and an SR head—and `none/none` delegates to the exact V2 graph.

CIMEX is also aligned with the audit's *optional-after-validation* direction
of one efficient long-range/content-aware mechanism. That conceptual alignment
does not establish novelty or performance. CIMEX remains a candidate research
contribution until matched-cost ablations and SR benchmarks falsify simpler
explanations.

## Frozen execution contracts

The open work is now specified rather than merely listed:

- V2 has 14 gates (`V2-G00` through `V2-G13`);
- V3 has 14 gates (`V3-G00` through `V3-G13`), including the Round-3
  stage-aware canonical-equivalence gate;
- every gate declares dependencies, cohort, hardware, command contract,
  artifacts, criteria, and automatic/review/conditional disposition; and
- `bass-gates prepare` creates a source-bound work order and Slurm template.

The protocol deliberately refuses several tempting but unsound shortcuts. It
does not invent one universal proxy-correlation threshold, treats absolute
Params/FLOPs Spearman of 0.95 as a review trigger rather than automatic
objective deletion, makes memory relative to a preregistered target device,
and calls CIMEX pairs matched-cost only when **both** Params and traced FLOPs
are within 5%.

The released CIMEX gate remains `0.01`: it avoids making enabled exchange
invisible to the current one-backward proxy. Exact `alpha=0` is still tested as
an identity boundary and matched-training ablation. Requiring nonzero projection
gradients at exact zero would contradict the chain rule, so the full subpath is
checked with the released/open gate instead.

See [`experiments/README.md`](../experiments/README.md), the
[`result contracts`](../experiments/RESULT_CONTRACTS.md), and the staged
[`CIMEX ablation matrix`](../experiments/CIMEX_ABLATIONS.md).

## Gates that are defined but not run

The following work remains mandatory before **GO FOR FULL NAS**:

1. profile at least 10,000 stratified candidates for real Params/FLOPs and
   family cost distributions;
2. execute the 500-model build/forward/backward gate with zero unexplained
   failures;
3. measure peak accelerator memory on the boundary architectures at target
   patch sizes and scales;
4. determine whether Params and FLOPs are independent enough to retain as
   separate objectives;
5. implement candidate proxies, then calibrate 500–1,000 family- and
   cost-balanced candidates;
6. validate proxy ranks against short-trained PSNR on a stratified sample;
7. run the residual-head and CIMEX mechanism/site/prototype ablations;
8. cross-validate `ReferenceDirectionEA` and run multi-seed dry searches with
   duplicate, transition, family-drift, width-drift, and objective-correlation
   reports; and
9. retrain any learned surrogate on the canonical schema actually searched.

Until those gates pass, the correct status is: **runtime implemented,
scientific full-NAS readiness NO-GO**. That is a stronger and more useful claim
than either “V3 is done” or “V3 is invalid.”

The ledger intentionally starts at 0/14 qualifying V2 gates and 0/13
qualifying V3 gates. Local unit and smoke evidence is retained, but is not
relabeled as hardware validation.

Both structural scripts accept `--sampling-prior canonical` (default) or
`--sampling-prior conditioned`. Reports must name the prior; results from the
two distributions must never be pooled as though they were interchangeable.

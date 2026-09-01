# BASS V2 audit response and scientific readiness

This document records the disposition of every issue in the independent audit
of revision `d846f8e`. Passing unit tests means that a search space executes; it
does not by itself establish fair NAS behavior. V2 therefore remains gated from
full publication-scale NAS until the empirical stages below are completed.

## What changed

| Audit issue | Disposition | Implementation |
|---|---|---|
| HN-001 genotype aliases | Resolved for scientific search | 10 semantic integers; 43 complete unit states; no modulo or inactive fields |
| HN-002 branch symmetry | Resolved | skips packed, repeat runs normalized, branches sorted before storage |
| HN-003 residual asymmetry | Resolved structurally | every searchable CNN and attention transform is residual |
| HN-004 inverted bottleneck | Resolved | expansion, depthwise transform, linear projection, shortcut |
| HN-005 missing position | Resolved | convolutional positional encoding in every window block |
| HN-006 cost-biased CNN pool | Resolved only at catalog level | seven CNN and seven attention configurations; measured cost balance remains an experimental gate |
| HN-007 SynFlow naming | Resolved | V2 exposes `gradient_flow`; `synflow` is rejected as a metric name |
| HN-008 AZ-score absent | Intentionally pending | no unvalidated aggregate proxy was invented; calibration is an experiment gate |
| HN-009 bit crossover | Resolved | crossover recombines the unordered six-parent branch multiset; mutation uses typed semantic moves |
| HN-010 shifted repetition | Resolved | searchable regular/shifted pair guarantees cross-window communication |
| HN-011 amplify-only channel gate | Resolved | gate modulates a signed learned delta under a scaled residual |
| HN-012 stride-1 transpose conv | Removed | retained only in the retired 93-bit import/export description |
| HN-013 global SR residual | Implemented, ablation required | bicubic image skip and linear residual head are V2 default |
| HN-014 simplified NSGA-III | Partly resolved and renamed | Deb-Jain normalization added; class is `ReferenceDirectionEA` pending external cross-validation |
| HN-015 silent gradients | Resolved | disconnected/non-finite variables are reported and fail strict evaluation |
| HN-016 V1 predictor reuse | Rejected | V1 predictors are explicitly invalid for V2 and must be retrained |
| HN-017 attention fraction | Clarified | active-family fraction plus full operator counts; never described as compute share |
| HN-018 objective collinearity | Empirical gate | measure real Params/FLOPs correlation before freezing objectives |
| HN-019 silent repair | Resolved | strict persisted specs; canonicalization never substitutes a nearest operation |
| HN-020 restrictive head | Ablation-ready | fixed residual-linear default and explicit `direct` legacy-head alternative |

## Corrections and deliberate resolutions

The proposed 43-state encoding is not automatically one-to-one. Adjacent
groupings such as `A x2` followed by `A x1` and one `A x3` unit describe the
same sequence of independently parameterized residual blocks. V2 additionally
expands and deterministically recompresses adjacent equal operations before
hashing. A run is stored in maximal repeat-3 chunks, followed by one remainder.
This removes repeat-boundary aliases while retaining three unit slots.

The proposed final catalog is also internally inconsistent: its table contains
seven non-skip CNN configurations and five attention configurations, while its
43-state calculation assumes fourteen non-skip configurations. V2 fills the two
missing entries with `hybrid_conv_window` at windows 4 and 8. This keeps a 7/7
configuration-level family prior and retains the only operator that explicitly
fuses local convolution with window context. It is a deliberate, ablation-ready
choice, not something established by the audit's arithmetic.

## Scientific V2 representation

```text
gene[0]      channel id in 0..3
gene[1:10]   nine complete semantic unit states in 0..42
```

State zero is `skip`. The other 42 states are 14 complete primitive
configurations times repeats 1, 2, and 3. Kernel/window selection is part of the
primitive configuration, so no field is inactive.

The old 93-bit representation is retained only in `bass.v2.legacy93` for
inspection and round-trip storage. It is not accepted by the V2 scientific
problem. Migration from V1 or legacy93 is explicit and approximate because the
new residual, cost-balanced catalog intentionally removes operations.

## Decisions not adopted blindly

- No global spatial attention was added; its quadratic LR feature-map cost is
  still a poor default for this search.
- No DAT, token dictionary, Mamba, frequency, or focused-attention primitive was
  added. Expanding the catalog before validating the core 14 configurations
  would weaken rather than strengthen the study.
- No synthetic AZ-score implementation was added. The intended proxies need
  family-balanced calibration against short-trained PSNR first.
- Channels remain 16/32/48/64 and three BASS branches remain fixed. The audit
  correctly labels these as defensible design choices, not defects.
- Exact FLOP matching between operations is not imposed. A 7/7 configuration
  count is categorical symmetry, not cost balance; real FLOPs, latency, and
  memory must be measured and reported by family.
- Primitive internals still have different learned depths. A residual scale is
  shared across families, but that does not make a convolution and a Transformer
  block computationally identical; proxy and short-training gates must test the
  remaining confound.

## Readiness gates

1. Unit gate: all 14 primitive configurations x four channel widths, forward,
   backward, finite values, save/load, non-divisible shapes, and scales.
2. Structural preflight: run `scripts/audit_v2_space.py --samples 10000`; the
   qualifying Round-2 gate repeats the frozen analysis at one million draws.
3. Executable gate: run `scripts/validate_v2_models.py --samples 500`; require
   zero unexplained failures and full gradient coverage.
4. Proxy gate: calibrate each intended proxy on 500-1000 family-balanced models;
   report finite rate, seed sensitivity, and correlation with width, depth,
   family, Params, and FLOPs.
5. Short-training gate: train a stratified sample and establish rank correlation
   with PSNR before aggregating proxies.
6. Search dry run: report unique phenotype rate, duplicate rejections, semantic
   mutation distance, family drift, and objective correlations.
7. Full NAS only after gates 1-6 pass.

The code now addresses structural and algorithm-interface biases. It does not
claim that proxy validity or publication-scale search conclusions have already
been established. The later Round-2 findings and their post-V3 disposition are
tracked in [`ROUND2_AUDIT_RESPONSE.md`](ROUND2_AUDIT_RESPONSE.md).
The exact hardware and result hand-off is
[`../experiments/README.md`](../experiments/README.md).

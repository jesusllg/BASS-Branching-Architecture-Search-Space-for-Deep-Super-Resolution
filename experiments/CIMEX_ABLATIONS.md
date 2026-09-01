# CIMEX staged ablation matrix

This matrix tests whether CIMEX contributes useful cross-branch information and
which part is responsible. It is intentionally staged: first establish that the
mechanism is viable, then test placement and searchability. Running every switch
in a kitchen-sink factorial would create costly, non-identifiable, and sometimes
algebraically null comparisons.

## Common controls

All cells use the same frozen SR recipe, architecture pair, initialization/data
seeds, evaluation code, and device class. Report PSNR, SSIM, any preregistered
perceptual metric, convergence, branch cosine/CKA similarity, gate trajectories,
update norms, Params, traced FLOPs, latency, and peak memory.

A pair is `matched_cost=true` only when both

\[
\frac{|P_a-P_b|}{\max(P_a,P_b)} \le 0.05
\quad\text{and}\quad
\frac{|F_a-F_b|}{\max(F_a,F_b)} \le 0.05.
\]

Otherwise retain the result as unmatched and include cost as a covariate or
construct a closer width/depth control. Never delete the larger model after
seeing quality.

## Stage A — mechanism viability

Run each row as a paired change from full CIMEX, plus the no-exchange V2
subspace control.

| ID | Factor | Control | Variant | Question |
|---|---|---|---|---|
| A00 | Exchange | `none` | full CIMEX | Does communication help at all at matched cost? |
| A01 | Feature split | consensus + innovation | raw branch features | Does the decomposition add value beyond generic token memory? |
| A02 | Memory source | innovation memory | consensus-conditioned shared memory | Is branch-specific evidence actually used? |
| A03 | Read set | other branches only | self only | Is the block exchanging rather than re-encoding? |
| A04 | Read set | other branches only | all branches | Does excluding self prevent shortcut behavior? |
| A05 | Correction | centered | uncentered | Does zero-sum correction preserve specialization or overconstrain useful drift? |
| A06 | Projections | shared across branches | branch-specific | Is symmetry/parameter sharing beneficial? |
| A07 | Initialization | released small nonzero alpha (`0.01`) | exact `alpha=0` identity | Does function-preserving insertion outweigh one-step proxy blindness and delayed subpath learning? |
| A08 | Compact context | prototype memory | matched local/window attention | Is the proposed memory better than a credible efficient context control? |

Stage A passes only if the full mechanism has a repeatable practical advantage
over no exchange or a simpler matched-cost control and its diagnostics support
cross-branch use. A significant improvement by a simpler variant is a reason to
revise CIMEX, not hide the ablation.

## Stage B — placement and searchability

Use only the Stage-A-selected mechanism.

| ID | Factor | Levels |
|---|---|---|
| B00 | Placement | none, early only, late only, both |
| B01 | Prototype count | k=8, k=16 |
| B02 | Decision policy | fixed best placement, searchable exchange genes |
| B03 | Search-space scope | exact V2 subspace, V3 mixed space |
| B04 | Weight budget | natural width, matched-cost width/depth control |

Placement comparisons require at least one active branch-specific transform
after an exchange. Searchable-vs-fixed comparisons use the same total search
and validation budget over at least five seeds.

## Forbidden or null cells

These combinations are not valid negative evidence:

- centered corrections immediately before final branch addition with no
  downstream branch-specific transform: their sum cancels exactly;
- identical consensus input, shared mapping, and centered outputs: every branch
  receives the same correction, which centering removes;
- `alpha=0` frozen for the whole run: this is exactly no exchange with dead
  parameters, not a learned CIMEX condition;
- a one-step gradient assertion requiring nonzero projection gradients while
  `alpha=0`: those gradients are zero by the chain rule; test alpha on step zero
  and the full path after the gate opens;
- “matched” comparisons based on Params alone while FLOPs differ by more than
  5%; and
- dropping OOM, divergence, or failed seeds from one condition.

The machine-readable work contracts refer to these IDs so the external runner
can reject forbidden cells before consuming GPU time.

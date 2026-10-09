# Whitened Decodability Retention (WDR): method spec and branch-test plan

Status: **PLAN + IMPLEMENTATION STAGED — NO FORMAL TRAINING DISPATCH BEFORE USER REVIEW**
(Rules `README.md` §2.1 requires a written campaign plan before launch.)
Scored by [10_Capability_Regret_Objective.md](../10_Capability_Regret_Objective.md).

## 1. What the measurements force the method to do

All five constraints below come from runs already on disk, not from assumption.

**C1 — Continued training preserves the variance head and destroys the tail.**
Ridge map from the late checkpoint (ck29279) to the early one (ck12687), fit on train
and scored on test, resolved per early-checkpoint eigen-direction:

| anchor eigen-rank band | 0-8 | 8-32 | 32-128 | 128-512 | 512-2048 |
|---|---:|---:|---:|---:|---:|
| bloodmnist | 0.896 | 0.613 | 0.402 | 0.169 | 0.038 |
| pathmnist | 0.978 | 0.892 | 0.741 | 0.417 | 0.147 |
| nct-crc-he | 0.974 | 0.818 | 0.489 | 0.251 | 0.151 |
| chammi-cp-task3 | 0.811 | 0.374 | 0.328 | 0.205 | 0.124 |
| organcmnist | 0.950 | 0.863 | 0.663 | 0.289 | 0.043 |

**C2 — Any variance-weighted retention loss therefore optimises the safe subspace.**
Plain feature L2, relational/Gram matching and parameter averaging all weight
directions by their variance, so their gradient is dominated by the ranks that already
have decodability above 0.9, and is blind to the ranks below 0.2.

**C3 — The published Gram anchoring makes things measurably worse.** Matched branch
from ck12687, identical stream, steps, optimiser and evaluator, scored at ck18543:
MNR 4.252 (vanilla) versus 8.665 (official Gram); retention subset 6.486 -> 10.107 and
plasticity subset 2.504 -> 7.535. It rescues a few cells (multimodal_cellseg regret
9.71 -> 0.00) while destroying others (MoNuSeg 10.44 -> 52.88, PanNuke 8.11 -> 33.18).

**C4 — Post-hoc combination does not solve it, so the fix must be during training.**
On the 29 cross-campaign-comparable capabilities, the oracle single-checkpoint floor is
MNR 2.475; the best post-hoc arm is parameter averaging (E+M+L)/3 at 2.391, and feature
concatenation is far worse (E+L 4.998, M+L 4.783).

**C5 — Maturation times span the whole run**, from update 487 to 28791, so a single
frozen anchor cannot cover the retention set.

## 2. The method

### 2.1 Anchor construction (once per anchor, offline)

A fixed unlabeled calibration set `X_cal` is drawn from the pretraining stream, frozen
by manifest, and never touched by any downstream evaluation. At anchor time the frozen
EMA teacher is run over `X_cal` to give `A` in `R^{n x d}`. Store the mean `mu_A` and
the eigendecomposition `(U_A, Lambda_A)` of its covariance. Keep the top `r`
directions that survive a **label-free reliability test**: split `X_cal` in half, keep
direction `j` only if its two half-sample eigenvalues agree within a fixed tolerance,
so pure-noise directions are never whitened up. Precompute

    Z = Lambda_r^{-1/2} U_r^T (A - mu_A)      in R^{n x r},  every column unit variance.

### 2.2 Retention loss

Every k-th step, draw a minibatch from `X_cal`, forward the **student only** to get `S`,
and apply a learned linear decoder `g_phi: R^d -> R^r`:

    L_ret = mean_j mean_B ( Z[B,j] - (S phi + b)[B,j] )^2        (= 1 - R^2, equally weighted over j)

`phi` trains with the model; the student receives the gradient through `S`.

Two design choices, each answering a constraint:

- **Whitening answers C1/C2.** Each of the `r` directions contributes equally, so the
  loss spends its capacity exactly where decodability has collapsed.
- **A learned linear decoder, rather than feature matching, protects plasticity.**
  `L_ret` is invariant under any invertible linear reparametrisation of `S`: the student
  may rotate, rescale and add directions freely and is forbidden only from *deleting*
  anchor information. This is the weakest constraint that still guarantees retention for
  a linear probe, and every readout in the protocol is a linear probe. Feature matching
  (`S ~ A`) forbids all of it, which is the C3 failure mode.

### 2.3 Erosion-adaptive gate (answers C3)

Track an EMA of the achieved whitened `R^2`. With `R2_ref` measured right after minting
(approximately 1, since the anchor is the model itself at that moment),

    lambda(t) = lambda_0 * clip( (R2_ref - R2_bar(t)) / R2_ref , 0, 1 )

The term is inactive while the student still decodes the anchor and switches on in
proportion to measured loss. On a trajectory that is not forgetting, WDR is a no-op;
a fixed-weight Gram term is not.

### 2.4 Multi-anchor memory (answers C5)

Mint a new anchor when the running whitened `R^2` against the newest anchor falls below
a threshold - a label-free trigger that never consults downstream metrics. Cap the
memory at three anchors; when full, drop the one whose whitened subspace is most
redundant with the union of the others (largest mean squared canonical correlation).
Per-anchor cost is one `n x r` fp16 bank (65 536 x 512 x 2 B = 64 MiB) and one `r x d`
decoder.

### 2.5 Cost

No frozen-teacher forward pass at training time: the anchor is a precomputed bank. One
extra student forward on a small calibration batch every k steps plus an `r x d` matmul.
At `|B_cal| = 64`, `k = 4`, `r = 512` the overhead is a few percent, strictly below
Gram anchoring, which needs a live clean-crop teacher forward every step.

### 2.6 Separate, non-training companion

Independently of WDR, the evaluation readout depth is itself costing accuracy: on
organcmnist, block 15 of 24 beats the final block by +0.029 at every checkpoint
(ck12687 0.9169 vs 0.8871; ck29279 0.9164 vs 0.8930), which is larger than the entire
early-to-late gap. A **fixed, declared-in-advance multi-block readout** is therefore
reported as a separate protocol variant, applied identically to every arm including all
baselines, and never mixed with WDR's effect.

## 3. Branch test

From ck20007 of the 5TB no-Gram run, 2440 updates. Identical data stream, step count,
learning-rate schedule, augmentation, optimiser state, crop geometry and checkpoint
cadence across arms; the added regulariser is the only difference.

| arm | added term | isolates |
|---|---|---|
| A0 | none (vanilla continuation) | control |
| A1 | official within-image Gram | published baseline |
| A2 | unwhitened feature distillation to the anchor | the whitening in 2.2 |
| A3 | WDR without the adaptive gate | the gate in 2.3 |
| A4 | WDR full | the method |

Fail-closed gates, following the ck29279 Gram plan: identical per-update sample-key
digests across arms, identical real and effective batch, all endpoint teacher snapshots
present, and no use of downstream labels to select the anchor, `lambda_0`, `r`, or the
stopping point.

**Pass condition.** Against the frozen reference and split at `t_branch = 20007`:
MNR below the oracle single-checkpoint floor of 3.421, MNR down on the retention set,
and MNR not up on the plasticity set. Anything else is reported as a failure of WDR,
not rewritten into a partial success.

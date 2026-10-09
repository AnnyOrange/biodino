# Capability regret: the objective every later method must optimise

Status: **proposed metric definition, frozen reference computed 2026-09-23.**
It does not retroactively rename or rescore any existing v1/v2/v3/v4 campaign.
Scope: the HS6-L 5TB no-Gram trajectory is the reference run; the same construction
transfers to any other reference run by re-freezing its own table.

## 1. Why endpoint average accuracy is the wrong target

Biological capabilities do not mature together. On the audited 60-checkpoint
full-registry curve of the 5TB no-Gram run
(`outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908`),
41 task-dataset metrics have complete coverage and their maturation times span
essentially the whole run: from update 487 (2% of training) to update 28791 (98%).
Twenty-five of the 41 end at least two noise sigma below their own best value
earlier in the same run. An endpoint average hides this completely: the endpoint can
look flat while individual capabilities are 10-17 sigma past their peak.

## 2. Definition

For capability `i` measured along a reference trajectory at checkpoints `t_1..t_T`:

| symbol | definition |
|---|---|
| `S~_i(t)` | centred moving average of `S_i` over 5 checkpoints (shrinking at the edges) |
| `sigma_i` | `1.4826 * MAD(S_i - S~_i)` — noise scale taken from the curve itself |
| `S*_i` | `max_t S~_i(t)` — the smoothed historical best, the **frozen reference** |
| `tau_i` | first `t` with `S~_i(t) >= S*_i - sigma_i` — maturation step |
| `R_i(T)` | `S*_i - S_i(T)` — historical regret |
| `r_i(T)` | `max(0, R_i(T)) / sigma_i` — normalised regret |
| `u_i(T)` | `max(0, -R_i(T)) / sigma_i` — normalised surplus |

Raw `max_t S_i(t)` is not used as the reference because it is biased upward by
evaluation noise; the smoothed best and a 3-checkpoint window best are both recorded,
and the smoothed best is the one that is frozen.

Aggregates:

```
MNR(T)   = mean_i r_i(T)      primary, minimise
maxNR(T) = max_i  r_i(T)      worst-case guard, minimise
MNS(T)   = mean_i u_i(T)      genuine gain beyond the reference, report
```

## 3. Why MNR encodes retention and plasticity at once

`S*_i` is the best value capability `i` ever reached **anywhere on the reference
trajectory**, so:

- Freezing an early checkpoint leaves every late-growing capability far below its
  own `S*_j`, which shows up as large `r_j`. Retention bought by stalling is punished.
- Plain continuation leaves every early-matured capability below its own `S*_i`.
  Plasticity bought by forgetting is punished.

MNR therefore falls only when a representation is simultaneously close to the
reference peak of early-matured and late-growing capabilities. Retention and
plasticity are not two scores to be traded; they are two ways of failing one score.
`maxNR` is reported alongside so that a method cannot average away a single
sacrificed family.

When a method must be reported against the two conditions separately, split the
capability set at the branch point: **retention set** = `tau_i <= t_branch`,
**plasticity set** = `tau_i > t_branch`, and report MNR on each. A method passes
only if MNR falls on the retention set and does not rise on the plasticity set.

## 4. The bar: the oracle single-checkpoint floor

Any checkpoint of the reference run is itself a candidate representation, so the
honest baseline is the best one, chosen with an oracle:

```
floor = min_t MNR(t)
```

On the 41-metric reference set this is **MNR = 3.421 at update 20007**, against
**MNR = 4.167 at the terminal update 29279**. A method that does not beat 3.421 has
not solved anything that picking a different checkpoint would not have solved. Note
that the floor uses test-side information to pick `t`, so it is an upper bound on
what checkpoint selection can do, not a usable selection rule.

## 5. Rules of use

1. The reference table `(S*_i, sigma_i, tau_i)` is frozen once per reference run and
   published: `outputs/00_reports/hs6_l5_capability_trajectory_20260923/capability_reference_frozen_20260923.json`.
   Methods are scored against that file; recomputing `S*` to include a method's own
   results is forbidden.
2. A method's checkpoints must be evaluated with the identical evaluator, splits,
   geometry, batch and seed as the reference curve. Cross-campaign reuse requires
   the arm to reproduce the reference at shared checkpoints within 1 sigma per
   metric; the scorer refuses the metric otherwise and states the offset.
3. Missing or blocked cells stay missing. They are never scored as zero and never
   dropped silently from the inventory.
4. OOD metrics are kept out of the ID aggregate and reported separately.
5. Segmentation and detection under a different probe budget are a different
   measurement of the same dataset; they need their own reference table and may not
   be mixed with the curve reference.

# CTC native HS6 candidate set (precommitted 2026-09-11)

The native 20-domain CTC run is intentionally a second stage after the complete
checkpoint screens. The candidate set is frozen before any native held-out CTC
result is observed:

| Family | Candidate checkpoints | Selection evidence available before native CTC |
|---|---|---|
| S+ 1TB | 8199, 15374 | count-proxy MAE/Spearman peak; late endpoint/control |
| B 1TB | 6149, 9224, 14349 | early boundary requested by user; count R2/MAE peak; full-Rx peak |
| L 1TB | 12299, 14349 | count R2/MAE peak; count Spearman control |
| H+ 1TB | 15374 | only complete H+ checkpoint |
| L 5TB | 15615, 17079, 20007, 21959 | full-Rx region; count Spearman peak; 59-metric registry peak/count R2 peak; late count R2/Spearman control |

This is 12 native CTC model/checkpoint rows, each evaluated on all five fixed
folds and all 20 domains. The 62-checkpoint quick screen remains useful for
curve diagnosis, but running all 62 through 364 GB of native 2-D/3-D tracking
would spend most compute on points already dominated by multiple independent
screens. No candidate may be added after viewing native held-out results.

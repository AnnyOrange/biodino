# CTC + RxRx3 HS6 scaling quick screen (2026-09-11)

## Scope

- Datasets: CTC and RxRx3_core only.
- Models: HS6 1TB S+, B, L, H+ and the latest HS6 5TB model (L).
- This is an observational frozen-feature quick screen. It does not satisfy the
  native CTC tracking or full eligible-gene RxRx3 formal protocols in v3.
- The image cache, preprocessing, split, probe, seed, readout and metrics are
  identical to `external4_hplus_fm_fixedbudget_3090qi_20260910`, so its H+ result
  is reused rather than recomputed.

## Model matrix

| model_id | corpus | architecture | checkpoint | selection |
|---|---:|---|---:|---|
| hs6_splus_1tb_ck15374 | 1TB | ViT-S+ | 15374 | final valid checkpoint |
| hs6_b_1tb_ck13324 | 1TB | ViT-B | 13324 | latest complete local checkpoint; 4099 is corrupt and no complete later checkpoint exists |
| hs6_l_1tb_ck15374 | 1TB | ViT-L | 15374 | final valid checkpoint |
| hs6_hplus_1tb_ck15374 | 1TB | ViT-H+ | 15374 | final valid checkpoint; exact-result reuse |
| hs6_l_5tb_ck23911 | 5TB mixed | ViT-L | 23911 | newest complete checkpoint as of launch |

All HS6 encoders use teacher weights, 224 px input, one final block, CLS plus
patch-mean frozen feature, L2 normalization, batch size 4 and seed 0.

## Dataset protocols

- CTC: `ctc_2d_count_proxy_v1`, 40 train + 40 held-out frames across 10 domains;
  Ridge(alpha=10); R2, MAE and Spearman. This is a count proxy, not native TRA/SEG.
- RxRx3: `rxrx3_plate_disjoint_128_v1`, 128 gallery + 128 query wells;
  cosine retrieval and KMeans; Recall@1/5/10, MRR@10 and NMI.

## Execution

- Host: `3090-qi`
- Physical GPUs: 2, 5, 6, 7 (one model process per GPU)
- Shared cache:
  `outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910/cache`
- Output:
  `outputs/02_eval_runs/ctc_rxrx3_hs6_scaling_quickscreen_3090qi_20260911`


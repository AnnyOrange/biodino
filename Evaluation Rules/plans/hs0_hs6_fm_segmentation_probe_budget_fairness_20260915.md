# HS0 / HS6 / External-FM segmentation probe-budget fairness

Status: `APPROVED_AND_LOCKED_BEFORE_LAUNCH`.

Approval record: user approved execution on 2026-09-15 and explicitly approved
H100 GPUs 0, 1, 2 and 3. Protocol changes after the first formal job starts must
be recorded as a versioned amendment; completed results from different protocol
versions may not be merged silently.

Date: 2026-09-15

## 1. Research question

Test whether HS6 frozen spatial embeddings outperform HS0 and 14 external
foundation models under equal linear-probe opportunity, and determine whether
20 or 50 probe epochs materially changes that conclusion.

The experiment must not choose a budget because it maximizes an HS6-minus-FM
test-set gap. Epoch, learning rate, head checkpoint, and any model ranking are
selected without test labels. Test is evaluated only after the selection rule
has been applied to validation results.

## 2. Models and frozen checkpoints

The full grid contains 23 frozen encoders:

- HS0 1TB: S+, B, L, H+ at ck8199.
- HS6 1TB retrospective-best display anchors: S+ ck9224, B ck10249,
  L ck13324, H+ ck15374.
- HS6 L-5TB retrospective-best display anchor: ck26351.
- External FM14: DINOv2, MAE, SigLIP2, PE, BioCLIP, CytoSelf, JUMP-CP,
  CytoImageNet, UNI, CONCH, Phikon-v2, Virchow2, GigaPath, H-optimus-0.

The retrospective HS6 checkpoints were selected using historical downstream
test results and therefore cannot support a confirmatory checkpoint-selection
claim. The paper must show this label explicitly. A confirmatory companion
table will use a checkpoint rule independent of downstream test labels:

- HS0 versus HS6 method comparison: common ck8199.
- HS6 endpoint sensitivity: ck15374 for each available 1TB size.
- L-5TB endpoint sensitivity: ck29279.

No dataset-specific encoder checkpoint selection is allowed.

## 3. Dataset matrix

Formal segmentation aggregate, seven datasets:

- Tier A: Cellpose, CoNIC, LIVECell, MoNuSeg, PanNuke, TissueNet.
- Tier B: Multimodal CellSeg.  The prepared source-heldout CSV manifests have
  876/176/101 rows; the evaluator's predeclared WSI/>50M-pixel/missing-file
  exclusion leaves 855/172/100 effective samples.  Both manifest and effective
  counts must be reported.

BBBC038 is run as an eighth, separately reported observation dataset. PanNuke
is three independent official fold rotations, so the physical run grid has ten
split cells per model: six non-PanNuke formal cells, three PanNuke cells, and
one BBBC038 observation cell.

Only current v3 split manifests and hashes are valid. In particular, no old
CoNIC 3984 split or PanNuke val=test result may be reused.  A Multimodal result
with 855 effective training samples is reusable only when its ordered sample
identity matches the current 876-row manifest after the documented exclusion;
the count alone cannot distinguish an old split from the current filtered one.

## 4. Primary cross-model representation protocol

Use the same last spatial feature map (`last1`) for every encoder. For DINO
models this means the final transformer block; for an external model it means
the final native spatial dense map. This is the primary cross-FM embedding
comparison because it avoids giving DINO four concatenated layers while an
external architecture supplies only one.

The existing dataset-best DINO `even4` protocol is retained only as a secondary
"best available readout" table. It must not be presented as the strict
cross-architecture linear-probe comparison.

Common probe and preprocessing:

- frozen encoder in eval mode; no backbone gradients;
- current dataset-specific image resolution, resize, channel policy and class
  weighting from protocol v3;
- head: the existing BatchNorm + Dropout(0.1) + 1x1 convolution linear head;
- AdamW, lr 1e-3, weight decay 1e-4;
- probe batch 32 for every model; seed set {0, 1, 2};
- primary selection metric: validation mIoU;
- reported test metrics: mDice primary, plus mIoU, AJI, AP50 and bPQ;
- no test-time augmentation unless enabled identically for the full grid.

Encoder extraction batch may vary by available memory because the frozen
encoder is in deterministic eval mode and extraction batch does not change
probe optimization. Any exception requires a numerical invariance check on 32
fixed samples (`max_abs_diff <= 1e-5`, identical sample order and cache hash).
Probe batch may not vary.

## 5. Probe-budget arms

Two independent optimizer schedules are required:

| Arm | Total epochs | Cosine horizon | Validation epochs | Role |
|---|---:|---:|---|---|
| E20 | 20 | 20 | every epoch, 1--20 | predeclared primary |
| E50 | 50 | 50 | every epoch, 1--50 | convergence sensitivity |

E20 cannot be obtained by taking epoch 20 from E50 because their cosine
schedules differ. For every HS0, HS6 and external-FM run, validate after every
epoch. Within E20 choose the best of epochs 1--20 by validation mIoU; within
E50 choose the best of epochs 1--50 by validation mIoU. Save that head and
evaluate it on test exactly once. Epoch 50 itself is never assumed to be best.

Three result views are allowed:

1. `fixed-E20`: primary comparison; all models receive 20 epochs and report
   their independently selected best-validation head within epochs 1--20.
2. `fixed-E50`: sensitivity comparison; all models receive 50 epochs and report
   their independently selected best-validation head within epochs 1--50.
3. `validation-selected-{E20,E50}`: for each model/dataset/seed, choose the
   budget with higher validation mIoU, then reveal the corresponding test
   result. Every model receives the same two-budget search space.

It is forbidden to report `test-selected-{E20,E50}` or to select one global
budget after examining which produces the largest HS6 advantage.

## 6. Required evaluator changes before launch

The current probe saves `best_head.pth` but does not record enough information
to audit budget selection. Before the pilot it must additionally save:

- `best_epoch`, `best_val_miou`, and all 20 or 50 per-epoch validation records;
- optimizer, scheduler type/horizon, lr, weight decay and dropout;
- global optimizer-step count and train batches per epoch;
- encoder extraction batch separately from probe batch;
- checkpoint/config fingerprint, feature-layer declaration, split hashes and
  cache sample-order hash;
- explicit `test_evaluations=1` assertion for each completed run.

Output paths must include model, dataset/fold, feature protocol, budget, lr and
seed. E20 and E50 never overwrite or resume from each other.

## 7. Execution stages

### Stage 0: implementation and smoke validation

Patch the shared DINO and external-FM runners so both call the same cached
probe function and emit the fields above. Verify a tiny smoke run, then exclude
all smoke outputs from aggregation.

### Stage 1: convergence pilot

Run DINOv2, BioCLIP, UNI, H-optimus-0, HS0-L ck8199, HS6-L ck13324 and HS6
L-5TB ck26351 on CoNIC, LIVECell, MoNuSeg and Multimodal. Run E20/E50 and all
three seeds. This is 168 inexpensive cached-head fits after feature extraction.

The pilot is a pipeline and convergence audit, not a place to choose the
winning model. Continue to Stage 2 unless a shared protocol defect is found.

### Stage 2: complete fair matrix

Run all 23 encoders across ten split cells, two budgets and three seeds:
23 x 10 x 2 x 3 = 1,380 cached-head fits. Extract each model/dataset feature
cache once and reuse it across budgets and seeds. Preserve result JSON, plan,
manifest, validation report and concise logs; caches may be removed after all
downstream result hashes validate.

### Stage 3: label-efficiency evidence

After Stage 2, select the three external comparators with the highest mean
validation rank, not test rank. Compare them with HS6 H+ 1TB and HS6 L-5TB at
1%, 10%, 25% and 100% of the training labels, E20, seeds 0/1/2. The same nested
subset indices are shared by every model. This is stronger evidence of
embedding quality than extending optimization until one model happens to win.

## 8. Machine placement

| Machine | Encoders |
|---|---|
| H100 | HS6 H+ ck15374 and H+ confirmatory anchors |
| hxw | HS6 B ck10249 and B confirmatory anchors |
| lyx | HS6 S+ ck9224 and S+ confirmatory anchors |
| Huawei shared GPU fleet | HS0 S+/B/L/H+, HS6 L-1TB, L-5TB, external FM14 |

Only feature extraction must remain close to the checkpoint. Once deterministic
feature caches are complete and fingerprinted, cached-head jobs can use any
compatible shared GPU without moving checkpoints.

H100 allocation: GPUs 0--3 are approved. Initial H+ extraction should prefer
the least occupied of these GPUs at launch time; cached-head work may use the
remaining approved GPUs. Resource placement does not alter the model, split,
feature, or probe protocol.

## 9. Reporting and success criteria

Report per-dataset values and mean +/- standard deviation across probe seeds.
For the seven formal datasets report macro mean, mean rank, win count, and the
paired per-dataset delta against the strongest external-FM test result. Use a
paired dataset bootstrap confidence interval for the macro delta. BBBC038 is a
separate observation and never affects rank.

The strongest defensible claim is conditional on observed results:

- "HS6 is the best frozen embedding under the fixed E20 protocol" only if it
  ranks first at E20;
- "the conclusion is robust to probe optimization budget" only if it remains
  first at E50 and under validation-selected budget;
- otherwise report exactly where the ranking changes rather than selecting the
  favorable panel as the headline.

Figure set:

1. main heatmap: fixed-E20, last1, seven formal datasets;
2. convergence panel: E20 versus E50 paired deltas for every model/dataset;
3. validation-selected budget table;
4. label-efficiency curves from Stage 3;
5. supplementary retrospective-best and dataset-best/even4 results with their
   selection caveats.

## 10. Launch gates

Do not launch until the user approves this plan and the following pass:

1. current split manifests and hashes exist for all ten cells;
2. result schema records best epoch and validation history;
3. one common probe implementation is used by HS0, HS6 and external FMs;
4. selected frozen checkpoint and teacher/EMA branch are fingerprinted;
5. extraction-batch invariance is validated where batch 32 is impossible;
6. no historical test score participates in head/budget selection;
7. output and cache space estimates are recorded per machine.

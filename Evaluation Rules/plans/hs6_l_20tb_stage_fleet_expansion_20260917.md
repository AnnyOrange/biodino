# User-approved staged fleet expansion

Approval: user requested 3090-qi GPUs 0/1/2/4 and multiple cpui hosts with
3–5 tests per GPU under 03_Machine_Code_Sync_Rules.md on 2026-09-17.
This amends placement/concurrency only, not the staged evaluation protocol.
The original campaign remains EXPERIMENTAL_NOT_REPORTABLE; missing tasks unchanged.

Placement: 3090-qi GPUs 0/1/2/4; cpu1/cpu2/cpu9/cpu10/cpu11/cpu15 GPU 0.
Preflight: all below 60 percent memory, with cpu1/qi2 already running one staged
worker. Other users' and existing project tasks are preserved and counted by the
GPU admission gate. No deepcad GPU used. Shared checkpoint/data paths only.

Initial target: 3 total concurrent tests per GPU, hard maximum 5. Existing
workers count toward target. Each added worker runs one child test at a time.
At >=60 percent memory, add at most one managed test with >=6.5GB free.
Dense segmentation requires no other compute app or managed reservation and
>=18GB free. No reduced batch/resolution/layers to avoid OOM.

One added worker per card prioritizes newest checkpoints, other workers backfill.
All use existing atomic shared lane claims. Jobs use 2 data workers and BLAS
threads 1. Admission checks GPU memory/apps before each test. Hosts/commands
and managed adapter SHA256 are in fleet_expansion_manifest.json. Existing
campaign manifest and protocol hashes remain unchanged; no baseline reuse.

#!/usr/bin/env python3
"""Build cheap weight-space baselines from the three 5TB EMA teachers (E/M/L).

theta_alpha = (1 - alpha) * theta_M + alpha * theta_L   for alpha in {0.25, 0.5, 0.75}
theta_avg3  = (theta_E + theta_M + theta_L) / 3

alpha = 0 and alpha = 1 are byte-identical to M and L; they are not re-saved,
their v4 results are reused from the coexistence campaign by checkpoint SHA256.
Every tensor (backbone and unused DINO/iBOT heads) is merged in float64 and
stored as float32 in the same {"teacher": OrderedDict} layout as the sources.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch

COEX = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921')
ROOT = Path('/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_weightspace_baseline_v4_20260922')
ARMS = {
    'WA025': {'kind': 'lerp_M_L', 'alpha': 0.25},
    'WA050': {'kind': 'lerp_M_L', 'alpha': 0.5},
    'WA075': {'kind': 'lerp_M_L', 'alpha': 0.75},
    'AVG3': {'kind': 'mean_E_M_L'},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_teacher(path: Path) -> OrderedDict:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or set(ck) != {'teacher'} or not isinstance(ck['teacher'], dict):
        raise ValueError(f'Expected a {{"teacher": state_dict}} checkpoint: {path} keys={list(ck)[:5]}')
    state = OrderedDict(ck['teacher'])
    if not all(torch.is_tensor(v) and v.is_floating_point() for v in state.values()):
        raise ValueError(f'Non-float tensors are not handled: {path}')
    return state


def flat_backbone(state: OrderedDict) -> torch.Tensor:
    return torch.cat([v.reshape(-1).double() for k, v in state.items() if k.startswith('backbone.')])


def main() -> int:
    manifest_in = json.loads((COEX / 'launch_manifest.json').read_text())
    teachers = manifest_in['teachers']
    sources = {}
    for role in 'EML':
        path = Path(teachers[role]['checkpoint'])
        digest = sha256(path)
        if digest != teachers[role]['sha256'] or path.stat().st_size != teachers[role]['bytes']:
            raise ValueError(f'Teacher {role} does not match the coexistence launch manifest: {path}')
        sources[role] = {'checkpoint': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
        print(f'[verified] {role} {path} sha256={digest}', flush=True)
    states = {role: load_teacher(Path(sources[role]['checkpoint'])) for role in 'EML'}
    keys = list(states['E'])
    for role in 'ML':
        if list(states[role]) != keys:
            raise ValueError(f'Key order differs between E and {role}')
        for k in keys:
            if states[role][k].shape != states['E'][k].shape or states[role][k].dtype != states['E'][k].dtype:
                raise ValueError(f'Shape/dtype mismatch for {k} in {role}')
    n_backbone = sum(1 for k in keys if k.startswith('backbone.'))
    print(f'[loaded] {len(keys)} tensors ({n_backbone} backbone); dtype={states["E"][keys[0]].dtype}', flush=True)

    # Parameter-space geometry of the three anchors (backbone only), for the report.
    fe, fm, fl = (flat_backbone(states[r]) for r in 'EML')
    def dist(a, b):
        return float(torch.linalg.vector_norm(a - b))
    geometry = {
        'backbone_numel': int(fe.numel()),
        'norm_E': float(torch.linalg.vector_norm(fe)), 'norm_M': float(torch.linalg.vector_norm(fm)),
        'norm_L': float(torch.linalg.vector_norm(fl)),
        'dist_E_M': dist(fe, fm), 'dist_M_L': dist(fm, fl), 'dist_E_L': dist(fe, fl),
        'cos_EM_ML': float(torch.dot(fm - fe, fl - fm) / (torch.linalg.vector_norm(fm - fe) * torch.linalg.vector_norm(fl - fm))),
        'cos_ME_ML': float(torch.dot(fe - fm, fl - fm) / (torch.linalg.vector_norm(fe - fm) * torch.linalg.vector_norm(fl - fm))),
    }
    del fe, fm, fl

    outputs = {}
    for arm, spec in ARMS.items():
        out_dir = ROOT / 'checkpoints' / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / 'teacher_checkpoint.pth'
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite {out}')
        merged = OrderedDict()
        for k in keys:
            e, m, l = (states[r][k].double() for r in 'EML')
            if spec['kind'] == 'lerp_M_L':
                a = spec['alpha']
                v = (1.0 - a) * m + a * l
            else:
                v = (e + m + l) / 3.0
            merged[k] = v.to(states['E'][k].dtype).contiguous()
        part = out.with_suffix('.pth.part')
        torch.save({'teacher': merged}, part)
        part.replace(out)
        # Independent reload + formula verification on every tensor.
        reloaded = load_teacher(out)
        max_err = 0.0
        for k in keys:
            e, m, l = (states[r][k].double() for r in 'EML')
            ref = ((1 - spec['alpha']) * m + spec['alpha'] * l) if spec['kind'] == 'lerp_M_L' else (e + m + l) / 3
            err = float((reloaded[k].double() - ref).abs().max())
            max_err = max(max_err, err)
            if not math.isfinite(err) or err > 1e-5:
                raise ValueError(f'{arm}: tensor {k} deviates from formula by {err}')
        fa = flat_backbone(reloaded)
        outputs[arm] = {
            **spec, 'checkpoint': str(out), 'sha256': sha256(out), 'bytes': out.stat().st_size,
            'n_tensors': len(reloaded), 'max_abs_formula_error_fp32_roundtrip': max_err,
            'dist_to_L': dist(fa, flat_backbone(states['L'])), 'dist_to_M': dist(fa, flat_backbone(states['M'])),
            'dist_to_E': dist(fa, flat_backbone(states['E'])),
        }
        del merged, reloaded, fa
        print(f'[built] {arm} {out} sha256={outputs[arm]["sha256"]} max_err={max_err:.3e}', flush=True)

    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'cheap weight-space (parameter-averaging) baseline for the 5TB E/M/L coexistence diagnostic',
        'definitions': {
            'WA_alpha': 'theta = (1-alpha)*theta_M + alpha*theta_L, alpha in {0, 0.25, 0.5, 0.75, 1}',
            'AVG3': 'theta = (theta_E + theta_M + theta_L)/3',
            'alpha_0': 'identical to M (ck20007); results reused from the coexistence campaign by SHA256',
            'alpha_1': 'identical to L (ck29279); results reused from the coexistence campaign by SHA256',
            'arithmetic': 'float64 merge of every stored tensor, cast back to float32; heads merged for format parity only',
        },
        'sources': sources, 'source_manifest': str(COEX / 'launch_manifest.json'),
        'train_config': manifest_in['train_config'],
        'backbone_geometry': geometry, 'outputs': outputs,
        'torch': torch.__version__, 'python': sys.version,
    }
    (ROOT / 'checkpoints' / 'checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(geometry, indent=1), flush=True)
    print(f'[done] {ROOT / "checkpoints" / "checkpoint_manifest.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

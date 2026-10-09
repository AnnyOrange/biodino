"""Export saved EMA teacher states for the user-requested early full-v4 evaluation."""
import hashlib,json,time
from pathlib import Path
import torch
import run_deepcad_method_20260927 as r

def main():
    torch.set_num_threads(2)
    plan=r.REPO/'outputs/00_reports/deepcad_method_20260927/ck_recovery/EARLY_V4_244_PLAN.json'
    arms=['ck_c_e12687_formal','ck_k_e12687_formal']
    r.atomic(plan,dict(time=r.now(),checkpoint=12931,updates_since_E12687=244,arms=arms,
        reason='User requested actual >50% GPU4 memory occupancy and evaluation of new experiments; prior pending task queue empty. Add one paired early diagnostic checkpoint.',
        checkpoint_selection='Same saved step244 for BOTH methods, fixed before seeing downstream scores',
        training_end=15127,training_configuration_changed=False,protocol='Full v4, all56 cells; supplementary diagnostic, not test-guided tuning'))
    for arm in arms:
        out=r.ROOT/arm;src=out/'ckpt/12931/checkpoint.pth';dest=out/'eval/training_12931/teacher_checkpoint.pth'
        assert src.is_file() and not dest.exists()
        state=torch.load(src,map_location='cpu',mmap=True,weights_only=False)
        assert int(state['iteration'])==12931
        teacher={k.removeprefix('teacher.'):v for k,v in state['model'].items() if k.startswith('teacher.')}
        assert len([k for k in teacher if k.startswith('backbone.')])>300
        assert all(torch.isfinite(v).all().item() for v in teacher.values() if v.is_floating_point())
        dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');torch.save({'teacher':teacher},tmp)
        check=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False)['teacher']
        assert check.keys()==teacher.keys() and all(torch.equal(v,check[k]) for k,v in teacher.items())
        tmp.replace(dest)
        r.atomic(dest.parent/'EXPORT_PROVENANCE.json',dict(time=r.now(),source=str(src),source_bytes=src.stat().st_size,
            extraction='model.teacher.* from consolidated full checkpoint; SSL model_ema aliases teacher',tensor_count=len(teacher),verified_exact_roundtrip=True,extra_early_evaluation=True))
        print(arm,'EXPORTED',len(teacher),'tensors',dest.stat().st_size,'bytes',flush=True)
        del state,teacher,check

if __name__=='__main__':main()

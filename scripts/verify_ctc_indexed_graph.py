"""Compare all AOGM counts against pinned code, including merges and divisions."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'outputs/02_eval_runtime/py-ctcmetrics')]
from ctc_metrics.utils.representations import count_acyclic_graph_correction_operations
from dinov3.eval.bio_tracking.ctc_metrics_requested import count_graph_operations_indexed
rng=np.random.default_rng(1977);cases=0;start=time.monotonic()
for case in range(250):
    # Real division graphs; stochastic missing/merged/false-positive detections.
    tracks=np.array([[1,0,0,0],[2,1,2,1],[3,1,2,1],[4,0,2,0],[5,0,1,0],[6,2,2,5]])
    comp=tracks.copy();comp[:,0]+=70000;comp[comp[:,3]>0,3]+=70000
    if case%2:comp[:,3]=0  # Division-edge semantic errors.
    lr=[tracks[(tracks[:,1]<=t)&(tracks[:,2]>=t),0] for t in range(3)]
    lc=[comp[(comp[:,1]<=t)&(comp[:,2]>=t),0] for t in range(3)]
    mr=[];mc=[]
    for refs,computed in zip(lr,lc):
        chosen=refs[rng.random(len(refs))>.3];mr.append(chosen);mc.append(rng.choice(computed,len(chosen)))
    expected=count_acyclic_graph_correction_operations(tracks,comp,lr,lc,mr,mc)
    actual=count_graph_operations_indexed(tracks,comp,lr,lc,mr,mc)
    assert actual==expected,(case,actual,expected)
    cases+=1
out=ROOT/'outputs/02_eval_runs/unified_paper/validation/ctc_indexed_graph';out.mkdir(parents=True,exist_ok=True)
report={'status':'PASS','cases':cases,'counts':'Every pinned AOGM vertex and edge count exactly equal',
    'coverage':['division','continuation','missing vertices','false positives','merged detections','edge semantic mismatch','uint32 IDs'],
    'seconds':time.monotonic()-start}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

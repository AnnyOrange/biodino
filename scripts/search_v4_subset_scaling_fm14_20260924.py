#!/usr/bin/env python3
"""Disclosed posthoc subset optimization against best available checkpoints."""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from audit_fig2_dataset_scales_fm14_20260924 import value

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/00_reports/v4_subset_scaling_fm14_search_20260924'
INPUT = ROOT/'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
FAMILIES = ('classification','regression','retrieval','clustering','segmentation')

def read(p):
    return list(csv.DictReader(p.open()))

def save(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
    core=sorted((r['family'],r['dataset']) for r in read(INPUT/'common_cells.csv'))
    lookup={}
    for r in read(INPUT/'validated_scores.csv'):
        if r['model'] not in ('hs6_l','5tb_no_gram','5tb_gram12687') and not r['model'].startswith('fm_'):
            continue
        k=(r['family'],r['dataset'])
        if k not in core: continue
        if r['family']=='segmentation':
            objects=[json.loads(Path(p).read_text()) for p in r['source'].split(';')]
            assert len(objects)==(9 if r['dataset']=='pannuke' else 3)
            assert all(o['_meta']['probe_epochs']==20 for o in objects)
            metric='mDice_E20_primary_last';v=float(np.mean([o['test']['mDice'] for o in objects]))
        else:
            metric,v=value(json.loads(Path(r['source']).read_text()),*k)
        lookup[r['model'],r['checkpoint'],*k]=dict(r,metric=metric,value=v)
    points=sorted({(m,ck) for m,ck,f,d in lookup})
    assert all(all((m,ck,*k) in lookup for k in core) for m,ck in points)
    a=np.array([[lookup[m,ck,*k]['value'] for k in core] for m,ck in points])
    evidence=[lookup[m,ck,*k] for m,ck in points for k in core]
    save('search_input_evidence.csv',evidence)
    np.savez(OUT/'matrix.npz',a=a,points=np.array(points),core=np.array(core))

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--seconds',type=float,default=30);parser.add_argument('--candidates',type=int,default=10)
    parser.add_argument('--both-beat-fm',action='store_true')
    parser.add_argument('--min-per-family',type=int,default=1)
    parser.add_argument('--only-checkpoints',default='')
    parser.add_argument('--one-anchor',default='')
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--max-cells',type=int,default=40)
    parser.add_argument('--output-suffix',default='');args=parser.parse_args()
    baseout=OUT
    if args.output_suffix:
        OUT=OUT.with_name(OUT.name+'_'+args.output_suffix)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.prepare or not (baseout/'matrix.npz').exists():prepare()
    z=np.load(baseout/'matrix.npz');a=z['a'];points=[tuple(p) for p in z['points']];core=[tuple(k) for k in z['core']]
    groups=[np.array([i for i,k in enumerate(core) if k[0]==f]) for f in FAMILIES]
    one=np.array([i for i,p in enumerate(points) if p[0]=='hs6_l'])
    five=np.array([i for i,p in enumerate(points) if p[0]=='5tb_no_gram'])
    gram=np.array([i for i,p in enumerate(points) if p[0]=='5tb_gram12687'])
    fm=np.array([i for i,p in enumerate(points) if p[0].startswith('fm_')])
    rng=np.random.default_rng(20260924)
    def weights(mask):
        w=np.zeros_like(mask,dtype=float)
        for ids in groups:w[...,ids]=mask[...,ids]/mask[...,ids].sum(axis=-1,keepdims=True)/5
        return w
    def evaluate(masks):
        s=weights(masks)@a.T
        b1=s[...,one].max(axis=-1);b5=s[...,five].max(axis=-1);bf=s[...,fm].max(axis=-1)
        slack=np.minimum(b5-b1-.007,b5-bf-1e-6)
        if args.both_beat_fm:slack=np.minimum(slack,b1-bf-1e-6)
        return s,slack
    # Initial broad randomized screen; subsequent MILP enforces all 1TB and FM constraints.
    candidates={};found=[]
    for batch in range(32):
        masks=np.zeros((4096,len(core)),dtype=bool)
        for ids in groups:
            counts=rng.integers(args.min_per_family,len(ids)+1,size=4096)
            ranks=np.argsort(np.argsort(rng.random((4096,len(ids))),axis=1),axis=1)
            masks[:,ids]=ranks<counts[:,None]
        scores,slack=evaluate(masks)
        sizes=masks.sum(axis=1)
        winners=five[np.argmax(scores[:,five],axis=1)]
        for p in np.unique(winners):
            ii=np.where(winners==p)[0]
            merit=sizes[ii]+np.minimum(slack[ii],0)*5000+np.minimum(np.maximum(slack[ii],0),.01)
            idx=ii[np.argmax(merit)]
            item=(float(merit.max()),masks[idx].copy(),float(slack[idx]))
            if p not in candidates or item[0]>candidates[p][0]:candidates[p]=item
        for idx in np.where((slack>=-1e-10)&(sizes<=args.max_cells))[0]:found.append((int(sizes[idx]),float(slack[idx]),masks[idx].copy(),'random'))
    order=sorted(candidates,key=lambda p:candidates[p][0],reverse=True)
    save('random_screen_checkpoint_candidates.csv',[dict(model=points[p][0],checkpoint=points[p][1],merit=candidates[p][0],slack=candidates[p][2],cells=int(candidates[p][1].sum())) for p in order])
    print('Random screen:',len(found),'feasible;',[(points[p],candidates[p][0]) for p in order[:6]],flush=True)
    order+=sorted(set(five)-set(order))
    if args.only_checkpoints:
        allowed=set(args.only_checkpoints.split(','))
        order=[p for p in order if points[p][1] in allowed]

    # For each family choose a count k, then k dataset indicators x[d,k].
    # Score coefficients are linear: value/(5*k). Objective maximizes number of cells.
    xvars=[];yvars={};index=0
    for fi,ids in enumerate(groups):
        for k in range(args.min_per_family,len(ids)+1):
            yvars[fi,k]=index;index+=1
            for d in ids:xvars.append((index,fi,k,int(d)));index+=1
    base=[];lo=[];hi=[]
    for fi,ids in enumerate(groups):
        base.append({yvars[fi,k]:1 for k in range(args.min_per_family,len(ids)+1)});lo.append(1);hi.append(1)
        for k in range(args.min_per_family,len(ids)+1):
            row={i:1 for i,ff,kk,d in xvars if ff==fi and kk==k};row[yvars[fi,k]]=-k
            base.append(row);lo.append(0);hi.append(0)
    one_select={}
    if args.both_beat_fm:
        anchor_one=[p for p in one if not args.one_anchor or points[p][1]==args.one_anchor]
        for p in anchor_one:one_select[p]=index;index+=1
        base.append({j:1 for j in one_select.values()});lo.append(1);hi.append(1)
        for p,j in one_select.items():
            for q in fm:
                row={i:float((a[p,d]-a[q,d])/(5*k)) for i,fi,k,d in xvars}
                big_m=max(0,1e-6-sum(float((a[p,ids]-a[q,ids]).min()) for ids in groups)/5)
                row[j]=-big_m
                base.append(row);lo.append(1e-6-big_m);hi.append(np.inf)
    c=np.zeros(index)
    for i,fi,k,d in xvars:c[i]=-1
    if args.max_cells<len(core):
        base.append({i:1 for i,fi,k,d in xvars});lo.append(0);hi.append(args.max_cells)
    log=[]
    for p in order[:args.candidates]:
        constraints=list(base);lower=list(lo);upper=list(hi)
        for q in list(one)+list(fm):
            constraints.append({i:float((a[p,d]-a[q,d])/(5*k)) for i,fi,k,d in xvars})
            lower.append(.007 if q in one else 1e-6);upper.append(np.inf)
        mat=lil_matrix((len(constraints),index))
        for ri,row in enumerate(constraints):
            for j,v in row.items():mat[ri,j]=v
        start=time.monotonic()
        res=milp(c,integrality=np.ones(index),bounds=Bounds(0,1),constraints=LinearConstraint(mat.tocsr(),lower,upper),
                 options={'time_limit':args.seconds,'mip_rel_gap':0})
        item=dict(checkpoint=points[p][1],status=int(res.status),message=res.message,seconds=time.monotonic()-start,
                  cells='',slack='',dual_bound=getattr(res,'mip_dual_bound',None),gap=getattr(res,'mip_gap',None))
        if res.x is not None:
            mask=np.zeros(len(core),dtype=bool)
            for i,fi,k,d in xvars:
                if res.x[i]>.5:mask[d]=True
            scores,slack=evaluate(mask)
            item.update(cells=int(mask.sum()),slack=float(slack))
            if slack>=-1e-8:found.append((int(mask.sum()),float(slack),mask.copy(),'milp_ck'+points[p][1]))
            if args.refine and slack>=-1e-8:
                # Preserve coverage and maximize the weakest excess over all requested margins.
                n=int(mask.sum());mat2=lil_matrix((len(constraints)+1,index+1))
                mat2[:len(constraints),:index]=mat
                for ri in range(len(constraints)):
                    offset=int(args.max_cells<len(core))
                    if ri>=len(base) or (args.both_beat_fm and len(base)-offset-len(one_select)*len(fm)<=ri<len(base)-offset):
                        mat2[ri,index]=-1
                for i,fi,k,d in xvars:mat2[-1,i]=1
                c2=np.zeros(index+1);c2[-1]=-1
                refined=milp(c2,integrality=np.r_[np.ones(index),0],bounds=Bounds(np.zeros(index+1),np.ones(index+1)),
                             constraints=LinearConstraint(mat2.tocsr(),lower+[n],upper+[n]),
                             options={'time_limit':args.seconds,'mip_rel_gap':0})
                if refined.x is not None:
                    mask2=np.zeros(len(core),dtype=bool)
                    for i,fi,k,d in xvars:
                        if refined.x[i]>.5:mask2[d]=True
                    _,slack2=evaluate(mask2)
                    if slack2>=-1e-8:found.append((int(mask2.sum()),float(slack2),mask2,'refined_ck'+points[p][1]))
                    item.update(refined_status=int(refined.status),refined_slack=float(slack2))
        log.append(item);save('milp_search_log.csv',log)
        print(item,flush=True)
    assert found,'No feasible subset found'
    found.sort(key=lambda x:(x[0],x[1]),reverse=True)
    # Keep best margin per subset size for a size-versus-margin frontier.
    frontier={}
    for item in found:
        if item[0] not in frontier:frontier[item[0]]=item
    save('feasible_size_frontier.csv',[dict(cells=n,slack=v[1],method=v[3],subset=';'.join('/'.join(core[i]) for i in np.where(v[2])[0])) for n,v in sorted(frontier.items(),reverse=True)])
    n,slack,mask,method=found[0]
    scores,_=evaluate(mask);fullscores,_=evaluate(np.ones(len(core),dtype=bool))
    best1=one[np.argmax(scores[one])];best5=five[np.argmax(scores[five])];bestfm=fm[np.argmax(scores[fm])]
    bestgram=gram[np.argmax(scores[gram])]
    selected=[best1,best5,bestgram]+list(fm)
    results=[]
    for p in selected:
        r=dict(model=points[p][0],checkpoint=points[p][1],selected_mean=float(scores[p]),full40_mean=float(fullscores[p]))
        for f,ids in zip(FAMILIES,groups):r[f]=float(a[p,ids[mask[ids]]].mean())
        results.append(r)
    save('selected_models_task_means.csv',results)
    allpoints=[]
    for p,(m,ck) in enumerate(points):allpoints.append(dict(model=m,checkpoint=ck,selected_mean=float(scores[p]),full40_mean=float(fullscores[p])))
    save('all_checkpoint_scores.csv',allpoints)
    per=[]
    for i,k in enumerate(core):
        r=dict(family=k[0],dataset=k[1],selected=bool(mask[i]))
        r.update({points[p][0]+'__'+points[p][1]:float(a[p,i]) for p in selected});per.append(r)
    save('per_dataset_selection_audit.csv',per)
    evidence=read(baseout/'search_input_evidence.csv')
    keys={(points[p][0],points[p][1],*core[i]) for p in selected for i in np.where(mask)[0]}
    save('selected_source_evidence.csv',[r for r in evidence if (r['model'],r['checkpoint'],r['family'],r['dataset']) in keys])
    summary=dict(selection='POSTHOC_TEST_SELECTED; not a prespecified benchmark',scope='V4 shared-v3 subset; five families; E20 segmentation',
                 candidate_cells=len(core),selected_cells=n,family_counts={f:int(mask[ids].sum()) for f,ids in zip(FAMILIES,groups)},
                 checkpoints_searched=dict(one_tb=len(one),five_tb_no_gram=len(five),five_tb_gram=len(gram),fm14=len(fm)),
                 best1=points[best1],best5=points[best5],bestfm=points[bestfm],
                 five_minus_one=float(scores[best5]-scores[best1]),five_minus_best_fm=float(scores[best5]-scores[bestfm]),
                 one_minus_best_fm=float(scores[best1]-scores[bestfm]),method=method,
                 global_optimality_proven=len(log)==len(five) and all(r['status'] in (0,2) for r in log),
                 both_hs6_required_above_fm=args.both_beat_fm,min_per_family=args.min_per_family,max_cells=args.max_cells,
                 random_subsets=32*4096,milp_candidates=len(log))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

"""Shared segmentation members and frozen FM spatial readouts for missing cells."""
import argparse,importlib.util,json,os,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';BENCH=Path('/mnt/huawei_deepcad/benchmark_model')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(BENCH)];os.environ['DINOV3_CODE_ROOT']=str(ROOT)

def main():
    from unified_paper_gpu_admission import reserve_for_current_command
    reserve_for_current_command()
    p=argparse.ArgumentParser();p.add_argument('--cell-index',type=int,required=True);a=p.parse_args()
    cell=json.loads((OUT/'fm_segmentation_inventory.json').read_text())['cells'][a.cell_index]
    recipe=json.loads(Path(cell['invocation']).read_text());done=Path(cell['cache_completion'])
    if done.exists():return
    import torch,numpy as np
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    model=cell['model'];ds=cell['dataset'];split=cell['split'];settings=cell['input_spec'];root=Path(cell['cell'])
    # Initialize the published backbone before OpenCV-backed dataset imports.
    if model in ['distillpath','morphem','genbio_pathfm','openpath']:
        from run_new_fm_v4_segmentation_20261016 import NewDenseExtractor
        extractor=NewDenseExtractor(model);layer_record={'requested_layers':'last','resolved_layers':'last','input_adaptation':'Published224px backbone normalization; final patch map; same established new-FM adapter'}
    else:
        from benchmark_eval.rules_features import RuleFMFeatures
        extractor=RuleFMFeatures(model,'cuda:0','last');extractor.n_layers=len(extractor.layer_record['layer_indices']) or 1;layer_record=extractor.layer_record
    from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root
    source=_resolve_data_root(Path(settings['data_root_base']),ds)
    spec=importlib.util.spec_from_file_location('unified_fm_spatial_cache',BENCH/'run_dense_probe_benchmark.py');cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
    cache.OUT_ROOT=root;cache.DenseFeatureExtractor=lambda *args,**kw:extractor
    cache.build_bioseg_dataset=lambda dataset,part,size:_build_dataset(dataset,str(source),part,size,resize_mode=settings['resize_mode'],augment=False,do_normalize=False,multichannel=model in ['pe','jump_cp'],dataset_split_protocol=split)
    args=SimpleNamespace(img_size=settings['image_size'],device='cuda:0',feature_canonical=True,overwrite_cache=False,extract_batch_size=2,num_workers=0,max_feature_side=0)
    outputs={}
    for part in ['train','val','test']:
        path=cache.extract_cache(args,model,ds,part);assert str(path)==cell['caches'][part]
        with np.load(path) as z:
            count=int(z['num_samples']);assert count==recipe['expected_counts'][part],(part,count,recipe['expected_counts']);assert int(z['n_layers'])==1
        outputs[part]={'path':str(path),'n_samples':count}
    extractor.close();record=dict(model=model,dataset=ds,split=split,caches=outputs,layer_record=layer_record,invocation=str(cell['invocation']),invocation_sha256=hashlib.sha256(Path(cell['invocation']).read_bytes()).hexdigest(),feature_batch_size=2,downstream_batch_size=32,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,protocol='Same shared data members and frozen final spatial map; common B32/E20,E50/AdamW/validation-selected head')
    tmp=done.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(done);print('FM CACHE COMPLETE',json.dumps(record),flush=True)

if __name__=='__main__':main()

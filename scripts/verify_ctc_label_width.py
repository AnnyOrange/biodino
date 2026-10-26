"""Meaningful serialization check: exact same graph at uint16 and >65535 uint32 IDs."""
from pathlib import Path
import sys,json
import numpy as np,tifffile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_tracking.ctc_metrics_requested import score_requested_metrics
out=ROOT/'outputs/02_eval_runs/unified_paper/validation/ctc_label_width';gt=out/'01_GT'
for p in [gt/'TRA',gt/'SEG',out/'uint16_RES',out/'uint32_RES']:p.mkdir(parents=True,exist_ok=True)
# Parent cell divides into two descendants, exercising both continuation and division edges.
tables=np.array([[1,0,0,0],[2,1,2,1],[3,1,2,1]])
np.savetxt(gt/'TRA/man_track.txt',tables,fmt='%d')
for frame in range(3):
 mask=np.zeros((32,32),np.uint16)
 if frame==0:mask[8:24,8:24]=1
 else:mask[8:15,8:24]=2;mask[17:24,8:24]=3
 for sub,prefix in [('TRA','man_track'),('SEG','man_seg')]:tifffile.imwrite(gt/sub/f'{prefix}{frame:03d}.tif',mask)
 tifffile.imwrite(out/'uint16_RES'/f'mask{frame:03d}.tif',mask)
 wide=mask.astype(np.uint32);wide[wide>0]+=70000;tifffile.imwrite(out/'uint32_RES'/f'mask{frame:03d}.tif',wide)
np.savetxt(out/'uint16_RES/res_track.txt',tables,fmt='%d')
tables[:,0]+=70000;tables[tables[:,3]>0,3]+=70000;np.savetxt(out/'uint32_RES/res_track.txt',tables,fmt='%d')
rows={}
for kind in ['uint16','uint32']:
 rows[kind]=score_requested_metrics(out/f'{kind}_RES',gt,ROOT/'outputs/02_eval_runtime/py-ctcmetrics',threads=1)[0]
assert rows['uint16']==rows['uint32'],rows
assert rows['uint32']['Valid']==1 and rows['uint32']['TRA']==1.,rows
(out/'result.json').write_text(json.dumps({'status':'PASS','metrics':rows,'scope':'Local pinned py-ctcmetrics graph scoring; uint32 is a lossless extension, not official 16-bit submission format'},indent=2));print(rows)

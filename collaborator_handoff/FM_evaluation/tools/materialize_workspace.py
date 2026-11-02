import argparse,json,shutil,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('workspace',type=Path);x=a.parse_args()
if x.workspace.exists():raise FileExistsError('Choose a new workspace')
x.workspace.mkdir(parents=True)
for name,h in json.loads((P/'PACKAGE_SHA256.json').read_text()).items():
 if not name.startswith('checkout/'):continue
 rel=name[9:];source=ROOT/rel;target=x.workspace/rel
 if hashlib.sha256(source.read_bytes()).hexdigest()!=h:raise ValueError('Source changed: '+rel)
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
(x.workspace/'.acceptance_workspace').write_text('FM_GITHUB_20261102\n')
print('WORKSPACE READY',x.workspace)

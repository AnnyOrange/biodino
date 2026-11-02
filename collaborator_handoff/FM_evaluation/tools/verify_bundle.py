import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
rows=json.loads((P/'PACKAGE_SHA256.json').read_text());bad=[]
for name,h in rows.items():
 p=ROOT/name[9:] if name.startswith('checkout/') else P/name
 if not p.is_file():bad.append(name);continue
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
 if digest.hexdigest()!=h:bad.append(name)
print('SOURCE', 'FAIL' if bad else 'PASS',len(rows),'files')
if bad:print('\n'.join(bad))
sys.exit(bool(bad))

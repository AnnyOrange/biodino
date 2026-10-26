"""Restore explicitly recorded evaluation mounts after a verified host reboot."""
import json,subprocess,shlex,time
from pathlib import Path
from unified_paper_transport import ssh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'

def restore(host):
    config=json.loads((OUT/'mount_recovery_config.json').read_text()).get('hosts',{}).get(host)
    if not config:return dict(status='not_configured',host=host)
    code=r'''import json,pathlib,socket,subprocess,sys
c=json.loads(sys.argv[1]);assert socket.gethostname()==c['hostname'],'Unexpected host identity'
uuids=set(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split())
assert uuids==set(c['gpu_uuids']),'GPU identity changed; mount recovery refused'
actions=[]
for item in c['mounts']:
 target=pathlib.Path(item['target'])
 current=subprocess.run(['findmnt','-rn','-M',str(target),'-o','SOURCE,FSTYPE'],capture_output=True,text=True)
 if current.returncode==0:
  fields=current.stdout.strip().split();assert fields[0]==item['source']and fields[1]in ['nfs','nfs4'],'Different filesystem already mounted'
  actions.append(dict(target=str(target),status='already_mounted'));continue
 assert target.is_dir()and not target.is_symlink()and not any(target.iterdir()),'Refuse to cover a nonempty or unexpected directory'
 subprocess.run(['sudo','-n','timeout','10s','mount','-t','nfs','-o',c['options'],item['source'],str(target)],check=True,timeout=15)
 actions.append(dict(target=str(target),status='restored'))
print(json.dumps(dict(status='ready',boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),actions=actions)))
'''
    try:
        result=subprocess.run(ssh(host,5)+[shlex.join(['python3','-c',code,json.dumps(config)])],capture_output=True,text=True,timeout=55)
        record=json.loads(result.stdout)if result.returncode==0 else dict(status='failed',error=result.stderr[-3000:])
    except Exception as error:record=dict(status='failed',error=repr(error))
    record.update(host=host,observed_at_controller=time.time())
    path=OUT/'states'/f'mount_recovery_{host}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
    with(OUT/'logs/mount_recovery.jsonl').open('a')as stream:stream.write(json.dumps(record)+'\n')
    return record

if __name__=='__main__':
    import sys
    print(json.dumps(restore(sys.argv[1]),indent=2))

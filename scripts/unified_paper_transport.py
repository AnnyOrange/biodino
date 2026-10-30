"""Task-local SSH routing; reuse authorized identities without global changes."""
import json
import socket
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'outputs/02_eval_runs/unified_paper'
def local_host(host):
 return host==socket.gethostname() or (socket.gethostname()=='server' and host in {'server','5090-zxr'})
def ssh(host,timeout=8):
 path=OUT/'ssh_routes.json';routes=json.loads(path.read_text())if path.exists()else{}
 command=['ssh','-o','BatchMode=yes','-o',f'ConnectTimeout={timeout}']
 jump=routes.get(host,{}).get('jump_host')
 if jump:command+=['-J',jump]
 return command+[host]

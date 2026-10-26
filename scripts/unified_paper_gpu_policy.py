"""User-reserved devices are excluded from every campaign scheduling path."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'outputs/02_eval_runs/unified_paper'
def excluded(host,gpu):
    path=OUT/'gpu_exclusions.json'
    if not path.exists():return False
    return int(gpu)in json.loads(path.read_text()).get('hosts',{}).get(host,{}).get('gpus',[])

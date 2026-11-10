#!/usr/bin/env python3
"""Verify complete input arrivals once; never admit partial transfers."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dinov3.eval.ood_transfer.common import read_json, save_json, sha256, resolve_input, verify_inputs, process_policy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);args=p.parse_args()
    root=Path(args.campaign);cfg=read_json(root/'campaign.json');process_policy()
    with (root/'input_verify.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report={}
        for task,role in [('pxct','frozen'),('film','frozen'),('bbbc041','development'),('bbbc041','test'),
                          ('bone_age','development'),('bone_age','test'),('pneumonia','development'),('pneumonia','test'),('cryo','frozen')]:
            key=task+'_'+role;path=Path(cfg['inputs'])/(key+'.json');receipt=path.with_name(key+'.verified.json')
            if not path.exists():report[key]=dict(status='MANIFEST_PENDING');continue
            data_root=cfg.get('task_data_roots',{}).get(task,cfg['data_root'])
            if receipt.exists():
                old=read_json(receipt)
                if old.get('manifest_sha256')==sha256(path) and old.get('data_root')==str(Path(data_root).resolve()):
                    report[key]=dict(status='ALL_SOURCE_BYTES_VERIFIED');continue
            d=read_json(path);rows=d['files'] if 'files' in d else d['records'];missing=[]
            for row in rows:
                local=resolve_input(data_root,row['path'])
                if not local.is_file() or local.stat().st_size!=row['bytes']:missing.append(row['sample_id'])
            if missing:report[key]=dict(status='INPUT_TRANSFER_PENDING',missing_count=len(missing),total=len(rows));continue
            try:
                report[key]=verify_inputs(path,data_root,receipt)
            except Exception as error:
                report[key]=dict(status='VERIFICATION_FAILED',error=repr(error))
            save_json(root/'input_status.json',dict(time=time.time(),tasks=report))
        save_json(root/'input_status.json',dict(time=time.time(),tasks=report))


if __name__=='__main__':main()

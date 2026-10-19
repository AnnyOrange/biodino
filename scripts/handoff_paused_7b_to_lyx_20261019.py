#!/usr/bin/env python3
"""After the approved 7B pause, export its exact EMA teacher and evaluate on lyx."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


REPORT = Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/l20tb_domain_7b_baseline_20261018')
STATE = REPORT / 'paused_7b_lyx_handoff_state.json'
HXW = '5090-hxw-xzj'
LYX = '5090-lyx-xr'
LYX_DIRECT = 'server@101.6.89.166'
RUN = '/home/xzj/storage/merged/vit7b_20tb_hs6_nogram_b1024_from249_20261019'
REFERENCE = ('/home/xzj/storage/merged/vit7b_20tb_recovered249_20261018'
             '/eval/training_499/sharded_teacher_checkpoint')
HXW_PY = '/home/xzj/miniconda3/envs/dinov3/bin/python'
EXPORT = '/home/xzj/biodino_l20tb_domain_snapshot_20261018/scripts/export_paused_7b_teacher_20261019.py'
LYX_ROOT = '/data/xuzijing/cooldown_20261013_eval'
ARM = '7b_20tb_hs6_nogram_b1024_from249_20261019'
LYX_ARM = f'{LYX_ROOT}/{ARM}'
DISPATCH = f'{LYX_ROOT}/bin/dispatch_20tb_priority_20261016.py'
OLD_CONFIG = f'{LYX_ROOT}/online_7b_nogram_20261018/config.json'
LANES = ('classification_a', 'classification_b', 'classification_c', 'classification_d',
         'regression', 'retrieval', 'segmentation_a', 'segmentation_b',
         'segmentation_c', 'segmentation_d', 'detection')


def save(state):
    state.update(pid=os.getpid(), updated_unix=time.time())
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


def remote(host, args, timeout=120, input_text=None):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                           shlex.join(map(str, args))], input=input_text, text=True,
                          capture_output=True, check=True, timeout=timeout).stdout.strip()


def paused_step():
    try:
        raw = remote(HXW, ['cat', f'{RUN}/paused_for_l_20261019.json'])
    except subprocess.CalledProcessError:
        return None
    pause = json.loads(raw)
    if pause.get('supervisor_active') or pause.get('train_active'):
        raise RuntimeError('7B pause marker reports an active trainer')
    step = int(pause['checkpoint']['step'])
    if step < 250:
        raise RuntimeError(f'Paused checkpoint {step} precedes the corrected 7B baseline')
    recovery = json.loads(remote(HXW, ['cat', f'{RUN}/recovery_state.json']))
    if recovery.get('global_batch') != 1024:
        raise RuntimeError('Paused 7B run does not have effective batch 1024')
    return step


def export_teacher(step, state):
    source = f'{RUN}/ckpt/{step}'
    eval_dir = f'{RUN}/eval/training_{step}'
    destination = f'{eval_dir}/teacher_checkpoint.pth'
    record_path = f'{eval_dir}/teacher_checkpoint.export.json'
    try:
        record = json.loads(remote(HXW, ['cat', record_path]))
        if record['source'] != source or record['destination'] != destination:
            raise RuntimeError('Existing teacher export belongs to another source')
        actual = remote(HXW, ['sha256sum', destination], timeout=3600).split()[0]
        if actual != record['sha256']:
            raise RuntimeError('Existing hxw export digest changed')
    except subprocess.CalledProcessError:
        state['status'] = 'EXPORTING_TEACHER'
        save(state)
        raw = remote(HXW, [HXW_PY, EXPORT, source, REFERENCE,
                           destination], timeout=6 * 3600)
        record = json.loads(raw.splitlines()[-1])
    state.update(status='EXPORTED', source=source, exported=destination,
                 bytes=record['bytes'], sha256=record['sha256'])
    save(state)
    return destination, record


def transfer_teacher(step, source, record, state):
    target = f'{LYX_ARM}/eval/training_{step}/teacher_checkpoint.pth'
    partial = target + '.partial'
    remote(LYX, ['mkdir', '-p', str(Path(target).parent)])
    try:
        actual = remote(LYX, ['sha256sum', target], timeout=3600).split()[0]
        if actual == record['sha256']:
            state.update(status='TRANSFERRED', destination=target)
            save(state)
            return target
        raise RuntimeError('An existing lyx teacher differs from the paused checkpoint')
    except subprocess.CalledProcessError:
        pass
    state['status'] = 'TRANSFERRING_TO_LYX'
    save(state)
    command = ['rsync', '-a', '--partial', '--append-verify',
               '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new',
               source, f'{LYX_DIRECT}:{partial}']
    remote(HXW, command, timeout=18 * 3600)
    verify = ('import hashlib,json,os,sys; p,d,w=sys.argv[1:4]; '
              'h=hashlib.sha256(); '
              'f=open(p,"rb"); '
              '[h.update(b) for b in iter(lambda:f.read(16*1024**2),b"")]; '
              'f.close(); '
              'assert h.hexdigest()==w,(h.hexdigest(),w); '
              'os.replace(p,d); print(json.dumps({"bytes":os.path.getsize(d),"sha256":w}))')
    verified = json.loads(remote(LYX, ['python3', '-c', verify, partial, target,
                                      record['sha256']], timeout=3600))
    if verified['bytes'] != record['bytes']:
        raise RuntimeError('lyx teacher size mismatch')
    state.update(status='TRANSFERRED', destination=target,
                 transferred_unix=time.time())
    save(state)
    return target


def publish_config(step, state):
    config = json.loads(remote(LYX, ['cat', OLD_CONFIG]))
    template = next(target for target in config['targets']
                    if target['arm'] == '7b_20tb_nogram_recovered249_20261018'
                    and target['checkpoint'] == 499)
    source_config = remote(HXW, ['cat', f'{RUN}/config.yaml']) + '\n'
    source_path = f'{LYX_ARM}/source/config.yaml'
    write_code = ('import os,sys; p=sys.argv[1]; os.makedirs(os.path.dirname(p),exist_ok=True); '
                  'q=p+".tmp"; open(q,"w").write(sys.stdin.read()); os.replace(q,p)')
    remote(LYX, ['python3', '-c', write_code, source_path], input_text=source_config)
    control = f'{LYX_ROOT}/online_7b_paused_{step}_b1024_20261019'
    config.update(control_root=control, max_running_jobs=2)
    config['targets'] = [dict(template, arm=ARM, checkpoint=step,
                              snapshot_root=f'{LYX_ARM}/eval',
                              effective_global_batch=1024, full_eval_period=1,
                              config_sha256=hashlib.sha256(source_config.encode()).hexdigest())]
    path = f'{control}/config.json'
    content = json.dumps(config, indent=2) + '\n'
    digest = hashlib.sha256(content.encode()).hexdigest()
    remote(LYX, ['python3', '-c', write_code, path], input_text=content)
    state.update(status='EVAL_CONFIG_READY', lyx_config=path, config_sha256=digest)
    save(state)
    return path, control


def launch_eval(config, control, state):
    jobs = [(0, 'frozen'), (1, 'frozen'), (4, 'dense'), (5, 'dense')]
    launched = state.setdefault('launchers', {})
    for gpu, group in jobs:
        key = f'{gpu}_{group}'
        if key in launched:
            try:
                remote(LYX, ['kill', '-0', str(launched[key]['pid'])])
                continue
            except subprocess.CalledProcessError:
                launched.pop(key)
        log = f'{control}/launcher_gpu{gpu}_{group}.log'
        command = ['/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python', '-u', DISPATCH,
                   '--config', config, '--host', LYX, '--gpu', str(gpu),
                   '--group', group, '--slot', f'paused_{group}']
        shell = f'nohup setsid {shlex.join(command)} > {shlex.quote(log)} 2>&1 < /dev/null & echo $!'
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', LYX, shell],
                                capture_output=True, text=True, check=True, timeout=30)
        pid = int(result.stdout.strip())
        time.sleep(2)
        remote(LYX, ['kill', '-0', str(pid)])
        launched[key] = dict(pid=pid, log=log)
        save(state)
    state['status'] = 'EVALUATING_ON_LYX'
    save(state)


def evaluate_until_done(step, state):
    base = f'{LYX_ARM}/results/_state'
    count_code = ('import json,os,sys; b,s=sys.argv[1],sys.argv[2]; '
                  'lanes=json.loads(s); '
                  'print(json.dumps({k:next((x for x in ("done","terminal") '
                  'if os.path.isfile(os.path.join(b,x,"ckpt_%s__%s.json"%(sys.argv[3],k)))),'
                  '"pending") for k in lanes}))')
    while True:
        statuses = json.loads(remote(LYX, ['python3', '-c', count_code, base,
                                         json.dumps(LANES), str(step)]))
        state['lane_statuses'] = statuses
        state['status'] = ('EVALUATION_COMPLETE' if all(x == 'done' for x in statuses.values())
                           else 'EVALUATION_HAS_TERMINAL_FAILURES' if all(x != 'pending' for x in statuses.values())
                           else 'EVALUATING_ON_LYX')
        save(state)
        if state['status'] != 'EVALUATING_ON_LYX':
            for launcher in state.get('launchers', {}).values():
                try:
                    remote(LYX, ['kill', '-TERM', str(launcher['pid'])])
                except subprocess.CalledProcessError:
                    pass
            return
        launch_eval(state['lyx_config'], str(Path(state['lyx_config']).parent), state)
        time.sleep(120)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'paused_7b_lyx_handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text()) if STATE.exists() else {
            'authorization': 'User requested copy to lyx and test after the 7B pause on 2026-10-19',
            'status': 'WAITING_FOR_7B_PAUSE',
        }
        while True:
            try:
                step = paused_step()
                if step is None:
                    state['status'] = 'WAITING_FOR_7B_PAUSE'
                    save(state)
                    time.sleep(60)
                    continue
                if state.get('step') not in (None, step):
                    raise RuntimeError('Pause step changed after handoff began')
                state['step'] = step
                save(state)
                source, record = export_teacher(step, state)
                transfer_teacher(step, source, record, state)
                config, control = publish_config(step, state)
                launch_eval(config, control, state)
                evaluate_until_done(step, state)
                return
            except Exception as exc:
                state.update(status='RETRY', last_error=repr(exc),
                             failures=state.get('failures', 0) + 1)
                save(state)
                time.sleep(120)


if __name__ == '__main__':
    main()

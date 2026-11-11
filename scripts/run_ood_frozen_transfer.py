#!/usr/bin/env python3
"""Portable complementary Cryo/PXCT/FILM analyses; never mislabeled full FT."""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dinov3.eval.eval_ood.datasets import MRCStack
from dinov3.eval.ood_transfer.common import (
    code_identity, process_policy, read_json, require_inputs, resolve_input,
    save_json, save_torch, seed_all, sha256,
)
from dinov3.eval.ood_transfer.models import load_backbone
from dinov3.eval.ood_transfer.metrics import pxct_confidence_metrics


class Encoder:
    def __init__(self, args):
        self.body, mean, std = load_backbone(args.checkpoint, args.train_config, False)
        self.body.cuda().eval().requires_grad_(False)
        self.mean = torch.tensor(mean, device='cuda')[None, :, None, None]
        self.std = torch.tensor(std, device='cuda')[None, :, None, None]

    @torch.no_grad()
    def __call__(self, planes, spatial=False):
        pixels = torch.stack([F.interpolate(torch.as_tensor(np.array(x, dtype=np.float32, copy=True))[None, None],
                              size=(256, 256), mode='bilinear', align_corners=False)[0].expand(3, -1, -1)
                              for x in planes]).cuda()
        pixels = pixels.clamp(0, 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if spatial:
                return self.body.get_intermediate_layers((pixels-self.mean)/self.std, n=1, reshape=True)[0].float().cpu()
            value = self.body.forward_features((pixels-self.mean)/self.std)
            feature = torch.cat((value['x_norm_clstoken'], value['x_norm_patchtokens'].mean(1)), dim=1).float()
        return F.normalize(feature, dim=1).cpu().numpy()


def minmax(x):
    x = np.asarray(x, np.float32)
    return (x-x.min()) / max(float(x.max()-x.min()), 1e-12)


def cryo(args, manifest, encoder, out, image_transform=minmax):
    records, stacks = [], {}
    for project in manifest['projects']:
        particles = np.load(resolve_input(args.data_root, project['particles']), allow_pickle=False)
        quality = read_json(resolve_input(args.data_root, project['quality_scores']))
        for row in particles:
            name = row['blob/path']
            if isinstance(name, bytes):
                name = name.decode()
            path = str(Path(project['stack_root']) / Path(str(name)).name)
            records.append(dict(sample_id=f"{project['project']}:{int(row['uid'])}", project=project['project'],
                                group_id=f"{project['project']}:{int(row['location/micrograph_uid'])}", path=path,
                                index=int(row['blob/idx']), quality=float(quality[str(int(row['alignments2D/class']))])))
    if len(records) != 320000:
        raise ValueError('Cryo population differs')
    features = None
    cache = out / 'temporary_features.npy'
    for start in range(0, len(records), args.batch):
        images = []
        for row in records[start:start+args.batch]:
            path = row['path']
            if path not in stacks:
                stacks[path] = MRCStack(resolve_input(args.data_root, path))
            images.append(image_transform(stacks[path].read(row['index'])))
        value = encoder(images)
        if features is None:
            features = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32, shape=(len(records), value.shape[1]))
        features[start:start+len(value)] = value
        if start % (args.batch*100) == 0:
            print('FEATURES', start, len(records), flush=True)
    features.flush()
    reference = F.normalize(torch.from_numpy(np.asarray(features)).cuda(), dim=1)
    neighbors = []
    for start in range(0, len(records), 256):
        end = min(start+256, len(records))
        similarity = reference[start:end] @ reference.T
        similarity[torch.arange(end-start, device='cuda'), torch.arange(start, end, device='cuda')] = -torch.inf
        neighbors.extend(similarity.argmax(1).cpu().tolist())
    labels = np.asarray([r['project'] for r in records])
    quality = np.asarray([r['quality'] for r in records])
    tiers = np.where(quality < .3, 0, np.where(quality > .7, 2, 1))
    neighbors = np.asarray(neighbors)
    metrics = dict(structure_top1=float(np.mean(labels[neighbors] == labels)),
                   quality_top1=float(np.mean(tiers[neighbors] == tiers)), n=len(records),
                   by_project={p:dict(structure_top1=float(np.mean(labels[neighbors][labels==p] == p)),
                                     quality_top1=float(np.mean(tiers[neighbors][labels==p] == tiers[labels==p])))
                               for p in sorted(set(labels))})
    save_torch(out / 'predictions.pth', dict(records=records, neighbor_indices=neighbors, metrics=metrics))
    del reference, features
    cache.unlink()
    return metrics


def pxct(args, manifest, encoder, out):
    rows, features = manifest['records'], []
    for index, row in enumerate(rows):
        volume = minmax(np.load(resolve_input(args.data_root, row['path']), allow_pickle=False))
        planes = [np.take(volume, j, axis=axis) for axis in range(3) for j in range(volume.shape[axis])]
        features.append(np.concatenate([encoder(planes[j:j+args.batch]) for j in range(0, len(planes), args.batch)]).mean(0))
        if index % 20 == 0:
            print('ROI', index, len(rows), flush=True)
    x = np.asarray(features)
    truth = np.asarray([r['em_mean'] for r in rows])
    group = np.asarray([r['group_id'] for r in rows])
    prediction = np.empty(len(rows))
    folds = {}
    for g in sorted(set(group)):
        test = group == g
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x[~test], truth[~test])
        prediction[test] = model.predict(x[test])
        folds[g] = pxct_confidence_metrics(truth[test], prediction[test])
    metrics = dict(pooled=pxct_confidence_metrics(truth, prediction), folds=folds, independent_specimens=2)
    save_json(out / 'predictions.json', [dict(sample_id=r['sample_id'], group_id=r['group_id'], truth=float(y), prediction=float(p))
                                        for r, y, p in zip(rows, truth, prediction)])
    return metrics


class ReconstructionHead(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv3d(channels, 1, 3, padding=1)
        torch.nn.init.zeros_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, features, raw):
        residual = F.interpolate(self.conv(features), size=raw.shape[-3:], mode='trilinear', align_corners=False)
        return raw + residual


def film(args, manifest, encoder, out):
    pairs = np.load(resolve_input(args.data_root, manifest['pairs']), allow_pickle=False)
    x, y = pairs['x'], pairs['y']
    if x.shape != (512,16,64,64) or y.shape != x.shape:
        raise ValueError('FILM author pairing changed')
    cache = out / 'temporary_spatial_features.npy'
    bank = None
    for i in range(len(x)):
        features = torch.cat([encoder(x[i,j:j+2], spatial=True) for j in range(0,16,2)]).permute(1,0,2,3).numpy()
        if bank is None:
            bank = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float16, shape=(len(x),*features.shape))
        bank[i] = features
        if i % 32 == 0:
            print('PATCH', i, len(x), flush=True)
    bank.flush()
    rows = []
    for seed in range(5):
        seed_all(seed)
        root = out / f'seed{seed}'
        root.mkdir(exist_ok=True)
        if (root/'complete.json').exists():
            rows.append(read_json(root/'complete.json'))
            continue
        head = ReconstructionHead(bank.shape[1]).cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr=4e-5, eps=1e-7)
        generator = np.random.default_rng(seed)
        best, bad, history, start = float('inf'), 0, [], 0
        if (root/'latest.pth').exists():
            resume = torch.load(root/'latest.pth', map_location='cpu', weights_only=False)
            head.load_state_dict(resume['head']); optimizer.load_state_dict(resume['optimizer'])
            generator.bit_generator.state=resume['rng']; best=resume['best'];bad=resume['bad'];history=resume['history'];start=len(history)
        for epoch in range(start,100):
            head.train()
            for _ in range(20):
                optimizer.zero_grad(set_to_none=True)
                for i in generator.integers(0,461,size=16):
                    feature = torch.from_numpy(np.array(bank[i],np.float32,copy=True))[None].cuda()
                    raw = torch.from_numpy(x[i])[None,None].cuda()
                    target = torch.from_numpy(y[i])[None,None].cuda()
                    loss = F.l1_loss(head(feature,raw),target)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite FILM loss')
                    (loss/16).backward()
                optimizer.step()
            head.eval(); values=[]
            with torch.inference_mode():
                for i in range(461,512):
                    feature=torch.from_numpy(np.array(bank[i],np.float32,copy=True))[None].cuda()
                    raw=torch.from_numpy(x[i])[None,None].cuda(); target=torch.from_numpy(y[i])[None,None].cuda()
                    values.append(float(F.l1_loss(head(feature,raw),target)))
            val=float(np.mean(values));history.append(dict(epoch=epoch+1,validation_self_permutation_mae=val))
            if val < best:
                best=val;bad=0;save_torch(root/'best.pth',dict(head=head.state_dict(),epoch=epoch+1,mae=val))
            else:
                bad+=1
                if bad>=10:
                    for group in optimizer.param_groups:group['lr']*=.5
                    bad=0
            save_torch(root/'latest.pth',dict(head=head.state_dict(),optimizer=optimizer.state_dict(),rng=generator.bit_generator.state,
                                            best=best,bad=bad,history=history))
            save_json(root/'history.json',history)
        row=dict(seed=seed,validation_self_permutation_mae=best,independent_test=False)
        save_json(root/'complete.json',row);rows.append(row)
    del bank
    cache.unlink()
    # Full126-band reconstruction; spectral planes are never averaged before the model.
    raw = tifffile.imread(resolve_input(args.data_root, manifest['raw'])).astype(np.float32)
    if raw.shape != (126,200,200):raise ValueError('Raw FILM cube changed')
    low,high=np.percentile(raw,[2,99.8]);scale=max(float(high-low),1e-12)
    normalized=(raw-low)/scale
    heads=[]
    for seed in range(5):
        state=torch.load(out/f'seed{seed}/best.pth',map_location='cpu',weights_only=False)
        head=ReconstructionHead(encoder.body.embed_dim).cuda().eval()
        head.load_state_dict(state['head']);heads.append(head)
    sums=[np.zeros_like(raw) for _ in heads];counts=np.zeros_like(raw)
    def starts(length,window,stride):return sorted(set([*range(0,length-window+1,stride),length-window]))
    with torch.inference_mode():
        for z in starts(126,16,8):
            for y0 in starts(200,64,32):
                for x0 in starts(200,64,32):
                    sl=(slice(z,z+16),slice(y0,y0+64),slice(x0,x0+64))
                    tile=normalized[sl]
                    features=torch.cat([encoder(tile[j:j+2],spatial=True) for j in range(0,16,2)]).permute(1,0,2,3)[None].cuda()
                    tensor=torch.from_numpy(tile.copy())[None,None].cuda()
                    for total,head in zip(sums,heads):total[sl]+=head(features,tensor)[0,0].cpu().numpy()
                    counts[sl]+=1
    outputs=[]
    for seed,total in enumerate(sums):
        reconstructed=total/counts*scale+low
        path=out/f'seed{seed}/reconstruction.tif'
        tifffile.imwrite(path,reconstructed)
        save_json(out/f'seed{seed}/spectra.json',dict(raw_spatial_mean=raw.mean((1,2)).tolist(),
                  reconstructed_spatial_mean=reconstructed.mean((1,2)).tolist(),bands=126))
        outputs.append(dict(seed=seed,path=str(path),sha256=sha256(path)))
    return dict(seeds=rows,reconstructions=outputs,scope='Single-volume self-supervised validation; no clean GT or independent test')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','input-receipt','data-root','checkpoint','train-config','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--batch',type=int,default=16)
    args=parser.parse_args();process_policy();seed_all(0)
    code=code_identity()
    if code['dirty']:raise ValueError('Requires committed clean source')
    require_inputs(args.manifest,args.data_root,args.input_receipt)
    if sha256(args.checkpoint)!='f63544ca11ab72d7c3e05b5765c613ebc9d6374ba45ee5ca0c619eaa2f4cc423':raise ValueError('Wrong lw5 weight')
    if sha256(args.train_config)!='e33cb26b1eecbec0efa7a466b17fb59f16f7a77ae25aa6b4b138028aa95eabe3':raise ValueError('Wrong model config')
    manifest=read_json(args.manifest);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=sha256(args.manifest),checkpoint_sha256=sha256(args.checkpoint),code=code,
                  protocol='ood-lw5-adaptation-v1-frozen-complement',encoder_frozen=True,input_size=256)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'identity.json').exists() and read_json(out/'identity.json')!=identity:raise ValueError('Identity changed')
        save_json(out/'identity.json',identity)
        if (out/'complete.json').exists():return
        encoder=Encoder(args)
        result={'cryo':cryo,'pxct':pxct,'film':film}[manifest['task']](args,manifest,encoder,out)
        save_json(out/'complete.json',dict(identity=identity,metrics=result))


if __name__=='__main__':main()

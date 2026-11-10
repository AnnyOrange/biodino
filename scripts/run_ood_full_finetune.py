#!/usr/bin/env python3
"""Full-backbone supervised adaptation; train/val and test are separate commands."""
import argparse
import contextlib
import fcntl
import math
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from dinov3.eval.ood_transfer.common import (
    code_identity, process_policy, read_json, require_inputs, restore_rng, rng_state,
    save_json, save_torch, seed_all, seed_worker, sha256,
)
from dinov3.eval.ood_transfer.data import TransferDataset, collate
from dinov3.eval.ood_transfer.evaluate import predict, score
from dinov3.eval.ood_transfer.models import BoneAgeModel, build_model, parameter_groups


def identity(args, recipe):
    code = code_identity()
    if code['dirty'] and not args.allow_dirty:
        raise ValueError('Training/evaluation requires a clean committed code tree')
    actual_checkpoint = sha256(args.checkpoint)
    actual_config = sha256(args.train_config)
    if actual_checkpoint != recipe['checkpoint_sha256'] or actual_config != recipe['train_config_sha256']:
        raise ValueError('Pretraining checkpoint/config identity mismatch')
    return dict(protocol='ood-lw5-adaptation-v1', recipe_sha256=sha256(args.recipe), code=code,
                checkpoint_sha256=actual_checkpoint, train_config_sha256=actual_config,
                seed=args.seed, phase=args.phase, torch_version=torch.__version__,
                cuda_version=torch.version.cuda, deterministic_algorithms='warn_only',
                tf32=False, autocast='bf16', parameter_dtype='float32')


def batch_tensor(images):
    # Optional multiscale bone training remains full-field even within a batch.
    height = max(x.shape[-2] for x in images)
    width = max(x.shape[-1] for x in images)
    return torch.stack([torch.nn.functional.pad(x, (0, width-x.shape[-1], 0, height-x.shape[-2])) for x in images])


def fit(args):
    recipe = read_json(args.recipe)
    if args.phase == 'formal':
        if args.seed not in range(5) or not args.recipe_lock:
            raise ValueError('Formal training needs seeds0..4 and a validation-only recipe lock')
        lock = read_json(args.recipe_lock)
        if lock['recipe_sha256'] != sha256(args.recipe) or lock.get('test_used') is not False:
            raise ValueError('Recipe lock mismatch')
    elif args.seed not in (101, 102):
        raise ValueError('Development seeds are101,102')
    require_inputs(args.development_manifest, args.data_root, args.input_receipt)
    train = TransferDataset(args.development_manifest, args.data_root, 'train', recipe, augment=True)
    val = TransferDataset(args.development_manifest, args.data_root, 'val', recipe)
    if recipe['effective_batch'] % args.microbatch:
        raise ValueError('Microbatch must divide effective batch')
    if recipe['task'] == 'bone_age':
        labels = np.asarray([r['age_months'] for r in train.records])
        recipe.update(target_mean=float(labels.mean()), target_std=float(labels.std()))
        if recipe['target_std'] <= 0:
            raise ValueError('Degenerate training labels')
    ident = identity(args, recipe)
    ident.update(development_manifest_sha256=sha256(args.development_manifest),
                 source_manifest_sha256=read_json(args.development_manifest)['source_manifest_sha256'], microbatch=args.microbatch,
                 loader_workers=args.workers, runtime_recipe=recipe)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        if read_json(out / 'complete.json')['identity'] != ident:
            raise ValueError('Completed run has another identity')
        return
    seed_all(args.seed)
    model = build_model(recipe, args.checkpoint, args.train_config).cuda()
    optimizer = torch.optim.AdamW(parameter_groups(model, recipe), betas=(.9, .999), eps=1e-8)
    epochs = recipe['epochs']
    steps_per_epoch = math.ceil(len(train) / recipe['effective_batch'])
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, round(total_steps * recipe['warmup_fraction']))
    history, best, update, start = [], -float('inf'), 0, 0
    if (out / 'latest.pth').exists():
        state = torch.load(out / 'latest.pth', map_location='cpu', weights_only=False)
        if state['identity'] != ident:
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.cuda()
        history, best, update, start = state['history'], state['best_score'], state['update'], state['epoch']
        restore_rng(state['rng'])
        del state
    save_json(out / 'identity.json', ident)
    save_json(out / 'runtime_recipe.json', recipe)
    for epoch in range(start, epochs):
        if shutil.disk_usage(out).free < args.min_disk_free_gib * 1024**3 + 12 * 1024**3:
            raise RuntimeError('Disk reserve reached before training epoch')
        started = time.time()
        generator = torch.Generator().manual_seed(args.seed + 1000003 * epoch)
        loader = DataLoader(train, batch_size=args.microbatch, shuffle=True, generator=generator,
                            num_workers=args.workers, worker_init_fn=seed_worker, collate_fn=collate,
                            pin_memory=True, persistent_workers=False)
        model.train()
        if not isinstance(model, BoneAgeModel):
            model.transform.min_size = tuple(recipe.get('train_sizes', [recipe['resolution']]))
        optimizer.zero_grad(set_to_none=True)
        samples, window_count, loss_sum = 0, 0, 0.0
        window_size = min(recipe['effective_batch'], len(train))
        for images, targets in loader:
            batch = len(images)
            images = [x.cuda(non_blocking=True) for x in images]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if recipe['task'] == 'bone_age':
                    sex = torch.stack([t['sex'] for t in targets])
                    age = torch.stack([t['age'] for t in targets])
                    target = (age - recipe['target_mean']) / recipe['target_std']
                    loss = torch.nn.functional.l1_loss(model(batch_tensor(images), sex).float(), target)
                else:
                    loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            (loss * batch / window_size).backward()
            loss_sum += float(loss.detach()) * batch
            window_count += batch
            samples += batch
            if window_count == window_size:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip'], error_if_nonfinite=True)
                if update == 0:
                    encoder = model.encoder if isinstance(model, BoneAgeModel) else model.backbone.encoder
                    gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
                    nonzero = sum(bool(torch.count_nonzero(g)) for g in gradients)
                    if nonzero == 0:
                        raise ValueError('Backbone received no nonzero gradients')
                    save_json(out / 'first_optimizer_step.json', dict(
                        encoder_frozen=False, trainable_encoder_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                        encoder_tensors_with_nonzero_gradient=nonzero, loss=float(loss.detach()), gradient_norm=float(grad_norm),
                        allocated_gpu_bytes=torch.cuda.max_memory_allocated(), actual_input_shapes=[list(x.shape) for x in images],
                        seed=args.seed, test_used=False))
                if update < warmup_steps:
                    multiplier = (update + 1) / warmup_steps
                else:
                    multiplier = .5 * (1 + math.cos(math.pi * (update-warmup_steps) / max(1, total_steps-warmup_steps)))
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * multiplier
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                window_count = 0
                window_size = min(recipe['effective_batch'], len(train)-samples)
                if update % 20 == 0:
                    save_json(out / 'progress.json', dict(epoch=epoch+1, max_epochs=epochs, updates=update,
                              samples_this_epoch=samples, total_train_samples=len(train), loss=loss_sum/samples,
                              peak_gpu_bytes=torch.cuda.max_memory_allocated(), time=time.time(), test_used=False))
                    print('TRAIN', epoch+1, update, samples, loss_sum/samples, flush=True)
        if window_count or samples != len(train):
            raise ValueError('Incomplete accumulation window/population')
        validation_predictions = predict(model, val, batch_size=args.microbatch, workers=args.workers)
        validation = score(val, validation_predictions)
        row = dict(epoch=epoch+1, loss=loss_sum/samples, validation=validation,
                   seconds=time.time()-started, updates=update, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(row)
        if validation['selection_score'] > best:
            best = validation['selection_score']
            save_torch(out / 'best.pth', dict(model=model.state_dict(), identity=ident, recipe=recipe,
                       epoch=epoch+1, validation=validation))
            save_json(out / 'best_validation.json', dict(epoch=epoch+1, validation=validation,
                      identity=ident, test_used=False))
        save_torch(out / 'latest.pth', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   identity=ident, history=history, epoch=epoch+1, best_score=best, update=update, rng=rng_state()))
        save_json(out / 'history.json', history)
        print('EPOCH', row, flush=True)
    save_json(out / 'complete.json', dict(identity=ident, status='TRAIN_VALIDATION_COMPLETE',
               best=read_json(out / 'best_validation.json'), test_used=False,
               best_checkpoint_sha256=sha256(out / 'best.pth')))


def test(args):
    directory = Path(args.trained_run)
    complete = read_json(directory / 'complete.json')
    if complete['identity']['phase'] != 'formal':
        raise ValueError('Development runs cannot be evaluated on test')
    current_code = code_identity()
    if current_code['dirty'] or current_code['commit'] != complete['identity']['code']['commit']:
        raise ValueError('Test must use the identical committed code checkout')
    if sha256(directory / 'best.pth') != complete['best_checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    require_inputs(args.test_manifest, args.data_root, args.input_receipt)
    if read_json(args.test_manifest)['source_manifest_sha256'] != complete['identity']['source_manifest_sha256']:
        raise ValueError('Test and development came from different canonical task manifests')
    state = torch.load(directory / 'best.pth', map_location='cpu', weights_only=False)
    recipe = state['recipe']
    # Validate the pretraining identity too; constructor only reads that checkpoint.
    if sha256(args.checkpoint) != recipe['checkpoint_sha256'] or sha256(args.train_config) != recipe['train_config_sha256']:
        raise ValueError('Initial model/config differs')
    model = build_model(recipe, args.checkpoint, args.train_config).cuda()
    model.load_state_dict(state['model'], strict=True)
    dataset = TransferDataset(args.test_manifest, args.data_root, 'test', recipe)
    predictions = predict(model, dataset, batch_size=args.microbatch, workers=args.workers)
    metrics = score(dataset, predictions, threshold=state['validation'].get('score_threshold'))
    save_torch(directory / 'test_predictions.pth', dict(records=dataset.records, predictions=predictions,
               classes=dataset.classes, test_manifest_sha256=sha256(args.test_manifest)))
    save_json(directory / 'test_result.json', dict(identity=state['identity'], metrics=metrics,
               selected_epoch=state['epoch'], test_manifest_sha256=sha256(args.test_manifest),
               best_checkpoint_sha256=complete['best_checkpoint_sha256'],
               predictions_sha256=sha256(directory / 'test_predictions.pth'), test_tuning=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--recipe', required=True)
    train.add_argument('--phase', choices=['development', 'formal'], required=True)
    train.add_argument('--seed', type=int, required=True)
    train.add_argument('--development-manifest', required=True)
    train.add_argument('--recipe-lock')
    train.add_argument('--output', required=True)
    train.add_argument('--allow-dirty', action='store_true')
    train.add_argument('--min-disk-free-gib', type=int, default=120)
    evaluate = sub.add_parser('test')
    evaluate.add_argument('--trained-run', required=True)
    evaluate.add_argument('--test-manifest', required=True)
    for item in (train, evaluate):
        item.add_argument('--checkpoint', required=True)
        item.add_argument('--train-config', required=True)
        item.add_argument('--data-root', required=True)
        item.add_argument('--input-receipt', required=True)
        item.add_argument('--microbatch', type=int, default=1)
        item.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    process_policy()
    output = Path(args.output if args.command == 'train' else args.trained_run)
    output.mkdir(parents=True, exist_ok=True)
    with (output / (args.command + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            (fit if args.command == 'train' else test)(args)
        except Exception as error:
            save_json(output / (args.command + '_failure.json'), dict(error=repr(error), time=time.time()))
            raise


if __name__ == '__main__':
    main()

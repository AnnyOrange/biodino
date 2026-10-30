"""Shared native CTC orchestration, relocated without changing the evaluation flow."""
import re,hashlib,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import run_ctc_native_full_hs6 as native
from dinov3.eval.bio_tracking.ctc_native import sha256
SHARED=Path(__file__).resolve().parents[1]
CTC=SHARED/'outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json'
# The external adapter is installed by the unified runner before seed reset.
FMSpatialBackbone=None

def verify_resident_inputs():
    if str(SHARED)=='/mnt/huawei_deepcad/dinov3':return
    out=SHARED/'outputs/02_eval_runs/unified_paper'
    proof=json.loads((out/'validation/ai_ctc_inputs_complete.json').read_text())
    canonical=out/'validation/ctc_canonical_manifest.json'
    assert proof['status']=='PASS' and proof['domains']==20
    assert hashlib.sha256(canonical.read_bytes()).hexdigest()==proof['canonical_manifest_sha256']
    assert hashlib.sha256(CTC.read_bytes()).hexdigest()==proof['resident_manifest_sha256']
    from prepare_unified_ctc_resident_data import relocate,signature
    canonical_data=json.loads(canonical.read_text())
    assert json.loads(CTC.read_text())==relocate(canonical_data)
    expected_files={relocate(r['path']):r['sha256'] for r in canonical_data['files']}
    checked=set()
    for path in proof['domain_proofs']:
        domain=json.loads(Path(path).read_text())
        for source,expected in domain['source_fingerprints'].items():
            assert expected_files[source]==expected and source not in checked
            p=Path(source)
            if signature(p)!=domain['source_stats'][source]:
                with p.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,source
            checked.add(source)
    assert checked==set(expected_files) and len(checked)==proof['files']

def ctcmetrics_commit() -> str:
    git_dir = native.PY_CTCMETRICS / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text().strip()
        if not marker.startswith("gitdir: "):
            raise RuntimeError(f"Invalid py-ctcmetrics gitdir marker: {marker!r}")
        git_dir = (git_dir.parent / marker[8:]).resolve()
    head = (git_dir / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            commit = ref_path.read_text().strip()
        else:
            commit = ""
            for line in (git_dir / "packed-refs").read_text().splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[1] == ref:
                    commit = fields[0]
                    break
    else:
        commit = head
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"Invalid py-ctcmetrics Git ref: {commit!r}")
    return commit

def native_ctc_modules():
    import run_ctc_native_full_hs6 as native
    base = native.base
    native.PY_CTCMETRICS = SHARED / 'outputs/02_eval_runtime/py-ctcmetrics'
    base.PY_CTCMETRICS = native.PY_CTCMETRICS
    base.IMAGECODECS_VENDOR = SHARED / 'outputs/02_eval_runtime/imagecodecs_py311_wheel_2026.3.6'
    base.IMAGECODECS_WHEEL = SHARED / 'outputs/02_eval_runtime/imagecodecs_wheel/imagecodecs-2026.3.6-cp311-abi3-manylinux_2_28_x86_64.whl'
    if base.ctcmetrics_commit() != base.PINNED_CTCMETRICS_COMMIT:
        raise ValueError('Unpinned official CTC evaluator')
    base.verify_imagecodecs_vendor()
    # Epoch-dependent samples are copied into fresh workers on each iterator.
    original_loader = DataLoader
    def loader(*args, **kwargs):
        kwargs['num_workers'] = 2
        return original_loader(*args, **kwargs)
    base.DataLoader = loader
    return native

def run_ctc(invocation, output):
    native = native_ctc_modules()
    from dinov3.eval.bio_segmentation.instance_seg.model import DINOHoVerNet, build_dino_hovernet
    from dinov3.eval.bio_tracking.ctc_native import load_cache_manifest
    native.base.set_determinism()
    if invocation['asset_kind'] == 'external':
        backbone = FMSpatialBackbone(invocation['checkpoint']['model_id'])
        model = DINOHoVerNet(backbone, layers=[0], freeze_backbone=True,
            num_types=0, feature_size=32, embed_proj=384).cuda()
        # Inference-mode feature tensors must be normal tensors before decoder training.
        model._extract = lambda images: [backbone.get_intermediate_layers(images, n=1, reshape=True)[0].clone()]
    else:
        from omegaconf import OmegaConf
        config = OmegaConf.load(invocation['checkpoint']['config_path'])
        layer = {'vit_small': 11, 'vit_base': 11, 'vit_large': 23, 'vit_huge2': 31}[config.student.arch]
        model = build_dino_hovernet(checkpoint=invocation['checkpoint']['path'],
            train_config=invocation['checkpoint']['config_path'], layers=[layer],
            num_types=0, freeze_backbone=True, feature_size=32, embed_proj=384, device=torch.device('cuda:0'))
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.decoder.state_dict().items()}
    data = load_cache_manifest(CTC)
    fold = invocation['dataset']['fold']
    candidate = dict(model=output.name, checkpoint=invocation['checkpoint'])
    manifest_sha = sha256(output / 'invocation_manifest.json')
    head = native.base.train_or_load_head(model, initial, data, candidate, fold, output,
        manifest_sha, torch.device('cuda:0'), {})
    records = native.training_records(data, fold)
    diameter, n_diameter = native.training_diameter(records, {})
    from unified_paper_gpu_admission import release_current_reservation
    test_domains = invocation['dataset']['test_domains']
    domains = [native.evaluate_domain(model, data, name, fold, output, diameter, n_diameter,
               manifest_sha, head['head_sha256'], torch.device('cuda:0'),
               after_inference=release_current_reservation if index == len(test_domains) - 1 else None)
               for index, name in enumerate(test_domains)]
    return dict(protocol_id=native.PROTOCOL_ID, fold=fold, head=head,
                domain_rows=domains, macro=native._macro(domains), n_domains=len(domains),
                feature_grid_policy='published-final-map-to-common-stride16-decoder-grid',
                extra_dice_definition='foreground Dice (legacy field instance_mDice); not matched-instance Dice')

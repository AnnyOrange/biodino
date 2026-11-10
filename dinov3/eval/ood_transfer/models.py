"""Strict checkpoint loading and trainable heads; no frozen feature wrappers."""
from collections import OrderedDict
import functools

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dinov3.configs import get_default_config
from dinov3.models import build_model_from_cfg


def load_backbone(path, config_path, activation_checkpointing=True):
    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(config_path))
    cfg.student.fp8_enabled = False
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    for key in ('teacher', 'model', 'state_dict'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    for prefix in ('teacher.backbone.', 'backbone.'):
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            break
    # Fail on every missing/unexpected key, including LayerScale parameters.
    model.load_state_dict(state, strict=True, assign=True)
    model = model.float()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if activation_checkpointing:
        for block in model.blocks:
            original = block.forward

            @functools.wraps(original)
            def forward(*args, _original=original, _block=block, **kwargs):
                if _block.training and torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)

            block.forward = forward
    return model, list(cfg.crops.rgb_mean), list(cfg.crops.rgb_std)


class BoneAgeModel(nn.Module):
    def __init__(self, backbone, mean, std, use_sex=False):
        super().__init__()
        self.encoder = backbone
        self.use_sex = use_sex
        self.register_buffer('image_mean', torch.tensor(mean)[None, :, None, None])
        self.register_buffer('image_std', torch.tensor(std)[None, :, None, None])
        self.head = nn.Sequential(nn.Linear(backbone.embed_dim * 2 + int(use_sex), 512),
                                  nn.GELU(), nn.Linear(512, 1))

    def forward(self, images, sex):
        features = self.encoder.forward_features((images - self.image_mean) / self.image_std)
        representation = torch.cat((features['x_norm_clstoken'], features['x_norm_patchtokens'].mean(1)), dim=1)
        if self.use_sex:
            representation = torch.cat((representation, sex[:, None]), dim=1)
        return self.head(representation).flatten()


class SimplePyramid(nn.Module):
    """Learned stride4/8/16/32 maps from the ViT's final stride16 map."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if encoder.patch_size != 16:
            raise ValueError('This pyramid recipe requires patch size16')
        channels = 256
        self.out_channels = channels
        self.projection = nn.Conv2d(encoder.embed_dim, channels, 1)
        self.up4 = nn.Sequential(nn.ConvTranspose2d(channels, channels, 2, 2), nn.GroupNorm(32, channels),
                                 nn.GELU(), nn.ConvTranspose2d(channels, channels, 2, 2))
        self.up2 = nn.ConvTranspose2d(channels, channels, 2, 2)
        self.refine = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(32, channels))
            for _ in range(4)])

    def forward(self, images):
        feature = self.encoder.get_intermediate_layers(images, n=1, reshape=True)[0]
        feature = self.projection(feature)
        maps = [self.up4(feature), self.up2(feature), feature, nn.functional.max_pool2d(feature, 2)]
        result = OrderedDict((str(i), layer(value)) for i, (layer, value) in enumerate(zip(self.refine, maps)))
        result['pool'] = nn.functional.max_pool2d(result['3'], 2)
        return result


def build_model(recipe, checkpoint_path, config_path):
    backbone, mean, std = load_backbone(checkpoint_path, config_path, recipe.get('activation_checkpointing', True))
    if recipe['task'] == 'bone_age':
        return BoneAgeModel(backbone, mean, std, recipe.get('use_sex', False))
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    sizes = tuple(recipe.get('train_sizes', [recipe['resolution']]))
    return FasterRCNN(
        SimplePyramid(backbone), num_classes=recipe['num_classes'] + 1,
        min_size=sizes, max_size=recipe['max_size'], image_mean=mean, image_std=std,
        rpn_anchor_generator=AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((.5, 1., 2.),) * 5),
        box_roi_pool=MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
        box_detections_per_img=300, box_score_thresh=.01, box_nms_thresh=.5,
        rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
    )


def parameter_groups(model, recipe):
    layers = len(model.encoder.blocks if isinstance(model, BoneAgeModel) else model.backbone.encoder.blocks)
    prefix = 'encoder.' if isinstance(model, BoneAgeModel) else 'backbone.encoder.'
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError('Unexpected frozen parameter in full fine-tuning: ' + name)
        encoder = name.startswith(prefix)
        if encoder:
            local = name[len(prefix):]
            level = int(local.split('.')[1]) + 1 if local.startswith('blocks.') else (layers + 1 if 'norm' in local else 0)
            multiplier = recipe['layer_decay'] ** (layers + 1 - level)
        else:
            multiplier = recipe['head_lr_multiplier']
        decay = 0.0 if parameter.ndim <= 1 or name.endswith('.bias') else recipe['weight_decay']
        key = (multiplier, decay)
        group = groups.setdefault(key, dict(params=[], lr=recipe['encoder_lr'] * multiplier,
                                            initial_lr=recipe['encoder_lr'] * multiplier, weight_decay=decay))
        group['params'].append(parameter)
    return list(groups.values())

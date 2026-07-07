# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from dinov3.layers import (
    DualRouteStem,
    LayerScale,
    Mlp,
    PatchEmbed,
    PatchEmbedPerChannel,
    ResidualMultiChannelStem,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, (PatchEmbed, PatchEmbedPerChannel, DualRouteStem, ResidualMultiChannelStem)):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        enable_channelvit: bool = False,
        stem_type: str | None = None,
        residual_mc_extra_scale_init: float = 1e-3,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.enable_channelvit = enable_channelvit
        self.stem_type = stem_type
        self.in_chans = in_chans

        # Branch logic: dual-route stem vs ChannelViT vs standard DINOv3
        if self.stem_type in ("residual_mc", "rgb_extra_residual", "residual_mc_v2", "rgb_extra_residual_v2"):
            # Conservative multi-channel stem: keep official RGB PatchEmbed for
            # channels 0/1/2 and add a tiny residual from channels 3+.
            rgb_fill_mode = (
                "repeat_low" if self.stem_type in ("residual_mc_v2", "rgb_extra_residual_v2") else "zero"
            )
            self.patch_embed = ResidualMultiChannelStem(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                extra_scale_init=residual_mc_extra_scale_init,
                rgb_fill_mode=rgb_fill_mode,
            )
            self.channel_embed = None
            logger.info(
                "Residual multi-channel stem enabled (RGB base + extra residual, fill=%s, extra_scale_init=%s)",
                rgb_fill_mode,
                residual_mc_extra_scale_init,
            )
        elif self.stem_type == "dualroute":
            # #1 dual-route stem: RGB Conv2d (joint <=3ch) || channel-adaptive
            # pooling (independent multichannel). Returns RGB-shaped tokens, so
            # the rest of the ViT is unchanged. channel_embed (ChannelViT vocab)
            # is not used here; identity comes from content (#3) inside the stem.
            self.patch_embed = DualRouteStem(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )
            self.channel_embed = None
            logger.info("Dual-route stem enabled (RGB Conv2d || content-pool)")
        elif self.enable_channelvit:
            # ChannelViT mode: use PatchEmbedPerChannel
            self.patch_embed = PatchEmbedPerChannel(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                flatten_embedding=False,  # We handle flattening in prepare_tokens_with_masks
            )
            # Initialize Channel Embedding (ChannelViT specific).
            # RGB callers do not need to pass channel ids; they default to
            # [0, 1, 2].  Multi-channel callers can pass explicit channel ids
            # and gather from this table without changing the RGB data path.
            self.channel_embed = nn.Parameter(torch.empty(1, in_chans, embed_dim, device=device))
            torch.nn.init.trunc_normal_(self.channel_embed, std=0.02)
            logger.info(f"ChannelViT enabled with {in_chans} channels")
        else:
            # Standard DINOv3 mode (default)
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                flatten_embedding=False,
            )
            self.channel_embed = None

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        if self.enable_channelvit and self.channel_embed is not None:
            nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def _channel_embeddings(
        self,
        *,
        channel_ids: Tensor | None,
        batch_size: int,
        n_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.channel_embed is None:
            raise RuntimeError("Channel embeddings requested when ChannelViT is disabled")

        if channel_ids is None:
            channel_ids = torch.arange(n_channels, device=device, dtype=torch.long)
        else:
            channel_ids = channel_ids.to(device=device, dtype=torch.long)

        if channel_ids.numel() == 0:
            raise ValueError("channel_ids must not be empty")

        if channel_ids.ndim == 1:
            if channel_ids.shape[0] != n_channels:
                raise ValueError(
                    f"channel_ids length ({channel_ids.shape[0]}) must match input channels ({n_channels})"
                )
            embed = self.channel_embed[:, channel_ids, :].to(dtype=dtype)
            return embed.unsqueeze(2).unsqueeze(3)

        if channel_ids.ndim == 2:
            if channel_ids.shape != (batch_size, n_channels):
                raise ValueError(
                    "batched channel_ids must have shape "
                    f"({batch_size}, {n_channels}), got {tuple(channel_ids.shape)}"
                )
            embed = self.channel_embed[0, channel_ids, :].to(dtype=dtype)
            return embed.unsqueeze(2).unsqueeze(3)

        raise ValueError(f"channel_ids must be 1D or 2D, got shape={tuple(channel_ids.shape)}")

    def prepare_tokens_with_masks(
        self,
        x: Tensor,
        masks=None,
        channel_ids: Tensor | None = None,
        channel_valid_mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tuple[int], Tensor | None]:
        B, C, H, W = x.shape
        
        # Patch embedding
        if self.stem_type in ("dualroute", "residual_mc", "rgb_extra_residual", "residual_mc_v2", "rgb_extra_residual_v2"):
            # Multi-channel stems consume channel metadata; they return
            # RGB-shaped (B, H', W', D), so the standard (else) branch below
            # applies unchanged (token count == H'*W', no channel explosion).
            x = self.patch_embed(x, channel_ids=channel_ids, channel_valid_mask=channel_valid_mask)
        else:
            x = self.patch_embed(x)

        if self.enable_channelvit:
            # === ChannelViT logic ===
            # x from PatchEmbedPerChannel: (B, embed_dim, C, H', W')
            # Permute to (B, C, H', W', embed_dim) for adding channel embedding
            x = x.permute(0, 2, 3, 4, 1)  # (B, C, H', W', embed_dim)
            
            # Add Channel Embedding
            # self.channel_embed: gather to (1 or B, C, 1, 1, embed_dim)
            chan_embed = self._channel_embeddings(
                channel_ids=channel_ids,
                batch_size=B,
                n_channels=C,
                device=x.device,
                dtype=x.dtype,
            )
            x = x + chan_embed  # Broadcast addition
            
            # Flatten: (B, C, H', W', embed_dim) -> (B, C*H'*W', embed_dim)
            x = x.flatten(1, 3)
            
            # Record spatial dimensions for RoPE (single channel size)
            H_patch, W_patch = H // self.patch_size, W // self.patch_size
            if channel_valid_mask is None:
                channel_valid_mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
            else:
                channel_valid_mask = channel_valid_mask.to(device=x.device, dtype=torch.bool)
                if channel_valid_mask.shape != (B, C):
                    raise ValueError(
                        f"channel_valid_mask must have shape {(B, C)}, got {tuple(channel_valid_mask.shape)}"
                    )
            patch_valid_mask = (
                channel_valid_mask.unsqueeze(-1)
                .expand(-1, -1, H_patch * W_patch)
                .reshape(B, C * H_patch * W_patch)
            )
        else:
            # === Standard DINOv3 logic ===
            # x: (B, H', W', embed_dim)
            H_patch, W_patch = x.shape[1], x.shape[2]
            x = x.flatten(1, 2)  # (B, H'*W', embed_dim)
            patch_valid_mask = None
        if masks is not None:
            if self.enable_channelvit and masks.shape[1] != x.shape[1]:
                # masks: (B, H'*W') → (B, C*H'*W') to match ChannelViT token count
                masks = masks.unsqueeze(1).expand(-1, C, -1).reshape(B, -1)
            if self.enable_channelvit and patch_valid_mask is not None:
                masks = masks & patch_valid_mask
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        token_valid_mask = None
        if patch_valid_mask is not None:
            extra_valid = torch.ones(
                B,
                self.n_storage_tokens + 1,
                dtype=torch.bool,
                device=x.device,
            )
            token_valid_mask = torch.cat([extra_valid, patch_valid_mask], dim=1)
            x = x.masked_fill(~token_valid_mask.unsqueeze(-1), 0)

        return x, (H_patch, W_patch), token_valid_mask

    def forward_features_list(
        self,
        x_list: List[Tensor],
        masks_list: List[Tensor],
        channel_ids_list: List[Tensor | None] | None = None,
        channel_valid_masks_list: List[Tensor | None] | None = None,
    ) -> List[Dict[str, Tensor]]:
        if channel_ids_list is None:
            channel_ids_list = [None for _ in x_list]
        if channel_valid_masks_list is None:
            channel_valid_masks_list = [None for _ in x_list]
        x = []
        rope = []
        token_valid_masks = []
        for t_x, t_masks, t_channel_ids, t_channel_valid_mask in zip(
            x_list,
            masks_list,
            channel_ids_list,
            channel_valid_masks_list,
        ):
            t2_x, hw_tuple, token_valid_mask = self.prepare_tokens_with_masks(
                t_x,
                t_masks,
                channel_ids=t_channel_ids,
                channel_valid_mask=t_channel_valid_mask,
            )
            x.append(t2_x)
            rope.append(hw_tuple)
            token_valid_masks.append(token_valid_mask)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                # Generate base RoPE for single channel spatial dimensions
                rope_sincos_list = [self.rope_embed(H=H, W=W) for H, W in rope]
                
                if self.enable_channelvit:
                    # === ChannelViT RoPE adaptation ===
                    # We need to repeat RoPE C times to match (C*H*W) token sequence
                    new_rope_sincos_list = []
                    for (cos, sin), t_x_tensor in zip(rope_sincos_list, x):
                        # Calculate number of channels
                        # Total tokens (excluding CLS and storage tokens) / spatial tokens
                        n_extra = 1 + self.n_storage_tokens  # CLS + Registers
                        total_tokens = t_x_tensor.shape[1] - n_extra
                        spatial_tokens = cos.shape[0]  # H*W
                        num_channels = total_tokens // spatial_tokens
                        
                        # Repeat RoPE: [Pos1, Pos2... PosN] -> [Pos1...PosN, Pos1...PosN, ...]
                        # Corresponding to flatten order: [Chan1_Spatial, Chan2_Spatial, ...]
                        cos_rep = torch.cat([cos] * num_channels, dim=0)
                        sin_rep = torch.cat([sin] * num_channels, dim=0)
                        new_rope_sincos_list.append((cos_rep, sin_rep))
                    rope_sincos = new_rope_sincos_list
                else:
                    # === Standard DINOv3 RoPE ===
                    rope_sincos = rope_sincos_list
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos, token_valid_masks)
            x = [
                t_x.masked_fill(~token_valid_mask.unsqueeze(-1), 0)
                if token_valid_mask is not None
                else t_x
                for t_x, token_valid_mask in zip(x, token_valid_masks)
            ]
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self,
        x: Tensor | List[Tensor],
        masks: Optional[Tensor] = None,
        channel_ids: Optional[Tensor | List[Tensor | None]] = None,
        channel_valid_mask: Optional[Tensor | List[Tensor | None]] = None,
    ) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks], [channel_ids], [channel_valid_mask])[0]
        else:
            if masks is None:
                masks = [None for _ in x]
            if channel_ids is None:
                channel_ids = [None for _ in x]
            if channel_valid_mask is None:
                channel_valid_mask = [None for _ in x]
            return self.forward_features_list(x, masks, channel_ids, channel_valid_mask)

    def _get_intermediate_layers_not_chunked(
        self,
        x: Tensor,
        n: int = 1,
        channel_ids: Tensor | None = None,
        channel_valid_mask: Tensor | None = None,
    ) -> List[Tensor]:
        x, (H_patch, W_patch), token_valid_mask = self.prepare_tokens_with_masks(
            x,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
        )
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        
        # H_patch and W_patch are already patch dimensions from prepare_tokens_with_masks
        
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos_base = self.rope_embed(H=H_patch, W=W_patch)
                
                if self.enable_channelvit:
                    # Repeat RoPE for ChannelViT
                    n_extra = 1 + self.n_storage_tokens
                    total_tokens = x.shape[1] - n_extra
                    spatial_tokens = rope_sincos_base[0].shape[0]
                    num_channels = total_tokens // spatial_tokens
                    
                    cos_rep = torch.cat([rope_sincos_base[0]] * num_channels, dim=0)
                    sin_rep = torch.cat([rope_sincos_base[1]] * num_channels, dim=0)
                    rope_sincos = (cos_rep, sin_rep)
                else:
                    rope_sincos = rope_sincos_base
            else:
                rope_sincos = None
            x = blk(x, rope_sincos, token_valid_mask)
            if token_valid_mask is not None:
                x = x.masked_fill(~token_valid_mask.unsqueeze(-1), 0)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
        channel_ids: Tensor | None = None,
        channel_valid_mask: Tensor | None = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(
            x,
            n,
            channel_ids=channel_ids,
            channel_valid_mask=channel_valid_mask,
        )
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, C, h, w = x.shape
            h_patch = h // self.patch_size
            w_patch = w // self.patch_size
            if self.enable_channelvit:
                # ChannelViT patch tokens are ordered as C x H x W.  Collapse
                # channels so dense evaluators receive one feature per patch.
                if channel_valid_mask is not None:
                    valid = channel_valid_mask.to(device=x.device, dtype=torch.bool)
                    if valid.shape != (B, C):
                        raise ValueError(f"channel_valid_mask must have shape {(B, C)}, got {tuple(valid.shape)}")
                    valid = valid[:, :, None, None, None].to(dtype=outputs[0].dtype)
                    denom = valid.sum(dim=1).clamp_min(1)
                    outputs = [
                        ((out.reshape(B, C, h_patch, w_patch, -1) * valid).sum(dim=1) / denom)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        for out in outputs
                    ]
                else:
                    outputs = [
                        out.reshape(B, C, h_patch, w_patch, -1).mean(dim=1).permute(0, 3, 1, 2).contiguous()
                        for out in outputs
                    ]
            else:
                outputs = [
                    out.reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2).contiguous()
                    for out in outputs
                ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    ffn_ratio = kwargs.pop("ffn_ratio", 4.0)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    ffn_ratio = kwargs.pop("ffn_ratio", 4.0)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    ffn_ratio = kwargs.pop("ffn_ratio", 4.0)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    ffn_ratio = kwargs.pop("ffn_ratio", 3.777777778)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    ffn_ratio = kwargs.pop("ffn_ratio", 4.0)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    ffn_ratio = kwargs.pop("ffn_ratio", 4.0)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    ffn_ratio = kwargs.pop("ffn_ratio", 3.0)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=ffn_ratio,
        **kwargs,
    )
    return model

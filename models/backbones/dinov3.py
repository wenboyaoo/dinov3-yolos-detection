# Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import compile_compatible_method_lru_cache
from transformers.utils import logging, is_torch_npu_available, is_torch_xpu_available
from transformers.utils.import_utils import is_torch_greater_or_equal
from transformers.utils.generic import check_model_inputs
from transformers.configuration_utils import PretrainedConfig

class DINOv3ViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DINOv3Model`]. It is used to instantiate an
    DINOv3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DINOv3
    [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        rope_theta (`float`, *optional*, defaults to 100.0):
            The base period of the RoPE embeddings.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        query_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the query projection.
        key_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the key projection.
        value_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the value projection.
        proj_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the output projection.
        mlp_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the MLP layers.
        layerscale_value (`float`, *optional*, defaults to 1.0):
            Initial value to use for layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_gated_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
        num_register_tokens (`int`, *optional*, defaults to 0):
            The number of register tokens.
        pos_embed_shift (`float`, *optional*):
            Amount to randomly shift position embedding coordinates in [-shift, shift],
            applied only in training mode if not `None`.
        pos_embed_jitter (`float`, *optional*):
            Amount to randomly jitter position embedding coordinates in log-uniform value in [1/jitter, jitter],
            applied only in training mode if not `None`.
        pos_embed_rescale (`float`, *optional*, defaults to 2.0):
            Amount to randomly rescale position embedding coordinates in log-uniform value in [1/rescale, rescale],
            applied only in training mode if not `None`.

    Example:

    ```python
    >>> from transformers import DINOv3ViTConfig, DINOv3ViTModel

    >>> # Initializing a DINOv3 ViT-small style configuration
    >>> config = DINOv3ViTConfig()

    >>> # Initializing a model (with random weights) from the config
    >>> model = DINOv3ViTModel(config)

    >>> # Accessing the model config
    >>> config = model.config
    ```"""

    model_type = "dinov3_vit"

    def __init__(
        self,
        patch_size: int = 16,
        hidden_size: int = 384,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        hidden_act: str = "gelu",
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 100.0,
        image_size: int = 224,
        num_channels: int = 3,
        query_bias: bool = True,
        key_bias: bool = False,
        value_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        layerscale_value: float = 1.0,
        drop_path_rate: float = 0.0,
        use_gated_mlp: bool = False,
        num_register_tokens: int = 0,
        # train augs
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = None,
        # experiments
        num_det_tokens: int = 100,
        det_token_gate: Optional[str] = None,
        enable_det_rope: bool = False,
        det_rope_type: Optional[str] = 'fixed',
        token_proj_dim: Optional[int] = 32,
        attn_interp_dim: Optional[int] = 50,
        attn_proj_dim: Optional[int] = 32,
        det_mlp_intermediate_size: Optional[int] = 64,
        max_stride: Optional[int] = 0.2,
        det_tem_power_scale: Optional[float] = 2,
        enable_det_additive_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_gated_mlp = use_gated_mlp
        self.rope_theta = rope_theta
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.proj_bias = proj_bias
        self.mlp_bias = mlp_bias
        self.num_register_tokens = num_register_tokens

        # train augs
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale

        # experiments
        self.num_det_tokens = num_det_tokens
        self.det_token_gate = det_token_gate
        self.enable_det_rope = enable_det_rope
        self.det_rope_type = det_rope_type
        self.attn_interp_dim = attn_interp_dim
        self.token_proj_dim = token_proj_dim
        self.attn_proj_dim = attn_proj_dim
        self.det_mlp_intermediate_size = det_mlp_intermediate_size
        self.max_stride = max_stride
        self.det_tem_power_scale = det_tem_power_scale
        self.enable_det_additive_embeddings = enable_det_additive_embeddings

class DINOv3ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.det_tokens = nn.Parameter(torch.empty(1, config.num_det_tokens, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype

        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patch_embeddings.dtype)
            patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        det_tokens = self.det_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_token, register_tokens, patch_embeddings, det_tokens), dim=1)

        return embeddings


@compile_compatible_method_lru_cache(maxsize=32)
def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Computes the 2D coordinates of the centers of image patches, normalized to the range [-1, +1].
    The center of each patch is exactly halfway between its top-left and bottom-right corners.

    Args:
        num_patches_h (int): Number of patches along the vertical (height) axis.
        num_patches_w (int): Number of patches along the horizontal (width) axis.
        dtype (torch.dtype): The desired data type of the returned tensor.

    Returns:
        torch.Tensor: A tensor of shape (height * width, 2), where each row contains the (y, x)
            coordinates of a patch center, normalized to [-1, +1].
    """
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords

def get_fixed_det_coordinates(
    num_h: int, num_w: int
) -> torch.Tensor:
    coords_h = torch.arange(0.5, num_h)
    coords_w = torch.arange(0.5, num_w)
    coords_h = coords_h / num_h
    coords_w = coords_w / num_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0
    return coords


def augment_patches_center_coordinates(
    coords: torch.Tensor,
    shift: Optional[float] = None,
    jitter: Optional[float] = None,
    rescale: Optional[float] = None,
) -> torch.Tensor:
    # Shift coords by adding a uniform value in [-shift, shift]
    if shift is not None:
        shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        shift_hw = shift_hw.uniform_(-shift, shift)
        coords = coords + shift_hw

    # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
    if jitter is not None:
        jitter_range = np.log(jitter)
        jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_hw

    # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
    if rescale is not None:
        rescale_range = np.log(rescale)
        rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype)
        rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_hw

    return coords


class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()

        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_patches_h = config.image_size // config.patch_size
        self.num_patches_w = config.image_size // config.patch_size

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, info) -> tuple[torch.Tensor, torch.Tensor]:
        num_patches_h, num_patches_w, device, dtype = (
            info["num_patches_h"],
            info["num_patches_w"],
            info["device"],
            info["dtype"],
        )
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # Although we could precompute static patch_coords from image_size and patch_size in the config,
            # the model was trained with random_scale, so it can process images of varying sizes.
            # Therefore, it's better to compute patch_coords dynamically (with lru_cache).
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=torch.float32, device=device
            )

            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DetRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, coords: torch.Tensor, info: dict) -> tuple[torch.Tensor, torch.Tensor]:
        device, dtype = info['device'], info['dtype']
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        # Accept coords as (N, 2) or (B, N, 2)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.dim() != 3 or coords.shape[-1] != 2:
            raise ValueError(f"coords must be (N,2) or (B,N,2), got {tuple(coords.shape)}")

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # Although we could precompute static patch_coords from image_size and patch_size in the config,
            # the model was trained with random_scale, so it can process images of varying sizes.
            # Therefore, it's better to compute patch_coords dynamically (with lru_cache).

            # coords: (B, N, 2) -> (B, N, 2, 1) to broadcast against inv_freq: (1, 1, 1, F)
            angles = 2 * math.pi * coords[:, :, :, None] * self.inv_freq[None, None, None, :]
            # (B, N, 2, F) -> (B, N, 2*F)
            angles = angles.flatten(2, 3)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        # (B, N, head_dim) -> (B, 1, N, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DetRopeUpdater(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.token_proj_dim = config.token_proj_dim
        self.attn_interp_dim = config.attn_interp_dim
        self.attn_proj_dim = config.attn_proj_dim
        self.intermediate_size = config.det_mlp_intermediate_size
        self.max_stride = config.max_stride
        self.feature_dim = 1 + self.attn_proj_dim + self.token_proj_dim + 1 + 1

        self.token_proj = nn.Linear(self.hidden_size, self.token_proj_dim)
        self.attn_proj = nn.Linear(self.attn_interp_dim, self.attn_proj_dim)
        self.gate_proj = nn.Linear(self.feature_dim, self.intermediate_size)
        self.up_proj = nn.Linear(self.feature_dim, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, 2)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, marginal_attn: tuple[torch.Tensor, torch.Tensor], det_tokens: torch.Tensor, rope_params: torch.Tensor) -> torch.Tensor:
        attn_h, attn_w = marginal_attn
        batch_size, num_det = attn_h.shape[0], attn_h.shape[1]
        if det_tokens.shape[0] != batch_size or det_tokens.shape[1] != num_det:
            raise ValueError(
                f"Shape mismatch: attn_h is [B={batch_size},N={num_det},H], det_tokens is {tuple(det_tokens.shape)}"
            )
        if rope_params.shape[0] != batch_size or rope_params.shape[1] != num_det:
            raise ValueError(
                f"Shape mismatch: attn_h is [B={batch_size},N={num_det},H], rope_params is {tuple(rope_params.shape)}"
            )

        device = det_tokens.device
        dtype = det_tokens.dtype
        attn_h = attn_h.to(device=device, dtype=dtype)
        attn_w = attn_w.to(device=device, dtype=dtype)
        rope_params = rope_params.to(device=device, dtype=dtype)

        y = rope_params[..., 0:1]
        x = rope_params[..., 1:2]
        ph = rope_params[..., 2:3]
        pw = rope_params[..., 3:4]

        def _interp_1d(attn_1d: torch.Tensor) -> torch.Tensor:
            if attn_1d.shape[-1] == self.attn_interp_dim:
                return attn_1d
            x_in = attn_1d.reshape(batch_size * num_det, 1, attn_1d.shape[-1])
            x_out = F.interpolate(x_in, size=self.attn_interp_dim, mode="linear", align_corners=False)
            return x_out.reshape(batch_size, num_det, self.attn_interp_dim)

        attn_h_proj = self.attn_proj(_interp_1d(attn_h))
        attn_w_proj = self.attn_proj(_interp_1d(attn_w))
        token_proj = self.token_proj(det_tokens)

        dir_h = torch.zeros((batch_size, num_det, 1), device=device, dtype=dtype)
        dir_w = torch.ones((batch_size, num_det, 1), device=device, dtype=dtype)

        feature_h = torch.cat((dir_h, attn_h_proj, token_proj, y, ph), dim=-1)
        feature_w = torch.cat((dir_w, attn_w_proj, token_proj, x, pw), dim=-1)

        delta_h = self.down_proj(self.act_fn(self.gate_proj(feature_h)) * self.up_proj(feature_h))
        delta_w = self.down_proj(self.act_fn(self.gate_proj(feature_w)) * self.up_proj(feature_w))

        d_y = self.max_stride * torch.tanh(delta_h[..., 0:1])
        d_ph = torch.tanh(delta_h[..., 1:2])
        d_x = self.max_stride * torch.tanh(delta_w[..., 0:1])
        d_pw = torch.tanh(delta_w[..., 1:2])

        y_next = torch.clamp(y + d_y, -1, 1)
        x_next = torch.clamp(x + d_x, -1, 1)
        ph_next = torch.tanh(ph + d_ph)
        pw_next = torch.tanh(pw + d_pw)

        rope_params_next = torch.cat((y_next, x_next, ph_next, pw_next), dim=-1)
        return rope_params_next
    
def get_marginal_attn(
        det_s: int, det_e: int,
        patch_s:int, patch_e:int, 
    num_patches_h:int, num_patches_w:int,
    attn_weights: Optional[torch.Tensor] = None,
    attention_weights: Optional[torch.Tensor] = None,
    **kwargs,
)-> tuple[torch.Tensor, torch.Tensor]:

    # Backward/forward compatible kwarg naming
    if attn_weights is None:
        attn_weights = attention_weights
    if attn_weights is None:
        raise TypeError("get_marginal_attn() missing required argument: 'attn_weights'")

    attn = attn_weights[:, :, det_s:det_e, patch_s:patch_e]
    attn = attn.mean(dim=1)

    attn = attn.reshape(attn.shape[0], attn.shape[1], num_patches_h, num_patches_w)

    attn_h = attn.sum(dim=-1)
    attn_w = attn.sum(dim=-2)

    return attn_h, attn_w
    
def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos_sin: list[torch.Tensor], 
    start: int, end: int, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors, but only to the patch tokens,
    ignoring the prefix tokens (cls token and register tokens).

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    q_tokens = q[..., start:end, :]
    k_tokens = k[..., start:end, :]

    cos, sin = cos_sin
    cos = cos.to(device=q_tokens.device, dtype=q_tokens.dtype)
    sin = sin.to(device=q_tokens.device, dtype=q_tokens.dtype)

    q_rot = (q_tokens * cos) + (rotate_half(q_tokens) * sin)
    k_rot = (k_tokens * cos) + (rotate_half(k_tokens) * sin)

    q_out = q.clone()
    k_out = k.clone()
    q_out[..., start:end, :] = q_rot
    k_out[..., start:end, :] = k_rot

    return q_out, k_out

def apply_det_temperature(
    q: torch.Tensor, k: torch.Tensor, ph_pw: tuple[torch.Tensor, torch.Tensor], p_scale: float,
    start: int, end: int, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    # IMPORTANT: do NOT scale slices in-place on `q`/`k` views (AsStrided).
    q_tokens = q[..., start:end, :]
    k_tokens = k[..., start:end, :]
    half_len = q.shape[-1] // 2

    ph, pw = ph_pw
    ph = ph * p_scale
    pw = pw * p_scale
    tem_h = torch.exp(ph)
    tem_w = torch.exp(pw)

    # Normalize shapes for broadcasting to (B, heads, N, head_dim/2)
    def _norm_tem(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:  # (N,)
            return t.view(1, 1, -1, 1)
        if t.dim() == 2:  # (B, N)
            return t.unsqueeze(1).unsqueeze(-1)
        if t.dim() == 3:  # (B, N, 1)
            return t.unsqueeze(1)
        return t

    tem_h = _norm_tem(tem_h).to(device=q_tokens.device, dtype=q_tokens.dtype)
    tem_w = _norm_tem(tem_w).to(device=q_tokens.device, dtype=q_tokens.dtype)

    q_scaled = torch.cat(
        (q_tokens[..., :half_len] * tem_h, q_tokens[..., half_len:] * tem_w),
        dim=-1,
    )
    k_scaled = torch.cat(
        (k_tokens[..., :half_len] * tem_h, k_tokens[..., half_len:] * tem_w),
        dim=-1,
    )

    q_out = q.clone()
    k_out = k.clone()
    q_out[..., start:end, :] = q_scaled
    k_out[..., start:end, :] = k_scaled

    return q_out, k_out


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _to_additive_gate(
    gate: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    gate = gate.to(device=device)

    if gate.dtype == torch.bool:
        neg_inf = torch.finfo(dtype).min
        return torch.where(
            gate,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), neg_inf, device=device, dtype=dtype),
        )
    else:
        return gate.to(dtype=dtype, device=device)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_gate: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_gate is not None:
        additive_gate = _to_additive_gate(gate=attention_gate, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = attn_weights + additive_gate
    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DINOv3ViTAttention(nn.Module):
    """
    Multi-headed attention compatible with ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False

        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

        self.det_tem_power_scale = config.det_tem_power_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_info: dict,
        attention_gate: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patch_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        det_rope_embeddings: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
		**kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, num_tokens = hidden_states.shape[0], hidden_states.shape[1]
        patch_s, patch_e, det_s, det_e = (
            batch_info["patch_s"],
            batch_info["patch_e"],
            batch_info["det_s"],
            batch_info["det_e"],
        )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if patch_position_embeddings is not None:
            query_states, key_states = apply_rotary_pos_emb(q=query_states, k=key_states, cos_sin=patch_position_embeddings, start=patch_s, end=patch_e)
        if det_rope_embeddings is not None:
            cos_sin = det_rope_embeddings[:2]
            ph_pw = det_rope_embeddings[2:]
            query_states, key_states = apply_rotary_pos_emb(q=query_states, k=key_states, cos_sin=cos_sin, start=det_s, end=det_e)
            query_states, key_states = apply_det_temperature(q=query_states, k=key_states, ph_pw=ph_pw, p_scale=self.det_tem_power_scale, start=det_s, end=det_e)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_gate=attention_gate,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, num_tokens, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class DINOv3ViTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class DINOv3ViTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTGatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DINOv3ViTLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = DINOv3ViTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(config)
        else:
            self.mlp = DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_info: dict,
        attention_gate: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patch_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        det_rope_embeddings: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        output_marginal_attention: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        if output_marginal_attention:
            det_s, det_e, patch_s, patch_e, num_patches_h, num_patches_w = batch_info['det_s'], batch_info['det_e'], batch_info['patch_s'], batch_info['patch_e'], batch_info['num_patches_h'], batch_info['num_patches_w']
            hidden_states, attention_weights = self.attention(
                hidden_states=hidden_states,
                batch_info=batch_info,
                attention_gate=attention_gate,
                attention_mask=attention_mask,
                patch_position_embeddings=patch_position_embeddings,
                det_rope_embeddings=det_rope_embeddings
            )
            marginal_attn = get_marginal_attn(attention_weights=attention_weights, det_s=det_s, det_e=det_e, patch_s=patch_s, patch_e=patch_e, num_patches_h=num_patches_h, num_patches_w=num_patches_w)
        else:
            hidden_states, _ = self.attention(
                hidden_states=hidden_states,
                batch_info=batch_info,
                attention_gate=attention_gate,
                attention_mask=attention_mask,
                patch_position_embeddings=patch_position_embeddings,
                det_rope_embeddings=det_rope_embeddings
            )

        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        if output_marginal_attention:
            return hidden_states, marginal_attn
        else:
            return hidden_states


class DINOv3ViTPreTrainedModel(PreTrainedModel):
    config: DINOv3ViTConfig
    base_model_prefix = "dinov3_vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DINOv3ViTLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DINOv3ViTLayer,
        "attentions": DINOv3ViTAttention,
    }

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, DINOv3ViTEmbeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)
            if module.config.num_register_tokens > 0:
                module.register_tokens.data = nn.init.trunc_normal_(
                    module.register_tokens.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.register_tokens.dtype)
            if module.config.num_det_tokens > 0:
                module.det_tokens.data = nn.init.trunc_normal_(
                    module.det_tokens.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.det_tokens.dtype)
            module.mask_token.data.zero_()
        elif isinstance(module, DINOv3ViTLayerScale):
            module.lambda1.data.fill_(self.config.layerscale_value)

def get_det_gate(gate_type, info):
    if gate_type is None:
        return None
    else:
        device, num_tokens, det_e, det_s, patch_e= info['device'], info['num_tokens'], info['det_e'], info['det_s'], info['patch_e']
        GATE_TYPES = {
            "block_non_det_read_det":[ (slice(0, patch_e),slice(det_s,det_e))],
            "block_all_read_det":[ (slice(0, num_tokens),slice(det_s,det_e))],
        }
        assert gate_type in GATE_TYPES.keys(), f"Unknown det_token_gate={gate_type}"
        gate = torch.ones((num_tokens, num_tokens), dtype=torch.bool, device=device)

        for qs, ks in GATE_TYPES[gate_type]:
            gate[qs, ks] = False

    return gate

    
class DINOv3ViTModel(DINOv3ViTPreTrainedModel):
    def __init__(self, config: DINOv3ViTConfig, **kwargs):
        super().__init__(config)
        self.config = config

        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.det_token_gate = config.det_token_gate
        self.enable_det_rope = config.enable_det_rope
        self.enable_det_additive_embeddings = config.enable_det_additive_embeddings

        self.gradient_checkpointing = False
        self.post_init()

        if self.enable_det_rope:
            self.det_rope = DetRopePositionEmbedding(config)
            self.det_rope_type = config.det_rope_type
            if self.det_rope_type == 'fixed':
                n = self.config.num_det_tokens
                hw = math.isqrt(n)
                assert hw * hw == n, f"num_det_tokens must be a perfect square, got {n}"
                coords = get_fixed_det_coordinates(num_h=hw, num_w=hw)
                params = F.pad(coords, (0, 2), "constant", 0)
                self.register_buffer("det_rope_params", params, persistent=False)
            elif self.det_rope_type == 'learnable':
                self.det_rope_params = nn.Parameter(torch.empty(self.config.num_hidden_layers, self.config.num_det_tokens, 4))
                torch.nn.init.uniform_(self.det_rope_params, a=-1, b=1)
                with torch.no_grad():
                    self.det_rope_params[..., 2:].zero_()
            elif self.det_rope_type == 'adaptive':
                self.det_rope_params = nn.Parameter(torch.empty(self.config.num_det_tokens, 4))
                self.det_rope_updater = nn.ModuleList([DetRopeUpdater(config) for _ in range(config.num_hidden_layers)])
                torch.nn.init.uniform_(self.det_rope_params, a=-1, b=1)
                with torch.no_grad():
                    self.det_rope_params[..., 2:].zero_()
            else:
                raise ValueError(f'unknown det RoPE type {self.det_rope_type}')
        
        if self.enable_det_additive_embeddings:
            self.det_additive_embeddings = nn.Parameter(torch.empty(self.config.num_hidden_layers, self.config.num_det_tokens, self.config.hidden_size))
            nn.init.trunc_normal_(
                self.det_additive_embeddings,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @check_model_inputs
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
		**kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """

        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        patch_size=self.config.patch_size
        batch_size, _, height, width = pixel_values.shape
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_tokens = hidden_states.shape[1]
        det_e = num_tokens
        det_s = num_tokens - self.config.num_det_tokens
        patch_e = det_s
        patch_s = 1 + self.config.num_register_tokens
        device = pixel_values.device
        dtype = pixel_values.dtype
        assert 0 <= patch_s <= patch_e <= det_s <= det_e == num_tokens
        assert num_patches_h * num_patches_w == patch_e - patch_s
        batch_info = {
            'batch_size': batch_size,
            'num_tokens':num_tokens,
            'patch_s': patch_s,
            'patch_e': patch_e,
            'det_s': det_s,
            'det_e': det_e,
            'num_patches_h': num_patches_h,
            'num_patches_w': num_patches_w,
            'device': device,
            'dtype': dtype
        }

        attention_gate = get_det_gate(gate_type=self.det_token_gate ,info=batch_info)
        patch_position_embeddings = self.rope_embeddings(batch_info)

        aux_layer_ids = (2, 5, 8)
        aux_det_tokens = []
        aux_rope_yx = []

        self.aux_det_tokens = None
        self.aux_rope_yx = None
        
        # For adaptive det RoPE, keep a per-forward, per-batch rope state.
        rope_params = None
        if self.enable_det_rope and self.det_rope_type == "adaptive":
            rope_params = self.det_rope_params.unsqueeze(0).expand(batch_size, -1, -1)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.enable_det_additive_embeddings:
                det_e = hidden_states.shape[1]
                det_s = det_e - self.config.num_det_tokens
                hidden_states[:,det_s:det_e,:] += self.det_additive_embeddings[i]
            if self.enable_det_rope:
                if self.det_rope_type == "adaptive":
                    layer_det_rope_param = rope_params
                else:
                    layer_det_rope_param = self.det_rope_params if self.det_rope_params.dim() == 2 else self.det_rope_params[i]

                coords = layer_det_rope_param[..., :2]
                tem_p = layer_det_rope_param[..., 2:]
                ph = tem_p[..., 0]
                pw = tem_p[..., 1]
                # Normalize ph/pw for broadcasting later
                if ph.dim() == 1:
                    ph = ph.view(1, -1)
                    pw = pw.view(1, -1)

                cos, sin = self.det_rope(coords, batch_info)
                det_rope_embeddings = (cos, sin, ph, pw)
                if self.det_rope_type == 'adaptive':
                    hidden_states, marginal_attn = layer_module(
                        hidden_states= hidden_states,
                        batch_info=batch_info,
                        attention_gate=attention_gate,
                        attention_mask=layer_head_mask,
                        patch_position_embeddings=patch_position_embeddings,
                        det_rope_embeddings=det_rope_embeddings,
                        output_marginal_attention=True,
                    )
                    marginal_attn = (marginal_attn[0].detach(), marginal_attn[1].detach())
                    det_tokens_for_updater = hidden_states[:, det_s:det_e, :].detach()
                    rope_params = self.det_rope_updater[i](
                        rope_params=rope_params,
                        marginal_attn=marginal_attn,
                        det_tokens=det_tokens_for_updater,
                    )

                    if self.training and i in aux_layer_ids:
                        aux_det_tokens.append(hidden_states[:, det_s:det_e, :])
                        aux_rope_yx.append(rope_params[..., :2])
                else:
                    hidden_states = layer_module(
                        hidden_states= hidden_states,
                        batch_info=batch_info,
                        attention_gate=attention_gate,
                        attention_mask=layer_head_mask,
                        patch_position_embeddings=patch_position_embeddings,
                        det_rope_embeddings=det_rope_embeddings,
                        output_marginal_attention=False,
                    )
            else:
                hidden_states = layer_module(
                    hidden_states= hidden_states,
                    batch_info=batch_info,
                    attention_gate=attention_gate,
                    attention_mask=layer_head_mask,
                    patch_position_embeddings=patch_position_embeddings,
                    output_marginal_attention=False,
                )

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]

        if self.training and self.enable_det_rope and self.det_rope_type == "adaptive":
            self.aux_det_tokens = aux_det_tokens if len(aux_det_tokens) > 0 else None
            self.aux_rope_xy = aux_rope_yx if len(aux_rope_yx) > 0 else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class DINOv3Backbone(DINOv3ViTModel):
    def __init__(self, config: DINOv3ViTConfig, gate=False, **kwargs):
        super().__init__(config)
        self.gate = gate

    def forward(self, x):
        out = super().forward(x)
        det_tokens = out.last_hidden_state[:, -self.config.num_det_tokens:, :]

        aux_det_tokens = getattr(self, "aux_det_tokens", None)
        aux_rope_yx = getattr(self, "aux_rope_yx", None)
        return {
            "det_tokens": det_tokens,
            "aux_det_tokens": aux_det_tokens,
            "aux_rope_yx": aux_rope_yx,
        }
    
    @torch.jit.ignore
    def get_attentions(self, x):
        return torch.stack(super().forward(x).attentions)

    @torch.jit.ignore
    def get_blocks(self):
        return list(self.layer)
    
    @torch.jit.ignore
    def init_unloaded_parameters(self, info):
        param_dict = dict(self.named_parameters())

        unloaded = info.get("missing_keys", []) + info.get("mismatched_keys", [])

        for name in unloaded:
            if name in param_dict:
                param = param_dict[name]
                if "det_rope_params" in name.split(".") and param.data.dim() >= 2 and param.data.shape[-1] == 4:
                    with torch.no_grad():
                        param.data[..., :2].uniform_(-1, 1)
                        param.data[..., 2:].zero_()
                else:
                    torch.nn.init.trunc_normal_(param.data, std=self.config.initializer_range)


def build_dinov3(pretrained=None, output_attentions=False, **kwargs):
    model, info = DINOv3Backbone.from_pretrained(pretrained, ignore_mismatched_sizes=True, output_loading_info=True, output_attentions=output_attentions, **kwargs)
    model.init_unloaded_parameters(info)
    return model, model.config.hidden_size

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
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import compile_compatible_method_lru_cache
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
        use_loc_hint: bool = False,
        use_det_rope: bool = False,
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
        self.use_loc_hint = use_loc_hint
        self.use_det_rope = use_det_rope

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

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> tuple[torch.Tensor,torch.Tensor]:
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
        embeddings = torch.cat((cls_token, register_tokens, patch_embeddings), dim=1)

        return embeddings, det_tokens


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

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_size
        num_patches_w = width // self.config.patch_size

        device = pixel_values.device
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

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

def get_rope_from_coords(config, coords:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = coords.device
    dtype = coords.dtype

    if coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (..., 2), got {tuple(coords.shape)}")

    base = config.rope_theta
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1 / base ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32)

    device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        coords_f32 = coords.to(device=device, dtype=torch.float32)
        inv_freq_f32 = inv_freq.to(device=device, dtype=torch.float32)

        # coords_f32: (..., 2)
        # inv_freq_f32: (head_dim/4,)
        # angles: (..., 2, head_dim/4) -> (..., head_dim/2) -> (..., head_dim)
        angles = 2 * math.pi * coords_f32[..., :, None] * inv_freq_f32
        angles = angles.reshape(*coords_f32.shape[:-1], -1)
        angles = torch.cat((angles, angles), dim=-1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

    return cos.to(dtype=dtype), sin.to(dtype=dtype)
    
def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos:torch.Tensor, sin: torch.Tensor, **kwargs
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
    k_len = k.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = k_len - num_patches
    if num_prefix_tokens < 0:
        raise ValueError(
            f"RoPE patch count ({num_patches}) exceeds key length ({k_len}). "
        )
    start = num_prefix_tokens
    end = k_len

    q_tokens = q[..., start:end, :]
    k_tokens = k[..., start:end, :]

    cos = cos.to(device=q_tokens.device, dtype=q_tokens.dtype)
    sin = sin.to(device=q_tokens.device, dtype=q_tokens.dtype)

    q_rot = (q_tokens * cos) + (rotate_half(q_tokens) * sin)
    k_rot = (k_tokens * cos) + (rotate_half(k_tokens) * sin)

    q_out = q.clone()
    k_out = k.clone()
    q_out[..., start:end, :] = q_rot
    k_out[..., start:end, :] = k_rot

    return q_out, k_out


def apply_rotary_pos_emb_q(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs
) -> torch.Tensor:
    if q.shape[-2] != cos.shape[-2] or q.shape[-2] != sin.shape[-2]:
        raise ValueError(
            "RoPE length mismatch: "
            f"q seq_len={q.shape[-2]}, cos seq_len={cos.shape[-2]}, sin seq_len={sin.shape[-2]}"
        )

    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)

    return (q * cos) + (rotate_half(q) * sin)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

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

        self.det_q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        det_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        det_rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        det_temperatures: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()
        _, num_det_tokens, _ = det_tokens.size()
        len_q = patches + num_det_tokens

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        det_query_states = self.det_q_proj(det_tokens)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        det_query_states = det_query_states.view(batch_size, num_det_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        if det_rope_cos_sin is not None:
            det_cos, det_sin = det_rope_cos_sin

            if det_cos.shape[-2] != num_det_tokens or det_sin.shape[-2] != num_det_tokens:
                raise ValueError(
                    "Det RoPE length mismatch: "
                    f"num_det_tokens={num_det_tokens}, cos seq_len={det_cos.shape[-2]}, sin seq_len={det_sin.shape[-2]}"
                )
            if det_cos.shape[-1] != self.head_dim or det_sin.shape[-1] != self.head_dim:
                raise ValueError(
                    "Det RoPE head_dim mismatch: "
                    f"head_dim={self.head_dim}, cos last_dim={det_cos.shape[-1]}, sin last_dim={det_sin.shape[-1]}"
                )

            if det_cos.dim() == 2:
                det_cos = det_cos.unsqueeze(0)
                det_sin = det_sin.unsqueeze(0)
            if det_cos.dim() == 3:
                det_cos = det_cos.unsqueeze(1)
                det_sin = det_sin.unsqueeze(1)

            det_cos = det_cos.to(device=det_query_states.device, dtype=det_query_states.dtype)
            det_sin = det_sin.to(device=det_query_states.device, dtype=det_query_states.dtype)
            det_query_states = (det_query_states * det_cos) + (rotate_half(det_query_states) * det_sin)

        if det_temperatures is not None:
            if isinstance(det_temperatures, torch.Tensor) and det_temperatures.dim() == 3:
                det_temperatures = det_temperatures.unsqueeze(1)
            det_query_states = det_query_states * det_temperatures

        query_states = torch.cat((query_states, det_query_states), dim=2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(q=query_states, k=key_states, cos=cos, sin=sin)


        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, len_q, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output[:, :patches, :], attn_output[:, patches:, :], attn_weights


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
        det_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        det_rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        det_temperatures: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention with residual connection
        residual = hidden_states
        residual_det = det_tokens

        hidden_states = self.norm1(hidden_states)
        det_tokens = self.norm1(det_tokens)

        hidden_states, det_tokens, _ = self.attention(
            hidden_states=hidden_states,
            det_tokens=det_tokens,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            det_rope_cos_sin = det_rope_cos_sin,
            det_temperatures = det_temperatures
        )

        hidden_states = self.layer_scale1(hidden_states)
        det_tokens = self.layer_scale1(det_tokens)

        hidden_states = self.drop_path(hidden_states) + residual
        det_tokens = self.drop_path(det_tokens) + residual_det

        # MLP with residual connection
        residual = hidden_states
        residual_det = det_tokens

        hidden_states = self.norm2(hidden_states)
        det_tokens = self.norm2(det_tokens)

        hidden_states = self.mlp(hidden_states)
        det_tokens = self.mlp(det_tokens)

        hidden_states = self.layer_scale2(hidden_states)
        det_tokens = self.layer_scale2(det_tokens)

        hidden_states = self.drop_path(hidden_states) + residual
        det_tokens = self.drop_path(det_tokens) + residual_det

        return hidden_states, det_tokens

class GatedMLP(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.output_size = output_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class LocalizationHint(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.location_embed = GatedMLP(config, 2)
        self.temperature_embed = GatedMLP(config, 1)
    
    def forward(self, det_tokens):
        coords = self.location_embed(det_tokens)
        coords = F.tanh(coords)
        temperatures = self.temperature_embed(det_tokens)
        temperatures = F.sigmoid(temperatures)
        cos_sin = get_rope_from_coords(config=self.config, coords=coords)

        return cos_sin, temperatures

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

    
class DINOv3ViTModel(DINOv3ViTPreTrainedModel):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self._init_det_q_proj_from_q_proj()

        if config.use_loc_hint:
            self.loc_hint = LocalizationHint(config)
        else:
            self.loc_hint = None
        
        if config.use_det_rope:
            if config.use_loc_hint:
                self.det_coords = nn.Parameter(torch.empty(config.num_hidden_layers-1, config.num_det_tokens, 2))
            else:
                self.det_coords = nn.Parameter(torch.empty(config.num_hidden_layers, config.num_det_tokens, 2))
            nn.init.uniform_(self.det_coords, -1.0, 1.0)
        else:
            self.det_coords = None

    def _init_det_q_proj_from_q_proj(self) -> None:
        for module in self.modules():
            if not isinstance(module, DINOv3ViTAttention):
                continue
            with torch.no_grad():
                if module.det_q_proj.weight.shape == module.q_proj.weight.shape:
                    module.det_q_proj.weight.copy_(module.q_proj.weight)
                if module.det_q_proj.bias is not None:
                    if module.q_proj.bias is not None and module.det_q_proj.bias.shape == module.q_proj.bias.shape:
                        module.det_q_proj.bias.copy_(module.q_proj.bias)
                    else:
                        module.det_q_proj.bias.zero_()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

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
        hidden_states, det_tokens = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            det_rope_cos_sin = None
            det_temperatures = None
            if self.loc_hint is not None and i == len(self.layer)-1:
                det_rope_cos_sin, det_temperatures = self.loc_hint(det_tokens)
            elif self.det_coords is not None and i < len(self.det_coords):
                det_rope_cos_sin = get_rope_from_coords(config=self.config, coords=self.det_coords[i])
            hidden_states, det_tokens = layer_module(
                hidden_states = hidden_states,
                det_tokens = det_tokens,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
                det_rope_cos_sin = det_rope_cos_sin,
                det_temperatures = det_temperatures
            )

        sequence_output = self.norm(hidden_states)
        det_output = self.norm(det_tokens)
        pooled_output = sequence_output[:, 0, :]

        last_hidden_state = torch.cat((sequence_output, det_output), dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class DINOv3Backbone(DINOv3ViTModel):
    def __init__(self, config: DINOv3ViTConfig, gate=False, **kwargs):
        super().__init__(config)

    def forward(self, x):
        return super().forward(x).last_hidden_state[:, -self.config.num_det_tokens:, :]

    @torch.jit.ignore
    def get_blocks(self):
        return list(self.layer)
    
    @torch.jit.ignore
    def init_unloaded_parameters(self, info):
        param_dict = dict(self.named_parameters())

        missing_keys = list(info.get("missing_keys", []) or [])
        mismatched_keys = list(info.get("mismatched_keys", []) or [])
        unloaded: list[str] = []
        unloaded.extend(missing_keys)
        for item in mismatched_keys:
            unloaded.append(item[0] if isinstance(item, tuple) else item)

        handled: set[str] = set()

        for name in unloaded:
            if ".det_q_proj." not in name:
                continue
            if name not in param_dict:
                continue

            src_name = name.replace(".det_q_proj.", ".q_proj.")
            dst_param = param_dict[name]

            if src_name in param_dict and param_dict[src_name].shape == dst_param.shape:
                with torch.no_grad():
                    dst_param.copy_(param_dict[src_name])
                handled.add(name)
                continue

            if name.endswith(".det_q_proj.bias"):
                with torch.no_grad():
                    dst_param.zero_()
                handled.add(name)

        for name in unloaded:
            if name in handled:
                continue
            if name in param_dict:
                param = param_dict[name]

                if name == "det_coords" or name.startswith("det_coords."):
                    with torch.no_grad():
                        nn.init.uniform_(param.data, -1.0, 1.0)
                    handled.add(name)
                    continue

                torch.nn.init.trunc_normal_(param.data, std=self.config.initializer_range)


def build_dinov3(pretrained=None, **kwargs):
    if pretrained is None or pretrained is False or pretrained == "":
        config = DINOv3ViTConfig(**kwargs)
        model = DINOv3Backbone(config)
        return model, config.hidden_size

    model, info = DINOv3Backbone.from_pretrained(
        pretrained,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        **kwargs,
    )
    model.init_unloaded_parameters(info)
    return model, model.config.hidden_size

import torch
import torch.utils.checkpoint as checkpoint 
from typing import Optional
from typing_extensions import Unpack

from transformers import DINOv3ViTConfig, DINOv3ViTModel
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.generic import check_model_inputs
from transformers.modeling_outputs import BaseModelOutputWithPooling

DINOV3_CONFIG = {
    'small':{'repo_id':'facebook/dinov3-vits16-pretrain-lvd1689m', 'hidden_dim':384}
}

class DINOv3ViTDetConfig(DINOv3ViTConfig):
    def __init__(self, finetune = False, use_checkpoint=False, patch_size = 16, hidden_size = 384, intermediate_size = 1536, num_hidden_layers = 12, num_attention_heads = 6, num_det_token = 100, hidden_act = "gelu", attention_dropout = 0, initializer_range = 0.02, layer_norm_eps = 0.00001, rope_theta = 100, image_size = 224, num_channels = 3, query_bias = True, key_bias = False, value_bias = True, proj_bias = True, mlp_bias = True, layerscale_value = 1, drop_path_rate = 0, use_gated_mlp = False, num_register_tokens = 0, pos_embed_shift = None, pos_embed_jitter = None, pos_embed_rescale = 2, **kwargs):
        super().__init__(patch_size=patch_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, hidden_act=hidden_act, attention_dropout=attention_dropout, initializer_range=initializer_range, layer_norm_eps=layer_norm_eps, rope_theta=rope_theta, image_size=image_size, num_channels=num_channels, query_bias=query_bias, key_bias=key_bias, value_bias=value_bias, proj_bias=proj_bias, mlp_bias=mlp_bias, layerscale_value=layerscale_value, drop_path_rate=drop_path_rate, use_gated_mlp=use_gated_mlp, num_register_tokens=num_register_tokens, pos_embed_shift=pos_embed_shift, pos_embed_jitter=pos_embed_jitter, pos_embed_rescale=pos_embed_rescale, **kwargs)
        self.finetune = finetune
        self.num_det_token = num_det_token
        self.use_checkpoint = use_checkpoint

class DINOv3ViTDet(DINOv3ViTModel):
    config_class = DINOv3ViTDetConfig
    
    def __init__(self, config: DINOv3ViTDetConfig):
        config.num_register_tokens = config.num_det_token
        super().__init__(config)

        self.gradient_checkpointing = config.use_checkpoint

        self.num_det_token = config.num_det_token
        self.det_tokens = self.embeddings.register_tokens
        torch.nn.init.trunc_normal_(self.det_tokens, std=config.initializer_range)

        self.det_pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_det_token, config.hidden_size))
        torch.nn.init.trunc_normal_(self.det_pos_embed, std=config.initializer_range)

        if not config.finetune:
            for p in self.parameters():
                p.requires_grad = False
            self.embeddings.register_tokens.requires_grad = True
            self.det_pos_embed.requires_grad = True

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Copied and modified from the original HuggingFace Transformers DINOv3ViTModel.forward implementation.

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """

        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)

        cls_token = hidden_states[:, :1, :]
        det_tokens = hidden_states[:, 1:1+self.num_det_token,:]
        patch_embeddings = hidden_states[:,1+self.num_det_token:,:]
        det_tokens += self.det_pos_embed
        hidden_states = torch.cat((cls_token,det_tokens,patch_embeddings),dim=1)
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint.checkpoint(
                    lambda *inputs: layer_module(*inputs),
                    hidden_states,
                    layer_head_mask,
                    position_embeddings,
                )
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=layer_head_mask,
                    position_embeddings=position_embeddings,
                )

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )
    
class DINOv3Backbone(DINOv3ViTDet):
    def forward(self, x):
        return super().forward(x, bool_masked_pos = None, head_mask = None).last_hidden_state[:, 1:1+self.num_det_token,:]

def build_dinov3(size='small', pretrained=False, **kwargs):
    assert size in DINOV3_CONFIG.keys()
    if pretrained:
        model = DINOv3Backbone.from_pretrained(DINOV3_CONFIG[size]['repo_id'], ignore_mismatched_sizes=True, **kwargs)
    else:
        config = DINOv3ViTDetConfig.from_pretrained(DINOV3_CONFIG[size]['repo_id'], **kwargs)
        model = DINOv3Backbone(config)
    return model, DINOV3_CONFIG[size]['hidden_dim']


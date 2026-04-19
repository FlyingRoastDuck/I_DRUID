# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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

    
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from .druidsd3_attention import AdapterDisenTransformerBlock
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.activations import FP32SiLU, get_activation
logger = logging.get_logger(__name__) 

def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x 4] representing the bounding boxes for GLIGEN pipeline
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, embed_dim * 2 * 4)

    return emb


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class TextBoundingboxProjectionFLUX(nn.Module):
    def __init__(self, positive_len, out_dim, fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim
        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy #64
        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]
        self.linears = PixArtAlphaTextProjection(in_features=self.positive_len + self.position_dim,hidden_size=out_dim//2,out_features=out_dim, act_fn="silu")
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self,
        boxes,#[B,10,4]
        masks,#[B,10]
        positive_embeddings, #torch.Size([B, 10, 512,1536])
    ):
        B,max_box,num_token,dim = positive_embeddings.shape
        masks = masks.unsqueeze(-1) #torch.Size([2, 10, 1])
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes)  # B*N*4 -> B*N*C #torch.Size([2, 10, 64])
        # learnable null embedding
        xyxy_null = self.null_position_feature.view(1, 1, -1) #torch.Size([1, 1, 64])
        # replace padding with learnable null embedding
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null #torch.Size([2, 10, 64])

        # 增加一个维度
        xyxy_embedding_unsqueezed = torch.unsqueeze(xyxy_embedding, 2)  # shape变为[2, 10, 1, 64]
        # 然后扩展到新的大小
        xyxy_embedding_expanded = xyxy_embedding_unsqueezed.expand(-1, -1, num_token, -1)  # torch.Size([2, 10, 30, 64])
        masks = masks.unsqueeze(-1) #torch.Size([2, 10, 1, 1])
        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, 1, -1) #从[1536]变到[1,1,1,1536]
        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null #torch.Size([2, 10, 30, 1536])

        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding_expanded], dim=-1)) # torch.Size([2, 10,30, 1536+64]) ->torch.Size([2, 10,30,1536])
        objs = objs.view(B, max_box*num_token, -1)
        return objs #[B,300,1536]


class TextBoundingboxProjection(nn.Module):
    def __init__(self, pooled_projection_dim, positive_len, out_dim, fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim
        self.pooled_projection_dim = pooled_projection_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy #64

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, positive_len, act_fn="silu")
        self.linears = PixArtAlphaTextProjection(in_features=self.positive_len + self.position_dim,hidden_size=out_dim//2,out_features=out_dim, act_fn="silu")
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self, boxes, masks, positive_embeddings,
    ):
        masks = masks.unsqueeze(-1) 
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes) 

        # learnable null embedding
        xyxy_null = self.null_position_feature.view(1, 1, -1) 
        # replace padding with learnable null embedding
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null 

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        positive_embeddings = self.text_embedder(positive_embeddings) 

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null 
        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        return objs


# total model
class AdapterLayoutSD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        attention_type = "layout",
        max_boxes_per_image =10, 
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size, width=self.config.sample_size,
            patch_size=self.config.patch_size, in_channels=self.config.in_channels,
            embed_dim=self.inner_dim, pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                AdapterDisenTransformerBlock(
                    dim=self.inner_dim, num_attention_heads=self.config.num_attention_heads, 
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1, attention_type=attention_type,
                    bbox_pre_only= i == num_layers - 1, bbox_with_temb=True, 
                ) for i in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False

        self.attention_type = attention_type
        self.max_boxes_per_image = max_boxes_per_image
        if self.attention_type == "layout":
            self.position_net = TextBoundingboxProjection(
                pooled_projection_dim=self.config.pooled_projection_dim, positive_len=self.inner_dim, 
                out_dim=self.inner_dim
            )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self, hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True, layout_kwargs=None, bbox_scale=1., bbox=[], spatial_mask=None
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)  
        temb = self.time_text_embed(timestep, pooled_projections) # time condition
        encoder_hidden_states = self.context_embedder(encoder_hidden_states) 
        
        if self.attention_type=="layout" and layout_kwargs is not None and layout_kwargs.get("layout", None) is not None:
            layout_args = layout_kwargs["layout"]
            bbox_raw = layout_args["boxes"]
            # only clip features
            bbox_text_embeddings = layout_args["positive_embeddings"].to(dtype=hidden_states.dtype,device=hidden_states.device) 
            bbox_masks = layout_args["masks"]
            # the only pos_net
            bbox_hidden_states = self.position_net(
                boxes=bbox_raw, masks=bbox_masks, positive_embeddings=bbox_text_embeddings
            )
        else:
            N = hidden_states.shape[0]
            bbox_hidden_states = torch.zeros(N, 2*self.max_boxes_per_image, self.config.pooled_projection_dim, dtype=hidden_states.dtype, device=hidden_states.device)
            bbox_masks = torch.zeros(N, self.max_boxes_per_image, dtype=hidden_states.dtype, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), hidden_states, encoder_hidden_states, temb, 
                    bbox_hidden_states, bbox_scale, spatial_mask, **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, 
                    temb=temb, bbox_hidden_states=bbox_hidden_states, bbox_scale=bbox_scale, spatial_mask=spatial_mask
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, )

        return Transformer2DModelOutput(sample=output)

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        if hasattr(self, "gradient_checkpointing"):
            self.gradient_checkpointing = enable
        else:
            for module in self.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = enable
                elif hasattr(module, "_set_gradient_checkpointing"):
                    module._set_gradient_checkpointing(enable=enable, gradient_checkpointing_func=gradient_checkpointing_func)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ldm.modules.druid.migc_layers import CBAM, CrossAttention, LayoutAttention
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
import warnings
from diffusers.models.attention_processor import Attention
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from enum import Enum
from diffusers.models.embeddings import apply_rotary_emb


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
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


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
        self.linears = PixArtAlphaTextProjection(in_features=self.positive_len + self.position_dim, hidden_size=out_dim//2,out_features=out_dim, act_fn="silu")
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(
        self, boxes, masks, positive_embeddings, phrases_masks=None, image_masks=None,
        phrases_embeddings=None, image_embeddings=None,
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


warnings.filterwarnings("ignore")

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)  # torch.Size([5, 30, 64])


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        # -------------------------------------------------------------- #
        self.linears_position = nn.Sequential(
            nn.Linear(self.position_dim, 512), nn.SiLU(), nn.Linear(512, 512), 
            nn.SiLU(), nn.Linear(512, out_dim),
        )

    def forward(self, boxes):
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*1*4 --> B*1*C torch.Size([5, 1, 64])
        xyxy_embedding = self.linears_position(xyxy_embedding)  # B*1*C --> B*1*768 torch.Size([5, 1, 768])
        return xyxy_embedding


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


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class IDM(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels=1536, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(IDM, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        inputs = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return inputs * x, inputs * (1 - x)

class DisenLayoutProcessor(torch.nn.Module):
    def __init__(self, context_dim=1536, hidden_dim=2048, scale=1,):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()
        self.box_scale = 1.0 
        self.add_k_proj_ip = zero_module(nn.Linear(context_dim, context_dim))
        self.add_v_proj_ip = zero_module(nn.Linear(context_dim, context_dim))
        self.ip_adapter = zero_module(nn.Linear(context_dim, context_dim))
        self.IDM = zero_module(IDM())

    def __call__(
        self, attn, hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None, 
        ip_hidden_states=None, scale=1, spatial_mask=None, 
        max_objs=10, *args, **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
        batch_size = encoder_hidden_states.shape[0]
        query = attn.to_q(hidden_states) # to_q was frozen
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        sample_query = query # used for bbox-query
        sample_key = key
        sample_val = value
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states) # should this be bbox-text features
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # joint attention on image-text
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # features before ff 
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :residual.shape[1]], hidden_states[:, residual.shape[1]:],
        )
        assert ip_hidden_states is not None, "ip_features not available"

        # for ip-adapter
        ip_key = self.add_k_proj_ip(ip_hidden_states)
        ip_value = self.add_v_proj_ip(ip_hidden_states)

        ip_query = sample_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) #input image
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) 

        # apply disen here, ip_query--(B, )
        ip_hidden_states = F.scaled_dot_product_attention(ip_query, ip_key, ip_value, dropout_p=0.0, is_causal=False)
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(ip_query.dtype) # (B, 4096, 1536)

        if spatial_mask is not None:
            f_dim = ip_hidden_states.shape[1] # 1536
            spatial_mask = spatial_mask.view(batch_size, max_objs, -1).unsqueeze(1) # B, 1, 10, 4096
            ip_hidden_states = ip_hidden_states.transpose(1, 2).unsqueeze(2) # B, 1536, 1, 4096
            net_in = ip_hidden_states*spatial_mask
            better_out, worse_out = self.IDM(net_in) # disen part 
            ip_hidden_states = better_out.mean(2).transpose(1, 2) # b, 1536, 10, 4096

        # linear proj
        hidden_states = hidden_states + scale * self.ip_adapter(ip_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        return hidden_states, encoder_hidden_states



class AttentionBackendName(str, Enum):
    # EAGER = "eager"

    # `flash-attn`
    FLASH = "flash"
    FLASH_VARLEN = "flash_varlen"
    _FLASH_3 = "_flash_3"
    _FLASH_VARLEN_3 = "_flash_varlen_3"
    _FLASH_3_HUB = "_flash_3_hub"
    # _FLASH_VARLEN_3_HUB = "_flash_varlen_3_hub"  # not supported yet.

    # `aiter`
    AITER = "aiter"

    # PyTorch native
    FLEX = "flex"
    NATIVE = "native"
    _NATIVE_CUDNN = "_native_cudnn"
    _NATIVE_EFFICIENT = "_native_efficient"
    _NATIVE_FLASH = "_native_flash"
    _NATIVE_MATH = "_native_math"
    _NATIVE_NPU = "_native_npu"
    _NATIVE_XLA = "_native_xla"

    # `sageattention`
    SAGE = "sage"
    SAGE_VARLEN = "sage_varlen"
    _SAGE_QK_INT8_PV_FP8_CUDA = "_sage_qk_int8_pv_fp8_cuda"
    _SAGE_QK_INT8_PV_FP8_CUDA_SM90 = "_sage_qk_int8_pv_fp8_cuda_sm90"
    _SAGE_QK_INT8_PV_FP16_CUDA = "_sage_qk_int8_pv_fp16_cuda"
    _SAGE_QK_INT8_PV_FP16_TRITON = "_sage_qk_int8_pv_fp16_triton"
    # TODO: let's not add support for Sparge Attention now because it requires tuning per model
    # We can look into supporting something "autotune"-ing in the future
    # SPARGE = "sparge"

    # `xformers`
    XFORMERS = "xformers"


class _AttentionBackendRegistry:
    _backends = {}
    _constraints = {}
    _supported_arg_names = {}
    _supports_context_parallel = {}
    ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
    DIFFUSERS_ATTN_CHECKS = os.getenv("DIFFUSERS_ATTN_CHECKS", "0") in ENV_VARS_TRUE_VALUES
    DIFFUSERS_ATTN_BACKEND = os.getenv("DIFFUSERS_ATTN_BACKEND", "native")
    _active_backend = AttentionBackendName(DIFFUSERS_ATTN_BACKEND)
    _checks_enabled = DIFFUSERS_ATTN_CHECKS

    @classmethod
    def register(
        cls, backend: AttentionBackendName,
        constraints: Optional[List[Callable]] = None,
        supports_context_parallel: bool = False,
    ):
        logger.debug(f"Registering attention backend: {backend} with constraints: {constraints}")

        def decorator(func):
            cls._backends[backend] = func
            cls._constraints[backend] = constraints or []
            cls._supported_arg_names[backend] = set(inspect.signature(func).parameters.keys())
            cls._supports_context_parallel[backend] = supports_context_parallel
            return func

        return decorator

    @classmethod
    def get_active_backend(cls):
        return cls._active_backend, cls._backends[cls._active_backend]

    @classmethod
    def list_backends(cls):
        return list(cls._backends.keys())

    @classmethod
    def _is_context_parallel_enabled(
        cls, backend: AttentionBackendName, parallel_config: Optional["ParallelConfig"]
    ) -> bool:
        supports_context_parallel = backend in cls._supports_context_parallel
        is_degree_greater_than_1 = parallel_config is not None and (
            parallel_config.context_parallel_config.ring_degree > 1
            or parallel_config.context_parallel_config.ulysses_degree > 1
        )
        return supports_context_parallel and is_degree_greater_than_1

# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip
from vllm.attention.ops.paged_attn import PagedAttention

class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QKVParallelLinear] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        sliding_window: Optional[int] = None,
        
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # This will be overwritten by model initialization if we are using it.
        # N.B. currently we only support per tensor scalar scaling factors
        # & only applicable to ROCm (AMD GPU).
        # The scaling factor convention we are assuming is
        # quantized_value * scaling_factor ~= true_value
        # which is consistent with the practice of setting
        # scaling_factor = tensor_amax / FPtype_max
        self.kv_scale = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=sliding_window)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                self.kv_scale)
        output, _ = self.o_proj(attn_output)
        return output
    
# class HcacheLlamaAttention(LlamaAttention):
#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         hidden_cache: torch.Tensor,
#         attn_metadata: AttentionMetadata,
#     ) -> torch.Tensor:
#         """前向传播，从 hidden states 计算注意力。
        
#         Args:
#             positions: 位置编码
#             hidden_states: 当前 token 的隐藏状态，shape (batch_size, seq_len, hidden_dim)
#             hidden_cache: 历史 token 的隐藏状态，shape (num_blocks, block_size, hidden_dim)
#             attn_metadata: 注意力的元数据
#         """
#         # 1. 为当前 token 计算 QKV
#         qkv, _ = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
#         # 2. 从 hidden states 重新计算历史 token 的 KV
#         if hidden_cache is not None:
#             # hidden_cache 已经是正确的形状: (num_blocks, block_size, hidden_dim)
#             num_blocks, block_size, _ = hidden_cache.shape
            
#             # 计算历史 token 的 KV
#             historical_qkv, _ = self.qkv_proj(hidden_cache)
#             _, k_hist, v_hist = historical_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            
#             # 创建符合 PagedAttention 预期的 KV cache 格式
#             computed_kv = torch.stack([k_hist, v_hist], dim=0)
#         else:
#             computed_kv = None
            
#         # 3. 应用 RoPE 位置编码
#         q, k = self.rotary_emb(positions, q, k)
        
#         # 4. 计算注意力输出
#         attn_output = self.attn(q, k, v, computed_kv, attn_metadata, self.kv_scale)
        
#         # 5. 输出投影
#         output, _ = self.o_proj(attn_output)
#         return output
    
class HcacheLlamaAttention(LlamaAttention):
    """使用hidden states缓存的Llama注意力层。
    
    复用现有KV缓存结构，但存储的是hidden states。
    对于每两层共享的缓存：
    - 偶数层的hidden states存储在K位置
    - 奇数层的hidden states存储在V位置
    """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """从hidden states计算注意力。"""
        # 1. 计算当前token的QKV
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 2. 从缓存计算历史KV
        computed_kv = None
        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_dim)
            
            # 根据层索引选择正确的缓存
            # 偶数层的hidden states存在K位置，奇数层存在V位置
            hidden_cache = key_cache if layer_idx % 2 == 0 else value_cache
            
            if hidden_cache.size(0) > 0:
                # 从缓存中获取hidden states
                flat_hidden = hidden_cache.reshape(-1, self.hidden_size)
                
                # 计算historical KV
                historical_qkv, _ = self.qkv_proj(flat_hidden)
                _, k_hist, v_hist = historical_qkv.split(
                    [self.q_size, self.kv_size, self.kv_size], dim=-1)
                
                # 重塑回缓存形状
                k_hist = k_hist.reshape(hidden_cache.shape)
                v_hist = v_hist.reshape(hidden_cache.shape)
                
                # 创建计算好的KV缓存
                computed_kv = torch.stack([k_hist, v_hist], dim=0)
        
        # 3. 应用rotary position embeddings
        q, k = self.rotary_emb(positions, q, k)
        
        # 4. 使用注意力机制
        attn_output = self.attn(q, k, v, computed_kv, attn_metadata, self.kv_scale)
        
        # 5. 输出投影
        output, _ = self.o_proj(attn_output)
        
        # # 6. 只有偶数层才存储hidden states到缓存
        # if kv_cache is not None and attn_metadata.slot_mapping is not None and layer_idx % 2 == 0:
        #     key_cache, value_cache = PagedAttention.split_kv_cache(
        #         kv_cache, self.num_kv_heads, self.head_dim)
            
        #     # 重要：将hidden_states重塑为PagedAttention期望的形状
        #     # hidden_states已经是[num_tokens, hidden_size]
        #     # 我们需要将其重塑为[num_tokens, num_kv_heads, head_dim]，与k和v相同
        #     reshaped_hidden = hidden_states.view(-1, self.num_kv_heads * self.head_dim)
        #     reshaped_output = output.view(-1, self.num_heads * self.head_dim)
            
        #     # 注意：这里我们用相同的hidden_states存储在K和V位置
        #     # 这样下一层可以用V位置的来计算
        #     PagedAttention.write_to_paged_cache(
        #         reshaped_hidden,  # 当前层hidden states
        #         reshaped_output,  # 同样是当前层hidden states
        #         key_cache,
        #         value_cache,
        #         attn_metadata.slot_mapping,
        #         attn_metadata.kv_cache_dtype,
        #         1.0
        #     )
        # 在 HcacheLlamaAttention.forward 方法中

        # 在 HcacheLlamaAttention.forward 方法中

        if kv_cache is not None and attn_metadata.slot_mapping is not None and layer_idx % 2 == 0:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_dim)
            
            # 将hidden_states按照注意力头数重塑为PagedAttention期望的形状
            # 改为按照模型的num_kv_heads和head_dim重塑张量
            hidden_reshaped = hidden_states.view(-1, self.num_kv_heads, self.head_dim)
            output_reshaped = output.view(-1, self.num_kv_heads, self.head_dim)
            
            # 现在用正确维度的张量调用write_to_paged_cache
            PagedAttention.write_to_paged_cache(
                hidden_reshaped,  # 形状为[num_tokens, num_kv_heads, head_dim]
                output_reshaped,  # 形状为[num_tokens, num_kv_heads, head_dim]
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                attn_metadata.kv_cache_dtype,
                1.0
            )
        
        return output


# class LlamaDecoderLayer(nn.Module):

#     def __init__(
#         self,
#         config: LlamaConfig,
#         quant_config: Optional[QuantizationConfig] = None,
#         layer_index: int = 0,
#     ) -> None:
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         rope_theta = getattr(config, "rope_theta", 10000)
#         rope_scaling = getattr(config, "rope_scaling", None)
#         if rope_scaling is not None and getattr(
#                 config, "original_max_position_embeddings", None):
#             rope_scaling["original_max_position_embeddings"] = (
#                 config.original_max_position_embeddings)
#         max_position_embeddings = getattr(config, "max_position_embeddings",
#                                           8192)
#         sliding_window = getattr(config, "sliding_window", None)
#         # Support abacusai/Smaug-72B-v0.1 with attention_bias
#         # Support internlm/internlm-7b with bias
#         attention_bias = getattr(config, "attention_bias", False) or getattr(
#             config, "bias", False)
#         # self.self_attn = LlamaAttention(
#         #     hidden_size=self.hidden_size,
#         #     num_heads=config.num_attention_heads,
#         #     num_kv_heads=getattr(config, "num_key_value_heads",
#         #                          config.num_attention_heads),
#         #     rope_theta=rope_theta,
#         #     rope_scaling=rope_scaling,
#         #     max_position_embeddings=max_position_embeddings,
#         #     quant_config=quant_config,
#         #     bias=attention_bias,
#         #     sliding_window=sliding_window,
#         # )
#         self.index = layer_index
#         self.self_attn = HcacheLlamaAttention(
#             hidden_size=self.hidden_size,
#             num_heads=config.num_attention_heads,
#             num_kv_heads=getattr(config, "num_key_value_heads",
#                                  config.num_attention_heads),
#             rope_theta=rope_theta,
#             rope_scaling=rope_scaling,
#             max_position_embeddings=max_position_embeddings,
#             quant_config=quant_config,
#             bias=attention_bias,
#             sliding_window=sliding_window,
#         )
#         self.mlp = LlamaMLP(
#             hidden_size=self.hidden_size,
#             intermediate_size=config.intermediate_size,
#             hidden_act=config.hidden_act,
#             quant_config=quant_config,
#         )
#         self.input_layernorm = RMSNorm(config.hidden_size,
#                                        eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size,
#                                                 eps=config.rms_norm_eps)

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         kv_cache: torch.Tensor,
#         attn_metadata: AttentionMetadata,
#         residual: Optional[torch.Tensor],
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Self Attention
#         if residual is None:
#             residual = hidden_states
#             hidden_states = self.input_layernorm(hidden_states)
#         else:
#             hidden_states, residual = self.input_layernorm(
#                 hidden_states, residual)
#         hidden_states = self.self_attn(
#             positions=positions,
#             hidden_states=hidden_states,
#             hidden_cache=kv_cache,
#             attn_metadata=attn_metadata,
#         )

#         # Fully Connected
#         hidden_states, residual = self.post_attention_layernorm(
#             hidden_states, residual)
#         hidden_states = self.mlp(hidden_states)
#         return hidden_states, residual
    
class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int = 0,  # 添加层索引参数
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx  # 保存层索引
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        sliding_window = getattr(config, "sliding_window", None)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        
        self.self_attn = HcacheLlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            sliding_window=sliding_window,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        
        # 传递层索引和上一层hidden states给注意力层
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            layer_idx=self.layer_idx,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        # 返回当前层的hidden states，用于下一层
        return hidden_states, residual


# class LlamaModel(nn.Module):

#     def __init__(
#         self,
#         config: LlamaConfig,
#         quant_config: Optional[QuantizationConfig] = None,
#         lora_config: Optional[LoRAConfig] = None,
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.padding_idx = config.pad_token_id
#         lora_vocab = (lora_config.lora_extra_vocab_size *
#                       (lora_config.max_loras or 1)) if lora_config else 0
#         self.vocab_size = config.vocab_size + lora_vocab
#         self.org_vocab_size = config.vocab_size
#         self.embed_tokens = VocabParallelEmbedding(
#             self.vocab_size,
#             config.hidden_size,
#             org_num_embeddings=config.vocab_size,
#         )
#         self.layers = nn.ModuleList([
#             LlamaDecoderLayer(config, quant_config)
#             for _ in range(config.num_hidden_layers)
#         ])
#         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.embed_tokens(input_ids)

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor],
#         positions: torch.Tensor,
#         kv_caches: List[torch.Tensor],
#         attn_metadata: AttentionMetadata,
#         inputs_embeds: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if inputs_embeds is not None:
#             hidden_states = inputs_embeds
#         else:
#             hidden_states = self.get_input_embeddings(input_ids)
#         residual = None
#         for i in range(len(self.layers)):
#             layer = self.layers[i]
#             hidden_states, residual = layer(
#                 positions,
#                 hidden_states,
#                 kv_caches[i],
#                 attn_metadata,
#                 residual,
#             )
#         hidden_states, _ = self.norm(hidden_states, residual)
#         return hidden_states
    
class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        # 创建层时传递层索引
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i, quant_config=quant_config)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        
        residual = None
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            
            # 计算当前层应该使用的缓存索引 (每两层共享一个缓存)
            cache_idx = i // 2
            
            # 使用正确的缓存索引，并传递上一层的hidden states
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[cache_idx],  # 使用计算出的cache_idx
                attn_metadata,
                residual,
            )
            
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, quant_config, lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")

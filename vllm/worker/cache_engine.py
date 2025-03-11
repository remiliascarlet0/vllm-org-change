"""CacheEngine class for managing the KV cache."""
from typing import Dict, List

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
#================================================================================================

class HiddenStateCacheEngine(CacheEngine):
    """管理基于hidden states的缓存引擎，以减少内存消耗。
    
    通过存储模型的中间态(hidden states)而不是KV缓存，能够减少约50%的内存使用。
    每两层共享一个缓存，进一步降低内存需求。
    """

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """分配hidden states缓存空间。
        
        Args:
            num_blocks: 区块数量
            device: 设备类型("cpu"或"cuda")
            
        Returns:
            hidden_cache: 每两层共享的hidden states缓存列表
        """
        if num_blocks == 0:
            return []
            
        # 获取hidden_dim
        hidden_dim = self.model_config.get_hidden_size()
        
        # 使用与原始VLLM一致的块结构
        # 注意：这里的形状仍然遵循KV缓存的形状，便于与PagedAttention兼容
        kv_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        
        logger.info(f"Allocating hidden states cache with shape {kv_shape} on {device}")
        
        
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        
        # 每两层共享一个缓存
        num_caches = (self.num_layers + 1) // 2  # 向上取整，确保奇数层情况也正确处理
        print(f'kv_shape in engine: {kv_shape}')
        print(f'num_caches in engine: {num_caches}')
        hidden_cache: List[torch.Tensor] = []
        for _ in range(num_caches):
            hidden_cache.append(
                torch.empty(kv_shape,
                          dtype=self.dtype,
                          pin_memory=pin_memory,
                          device=device))
            
        return hidden_cache

    @staticmethod 
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        """计算hidden states缓存块大小
        
        与原始KV缓存相比，每两层共享一个缓存，因此理论上内存使用减少约50%。
        
        Args:
            cache_config: 缓存配置
            model_config: 模型配置 
            parallel_config: 并行配置

        Returns:
            block_size: 以字节为单位的块大小
        """
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)
        num_caches = (num_layers + 1) // 2  # 向上取整，确保奇数层情况也正确处理
        
        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_caches * (key_cache_block + value_cache_block)
        
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype] 
        dtype_size = _get_dtype_size(dtype)
        
        return dtype_size * total
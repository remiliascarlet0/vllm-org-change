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
# class HiddenStateCacheEngine(CacheEngine):
#     def _allocate_kv_cache(
#         self,
#         num_blocks: int,
#         device: str,
#     ) -> List[torch.Tensor]:
#         """分配 hidden states 缓存空间。
        
#         每个区块存储 block_size 个 token 的 hidden states。
#         每个 token 的 hidden state 维度是 hidden_dim。
#         """
#         if num_blocks == 0:
#             return []
            
#         # 获取隐藏层维度
#         hidden_dim = self.model_config.get_hidden_size()
        
#         # 形状应该是: (num_blocks, block_size, hidden_dim)
#         # - num_blocks: 区块数量
#         # - block_size: 每个区块存储的 token 数量
#         # - hidden_dim: 每个 token 的 hidden state 维度
#         shape = (num_blocks, self.block_size, hidden_dim)
        
#         pin_memory = is_pin_memory_available() if device == "cpu" else False
        
#         hidden_cache: List[torch.Tensor] = []
#         for _ in range(self.num_layers):
#             hidden_cache.append(
#                 torch.empty(shape,
#                           dtype=self.dtype,
#                           pin_memory=pin_memory,
#                           device=device))
#         return hidden_cache

#     @staticmethod
#     def get_cache_block_size(
#         cache_config: CacheConfig,
#         model_config: ModelConfig,
#         parallel_config: ParallelConfig,
#     ) -> int:
#         """计算 hidden states 缓存块大小。
        
#         这里只需要存储 hidden states，不需要存储 K 和 V。
#         """
#         hidden_size = model_config.get_hidden_size()
#         num_layers = model_config.get_num_layers(parallel_config)
        
#         # 每个区块存储 block_size 个 token 的 hidden states
#         hidden_block = cache_config.block_size * hidden_size
#         total = num_layers * hidden_block

#         if cache_config.cache_dtype == "auto":
#             dtype = model_config.dtype
#         else:
#             dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
#         dtype_size = _get_dtype_size(dtype)
        
#         return dtype_size * total
class HiddenStateCacheEngine(CacheEngine):
    """管理基于hidden states的缓存引擎，以减少内存消耗。
    
    通过存储模型的中间态(hidden states)而不是KV缓存，能够减少约50%的内存使用。
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
            hidden_cache: 每层的hidden states缓存列表
        """
        if num_blocks == 0:
            return []
            
        # 获取hidden_dim，对应于transformer的隐藏层大小
        hidden_dim = self.model_config.get_hidden_size()
        
        # 使用与原始VLLM一致的块结构: (num_blocks, block_size, hidden_dim)
        # 这样可以保持与PagedAttention机制的兼容性
        shape = (num_blocks, self.block_size, hidden_dim)
        
        logger.info(f"Allocating hidden states cache with shape {shape} on {device}")
        
        # 对CPU缓存使用pin_memory可加速GPU-CPU传输
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        
        hidden_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            hidden_cache.append(
                torch.empty(shape,
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
        
        与原始KV缓存相比，hidden states只需存储一个张量而非K和V两个，
        因此理论上内存使用减少了一半。
        
        Args:
            cache_config: 缓存配置
            model_config: 模型配置 
            parallel_config: 并行配置

        Returns:
            block_size: 以字节为单位的块大小
        """
        hidden_size = model_config.get_hidden_size()
        num_layers = model_config.get_num_layers(parallel_config)
        
        # 计算每个块存储hidden states所需的内存
        # 与KV缓存相比，内存减少一半
        hidden_block = cache_config.block_size * hidden_size
        total = num_layers * hidden_block

        # 获取数据类型大小
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype] 
        dtype_size = _get_dtype_size(dtype)
        
        return dtype_size * total
        
    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        """从CPU缓存交换hidden states到GPU缓存。
        
        与原始方法相比，这里只处理hidden states而不是K和V。
        
        Args:
            src_to_dst: 源块到目标块的映射
        """
        for i in range(self.num_layers):
            # 使用原始的swap_blocks方法，但只处理一个tensor而不是两个
            ops.swap_blocks(self.cpu_cache[i], self.gpu_cache[i], src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        """从GPU缓存交换hidden states到CPU缓存。
        
        Args:
            src_to_dst: 源块到目标块的映射
        """
        for i in range(self.num_layers):
            ops.swap_blocks(self.gpu_cache[i], self.cpu_cache[i], src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        """在GPU内复制hidden states块。
        
        Args:
            src_to_dsts: 源块到目标块列表的映射
        """
        # 直接调用底层操作来复制hidden states块
        ops.copy_blocks(self.gpu_cache, src_to_dsts)
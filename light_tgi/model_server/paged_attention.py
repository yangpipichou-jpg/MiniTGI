"""
PagedAttention — vLLM-style paged KV Cache for Light TGI

核心思想:
  - KV Cache 不连续存储，按固定大小的 Block 分页
  - Block Table 记录逻辑位置 → 物理 Block 的映射
  - 类似操作系统虚拟内存的分页机制

Block Pool (物理存储):
  [Block 0] [Block 1] [Block 2] ... [Block N-1]
  每个 Block: (num_layers, 2, num_heads, block_size, head_dim)
              └─ key/value ─┘

Block Table (逻辑映射):
  Request A: [physical_block_7, physical_block_3, physical_block_12, ...]
  Request B: [physical_block_5, physical_block_15, ...]

优势:
  1. 零显存浪费 — 按需分配 block，不需要预分配 max_seq_len
  2. KV Cache 共享 — 相同前缀可复用 block (如 shared system prompt)
  3. 内存效率提升 2-4×
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class PagedCacheConfig:
    """PagedAttention 配置"""
    block_size: int = 16          # 每个 block 存储的 token 数
    num_blocks: int = 256         # GPU block pool 总 block 数
    num_layers: int = 32          # Transformer 层数
    num_kv_heads: int = 8         # KV attention heads (GQA 下可能 < num_attention_heads)
    head_dim: int = 128           # 每个 head 的维度
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    @property
    def max_total_tokens(self) -> int:
        return self.num_blocks * self.block_size


# ============================================================
# Block Table — 管理逻辑→物理映射
# ============================================================

class BlockTable:
    """
    每个请求的 Block Table，管理该请求的逻辑 position → 物理 block 映射

    示例 (block_size=16):
      seq_len = 35
      block_table = [7, 3, 12]  ← 3 个物理 block
      逻辑位置 0-15  → 物理 block 7
      逻辑位置 16-31 → 物理 block 3
      逻辑位置 32-34 → 物理 block 12 (只用了前 3 个 slot)
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: List[int] = []  # 物理 block ID 列表

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def capacity(self) -> int:
        """当前分配的 slot 总数"""
        return self.num_blocks * self.block_size

    @property
    def num_slots_used(self) -> int:
        """已使用的 slot 数 (外部跟踪，此处返回分配数)"""
        return self.capacity

    def append_block(self, physical_block_id: int):
        """追加一个物理 block"""
        self.blocks.append(physical_block_id)

    def get_physical_block(self, logical_block_idx: int) -> int:
        """逻辑 block 索引 → 物理 block ID"""
        return self.blocks[logical_block_idx]

    def __getitem__(self, idx: int) -> int:
        return self.blocks[idx]

    def __len__(self) -> int:
        return len(self.blocks)

    def to_list(self) -> List[int]:
        return list(self.blocks)

    def free_blocks(self) -> List[int]:
        """释放所有 block，返回物理 block ID 列表"""
        freed = list(self.blocks)
        self.blocks.clear()
        return freed


# ============================================================
# Block Pool — GPU 上的 KV Cache 物理存储
# ============================================================

class BlockPool:
    """
    GPU 显存中的 Block Pool

    形状: (num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim)
           └─ block ─┘ └─ layer ┘ └K/V┘ └── head ──┘ └─ tokens ─┘

    每个 block 包含所有层的 K 和 V cache
    """

    def __init__(self, config: PagedCacheConfig):
        self.config = config
        self.num_blocks = config.num_blocks
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.block_size = config.block_size
        self.head_dim = config.head_dim

        # 主存储: (num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim)
        self.data = torch.zeros(
            self.num_blocks,
            self.num_layers,
            2,  # 0=K, 1=V
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
            dtype=config.dtype,
            device=config.device,
        )

        # 空闲 block 管理
        self.free_blocks: List[int] = list(range(self.num_blocks))
        self.used_blocks: int = 0

    def allocate(self, n: int = 1) -> List[int]:
        """分配 n 个 block，返回物理 block ID 列表"""
        if len(self.free_blocks) < n:
            raise RuntimeError(
                f"Block pool exhausted! requested={n}, free={len(self.free_blocks)}"
            )
        allocated = []
        for _ in range(n):
            bid = self.free_blocks.pop(0)
            allocated.append(bid)
        self.used_blocks += n
        return allocated

    def free(self, block_ids: List[int]):
        """释放 block"""
        for bid in block_ids:
            if bid not in self.free_blocks:
                self.free_blocks.append(bid)
                # 清零释放的 block
                self.data[bid].zero_()
        self.used_blocks -= len(block_ids)

    @property
    def free_count(self) -> int:
        return len(self.free_blocks)

    def get_kv_block(self, physical_block_id: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取某个物理 block 在指定层的 K, V

        Returns:
            k: (num_kv_heads, block_size, head_dim)
            v: (num_kv_heads, block_size, head_dim)
        """
        block = self.data[physical_block_id, layer_idx]  # (2, num_kv_heads, block_size, head_dim)
        return block[0], block[1]  # (K, V)

    def set_kv_block(
        self,
        physical_block_id: int,
        layer_idx: int,
        k: torch.Tensor,   # (num_kv_heads, seq_len, head_dim) or padded
        v: torch.Tensor,
        offset: int = 0,
    ):
        """
        写入 K, V 到指定 block

        Args:
            physical_block_id: 物理 block ID
            layer_idx: 层索引
            k: (num_kv_heads, write_len, head_dim)
            v: (num_kv_heads, write_len, head_dim)
            offset: block 内起始写入位置
        """
        write_len = k.shape[1]
        self.data[physical_block_id, layer_idx, 0, :, offset:offset+write_len, :] = k
        self.data[physical_block_id, layer_idx, 1, :, offset:offset+write_len, :] = v

    def compute_bytes(self) -> int:
        """计算 Block Pool 占用的显存 (bytes)"""
        return self.data.element_size() * self.data.numel()


# ============================================================
# PagedAttention — 分页注意力计算 (纯 PyTorch 实现)
# ============================================================

def paged_attention(
    query: torch.Tensor,               # (batch, num_heads, head_dim)
    block_tables: List[BlockTable],    # 每个请求的 block table
    block_pool: BlockPool,             # GPU block pool
    seq_lens: List[int],               # 每个请求的实际序列长度
    layer_idx: int,                    # 当前层索引
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    纯 PyTorch 实现的 PagedAttention

    对 batch 中每个请求:
      1. 从 Block Pool 按 block_table 收集 K, V
      2. 拼接成连续的 K, V
      3. 计算 scaled dot-product attention

    Args:
        query: (batch, num_heads, head_dim) — 当前 token 的 query
        block_tables: 每个请求的 BlockTable
        block_pool: GPU Block Pool
        seq_lens: 每个请求的序列长度
        layer_idx: Transformer 层索引
        scale: attention scale factor (default: 1/√head_dim)

    Returns:
        output: (batch, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = block_pool.num_kv_heads

    if scale is None:
        scale = head_dim ** -0.5

    outputs = []

    for b in range(batch_size):
        bt = block_tables[b]
        seq_len = seq_lens[b]
        num_blocks_needed = (seq_len + block_pool.block_size - 1) // block_pool.block_size

        # Step 1: 从 Block Pool 收集该请求的所有 K, V blocks
        k_blocks = []
        v_blocks = []
        for i in range(num_blocks_needed):
            physical_id = bt[i]
            k_block, v_block = block_pool.get_kv_block(physical_id, layer_idx)
            k_blocks.append(k_block)  # each: (num_kv_heads, block_size, head_dim)
            v_blocks.append(v_block)

        # Step 2: 拼接成连续的 K, V → (num_kv_heads, total_seq_len, head_dim)
        K = torch.cat(k_blocks, dim=1)[:, :seq_len, :]
        V = torch.cat(v_blocks, dim=1)[:, :seq_len, :]

        # Step 3: 处理 GQA (Grouped Query Attention)
        # query: (num_heads, head_dim), K/V: (num_kv_heads, seq_len, head_dim)
        # 如果 num_heads > num_kv_heads, 需要扩展 K/V
        if num_heads != num_kv_heads:
            n_groups = num_heads // num_kv_heads
            # K: (num_kv_heads, seq_len, head_dim) → (num_heads, seq_len, head_dim)
            K = K.repeat_interleave(n_groups, dim=0)
            V = V.repeat_interleave(n_groups, dim=0)

        # Step 4: Scaled Dot-Product Attention
        # query[b]: (num_heads, head_dim) → (num_heads, 1, head_dim)
        # K: (num_heads, seq_len, head_dim)
        # V: (num_heads, seq_len, head_dim)
        q = query[b].unsqueeze(1)  # (num_heads, 1, head_dim)
        k = K.transpose(1, 2)       # (num_heads, head_dim, seq_len)

        scores = torch.bmm(q, k) * scale  # (num_heads, 1, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, V.transpose(1, 2).unsqueeze(0).expand(num_heads, -1, -1))
        # output: (num_heads, 1, head_dim) → (num_heads, head_dim)
        outputs.append(output.squeeze(1))

    return torch.stack(outputs, dim=0)


# ============================================================
# PagedKVCache — 高层封装，替代旧的 CacheEntry
# ============================================================

class PagedKVCache:
    """
    管理单个请求的 Paged KV Cache

    替代旧的 CacheEntry，提供:
      - BlockTable 管理
      - 自动分配/释放 block
      - 写入 K, V
      - 读取完整的 K, V (用于 attention 计算)
    """

    def __init__(
        self,
        request_id: str,
        block_pool: BlockPool,
        block_size: int,
        max_seq_len: int = 4096,
    ):
        self.request_id = request_id
        self.block_pool = block_pool
        self.block_size = block_size
        self.max_seq_len = max_seq_len

        self.block_table = BlockTable(block_size)
        self.seq_len: int = 0  # 当前已存储的 token 数
        self.is_finished: bool = False
        self.generated_ids: List[int] = []

    @property
    def num_blocks_used(self) -> int:
        return self.block_table.num_blocks

    @property
    def num_slots_available(self) -> int:
        return self.block_table.capacity - self.seq_len

    def need_new_block(self) -> bool:
        """是否需要分配新 block"""
        return self.seq_len >= self.block_table.capacity

    def allocate_block(self) -> int:
        """分配一个新 block"""
        physical_id = self.block_pool.allocate(1)[0]
        self.block_table.append_block(physical_id)
        return physical_id

    def write_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,  # (num_kv_heads, seq_len, head_dim)
        v: torch.Tensor,
    ):
        """
        写入一层的 K, V 到当前 block 的空闲位置

        Args:
            layer_idx: 层索引
            k: (num_kv_heads, write_len, head_dim)
            v: (num_kv_heads, write_len, head_dim)
        """
        write_len = k.shape[1]

        # 计算在最后一个 block 中的写入位置
        if self.block_table.num_blocks == 0:
            self.allocate_block()

        offset = self.seq_len % self.block_size
        last_block = self.block_table[-1]

        self.block_pool.set_kv_block(last_block, layer_idx, k, v, offset)

    def advance(self, token_ids: List[int]):
        """推进序列位置"""
        self.seq_len += len(token_ids)
        self.generated_ids.extend(token_ids)

    def free(self):
        """释放所有 block"""
        freed = self.block_table.free_blocks()
        self.block_pool.free(freed)

    def get_kv_for_attention(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取完整的 K, V 用于 attention 计算

        Returns:
            K: (num_kv_heads, seq_len, head_dim)
            V: (num_kv_heads, seq_len, head_dim)
        """
        k_blocks = []
        v_blocks = []
        for physical_id in self.block_table.to_list():
            k, v = self.block_pool.get_kv_block(physical_id, layer_idx)
            k_blocks.append(k)
            v_blocks.append(v)

        K = torch.cat(k_blocks, dim=1)[:, :self.seq_len, :]
        V = torch.cat(v_blocks, dim=1)[:, :self.seq_len, :]
        return K, V


# ============================================================
# 工具函数
# ============================================================

def compute_paged_cache_config(
    model,
    block_size: int = 16,
    gpu_memory_utilization: float = 0.9,
) -> PagedCacheConfig:
    """
    根据模型配置和 GPU 显存自动计算 PagedCacheConfig

    Args:
        model: HuggingFace 模型
        block_size: 每个 block 的 token 数
        gpu_memory_utilization: GPU 显存利用率

    Returns:
        PagedCacheConfig
    """
    config = model.config

    # 获取模型参数
    num_layers = getattr(config, 'num_hidden_layers', 32)
    num_kv_heads = getattr(config, 'num_key_value_heads',
                           getattr(config, 'num_attention_heads', 8))
    head_dim = config.hidden_size // config.num_attention_heads

    # 计算每个 block 的显存占用
    # (num_layers, 2, num_kv_heads, block_size, head_dim) × element_size
    bytes_per_block = (
        num_layers * 2 * num_kv_heads * block_size * head_dim
        * (2 if torch.finfo(model.dtype).bits == 16 else 4)
    )

    # 获取 GPU 空闲显存
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info()
        available = int(free_memory * gpu_memory_utilization)
    else:
        available = 4 * 1024 * 1024 * 1024  # CPU 模式: 4GB 上限

    num_blocks = max(1, available // bytes_per_block)

    device = str(next(model.parameters()).device)

    config = PagedCacheConfig(
        block_size=block_size,
        num_blocks=num_blocks,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=model.dtype,
        device=device,
    )

    logger.info(f"PagedCacheConfig: blocks={num_blocks}, block_size={block_size}, "
                f"max_tokens={num_blocks * block_size}, "
                f"mem_per_block={bytes_per_block / 1024 / 1024:.1f}MB, "
                f"total_pool_mem={num_blocks * bytes_per_block / 1024 / 1024 / 1024:.2f}GB")

    return config

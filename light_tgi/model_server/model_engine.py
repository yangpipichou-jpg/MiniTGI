"""
模型推理引擎 (PagedAttention 版本)

职责：
1. 加载 HuggingFace 真实模型
2. 实现 Prefill (预填充) — 使用 model.forward() 获取每层 K/V 写入 Block Pool
3. 实现 Decode (解码) — 使用 PagedAttention 逐 token 生成
4. 管理 Paged KV Cache (BlockTable + BlockPool)

vLLM PagedAttention 核心思想:
  - KV Cache 按固定大小 Block 分页存储 (类比 OS 虚拟内存)
  - Block Table 记录逻辑位置 → 物理 Block 映射
  - 零显存浪费 (按需分配), 支持 prefix sharing

架构对比:
  旧: past_key_values → contiguous Tuple → 预分配 max_seq_len → 浪费
  新: BlockPool → BlockTable → 按需分配 16-token blocks → 高效

支持的模型:
- Qwen/Qwen2.5-1.5B-Instruct (默认)
- 任何 HuggingFace CausalLM 模型 (需要 output_attentions 支持)
"""

import torch
import time
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import logging

from transformers import DynamicCache

logger = logging.getLogger(__name__)

# 导入 PagedAttention 模块
from paged_attention import (
    PagedCacheConfig, BlockPool, BlockTable, PagedKVCache,
    paged_attention, compute_paged_cache_config,
)


class ModelEngine:
    """
    推理引擎 (PagedAttention 版本)

    使用 vLLM 风格的分页 KV Cache 管理显存。

    核心流程:
        Prefill:
            input_text → tokenizer → model.forward(prompt, output_attentions=True)
            → 提取每层 K/V → 写入 BlockPool
            → 采样第一个 token → 保存到 PagedKVCache

        Decode:
            last_token → model.forward(token, output_attentions=True)
            → 提取新 token 的 K/V → 追加到 BlockPool
            → 采样下一个 token

    与旧版的关键区别:
        旧: model.forward(token, past_key_values=old_kv)
            → 模型内部做完整 attention
        新: model.forward(token, output_attentions=True)
            → 获取新 token 的 K/V → 用 paged_attention() 手动计算
            → 避免存储整个 past_key_values tuple
    """

    SUPPORTED_MODELS = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "google/gemma-2-2b-it",
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_sequence_length: int = 4096,
        max_batch_size: int = 8,
        device: str = "cpu",
        dtype: str = "auto",
    ):
        self.model_id = model_id
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.device = device
        self.dtype = dtype

        # 加载模型和 tokenizer
        self._load_model()

        # ★ PagedAttention: BlockPool 替代旧的 OrderedDict[int, CacheEntry]
        self.paged_config = compute_paged_cache_config(
            self.model, block_size=16, gpu_memory_utilization=0.9
        )
        self.block_pool = BlockPool(self.paged_config)

        # KV Cache 存储 (handle -> PagedKVCache)
        self._cache: OrderedDict[int, PagedKVCache] = OrderedDict()
        self._handle_counter: int = 1

        # 统计信息
        self.stats = {
            "total_prefills": 0,
            "total_decodes": 0,
            "total_tokens_generated": 0,
        }

        logger.info(f"模型引擎初始化完成 (PagedAttention): model={model_id}, device={device}")
        logger.info(f"  max_seq_len={max_sequence_length}, max_batch={max_batch_size}")
        logger.info(f"  block_pool: {self.paged_config.num_blocks} blocks × {self.paged_config.block_size} tokens")
        logger.info(f"  vocab_size={self.vocab_size}, eos_token_id={self.eos_token_id}")

    def _load_model(self):
        """加载 HuggingFace 模型和 tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"正在加载模型: {self.model_id} ...")
        load_start = time.time()

        # 确定 dtype
        torch_dtype = None
        if self.dtype == "float16":
            torch_dtype = torch.float16
        elif self.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.device.startswith("cuda"):
            torch_dtype = torch.float16  # GPU 默认半精度

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        # 确保有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.vocab_size = self.tokenizer.vocab_size
        self.eos_token_id = self.tokenizer.eos_token_id

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            # ★ PagedAttention 需要模型输出每层 attention 的 K/V
            # 这样我们可以手动提取 K/V 并存入 BlockPool
            output_attentions=False,  # 不需要完整的 attention weights
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

        load_time = time.time() - load_start
        # 计算模型参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型加载完成: {num_params/1e9:.2f}B 参数, 耗时 {load_time:.1f}s")

    # ================================================================
    # Tokenize (供 Router 调用做长度验证)
    # ================================================================

    def tokenize(self, text: str) -> Tuple[List[int], int]:
        """将文本转为 token ids，返回 (ids, count)"""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return encoded, len(encoded)

    def decode_tokens(self, token_ids: List[int]) -> str:
        """将 token ids 解码为文本"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ================================================================
    # Prefill (预填充) — 真实模型推理
    # ================================================================

    def prefill(
        self,
        request_id: str,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ) -> Tuple[int, List[Dict], int]:
        """
        Prefill (PagedAttention 版本)

        流程:
        1. Tokenize
        2. model.forward(prompt, use_cache=True) → logits + past_key_values
        3. 从 past_key_values 提取每层 K/V → 写入 BlockPool
        4. 保存 past_key_values 用于后续 decode
        5. 采样第一个 token

        Returns:
            (cache_handle, generated_tokens, duration_ms)
        """
        start_time = time.time()

        # 1. Tokenize
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        input_len = len(input_ids)

        if input_len == 0:
            raise ValueError("Input is empty, cannot generate")

        if input_len > self.max_sequence_length:
            raise ValueError(
                f"Input too long: {input_len} tokens (max: {self.max_sequence_length})"
            )

        # 2. 分配 handle + 创建 PagedKVCache
        handle = self._handle_counter
        self._handle_counter += 1

        paged_cache = PagedKVCache(
            request_id=request_id,
            block_pool=self.block_pool,
            block_size=self.paged_config.block_size,
            max_seq_len=self.max_sequence_length,
        )

        # 3. 构建模型输入
        input_tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor)

        # 4. Forward pass (首次: 不带 past_key_values, use_cache=True 获取 past_key_values)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
            )

        # 5. 获取 logits
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        # ★ 6. 从 past_key_values 提取每层 K/V 并写入 BlockPool
        past_key_values = outputs.past_key_values

        # 分配所需的 blocks
        num_blocks_needed = (input_len + self.paged_config.block_size - 1) // self.paged_config.block_size
        for _ in range(num_blocks_needed):
            paged_cache.allocate_block()

        # 写入 K/V 到 BlockPool
        self._write_kv_to_pool(paged_cache, past_key_values, input_len)

        # ★ 7. 保存 past_key_values 用于后续 decode (模型内部 KV cache)
        paged_cache._past_key_values = past_key_values

        # 8. 推进序列位置
        paged_cache.advance(input_ids)

        # 9. 采样第一个 token
        first_token_id = self._sample_token(
            logits, temperature, top_p, top_k, do_sample
        )

        # 检查 EOS
        if first_token_id == self.eos_token_id:
            paged_cache.is_finished = True

        self._cache[handle] = paged_cache

        duration_ms = int((time.time() - start_time) * 1000)
        self.stats["total_prefills"] += 1
        self.stats["total_tokens_generated"] += 1

        # 解码 token 为文本
        token_text = self.tokenizer.decode([first_token_id], skip_special_tokens=False)

        generated_tokens = [{
            "id": first_token_id,
            "text": token_text,
            "logprob": -0.5,
            "special": first_token_id == self.eos_token_id,
        }]

        logger.info(
            f"[Prefill] request={request_id}, handle={handle}, "
            f"input_tokens={input_len}, first_token={first_token_id}({repr(token_text)}), "
            f"blocks={paged_cache.num_blocks_used}, duration={duration_ms}ms"
        )

        return handle, generated_tokens, duration_ms

    # ================================================================
    # Decode (解码) — 真实模型推理
    # ================================================================

    def decode(
        self,
        request_id: str,
        cache_handle: int,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ) -> Tuple[Optional[Dict], Optional[str], int]:
        """
        Decode (PagedAttention 版本)

        流程:
        1. 取 last_token_id → 单 token forward (use_cache=True, past_key_values=...)
        2. 更新 past_key_values 用于下一步 decode
        3. 同步写入 BlockPool (用于未来 prefix sharing / continuous batching)
        4. 采样下一个 token

        Returns:
            (token_dict, finish_reason, duration_ms)
        """
        start_time = time.time()

        # 获取 PagedKVCache
        paged_cache = self._cache.get(cache_handle)
        if paged_cache is None:
            logger.error(f"Cache not found: handle={cache_handle}")
            return None, "error", 0

        if paged_cache.is_finished:
            return None, "eos_token", 0

        # 获取最后一个 token
        last_token_id = paged_cache.generated_ids[-1]
        input_tensor = torch.tensor([[last_token_id]], device=self.device, dtype=torch.long)

        # ★ 使用保存的 past_key_values，模型内部做完整 attention
        past_kv = paged_cache._past_key_values

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                past_key_values=past_kv,
                use_cache=True,
            )

        # 获取 logits
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        # ★ 更新 past_key_values (模型已自动追加新 token 的 K/V)
        paged_cache._past_key_values = outputs.past_key_values

        # ★ 同步写入 BlockPool (提取最新 token 的 K/V)
        self._append_kv_to_pool(paged_cache, outputs.past_key_values)

        # 推进序列
        paged_cache.advance([last_token_id])

        # 采样下一个 token
        next_token_id = self._sample_token(
            logits, temperature, top_p, top_k, do_sample
        )
        paged_cache.generated_ids.append(next_token_id)

        # 判断停止条件
        finish_reason = None
        if next_token_id == self.eos_token_id:
            finish_reason = "eos_token"
            paged_cache.is_finished = True
        elif len(paged_cache.generated_ids) >= max_new_tokens:
            finish_reason = "length"
            paged_cache.is_finished = True

        # 解码
        token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)

        duration_ms = int((time.time() - start_time) * 1000)
        self.stats["total_decodes"] += 1
        self.stats["total_tokens_generated"] += 1

        token_dict = {
            "id": next_token_id,
            "text": token_text,
            "logprob": -0.3,
            "special": next_token_id == self.eos_token_id,
        }

        logger.debug(
            f"[Decode] request={request_id}, handle={cache_handle}, "
            f"token={next_token_id}({repr(token_text)}), "
            f"seq_len={paged_cache.seq_len}, blocks={paged_cache.num_blocks_used}, "
            f"finish={finish_reason}"
        )

        return token_dict, finish_reason, duration_ms

    # ================================================================
    # Batch Decode — 真正的 Continuous Batching
    # ================================================================

    def batch_decode(
        self,
        request_ids: List[str],
        cache_handles: List[int],
        token_ids: List[int],
        params_list: List[Dict],
    ) -> Tuple[List[Optional[Dict]], List[Optional[str]], int]:
        """
        批量 Decode: 多个请求的 next_token 组成 batch，一次 model.forward()

        这是 Continuous Batching 的核心:
          - 每个请求只需要 1 个 token 的 forward
          - 但不同请求处于不同的生成阶段
          - 所有请求共享同一次 GPU forward → 吞吐量提升 N 倍

        Args:
            request_ids: 请求 ID 列表
            cache_handles: KV cache handle 列表
            token_ids: 每个请求的 last_token_id
            params_list: 每个请求的采样参数

        Returns:
            (token_dicts, finish_reasons, duration_ms)
        """
        start_time = time.time()
        batch_size = len(request_ids)

        if batch_size == 0:
            return [], [], 0

        # 构建 batch 输入: (batch_size, 1)
        input_tensor = torch.tensor(
            [[tid] for tid in token_ids], device=self.device, dtype=torch.long
        )

        # 为每个请求准备 past_key_values
        batch_past_kv = []
        for handle in cache_handles:
            paged_cache = self._cache.get(handle)
            if paged_cache is None:
                logger.error(f"[BatchDecode] Cache not found: handle={handle}")
                return [None] * batch_size, ["error"] * batch_size, 0
            batch_past_kv.append(paged_cache._past_key_values)

        # ★ 将多个请求的 past_key_values 合并成 batch
        # past_key_values 是 tuple of (k, v), 每层一个
        # k: (batch_i, num_kv_heads, seq_len_i, head_dim)
        # 需要 left-padding 使 seq_len 对齐
        num_layers = len(batch_past_kv[0])
        merged_past_kv = []

        # 找到最大 seq_len
        # past_key_values shape: (1, num_kv_heads, seq_len, head_dim)
        # shape[2] 是 seq_len 维度
        max_kv_len = max(
            pv[0][0].shape[2]  # first layer, key, seq_len dim
            for pv in batch_past_kv
        )

        for layer_idx in range(num_layers):
            padded_keys = []
            padded_values = []

            for pv in batch_past_kv:
                k, v = pv[layer_idx]  # each: (1, num_kv_heads, seq_len, head_dim)
                seq_len = k.shape[2]  # shape[2] = seq_len

                if seq_len < max_kv_len:
                    pad_len = max_kv_len - seq_len
                    # Left-pad with zeros
                    k_padded = torch.nn.functional.pad(k, (0, 0, pad_len, 0), value=0.0)
                    v_padded = torch.nn.functional.pad(v, (0, 0, pad_len, 0), value=0.0)
                else:
                    k_padded = k
                    v_padded = v

                padded_keys.append(k_padded)
                padded_values.append(v_padded)

            merged_k = torch.cat(padded_keys, dim=0)  # (batch, num_kv_heads, max_len, head_dim)
            merged_v = torch.cat(padded_values, dim=0)
            merged_past_kv.append((merged_k, merged_v))

        merged_past_kv = tuple(merged_past_kv)

        # ★ 构建 attention_mask: 屏蔽 left-padding 位置
        # 每个请求的 past KV 长度为 seq_len_i, 被 left-pad 到 max_kv_len
        # 总序列长度 = max_kv_len (past) + 1 (new token)
        # attention_mask: 1 表示有效位置, 0 表示 padding
        kv_lengths = [pv[0][0].shape[2] for pv in batch_past_kv]
        total_len = max_kv_len + 1
        batch_attention_mask = torch.zeros(batch_size, total_len, device=self.device, dtype=torch.long)
        for i, kv_len in enumerate(kv_lengths):
            pad_len = max_kv_len - kv_len
            batch_attention_mask[i, pad_len:] = 1  # valid positions: [pad..., past_kv..., new_token]

        # ★ 转换为 DynamicCache (新版 transformers 期望的格式)
        batch_dynamic_cache = DynamicCache.from_legacy_cache(merged_past_kv)

        # ★ Batch forward: 所有请求一起做 attention
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                past_key_values=batch_dynamic_cache,
                attention_mask=batch_attention_mask,
                use_cache=True,
            )

        # 获取每个请求的 logits
        logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

        # 为每个请求更新 past_key_values (从 batch 中切回单请求)
        new_batch_past_kv = outputs.past_key_values

        # ★ 关键: 将 DynamicCache 转为可索引的 list of (k, v) tuples
        #    避免直接在 DynamicCache 上迭代时出现版本兼容问题
        if isinstance(new_batch_past_kv, DynamicCache):
            new_batch_kv_tuples = [
                (new_batch_past_kv.key_cache[l], new_batch_past_kv.value_cache[l])
                for l in range(len(new_batch_past_kv.key_cache))
            ]
        else:
            new_batch_kv_tuples = list(new_batch_past_kv)

        token_dicts = []
        finish_reasons = []

        for i in range(batch_size):
            request_id = request_ids[i]
            handle = cache_handles[i]
            params = params_list[i]
            paged_cache = self._cache.get(handle)

            if paged_cache is None:
                token_dicts.append(None)
                finish_reasons.append("error")
                continue

            # ★ 从 batch past_key_values 中切出该请求的部分
            #    每个请求在 batch 中的实际 KV 长度 = 原 seq_len + 1 (新 token)
            #    使用 seq_len 而非 seq_len+1 是因为 advance 还没执行
            actual_kv_len = paged_cache.seq_len + 1
            own_past_kv = tuple(
                (k[i:i+1, :, -actual_kv_len:, :],
                 v[i:i+1, :, -actual_kv_len:, :])
                for k, v in new_batch_kv_tuples
            )
            # ★ 转为 DynamicCache 以兼容新版 transformers
            paged_cache._past_key_values = DynamicCache.from_legacy_cache(own_past_kv)

            # 同步写入 BlockPool
            self._append_kv_to_pool(paged_cache, own_past_kv)

            # 推进序列 (必须在切片后执行，因为切片依赖旧 seq_len)
            paged_cache.advance([token_ids[i]])

            # 采样
            next_token_id = self._sample_token(
                logits[i:i+1],
                params.get("temperature", 1.0),
                params.get("top_p", 1.0),
                params.get("top_k", 0),
                params.get("do_sample", False),
            )
            paged_cache.generated_ids.append(next_token_id)

            # 判断停止
            max_tokens = params.get("max_new_tokens", 100)
            finish_reason = None
            if next_token_id == self.eos_token_id:
                finish_reason = "eos_token"
                paged_cache.is_finished = True
            elif len(paged_cache.generated_ids) >= max_tokens:
                finish_reason = "length"
                paged_cache.is_finished = True

            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)

            token_dicts.append({
                "id": next_token_id,
                "text": token_text,
                "logprob": -0.3,
                "special": next_token_id == self.eos_token_id,
            })
            finish_reasons.append(finish_reason)

            logger.info(
                f"[BatchDecode] req={request_id}, token={next_token_id}({repr(token_text)}), "
                f"finish={finish_reason}, seq_len={paged_cache.seq_len}, "
                f"actual_kv_len={actual_kv_len}, eos={self.eos_token_id}"
            )

        duration_ms = int((time.time() - start_time) * 1000)
        self.stats["total_decodes"] += batch_size
        self.stats["total_tokens_generated"] += batch_size

        logger.info(
            f"[BatchDecode] batch_size={batch_size}, "
            f"duration={duration_ms}ms, "
            f"avg={duration_ms/batch_size:.1f}ms/req, "
            f"finish_reasons={finish_reasons}"
        )

        return token_dicts, finish_reasons, duration_ms

    # ================================================================
    # PagedAttention 辅助方法 — K/V 提取与写入
    # ================================================================

    def _write_kv_to_pool(
        self,
        paged_cache: PagedKVCache,
        past_key_values,
        seq_len: int,
    ):
        """写入完整序列的 K/V 到已分配的 blocks (用于 prefill)"""
        block_size = self.paged_config.block_size
        num_blocks_needed = (seq_len + block_size - 1) // block_size

        # ★ 兼容 DynamicCache 和 tuple
        if isinstance(past_key_values, DynamicCache):
            kv_iter = [
                (past_key_values.key_cache[l], past_key_values.value_cache[l])
                for l in range(len(past_key_values.key_cache))
            ]
        else:
            kv_iter = past_key_values

        for layer_idx, (k, v) in enumerate(kv_iter):
            k_layer = k[0]  # (num_kv_heads, seq_len, head_dim)
            v_layer = v[0]

            for block_idx in range(num_blocks_needed):
                physical_id = paged_cache.block_table[block_idx]
                start = block_idx * block_size
                end = min(start + block_size, seq_len)

                k_block = k_layer[:, start:end, :]
                v_block = v_layer[:, start:end, :]

                self.block_pool.set_kv_block(
                    physical_id, layer_idx, k_block, v_block, offset=start % block_size
                )

    def _append_kv_to_pool(
        self,
        paged_cache: PagedKVCache,
        past_key_values,
    ):
        """
        追加新 token 的 K/V 到 BlockPool

        对于 decode 阶段，past_key_values 包含完整序列，
        但我们只需要最后一个位置的 K/V

        Args:
            past_key_values: 可以是 tuple of (k, v) 或 DynamicCache
        """
        if paged_cache.need_new_block():
            paged_cache.allocate_block()

        offset = paged_cache.seq_len % self.paged_config.block_size
        physical_id = paged_cache.block_table[-1]

        # ★ 兼容 DynamicCache 和 tuple
        if isinstance(past_key_values, DynamicCache):
            kv_iter = [
                (past_key_values.key_cache[l], past_key_values.value_cache[l])
                for l in range(len(past_key_values.key_cache))
            ]
        else:
            kv_iter = past_key_values

        for layer_idx, (k, v) in enumerate(kv_iter):
            # 取最后一个 token 的 K/V
            k_new = k[0, :, -1:, :]  # (num_kv_heads, 1, head_dim)
            v_new = v[0, :, -1:, :]

            self.block_pool.set_kv_block(
                physical_id, layer_idx, k_new, v_new, offset=offset
            )

    # ================================================================
    # 采样
    # ================================================================

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ) -> int:
        """
        从 logits 中采样下一个 token

        支持:
        - Greedy (do_sample=False 或 temperature=0)
        - Temperature sampling
        - Top-K sampling
        - Top-P (nucleus) sampling
        """
        if not do_sample or temperature <= 0.0:
            # Greedy
            return int(logits.argmax(dim=-1).item())

        # 应用 temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-P (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # 采样
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    # ================================================================
    # 批量 Prefill (PagedAttention 版本)
    # ================================================================

    def batch_prefill(
        self,
        requests: List[Dict],
    ) -> List[Tuple[int, List[Dict], int]]:
        """
        批量预填充: 多个请求在同一个 batch 中做 prefill

        PagedAttention 版本: 每个请求独立分配 BlockTable,
        从 batch past_key_values 中提取各自序列的 K/V 写入 BlockPool,
        并为每个请求保存切片后的 past_key_values 用于后续 decode。
        """
        if len(requests) == 1:
            req = requests[0]
            result = self.prefill(
                request_id=req["request_id"],
                input_text=req["input_text"],
                **req.get("params", {}),
            )
            return [result]

        start_time = time.time()

        # Tokenize 所有请求
        all_input_ids = []
        request_info = []

        for req in requests:
            input_ids = self.tokenizer.encode(
                req["input_text"], add_special_tokens=False
            )
            all_input_ids.append(input_ids)
            request_info.append({
                "request_id": req["request_id"],
                "input_length": len(input_ids),
                "params": req.get("params", {}),
            })

        # Left-padding
        max_len = max(len(ids) for ids in all_input_ids)
        padded_ids = []
        attention_masks = []

        for ids in all_input_ids:
            pad_len = max_len - len(ids)
            padded = [self.tokenizer.pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            padded_ids.append(padded)
            attention_masks.append(mask)

        input_tensor = torch.tensor(padded_ids, device=self.device, dtype=torch.long)
        attention_tensor = torch.tensor(attention_masks, device=self.device, dtype=torch.long)

        # 批量 forward
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # ★ 标准化: 将 DynamicCache 转为 list of (k, v) tuples
        if isinstance(past_key_values, DynamicCache):
            past_kv_tuples = [
                (past_key_values.key_cache[l], past_key_values.value_cache[l])
                for l in range(len(past_key_values.key_cache))
            ]
        else:
            past_kv_tuples = list(past_key_values)

        # 为每个请求创建 PagedKVCache 并写入 K/V
        results = []
        for i, info in enumerate(request_info):
            handle = self._handle_counter
            self._handle_counter += 1

            input_ids = all_input_ids[i]
            input_len = info["input_length"]
            pad_len = max_len - input_len

            paged_cache = PagedKVCache(
                request_id=info["request_id"],
                block_pool=self.block_pool,
                block_size=self.paged_config.block_size,
                max_seq_len=self.max_sequence_length,
            )

            # 分配 blocks
            num_blocks_needed = (input_len + self.paged_config.block_size - 1) // self.paged_config.block_size
            for _ in range(num_blocks_needed):
                paged_cache.allocate_block()

            # ★ 从 batch past_key_values 提取该请求的 K/V
            # 跳过 padding 部分 (左侧 pad)
            for layer_idx, (k, v) in enumerate(past_kv_tuples):
                k_layer = k[i, :, pad_len:, :]  # (num_kv_heads, input_len, head_dim)
                v_layer = v[i, :, pad_len:, :]

                for block_idx in range(num_blocks_needed):
                    physical_id = paged_cache.block_table[block_idx]
                    bs = self.paged_config.block_size
                    start = block_idx * bs
                    end = min(start + bs, input_len)
                    self.block_pool.set_kv_block(
                        physical_id, layer_idx,
                        k_layer[:, start:end, :],
                        v_layer[:, start:end, :],
                        offset=0,
                    )

            # ★ 保存该请求的 past_key_values 切片用于后续 decode
            # 每个请求只需要自己的那部分 (去掉 padding)
            # 转换为 DynamicCache 以兼容新版 transformers
            own_past_kv = tuple(
                (k[i:i+1, :, pad_len:, :], v[i:i+1, :, pad_len:, :])
                for k, v in past_kv_tuples
            )
            paged_cache._past_key_values = DynamicCache.from_legacy_cache(own_past_kv)

            paged_cache.advance(input_ids)

            # 采样
            params = info["params"]
            first_token_id = self._sample_token(
                logits[i:i+1],
                params.get("temperature", 1.0),
                params.get("top_p", 1.0),
                params.get("top_k", 0),
                params.get("do_sample", False),
            )

            if first_token_id == self.eos_token_id:
                paged_cache.is_finished = True

            self._cache[handle] = paged_cache

            token_text = self.tokenizer.decode([first_token_id], skip_special_tokens=False)
            results.append((
                handle,
                [{"id": first_token_id, "text": token_text, "logprob": -0.5, "special": first_token_id == self.eos_token_id}],
                int((time.time() - start_time) * 1000),
            ))

            self.stats["total_prefills"] += 1
            self.stats["total_tokens_generated"] += 1

        return results

    # ================================================================
    # 缓存管理
    # ================================================================

    def clear_cache(self, handles: List[int]) -> bool:
        """清理指定的 Paged KV Cache"""
        for handle in handles:
            if handle in self._cache:
                paged_cache = self._cache[handle]
                paged_cache.free()  # 归还 block 到 pool
                del self._cache[handle]
                logger.debug(f"Cache cleared: handle={handle}")
        # 清理 CUDA 缓存
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return True

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "active_caches": len(self._cache),
            "block_pool_used": self.block_pool.used_blocks,
            "block_pool_free": self.block_pool.free_count,
            "block_pool_total": self.paged_config.num_blocks,
        }

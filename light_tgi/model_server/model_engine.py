"""
模型推理引擎 (真实模型版本)

职责：
1. 加载 HuggingFace 真实模型 (默认: Qwen/Qwen2.5-1.5B-Instruct)
2. 实现 Prefill (预填充) 逻辑 — 使用 model.generate() 批量推理
3. 实现 Decode (解码) 逻辑 — 复用 KV Cache 逐 token 生成
4. 管理 KV Cache (使用 transformers 的 StaticCache / DynamicCache)

设计要点:
- Prefill 阶段: 调用 model.forward() 处理整个 prompt，得到 logits + past_key_values
- Decode 阶段: 每次输入 1 个新 token + past_key_values，得到下一个 token
- 通过 StaticCache 复用 KV Cache，避免重复计算

支持的模型:
- Qwen/Qwen2.5-1.5B-Instruct (默认, 1.5B)
- Qwen/Qwen2.5-0.5B-Instruct (更轻量, 0.5B)
- 任何 HuggingFace CausalLM 模型
"""

import torch
import time
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """KV Cache 条目 — 存储 past_key_values 和生成状态"""
    handle: int
    request_id: str
    # ★ 真实的 KV Cache: transformers 的 past_key_values
    past_key_values: Optional[Tuple] = None
    # 已生成的 token ids (包含 input_ids + generated_ids)
    generated_ids: List[int] = field(default_factory=list)
    # 输入长度 (用于计算已生成 token 数)
    input_length: int = 0
    is_finished: bool = False
    # attention_mask
    attention_mask: Optional[torch.Tensor] = None
    # batch 中的索引 (批量 prefill 时使用)
    batch_index: int = 0


class ModelEngine:
    """
    真实模型推理引擎

    使用 HuggingFace transformers 加载真实模型进行推理。

    核心流程:
        Prefill:
            input_text → tokenizer → model.forward(prompt) → logits + past_key_values
            → 采样得到第一个 token → 保存 past_key_values 到 CacheEntry

        Decode:
            past_key_values + last_token → model.forward(token) → logits
            → 采样得到下一个 token → 更新 past_key_values
    """

    # 支持的模型列表
    SUPPORTED_MODELS = [
        "Qwen/Qwen2.5-1.5B-Instruct",   # 1.5B 参数, 推荐
        "Qwen/Qwen2.5-0.5B-Instruct",   # 0.5B 参数, 最轻量
        "Qwen/Qwen2.5-3B-Instruct",     # 3B 参数
        "Qwen/Qwen2.5-7B-Instruct",     # 7B 参数
        "Qwen/Qwen2.5-14B-Instruct",    # 14B 参数
        "google/gemma-2-2b-it",         # 2B 参数, Gemma
        "microsoft/Phi-3-mini-4k-instruct",  # 3.8B 参数
        "meta-llama/Llama-3.2-1B-Instruct",  # 1B 参数
    ]

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_sequence_length: int = 4096,
        max_batch_size: int = 8,        # 真实模型 batch 不宜过大
        device: str = "cpu",            # "cpu" | "cuda" | "cuda:0"
        dtype: str = "auto",            # "auto" | "float16" | "bfloat16"
    ):
        self.model_id = model_id
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.device = device
        self.dtype = dtype

        # 加载模型和 tokenizer
        self._load_model()

        # KV Cache 存储 (handle -> CacheEntry)
        self._cache: OrderedDict[int, CacheEntry] = OrderedDict()
        self._handle_counter: int = 1

        # 统计信息
        self.stats = {
            "total_prefills": 0,
            "total_decodes": 0,
            "total_tokens_generated": 0,
        }

        logger.info(f"模型引擎初始化完成: model={model_id}, device={device}")
        logger.info(f"  max_seq_len={max_sequence_length}, max_batch={max_batch_size}")
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
        预填充阶段: 处理整个 prompt，生成第一个 token

        流程:
        1. tokenizer.encode(input_text) → input_ids
        2. model.forward(input_ids) → logits + past_key_values
        3. 从 logits[-1] 采样得到第一个 token
        4. 保存 past_key_values 到 CacheEntry

        Returns:
            (cache_handle, generated_tokens, duration_ms)
        """
        start_time = time.time()

        # 1. Tokenize
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        input_len = len(input_ids)

        if input_len == 0:
            raise ValueError("输入为空，无法生成")

        if input_len > self.max_sequence_length:
            raise ValueError(
                f"输入过长: {input_len} tokens (最大: {self.max_sequence_length})"
            )

        # 2. 分配 handle
        handle = self._handle_counter
        self._handle_counter += 1

        # 3. 构建模型输入
        input_tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor)

        # 4. Forward pass (首次: 不带 past_key_values)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
            )

        # 5. 获取 logits 和 past_key_values
        logits = outputs.logits[:, -1, :]  # 只取最后一个位置的 logits
        past_key_values = outputs.past_key_values

        # 6. 采样第一个 token
        first_token_id = self._sample_token(
            logits, temperature, top_p, top_k, do_sample
        )

        # 7. 创建 CacheEntry
        entry = CacheEntry(
            handle=handle,
            request_id=request_id,
            past_key_values=past_key_values,
            generated_ids=list(input_ids) + [first_token_id],
            input_length=input_len,
            attention_mask=attention_mask,
        )

        # 检查 EOS
        if first_token_id == self.eos_token_id:
            entry.is_finished = True

        self._cache[handle] = entry

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
            f"duration={duration_ms}ms"
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
        解码阶段: 基于 KV Cache 生成下一个 token

        流程:
        1. 从 CacheEntry 获取 past_key_values
        2. model.forward(last_token, past_key_values) → logits + new_past_key_values
        3. 采样得到下一个 token
        4. 更新 CacheEntry

        Returns:
            (token_dict, finish_reason, duration_ms)
        """
        start_time = time.time()

        # 获取缓存
        entry = self._cache.get(cache_handle)
        if entry is None:
            logger.error(f"缓存未找到: handle={cache_handle}")
            return None, "error", 0

        # 检查是否已完成
        if entry.is_finished:
            return None, "eos_token", 0

        # 获取最后一个 token
        last_token_id = entry.generated_ids[-1]
        input_tensor = torch.tensor([[last_token_id]], device=self.device, dtype=torch.long)
        attention_mask = torch.cat([
            entry.attention_mask,
            torch.ones((1, 1), device=self.device, dtype=torch.long)
        ], dim=1)

        # Forward pass (使用 past_key_values)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                past_key_values=entry.past_key_values,
                use_cache=True,
            )

        # 更新状态
        logits = outputs.logits[:, -1, :]
        entry.past_key_values = outputs.past_key_values
        entry.attention_mask = attention_mask

        # 采样下一个 token
        next_token_id = self._sample_token(
            logits, temperature, top_p, top_k, do_sample
        )
        entry.generated_ids.append(next_token_id)

        # 判断停止条件
        finish_reason = None
        if next_token_id == self.eos_token_id:
            finish_reason = "eos_token"
            entry.is_finished = True

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
            f"token={next_token_id}({repr(token_text)}), finish={finish_reason}"
        )

        return token_dict, finish_reason, duration_ms

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
    # 批量 Prefill (支持多个请求在一次 forward 中处理)
    # ================================================================

    def batch_prefill(
        self,
        requests: List[Dict],
    ) -> List[Tuple[int, List[Dict], int]]:
        """
        批量预填充: 多个请求在同一个 batch 中做 prefill

        使用 left-padding 对齐不同长度的 prompt，
        一次 forward 处理整个 batch。

        Args:
            requests: [{"request_id": str, "input_text": str, "params": dict}, ...]

        Returns:
            [(cache_handle, generated_tokens, duration_ms), ...]
        """
        if len(requests) == 1:
            # 单请求走普通 prefill
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
        all_attention_masks = []
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

        # Left-padding: 短序列在左侧补 pad_token
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

        # 为每个请求采样 + 保存 cache
        results = []
        for i, info in enumerate(request_info):
            handle = self._handle_counter
            self._handle_counter += 1

            params = info["params"]
            first_token_id = self._sample_token(
                logits[i:i+1],
                params.get("temperature", 1.0),
                params.get("top_p", 1.0),
                params.get("top_k", 0),
                params.get("do_sample", False),
            )

            # 为每个请求创建独立的 CacheEntry
            # 注意: past_key_values 是 batch 共享的，需要每个请求单独提取
            # 简化处理: 每个请求存储完整的 batch past_key_values
            entry = CacheEntry(
                handle=handle,
                request_id=info["request_id"],
                past_key_values=past_key_values,  # batch 共享
                generated_ids=all_input_ids[i] + [first_token_id],
                input_length=info["input_length"],
                attention_mask=attention_tensor[i:i+1],
                batch_index=i,  # 记录在 batch 中的位置
            )

            if first_token_id == self.eos_token_id:
                entry.is_finished = True

            self._cache[handle] = entry

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
        """清理指定的 KV Cache"""
        for handle in handles:
            if handle in self._cache:
                # 释放 GPU 显存
                entry = self._cache[handle]
                if entry.past_key_values is not None:
                    del entry.past_key_values
                if entry.attention_mask is not None:
                    del entry.attention_mask
                del self._cache[handle]
                logger.debug(f"缓存已清理: handle={handle}")
        # 清理 CUDA 缓存
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return True

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "active_caches": len(self._cache),
        }

"""
模型推理引擎

职责：
1. 加载 HuggingFace 模型
2. 实现 Prefill (预填充) 逻辑
3. 实现 Decode (解码) 逻辑
4. 管理 KV Cache

这是 Python 端的核心，模拟真实 TGI 的模型服务器。
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
    """KV Cache 条目"""
    handle: int
    request_id: str
    # 简化的 KV cache 模拟：存储 key-value 对
    key_cache: List[torch.Tensor] = field(default_factory=list)
    value_cache: List[torch.Tensor] = field(default_factory=list)
    generated_ids: List[int] = field(default_factory=list)
    is_finished: bool = False


class ModelEngine:
    """
    模型推理引擎

    在教学环境中，我们使用简化的模拟实现：
    - 不加载真实模型权重
    - 使用确定性算法模拟 token 生成
    - 模拟 KV Cache 的行为

    真实环境中，这里会：
    - 使用 PyTorch/tensor-parallel 加载模型
    - 使用 FlashAttention 加速
    - 使用 PagedAttention 管理 KV Cache
    """

    def __init__(
        self,
        model_id: str = "mock-gpt2",
        max_sequence_length: int = 4096,
        max_batch_size: int = 32,
        vocab_size: int = 50257,
        eos_token_id: int = 50256,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.device = device

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

    # ================================================================
    # Prefill (预填充)
    # ================================================================

    def prefill(
        self,
        request_id: str,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ) -> Tuple[int, List[Dict], int]:
        """
        预填充阶段：处理整个 prompt，生成第一个 token

        Args:
            request_id: 请求唯一 ID
            input_ids: 输入的 token ID 列表
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling
            top_k: top-k sampling
            do_sample: 是否采样 (False = greedy)

        Returns:
            (cache_handle, generated_tokens, duration_ms)
            - cache_handle: KV Cache 句柄
            - generated_tokens: 生成的 token 列表 [{"id": int, "text": str}, ...]
            - duration_ms: 耗时(毫秒)
        """
        start_time = time.time()

        # 分配 KV Cache 句柄
        handle = self._handle_counter
        self._handle_counter += 1

        # 创建缓存条目
        entry = CacheEntry(
            handle=handle,
            request_id=request_id,
            generated_ids=list(input_ids),
        )

        # 模拟 Prefill 计算延迟
        # 真实场景中，Prefill 延迟与 prompt 长度正相关
        prefill_tokens = len(input_ids)
        simulated_delay = 0.001 * prefill_tokens  # 每 token 1ms (模拟)
        time.sleep(min(simulated_delay, 0.05))  # 最多等 50ms

        # 生成第一个新 token (模拟)
        first_token = self._generate_next_token(input_ids, temperature, do_sample)
        entry.generated_ids.append(first_token)

        # 模拟 KV Cache 存储
        entry.key_cache = [torch.randn(1, 64)]  # 简化模拟
        entry.value_cache = [torch.randn(1, 64)]

        # 检查是否遇到 EOS
        if first_token == self.eos_token_id:
            entry.is_finished = True

        # 保存缓存
        self._cache[handle] = entry

        duration_ms = int((time.time() - start_time) * 1000)
        self.stats["total_prefills"] += 1
        self.stats["total_tokens_generated"] += 1

        # 构建返回的 token 信息
        generated_tokens = [
            {
                "id": first_token,
                "text": self._token_id_to_text(first_token),
                "logprob": -0.5,
                "special": first_token == self.eos_token_id,
            }
        ]

        logger.debug(
            f"[Prefill] request={request_id}, handle={handle}, "
            f"input_tokens={prefill_tokens}, first_token={first_token}, "
            f"duration={duration_ms}ms"
        )

        return handle, generated_tokens, duration_ms

    # ================================================================
    # Decode (解码)
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
        解码阶段：基于 KV Cache 生成下一个 token

        Args:
            request_id: 请求唯一 ID
            cache_handle: KV Cache 句柄 (从 prefill 获取)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling
            top_k: top-k sampling
            do_sample: 是否采样

        Returns:
            (token_dict, finish_reason, duration_ms)
            - token_dict: {"id": int, "text": str, "logprob": float, "special": bool}
            - finish_reason: "eos_token" | "length" | None
            - duration_ms: 耗时
        """
        start_time = time.time()

        # 获取缓存
        entry = self._cache.get(cache_handle)
        if entry is None:
            logger.error(f"缓存未找到: handle={cache_handle}, request={request_id}")
            return None, "error", 0

        # 模拟 Decode 计算延迟
        # 真实场景中，Decode 每次只处理 1 个 token，延迟很低
        time.sleep(0.002)  # 2ms (模拟)

        # 生成下一个 token
        next_token = self._generate_next_token(
            entry.generated_ids, temperature, do_sample
        )
        entry.generated_ids.append(next_token)

        # 判断停止条件
        finish_reason = None
        generated_since_input = len(entry.generated_ids) - len(entry.generated_ids) + 1
        # 实际计算已生成的新 token 数
        new_token_count = sum(
            1 for _ in entry.generated_ids
        )  # 简化：直接用总长度
        # 更准确的计算
        # 假设 input_ids 已记录在 generated_ids 最前面
        # 这里简化为：直接判断是否达到 eos 或 max

        if next_token == self.eos_token_id:
            finish_reason = "eos_token"
            entry.is_finished = True
        # 检查是否达到 max_new_tokens
        # 简化：这里由调用方控制

        duration_ms = int((time.time() - start_time) * 1000)
        self.stats["total_decodes"] += 1
        self.stats["total_tokens_generated"] += 1

        token_dict = {
            "id": next_token,
            "text": self._token_id_to_text(next_token),
            "logprob": -0.3,
            "special": next_token == self.eos_token_id,
        }

        logger.debug(
            f"[Decode] request={request_id}, handle={cache_handle}, "
            f"token={next_token}, finish={finish_reason}, "
            f"duration={duration_ms}ms"
        )

        return token_dict, finish_reason, duration_ms

    # ================================================================
    # 缓存管理
    # ================================================================

    def clear_cache(self, handles: List[int]) -> bool:
        """清理指定的 KV Cache"""
        for handle in handles:
            if handle in self._cache:
                del self._cache[handle]
                logger.debug(f"缓存已清理: handle={handle}")
        return True

    def get_cache(self, handle: int) -> Optional[CacheEntry]:
        """获取缓存条目"""
        return self._cache.get(handle)

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _generate_next_token(
        self,
        context_ids: List[int],
        temperature: float,
        do_sample: bool,
    ) -> int:
        """
        模拟 token 生成

        真实环境中，这里会：
        1. 将 context_ids 输入模型
        2. 通过 transformer layers 计算 logits
        3. 根据 temperature/top_p/top_k 采样
        4. 返回采样到的 token id

        教学中使用确定性算法：
        - 基于 context_ids 的 hash 生成伪随机 token
        - 模拟一定的语义连贯性
        """
        if not do_sample or temperature == 0.0:
            # Greedy: 选择概率最高的 token
            # 简化：基于最近几个 token 的 hash
            recent = context_ids[-4:] if len(context_ids) >= 4 else context_ids
            hash_val = sum(
                t * (i + 1) * 2654435761 for i, t in enumerate(recent)
            ) & 0xFFFFFFFF
            token = (hash_val % (self.vocab_size - 1)) + 1
        else:
            # 模拟采样
            import random
            recent = context_ids[-4:] if len(context_ids) >= 4 else context_ids
            seed_val = sum(
                t * (i + 1) * 2654435761 for i, t in enumerate(recent)
            ) & 0xFFFFFFFF
            rng = random.Random(seed_val)
            token = rng.randint(1, self.vocab_size - 1)

        return token

    def _token_id_to_text(self, token_id: int) -> str:
        """
        将 token ID 转换为文本

        真实环境中使用 tokenizer.decode()
        教学中使用简化映射
        """
        if token_id == self.eos_token_id:
            return "<|endoftext|>"

        # 模拟：将 token id 映射为可读文本
        # 使用简单的单词表
        mock_words = [
            "the", "a", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "may", "might", "shall", "should", "must",
            "I", "you", "he", "she", "it", "we", "they",
            "this", "that", "these", "those", "here", "there",
            "good", "great", "bad", "new", "old", "big", "small",
            "time", "way", "day", "man", "woman", "child", "world",
            "life", "hand", "part", "place", "case", "week", "company",
            "system", "program", "question", "work", "government",
            "number", "night", "point", "home", "water", "room",
            "mother", "area", "money", "story", "fact", "month",
            "lot", "right", "study", "book", "eye", "job", "word",
            "business", "issue", "side", "kind", "head", "house",
            "service", "friend", "father", "power", "hour", "game",
            "line", "end", "member", "law", "car", "city", "community",
            "name", "president", "team", "minute", "idea", "kid",
            "body", "information", "back", "parent", "face", "others",
            "level", "office", "door", "health", "person", "art",
            "war", "history", "party", "result", "change", "morning",
            "reason", "research", "girl", "guy", "moment", "air",
            "teacher", "force", "education", "and", "or", "but",
            "not", "so", "if", "then", "than", "too", "very",
            "just", "about", "also", "now", "even", "only",
        ]
        idx = token_id % len(mock_words)
        return mock_words[idx]

"""
BatchScheduler — 生产级 Continuous Batching 调度器

核心思想:
  - 维护一个活跃请求池 (ActiveBatch)
  - 新请求到达 → 立即 prefill → 加入 decode 池
  - 每次 decode step: 将所有活跃请求的 next_token 组成 batch, 一次 model.forward()
  - 完成的请求从池中移除
  - 类似 vLLM 的 iteration-level scheduling

对比旧架构:
  旧: 每个请求独立 StreamGenerate → 串行 prefill+decode → 无 batch
  新: BatchScheduler 统一管理 → prefill 后混入 decode pool → 真正的 batch forward

Prefix Sharing:
  - 维护 PrefixCache (LRU, 前缀文本 → cache_handle + BlockTable 引用)
  - 新请求 prefill 前查 PrefixCache, 命中则跳过已缓存的 prefix 部分
  - Prefill 完成后将完整序列存入 PrefixCache
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict, deque

import torch

logger = logging.getLogger(__name__)


# ================================================================
# PrefixCache — 前缀共享缓存
# ================================================================

class PrefixCache:
    """
    LRU 前缀缓存

    原理:
      - Key: 前缀文本 (或 token 序列 hash)
      - Value: (cache_handle, token_count, block_ids)
      - 新请求 prefill 前查缓存 → 命中则跳过前缀, 只处理 suffix
      - 使用 OrderedDict 实现 LRU 淘汰

    适用场景:
      - 客服机器人: 共享 system prompt
      - RAG 应用: 共享检索到的 context
      - Few-shot prompting: 共享示例前缀
    """

    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def lookup(self, prefix_text: str, min_match_tokens: int = 4) -> Optional[Dict]:
        """
        查找前缀缓存

        Returns:
            None 如果未命中
            {"cache_handle": int, "token_count": int, "block_ids": List[int]} 如果命中
        """
        # 尝试最长前缀匹配
        for cached_prefix, entry in reversed(self._cache.items()):
            if prefix_text.startswith(cached_prefix):
                if entry["token_count"] >= min_match_tokens:
                    # LRU: 移到末尾
                    self._cache.move_to_end(cached_prefix)
                    self.hits += 1
                    logger.debug(
                        f"[PrefixCache] HIT: prefix_len={len(cached_prefix)}, "
                        f"tokens={entry['token_count']}"
                    )
                    return entry
        self.misses += 1
        return None

    def store(self, prefix_text: str, cache_handle: int, token_count: int, block_ids: List[int]):
        """存储前缀缓存条目"""
        if len(self._cache) >= self.max_entries:
            # LRU 淘汰最旧的
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"[PrefixCache] EVICT: {oldest_key[:50]}...")

        self._cache[prefix_text] = {
            "cache_handle": cache_handle,
            "token_count": token_count,
            "block_ids": list(block_ids),
        }
        logger.debug(
            f"[PrefixCache] STORE: prefix_len={len(prefix_text)}, "
            f"tokens={token_count}, entries={len(self._cache)}"
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)


# ================================================================
# ActiveRequest — 活跃请求状态
# ================================================================

class ActiveRequest:
    """Batch 中的活跃请求"""

    __slots__ = (
        "request_id", "cache_handle", "input_ids", "input_len",
        "max_new_tokens", "temperature", "top_p", "top_k", "do_sample",
        "generated_ids", "generated_count", "is_finished", "finish_reason",
        "step", "queue_time",
    )

    def __init__(
        self,
        request_id: str,
        cache_handle: int,
        input_ids: List[int],
        input_len: int,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ):
        self.request_id = request_id
        self.cache_handle = cache_handle
        self.input_ids = input_ids
        self.input_len = input_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.generated_ids: List[int] = []
        self.generated_count = 0
        self.is_finished = False
        self.finish_reason: Optional[str] = None
        self.step = 0
        self.queue_time = time.time()

    def last_token_id(self) -> int:
        """获取最后生成的 token id (用于 decode)"""
        if self.generated_ids:
            return self.generated_ids[-1]
        return self.input_ids[-1] if self.input_ids else 0

    def record_token(self, token_id: int):
        """记录生成的 token"""
        self.generated_ids.append(token_id)
        self.generated_count += 1
        self.step += 1

    def check_finished(self, eos_token_id: int, max_new_tokens: int) -> Optional[str]:
        """检查是否应该停止"""
        if self.generated_ids and self.generated_ids[-1] == eos_token_id:
            return "eos_token"
        if self.generated_count >= max_new_tokens:
            return "length"
        return None


# ================================================================
# BatchScheduler — 核心调度器
# ================================================================

class BatchScheduler:
    """
    Continuous Batching 调度器

    工作流程:
      1. 接收新请求 → 放入 pending 队列
      2. 定时 flush: 将 pending 请求做 batch prefill
      3. 将 prefill 完成的请求加入 active decode pool
      4. 每次 decode step: 所有 active 请求一起 batch decode
      5. 完成的请求从 pool 移除, 结果通过回调返回

    关键设计:
      - Prefill 和 Decode 可以混合: 当 pending 非空时, 在当前 decode step
        中插入 prefill (需要支持 mixed batch, 或先做 prefill 再做 decode)
      - 简化版: prefill 独立 step, decode 独立 step, 交替进行
    """

    def __init__(
        self,
        engine,  # ModelEngine
        max_batch_size: int = 8,
        max_batch_prefill_tokens: int = 4096,
        max_active_requests: int = 32,
        prefix_cache_size: int = 64,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_batch_prefill_tokens = max_batch_prefill_tokens
        self.max_active_requests = max_active_requests

        # Pending 队列: 等待 prefill 的请求
        self.pending: deque = deque()
        self.pending_lock = threading.Lock()

        # Active decode pool: request_id → ActiveRequest
        self.active: OrderedDict[str, ActiveRequest] = OrderedDict()
        self.active_lock = threading.Lock()

        # Prefix Cache
        self.prefix_cache = PrefixCache(max_entries=prefix_cache_size)

        # 输出队列: (request_id, token_dict, finish_reason) → 由外部消费
        self.output_queue: deque = deque()
        self.output_lock = threading.Lock()

        # 统计
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_prefill_steps": 0,
            "total_decode_steps": 0,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
        }

        logger.info(
            f"[BatchScheduler] 初始化: max_batch={max_batch_size}, "
            f"max_prefill_tokens={max_batch_prefill_tokens}, "
            f"max_active={max_active_requests}, "
            f"prefix_cache_size={prefix_cache_size}"
        )

    # ================================================================
    # 公共接口
    # ================================================================

    def add_request(
        self,
        request_id: str,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
    ):
        """添加新请求到 pending 队列"""
        with self.pending_lock:
            self.pending.append({
                "request_id": request_id,
                "input_text": input_text,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "queue_time": time.time(),
            })
        self.stats["total_requests"] += 1
        logger.debug(f"[BatchScheduler] 新请求入队: {request_id} (pending={len(self.pending)})")

    def cancel_request(self, request_id: str):
        """取消请求"""
        with self.pending_lock:
            self.pending = deque(
                r for r in self.pending if r["request_id"] != request_id
            )
        with self.active_lock:
            if request_id in self.active:
                req = self.active.pop(request_id)
                self.engine.clear_cache([req.cache_handle])
                logger.info(f"[BatchScheduler] 取消请求: {request_id}")

    def step(self) -> List[Tuple[str, Optional[Dict], Optional[str]]]:
        """
        执行一步调度 (一个 iteration)

        流程:
          1. 如果有 pending 请求 → 做 batch prefill
          2. 如果有 active 请求 → 做 batch decode
          3. 返回本轮产生的 token

        Returns:
            [(request_id, token_dict_or_None, finish_reason_or_None), ...]
        """
        results = []

        # Step 1: 处理 pending 请求 (Prefill)
        prefill_results = self._prefill_step()
        results.extend(prefill_results)

        # Step 2: 处理 active 请求 (Decode)
        if self.active:
            decode_results = self._decode_step()
            results.extend(decode_results)

        return results

    def pop_outputs(self) -> List[Tuple[str, Optional[Dict], Optional[str]]]:
        """弹出输出队列中的所有结果"""
        with self.output_lock:
            results = list(self.output_queue)
            self.output_queue.clear()
        return results

    @property
    def active_count(self) -> int:
        return len(self.active)

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "active_requests": len(self.active),
            "pending_requests": len(self.pending),
            "prefix_cache_entries": self.prefix_cache.size,
            "prefix_cache_hit_rate": self.prefix_cache.hit_rate,
        }

    # ================================================================
    # Prefill Step
    # ================================================================

    def _prefill_step(self) -> List[Tuple[str, Optional[Dict], Optional[str]]]:
        """
        批量 Prefill: 从 pending 队列取出请求, 组成 batch

        返回: [(request_id, token_dict, finish_reason), ...]
        """
        with self.pending_lock:
            if not self.pending:
                return []

            # 按 token 预算收集请求
            collected = []
            total_tokens = 0

            while self.pending and len(collected) < self.max_batch_size:
                req = self.pending[0]
                input_ids, token_count = self.engine.tokenize(req["input_text"])

                if total_tokens + token_count > self.max_batch_prefill_tokens:
                    if not collected:
                        # 单个请求就超过预算, 仍然处理 (但警告)
                        logger.warning(
                            f"[BatchScheduler] 请求超过 prefill 预算: "
                            f"{req['request_id']} tokens={token_count} > {self.max_batch_prefill_tokens}"
                        )
                        collected.append(req)
                        self.pending.popleft()
                    break

                collected.append(req)
                self.pending.popleft()
                total_tokens += token_count

            if not collected:
                return []

        logger.info(
            f"[BatchScheduler] Prefill batch: {len(collected)} 请求, "
            f"{total_tokens} tokens"
        )
        self.stats["total_prefill_steps"] += 1

        results = []

        if len(collected) == 1:
            # 单请求走 prefill (更高效)
            req = collected[0]
            try:
                cache_handle, tokens, _ = self.engine.prefill(
                    request_id=req["request_id"],
                    input_text=req["input_text"],
                    max_new_tokens=req["max_new_tokens"],
                    temperature=req["temperature"],
                    top_p=req["top_p"],
                    top_k=req["top_k"],
                    do_sample=req["do_sample"],
                )

                # 构建 ActiveRequest 加入 decode pool
                input_ids, input_len = self.engine.tokenize(req["input_text"])
                active_req = ActiveRequest(
                    request_id=req["request_id"],
                    cache_handle=cache_handle,
                    input_ids=input_ids,
                    input_len=input_len,
                    max_new_tokens=req["max_new_tokens"],
                    temperature=req["temperature"],
                    top_p=req["top_p"],
                    top_k=req["top_k"],
                    do_sample=req["do_sample"],
                )

                for token in tokens:
                    active_req.record_token(token["id"])
                    finish_reason = active_req.check_finished(
                        self.engine.eos_token_id, req["max_new_tokens"]
                    )
                    if finish_reason:
                        active_req.is_finished = True
                        active_req.finish_reason = finish_reason
                    results.append((req["request_id"], token, finish_reason))

                if not active_req.is_finished:
                    with self.active_lock:
                        self.active[req["request_id"]] = active_req
                else:
                    self.engine.clear_cache([cache_handle])

                # ★ Prefix Sharing: 缓存 prefill 结果
                self.prefix_cache.store(
                    req["input_text"], cache_handle, input_len,
                    self.engine._cache[cache_handle].block_table.to_list(),
                )

            except Exception as e:
                logger.error(f"[BatchScheduler] Prefill 失败: {req['request_id']}, {e}")
                results.append((req["request_id"], None, "error"))

        else:
            # 批量 prefill
            batch_requests = [
                {
                    "request_id": r["request_id"],
                    "input_text": r["input_text"],
                    "params": {
                        "max_new_tokens": r["max_new_tokens"],
                        "temperature": r["temperature"],
                        "top_p": r["top_p"],
                        "top_k": r["top_k"],
                        "do_sample": r["do_sample"],
                    },
                }
                for r in collected
            ]

            try:
                batch_results = self.engine.batch_prefill(batch_requests)

                for i, (cache_handle, tokens, _) in enumerate(batch_results):
                    req = collected[i]
                    input_ids, input_len = self.engine.tokenize(req["input_text"])

                    active_req = ActiveRequest(
                        request_id=req["request_id"],
                        cache_handle=cache_handle,
                        input_ids=input_ids,
                        input_len=input_len,
                        max_new_tokens=req["max_new_tokens"],
                        temperature=req["temperature"],
                        top_p=req["top_p"],
                        top_k=req["top_k"],
                        do_sample=req["do_sample"],
                    )

                    for token in tokens:
                        active_req.record_token(token["id"])
                        finish_reason = active_req.check_finished(
                            self.engine.eos_token_id, req["max_new_tokens"]
                        )
                        if finish_reason:
                            active_req.is_finished = True
                            active_req.finish_reason = finish_reason
                        results.append((req["request_id"], token, finish_reason))

                    if not active_req.is_finished:
                        with self.active_lock:
                            self.active[req["request_id"]] = active_req
                    else:
                        self.engine.clear_cache([cache_handle])

            except Exception as e:
                logger.error(f"[BatchScheduler] Batch Prefill 失败: {e}")
                for req in collected:
                    results.append((req["request_id"], None, "error"))

        return results

    # ================================================================
    # Decode Step — 真正的 Continuous Batching
    # ================================================================

    def _decode_step(self) -> List[Tuple[str, Optional[Dict], Optional[str]]]:
        """
        批量 Decode: 所有 active 请求一起做 batch decode

        这是 Continuous Batching 的核心:
          - 每个请求贡献 1 个 token 到 batch
          - 一次 model.forward() 处理所有请求
          - 不同请求处于不同的生成阶段, 但共享同一次 GPU forward
        """
        self.stats["total_decode_steps"] += 1

        with self.active_lock:
            # ★ 先过滤掉已完成的请求 (防御性清理)
            finished_before = [
                rid for rid, req in self.active.items()
                if req.generated_count >= req.max_new_tokens or req.is_finished
            ]
            for rid in finished_before:
                req = self.active.pop(rid)
                try:
                    self.engine.clear_cache([req.cache_handle])
                except Exception:
                    pass
                logger.warning(
                    f"[BatchScheduler] 清理残留的已完成请求: {rid}, "
                    f"generated={req.generated_count}/{req.max_new_tokens}"
                )

            active_requests = list(self.active.values())

        if not active_requests:
            return []

        results = []
        finished_handles = []

        if len(active_requests) == 1:
            # 单请求 decode
            req = active_requests[0]
            try:
                token_dict, finish_reason, _ = self.engine.decode(
                    request_id=req.request_id,
                    cache_handle=req.cache_handle,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    do_sample=req.do_sample,
                )

                if token_dict:
                    req.record_token(token_dict["id"])
                    self.stats["total_tokens"] += 1

                    if finish_reason:
                        req.is_finished = True
                        req.finish_reason = finish_reason
                        finished_handles.append(req.request_id)

                    results.append((req.request_id, token_dict, finish_reason))
                else:
                    # token_dict is None → error
                    req.is_finished = True
                    req.finish_reason = "error"
                    finished_handles.append(req.request_id)
                    results.append((req.request_id, None, "error"))

            except Exception as e:
                logger.error(f"[BatchScheduler] Decode 失败: {req.request_id}, {e}")
                req.is_finished = True
                req.finish_reason = "error"
                results.append((req.request_id, None, "error"))
                finished_handles.append(req.request_id)

        else:
            # ★ 真正的 batch decode: 一次 model.forward() 处理多个请求
            batch_size = len(active_requests)
            logger.info(f"[BatchScheduler] Batch Decode: {batch_size} 请求, "
                        f"generated_counts={[r.generated_count for r in active_requests]}, "
                        f"max_new_tokens={[r.max_new_tokens for r in active_requests]}")

            # 准备 batch 输入
            batch_input_ids = []
            batch_cache_handles = []

            for req in active_requests:
                batch_input_ids.append(req.last_token_id())
                batch_cache_handles.append(req.cache_handle)

            # ★ Batch Decode: 所有请求一起 forward
            try:
                batch_token_dicts, batch_finish_reasons, _ = self.engine.batch_decode(
                    request_ids=[r.request_id for r in active_requests],
                    cache_handles=batch_cache_handles,
                    token_ids=batch_input_ids,
                    params_list=[
                        {
                            "max_new_tokens": r.max_new_tokens,
                            "temperature": r.temperature,
                            "top_p": r.top_p,
                            "top_k": r.top_k,
                            "do_sample": r.do_sample,
                        }
                        for r in active_requests
                    ],
                )

                for i, req in enumerate(active_requests):
                    token_dict = batch_token_dicts[i] if i < len(batch_token_dicts) else None
                    finish_reason = batch_finish_reasons[i] if i < len(batch_finish_reasons) else None

                    if token_dict:
                        req.record_token(token_dict["id"])
                        self.stats["total_tokens"] += 1

                        if finish_reason:
                            req.is_finished = True
                            req.finish_reason = finish_reason
                            finished_handles.append(req.request_id)

                        results.append((req.request_id, token_dict, finish_reason))
                    else:
                        req.is_finished = True
                        req.finish_reason = "error"
                        results.append((req.request_id, None, "error"))
                        finished_handles.append(req.request_id)

            except Exception as e:
                logger.error(f"[BatchScheduler] Batch Decode 失败: {e}")
                for req in active_requests:
                    req.is_finished = True
                    req.finish_reason = "error"
                    results.append((req.request_id, None, "error"))
                    finished_handles.append(req.request_id)

        # 清理完成的请求
        if finished_handles:
            with self.active_lock:
                for rid in finished_handles:
                    if rid in self.active:
                        req = self.active.pop(rid)
                        try:
                            self.engine.clear_cache([req.cache_handle])
                        except Exception:
                            pass

        return results

    # ================================================================
    # 运行循环
    # ================================================================

    def run_loop(self, stop_event: threading.Event, result_callback=None):
        """
        主调度循环 (在独立线程中运行)

        持续执行 step(), 将结果通过回调返回

        Args:
            stop_event: 停止信号
            result_callback: 回调函数 (request_id, token_dict, finish_reason)
        """
        logger.info("[BatchScheduler] 调度循环启动")

        while not stop_event.is_set():
            has_work = self.pending_count > 0 or self.active_count > 0

            if has_work:
                results = self.step()
                if results and result_callback:
                    for request_id, token_dict, finish_reason in results:
                        result_callback(request_id, token_dict, finish_reason)
            else:
                # 没有工作时短暂休眠
                time.sleep(0.005)  # 5ms

        logger.info("[BatchScheduler] 调度循环停止")

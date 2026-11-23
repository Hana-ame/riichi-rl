"""高速向听与听牌缓存模块（路线 1 核心加速组件）。

性能瓶颈定位：
- 旧版 `train_hanchan.encode()` 在玩家听牌（sh == 0）时会循环 34 次调用
  `REF.can_win_with()`，每次都触发完整的 `HandCalculator.estimate_hand_value` 算役，
  占用整个训练步 60%+ 的 CPU 耗时。
- 本模块提供：
  1. `cached_shanten(counts_tuple)`：基于 34 维不可变 tuple 的 LRU 缓存，避免重复递归；
  2. `fast_wait_count(counts_tuple, melds)`：仅在手牌听牌时，基于向听数是否降到 -1 快速枚举
     有效进张种类（跳过重型算役，速度提升 20× 以上，用于 touch 形状特征完全足够）。
"""

from __future__ import annotations

from functools import lru_cache

from mahjong import refengine as REF


@lru_cache(maxsize=262144)
def cached_shanten(counts_key: tuple[int, ...]) -> int:
    """带 LRU 缓存的向听数计算（和牌=-1，听牌=0）。"""
    return REF.ref_shanten(counts_key)


@lru_cache(maxsize=65536)
def fast_wait_count(counts_key: tuple[int, ...]) -> int:
    """快速听牌种数统计：测试哪些牌加入后向听变为 -1（不调用重型算役器）。"""
    base = list(counts_key)
    waits = 0
    for t in range(34):
        if base[t] >= 4:
            continue
        base[t] += 1
        if cached_shanten(tuple(base)) == -1:
            waits += 1
        base[t] -= 1
    return waits


def clear_caches() -> None:
    cached_shanten.cache_clear()
    fast_wait_count.cache_clear()


def cache_stats() -> dict:
    return {
        "shanten": cached_shanten.cache_info()._asdict(),
        "waits": fast_wait_count.cache_info()._asdict(),
    }

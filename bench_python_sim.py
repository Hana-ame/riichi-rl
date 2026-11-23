"""纯 Python 高效模拟器基准测试：对比单进程 vs 多进程并行池 + 向听缓存吞吐量。

用法：
    python bench_python_sim.py --deals 8 --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import fast_cache as FC
from mahjong import obs2 as O2
from mahjong.hanchan import HanchanState
from mahjong.mp_env import ParallelDealPool, _encode_fast, _filter_legal, _slot_of


def run_serial(n_deals: int, event_len: int = 48):
    enc = O2.ObsEncoder(order=False)
    steps = 0
    for i in range(n_deals):
        h = HanchanState(dealer=i % 4, start_points=25000)
        env = h.new_env(seed=5000 + i)
        res = env.start_turn()
        g = 0
        while res is None and g < 400:
            g += 1
            acts = {}
            for p in env.pending_players():
                legal = _filter_legal(env.legal_actions(p), allow_ryuukyoku=False)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                _encode_fast(env, enc, h, p, event_len)
                acts[p] = legal[0]
                steps += 1
            res = env.step(acts)
    return steps


def run_parallel(n_deals: int, n_workers: int, event_len: int = 48):
    hstates = [HanchanState(dealer=i % 4, start_points=25000) for i in range(n_deals)]
    seeds = [5000 + i for i in range(n_deals)]
    steps = [0]

    def fast_policy(big_v, big_tf, big_ev, slot_lists):
        steps[0] += len(slot_lists)
        return [(sl[0], 0.0, -1) for sl in slot_lists]

    with ParallelDealPool(n_workers=n_workers, obs_order=False, event_len=event_len) as pool:
        pool.run_deals(hstates, seeds, fast_policy)
    return steps[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--event-len", type=int, default=48)
    args = ap.parse_args()

    FC.clear_caches()
    t0 = time.perf_counter()
    s1 = run_serial(args.deals, args.event_len)
    dt1 = time.perf_counter() - t0
    stats = FC.cache_stats()["shanten"]
    hit_rate = 100.0 * stats["hits"] / max(1, stats["hits"] + stats["misses"])

    t1 = time.perf_counter()
    s2 = run_parallel(args.deals, args.workers, args.event_len)
    dt2 = time.perf_counter() - t1

    print(f"[Serial  1 worker ] deals={args.deals} steps={s1} time={dt1:.2f}s ({s1/dt1:.1f} steps/s, shanten cache hit={hit_rate:.1f}%)")
    print(f"[Parallel {args.workers} workers] deals={args.deals} steps={s2} time={dt2:.2f}s ({s2/dt2:.1f} steps/s, speedup={dt1/max(dt2,1e-6):.2f}x)")


if __name__ == "__main__":
    main()

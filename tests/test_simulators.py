"""纯 Python 高效模拟器回归测试（LRU 向听/听牌缓存 + 多进程 Actor-Learner 并行池）。

运行：python -m tests.test_simulators
"""

from __future__ import annotations

import os
import py_compile
import unittest

import tests.utils as U
from mahjong import fast_cache as FC
from mahjong import refengine as REF
from mahjong.hanchan import HanchanState
from mahjong.mp_env import ParallelDealPool


class TestPythonSimulatorSpeedup(unittest.TestCase):
    def test_fast_cache_shanten_and_waits(self):
        FC.clear_caches()
        # 123m 456m 789m 23p 东东 -> 听 1p / 4p 两面听（2 种听牌）
        c = [0] * 34
        for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 27, 27):
            c[t] += 1
        key = tuple(c)
        self.assertEqual(FC.cached_shanten(key), 0)
        self.assertEqual(FC.cached_shanten(key), REF.ref_shanten(c))
        self.assertEqual(FC.fast_wait_count(key), 2)
        self.assertGreaterEqual(FC.cache_stats()["shanten"]["hits"], 1)

    def test_parallel_deal_pool_smoke(self):
        hstates = [HanchanState(dealer=i % 4, start_points=25000) for i in range(4)]
        seeds = [2026 + i for i in range(4)]

        def dummy_policy(big_v, big_tf, big_ev, slot_lists):
            self.assertEqual(big_ev.ndim, 3)
            self.assertEqual(big_ev.shape[2], 4)
            return [(slots[0], 0.0, -1) for slots in slot_lists]

        with ParallelDealPool(n_workers=2, obs_order=False, event_len=48) as pool:
            outs = pool.run_deals(hstates, seeds, dummy_policy)

        self.assertEqual(len(outs), 4)
        for env, res, info in outs:
            self.assertIsNotNone(res)
            self.assertGreater(len(info), 0)
            self.assertEqual(len(info[0]), 11)
            self.assertEqual(info[0][10].shape, (48, 4))

    def test_python_sim_files_compile_and_no_rust_jax(self):
        self.assertFalse(os.path.exists(os.path.join(U.ROOT, "rust_sim")))
        self.assertFalse(os.path.exists(os.path.join(U.ROOT, "jax_sim")))
        for rel in (
            "mahjong/fast_cache.py",
            "mahjong/mp_env.py",
            "bench_python_sim.py",
            "gen_teacher_data.py",
            "train_hanchan.py",
        ):
            py_compile.compile(os.path.join(U.ROOT, rel), doraise=True)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))

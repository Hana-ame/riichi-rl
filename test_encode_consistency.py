"""编码一致性测试：训练侧 build_cache.encode 与推理侧 arena_model.encode_live。

这是**静默失败**的守卫：两边字段只要有一处不一致，模型在自己编码上
跑就变弱，但不会抛异常、不会掉点、看不出理由。这里用同一局面
分别喂进两个编码器，逐元素断言相等。

用法：python test_encode_consistency.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_cache as BC          # noqa: E402
import houou_data as H            # noqa: E402
from arena_model import encode_live  # noqa: E402
from mahjong import env as E      # noqa: E402
from mahjong import policies as POL  # noqa: E402


def build_env(seed=1234, dealer=0):
    env = E.MahjongEnv(seed=seed, dealer=dealer, honba=1,
                       round_wind=28, kyoku=2)
    env.start_turn()
    return env


def public_state(env, turn):
    """从引擎当前局面抓出 build_cache.encode 需要的公开字段。"""
    discards_by = np.zeros((4, 34), dtype=np.uint8)
    meld_cnt = np.zeros(4, dtype=np.int32)
    riichi = np.zeros(4, dtype=np.uint8)
    for i in range(4):
        p = env.players[i]
        for t in p.river:
            discards_by[i, t] += 1
        meld_cnt[i] = len(p.melds)
        if p.riichi:
            riichi[i] = 1
    return discards_by, meld_cnt, riichi


def main():
    env = build_env()
    turn = 0
    checked = 0
    mismatches = []
    r = None
    for step_i in range(80):
        if r is not None:
            break
        acts = {}
        for p in env.pending_players():
            legal = env.legal_actions(p)
            if not legal:
                acts[p] = E.PASS_ACTION
                continue
            acts[p] = POL.RandomLegal(0).choose(env, p, legal) \
                if hasattr(POL, "RandomLegal") else legal[0]
        r = env.step(acts)
        turn += 1
        for p in range(4):
            discards_by, meld_cnt, riichi = public_state(env, turn)
            p0 = p
            hand = np.zeros(34, dtype=np.uint8)
            for t in env.players[p0].hand:
                hand[t] += 1
            ind = env.dora_indicator
            # 训练侧：直接调 build_cache.encode
            d = H.Decision(
                player=p0, trigger_type=0, trigger_actor=p0, trigger_tile=-1,
                hand_counts=hand, discards=discards_by[p0],
                discards_by=discards_by,
                melds=list(env.players[p0].melds),
                melds_by=[list(x.melds) for x in env.players],
                legal=[], expert=None,
                dora_indicators=np.array([ind, 0, 0, 0, 0], dtype=np.int32),
                scores=np.array([x.points for x in env.players]),
                seat_wind=int(env.players[p0].seat_wind),
                round_wind=int(env.round_wind), kyoku=int(env.kyoku),
                honba=int(env.honba),
                tiles_left=len(env.wall) - env.wall_idx,
                turn=turn, meld_cnt=meld_cnt, riichi_flags=riichi,
                score_delta=0, final_rank=0, steps_to_done=0,
            )
            pl_a, sc_a = BC.encode(d, {})
            pl_b, sc_b = encode_live(
                env, p0, discards_by.astype(np.float32),
                riichi.astype(np.float32), meld_cnt.astype(np.float32),
                turn, 0, trigger_actor=p0, trigger_tile=-1)
            if not (np.allclose(pl_a, pl_b) and np.allclose(sc_a, sc_b)):
                diff = np.abs(pl_a - pl_b).max(), np.abs(sc_a - sc_b).max()
                mismatches.append((step_i, p0, diff))
            checked += 1
        if checked >= 60:
            break

    print(f"checked {checked} (env, player) pairs")
    if mismatches:
        print(f"MISMATCH {len(mismatches)} times, first 3:")
        for m in mismatches[:3]:
            print("  step", m[0], "player", m[1], "max diff", m[2])
        sys.exit(1)
    print("PASS: 训练侧与推理侧编码逐元素一致")


if __name__ == "__main__":
    main()
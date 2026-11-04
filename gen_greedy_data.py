"""用修好的 GreedyTenpai（向听最小化+立直+不早宣言流局）生成 BC 数据。

产出 npz（obs/act/seat/pol）与 serve_game 记录同格式，供 train_from_games.py 蒸馏。
这是「先弄出一个能 agari 的模型」的快速路径：让神经网络先学会这款能听牌/立直的策略。

用法：
    python gen_greedy_data.py --games 120 --out-dir data/selfplay
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import policies as POL
from train_reinforce import slot_of

ROOT = os.path.dirname(os.path.abspath(__file__))


def next_seq(out_dir):
    mx = 0
    for fn in glob.glob(os.path.join(out_dir, "g*.npz")):
        m = re.search(r"g(\d+)\.npz$", os.path.basename(fn))
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed", type=int, default=50000)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "selfplay"))
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = O2.ObsEncoder()
    seq = next_seq(args.out_dir)
    total_steps = total_games = wins = 0
    for g in range(args.games):
        seed = args.seed + g
        env = E.MahjongEnv(seed=seed, dealer=seed % 4)
        pols = {i: POL.make_policy("greedy_tenpai", seed=seed * 97 + i)
                for i in range(4)}
        obs_l, act_l, seat_l, pol_l = [], [], [], []
        r = env.start_turn()
        steps = 0
        while r is None and steps < 600:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                a = pols[p].choose(env, p, legal)
                acts[p] = a
                # 只记录「打牌阶段的决策」和「和牌动作」。
                # 响应阶段的 PASS 占全部决策点 ~73%，全是噪声，会把模型
                # 训成保守型（实测蒸馏模型立直 0 次、和了 0 局）。
                keep = (env.phase == E.Phase.DISCARD
                        or a.type in (E.ACT_RON, E.ACT_TSUMO))
                s = slot_of(a)
                if keep and s is not None and s >= 0:
                    planes, scalar = enc.encode(env, p)
                    obs_l.append(np.concatenate([planes.ravel(), scalar]).astype(np.float32))
                    act_l.append(int(s))
                    seat_l.append(int(p))
                    pol_l.append("greedy_tenpai")
            r = env.step(acts)
        if not obs_l:
            continue
        np.savez_compressed(
            os.path.join(args.out_dir, f"g{seq:05d}.npz"),
            obs=np.stack(obs_l), act=np.array(act_l, np.int64),
            seat=np.array(seat_l, np.int64), pol=np.array(pol_l),
            winner=r.winner if r else -1,
            result_type=r.result_type if r else "unfinished",
            points_delta=np.array(r.points_delta if r else [0, 0, 0, 0]),
            seed=seed,
        )
        seq += 1
        total_steps += len(obs_l)
        total_games += 1
        if r and r.result_type != "ryuukyoku":
            wins += 1
        print(f"[gen] game {g+1}/{args.games} steps={len(obs_l)} "
              f"result={r.result_type if r else '?'} "
              f"累计和了率={wins/max(1,total_games)*100:.0f}%", flush=True)
    print(f"done. {total_games} games, {total_steps} decision points", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
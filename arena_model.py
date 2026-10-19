"""竞技场：BC 模型 vs 四个规则基线 bot，用可证伪判据排名。

判据（严格沿用 arena.py 的口径，不放宽）：
  1. 每批 4 局覆盖全部座次，消除庄家偏差
  2. 批次为统计独立单元（cluster-robust），不拿单局当独立样本
  3. 95% CI 跨 0 就写「无法区分」，不写「略强」

用法：
    python arena_model.py --ckpt checkpoints/bc_best.pt --batches 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_only   # noqa: E402  必须早于 torch
import torch      # noqa: E402

import build_cache as BC          # noqa: E402
import houou_data as H            # noqa: E402
from action_bridge import env_action_to_slot   # noqa: E402
from mahjong import env as E                  # noqa: E402
from mahjong import policies as POL           # noqa: E402
from mahjong.hanchan import HanchanState      # noqa: E402
from train_bc import MahjongNet              # noqa: E402

BASELINES = ["random", "greedy_tenpai", "call_first", "yaochu_first"]


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MahjongNet(ck.get("channels", 256), ck.get("blocks", 8)).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def encode_live(env, p_idx, discards_by, riichi_flags, meld_cnt, turn,
                trigger_type, trigger_actor=None, trigger_tile=-1):
    """把引擎局面编码成与训练一致的 (planes(16,34), scalar(12))。

    必须与 build_cache.encode **逐字段一致**。推理/训练编码不一致会静默
    降低强度且不报错，所以这里逐行对齐而不是「差不多」。

    引擎侧对应关系（2026-10-05 核对 mahjong/env.py:194-251）：
      剩余牌数 = len(env.wall) - env.wall_idx   （env.py:256 同一算法）
      自风     = env.players[i].seat_wind        （27..30，env.py:210）
      场风     = env.round_wind
      宝牌指示 = env.dora_indicator              （0-based 牌号）
    """
    pl = np.zeros((BC.N_PLANES, 34), dtype=np.float32)
    sc = np.zeros(BC.SCALAR_DIM, dtype=np.float32)

    left = len(env.wall) - env.wall_idx

    hand = np.zeros(34, dtype=np.float32)
    for t in env.players[p_idx].hand:
        hand[t] += 1
    pl[0] = hand
    pl[1] = discards_by[p_idx].copy()
    for i in range(4):
        if i != p_idx:
            pl[2 + i] = discards_by[i].copy()
    ind = env.dora_indicator
    if ind is not None and ind >= 0:
        pl[6, min(int(ind), 33)] = 1.0
    pl[7, :4] = np.array([x.points for x in env.players], np.float32) / 40000.0
    pl[8, :] = left / 136.0
    for i in range(4):
        if riichi_flags[i]:
            pl[9, min(i, 33)] = 1.0
    seat_wind = env.players[p_idx].seat_wind
    pl[10, 0] = (int(seat_wind) - 27) / 3.0
    pl[10, 1] = float(env.kyoku == 1)
    pl[11, min(int(trigger_type), 3)] = 1.0
    pl[12, 0] = (env.players[p_idx].points - 25000) / 20000.0
    pl[13, 0] = turn / 18.0
    for i in range(4):
        pl[14, i] = meld_cnt[i] / 4.0

    sc[0] = env.kyoku / 4.0
    sc[1] = env.honba / 5.0
    sc[2] = (int(env.round_wind) - 27) / 3.0
    sc[3] = meld_cnt[p_idx]
    # sc[4]/sc[5] 训练侧来自数据集的 trigger_actor/trigger_tile；
    # 推理侧自己摸牌时 actor==自己、tile 无意义，故取同值语义。
    sc[4] = 1.0 if trigger_actor == p_idx else 0.0
    sc[5] = (trigger_tile / 33.0) if trigger_tile >= 0 else -1.0
    sc[6] = (ind / 33.0) if (ind is not None and ind >= 0) else -1.0
    sc[7] = left / 136.0
    sc[8] = turn / 18.0
    sc[9] = (env.players[p_idx].points - 25000) / 20000.0
    sc[10] = (int(seat_wind) - 27) / 3.0
    sc[11] = trigger_type / 4.0
    return pl, sc


def play_kyoku(model, dev, env, model_seat, base_name, seed, temp=0.0):
    """跑完一局（不是半庄）：模型坐 model_seat，其余三席同一基线。"""
    r = env.start_turn()
    pols = {i: POL.make_policy(base_name, seed=seed * 97 + i)
            for i in range(4) if i != model_seat}
    discards_by = np.zeros((4, 34), dtype=np.float32)
    riichi_flags = np.zeros(4, dtype=np.float32)
    meld_cnt = np.zeros(4, dtype=np.float32)
    turn = 0
    steps = 0
    while r is None and steps < 400:
        steps += 1
        acts = {}
        for p in env.pending_players():
            legal = env.legal_actions(p)
            if not legal:
                acts[p] = E.PASS_ACTION
                continue
            if p != model_seat:
                acts[p] = pols[p].choose(env, p, legal)
                continue
            slots = [env_action_to_slot(x, BC.TILED_SLOT_BASE,
                                       BC.PARAM_SLOT, H.TILE_FIELDS)
                     for x in legal]
            if all(s < 0 for s in slots):
                acts[p] = legal[0]
                continue
            pl, sc = encode_live(env, p, discards_by, riichi_flags,
                                meld_cnt, turn, 0, trigger_actor=p,
                                trigger_tile=-1)
            with torch.no_grad():
                logit, _ = model(
                    torch.from_numpy(pl).unsqueeze(0).to(dev),
                    torch.from_numpy(sc).unsqueeze(0).to(dev))
            lg = logit[0].float().cpu().numpy()
            if temp > 0:
                lg = lg + np.random.default_rng(seed + steps).gumbel(0, temp)
            bi_, bv = 0, -1e18
            for i, s in enumerate(slots):
                if s >= 0 and lg[s] > bv:
                    bv, bi_ = lg[s], i
            acts[p] = legal[bi_]
        r = env.step(acts)
        turn += 1
        discards_by[:] = 0
        for i in range(4):
            for t in env.players[i].river:
                discards_by[i, t] += 1
        meld_cnt[:] = 0
        for i in range(4):
            meld_cnt[i] = len(env.players[i].melds)
            if getattr(env.players[i], "riichi", False):
                riichi_flags[i] = 1.0
    return r


def play_hanchan(model, dev, base_name, dealer, seed, temp=0.0, max_kyoku=8):
    """模型固定坐 0 号位，跑一个半庄。返回 (name_table, 是否和了, 点数变化)。"""
    h = HanchanState(dealer=dealer)
    n_win = 0
    n_deal = 0
    pt = [0, 0, 0, 0]
    while not h.finished and n_deal < max_kyoku:
        env = h.new_env(seed=seed + n_deal * 131)
        r = play_kyoku(model, dev, env, 0, base_name,
                       seed=seed + n_deal * 131, temp=temp)
        if r is None:
            n_deal += 1
            continue
        n_deal += 1
        for i in range(4):
            pt[i] += r.points_delta[i]
        if r.result_type != "ryuukyoku" and r.winner == 0:
            n_win += 1
        h.advance(r)
    return n_win, n_deal, pt[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bc_best.pt")
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", default="rules/arena_model.json")
    ap.add_argument("--baseline", default="")
    a = ap.parse_args()

    dev = cpu_only.assert_cpu_only()
    model, ck = load_model(a.ckpt, dev)
    print(f"model {os.path.basename(a.ckpt)} epoch={ck.get('epoch')} "
          f"val_acc={ck.get('val_acc'):.4f} device={dev}", flush=True)

    basel = BASELINES if not a.baseline else a.baseline.split(",")
    results = {}
    for bname in basel:
        t0 = time.time()
        per_batch = []
        W = D = 0
        for bi in range(a.batches):
            # 座位轮换：这一批 4 个半庄，模型坐 0，但庄家走 0..3
            batch_pt = []
            for seat in range(4):
                w, d, p0 = play_hanchan(model, dev, bname, dealer=seat,
                                        seed=700000 + bi * 101 + seat * 7,
                                        temp=a.temp)
                W += w
                D += d
                batch_pt.append(p0)
            per_batch.append(sum(batch_pt) / len(batch_pt))
        mean = sum(per_batch) / len(per_batch)
        var = sum((x - mean) ** 2 for x in per_batch) / (len(per_batch) - 1)
        ci = 1.96 * math.sqrt(var / len(per_batch))
        # 0/N 的 95% 上界（Wilson）
        n_w, n_d = W, max(1, D)
        z = 1.96
        ph = n_w / n_d
        den = 1 + z * z / n_d
        ctr = (ph + z * z / (2 * n_d)) / den
        hw = z * math.sqrt(ph * (1 - ph) / n_d + z * z / (4 * n_d * n_d)) / den
        results[bname] = {
            "mean_pt": mean, "ci_lo": mean - ci, "ci_hi": mean + ci,
            "batches": len(per_batch), "win_rate": ph,
            "win_ci_lo": ctr - hw, "win_ci_hi": ctr + hw,
            "deals": n_d, "secs": time.time() - t0,
            "verdict": "MODEL 强" if mean - ci > 0 else
                       "BASELINE 强" if mean + ci < 0 else "无法区分",
        }
        r_ = results[bname]
        print(f"vs {bname:<15} pt {mean:>8.0f} [{mean-ci:>8.0f},{mean+ci:>8.0f}]  "
              f"win {ph:.3f} [{r_['win_ci_lo']:.3f},{r_['win_ci_hi']:.3f}]  "
              f"n={n_d}  {r_['verdict']}  ({r_['secs']:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"ckpt": a.ckpt, "epoch": ck.get("epoch"),
                   "val_acc": ck.get("val_acc"),
                   "temp": a.temp, "batches": a.batches,
                   "results": results}, f, indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
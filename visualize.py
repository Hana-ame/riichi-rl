"""可视化纯 RL 模型对局：逐轮打印局势 + 模型预测的下一个出牌概率。

用法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python visualize.py --games 1 --seed 1000
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python visualize.py --top 8   # 每步看前8个候选

只看模型「认为下一步该打什么」就够；不加 --sample 即贪心取概率最高者。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import tiles as T
from mahjong.net import MahjongNet
from train_reinforce import slot_of, N_ACTIONS

DEV = "cpu"


def load_model(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = MahjongNet().to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("iters")


def hand_str(hand):
    """手牌（仅 hand，不含副露）转紧凑字符串 123m456p789s1z。"""
    c = [0] * 34
    for t in hand:
        c[t] += 1
    return T.counts_str(c)


def river_str(river, riichi_turn):
    """牌河；立直那张用 * 标记。"""
    out = []
    for i, t in enumerate(river):
        s = T.tile_name(t)
        if riichi_turn >= 0 and i == riichi_turn:
            s += "*"
        out.append(s)
    return " ".join(out)


def meld_str(melds):
    if not melds:
        return ""
    return " ".join(f"{kind}({T.tile_name(t)})" for kind, t, _ in melds)


def render(env, turn_no, model_seat):
    """打印当前局势快照。"""
    left = len(env.wall) - env.wall_idx
    print(f"\n──────── 巡目 {turn_no} | {T.tile_name(env.round_wind)}{env.kyoku}局 "
          f"{env.honba}本场 | 宝牌 {T.tile_name(env.dora_indicator) if env.dora_indicator is not None and env.dora_indicator >= 0 else '?'} "
          f"| 墙余 {left} | 阶段 {env.phase} ────────")
    for p in range(4):
        pl = env.players[p]
        tag = "◀模型" if p == model_seat else "     "
        hand = hand_str(pl.hand)
        n_hand = len(pl.hand)
        flags = []
        if pl.riichi:
            flags.append("立直")
        ms = meld_str(pl.melds)
        print(f"P{p}{tag} 点{pl.points:<6} 手{n_hand:>2} [{hand}]"
              f"{' #' + ','.join(flags) if flags else ''}"
              f"{' 副露:' + ms if ms else ''}")
        print(f"    牌河: {river_str(pl.river, pl.riichi_turn)}")


def predict(model, encoder, env, p, top_n):
    """返回 [(动作文本, 概率)] 降序，及最高概率动作。"""
    planes, scalar = encoder.encode(env, p)
    vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
    legal = env.legal_actions(p)
    if not legal:
        return [], E.PASS_ACTION, 1.0
    slots = [slot_of(a) for a in legal]
    with torch.no_grad():
        logits = model(torch.from_numpy(vec).unsqueeze(0).to(DEV))[0]
    mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32)
    mask[slots] = 0.0
    logits = logits.reshape(-1) + mask.to(DEV)   # reshape [1,54]->[54] 再掩码
    probs = F.softmax(logits, dim=0)

    rows = []
    for a, s in zip(legal, slots):
        name = E.ACT_NAMES.get(a.type, str(a.type))
        tile = T.tile_name(a.tile) if a.tile is not None and a.tile >= 0 else ""
        rows.append((f"{name}{('(' + tile + ')') if tile else ''}", float(probs[s]), a))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top_n], rows[0][2], rows[0][1]


def play(model, encoder, seed, top_n, sample=False, temp=1.0):
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    result = env.start_turn()
    model_seat = 0  # 模型坐 0 位，逐轮详细渲染
    steps = turn_no = 0
    while result is None and steps < 400:
        steps += 1
        actions = {}
        for p in env.pending_players():
            rows, best, best_p = predict(model, encoder, env, p, top_n)
            if not rows:
                actions[p] = E.PASS_ACTION
                continue
            if p == model_seat:
                render(env, turn_no, model_seat)
                print(f"  ▶ P{p} 模型预测（前{len(rows)}）：")
                for i, (txt, pr, _) in enumerate(rows, 1):
                    bar = "█" * max(1, int(pr * 30))
                    print(f"    {i:>2}. {txt:<16} {pr*100:5.1f}% {bar}")
            if sample:
                prs = np.array([r[1] for r in rows])
                pick_row = rows[int(np.random.choice(len(rows), p=prs / prs.sum()))]
                actions[p] = pick_row[2]
            else:
                actions[p] = best
        result = env.step(actions)
        if env.phase == E.Phase.DISCARD:
            turn_no += 1
    return env, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/riichi_net.pt")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--sample", action="store_true", help="按概率采样而非贪心")
    args = ap.parse_args()

    encoder = O2.ObsEncoder()
    for g in range(args.games):
        model, iters = load_model(args.ckpt)
        print(f"=== 模型 {args.ckpt} (iters={iters}) | 局 {g} seed={args.seed+g} ===")
        seed = args.seed + g
        _, result = play(model, encoder, seed, args.top, sample=args.sample)
        if result is None:
            print("\n未结束")
            continue
        print(f"\n=== 结果：{result.result_type} "
              f"winner={result.winner} from={result.from_player} "
              f"役={result.yaku} {result.han}番{result.fu}符 ===")
        print(f"点数变化：{result.points_delta}")


if __name__ == "__main__":
    main()

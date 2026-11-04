"""让 SenseNova-6.8-flash-lite（大语言模型）读牌面文字、自己推理出牌。

原理：把当前局势渲染成中文 prompt → 发给 LLM → 它从「合法动作编号」里选一个
     → 解析回引擎 Action → 执行。LLM 答非所问时兜底取第一个合法动作。

用法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python llm_player.py --games 1 --seed 1000
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python llm_player.py --opp greedy_tenpai

LLM 只打 0 位；其余三家用项目自带规则基线（random/greedy_tenpai/call_first/yaochu_first）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import tiles as T
from mahjong import policies as POL
from mahjong.hanchan import HanchanState

BASE_URL = "https://token.sensenova.cn/v1"
MODEL = "sensenova-6.8-flash-lite"
CRED = "/home/lumin/.dsh/plugins/dsh-sensenova-keypool/settings.json"


def get_key():
    d = json.load(open(CRED))
    return d["keys"][0]["key"]


# ---------- 局面 → 文字 ----------

def hand_str(hand):
    c = [0] * 34
    for t in hand:
        c[t] += 1
    return T.counts_str(c)


def river_str(river):
    return " ".join(T.tile_name(t) for t in river) or "（无）"


def action_label(a):
    name = E.ACT_NAMES.get(a.type, str(a.type))
    cn = {"DISCARD": "打", "RIICHI": "立直打", "TSUMO": "自摸和", "RON": "荣和",
          "PON": "碰", "CHI": "吃", "ANKAN": "暗杠", "ADD_KAN": "加杠",
          "RYUUKYOKU": "流局", "NAGASHI": "流局宣言", "PASS": "跳过"}
    tile = T.tile_name(a.tile) if a.tile is not None and a.tile >= 0 else ""
    base = cn.get(name, name)
    return f"{base} {tile}".strip()


def situation_text(env, p):
    left = len(env.wall) - env.wall_idx
    pl = env.players[p]
    lines = [f"东/南{env.kyoku}局 {env.honba}本场，第 "
             f"{18 - left // 4} 巡左右，宝牌指示牌 {T.tile_name(env.dora_indicator) if env.dora_indicator is not None and env.dora_indicator >= 0 else '?'}，墙余 {left}。"]
    for i in range(4):
        q = env.players[i]
        who = "你" if i == p else f"对手{i}"
        riichi = "【已立直】" if q.riichi else ""
        hand = hand_str(q.hand) if i == p else f"{len(q.hand)}张(看不到)"
        melds = (" 副露:" + " ".join(T.tile_name(t) for _, t, _ in q.melds)) if q.melds else ""
        lines.append(f"  {who} 点{q.points} {riichi}{melds} 牌河: {river_str(q.river)}"
                     + (f" 手牌: {hand}" if i == p else ""))
    return "\n".join(lines)


# ---------- LLM 调用 ----------

def ask_llm(key, situation, choices):
    sysmsg = ("你是日本立直麻将高手。只根据局面和合法动作做选择，"
              "最终必须只回复一个动作的编号数字（阿拉伯数字），不要解释、不要多余字。")
    user = (situation + "\n\n你可执行的动作：\n" + choices
            + "\n\n只回复编号数字：")
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": user}],
            "stream": False, "max_tokens": 2048, "temperature": 0.2}
    req = urllib.request.Request(BASE_URL + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json",
                                          "User-Agent": "mahjong-llm/1.0"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return r["choices"][0]["message"].get("content", "").strip()


def parse_choice(reply, n_legal):
    """从回复里提取第一个 1..n_legal 的编号；失败返回 None。"""
    m = re.search(r"\b(\d+)\b", reply.replace("（", "("))
    if not m:
        return None
    v = int(m.group(1))
    return v if 1 <= v <= n_legal else None


# ---------- 对局 ----------

def play_game(llm_seat, opp_name, key, seed, verbose=True):
    h = HanchanState(dealer=seed % 4)
    env = h.new_env(seed=seed)
    result = env.start_turn()
    turn = 0
    while result is None and turn < 8:
        steps = 0
        while result is None and steps < 400:
            steps += 1
            actions = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    actions[p] = E.PASS_ACTION
                    continue
                if len(legal) == 1:
                    actions[p] = legal[0]
                    continue
                if p == llm_seat:
                    choices = "\n".join(f"  {i}. {action_label(a)}"
                                        for i, a in enumerate(legal, 1))
                    sit = situation_text(env, p)
                    reply = ask_llm(key, sit, choices)
                    idx = parse_choice(reply, len(legal))
                    if idx is None:
                        idx = 1  # 兜底：第一个合法动作
                    act = legal[idx - 1]
                    if verbose:
                        print(f"  [P{p} 巡{turn}] 选 {idx}: {action_label(act)}"
                              f"  (手牌 {hand_str(env.players[p].hand)})"
                              f"  LLM回:{reply[:20]!r}", flush=True)
                    actions[p] = act
                else:
                    actions[p] = POL.make_policy(opp_name, seed=seed * 97 + p).choose(env, p, legal)
            result = env.step(actions)
            if env.phase == E.Phase.DISCARD:
                turn += 1
        if result is None:
            break
        h.advance(result)
        if not h.finished:
            env = h.new_env(seed=seed + turn * 131)
            result = env.start_turn()
    return result, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--opp", default="greedy_tenpai",
                    choices=["random", "greedy_tenpai", "call_first", "yaochu_first"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    key = get_key()
    print(f"模型 {MODEL} 坐 0 位 | 对手 {args.opp} | seed={args.seed}")
    result, h = play_game(0, args.opp, key, args.seed, verbose=not args.quiet)
    if result is None:
        print("未结束")
        return
    print(f"\n=== 结果 {result.result_type} winner={result.winner} "
          f"from={result.from_player} 役={result.yaku} {result.han}番{result.fu}符 ===")
    print(f"点数变化：{result.points_delta}")


if __name__ == "__main__":
    main()

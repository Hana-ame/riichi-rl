"""WebSocket 服务器：SenseNova-6.8-flash-lite 打麻将，实时推送给浏览器看。

跑法（后台）：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python server.py
然后浏览器打开 index.html（连 ws://127.0.0.1:8765）。

浏览器发命令：
    {"cmd":"start","seed":1000,"opp":"greedy_tenpai"}
    {"cmd":"pause"} / {"cmd":"resume"} / {"cmd":"step"}
    {"cmd":"speed","ms":800}

LLM 只打 0 位；其余三家用规则基线。每步推 state + decision 两类消息。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets

from mahjong import env as E
from mahjong import tiles as T
from mahjong import policies as POL
from mahjong.hanchan import HanchanState

BASE_URL = "https://token.sensenova.cn/v1"
MODEL = "sensenova-6.8-flash-lite"
CRED = "/home/lumin/.dsh/plugins/dsh-sensenova-keypool/settings.json"

STATE = {"paused": False, "step_once": False, "sleep": 0.6}


def get_key():
    return json.load(open(CRED))["keys"][0]["key"]


# ---------- 渲染 / 编码 ----------

def hand_list(hand):
    return [T.tile_name(t) for t in hand]


def river_list(river):
    return [T.tile_name(t) for t in river]


def meld_list(melds):
    return [{"kind": k, "tile": T.tile_name(t)} for k, t, _ in melds]


def action_label(a):
    name = E.ACT_NAMES.get(a.type, str(a.type))
    cn = {"DISCARD": "打", "RIICHI": "立直打", "TSUMO": "自摸和", "RON": "荣和",
          "PON": "碰", "CHI": "吃", "ANKAN": "暗杠", "ADD_KAN": "加杠",
          "RYUUKYOKU": "流局", "NAGASHI": "流局宣言", "PASS": "跳过"}
    tile = T.tile_name(a.tile) if a.tile is not None and a.tile >= 0 else ""
    return (cn.get(name, name) + " " + tile).strip()


def snapshot(env, turn):
    return {
        "type": "state",
        "kyoku": env.kyoku, "honba": env.honba,
        "round_wind": T.tile_name(env.round_wind),
        "dora": T.tile_name(env.dora_indicator) if env.dora_indicator is not None and env.dora_indicator >= 0 else "?",
        "wall_left": len(env.wall) - env.wall_idx,
        "turn": turn, "phase": env.phase,
        "players": [{
            "seat": i, "points": pl.points,
            "hand": hand_list(pl.hand), "river": river_list(pl.river),
            "riichi": pl.riichi, "melds": meld_list(pl.melds),
        } for i, pl in enumerate(env.players)],
    }


def situation_text(env, p):
    left = len(env.wall) - env.wall_idx
    lines = [f"{T.tile_name(env.round_wind)}{env.kyoku}局 {env.honba}本场，约第 {18 - left // 4} 巡，"
             f"宝牌指示牌 {snapshot(env, 0)['dora']}，墙余 {left}。"]
    for i in range(4):
        q = env.players[i]
        who = "你" if i == p else f"对手{i}"
        riichi = "【已立直】" if q.riichi else ""
        melds = (" 副露:" + " ".join(T.tile_name(t) for _, t, _ in q.melds)) if q.melds else ""
        hand = (" 手牌: " + "".join(T.tile_name(t) for t in q.hand)) if i == p else ""
        lines.append(f"  {who} 点{q.points} {riichi}{melds} 牌河:{' '.join(T.tile_name(t) for t in q.river) or '无'}{hand}")
    return "\n".join(lines)


# ---------- LLM ----------

def ask_llm_sync(key, situation, choices):
    sysmsg = ("你是日本立直麻将高手。根据局面和给出的合法动作做选择。"
              "严格按两行回复：第一行只写动作的编号数字，第二行用一句话说明理由。")
    user = situation + "\n\n你可执行的动作：\n" + choices + "\n\n按格式回复（第一行编号，第二行理由）："
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": user}],
            "stream": False, "max_tokens": 65535, "temperature": 0.2}
    req = urllib.request.Request(BASE_URL + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json",
                                          "User-Agent": "mahjong-llm/1.0"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return r["choices"][0]["message"].get("content", "").strip()


def parse_choice(reply, n_legal):
    m = re.search(r"\b(\d+)\b", reply)
    if not m:
        return None
    v = int(m.group(1))
    return v if 1 <= v <= n_legal else None


# ---------- 对局循环 ----------

async def run_game(ws, cmd, key):
    seed = int(cmd.get("seed", 1000))
    opp = cmd.get("opp", "greedy_tenpai")
    STATE.update(paused=False, step_once=False)
    await ws.send(json.dumps({"type": "info", "msg": f"开始 seed={seed} 对手={opp} 模型={MODEL}"}))
    h = HanchanState(dealer=seed % 4)
    env = h.new_env(seed=seed)
    result = env.start_turn()
    turn = 0
    while result is None and turn < 8:
        steps = 0
        while result is None and steps < 400:
            await gate()
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
                if p == 0:
                    choices = "\n".join(f"  {i}. {action_label(a)}" for i, a in enumerate(legal, 1))
                    sit = situation_text(env, p)
                    try:
                        reply = await asyncio.to_thread(ask_llm_sync, key, sit, choices)
                    except Exception as ex:
                        reply = f"1 (LLM 出错兜底: {type(ex).__name__})"
                    idx = parse_choice(reply, len(legal)) or 1
                    act = legal[idx - 1]
                    lines = [l for l in reply.splitlines() if l.strip()]
                    reason = lines[1].strip() if len(lines) > 1 else ""
                    await ws.send(json.dumps({
                        "type": "decision", "player": p, "turn": turn,
                        "options": [action_label(a) for a in legal],
                        "chosen": action_label(act), "chosen_idx": idx,
                        "llm_reply": reply[:200], "reason": reason,
                        "hand": hand_list(env.players[p].hand),
                    }, ensure_ascii=False))
                    actions[p] = act
                else:
                    actions[p] = POL.make_policy(opp, seed=seed * 97 + p).choose(env, p, legal)
            result = env.step(actions)
            if env.phase == E.Phase.DISCARD:
                turn += 1
            await ws.send(json.dumps(snapshot(env, turn), ensure_ascii=False))
            await asyncio.sleep(STATE["sleep"])
        if result is None:
            break
        await ws.send(json.dumps({
            "type": "kyoku_end", "result_type": result.result_type,
            "winner": result.winner, "from": result.from_player,
            "yaku": result.yaku, "han": result.han, "fu": result.fu,
            "delta": result.points_delta, "scores": h.scores,
        }, ensure_ascii=False))
        h.advance(result)
        if not h.finished:
            env = h.new_env(seed=seed + turn * 131)
            result = env.start_turn()
            await asyncio.sleep(0.8)
    await ws.send(json.dumps({"type": "game_over", "scores": h.scores}, ensure_ascii=False))


async def gate():
    """暂停/单步控制。"""
    while STATE["paused"] and not STATE["step_once"]:
        await asyncio.sleep(0.1)
    STATE["step_once"] = False


async def handler(ws):
    key = get_key()
    async for msg in ws:
        try:
            cmd = json.loads(msg)
        except Exception:
            continue
        c = cmd.get("cmd")
        if c == "start":
            asyncio.create_task(run_game(ws, cmd, key))
        elif c == "pause":
            STATE["paused"] = True
            await ws.send(json.dumps({"type": "info", "msg": "已暂停"}))
        elif c == "resume":
            STATE["paused"] = False
            await ws.send(json.dumps({"type": "info", "msg": "继续"}))
        elif c == "step":
            STATE["step_once"] = True
        elif c == "speed":
            STATE["sleep"] = max(0, int(cmd.get("ms", 600)) / 1000)


async def main():
    async with websockets.serve(handler, "127.0.0.1", 8765, ping_interval=None):
        print("server ready ws://127.0.0.1:8765", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

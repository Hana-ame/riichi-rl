"""LLM 麻将实时观战服务器。

- WS  : ws://localhost:8775      推送每步局面 + LLM 决策
- HTTP: http://localhost:8776    托管 mahjong_view.html

浏览器打开 http://localhost:8766/mahjong_view.html 即可一边跑一边看。

跑法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python serve_game.py --seed 1000 --opp greedy_tenpai
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler

import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np             # noqa: E402
import cpu_only                # noqa: E402,F401 —— 必须先于 torch 锁 CPU-only
import torch                   # noqa: E402

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import policies as POL
from mahjong.hanchan import HanchanState
from mahjong.net import MahjongNet
from train_reinforce import slot_of, N_ACTIONS
from train_hanchan import hand_shanten
from opponent_model import OpponentNet, build_features
from hand_predictor import HandPredictor, build_feature as build_hfeature
import llm_player as LP

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIENTS: set = set()
PAUSED = False
STEP_ONCE = False

REALTIME_BUDGET_MS = 5000.0     # 单步决策预算（用户要求 5 秒内 realtime）

POLICY_NAMES = {"llm", "net", "random", "greedy_tenpai", "damaten",
                "riichi_needed", "riichi_ev", "call_first", "yaochu_first"}
# 每家策略，观战途中可通过界面下拉随时切换（下一生效决策点即用新策略）
PER_SEAT = {0: "greedy_tenpai", 1: "call_first", 2: "call_first", 3: "call_first"}

# ---- 你的神经网络模型（riichi_net.pt，54 槽 convformer_v1）----
_NET = None
_NET_ENC = None
NET_CKPT = os.path.join(ROOT, "checkpoints", "riichi_net.pt")

# ---- 对弈数据记录 ----
# 每个决策点存：(网络观测, 动作槽, 座位, 策略) + 牌面快照（真实手牌/牌河/副露/立直）
# 前者训策略（蒸馏/离线RL），后者训「对手手牌推断模型」（有监督，真实手牌当标签）。
REC_DIR = os.path.join(ROOT, "data", "selfplay")
os.makedirs(REC_DIR, exist_ok=True)
_REC_ENC = O2.ObsEncoder()
_REC = {"obs": [], "act": [], "seat": [], "pol": [],
        "hands": [], "rivers": [], "melds": [], "riichi": [],
        "left": [], "dora": [], "tsumogiri": [], "shanten": []}
def _next_game_seq():
    """重启服务器时从已有文件名续号，避免覆盖旧数据。"""
    import re as _re
    mx = 0
    try:
        for fn in os.listdir(REC_DIR):
            m = _re.match(r"g(\d+)\.npz$", fn)
            if m:
                mx = max(mx, int(m.group(1)))
    except Exception:
        pass
    return mx + 1


_GAME_SEQ = [_next_game_seq()]
# 每家「最近一次打牌是否摸切」，用于牌河标记（摸切=刚摸的牌直接打出）
_TSUMO_LAST = {0: None, 1: None, 2: None, 3: None}


def rec_step(env, p, act, pol):
    """记录一个决策点；slot 无法映射的动作（-1）跳过。"""
    s = slot_of(act)
    if s is None or s < 0:
        return
    planes, scalar = _REC_ENC.encode(env, p)
    vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
    _REC["obs"].append(vec)
    _REC["act"].append(int(s))
    _REC["seat"].append(int(p))
    _REC["pol"].append(pol)

    # 牌面快照（对手手牌推断的输入输出）
    hands = np.zeros((4, 34), dtype=np.int8)
    rivers = np.zeros((4, 34), dtype=np.int8)
    meldc = np.zeros(4, dtype=np.int8)
    rich = np.zeros(4, dtype=np.int8)
    for i in range(4):
        pl = env.players[i]
        for t in pl.hand:
            hands[i, t] += 1
        for t in pl.river:
            rivers[i, t] += 1
        meldc[i] = len(pl.melds)
        rich[i] = 1 if pl.riichi else 0
    di = env.dora_indicator
    # 摸切判断：打出的牌正好是刚摸的那张（14 张时最后一张）
    pl0 = env.players[p]
    tsg = 0
    if (act.type in (E.ACT_DISCARD, E.ACT_RIICHI) and len(pl0.hand) == 14
            and act.tile == pl0.hand[-1]):
        tsg = 1
    _REC["tsumogiri"].append(tsg)
    if act.type in (E.ACT_DISCARD, E.ACT_RIICHI):
        _TSUMO_LAST[p] = bool(tsg)

    _REC["hands"].append(hands)
    _REC["rivers"].append(rivers)
    _REC["melds"].append(meldc)
    _REC["riichi"].append(rich)
    _REC["left"].append(len(env.wall) - env.wall_idx)
    _REC["dora"].append(int(di) if (di is not None and di >= 0) else -1)
    try:
        sh = int(hand_shanten(env, p))
    except Exception:
        sh = 99
    _REC["shanten"].append(sh)


def flush_rec(result, seed):
    """一局结束，把这局所有决策点 + 终局结果存成 npz。"""
    if not _REC["obs"]:
        return
    _GAME_SEQ[0] += 1
    path = os.path.join(REC_DIR, f"g{_GAME_SEQ[0]:05d}.npz")
    np.savez_compressed(
        path,
        obs=np.stack(_REC["obs"]),
        act=np.array(_REC["act"], dtype=np.int64),
        seat=np.array(_REC["seat"], dtype=np.int64),
        pol=np.array(_REC["pol"]),
        hands=np.stack(_REC["hands"]),
        rivers=np.stack(_REC["rivers"]),
        melds=np.stack(_REC["melds"]),
        riichi=np.stack(_REC["riichi"]),
        left=np.array(_REC["left"], dtype=np.int64),
        dora=np.array(_REC["dora"], dtype=np.int64),
        tsumogiri=np.array(_REC["tsumogiri"], dtype=np.int8),
        shanten=np.array(_REC["shanten"], dtype=np.int8),
        result_type=result.result_type if result else "unfinished",
        winner=result.winner if result else -1,
        han=result.han if result else 0,
        fu=result.fu if result else 0,
        points_delta=np.array(result.points_delta if result else [0, 0, 0, 0]),
        seed=seed,
    )
    n = len(_REC["obs"])
    for k in _REC:
        _REC[k] = []
    for i in range(4):
        _TSUMO_LAST[i] = None
    print(f"[rec] {path}  {n} 决策点", flush=True)


def get_net():
    """懒加载神经网络，只加载一次。"""
    global _NET, _NET_ENC
    if _NET is None:
        ck = torch.load(NET_CKPT, map_location="cpu", weights_only=False)
        _NET = MahjongNet()
        _NET.load_state_dict(ck["state_dict"])
        _NET.eval()
        _NET_ENC = O2.ObsEncoder()
    return _NET, _NET_ENC


def net_decide(env, p, legal):
    """神经网络推理 + 规则接管和牌/立直。

    纯蒸馏模型实测从不主动立直/和牌（立直样本占比低、argmax 永远不选），
    所以和牌与立直由规则保证，模型只负责普通打牌。
    """
    for a in legal:
        if a.type in (E.ACT_TSUMO, E.ACT_RON):
            return a
    riichis = [a for a in legal if a.type == E.ACT_RIICHI]
    if riichis:
        return riichis[0]
    net, enc = get_net()
    planes, scalar = enc.encode(env, p)
    vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
    slots = [slot_of(a) for a in legal]
    with torch.no_grad():
        logits = net(torch.from_numpy(vec).unsqueeze(0))[0].reshape(-1)
    mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32)
    mask[slots] = 0.0
    pick = int((logits + mask).argmax())
    if pick not in slots:
        pick = slots[0]
    return legal[slots.index(pick)]


def net_probs(env, p, legal, top=5):
    """NET 对各合法动作的 softmax 概率，降序 top-N：[(label, prob)]。
    LLM/规则决策时也拿它当「概率参考」写在日志里。"""
    net, enc = get_net()
    planes, scalar = enc.encode(env, p)
    vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
    slots = [slot_of(a) for a in legal]
    with torch.no_grad():
        logits = net(torch.from_numpy(vec).unsqueeze(0))[0].reshape(-1)
    mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32)
    mask[slots] = 0.0
    probs = torch.softmax(logits + mask, dim=0).numpy()
    rows = [(LP.action_label(legal[i]), float(probs[slots[i]]))
            for i in range(len(legal))]
    rows.sort(key=lambda r: -r[1])
    return rows[:top]


def remain_counts(env):
    """每张牌的剩余张数（34 维）：4 - 牌河 - 副露代表牌 - 宝牌指示。
    粗估版对手手牌预测的基础（副露只计代表牌，偏乐观）。"""
    vis = np.zeros(34, dtype=np.int64)
    for i in range(4):
        for t in env.players[i].river:
            vis[t] += 1
        for _kind, t, _arg in env.players[i].melds:
            vis[t] += 1
    di = env.dora_indicator
    if di is not None and di >= 0:
        vis[di] += 1
    return (4 - vis).clip(0).tolist()


# ---- 和牌率统计（整场观战累计）----
STATS = {i: {"win": 0, "deal": 0, "pt": 0} for i in range(4)}

# ---- 对手手牌推断模型（热更新：checkpoints/opponent_model.pt 变了就重载）----
_OPP_MODEL = {"m": None, "mtime": None}
OPP_CKPT = os.path.join(ROOT, "checkpoints", "opponent_model.pt")


def get_opp_model():
    if not os.path.exists(OPP_CKPT):
        return None
    mt = os.path.getmtime(OPP_CKPT)
    if _OPP_MODEL["mtime"] != mt:
        try:
            ck = torch.load(OPP_CKPT, map_location="cpu", weights_only=False)
            m = OpponentNet()
            m.load_state_dict(ck["state_dict"])
            m.eval()
            _OPP_MODEL["m"] = m
            _OPP_MODEL["mtime"] = mt
            print(f"[opp] 加载对手推断模型 samples={ck.get('samples')} "
                  f"val_loss={ck.get('val_loss'):.4f}", flush=True)
        except Exception as e:
            print(f"[opp] 加载失败: {e}", flush=True)
            return None
    return _OPP_MODEL["m"]


def opp_predictions(env, observer=0):
    """从 observer 视角预测另外三家每张牌的持有概率，返回 {seat: [34]}。"""
    m = get_opp_model()
    if m is None:
        return None
    out = {}
    hands = np.zeros((4, 34), dtype=np.int8)
    rivers = np.zeros((4, 34), dtype=np.int8)
    for i in range(4):
        pl = env.players[i]
        for t in pl.hand:
            hands[i, t] += 1
        for t in pl.river:
            rivers[i, t] += 1
    left = len(env.wall) - env.wall_idx
    di = env.dora_indicator
    dora = int(di) if (di is not None and di >= 0) else -1
    feats, seats = [], []
    for t in range(4):
        if t == observer:
            continue
        other = np.zeros(34, dtype=np.int8)
        for i in range(4):
            if i != observer and i != t:
                other += rivers[i]
        feats.append(build_features(
            hands[observer], rivers[t], other,
            len(env.players[t].melds), 1 if env.players[t].riichi else 0,
            int(rivers[t].sum()), left, dora))
        seats.append(t)
    with torch.no_grad():
        pr = m(torch.from_numpy(np.stack(feats))).numpy()
    for k, t in enumerate(seats):
        out[t] = [round(float(x), 3) for x in pr[k]]
    return out


# ---- 和牌预测器（热更新）----
_HAND_MODEL = {"m": None, "mtime": None}
HAND_CKPT = os.path.join(ROOT, "checkpoints", "hand_predictor.pt")


def get_hand_model():
    if not os.path.exists(HAND_CKPT):
        return None
    mt = os.path.getmtime(HAND_CKPT)
    if _HAND_MODEL["mtime"] != mt:
        try:
            ck = torch.load(HAND_CKPT, map_location="cpu", weights_only=False)
            m = HandPredictor()
            m.load_state_dict(ck["state_dict"])
            m.eval()
            _HAND_MODEL["m"] = m
            _HAND_MODEL["mtime"] = mt
            print(f"[hand] 加载和牌预测器 samples={ck.get('samples')}", flush=True)
        except Exception as e:
            print(f"[hand] 加载失败: {e}", flush=True)
            return None
    return _HAND_MODEL["m"]


def hand_predictions(env):
    """预测四家各自：和牌概率 / 预期点数 / 还需几巡听牌。"""
    m = get_hand_model()
    if m is None:
        return None
    left = len(env.wall) - env.wall_idx
    di = env.dora_indicator
    dora = int(di) if (di is not None and di >= 0) else -1
    feats = []
    for i in range(4):
        pl = env.players[i]
        c = np.zeros(34, dtype=np.int8)
        for t in pl.hand:
            c[t] += 1
        try:
            sh = int(hand_shanten(env, i))
        except Exception:
            sh = 8
        feats.append(build_hfeature(c, len(pl.melds),
                                    1 if pl.riichi else 0, sh, left, dora))
    with torch.no_grad():
        pw, pp, pt = m(torch.from_numpy(np.stack(feats)))
    out = {}
    for i in range(4):
        out[i] = {"win": round(float(pw[i, 0]), 3),
                  "pts": int(pp[i].item()),
                  "turns": round(float(pt[i].item()), 1)}
    return out

# ---------- 局面 → JSON ----------

def player_json(p):
    return {
        "hand": list(p.hand),
        "river": list(p.river),
        "melds": [[k, t, a] for k, t, a in p.melds],
        "riichi": bool(p.riichi),
        "points": p.points,
        "riichi_turn": p.riichi_turn,
    }


def state_json(env, turn, result=None):
    left = len(env.wall) - env.wall_idx
    d = {
        "type": "state", "kyoku": env.kyoku, "honba": env.honba,
        "turn": turn, "left": left, "phase": env.phase,
        "dora": env.dora_indicator if (env.dora_indicator is not None and env.dora_indicator >= 0) else -1,
        "doras": list(getattr(env, "dora_indicators", []) or []),
        "round_wind": int(env.round_wind) - 27,
        "per_seat": [PER_SEAT.get(i, "greedy_tenpai") for i in range(4)],
        "stats": [dict(STATS[i]) for i in range(4)],
        "remain": remain_counts(env),
        "opred": opp_predictions(env),
        "hpred": hand_predictions(env),
        "tsumo_last": [_TSUMO_LAST[i] for i in range(4)],
        # 当前轮到谁（打牌阶段=该摸牌打牌者；响应阶段=第一个响应者）
        "current": (int(env.turn_player) if getattr(env, "turn_player", None) is not None
                    else -1),
        "players": [player_json(env.players[i]) for i in range(4)],
    }
    if result is not None:
        d["result"] = {
            "result_type": result.result_type, "winner": result.winner,
            "from_player": result.from_player, "yaku": result.yaku,
            "han": result.han, "fu": result.fu,
            "points_delta": list(result.points_delta),
        }
    return d


def decision_json(seat, turn, phase, picked_text, think, acts,
                  probs=None, shanten=None, tag=""):
    d = {"type": "decision", "seat": seat, "turn": turn, "phase": phase,
         "picked_text": picked_text, "think": think, "acts": acts}
    if probs:
        d["probs"] = [{"text": t, "p": round(pr, 3)} for t, pr in probs]
    if shanten is not None:
        d["shanten"] = int(shanten)
    if tag:
        d["tag"] = tag
    return d


# ---------- LLM（同步阻塞，放线程池跑） ----------

def ask_llm2(key, situation, choices):
    sysmsg = ("你是日本立直麻将高手。只根据局面和合法动作做选择，"
              "最终必须只回复一个动作的编号数字（阿拉伯数字），不要解释、不要多余字。")
    user = (situation + "\n\n你可执行的动作：\n" + choices + "\n\n只回复编号数字：")
    body = {"model": LP.MODEL,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": user}],
            "stream": False, "max_tokens": 2048, "temperature": 0.2}
    req = urllib.request.Request(LP.BASE_URL + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json",
                                          "User-Agent": "mahjong-llm/1.0"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    m = r["choices"][0]["message"]
    content = m.get("content", "")
    think = m.get("reasoning_content", "") or ""
    return content.strip(), think


# ---------- 广播 ----------

async def broadcast(msg):
    if not CLIENTS:
        return
    data = json.dumps(msg, ensure_ascii=False)
    dead = []
    for ws in list(CLIENTS):
        try:
            await ws.send(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


async def hold():
    """暂停时挂起；单步放行一步后继续挂起。"""
    global STEP_ONCE
    while PAUSED and not STEP_ONCE:
        await asyncio.sleep(0.08)
    STEP_ONCE = False


# ---------- 对局主循环 ----------

async def run_game(key, seed, step_delay):
    loop = asyncio.get_event_loop()
    h = HanchanState(dealer=seed % 4)
    env = h.new_env(seed=seed)
    result = env.start_turn()
    await broadcast(state_json(env, 0))
    turn = 0
    deals = 0          # 半庄内已打完的局数（旧代码误用 turn 当上限：
    #                    turn 是决策计数，打完第一局就 >8，外层直接退出，
    #                    观战永远只看得到一局）
    while result is None and deals < 8:
        steps = 0
        while result is None and steps < 400:
            steps += 1
            await hold()
            actions = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    actions[p] = E.PASS_ACTION
                    continue
                if len(legal) == 1:
                    actions[p] = legal[0]
                    continue
                pol = PER_SEAT.get(p, "greedy_tenpai")
                _t0 = time.perf_counter()
                if pol == "llm":
                    choices = "\n".join(f"  {i}. {LP.action_label(a)}"
                                        for i, a in enumerate(legal, 1))
                    sit = LP.situation_text(env, p)
                    content, think = await loop.run_in_executor(
                        None, ask_llm2, key, sit, choices)
                    idx = LP.parse_choice(content, len(legal))
                    if idx is None:
                        idx = 1
                    act = legal[idx - 1]
                    probs = await loop.run_in_executor(None, net_probs, env, p, legal)
                    await broadcast(decision_json(
                        p, turn, env.phase, LP.action_label(act),
                        think or content,
                        ", ".join(LP.action_label(a) for a in legal[:12]),
                        probs=probs, tag="LLM"))
                elif pol == "net":
                    probs = net_probs(env, p, legal)
                    act = net_decide(env, p, legal)
                    await broadcast(decision_json(
                        p, turn, env.phase, LP.action_label(act),
                        "神经网络 argmax（riichi_net.pt）",
                        ", ".join(LP.action_label(a) for a in legal[:12]),
                        probs=probs, tag="NET"))
                else:
                    act = POL.make_policy(pol, seed=seed * 97 + p).choose(env, p, legal)
                    sh = hand_shanten(env, p)
                    await broadcast(decision_json(
                        p, turn, env.phase, LP.action_label(act),
                        "", "", probs=net_probs(env, p, legal),
                        shanten=sh, tag=pol))
                _ms = (time.perf_counter() - _t0) * 1000.0
                if _ms > REALTIME_BUDGET_MS:
                    print(f"[realtime] seat{p} 单步 {_ms:.0f}ms 超 "
                          f"{REALTIME_BUDGET_MS:.0f}ms 预算，退回规则基线",
                          flush=True)
                    act = POL.make_policy("greedy_tenpai",
                                          seed=seed * 97 + p).choose(
                                              env, p, legal)
                actions[p] = act
                rec_step(env, p, actions[p], pol)   # 记录这步决策（蒸馏/离线RL用）
            result = env.step(actions)
            await broadcast(state_json(env, turn))
            await asyncio.sleep(step_delay)
            if env.phase == E.Phase.DISCARD:
                turn += 1
        if result is not None:
            await broadcast(state_json(env, turn, result))
            flush_rec(result, seed + turn * 131)
            net = E.net_points(result, env.initial_points)
            for i in range(4):
                STATS[i]["deal"] += 1
                STATS[i]["pt"] += net[i]
            if result.result_type != "ryuukyoku" and result.winner is not None and result.winner >= 0:
                STATS[result.winner]["win"] += 1
        if result is None:
            break
        deals += 1
        h.advance(result)
        if not h.finished:
            env = h.new_env(seed=seed + turn * 131)
            result = env.start_turn()
            await broadcast(state_json(env, turn))
    print("[game over]", result.result_type if result else "未结束", flush=True)


# ---------- WS / HTTP ----------

async def handler(ws):
    CLIENTS.add(ws)
    try:
        async for raw in ws:
            try:
                d = json.loads(raw)
            except Exception:
                continue
            global PAUSED, STEP_ONCE
            c = d.get("ctrl")
            if c == "pause":
                PAUSED = True
            elif c == "start":
                PAUSED = False
            elif c == "step":
                PAUSED = True
                STEP_ONCE = True
            elif c == "policy":
                # 中途切换某家策略：{"ctrl":"policy","seat":N,"name":"llm|random|..."}
                try:
                    seat = int(d.get("seat", -1))
                    name = d.get("name", "")
                except Exception:
                    continue
                if 0 <= seat <= 3 and name in POLICY_NAMES:
                    PER_SEAT[seat] = name
                    await broadcast({"type": "per_seat",
                                     "per_seat": [PER_SEAT.get(i) for i in range(4)]})
    finally:
        CLIENTS.discard(ws)


def serve_http():
    os.chdir(ROOT)
    HTTPServer(("0.0.0.0", 8776), SimpleHTTPRequestHandler).serve_forever()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--opp", default="call_first",
                    choices=["net", "random", "greedy_tenpai", "damaten",
                              "riichi_needed", "riichi_ev", "call_first",
                              "yaochu_first"],
                    help="P1-3 初始策略；call_first 会主动吃碰（默认，便于观察副露）")
    ap.add_argument("--p0", default="greedy_tenpai",
                    choices=["net", "llm", "random", "greedy_tenpai",
                              "damaten", "riichi_needed", "riichi_ev",
                              "call_first", "yaochu_first"],
                    help="P0 初始策略（默认 net 你的模型；LLM 需显式指定）")
    ap.add_argument("--delay", type=float, default=0.35)
    args = ap.parse_args()

    from cpu_only import set_inference_threads
    set_inference_threads(2)
    key = LP.get_key()
    PER_SEAT.update({0: args.p0, 1: args.opp, 2: args.opp, 3: args.opp})
    threading.Thread(target=serve_http, daemon=True).start()
    async with websockets.serve(handler, "0.0.0.0", 8775,
                                ping_interval=20, ping_timeout=20,
                                max_size=2 ** 20):
        print("=" * 56)
        print("▶ 观战页面:  http://localhost:8776/mahjong_view.html")
        print("  WS 推送 :  ws://localhost:8775")
        print(f"  初始: P0={args.p0}, P1-3={args.opp}（默认不含 LLM；界面可切）| seed={args.seed}")
        print("  等待浏览器连入…", flush=True)
        t0 = asyncio.get_event_loop().time()
        while not CLIENTS:
            if asyncio.get_event_loop().time() - t0 > 300:
                print("  300s 无人连入，仍继续开跑", flush=True)
                break
            await asyncio.sleep(0.5)
        if CLIENTS:
            print(f"  ✅ 已有 {len(CLIENTS)} 个客户端，开始对局", flush=True)
        round_no = 0
        while True:   # 打完一个半庄自动开新的，持续产数据、持续可观战
            await run_game(key, args.seed + round_no * 7919, args.delay)
            round_no += 1
            await broadcast({"type": "decision", "seat": -1, "turn": 0,
                             "phase": "end", "picked_text": f"—— 新半庄 #{round_no + 1} 开始 ——",
                             "think": "", "acts": ""})


if __name__ == "__main__":
    asyncio.run(main())

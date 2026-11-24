"""给牌谱每步弃牌加注释：进攻/防守意图 + 牌效(速度)/打点(大小)/安全度(防守)。"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mahjong.tiles import tile_name, NUM_TILES
from mahjong import refengine as REF


def tile_id_from_name(name):
    for t in range(NUM_TILES):
        if tile_name(t) == name:
            return t
    return -1


def hand34_from_names(names):
    h = np.zeros(34, dtype=np.int8)
    for n in names:
        tid = tile_id_from_name(n)
        if tid >= 0:
            h[tid] += 1
    return h


def shanten_of(hand34):
    return REF.ref_shanten(list(hand34))


def uke_count(hand34):
    cur = shanten_of(hand34)
    if cur < 0:
        return 0
    n = 0
    for t in range(34):
        if hand34[t] >= 4:
            continue
        hand34[t] += 1
        try:
            if REF.ref_shanten(hand34) < cur:
                n += 1
        except Exception:
            pass
        hand34[t] -= 1
    return n


def wait_tiles(hand34):
    if shanten_of(hand34) != 0:
        return []
    waits = []
    for t in range(34):
        if hand34[t] >= 4:
            continue
        hand34[t] += 1
        try:
            if REF.ref_shanten(hand34) < 0:
                waits.append(t)
        except Exception:
            pass
        hand34[t] -= 1
    return waits


def hand_han_potential(hand34, melds_n, dora_inds):
    n_dora = sum(hand34[t] for t in dora_inds if 0 <= t < 34)
    n_yakuhai_pair = sum(1 for t in (31, 32, 33) if hand34[t] >= 2)
    han = n_dora * 1.0 + n_yakuhai_pair * 0.5
    if melds_n == 0:
        han += 0.5
    return han


def safety_of(tile_str, snap, p_idx):
    score = 5.0
    seen = 0
    for i, q in enumerate(snap["players"]):
        if i == p_idx:
            continue
        seen += q["river"].count(tile_str)
        for m in q["melds"]:
            if tile_str in m["tiles"]:
                seen += 3
    score += min(seen, 4) * 1.2
    riichi_threat = any(snap["players"][i]["riichi"] for i in range(4) if i != p_idx)
    if riichi_threat:
        score -= 2.0
    tid = tile_id_from_name(tile_str)
    if tid >= 27:
        score += 1.0
    return max(0.0, min(10.0, score))


def intent_of(shanten_after, opp_riichi, turn):
    if opp_riichi and shanten_after >= 2 and turn >= 8:
        return "防守", f"对手立直+自己{shanten_after}向听+第{turn}巡，弃和保命"
    if opp_riichi and shanten_after >= 3:
        return "防守", f"对手立直+自己{shanten_after}向听，跟牌防守"
    if shanten_after <= 1:
        return "进攻", f"自己{shanten_after}向听（{'听牌' if shanten_after==0 else '一向听'}），推进牌型"
    if shanten_after == 2:
        return "进攻", "二向听，持续推进"
    return "中立", f"{shanten_after}向听，整理牌型"


def annotate_game(log_path, out_path):
    data = json.load(open(log_path, encoding="utf-8"))
    annotated = []
    for i, step in enumerate(data["steps"]):
        snap = step["snap"]
        turn = snap["turn"]
        dora_inds = [tile_id_from_name(dn) for dn in snap["dora"]]
        for ev in step["events"]:
            if ev["kind"] != "DISCARD":
                continue
            seat = ev["seat"]
            tile_str = ev["tile"]
            tile_id = tile_id_from_name(tile_str)
            if tile_id < 0:
                continue
            hand_after_names = snap["players"][seat]["hand"]
            hand_before = hand34_from_names(hand_after_names)
            hand_before[tile_id] += 1
            hand_after = hand_before.copy()
            hand_after[tile_id] -= 1
            sh_after = shanten_of(hand_after)
            uke = uke_count(hand_after)
            waits = wait_tiles(hand_after)
            melds_n = len(snap["players"][seat]["melds"])
            han_pot = hand_han_potential(hand_after, melds_n, dora_inds)
            safety = safety_of(tile_str, snap, seat)
            opp_riichi = any(snap["players"][j]["riichi"] for j in range(4) if j != seat)
            intent, reason = intent_of(sh_after, opp_riichi, turn)
            annotated.append({
                "step": i + 1,
                "seat": seat,
                "tile": tile_str,
                "intent": intent,
                "reason": reason,
                "shanten_after": int(sh_after),
                "uke_count": uke,
                "waits": [tile_name(t) for t in waits],
                "han_potential": round(han_pot, 1),
                "safety": round(safety, 1),
            })
    data["annotations"] = annotated
    json.dump(data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"annotated {len(annotated)} discards -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="rules/sample_game_d2.json")
    ap.add_argument("--out", default="rules/sample_game_d2_annot.json")
    a = ap.parse_args()
    annotate_game(a.log, a.out)

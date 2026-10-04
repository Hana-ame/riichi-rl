"""读数探针：每 30 秒采一次训练日志 / checkpoint / 策略实际表现。

按 M2 决断 1 实现：不再 sleep 盯日志，改为轮询式采样。

产出 metrics_probe.jsonl（每行一次采样）与控制台摘要。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import numpy as np
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import refengine as REF
from mahjong.net import MahjongNet
from train_hanchan import encode, slot_of, hand_shanten, N_ACTIONS
from mahjong import touch as TOUCH

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_a2c.log")
CKPT_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "checkpoints", "a2c_r*_it*.pt")

ACTION_NAMES = {
    34: "OTHER", 35: "OTHER", 36: "OTHER", 37: "OTHER", 38: "OTHER",
    39: "RYUUKYOKU", 40: "TSUMO", 41: "NAGASHI", 42: "PASS",
    43: "OTHER", 44: "OTHER", 45: "OTHER", 46: "OTHER", 47: "OTHER",
    48: "OTHER", 49: "OTHER", 50: "OTHER", 51: "OTHER", 52: "OTHER",
    53: "OTHER",
}


def parse_log():
    """从训练日志解析最新的 iter 行。"""
    if not os.path.exists(LOG):
        return None
    last = None
    with open(LOG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("iter "):
                last = line.strip()
    if not last:
        return None
    d = {"raw": last}
    for k, pat in (("iter", r"iter\s+(\d+)"),
                   ("deals", r"deals=\s*(\d+)"),
                   ("win_pct", r"win=\s*([\d.]+)%"),
                   ("R", r"R=([+-][\d.]+)"),
                   ("pol", r"pol=([+-][\d.]+)"),
                   ("vloss", r"vloss=([\d.]+)"),
                   ("ent", r"ent=([\d.]+)"),
                   ("giveup", r"giveup=(\d+)"),
                   ("defend", r"defend=(\d+)"),
                   ("last", r"last=(\d+)"),
                   ("secs", r"(\d+)s$")):
        m = re.search(pat, last)
        if m:
            d[k] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return d


def load_model(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ch = ck.get("channels", 64)
    nb = ck.get("blocks", 4)
    td = ck.get("touch_dim", TOUCH.TOUCH_DIM)
    need = ck["state_dict"]["head_fc.0.weight"].shape[1]
    probe = MahjongNet(channels=ch, n_blocks=nb, touch_dim=td)
    extra = need - probe.head_fc[0].in_features
    m = MahjongNet(channels=ch, n_blocks=nb, touch_dim=td,
                  extra_scalar=extra)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def probe_policy(model, n_deals, enc, seed0):
    """实际跑 n_deals 局，统计策略真实表现。"""
    import torch
    wins = 0
    deals = 0
    sh_sum = 0
    sh_n = 0
    disc_hist = [0] * 34
    action_slots = {}
    tenpai_hits = 0

    for i in range(n_deals):
        e = E.MahjongEnv(seed=seed0 + i, dealer=i % 4)
        r = e.start_turn()
        g = 0
        while r is None and g < 400:
            g += 1
            recs = {}
            idxs = []
            acts = {}
            for p in e.pending_players():
                legal = e.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                slots = [slot_of(a) for a in legal]
                recs[p] = (slots, legal)
                idxs.append(p)
            if idxs:
                vecs, tfs = [], []
                for p in idxs:
                    c = e.players[p].full_counts()
                    sh = hand_shanten(e, p)
                    sh_sum += max(0, sh)
                    sh_n += 1
                    planes, scalar = enc.encode(e, p)
                    vecs.append(planes.ravel())
                    tfs.append(TOUCH.shape_features(
                        c, len(e.players[p].melds), sh, 0))
                x = torch.from_numpy(
                    np.stack([np.concatenate(v) for v in vecs]).astype("float32"))
                tx = torch.from_numpy(np.stack(tfs))
                with torch.no_grad():
                    lg, _ = model(x, tx)
                for k, p in enumerate(idxs):
                    slots, legal = recs[p]
                    msk = torch.full((N_ACTIONS,), -1e9)
                    msk[slots] = 0.0
                    pick = slots[int((lg[k] + msk).argmax())]
                    a = legal[slots.index(pick)]
                    acts[p] = a
                    if a.type == E.ACT_DISCARD and 0 <= a.tile < 34:
                        disc_hist[a.tile] += 1
                    slot_name = (ACTION_NAMES.get(pick)
                                 if pick >= 34 else "DISCARD")
                    action_slots[slot_name] = action_slots.get(
                        slot_name, 0) + 1
                    if a.type in (E.ACT_TSUMO, E.ACT_RON):
                        tenpai_hits += 1
            r = e.step(acts)
        if r is None:
            continue
        deals += 1
        if r.result_type != "ryuukyoku":
            wins += 1

    total_disc = sum(disc_hist)
    top = sorted(range(34), key=lambda t: -disc_hist[t])[:5]
    return {
        "deals": deals,
        "wins": wins,
        "win_pct": 100.0 * wins / max(1, deals),
        "avg_shanten": sh_sum / max(1, sh_n),
        "top_discards": [(t, disc_hist[t]) for t in top],
        "action_mix": action_slots,
        "tenpai_hits": tenpai_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--probe-deals", type=int, default=6)
    ap.add_argument("--out", default="metrics_probe.jsonl")
    a = ap.parse_args()

    enc = O2.ObsEncoder()
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    t_start = time.time()
    seen_ckpt = ""

    for rnd in range(a.rounds):
        row = {"t": round(time.time() - t_start, 1)}
        lg = parse_log()
        if lg:
            row["train"] = lg
        cks = sorted(glob.glob(CKPT_GLOB), key=os.path.getmtime)
        row["n_ckpt"] = len(cks)
        if cks:
            newest = cks[-1]
            row["ckpt"] = os.path.basename(newest)
            row["ckpt_age_s"] = round(time.time() - os.path.getmtime(newest), 1)
            if newest != seen_ckpt:
                seen_ckpt = newest
                try:
                    model, ck = load_model(newest)
                    row["probe"] = probe_policy(model, a.probe_deals,
                                              enc, 90000 + rnd * 17)
                    row["ck_iter"] = ck.get("iter")
                    row["ck_deals"] = ck.get("deals")
                except Exception as e:
                    row["probe_error"] = repr(e)[:200]
        with open(outp, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        t = lg or {}
        pr = row.get("probe")
        print("[%6.0fs] iter=%s deals=%s win=%s%% vloss=%s ent=%s ckpt=%s | "
              "probe: win=%s avg_shanten=%s" % (
            row["t"], t.get("iter"), t.get("deals"), t.get("win_pct"),
            t.get("vloss"), t.get("ent"), row.get("ckpt", "-"),
            ("%.1f%%" % pr["win_pct"]) if pr else "-",
            ("%.2f" % pr["avg_shanten"]) if pr else "-"), flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()

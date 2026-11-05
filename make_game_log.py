"""导出一局完整牌谱（每步快照），供 HTML 牌谱游览器查看。

python make_game_log.py --ckpt checkpoints/teacher47_100h_v2.pt --seed 7101 --out rules/sample_game.json
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong.hanchan import HanchanState
from mahjong import env as E
from mahjong.tiles import tile_name
from version_benchmark import make_agent

ACT_NAMES = {0:"DISCARD",1:"TSUMO",2:"RIICHI",3:"ANKAN",4:"MINKAN",5:"CHI",6:"PON",7:"ADD_KAN",8:"RON",9:"RYUUKYOKU",11:"PASS"}

def meld_tiles(m):
    kind,t,_=m
    if kind=="chi": return [t,t+1,t+2]
    n=4 if kind in ("ankan","minkan","addkan") else 3
    return [t]*n

def snap(env,h):
    return {
        "wall": len(env.wall)-env.wall_idx,
        "turn": env.turn,
        "dora": [tile_name(t) for t in env.dora_indicators],
        "players": [{
            "seat": p.seat_wind, "dealer": p.is_dealer, "points": p.points,
            "hand": [tile_name(t) for t in sorted(p.hand)],
            "river": [tile_name(t) for t in p.river],
            "melds": [{"kind": m[0], "tiles": [tile_name(t) for t in meld_tiles(m)]} for m in p.melds],
            "riichi": p.riichi,
        } for p in env.players],
        "round": {"wind": tile_name(env.round_wind), "kyoku": env.kyoku, "honba": env.honba},
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/teacher47_100h_v2.pt")
    ap.add_argument("--seed", type=int, default=7101)
    ap.add_argument("--opponents", default="self",
                    help="self=4×同模型；否则逗号分隔 3 个规则名（如 call_first,greedy_tenpai,call_first）")
    ap.add_argument("--out", default="rules/sample_game.json")
    a=ap.parse_args()
    from mahjong import policies as POL
    main = make_agent("teacher47_100h_v2", a.ckpt)
    if a.opponents == "self":
        agents = [make_agent("teacher47_100h_v2", a.ckpt) for _ in range(4)]
    else:
        names = [x.strip() for x in a.opponents.split(",")][:3]
        agents = [main] + [POL.make_policy(n, seed=a.seed + i) for i, n in enumerate(names)]
    # 4 家：座位0=主模型，其余按规则（或同模型）
    h=HanchanState(dealer=a.seed % 4); env=h.new_env(seed=a.seed)
    for ag in agents:
        if hasattr(ag, "set_game"): ag.set_game(env,h)
    data={"meta":{"ckpt":os.path.basename(a.ckpt),"seed":a.seed},
          "initial":snap(env,h),"steps":[],"result":None}
    r=env.start_turn(); steps=0
    while r is None and steps<400:
        steps+=1
        pending=list(env.pending_players())
        acts={}
        for p in pending:
            legal=env.legal_actions(p)
            if not legal: acts[p]=E.PASS_ACTION; continue
            ag=agents[p]
            # 规则策略只有 choose(env,p,legal)；神经网络 Agent 才有 act(env,p,legal)
            act = ag.choose(env,p,legal) if hasattr(ag, "choose") else ag.act(env,p,legal)
            if act not in legal: act=E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
            acts[p]=act
        ev=[{"seat":p,"kind":ACT_NAMES.get(acts[p].type,str(acts[p].type)),
             "tile":tile_name(acts[p].tile) if acts[p].tile>=0 else None,
             "is_call":acts[p].type in (E.ACT_CHI,E.ACT_PON,E.ACT_MINKAN,E.ACT_ANKAN,E.ACT_ADD_KAN)} for p in pending]
        r=env.step(acts)
        data["steps"].append({"events":ev,"snap":snap(env,h)})
    if r is not None:
        data["result"]={"type":r.result_type,"winner":getattr(r,"winner",None),
                        "from":getattr(r,"from_player",None),
                        "dora_count":getattr(r,"dora_count",0),"aka":getattr(r,"aka_count",0),
                        "ura":getattr(r,"ura_dora_count",0),
                        "yaku":getattr(r,"yaku",None),"han":getattr(r,"han",None)}
    os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
    json.dump(data, open(a.out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"steps={len(data['steps'])} result={data['result']} -> {a.out}")

if __name__=="__main__":
    raise SystemExit(main())

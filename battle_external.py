"""外部模型（mahjonglm 系列）vs 自研：1 席外部 vs 3 席 teacher47_100h_v2。

python battle_external.py --model external/mahjonglm100 --name mahjonglm100 --deals 20
口径：外部模型只接管弃牌（合法 discard token 最高 logit），吃碰杠/立直/和牌=规则。
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong.hanchan import HanchanState
from mahjong import env as E
from external_mahjonglm import MahjongLMAgent
from version_benchmark import make_agent
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="external/mahjonglm10")
    ap.add_argument("--name", default="mahjonglm")
    ap.add_argument("--ckpt", default="checkpoints/teacher47_100h_v2.pt")
    ap.add_argument("--deals", type=int, default=20)
    ap.add_argument("--seat0", type=int, default=0)
    a=ap.parse_args()
    torch.set_num_threads(2)
    ext=MahjongLMAgent(a.model, seat_name=a.name)
    ours=[make_agent("teacher47_100h_v2", a.ckpt) for _ in range(4)]
    ext_w=ext_d=our_w=our_d=0; ron=tsumo=0; t0=time.time()
    for g in range(a.deals):
        ext_seat=g%4
        h=HanchanState(dealer=g%4); env=h.new_env(seed=41000+g)
        ext.set_game(env,h)
        for ag in ours: ag.set_game(env,h)
        r=env.start_turn(); steps=0
        while r is None and steps<500:
            steps+=1; acts={}
            for p in env.pending_players():
                legal=env.legal_actions(p)
                if not legal: acts[p]=E.PASS_ACTION; continue
                act=ext.act(env,p,legal) if p==ext_seat else ours[p].act(env,p,legal)
                if act not in legal: act=E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
                acts[p]=act
            r=env.step(acts)
            ext.observe(env, acts)
        if r is None or r.result_type=="ryuukyoku":
            print(f"  deal {g+1}: ryuukyoku ({time.time()-t0:.0f}s)", flush=True); continue
        if r.result_type=="ron": ron+=1; loser=r.from_player
        else: tsumo+=1; loser=None
        if r.winner==ext_seat: ext_w+=1
        else: our_w+=1
        if r.result_type=="ron" and loser==ext_seat: ext_d+=1
        if r.result_type=="ron" and loser is not None and loser!=ext_seat: our_d+=1
        print(f"  deal {g+1}: {r.result_type} winner={r.winner} loser={loser} ext_seat={ext_seat} ({time.time()-t0:.0f}s)", flush=True)
    n=a.deals
    print(f"自研 vs {a.name}: {n} 局（{time.time()-t0:.0f}s，荣{ron}/自摸{tsumo}）")
    print(f"  {a.name} 座: 和牌 {ext_w}/{n}={100*ext_w/n:.1f}% 放铳 {ext_d}/{n}={100*ext_d/n:.1f}%")
    print(f"  自研3席: 和牌局 {our_w}/{n} 放铳局 {our_d}/{n}")
    print(f"  {a.name} 决策计数: {getattr(ext,'decisions',{})}")

if __name__=="__main__":
    raise SystemExit(main())

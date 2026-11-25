"""Tenhou Decision → obs2(786) BC npz 转换器。

思路：写一个 TenhouPseudoEnv 适配器，直接从 houou_data.Decision 构造出
ObsEncoder.encode(env, p_idx) 所需的全部 env 字段（players/river/melds/dora/...），
避免真实 MahjongEnv 回放（Tenhou 规则细节与本引擎不完全兼容，回放会崩）。

用法：
    python tenhou_to_obs2.py --shard-glob "data/houou/2024_*.tar" \
        --max-shards 6 --max-decisions 200000 \
        --out data/tenhou_obs2.npz
"""
import sys, glob, os, io, tarfile, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from houou_data import decode_match, ACT_DISCARD, ACT_RIICHI_DISCARD, ACT_CHI, \
    ACT_PON, ACT_KAKAN, ACT_ANKAN, ACT_RON, ACT_TSUMO, ACT_PASS
from mahjong import obs2 as O2
from mahjong.obs2 import ObsEncoder
from mahjong import event_stream as ES
from mahjong.tiles import NUM_TILES
from train_hanchan import slot_of, N_ACTIONS, SIT_DIM, SLOT_PASS, encode, TOUCH
from mahjong import env as E


class _FakeHState:
    """给 situation_features 用的假 HanchanState（只有它读的字段）。"""
    def __init__(self, kyoku, honba, scores):
        self.round = "E" if kyoku <= 4 else "S"
        self.kyoku = kyoku
        self.honba = honba
        self.sticks = scores  # assess() 用它作为各玩家点数


class _Player:
    def __init__(self, points=25000, is_dealer=False, seat_wind=27):
        self.hand = []
        self.river = []
        self.melds = []
        self.riichi = False
        self.points = points
        self.is_dealer = is_dealer
        self.seat_wind = seat_wind
        self.furiten = False
        self.genbutsu = set()

    def counts(self):
        """返回 (34,) 手牌计数，供 obs2.encode 使用。"""
        c = np.zeros(NUM_TILES, dtype=np.int8)
        for t in self.hand:
            if 0 <= t < NUM_TILES:
                c[t] += 1
        return c

    def full_counts(self):
        """返回 (34,) 手牌+副露合并计数，供 hand_shanten 使用。"""
        c = self.counts()
        for meld in self.melds:
            for t in meld:
                if 0 <= t < NUM_TILES:
                    c[t] += 1
        return c


class TenhouPseudoEnv:
    """从 houou_data.Decision 构造 ObsEncoder.encode 所需字段。"""

    def __init__(self, d):
        # 4 个 Player
        dealer = d.player  # houou 数据里 dealer 未知，用 player 当近似
        self.players = []
        for i in range(4):
            p = _Player(int(d.scores[i]), is_dealer=(i == dealer), seat_wind=27 + i)
            self.players.append(p)
        me = d.player
        # 自己手牌
        self.players[me].hand = [t for t in range(34) for _ in range(int(d.hand_counts[t]))]
        # 自己弃牌（river）
        self.players[me].river = [t for t in range(34) for _ in range(int(d.discards[t]))]
        # 自己副露（melds_by 是 [[t1,t2,t3],...] 列表的列表，转成 (kind, tile, meta) 格式）
        self.players[me].melds = []
        for m in d.melds_by[me]:
            if len(m) == 3:
                kind = "chi"
            elif len(m) == 4:
                kind = "pon"
            else:
                kind = "kan"
            self.players[me].melds.append((kind, m[0], (-1,)))
        # 自己立直
        self.players[me].riichi = bool(d.riichi_flags[me])

        # 其他三家（相对座位 1/2/3）
        for offset in range(1, 4):
            other = (me + offset) % 4
            q = self.players[other]
            for t in range(34):
                cnt = int(d.discards_by[other, t])
                q.river.extend([t] * cnt)
            q.melds = []
            for m in d.melds_by[other]:
                if len(m) == 3:
                    kind = "chi"
                elif len(m) == 4:
                    kind = "pon"
                else:
                    kind = "kan"
                q.melds.append((kind, m[0], (-1,)))
            q.riichi = bool(d.riichi_flags[other])
            q.genbutsu = set(q.river)

        # dora
        self.dora_indicator = int(d.dora_indicators[0]) - 1 if len(d.dora_indicators) > 0 and d.dora_indicators[0] > 0 else None
        self.dora_indicators = [int(x) - 1 for x in d.dora_indicators[:5] if x > 0]
        self.ura_dora_indicator = None

        # 局面
        self.kyoku = d.kyoku
        self.honba = d.honba
        self.turn = d.turn
        self.round_wind = d.round_wind
        wall_left = int(d.tiles_left)
        self.dead_wall_size = 14
        self.wall_idx = max(0, 136 - self.dead_wall_size - wall_left)
        self.wall = [0] * 136  # 占位，encode 只用 wall_idx

        # 还原真实公开事件流（含手切/摸切/立直/副露）并据此精确计算各立直家现物
        self.log = list(getattr(d, "history_events", None) or [])
        riichi_seen = [False, False, False, False]
        for _turn, _actor, _kind, _tile in self.log:
            if 0 <= _actor < 4 and 0 <= _tile < NUM_TILES:
                self.players[_actor].genbutsu.add(_tile)
                if _kind == "riichi":
                    riichi_seen[_actor] = True
                for _q_idx in range(4):
                    if riichi_seen[_q_idx]:
                        self.players[_q_idx].genbutsu.add(_tile)

        # phase
        self.phase = E.Phase.RESPONSE if d.trigger_type == 1 else E.Phase.DISCARD
        self.last_discard = d.trigger_tile if d.trigger_type == 1 else None
        self.last_discard_player = (d.trigger_actor if d.trigger_type == 1
                                    else d.player)


def decision_to_obs2(d, encoder):
    """Decision → (obs786, tf8, label_slot)。返回 (None, None, -1) 表示不可用。"""
    env = TenhouPseudoEnv(d)
    hstate = _FakeHState(d.kyoku, d.honba, list(d.scores))
    act_type, tile_base, consumed = d.expert
    act_map = {
        ACT_DISCARD: E.ACT_DISCARD, ACT_RIICHI_DISCARD: E.ACT_RIICHI,
        ACT_CHI: E.ACT_CHI, ACT_PON: E.ACT_PON, ACT_KAKAN: E.ACT_ADD_KAN,
        ACT_ANKAN: E.ACT_ANKAN, ACT_RON: E.ACT_RON, ACT_TSUMO: E.ACT_TSUMO,
        ACT_PASS: E.ACT_PASS,
    }
    if act_type not in act_map:
        return None, None, -1
    e_type = act_map[act_type]
    e_tile = tile_base if tile_base is not None and tile_base >= 0 else -1
    e_arg = consumed[0] if len(consumed) > 0 else -1
    if e_type == E.ACT_CHI and e_tile >= 0 and len(consumed) >= 2:
        # 统一与 MahjongEnv 约定：act.tile = 顺子起始牌 start，act.arg = 被吃弃牌
        seq_tiles = [int(e_tile), int(consumed[0]), int(consumed[1])]
        act = E.Action(e_type, min(seq_tiles), int(e_tile))
    else:
        act = E.Action(e_type, e_tile, e_arg)
    slot = slot_of(act)
    if slot < 0 or slot >= N_ACTIONS:
        return None, None, None, None, -1, 0.0
    try:
        vec, sh, sit, give_up, tf = encode(env, encoder, hstate, d.player)
        ev = ES.encode_event_stream(env, d.player)
    except Exception:
        return None, None, None, None, -1, 0.0
    # 从 houou_data.Decision.legal 构造 100% 真实的 47 槽合法动作掩码
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for l_type, l_tile, l_cons in getattr(d, "legal", ()):
        if l_type not in act_map:
            continue
        le_type = act_map[l_type]
        le_tile = l_tile if l_tile is not None and l_tile >= 0 else -1
        if le_type == E.ACT_CHI and le_tile >= 0 and len(l_cons) >= 2:
            seq_t = [int(le_tile), int(l_cons[0]), int(l_cons[1])]
            la = E.Action(le_type, min(seq_t), int(le_tile))
        else:
            le_arg = l_cons[0] if len(l_cons) > 0 else -1
            la = E.Action(le_type, le_tile, le_arg)
        ls = slot_of(la)
        if 0 <= ls < N_ACTIONS:
            mask[ls] = True
    mask[slot] = True
    # 离线 Value 监督目标：局内净得分 + 终局顺位 Uma 混合归一化
    rank_uma = {0: 1.0, 1: 0.33, 2: -0.33, 3: -1.0}.get(int(getattr(d, "final_rank", 1)), 0.0)
    val_target = float(np.clip(getattr(d, "score_delta", 0) / 10000.0, -3.0, 3.0) + 0.5 * rank_uma)
    return vec, tf, ev, mask, slot, val_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="data/houou/2024_*.tar")
    ap.add_argument("--max-shards", type=int, default=6)
    ap.add_argument("--max-decisions", type=int, default=200000)
    ap.add_argument("--out", default="data/tenhou_obs2.npz")
    args = ap.parse_args()

    torch_threads = 4
    import torch
    torch.set_num_threads(torch_threads)

    paths = sorted(glob.glob(args.shard_glob))[:args.max_shards]
    print(f"使用 {len(paths)} shard: {[os.path.basename(p) for p in paths]}", flush=True)
    encoder = ObsEncoder()

    cap = args.max_decisions
    obs_dim = O2.OBS_TENSOR_DIM + SIT_DIM
    X = np.empty((cap, obs_dim), dtype=np.float32)
    TF = np.empty((cap, 8), dtype=np.float32)  # touch features
    EV = np.empty((cap, ES.DEFAULT_EVENT_LEN, 4), dtype=np.int64)
    M = np.empty((cap, N_ACTIONS), dtype=bool)
    Y = np.empty(cap, dtype=np.int64)
    V = np.empty(cap, dtype=np.float32)
    total = 0

    for path in paths:
        if total >= cap:
            break
        print(f"读取 {os.path.basename(path)}...", flush=True)
        try:
            with tarfile.open(path, 'r') as tf:
                for m in tf.getmembers():
                    if total >= cap:
                        break
                    if not m.name.endswith('.npz'):
                        continue
                    try:
                        data = np.load(io.BytesIO(tf.extractfile(m).read()))
                        decisions = decode_match(data)
                    except Exception:
                        continue
                    for d in decisions:
                        if total >= cap:
                            break
                        vec, tf_feat, ev_feat, m_row, slot, val_t = decision_to_obs2(d, encoder)
                        if vec is None:
                            continue
                        # 跳过仅有单一合法动作（如唯一 PASS）的零信息样本
                        if int(m_row.sum()) <= 1:
                            continue
                        X[total] = vec
                        TF[total] = tf_feat
                        EV[total] = ev_feat
                        M[total] = m_row
                        Y[total] = slot
                        V[total] = val_t
                        total += 1
            print(f"  累计 {total}", flush=True)
        except Exception as e:
            print(f"  跳过 {type(e).__name__}:{e}", flush=True)

    X, TF, EV, M, Y, V = X[:total], TF[:total], EV[:total], M[:total], Y[:total], V[:total]
    print(f"最终样本 {len(X)} obs={X.shape[1]} tf={TF.shape[1]} 标签分布 {np.bincount(Y, minlength=N_ACTIONS)[:10]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, x=X, tf=TF, events=EV, mask=M,
                        y=Y, value=V, aux=np.zeros(len(X), dtype=np.float32))
    print(f"保存 {args.out}")


if __name__ == "__main__":
    main()

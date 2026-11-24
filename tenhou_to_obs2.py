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
from mahjong.obs2 import ObsEncoder
from mahjong.tiles import NUM_TILES
from train_hanchan import slot_of, N_ACTIONS, encode, TOUCH
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
        for kind, t, consumed in self.melds:
            if kind == "chi":
                for dt in range(3):
                    if t + dt < NUM_TILES:
                        c[t + dt] += 1
            elif kind == "pon":
                if t < NUM_TILES:
                    c[t] += 3
            elif kind in ("ankan", "minkan", "addkan"):
                if t < NUM_TILES:
                    c[t] += 4
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
        # 自己副露
        self.players[me].melds = [(("chi" if len(m) == 3 else "pon"), m[0], (-1,))
                                  for m in d.melds_by[me] if len(m) >= 2]
        # 自己立直
        self.players[me].riichi = bool(d.riichi_flags[me])

        # 其他三家（相对座位 1/2/3）
        for offset in range(1, 4):
            other = (me + offset) % 4
            q = self.players[other]
            for t in range(34):
                cnt = int(d.discards_by[other, t])
                q.river.extend([t] * cnt)
            q.melds = [(("chi" if len(m) == 3 else "pon"), m[0], (-1,))
                       for m in d.melds_by[other] if len(m) >= 2]
            q.riichi = bool(d.riichi_flags[other])

        # dora
        self.dora_indicator = int(d.dora_indicators[0]) - 1 if len(d.dora_indicators) > 0 and d.dora_indicators[0] > 0 else None
        self.dora_indicators = [int(x) - 1 for x in d.dora_indicators[:5] if x > 0]
        self.ura_dora_indicator = None

        # 局面
        self.kyoku = d.kyoku
        self.honba = d.honba
        self.turn = d.turn
        self.round_wind = d.round_wind
        wall_left = d.tiles_left
        self.wall_idx = 136 - wall_left
        self.wall = [0] * 136  # 占位，encode 只用 wall_idx

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
    act = E.Action(e_type, e_tile, e_arg)
    slot = slot_of(act)
    if slot < 0 or slot >= N_ACTIONS:
        return None, None, -1
    try:
        vec, sh, sit, give_up, tf = encode(env, encoder, hstate, d.player)
    except Exception:
        return None, None, -1
    return vec, tf, slot


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
    obs_dim = 786  # encode() 输出：planes(748) + scalar(26) + sit_vec(12) = 786
    X = np.empty((cap, obs_dim), dtype=np.float32)
    TF = np.empty((cap, 8), dtype=np.float32)  # touch features
    Y = np.empty(cap, dtype=np.int64)
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
                        vec, tf_feat, slot = decision_to_obs2(d, encoder)
                        if vec is None:
                            continue
                        X[total] = vec
                        TF[total] = tf_feat
                        Y[total] = slot
                        total += 1
            print(f"  累计 {total}", flush=True)
        except Exception as e:
            print(f"  跳过 {type(e).__name__}:{e}", flush=True)

    X, TF, Y = X[:total], TF[:total], Y[:total]
    print(f"最终样本 {len(X)} obs={X.shape[1]} tf={TF.shape[1]} 标签分布 {np.bincount(Y, minlength=N_ACTIONS)[:10]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, x=X, tf=TF, mask=np.ones((len(X), N_ACTIONS), dtype=bool),
                        y=Y, aux=np.zeros(len(X), dtype=np.float32))
    print(f"保存 {args.out}")


if __name__ == "__main__":
    main()

"""天鳳鳳凰室牌谱（WebDataset .npz）解码器 —— 行为克隆数据源。

数据来源（2026-10-05 实测）：
  HuggingFace `1nfrared218/tenhou-mjai-riichi-training`（CC-BY-4.0，ungated）
  上游 `NikkeTryHard/tenhou-to-mjai`（鳳凰卓 2009-2026，2.5M+ 局）
  一个 shard = 1024 局 / ~67 万决策点 / 23MB。

**牌编码（本文件唯一权威，实测 8270/8270 样本零违规定出）**：

    1..34   基础牌，base = id - 1
    34      红 5m  -> base 4
    35      红 5p  -> base 13
    36      红 5s  -> base 22

即「1-based 的 37 牌空间」，不是 0-based 也不是 136 牌空间。
注意 `hand_counts[34]` **不含**红五，红五单独存在 `hand_red_counts`。

**动作 type 编码（由 manifest 的 expert_action_counts 与 trigger 交叉验证定出）**：

    0 DISCARD(弃牌)     1 RIICHI_DISCARD(立直打牌)
    2 CHI   3 PON   4 KAKAN/DAIMINKAN   5 TSUMO(自摸)
    6 RON   7 ANKAN   8 ?(少出现的副露类)
    9 PASS  10 KYUSHU_KYUHAI(九种九牌)

    trigger_type: 0=自己摸牌后决策  1=他家打牌后的响应  4=其他(立直宣告等)

本文件只做解码，不含任何麻将策略知识（不含牌效、不含人类打牌偏好）。
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass, field

import numpy as np

#: 37 牌空间的 id -> 0-based 基础牌号
#:
#: **实测定出（2026-10-05，120 局未见样本零违例）**：
#:   id 1..34  -> base id-1
#:   id 35/36/37 -> 红 5m / 5p / 5s，base 4 / 13 / 22
#:
#: 注意红五的 id 是 **35/36/37 而不是 34/35/36**——早先按 34/35/36 猜，
#: 每次弃牌合法动作都有 1909+ 次「手里没有这张牌」的违例。
#: 用「弃牌候选必须在 hand_counts+hand_red_counts 里存在」这个硬不变量
#: 做匈牙利匹配，唯一零违例解就是上面这个。
RED_ID_TO_BASE = {35: 4, 36: 13, 37: 22}


def tile_base(tile_id: int) -> int:
    """37 牌空间 id -> 0-based 基础牌号（0..33）。"""
    if tile_id in RED_ID_TO_BASE:
        return RED_ID_TO_BASE[tile_id]
    return tile_id - 1


# 动作类型常量（见模块 docstring）
ACT_DISCARD = 0
ACT_RIICHI_DISCARD = 1
ACT_CHI = 2
ACT_PON = 3
ACT_KAKAN = 4
ACT_TSUMO = 5
ACT_RON = 6
ACT_ANKAN = 7
ACT_ACT7 = 7
ACT_ACT8 = 8
ACT_PASS = 9
ACT_KYUSHU_KYUHAI = 10

#: 需要带牌的动作（其余为无参动作）
TILED_ACTS = frozenset({ACT_DISCARD, ACT_RIICHI_DISCARD, ACT_CHI, ACT_PON,
                        ACT_KAKAN, ACT_ANKAN, ACT_RON})

TRIGGER_SELF = 0
TRIGGER_CALL = 1
TRIGGER_OTHER = 4


@dataclass
class Decision:
    """一个决策点（自观测，不含对手暗牌）。"""

    player: int
    trigger_type: int
    trigger_actor: int
    trigger_tile: int              # 0-based base
    hand_counts: np.ndarray        # (34,) uint8，本家手牌计数
    melds: list                    # 本家副露
    discards: np.ndarray           # (34,) uint8，本人已打出
    dora_indicators: np.ndarray    # (5,) 公开的宝牌指示牌号（0 = 空）
    discards_by: np.ndarray        # (4,34) 他家已打出（公开信息）
    riichi_flags: np.ndarray       # (4,) uint8 公开立直宣告
    melds_by: list                 # 四家副露
    meld_cnt: np.ndarray           # (4,) 各家副露组数
    scores: np.ndarray             # (4,) int32 公开点数
    seat_wind: int                 # 本家风位（牌号 27..30 = 东/南/西/北） uint8
    round_wind: int                # 场风（牌号 27..30）
    kyoku: int
    honba: int
    tiles_left: int
    turn: int                      # 该局已摸牌次数
    legal: list                    # [(act_type, tile_base|None, consumed_tuple)]
    expert: tuple                  # 专家动作，同 legal 元素类型
    score_delta: int               # 该玩家本局最终点数变化（终局标签）
    final_rank: int
    steps_to_done: int
    history_events: list = field(default_factory=list)  # [(turn, actor, kind, tile_id), ...] 含真实手切/摸切


#: 各动作 type 的 `legal_tiles` 行里有意义的牌数字段个数。
#: 实测该数组用 0 填充，而**牌号 1 本身是合法值（1m）**，
#: 所以不能靠 `if x > 0` 判断有效牌——必须按 type 知道字段数。
#:
#: type 7 / 8 实测**字段数为 0**（与 PASS 同类，是无参动作）：
#: 它们出现在 trigger=0（自己摸牌后，如九种九牌/流局满贯）和
#: trigger=1（他家打牌后）的位置，故单列为无参槽位。
TILE_FIELDS = {
    ACT_DISCARD: 1,
    ACT_RIICHI_DISCARD: 1,
    ACT_CHI: 3,
    ACT_PON: 3,
    ACT_KAKAN: 1,
    ACT_ANKAN: 1,
    ACT_RON: 1,
    ACT_TSUMO: 0,
    ACT_PASS: 0,
    ACT_ACT7: 0,
    ACT_ACT8: 0,
    ACT_KYUSHU_KYUHAI: 0,
}


def _legal_from_flat(lo, lt, lti, i):
    a, b = int(lo[i]), int(lo[i + 1])
    out = []
    for j in range(a, b):
        t = int(lt[j])
        k = TILE_FIELDS.get(t, 0)
        if k:
            row = [int(x) for x in lti[j][:k]]
            tile = tile_base(row[0])
            consumed = tuple(sorted(tile_base(x) for x in row[1:])) if k > 1 else ()
            out.append((t, tile, consumed))
        else:
            out.append((t, None, ()))
    return out


def decode_match(data) -> list:
    """一个 .npz -> Decision 列表。"""
    tt = data["trigger_type"]
    lo = data["legal_offsets"]
    lt = data["legal_type"]
    lti = data["legal_tiles"]
    ei = data["expert_index"]
    hc = data["hand_counts"]
    hrc = data["hand_red_counts"]
    dw = data["dora"]
    dh = data["history_offsets"]
    dht = data["history_type"]
    dhact = data["history_actor"]
    dhti = data["history_tiles"]
    hs = data["history_tsumogiri"]
    dhref = data["history_ref"]
    dhlen = data["history_len"]
    scores = data["scores"]
    sw = data["seat_wind"]
    rw, ky, hb = data["round_wind"], data["kyoku"], data["honba"]
    tl = data["tiles_left"]
    sdelta = data["score_delta"]
    frank = data["final_rank"]
    std = data["steps_to_done"]
    player = data["player"]

    out = []
    for i in range(len(tt)):
        legal = _legal_from_flat(lo, lt, lti, i)
        k = int(ei[i])
        if k >= len(legal):
            continue
        # 本家手牌（含红五还原）
        hand = hc[i].astype(np.int16).copy()
        for red_id, base in RED_ID_TO_BASE.items():
            hand[base] += int(hrc[i][base])

        # 公开信息：回放本决策点之前已发生的事件
        # history_ref[i] 指向 history_offsets 的下标；history_len[i] 是本样本的事件数
        discards = np.zeros((4, 34), dtype=np.uint8)
        melds = [[] for _ in range(4)]
        riichi = np.zeros(4, dtype=np.uint8)
        ha = int(dh[int(dhref[i])])
        hb_ = ha + int(dhlen[i])
        turn = 0
        hist_events = []
        pending_riichi_actor = -1
        for k2 in range(ha, hb_):
            ty = int(dht[k2])
            ac = int(dhact[k2])
            tiles = [tile_base(int(x)) for x in dhti[k2] if x > 0]
            if ty == 0:            # tsumo
                turn += 1
            elif ty == 1:          # dahai（区分手切 discard 与摸切 tsumogiri）
                if tiles:
                    t0 = tiles[0]
                    discards[ac, t0] += 1
                    is_tsumogiri = bool(int(hs[k2])) if k2 < len(hs) else False
                    if pending_riichi_actor == ac:
                        ev_kind = "riichi"
                        pending_riichi_actor = -1
                    else:
                        ev_kind = "tsumogiri" if is_tsumogiri else "discard"
                    hist_events.append((turn, ac, ev_kind, t0))
            elif ty == 2:          # riichi 宣告（下一张弃牌标为 riichi 宣言牌）
                riichi[ac] = 1
                pending_riichi_actor = ac
            elif ty in (3, 4, 5, 6, 7, 8):
                melds[ac].append(tuple(tiles))
                if tiles:
                    m_kind = "chi" if ty == 3 else ("pon" if ty == 4 else "kan")
                    hist_events.append((turn, ac, m_kind, tiles[0]))

        out.append(Decision(
            player=int(player[i]), trigger_type=int(tt[i]),
            trigger_actor=int(data["trigger_actor"][i]),
            trigger_tile=tile_base(int(data["trigger_tile"][i])) if int(data["trigger_tile"][i]) > 0 else -1,
            hand_counts=hand.astype(np.uint8), melds=melds[int(player[i])],
            discards=discards[int(player[i])], dora_indicators=dw[i],
            discards_by=discards, riichi_flags=riichi, melds_by=melds,
            meld_cnt=np.array([len(m) for m in melds], dtype=np.int32),
            scores=scores[i], seat_wind=sw[i], round_wind=int(rw[i]),
            kyoku=int(ky[i]), honba=int(hb[i]), tiles_left=int(tl[i]),
            turn=turn, legal=legal, expert=legal[k],
            score_delta=int(sdelta[i]), final_rank=int(frank[0]),
            steps_to_done=int(std[i]),
            history_events=hist_events,
        ))
    return out


def read_shard(path):
    """tar -> [(match_id, [Decision, ...]), ...]"""
    res = []
    with tarfile.open(path) as tf:
        for nm in tf.getnames():
            if not nm.endswith(".npz"):
                continue
            try:
                data = np.load(io.BytesIO(tf.extractfile(nm).read()),
                                allow_pickle=True)
                res.append((nm[:-4], decode_match(data)))
            except Exception:
                continue
    return res


if __name__ == "__main__":
    import glob
    import sys
    import collections

    fs = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                          else "/tmp/houou/*.tar"))
    ms = read_shard(fs[0])
    print(f"shard={fs[0]} matches={len(ms)} decisions={sum(len(d) for _, d in ms)}")
    acts = collections.Counter()
    for _, ds in ms:
        for d in ds:
            acts[d.expert[0]] += 1
    print("expert action histogram:", dict(sorted(acts.items())))
    print("legal/decision avg: %.2f" % (
        sum(len(d.legal) for _, ds in ms for d in ds) / max(1, sum(len(ds) for _, ds in ms))))
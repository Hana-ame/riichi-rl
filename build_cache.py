"""把天鳳鳳凰室 .npz 编译成紧凑的训练张量（.npz cache）。

为什么要这一步（2026-10-05 实测）：
  原始 shard 每局 24k 决策点要在 Python 里回放历史才能得到公开信息，
  实测 368s/shard（1024 局），16 shard 要 1.6 小时，纯 Python 回放是瓶颈。
  这里把每局的公开状态**增量累加**（决策点沿时间递增），
  每个事件只处理一次，复杂度从 O(样本数 x 事件数) 降到 O(事件数)。

输出（每局一组）：
  obs     (N, C, 34) float16   观测平面
  scalar  (N, S)     float16   标量
  mask    (N, A)     bool      合法动作
  target  (N,)       int16     专家动作的槽位（-1 = 无效）
  value   (N,)       float32   终局点数变化（可选的价值标签）
"""

from __future__ import annotations

import io
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import houou_data as H  # noqa: E402

# ---------------------------------------------------------------------------
# 动作槽位：type -> 槽位基址。每个 type 一段，段内按牌号展开。
# 弃牌/立直弃 34 槽（0..33），副露类 34 槽，其余单槽。
# ---------------------------------------------------------------------------
#: 带牌动作的槽位基址（段长 34）；无参动作单占一个槽。
#:
#: 注意 type 7 有两个来源共用（ANKAN 与实测的无参 type 7），
#: 因此**不能**把 SLOT_BASE 写成「type -> 基址」的普通 dict——
#: 同键会互相覆盖。真正的判据是 H.TILE_FIELDS（该 type 是否有牌字段）。
TILED_SLOT_BASE = {
    H.ACT_DISCARD: 0,          # 0..33
    H.ACT_RIICHI_DISCARD: 34,  # 34..67
    H.ACT_CHI: 68,             # 68..101
    H.ACT_PON: 102,            # 102..135
    H.ACT_KAKAN: 136,          # 136..169
    H.ACT_ANKAN: 170,          # 170..203
    H.ACT_RON: 204,            # 204..237
}
PARAM_SLOT = {
    H.ACT_TSUMO: 238,
    H.ACT_PASS: 239,
    H.ACT_ACT8: 240,
    H.ACT_KYUSHU_KYUHAI: 241,
    # type 7 实测字段数为 0，但它与 ANKAN 同 type。
    # 牌副露/暗杠在 trigger=1 带 1 个牌字段，已被 TILED_SLOT_BASE 覆盖；
    # 无参的 type 7 落到这里。
    7: 170 + 34,               # 204 与 RON 冲突，改用尾段
}
PARAM_SLOT[7] = 242
N_ACTIONS = 243


def slot_of(act) -> int:
    """(type, tile, consumed) -> 槽位号；不能唯一编码时返回 -1。

    判据用 H.TILE_FIELDS（该 type 实际有几个牌字段），而不是槽位表本身——
    否则 type 7 这种「同 type 但有时带牌有时不带」的情况会二义。
    """
    t, tile, _ = act
    nfield = H.TILE_FIELDS.get(t, 0)
    if nfield == 0:
        return PARAM_SLOT.get(t, -1)
    base = TILED_SLOT_BASE.get(t)
    if base is None or tile is None or tile < 0 or tile > 33:
        return -1
    return base + tile


N_PLANES = 16
SCALAR_DIM = 12


def encode(dec: H.Decision, state: dict):
    """Decision -> (planes(16,34), scalar(12,))"""
    pl = np.zeros((N_PLANES, 34), dtype=np.float32)
    sc = np.zeros(SCALAR_DIM, dtype=np.float32)

    # 0: 自己手牌计数
    pl[0] = dec.hand_counts
    # 1: 自己已打出的牌
    pl[1] = dec.discards
    # 2-5: 他家已打出的牌（公开）
    for i in range(4):
        if i != dec.player:
            pl[2 + i] = dec.discards_by[i]
    # 6: 宝牌指示（公开）。实测 dora 形状是 (5,)，装的是**指示牌**的
    # 0-based 牌号（0 = 该槽空），不是 34 维计数——直接 pl[6]=dec.dora 会崩。
    for k, ind in enumerate(dec.dora_indicators[:5]):
        if ind > 0:
            pl[6, min(int(ind), 33)] = 1.0
    # 7: 各家点数（归一）
    pl[7, :4] = dec.scores / 40000.0
    # 8: 剩余牌数（全局）
    pl[8, :] = dec.tiles_left / 136.0
    # 9: 公开立直宣告（四家合并到 honours 段会串味，改为广播到 4 个固定列）
    for i in range(4):
        if dec.riichi_flags[i]:
            pl[9, min(i, 33)] = 1.0
    # 10: 自己风位/场次。实测 seat_wind 是**单个** per-sample 值，
    # 且用牌号编码（27..30 = 东南西北），不是 0..3，也不是 (4,) 向量。
    pl[10, 0] = (int(dec.seat_wind) - 27) / 3.0
    pl[10, 1] = float(dec.kyoku == 1)
    # 11: 触发类型 one-hot（低 4 列）
    pl[11, min(int(dec.trigger_type), 3)] = 1.0
    # 12: 本家点数偏差
    pl[12, 0] = (dec.scores[dec.player] - 25000) / 20000.0
    # 13: 巡目进度
    pl[13, 0] = dec.turn / 18.0
    # 14: 副露张数（四家）
    for i in range(4):
        pl[14, i] = dec.meld_cnt[i] / 4.0
    # 15: 荣和/和了可用的粗粒度信号（后续可拆细）

    sc[0] = dec.kyoku / 4.0
    sc[1] = dec.honba / 5.0
    sc[2] = (int(dec.round_wind) - 27) / 3.0
    sc[3] = len(dec.melds)
    sc[4] = float(dec.trigger_actor == dec.player)
    sc[5] = dec.trigger_tile / 33.0 if dec.trigger_tile >= 0 else -1.0
    sc[6] = (dec.dora_indicators[0] / 33.0) if dec.dora_indicators[0] > 0 else -1.0
    sc[7] = dec.tiles_left / 136.0
    sc[8] = dec.turn / 18.0
    sc[9] = (dec.scores[dec.player] - 25000) / 20000.0
    sc[10] = (int(dec.seat_wind) - 27) / 3.0
    sc[11] = dec.trigger_type / 4.0
    return pl, sc


def compile_match(data, max_dec: int = 4000):
    """一个 .npz -> (obs, scalar, mask, target, value)。增量累加公开状态。"""
    n = int(data["sample_count"][0])
    idx = np.arange(n)
    tt = data["trigger_type"]
    lo, lt, lti = data["legal_offsets"], data["legal_type"], data["legal_tiles"]
    ei, hc, hrc = data["expert_index"], data["hand_counts"], data["hand_red_counts"]
    dw = data["dora"]
    scores, sw = data["scores"], data["seat_wind"]
    rw, ky, hb, tl = data["round_wind"], data["kyoku"], data["honba"], data["tiles_left"]
    sdelta = data["score_delta"]
    player = data["player"]
    dht, dhact, dhti = data["history_type"], data["history_actor"], data["history_tiles"]
    hs = data["history_tsumogiri"]
    dhref, dhlen = data["history_ref"], data["history_len"]
    kidxs = data["kyoku_index"]
    dh_off = data["history_offsets"]

    obs = np.zeros((n, N_PLANES, 34), dtype=np.float16)
    scl = np.zeros((n, SCALAR_DIM), dtype=np.float16)
    mask = np.zeros((n, N_ACTIONS), dtype=bool)
    tgt = np.full(n, -1, dtype=np.int16)
    val = sdelta.astype(np.float32) / 8000.0

    # 增量状态
    discards = np.zeros((4, 34), dtype=np.uint8)
    meld_cnt = np.zeros(4, dtype=np.int32)
    riichi = np.zeros(4, dtype=np.uint8)
    turn = 0
    cur = 0                      # 已消费的事件下标
    prev_kyoku_idx = -1

    for i in range(n):
        # 进入新的一巡：重置增量状态。
        # 实测 history_type 里**没有** start_kyoku（历史按巡截断，
        # 只留 tsumo/dahai/riichi/meld），真正的巡边界信号是 kyoku_index。
        kyidx = int(kidxs[i])
        if kyidx != prev_kyoku_idx:
            discards[:] = 0
            meld_cnt[:] = 0
            riichi[:] = 0
            turn = 0
            prev_kyoku_idx = kyidx
        # 推进历史到本决策点
        start = int(dh_off[int(dhref[i])])
        want = start + int(dhlen[i])
        while cur < want and cur < len(dht):
            ty = int(dht[cur])
            if ty == 0:                 # tsumo（摸牌）
                turn += 1
            elif ty == 1:               # dahai（打牌）：第 0 字段是牌
                discards[int(dhact[cur]), H.tile_base(int(dhti[cur][0]))] += 1
            elif ty == 2:               # riichi（宣告）：牌在 1..3
                riichi[int(dhact[cur])] = 1
            elif ty in (3, 4, 5, 6, 7, 8):   # 副露类
                meld_cnt[int(dhact[cur])] += 1
            cur += 1

        p = int(player[i])
        d = H.Decision(
            player=p, trigger_type=int(tt[i]),
            trigger_actor=int(data["trigger_actor"][i]),
            trigger_tile=H.tile_base(int(data["trigger_tile"][i]))
            if int(data["trigger_tile"][i]) > 0 else -1,
            hand_counts=hc[i].astype(np.uint8), melds=[],
            discards=discards[p], dora_indicators=dw[i],
            discards_by=discards, riichi_flags=riichi,
            melds_by=[[] for _ in range(4)], meld_cnt=meld_cnt,
            scores=scores[i], seat_wind=sw[i],
            round_wind=int(rw[i]), kyoku=int(ky[i]), honba=int(hb[i]),
            tiles_left=int(tl[i]), turn=turn, legal=[],
            expert=(0, -1, ()), score_delta=0, final_rank=0, steps_to_done=0)

        pl, sc = encode(d, {})
        obs[i] = pl
        scl[i] = sc

        a, b = int(lo[i]), int(lo[i + 1])
        leg = H._legal_from_flat(lo, lt, lti, i)
        for act in leg:
            s = slot_of(act)
            if s >= 0:
                mask[i, s] = True
        tgt[i] = slot_of(leg[int(ei[i])]) if int(ei[i]) < len(leg) else -1

    return obs, scl, mask, tgt, val


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    t0 = time.time()
    n_dec = 0
    n_match = 0
    parts = []
    with tarfile.open(src) as tf:
        for nm in tf.getnames():
            if not nm.endswith(".npz"):
                continue
            try:
                data = np.load(io.BytesIO(tf.extractfile(nm).read()), allow_pickle=True)
                parts.append(compile_match(data))
                n_dec += int(data["sample_count"][0])
                n_match += 1
            except Exception as e:
                print("skip", nm, e, file=sys.stderr)
            if n_match % 100 == 0:
                print(f"  {n_match} matches {n_dec} dec {time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
    o = np.concatenate([p[0] for p in parts])
    s = np.concatenate([p[1] for p in parts])
    m = np.concatenate([p[2] for p in parts])
    t = np.concatenate([p[3] for p in parts])
    v = np.concatenate([p[4] for p in parts])
    np.savez_compressed(out, obs=o, scalar=s, mask=m, target=t, value=v)
    print(f"saved {out}  matches={n_match} decisions={n_dec} "
          f"obs={o.shape} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

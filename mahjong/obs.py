"""观测编码与合法动作掩码。

对应 STATE_REPRESENTATION.md §2、§3

铁律：**观测里不能出现真实对局中该玩家看不到的信息。**
具体禁止：对手手牌、牌山顺序（只给「已见的牌」计数）、未来的牌。
允许：自己的手牌、四家牌河、四家副露、宝牌指示牌、点数、巡目、
      风位、剩余牌数、各家剩余张数（公开信息）。
"""

from __future__ import annotations

from . import env as E
from . import ruleset as RS
from . import tiles as T
from .tiles import NUM_TILES

# --- 特征维度 ---
N_TILE_FEATURES = NUM_TILES   # 自己手牌计数 34
ACTION_SPACE_SIZE = NUM_TILES + 20   # 34 张牌 + 若干特殊动作


def encode_observation(env, p_idx):
    """把局面编码成定长 float 向量（长度见 OBS_DIM）。

    只包含该玩家合法可见的信息。
    """
    p = env.players[p_idx]
    obs = []

    # 1. 自己手牌计数 34
    hc = p.counts()
    obs.extend(float(x) for x in hc)

    # 2. 各家牌河计数（自己 34 + 三家 34 = 136）——牌河是公开的
    for q in env.players:
        c = [0.0] * NUM_TILES
        for t in q.river:
            c[t] += 1.0
        obs.extend(c)

    # 3. 各家副露计数 34
    for q in env.players:
        c = [0.0] * NUM_TILES
        for kind, t, _ in q.melds:
            n = 4 if kind in ("ankan", "minkan", "addkan") else 3
            if kind == "chi":
                c[t] += 1; c[t + 1] += 1; c[t + 2] += 1
            else:
                c[t] += n
        obs.extend(c)

    # 4. 副露组数（4）
    obs.extend(float(len(q.melds)) for q in env.players)

    # 5. 点数归一化（4）：points / 25000 * 0.1，限制到 [-1, 1]
    for q in env.players:
        v = (q.points - RS.STARTING_POINTS) / 25000.0
        obs.append(max(-1.0, min(1.0, v * 0.1)))

    # 6. 自己是否庄家（1）
    obs.append(1.0 if p.is_dealer else 0.0)

    # 7. 巡目归一化（1）
    obs.append(min(1.0, env.turn / float(RS.MAX_TURNS)))

    # 8. 场风 one-hot（4）：E/S/W/N
    for t in (27, 28, 29, 30):
        obs.append(1.0 if env.round_wind == t else 0.0)

    # 9. 自风 one-hot（4）
    for t in (27, 28, 29, 30):
        obs.append(1.0 if p.seat_wind == t else 0.0)

    # 10. 宝牌指示牌 one-hot（34）
    ind = env.dora_indicator if env.dora_indicator is not None else -1
    obs.extend(1.0 if i == ind else 0.0 for i in range(NUM_TILES))

    # 11. 里宝牌指示牌 one-hot（34）
    uind = env.ura_dora_indicator
    obs.extend(1.0 if (uind is not None and i == uind) else 0.0
               for i in range(NUM_TILES))

    # 12. 振听标志（1）
    obs.append(1.0 if p.furiten else 0.0)

    # 13. 立直标志（1）
    obs.append(1.0 if p.riichi else 0.0)

    # 14. 牌山剩余张数归一化（1）
    left = len(env.wall) - env.wall_idx
    obs.append(min(1.0, left / 70.0))

    # 15. 各家手牌张数（4）：自己 13/14，副露玩家更少
    for q in env.players:
        obs.append(min(1.0, len(q.hand) / 14.0))

    # 16. 是否有上一张弃牌（1）+ 上一张弃牌 one-hot（34）
    ld = env.last_discard
    obs.append(1.0 if ld is not None else 0.0)
    obs.extend(1.0 if (ld is not None and i == ld) else 0.0
               for i in range(NUM_TILES))

    # 17. 当前阶段 one-hot（2）：discard / response
    obs.append(1.0 if env.phase == E.Phase.DISCARD else 0.0)
    obs.append(1.0 if env.phase == E.Phase.RESPONSE else 0.0)

    return obs


#: 观测向量长度（供训练脚本引用）
OBS_DIM = (N_TILE_FEATURES            # 34 自己手牌
           + 4 * NUM_TILES              # 136 牌河
           + 4 * NUM_TILES              # 136 副露
           + 4                          # 副露组数
           + 4                          # 点数
           + 1                          # 是否庄家
           + 1                          # 巡目
           + 4 + 4                      # 场风/自风
           + NUM_TILES                  # 宝牌指示
           + NUM_TILES                  # 里宝指示
           + 1 + 1                      # 振听/立直
           + 1                          # 牌山剩余
           + 4                          # 各家手牌张数
           + 1 + NUM_TILES              # 上一张弃牌
           + 2)                         # 阶段
def encode_observation_for(env, p_idx):
    """带长度校验的版本（自测用）。"""
    obs = encode_observation(env, p_idx)
    assert len(obs) == OBS_DIM, f"obs dim {len(obs)} != {OBS_DIM}"
    return obs


def legal_action_mask(env, p_idx):
    """合法动作掩码。

    返回 (mask, actions)：
        mask: 长度 ACTION_SPACE_SIZE 的 0/1 列表
        actions: 该位置上对应的 Action 列表（与 mask 中 1 的位置一一对应）
    """
    mask = [0.0] * ACTION_SPACE_SIZE
    actions = env.legal_actions(p_idx)
    slot_map = _slot_map(env, p_idx, actions)
    for idx, slot in slot_map:
        mask[slot] = 1.0
    return mask, actions, slot_map


# --- 动作 -> 固定槽位 ---
# 槽位 0..33 对应「打出 tile t」；
# 槽位 34.. 对应特殊动作，按 (type, tile, arg) 排序后的位置映射。
# 为了保证固定长度且可微，这里用一个稳定的哈希式映射：
# 特殊动作的槽位 = 34 + (type * 34 + tile) % (ACTION_SPACE_SIZE - 34)
_SPECIAL_BASE = NUM_TILES
_SPECIAL_SIZE = ACTION_SPACE_SIZE - NUM_TILES


def _special_slot(action):
    h = (action.type * 1000003 + action.tile * 101 + action.arg) % _SPECIAL_SIZE
    return _SPECIAL_BASE + h


def _slot_map(env, p_idx, actions):
    """返回 [(action_index, slot), ...]。同槽位冲突时取先出现者。"""
    out = []
    used = set()
    for i, a in enumerate(actions):
        if a.type == E.ACT_DISCARD and 0 <= a.tile < NUM_TILES:
            slot = a.tile
        else:
            slot = _special_slot(a)
        if slot in used:
            continue
        used.add(slot)
        out.append((i, slot))
    return out


def action_from_slot(env, p_idx, slot):
    """把槽位还原成合法动作（用于人类/测试驱动）。"""
    mask, actions, slot_map = legal_action_mask(env, p_idx)
    for i, s in slot_map:
        if s == slot:
            return actions[i]
    return None


class _FakeEnv:
    """占位，仅用于 OBS_DIM 断言的 import 期不执行。"""
    pass

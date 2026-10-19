"""动作编码桥接：数据集体系 <-> 引擎体系。

**这是整个项目最容易静默出错的地方**，因此单独成文件并自带验证。

两套编码完全不同（2026-10-05 实测）：

| 语义   | 数据集（houou npz） | 引擎（mahjong/env.py） |
|--------|---------------------|------------------------|
| 打牌   | 0                   | 0 `ACT_DISCARD`       |
| 立直打 | 1                   | 2 `ACT_RIICHI`        |
| 吃     | 2                   | 5 `ACT_CHI`           |
| 碰     | 3                   | 6 `ACT_PON`           |
| 加杠   | 4                   | 7 `ACT_ADD_KAN`       |
| 自摸   | 5                   | 1 `ACT_TSUMO`         |
| 荣和   | 6                   | 8 `ACT_RON`           |
| 暗杠   | 7（另有无参 type 7） | 3 `ACT_ANKAN`         |
| type 8 | 8（无参）           | —                     |
| 跳过   | 9                   | 11 `ACT_PASS`         |
| 九九   | 10                  | —                     |

数据集用「吃/碰带 3 个牌字段」区分吃法，本项目按 (type, 主牌) 归并到槽位；
引擎的吃法还需要 `arg`（顺子起始牌），因此**吃不能只按主牌建槽**——
本项目的 TILED_SLOT_BASE 对吃只取第 0 字段，若 arg 不同会撞槽。
build_cache 侧已用 `consumed` 记录另外两张，故训练无歧义；
推理侧必须用同样的归并规则，见 legal_slot()。
"""

from __future__ import annotations

# 数据集 type -> 引擎 type
DS_TO_ENV = {
    0: 0,    # DISCARD
    1: 2,    # RIICHI_DISCARD -> ACT_RIICHI
    2: 5,    # CHI
    3: 6,    # PON
    4: 7,    # KAKAN -> ACT_ADD_KAN
    5: 1,    # TSUMO
    6: 8,    # RON
    7: 3,    # ANKAN
    9: 11,   # PASS
}

# 引擎 type -> 数据集 type（RIICHI 复用「立直打」槽）
ENV_TO_DS = {0: 0, 2: 1, 5: 2, 6: 3, 7: 4, 1: 5, 8: 6, 3: 7, 11: 9}


def env_action_to_slot(act, TILED_SLOT_BASE, PARAM_SLOT, TILE_FIELDS):
    """引擎 Action -> 训练槽位号；无法映射返回 -1。

    关键：**吃法按 (主牌, consumed) 归并**，与 build_cache 一致；
    只按主牌会让不同吃法撞同一个槽，训练时标签有歧义。
    """
    et = act.type
    ds = ENV_TO_DS.get(et)
    if ds is None:
        return -1
    nfield = TILE_FIELDS.get(ds, 0)
    if nfield == 0:
        return PARAM_SLOT.get(ds, -1)
    t = act.tile
    if t is None or t < 0 or t > 33:
        return -1
    base = TILED_SLOT_BASE.get(ds)
    if base is None:
        return -1
    return base + t
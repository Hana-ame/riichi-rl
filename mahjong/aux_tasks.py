"""额外任务（auxiliary tasks）：每个模型/参赛者每局做完就加奖励点数。

动机（用户 2026-11-05）：「每个模型都给额外任务：额外任务完成也有奖励」。
本模块把「额外任务」定义成可判定的成就，完成即给 bonus 点数；
arena 排名与训练 reward 都可用同一套判定（同一局里对所有模型公平）。

任务分两类：
- **局末可判**（无需轨迹）：和牌 / 自摸 / 立直 / 满贯以上和牌 / 未放铳 / 流局听牌；
- **需轨迹**（Tracker 逐步观察）：8 巡内进听。

默认奖励（可调）：
    win         +1000   本局和牌
    tsumo       +300    自摸和牌
    mangan      +500    和牌且 >=5 番
    riichi      +300    本局宣告过立直
    no_dealin   +200    本局未放铳（流局也算未放铳）
    tenpai_draw +200    流局时听牌
    fast_tenpai +300    8 巡内进入听牌（需 Tracker）
"""

from __future__ import annotations

from . import env as E


#: 任务定义：name -> (bonus, 描述)
TASKS = {
    "win": (1000, "本局和牌"),
    "tsumo": (300, "自摸和牌"),
    "mangan": (500, "和牌且 >=5 番"),
    "riichi": (300, "本局宣告立直"),
    "no_dealin": (200, "本局未放铳"),
    "tenpai_draw": (200, "流局时听牌"),
    "fast_tenpai": (300, "8 巡内进入听牌（需轨迹）"),
    # 役种风格任务：让「断幺人 / 平胡人」都收进同一个模型，用权重调参
    "tanyao": (300, "和牌且役含断幺九"),
    "pinfu": (300, "和牌且役含平和"),
    # dora 任务（P4）：和牌且宝牌/赤/里宝至少 1 番
    "dora": (300, "和牌且宝牌/赤/里宝≥1 番"),
}

#: 局末静态任务（不需要逐步轨迹）
STATIC_TASKS = ("win", "tsumo", "mangan", "riichi", "no_dealin",
                "tenpai_draw", "tanyao", "pinfu", "dora")


def _is_tenpai(env, p):
    try:
        return bool(env._is_tenpai(p))
    except Exception:
        return False


def static_tasks(env, result, p_idx):
    """局末可判的额外任务：返回 {task_name: bool}。"""
    done = {name: False for name in STATIC_TASKS}
    if result is None:
        return done
    pl = env.players[p_idx]
    is_win = (result.result_type in ("tsumo", "ron")
              and result.winner == p_idx)
    done["win"] = is_win
    done["tsumo"] = result.result_type == "tsumo" and result.winner == p_idx
    done["mangan"] = is_win and (result.han or 0) >= 5
    done["riichi"] = bool(pl.riichi)
    done["no_dealin"] = (result.result_type != "ron"
                         or result.from_player != p_idx)
    done["tenpai_draw"] = (result.result_type == "ryuukyoku"
                           and bool(result.tenpai[p_idx]))
    # 役种风格任务（同一模型内用 --aux-weights 调偏重）
    yaku = set(result.yaku or ())
    done["tanyao"] = is_win and "断幺九" in yaku
    done["pinfu"] = is_win and "平和" in yaku
    done["dora"] = is_win and (int(getattr(result, "dora_count", 0))
                               + int(getattr(result, "aka_count", 0))
                               + int(getattr(result, "ura_dora_count", 0))) > 0
    return done


def bonus_of(done, weights=None):
    """{task: bool} -> 奖励点数合计。

    weights: 可选的 {task_name: 倍率}——同一个模型用不同风格权重调参
    （如 {"tanyao": 2.0, "pinfu": 2.0}），默认全 1。
    """
    total = 0.0
    for name, ok in done.items():
        if not ok:
            continue
        w = 1.0 if not weights else float(weights.get(name, 1.0))
        total += TASKS[name][0] * w
    return total


class AuxTaskTracker:
    """每局给一个玩家发一份「额外任务清单」，逐步观察并结算奖励。

    用法：
        tracker = AuxTaskTracker(p)
        ... 每个 env.step 后：tracker.observe(env)
        done, bonus = tracker.finish(env, result)
    """

    def __init__(self, p_idx):
        self.p_idx = int(p_idx)
        self.fast_tenpai = False

    def observe(self, env):
        """每一步调用。8 巡（env.turn < 8）内进听即视为完成 fast_tenpai。"""
        if self.fast_tenpai:
            return
        if env.turn < 8 and _is_tenpai(env, self.p_idx):
            self.fast_tenpai = True

    def finish(self, env, result):
        done = static_tasks(env, result, self.p_idx)
        done["fast_tenpai"] = bool(self.fast_tenpai)
        return done, bonus_of(done)


def deal_bonus(env, result, p_idx, trackers=None, weights=None):
    """便捷接口：算一个玩家本局的额外任务奖励。

    trackers 可选：传入 {p_idx: AuxTaskTracker} 时复用其 fast_tenpai 记录。
    """
    if trackers is not None and p_idx in trackers:
        done = trackers[p_idx].finish(env, result)[0]
    else:
        done = static_tasks(env, result, p_idx)
    return done, bonus_of(done, weights)
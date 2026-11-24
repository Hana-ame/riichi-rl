"""对局引擎：4 人混战，多 agent 原生。

对应 RULES.md §3、§5、§9、§10、§13；竞技接口见 ARENA.md §2。
"""

from __future__ import annotations

import random

from . import ruleset as RS
from . import hand as H
from . import scoring as S
from . import tiles as T
from . import refengine as REF
from .tiles import NUM_TILES, NUM_TOTAL_TILES

N_PLAYERS = 4

ACT_DISCARD = 0
ACT_TSUMO = 1
ACT_RIICHI = 2
ACT_ANKAN = 3
ACT_MINKAN = 4
ACT_CHI = 5
ACT_PON = 6
ACT_ADD_KAN = 7
ACT_RON = 8
ACT_RYUUKYOKU = 9
ACT_NAGASHI = 10
ACT_PASS = 11

ACT_NAMES = {
    ACT_DISCARD: "DISCARD", ACT_TSUMO: "TSUMO", ACT_RIICHI: "RIICHI",
    ACT_ANKAN: "ANKAN", ACT_MINKAN: "MINKAN", ACT_CHI: "CHI",
    ACT_PON: "PON", ACT_ADD_KAN: "ADD_KAN", ACT_RON: "RON",
    ACT_RYUUKYOKU: "RYUUKYOKU", ACT_NAGASHI: "NAGASHI", ACT_PASS: "PASS",
}


class Action:
    """动作 (type, tile, arg)。arg 用于吃法的起始牌。"""

    __slots__ = ("type", "tile", "arg")

    def __init__(self, act_type, tile=-1, arg=-1):
        self.type = act_type
        self.tile = tile
        self.arg = arg

    def __eq__(self, other):
        return (isinstance(other, Action) and self.type == other.type
                and self.tile == other.tile and self.arg == other.arg)

    def __hash__(self):
        return hash((self.type, self.tile, self.arg))

    def __repr__(self):
        n = ACT_NAMES.get(self.type, str(self.type))
        if self.type == ACT_CHI:
            return f"{n}({T.tile_name(self.tile)},{self.arg})"
        if self.tile >= 0:
            return f"{n}({T.tile_name(self.tile)})"
        return n


PASS_ACTION = Action(ACT_PASS)


def _chi_starts(tile):
    """能吃出顺子的起始牌 id（tile 作为顺子第 1/2/3 张时的起手）。"""
    if T.is_honor(tile):
        return ()
    suit = tile // 9
    rank = tile % 9
    out = []
    if rank <= 6:
        out.append(suit * 9 + rank)
    if 1 <= rank <= 7:
        out.append(suit * 9 + rank - 1)
    if 2 <= rank <= 8:
        out.append(suit * 9 + rank - 2)
    return tuple(out)


class Player:
    __slots__ = ("idx", "hand", "melds", "river", "points", "is_dealer",
                 "seat_wind", "riichi", "double_riichi", "ippatsu",
                 "riichi_turn", "furiten", "just_kan", "last_drawn",
                 "genbutsu")

    def __init__(self, idx, points, seat_wind):
        self.idx = idx
        self.hand = []
        self.melds = []
        self.river = []
        self.points = points
        self.is_dealer = False
        self.seat_wind = seat_wind
        self.riichi = False
        self.double_riichi = False
        self.ippatsu = False
        self.riichi_turn = -1
        self.furiten = False
        self.just_kan = False
        self.last_drawn = -1   # 最近一次摸到的牌（自摸和了判定用）
        self.genbutsu = set()  # 现物集合（自家牌河 + 自家立直后全员弃牌）

    def is_menzen(self):
        return len(self.melds) == 0

    def counts(self):
        c = [0] * NUM_TILES
        for t in self.hand:
            c[t] += 1
        return c

    def full_counts(self):
        """暗牌 + 副露牌（不含待和的那张）。"""
        c = self.counts()
        for kind, t, _ in self.melds:
            if kind == "chi":
                c[t] += 1; c[t + 1] += 1; c[t + 2] += 1
            elif kind == "pon":
                c[t] += 3
            else:
                c[t] += 4
        return c

    def owned_tiles(self):
        out = list(self.hand)
        for kind, t, _ in self.melds:
            if kind == "chi":
                out.extend((t, t + 1, t + 2))
            elif kind == "pon":
                out.extend((t, t, t))
            else:
                out.extend((t, t, t, t))
        return out

    def hand_count_for_meld(self, tile):
        """暗牌里 tile 的张数。"""
        return sum(1 for t in self.hand if t == tile)

    def meld_count_for_tile(self, tile):
        """副露里 tile 的张数（加杠的 pon 算 3 张）。"""
        n = 0
        for kind, t, _ in self.melds:
            if t == tile:
                n += 4 if kind in ("ankan", "minkan", "addkan") else 3
        return n

    def total_of_tile(self, tile):
        return self.hand_count_for_meld(tile) + self.meld_count_for_tile(tile)


class Phase:
    DISCARD = "discard"
    RESPONSE = "response"
    END = "end"


class GameResult:
    __slots__ = ("result_type", "winner", "from_player", "points_delta",
                 "scores", "yaku", "han", "fu", "tenpai", "log", "turns",
                 "deltas", "kyokuten", "reason",
                 "dora_count", "aka_count", "ura_dora_count")

    def __init__(self, result_type, winner, from_player, points_delta,
                 scores, yaku, han, fu, tenpai, log, turns, kyokuten=0,
                 reason="", dora_count=0, aka_count=0, ura_dora_count=0):
        self.result_type = result_type
        self.winner = winner
        self.from_player = from_player
        self.points_delta = points_delta
        self.scores = scores
        self.yaku = yaku
        self.han = han
        self.fu = fu
        self.tenpai = tenpai
        self.log = log
        self.turns = turns
        self.kyokuten = kyokuten
        self.reason = reason
        self.dora_count = int(dora_count)
        self.aka_count = int(aka_count)
        self.ura_dora_count = int(ura_dora_count)

    def __repr__(self):
        return (f"Game[{self.result_type} winner={self.winner} "
                f"from={self.from_player} delta={self.points_delta}]")


def net_points(result, start_points=None):
    """本局每家净点数变化：优先 `scores - start_points`，否则回退 delta。

    start_points 为该局各家起手点数（MahjongEnv.initial_points）。
    统计/排名用这个口径；points_delta 只是转账流水，会漏掉立直成本。
    """
    scores = getattr(result, "scores", None)
    if scores is None or start_points is None:
        return list(result.points_delta)
    return [scores[i] - start_points[i] for i in range(N_PLAYERS)]


class MahjongEnv:
    """单局环境。4 agent 混战，多 agent 原生（ARENA.md §2）。

    对外接口：
        reset(seed) -> None
        pending_players -> list[int]   本步需要决策的主体
        legal_actions(p) -> list[Action]
        step(actions) -> GameResult or None
    """

    def __init__(self, seed=None, dealer=0, honba=0, round_wind=27,
                 kyoku=1, wall_dead=False, start_kyokuten=0,
                 start_points=None):
        """start_points: 各家本局起手点数（半庄接力用）。默认每家 25000。

        半庄里上一局的结算结果必须作为下一局的起手——旧实现每局都从
        25000 重开，导致持越点、立直资格（<1000 不能立直）、观测里的
        点数特征全部失真。`HanchanState.new_env` 现在传本局真实点数。
        """
        if start_points is None:
            start_points = [RS.STARTING_POINTS] * N_PLAYERS
        if len(start_points) != N_PLAYERS:
            raise ValueError(f"start_points 需要 {N_PLAYERS} 个数，"
                             f"收到 {len(start_points)}")
        self.initial_points = [int(x) for x in start_points]
        self.initial_kyokuten = start_kyokuten
        self.rng = random.Random(seed)
        self.dealer = dealer
        self.honba = honba
        self.round_wind = round_wind        # 场风 tile id
        self.kyoku = kyoku
        self.wall_dead = wall_dead
        self.reset(seed)

    # ---------------- 初始化 ----------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.players = []
        for i in range(N_PLAYERS):
            seat_wind = 27 + ((i - self.dealer) % 4)
            p = Player(i, self.initial_points[i], seat_wind)
            p.is_dealer = (i == self.dealer)
            self.players.append(p)

        # 牌山：136 张洗牌，末尾 14 张 wall[122:136] 严格作为王牌（死牌山）：
        #   122..126: 表宝牌指示牌（最多 5 张）
        #   127..131: 里宝牌指示牌（最多 5 张，与表宝牌 1-to-1 对应）
        #   132..135: 岭上补牌（最多 4 杠）
        self.wall = [t for t in range(NUM_TILES) for _ in range(4)]
        self.rng.shuffle(self.wall)
        self.wall_idx = 0
        self.dead_wall_size = 14 if (self.wall_dead or RS.USE_WARIBASHI) else 0
        self.kan_count = 0                  # 本局累计开杠次数（上限 4，每次杠活牌尾前移 1）
        self.rinshan_drawn = 0              # 已从王牌岭上摸出的张数
        self.dora_indicator = None          # 第一张（兼容旧接口）
        self.dora_indicators = []           # 全部宝牌指示牌（初始 1 张，每次杠 +1）
        self.ura_dora_indicator = None
        self.ura_dora_indicators = []
        self._set_dora_indicator()

        # 配牌
        for p in self.players:
            p.hand = [self.wall[self.wall_idx + k] for k in range(13)]
            self.wall_idx += 13
            self._sort_hand(p)
            p.furiten = False
            p.riichi = False
            p.riichi_turn = -1
            p.double_riichi = False
            p.ippatsu = False

        self.turn = 0               # 当前巡目（从 0 开始）
        self.turn_player = self.dealer
        self.phase = Phase.DISCARD
        self.last_discard = None
        self.last_discard_player = None
        self.pending = [self.turn_player]
        self.log = []
        self.kan_pending_draw = False
        self.kyokuten = getattr(self, "initial_kyokuten", 0)
        self.renchan_player = self.dealer
        self._must_discard_after_kan = -1

    def _set_dora_indicator(self):
        self.dora_indicators = []
        self._flip_dora()

    def _flip_dora(self):
        """从王牌（wall[122..126]）翻一张表宝牌指示牌：初始 1 张，每次杠 +1，最多 5 张。"""
        idx = len(self.dora_indicators)
        if idx >= 5:
            return
        pos = len(self.wall) - 14 + idx
        if pos < 0 or pos >= len(self.wall):
            return
        t = self.wall[pos]
        self.dora_indicators.append(t)
        self.dora_indicator = self.dora_indicators[0]

    def _live_wall_end(self):
        """活牌山可摸末尾下标：136 - 14(王牌) - kan_count(每次杠补岭上，尾端前移 1)。"""
        dead = getattr(self, "dead_wall_size", 0)
        return max(0, len(self.wall) - dead - getattr(self, "kan_count", 0))

    def _drawable_left(self):
        """当前活牌山剩余可摸张数（0 杠起手 = 70）。"""
        return max(0, self._live_wall_end() - self.wall_idx)

    def _sort_hand(self, p):
        p.hand.sort()

    def _draw_tile(self, p):
        """从活牌山摸一张。活牌山耗尽返回 None。"""
        if self.wall_idx >= self._live_wall_end():
            return None
        t = self.wall[self.wall_idx]
        self.wall_idx += 1
        p.hand.append(t)
        p.last_drawn = t
        return t

    def _draw_rinshan(self, p):
        """开杠后从王牌岭上（wall[132..135]）补摸一张。"""
        if self.rinshan_drawn >= 4:
            return None
        pos = len(self.wall) - 4 + self.rinshan_drawn
        t = self.wall[pos]
        self.rinshan_drawn += 1
        self.kan_count += 1
        p.hand.append(t)
        p.last_drawn = t
        return t

    # ---------------- 牌数守恒 ----------------

    def check_tile_conservation(self):
        """返回 (是否守恒, 详情)。selftest T1 用。"""
        total = 0
        detail = {}
        for p in self.players:
            n = len(p.owned_tiles())
            total += n
            detail[f"hand{p.idx}"] = n
        for p in self.players:
            n = len(p.river)
            total += n
            detail[f"river{p.idx}"] = n
        # 牌山未摸部分（活牌山未摸 + 14 张王牌扣除已摸岭上牌；指示牌仍在王牌内不重复计）
        wall_left = len(self.wall) - self.wall_idx - getattr(self, "rinshan_drawn", 0)
        total += wall_left
        detail["wall"] = wall_left
        return total == NUM_TOTAL_TILES, total, detail

    def visible_counts(self):
        """所有已公开的牌（手牌 + 副露 + 牌河 + 牌山剩 + 指示牌）。"""
        c = [0] * NUM_TILES
        for p in self.players:
            for t in p.owned_tiles():
                c[t] += 1
        for t in self.wall[self.wall_idx:]:
            c[t] += 1
        if self.dora_indicator is not None:
            c[self.dora_indicator] += 1
        if self.ura_dora_indicator is not None:
            c[self.ura_dora_indicator] += 1
        return c

    # ------------------------------------------------------------------
    # 合法动作枚举（RULES.md §5）
    # ------------------------------------------------------------------

    def legal_actions(self, p_idx):
        """返回该主体本步的全部合法动作。"""
        if self.phase == Phase.DISCARD:
            return self._legal_discard_phase(p_idx)
        if self.phase == Phase.RESPONSE:
            return self._legal_response_phase(p_idx)
        return [PASS_ACTION]

    # --- 自摸后（打牌 / 和了 / 立直 / 杠 / 流局） ---

    def _legal_discard_phase(self, p_idx):
        p = self.players[p_idx]
        hand = p.hand
        if p.riichi:
            # 立直后手牌锁定：只能打刚摸的那张（摸切），不能换牌。
            # 仍允许自摸和了。
            acts = []
            if self._can_tsumo(p_idx):
                acts.append(Action(ACT_TSUMO))
            if p.last_drawn is not None and p.last_drawn >= 0 and p.last_drawn in hand:
                acts.append(Action(ACT_DISCARD, p.last_drawn))
            elif hand:
                acts.append(Action(ACT_DISCARD, hand[-1]))
            return acts
        acts = []
        seen = set()
        for t in hand:
            if t in seen:
                continue
            seen.add(t)
            acts.append(Action(ACT_DISCARD, t))

        c = p.counts()

        # 和了（自摸）：14 张 + 非振听 + 有役
        if self._can_tsumo(p_idx):
            acts.append(Action(ACT_TSUMO))

        # 立直：门前 + 巡目 >= 4 + 点棒 >= 1000 + 打后听牌
        # 每一张「打出后能听牌」的牌各生成一个 RIICHI(t) 动作——
        # 立直与打牌是同一个动作（宣告后立刻打出那张牌），
        # 原实现不传 tile，导致动作里没有「打哪张」的信息。
        if self._can_riichi(p_idx):
            for t in sorted(seen):
                if self._is_tenpai(p_idx, extra=t):
                    acts.append(Action(ACT_RIICHI, t))

        # 宣言流局：听牌
        if self._is_tenpai(p_idx):
            acts.append(Action(ACT_RYUUKYOKU))
            acts.append(Action(ACT_NAGASHI))

        # 暗杠/加杠：非立直 + 全局杠数 < 4 + 活牌山仍有可摸牌
        can_kan_now = (not p.riichi and self.kan_count < 4 and self._drawable_left() > 0)
        if can_kan_now:
            for t in seen:
                if hand.count(t) == 4:
                    acts.append(Action(ACT_ANKAN, t))
            for kind, t, _ in p.melds:
                if kind == "pon" and p.hand_count_for_meld(t) >= 1:
                    acts.append(Action(ACT_ADD_KAN, t))

        return acts

    def _can_tsumo(self, p_idx):
        """能否自摸和了：牌型成立 + 有役。

        规则（RULES.md §11，2026-11-04 修正）：振听**只禁止荣和**，
        自摸和了不受振听影响。旧实现在这里拦了 furiten、却允许振听荣和，
        方向正好反了。

        算分与役判定改用**开源引擎**（refengine），
        自研的 hand.py/yaku.py 在使用中暴露过多个真 bug。
        """
        p = self.players[p_idx]
        c = p.full_counts()
        # 杠牌每组占 4 张，归一化成 3 张做牌型张数校验
        c_norm = list(c)
        for kind, t, _ in p.melds:
            if kind in ("ankan", "minkan", "addkan") and c_norm[t] > 0:
                c_norm[t] -= 1
        if sum(c_norm) != 14:
            return False
        if REF.ref_shanten(c_norm) != -1:
            return False
        score = self._best_score(p_idx, c, len(p.melds), is_tsumo=True,
                                 winning_tile=None, from_player=None)
        return score is not None and score.is_valid

    def _can_riichi(self, p_idx):
        """能否立直：门前 + 第 4 巡后 + 点棒 >= 1000 + 存在「打后听牌」的牌。"""
        p = self.players[p_idx]
        if not p.is_menzen() or p.riichi:
            return False
        if self.turn < 3:          # 巡目从 0 起，第 4 巡 = turn 3
            return False
        if p.points < RS.START_KYOTEN:
            return False
        # 必须存在一张牌，打出后听牌。
        # 用 _is_tenpai(extra=t) 直接判定「打掉 t 后是否听牌」，
        # 不再反复摘牌/装牌（那是更早版本的主要开销来源）。
        for t in set(p.hand):
            if self._is_tenpai(p_idx, extra=t):
                return True
        return False

    def _is_tenpai(self, p_idx, extra=None):
        """听牌判定：13 张手牌 + 待 1 张即成和了牌型。

        这是**规则判定**（能否宣言流局 / 能否立直），不是策略先验。

        extra: 可选的「再打一张」参数（tile id），用于一次算完
        「打哪张能听牌」，避免 _can_riichi 对 14 种牌各算一遍。
        """
        p = self.players[p_idx]
        c = p.full_counts()
        # 杠牌每组多 1 张，扣掉 1 张使暗牌+副露归一化回 13/14 张
        for kind, t, _ in p.melds:
            if kind in ("ankan", "minkan", "addkan") and c[t] > 0:
                c[t] -= 1
        meld_count = len(p.melds)
        total = sum(c)
        if total == 14:
            # 14 张时，先去掉「刚摸到的」或「指定要打的那张」。
            # 默认必须去 last_drawn（摸切）：旧实现去「最小牌号的牌」，
            # 等于拿错误的 13 张判听牌，导致摸牌后几乎永远判不出听牌——
            # 直接后果是弃牌阶段从不生成 RYUUKYOKU/NAGASHI 动作
            # （回归测试 test_never_declares_ryuukyoku 在这里抓到）。
            drop = extra
            if drop is None or drop < 0 or c[drop] == 0:
                ld = getattr(p, "last_drawn", -1)
                if ld is not None and 0 <= ld < NUM_TILES and c[ld] > 0:
                    drop = ld
                else:
                    drop = -1
                    for t in range(NUM_TILES):
                        if c[t] > 0:
                            drop = t
                            break
            if drop < 0:
                return False
            c[drop] -= 1
            total = 13
        if total != 13:
            return False
        # 用开源向听数判定：0 = 听牌，-1 = 已和了
        return REF.ref_shanten(c) <= 0

    def _has_any_yaku(self, p_idx, counts, meld_count, is_tsumo=True,
                       winning_tile=None):
        """是否有役。遍历所有分解取最高番。

        winning_tile 必须传，否则放铳时枚举器看不到那张牌，
        会把合法的和了判成「无役」。
        """
        res = self._best_score(p_idx, counts, meld_count, is_tsumo=is_tsumo,
                               winning_tile=winning_tile)
        return res is not None and res.is_valid

    def _best_score(self, p_idx, counts, meld_count, is_tsumo,
                    winning_tile=None, from_player=None):
        """枚举全部和了分解，返回分最高的一个。无役返回 None。

        counts 是**含和了牌**的完整 14 张计数（和了牌可能是刚打出的、
        并不在玩家手上的那张）。winning_tile 用于把那张补进暗牌计数，
        否则枚举器看不到它，13 张永远凑不出 4 面子 + 雀头。
        """
        p = self.players[p_idx]
        concealed = p.counts()
        if winning_tile is not None:
            concealed[winning_tile] += 1

        best = None
        # 标准型
        for melds, pair_tile in H.enumerate_all_with_melds(concealed, p.melds):
            # 面子总数必须是 4 组（3 张 1 组 = 1 个元素）
            if len(melds) != 4:
                continue
            cand = self._score_decomposition(
                p_idx, melds, pair_tile, counts, meld_count, is_tsumo)
            if cand is not None and cand.is_valid:
                if best is None or (cand.han > best.han):
                    best = cand
        # 七对子
        if meld_count == 0 and H.is_chiitoitsu(counts, 0):
            cand = self._score_chiitoi(p_idx, counts, is_tsumo)
            if cand is not None and cand.is_valid:
                if best is None or cand.han > best.han:
                    best = cand
        # 国士
        if H.is_kokushi(counts, meld_count):
            cand = self._score_kokushi(p_idx, counts, meld_count, is_tsumo)
            if cand is not None and best is None:
                best = cand
        return best

    def _score_decomposition(self, p_idx, melds, pair_tile, counts,
                             meld_count, is_tsumo):
        p = self.players[p_idx]
        # 把明副露（chi/pon/minkan/addkan）标记为 concealed=False，暗杠与暗牌分解标记为 True
        open_pool = [(m[0], m[1]) for m in p.melds if m[0] != "ankan"]
        tagged = []
        for kind, t, *rest in melds:
            if (kind, t) in open_pool:
                open_pool.remove((kind, t))
                tagged.append((kind, t, False))
            else:
                tagged.append((kind, t, True))
        is_menzen = p.is_menzen()
        dora_c = self._dora_count(p_idx, tagged, pair_tile)
        aka_c = self._aka_count(p_idx)
        ura_c = self._ura_dora_count(p_idx, tagged, pair_tile)
        self._last_dora_counts = (dora_c, aka_c, ura_c)
        return S.evaluate(
            tagged, pair_tile, counts, meld_count, is_menzen,
            is_tsumo, p.riichi, p.ippatsu, p.double_riichi,
            p.seat_wind, self.round_wind, False,
            dora_c, aka_c, ura_c,
        )

    def _score_chiitoi(self, p_idx, counts, is_tsumo):
        p = self.players[p_idx]
        aka_c = self._aka_count(p_idx)
        self._last_dora_counts = (0, aka_c, 0)
        return S.evaluate(
            [(H.CHI, 0)], 0, counts, 0, True, is_tsumo, p.riichi,
            p.ippatsu, p.double_riichi, p.seat_wind, self.round_wind, True,
            0, aka_c, 0,
        )

    def _score_kokushi(self, p_idx, counts, meld_count, is_tsumo):
        p = self.players[p_idx]
        pair = H.kokushi_pair(counts)
        aka_c = self._aka_count(p_idx)
        self._last_dora_counts = (0, aka_c, 0)
        return S.evaluate(
            [(H.CHI, 0)], pair if pair is not None else 0, counts, meld_count,
            True, is_tsumo, p.riichi, p.ippatsu, p.double_riichi,
            p.seat_wind, self.round_wind, False, 0, aka_c, 0,
        )

    # ------------------------------------------------------------------
    # 宝牌计数（RULES.md §7.5）
    # ------------------------------------------------------------------

    def _dora_set(self):
        """当前有效的宝牌 tile id 集合（遍历全部指示牌：初始 1 + 杠后追加）。"""
        out = set()
        inds = getattr(self, "dora_indicators", None)
        if inds:
            for ind in inds:
                out.add(T.dora_from_indicator(ind))
        elif self.dora_indicator is not None:
            out.add(T.dora_from_indicator(self.dora_indicator))
        return out

    def _ura_set(self):
        out = set()
        inds = getattr(self, "ura_dora_indicators", None)
        if inds:
            for ind in inds:
                out.add(T.dora_from_indicator(ind))
        elif self.ura_dora_indicator is not None:
            out.add(T.dora_from_indicator(self.ura_dora_indicator))
        return out

    def _aka_count(self, p_idx):
        """**和牌者**手里（含副露）赤牌的张数（每种赤牌最多 1 张，上限 3）。"""
        p = self.players[p_idx]
        owned = p.owned_tiles()
        n = 0
        for t in RS.AKA_TILES:
            if t in owned:
                n += 1
        return min(n, 3)

    def _dora_count(self, p_idx, melds, pair_tile):
        """和了牌中的宝牌数（含手牌与全部副露牌）。"""
        dora = self._dora_set()
        p = self.players[p_idx]
        return sum(1 for t in p.owned_tiles() if t in dora)

    def _ura_dora_count(self, p_idx, melds, pair_tile):
        p = self.players[p_idx]
        if not p.riichi:
            return 0
        ura = self._ura_set()
        return sum(1 for t in p.owned_tiles() if t in ura)

    # ------------------------------------------------------------------
    # 响应阶段（RULES.md §5.2）
    # ------------------------------------------------------------------

    def _legal_response_phase(self, p_idx):
        p = self.players[p_idx]
        discarder = self.last_discard_player
        tile = self.last_discard
        acts = [PASS_ACTION]
        if tile is None:
            return acts
        # 立直中不能吃碰杠
        if p.riichi:
            if self._can_ron(p_idx):
                acts.insert(0, Action(ACT_RON, tile))
            return acts

        # 和了
        if self._can_ron(p_idx):
            acts.insert(0, Action(ACT_RON, tile))

        # 宣言流局
        if self._is_tenpai(p_idx):
            acts.append(Action(ACT_RYUUKYOKU))
            acts.append(Action(ACT_NAGASHI))

        # 吃：只有上家，字牌不行
        if p_idx == (discarder + 1) % N_PLAYERS and not T.is_honor(tile):
            for start in _chi_starts(tile):
                # 顺子三张是 start/start+1/start+2，其中 tile 来自弃牌，
                # 手牌必须另外凑齐剩下两张（各至少 1 张，且不含 tile 本身）。
                need = [t for t in (start, start + 1, start + 2) if t != tile]
                if len(need) == 2 and all(p.hand_count_for_meld(t) >= 1
                                          for t in need):
                    acts.append(Action(ACT_CHI, start, tile))

        # 碰：手牌（暗牌）里必须有 2 张——副露里的牌不能用于新碰，
        # 因为碰出的刻子要从暗牌里取。规则上「持 4 副共 16 张」指的是
        # 手上真的有牌，不是副露里已有。
        if p.hand_count_for_meld(tile) >= 2 and len(p.melds) < 4:
            acts.append(Action(ACT_PON, tile))

        # 明杠：手牌里必须有 3 张 + 全局杠数 < 4 + 活牌山仍有可摸牌
        if (p.hand_count_for_meld(tile) >= 3 and len(p.melds) < 4
                and self.kan_count < 4 and self._drawable_left() > 0):
            acts.append(Action(ACT_MINKAN, tile))

        return acts

    def _is_furiten_for_tile(self, p_idx, tile):
        """振听判定：对指定牌是否振听（舍张振听）。

        规则（RULES.md §11）：自己牌河里出现过的牌，若它能让手牌和了，
        则处于振听状态。

        返回 True 表示「这张牌一来就振听」。
        """
        p = self.players[p_idx]
        if tile not in p.river:
            return False
        return REF.can_win_with(p.full_counts(), tile, len(p.melds))

    def _can_ron(self, p_idx, from_chankan=False):
        """能否和这张弃牌。

        振听规则（RULES.md §11，2026-11-04 修正）：**振听禁止一切荣和**
        （普通放铳与抢杠都不行）；自摸不受影响（见 `_can_tsumo`）。
        旧实现方向反了：振听可以放铳和、反而禁止自摸。
        """
        p = self.players[p_idx]
        tile = self.last_discard
        if tile is None:
            return False
        if p.furiten:
            return False          # 振听：不能荣和（含抢杠）
        c = p.full_counts()
        if c[tile] >= 4:
            return False
        c_norm = list(c)
        for kind, t, _ in p.melds:
            if kind in ("ankan", "minkan", "addkan") and c_norm[t] > 0:
                c_norm[t] -= 1
        c_norm[tile] += 1
        meld_count = len(p.melds)
        if sum(c_norm) != 14 or REF.ref_shanten(c_norm) != -1:
            return False
        # 有役检查：否则会生成「牌型成立但无役」的 RON，
        # 策略选中后 _finish_ron 又静默丢弃（实测 21 次选 RON 只结算 1 次）。
        c2 = list(c)
        c2[tile] += 1
        score = self._best_score(p_idx, c2, meld_count, is_tsumo=False,
                                 winning_tile=tile,
                                 from_player=self.last_discard_player)
        return score is not None and score.is_valid
    # ------------------------------------------------------------------
    # 推进对局
    # ------------------------------------------------------------------

    def start_turn(self):
        """开始一巡：庄家先摸牌。"""
        p = self.players[self.turn_player]
        t = self._draw_tile(p)
        if t is None:
            return self._exhaustive_draw()
        p.hand.sort()
        self._update_furiten(p)
        self.phase = Phase.DISCARD
        self.pending = [self.turn_player]
        self.last_discard = None
        self.kan_pending_draw = False
        return None

    def _update_furiten(self, p):
        """振听判定（RULES.md §11）。

        舍张振听：自己牌河里出现过的牌，若它能让手牌和了，则振听。
        同巡振听：并入舍张判定（本实现不单独区分，因同巡打出的牌
        也在牌河里）。
        """
        c = p.full_counts()
        meld_count = len(p.melds)
        if sum(c) == 14 and p.last_drawn is not None and p.last_drawn >= 0:
            # 刚摸牌后是 14 张；振听判定必须回到 13 张（can_win_with 的约定），
            # 否则等于拿 15 张去判，结果不可信。
            c = list(c)
            c[p.last_drawn] -= 1
        river = set(p.river)
        if not river:
            p.furiten = False
            return
        # 尝试每张「已打出的牌」能否和了（用开源引擎判定）
        for t in river:
            if REF.can_win_with(c, t, meld_count):
                p.furiten = True
                return
        p.furiten = False

    def pending_players(self):
        return list(self.pending)

    def step(self, actions):
        """执行本步动作。

        actions: dict {player_idx: Action}，只包含 pending 里的主体。
        返回 None 表示继续，或 GameResult 表示终局。
        """
        if self.phase == Phase.DISCARD:
            return self._step_discard(actions)
        if self.phase == Phase.RESPONSE:
            return self._step_response(actions)
        return None

    def _step_discard(self, actions):
        p_idx = self.pending[0]
        p = self.players[p_idx]
        act = actions.get(p_idx, PASS_ACTION)

        if act.type == ACT_TSUMO:
            return self._finish_tsumo(p_idx)
        if act.type == ACT_RYUUKYOKU:
            return self._finish_ryuukyoku(reason="ryuukyoku")
        if act.type == ACT_NAGASHI:
            return self._finish_ryuukyoku(reason="nagashi")

        if act.type in (ACT_ANKAN, ACT_MINKAN, ACT_ADD_KAN):
            return self._do_kan(p_idx, act)

        if act.type == ACT_RIICHI:
            # 立直宣告：牌由 act.tile 指定（宣告立直后立即打出该牌）。
            p.riichi = True
            p.genbutsu.update(p.river)
            p.double_riichi = (self.turn >= 3 and len(p.river) == 0)
            p.riichi_turn = self.turn
            p.ippatsu = True
            p.points = max(0, p.points - RS.START_KYOTEN)
            self.kyokuten += RS.START_KYOTEN
            if act.tile not in p.hand:
                act = Action(ACT_DISCARD, p.hand[-1])

        # 打牌
        tile = act.tile
        if tile not in p.hand:
            # 非法动作的兜底：打摸到的那张
            tile = p.hand[-1]
        p.hand.remove(tile)
        p.hand.sort()
        p.river.append(tile)
        p.genbutsu.add(tile)
        # 立直宣告后全员打出的牌均成为该立直家的现物
        for q in self.players:
            if q.riichi:
                q.genbutsu.add(tile)
        self.last_discard = tile
        self.last_discard_player = p_idx
        _ev_kind = "riichi" if act.type == ACT_RIICHI else ("tsumogiri" if tile == p.last_drawn else "discard")
        self.log.append((self.turn, p_idx, _ev_kind, T.tile_name(tile)))

        # 立直后的一发在任何人动作后失效
        for q in self.players:
            if q.riichi and q.idx != p_idx:
                q.ippatsu = False

        # 进入响应阶段
        self.phase = Phase.RESPONSE
        # 注意：**立直玩家也要进响应阶段**——立直后不能吃碰杠，但可以
        # 荣和（放铳和）。旧实现把立直者排除，导致立直后只能自摸、
        # 荣和恒为 0（实测 50 局全自摸）。
        self.pending = [q.idx for q in self.players if q.idx != p_idx]
        if not self.pending:
            return self._advance()
        return None

    def _advance(self):
        """无人响应，推进到下一家摸牌。

        RULES.md §3.1：一「巡」= 四家各摸打一次。本字段记的是**摸牌次数**，
        所以单局上限是 MAX_TURNS * N_PLAYERS 次摸牌（默认 18*4 = 72）。
        """
        self.turn += 1
        if self.turn >= RS.MAX_TURNS * N_PLAYERS:
            return self._finish_ryuukyoku(reason="exhaustive")
        if self._drawable_left() <= 0:
            return self._finish_ryuukyoku(reason="exhaustive")
        self.turn_player = (self.last_discard_player + 1) % N_PLAYERS
        self.phase = Phase.DISCARD
        return self.start_turn()

    def _step_response(self, actions):
        """响应阶段：按优先级裁决（RULES.md §5.2）。"""
        tile = self.last_discard
        discarder = self.last_discard_player
        chooser = [i for i in self.pending if i in actions]

        # 1. 和了（含抢杠）
        rons = [i for i in chooser
                if actions[i].type == ACT_RON and self._can_ron(i)]
        if rons:
            rons.sort(key=lambda i: ((discarder - i) % N_PLAYERS))
            winner = rons[0]
            return self._finish_ron(winner, discarder, tile)

        # 2. 碰 / 明杠优先于吃（标准立直麻将规则）
        for act_type in (ACT_PON, ACT_MINKAN, ACT_CHI):
            claimants = [i for i in chooser if actions[i].type == act_type]
            if not claimants:
                continue
            claimants.sort(key=lambda i: ((discarder - i) % N_PLAYERS))
            actor = claimants[0]
            act = actions[actor]
            if act_type == ACT_CHI:
                self._do_chi(actor, act, tile)
            elif act_type == ACT_PON:
                self._do_pon(actor, tile)
            else:
                self._do_minkan_from_discard(actor, act_type, tile)
            return None

        # 5. 宣言流局
        ryuu = [i for i in chooser
                if actions[i].type in (ACT_RYUUKYOKU, ACT_NAGASHI)]
        if ryuu and len(ryuu) == len(self.pending):
            return self._finish_ryuukyoku(reason="ryuukyoku_all")

        return self._advance()

    def _consume_from_river(self, tile):
        """弃牌被副露吃掉时，从原弃牌者的牌河移除（牌数守恒）。

        被吃/碰/杠的牌不再留在牌河里，它成为副露的一部分。
        """
        src = self.players[self.last_discard_player]
        if src.river and src.river[-1] == tile:
            src.river.pop()
        else:
            # 兜底：若最后一张不是（异常），从牌河移除一个
            if tile in src.river:
                src.river.remove(tile)

    def _do_chi(self, actor, act, tile):
        """吃：act.tile = 顺子起始牌，tile = 被吃的弃牌。

        手牌里要取的 2 张 = 顺子的另外两张（不含 tile 本身）。
        """
        p = self.players[actor]
        called_from = self.last_discard_player
        start = act.tile
        seq = (start, start + 1, start + 2)
        need = [t for t in seq if t != tile]
        # 先确认手牌够，再动牌（保证要么成功要么完全不变）
        if not all(p.hand_count_for_meld(t) >= 1 for t in need):
            return
        self._consume_from_river(tile)
        for t in need:
            p.hand.remove(t)
        p.hand.sort()
        p.melds.append(("chi", start, called_from))
        self.log.append((self.turn, actor, "chi", T.tile_name(tile)))
        self.last_discard_player = actor
        self.turn_player = actor
        self.phase = Phase.DISCARD
        self.pending = [actor]
        self.last_discard = None
        self._after_claim(actor)

    def _do_pon(self, actor, tile):
        p = self.players[actor]
        called_from = self.last_discard_player
        self._consume_from_river(tile)
        removed = 0
        while removed < 2 and tile in p.hand:
            p.hand.remove(tile)
            removed += 1
        p.hand.sort()
        if removed == 2:
            p.melds.append(("pon", tile, called_from))
            self.log.append((self.turn, actor, "pon", T.tile_name(tile)))
        else:
            # 手牌不足（含副露时可能发生）：把弃牌放回原牌河，不成立
            src = self.players[called_from]
            src.river.append(tile)
            return
        self.last_discard_player = actor
        self.turn_player = actor
        self.phase = Phase.DISCARD
        self.pending = [actor]
        self.last_discard = None
        self._after_claim(actor)

    def _do_minkan_from_discard(self, actor, act_type, tile):
        p = self.players[actor]
        called_from = self.last_discard_player
        if act_type == ACT_MINKAN:
            # 先确认手牌有 3 张，否则不动牌（保持牌数守恒）
            if p.hand_count_for_meld(tile) < 3:
                return
            self._consume_from_river(tile)
            for _ in range(3):
                p.hand.remove(tile)
            p.hand.sort()
            p.melds.append(("minkan", tile, called_from))
        else:
            # 加杠：自己的 pon 升级为 addkan，并吃掉这张弃牌
            target = -1
            for i, (kind, t, frm) in enumerate(p.melds):
                if kind == "pon" and t == tile:
                    target = i
                    break
            if target < 0:
                return
            self._consume_from_river(tile)
            kind0, t0, frm0 = p.melds[target]
            p.melds[target] = ("addkan", t0, frm0)
        p.hand.sort()
        self.log.append((self.turn, actor, "kan", T.tile_name(tile)))
        self._flip_dora()          # 杠后翻新宝牌指示牌
        self.last_discard_player = actor
        self.turn_player = actor
        self.phase = Phase.DISCARD
        self.pending = [actor]
        self.last_discard = None
        self._after_claim(actor, need_draw=True)

    def _do_kan(self, actor, act):
        p = self.players[actor]
        tile = act.tile
        if act.type == ACT_ANKAN:
            for _ in range(4):
                if tile in p.hand:
                    p.hand.remove(tile)
            p.melds.append(("ankan", tile, actor))
        elif act.type == ACT_ADD_KAN:
            for i, (kind, t, frm) in enumerate(p.melds):
                if kind == "pon" and t == tile:
                    p.melds[i] = ("addkan", t, frm)
                    if tile in p.hand:
                        p.hand.remove(tile)
                    break
        p.hand.sort()
        self.log.append((self.turn, actor, "kan", T.tile_name(tile)))
        self._flip_dora()          # 杠后翻新宝牌指示牌
        self._after_claim(actor, need_draw=True)

    def _after_claim(self, actor, need_draw=False):
        p = self.players[actor]
        p.just_kan = True
        self._update_furiten(p)
        for q in self.players:
            if q.riichi:
                q.ippatsu = False
        if need_draw:
            t = self._draw_rinshan(p)
            if t is None:
                return self._finish_ryuukyoku(reason="wall_empty")
            p.hand.sort()
        self.pending = [actor]
        self.phase = Phase.DISCARD
        self.last_discard = None

    def _exhaustive_draw(self):
        return self._finish_ryuukyoku(reason="wall_empty")

    # ------------------------------------------------------------------
    # 结算（RULES.md §9.5、§10）
    # ------------------------------------------------------------------

    def _finish_tsumo(self, winner):
        p = self.players[winner]
        c = p.full_counts()
        score = self._best_score(winner, c, len(p.melds), is_tsumo=True,
                                 winning_tile=None, from_player=None)
        delta = [0, 0, 0, 0]
        if score is None or not score.is_valid:
            # 无役不能和了（引擎不该走到这里）
            return self._finish_ryuukyoku(reason="no_yaku")

        if p.riichi and RS.USE_URA_DORA:
            self._set_ura_dora()

        total = score.points
        if score.yakuman_mult:
            total = RS.limit_points(score.yakuman_mult)
            if p.is_dealer:
                total *= 3 if score.yakuman_mult <= 3 else 3

        # 自摸：3 家各付一份。
        #   庄家和了：每家付「庄家收点数」（表中庄家列）
        #   子家和了：每家付「子家收点数 / 3」，不足 1 的余数按座次分配
        payers = [i for i in range(N_PLAYERS) if i != winner]
        if p.is_dealer:
            share = total
            for i in payers:
                delta[i] = -share
        else:
            per = total // len(payers)
            rem = total - per * len(payers)
            for k, i in enumerate(payers):
                delta[i] = -(per + (1 if k < rem else 0))
        delta[winner] = -sum(delta) + self.kyokuten   # 立直棒归和牌者
        self.kyokuten = 0
        return self._settle("tsumo", winner, None, delta, score)

    def _order(self, i, dealer):
        """i 相对 dealer 的逆时针距离（0..3）。"""
        return (dealer - i) % N_PLAYERS

    def _finish_ron(self, winner, from_player, tile):
        p = self.players[winner]
        c = p.full_counts()
        c[tile] += 1
        score = self._best_score(winner, c, len(p.melds), is_tsumo=False,
                                 winning_tile=tile, from_player=from_player)
        if score is None or not score.is_valid:
            return self._advance()
        if p.riichi and RS.USE_URA_DORA:
            self._set_ura_dora()

        total = score.points
        if score.yakuman_mult:
            total = RS.limit_points(score.yakuman_mult)
        if p.is_dealer:
            total *= 2
        delta = [0, 0, 0, 0]
        delta[from_player] = -total
        delta[winner] = total + self.kyokuten   # 立直棒归和牌者
        self.kyokuten = 0
        return self._settle("ron", winner, from_player, delta, score)

    def _set_ura_dora(self):
        if self.ura_dora_indicators:
            return
        n_dora = max(1, min(5, len(self.dora_indicators)))
        base = len(self.wall) - 14 + 5  # wall[127..131]
        self.ura_dora_indicators = [self.wall[base + k] for k in range(n_dora) if 0 <= base + k < len(self.wall)]
        if self.ura_dora_indicators:
            self.ura_dora_indicator = self.ura_dora_indicators[0]

    def _finish_ryuukyoku(self, reason):
        """流局结算：听牌罚符 + 连棒。"""
        tenpai = [False, False, False, False]
        for i in range(N_PLAYERS):
            tenpai[i] = self._is_tenpai(i)
        delta = [0, 0, 0, 0]
        n_tenpai = sum(tenpai)
        if 0 < n_tenpai < N_PLAYERS:
            # 标准立直麻将不听罚符（总池恒为 3000）：
            # 1 听 3 不听：+3000 / -1000；2 听 2 不听：+1500 / -1500；3 听 1 不听：+1000 / -3000
            gain = 3000 // n_tenpai
            loss = 3000 // (N_PLAYERS - n_tenpai)
            for i in range(N_PLAYERS):
                delta[i] = gain if tenpai[i] else -loss
        result = self._settle("ryuukyoku", None, None, delta, None,
                              tenpai=tenpai, reason=reason)
        return result

    def _settle(self, result_type, winner, from_player, delta, score,
                tenpai=None, reason=""):
        """统一结算：应用点数变化，检查守恒。

        `scores` = 各家本局结束时的绝对点数（已含立直供托的扣减与和牌者
        收取的供托）。注意 `points_delta` 只是本局的转账流水，**不含**
        立直时直接扣走的 1000 点（那 1000 点作为供托由和牌者收走，记在
        winner 的 delta 里）——所以跨局流转/排名必须用 `scores`，只加
        `points_delta` 会漏掉立直成本。
        """
        scores = [self.players[i].points + delta[i] for i in range(N_PLAYERS)]
        for i in range(N_PLAYERS):
            self.players[i].points = scores[i]
        self.phase = Phase.END
        self.pending = []
        res = GameResult(
            result_type=result_type,
            winner=winner,
            from_player=from_player,
            points_delta=list(delta),
            scores=scores,
            yaku=tuple(score.yakus) if score else (),
            han=score.han if score else 0,
            fu=score.fu if score else 0,
            tenpai=tenpai if tenpai is not None else [False] * N_PLAYERS,
            log=self.log,
            turns=self.turn,
            kyokuten=self.kyokuten,
            reason=reason,
            dora_count=(self._last_dora_counts[0] if score is not None
                        and getattr(self, "_last_dora_counts", None) else 0),
            aka_count=(self._last_dora_counts[1] if score is not None
                       and getattr(self, "_last_dora_counts", None) else 0),
            ura_dora_count=(self._last_dora_counts[2] if score is not None
                            and getattr(self, "_last_dora_counts", None) else 0),
        )
        self.last_result = res
        return res


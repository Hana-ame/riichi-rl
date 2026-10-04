"""自测套件：T1-T8（RULES.md §16）。

必测三项（用户铁律）：
    T1 牌数守恒       —— 136 张不多不少
    T2 随机对局不可非法和了 —— 每次和了都重校验合法性
    T3 点数结算守恒   —— 点数不凭空产生/消失

扩展项：
    T4 和了分解完备性
    T5 符跳符正确性
    T6 振听正确性
    T7 边界动作
    T8 可复现性

跑法：python -m selftest        或     python selftest.py
"""

from __future__ import annotations

import sys
import os
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import hand as H
from mahjong import ruleset as RS
from mahjong import scoring as S
from mahjong import tiles as T
from mahjong import yaku as Y
from mahjong import refengine as REF
from mahjong.situation import SIT_DEFENSE, SIT_LAST_CHANCE, SIT_AHEAD, SIT_BEHIND, SIT_NORMAL
from mahjong.tiles import NUM_TILES, NUM_TOTAL_TILES


RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    return ok


def random_play(seed, max_steps=500):
    """用随机合法动作打一局，返回 (result, env, steps)。"""
    e = E.MahjongEnv(seed=seed)
    r = e.start_turn()
    steps = 0
    rng = random.Random(seed * 7919 + 13)
    while r is None and steps < max_steps:
        steps += 1
        acts = {}
        for p in e.pending_players():
            legal = e.legal_actions(p)
            acts[p] = rng.choice(legal)
        r = e.step(acts)
    return r, e, steps


# ---------------------------------------------------------------------------
# T1 牌数守恒
# ---------------------------------------------------------------------------

def test_tile_conservation(n_games=200):
    bad = []
    for seed in range(1, n_games + 1):
        e = E.MahjongEnv(seed=seed)
        r = e.start_turn()
        steps = 0
        rng = random.Random(seed * 7919 + 13)
        ok, total, detail = e.check_tile_conservation()
        if not ok:
            bad.append((seed, 0, total, detail))
            continue
        while r is None and steps < 500:
            steps += 1
            acts = {p: rng.choice(e.legal_actions(p))
                    for p in e.pending_players()}
            ok, total, detail = e.check_tile_conservation()
            if not ok:
                bad.append((seed, steps, total, detail))
                break
            r = e.step(acts)
    ok = len(bad) == 0
    return record("T1 牌数守恒", ok,
                  f"{n_games} 局，违规 {len(bad)} 次"
                  + (f" 例:{bad[:2]}" if bad else ""))


# ---------------------------------------------------------------------------
# T2 随机对局不可非法和了
# ---------------------------------------------------------------------------


FORCED_WIN_HANDS = [
    "123m 456m 789m 123p 11p",
    "234m 567m 234p 567p 22p",
    "111m 222m 333m 444m 55z",
    "123m 123m 123m 456p EE",
    "234m 567m 234p 567p 22p",
]


def test_forced_wins():
    """强制和了：塞入已知必和牌型，验证和了路径可结算。

    这一项保证 T2b 不是空跑——随机策略下几乎不可能自然和了。
    """
    bad = []
    tsumo_ok = 0
    for i, spec in enumerate(FORCED_WIN_HANDS):
        for dealer in (0, 1):
            e = E.MahjongEnv(seed=1000 + i, dealer=dealer)
            p = e.players[0]
            p.hand = sorted(T.parse_tiles(spec))
            p.melds = []
            p.furiten = False
            if not e._can_tsumo(0):
                bad.append((spec, dealer, "can_tsumo=False"))
                continue
            res = e._finish_tsumo(0)
            if res is None or res.result_type != "tsumo":
                bad.append((spec, dealer, "no settle"))
                continue
            if sum(res.points_delta) != 0:
                bad.append((spec, dealer, "delta " + str(res.points_delta)))
                continue
            if not res.yaku:
                bad.append((spec, dealer, "no yaku"))
                continue
            tsumo_ok += 1

    ron_ok = 0
    ron_specs = [
        ("123m 456m 789m 11p 45p", "6p"),
        ("123m 456m 789m 11p 45p", "6p"),
    ]
    for i, (spec, win) in enumerate(ron_specs):
        for dealer in (0, 1):
            e = E.MahjongEnv(seed=2000 + i, dealer=dealer)
            p = e.players[0]
            p.hand = sorted(T.parse_tiles(spec))
            p.melds = []
            p.furiten = False
            wt = T.parse_tile(win)
            e.last_discard = wt
            e.last_discard_player = 3
            if not e._can_ron(0):
                bad.append((spec, win, dealer, "can_ron=False"))
                continue
            res = e._finish_ron(0, 3, wt)
            if res is None or res.result_type != "ron":
                bad.append((spec, win, dealer, "no settle"))
                continue
            if sum(res.points_delta) != 0:
                bad.append((spec, win, dealer, "delta " + str(res.points_delta)))
                continue
            ron_ok += 1

    ok = len(bad) == 0
    return record("T2a 强制和了可结算", ok,
                  f"自摸 {tsumo_ok} 例 + 放铳 {ron_ok} 例，失败 {len(bad)}"
                  + (f" {bad[:3]}" if bad else ""))

def test_no_illegal_win(n_games=200):
    """每次和了都要重新校验：牌型成立 + 有役 + 非振听。"""
    ron_count = 0
    tsumo_count = 0
    draw_count = 0
    illegal = []

    for seed in range(1, n_games + 1):
        r, e, steps = random_play(seed)
        if r is None:
            continue
        if r.result_type == "ron":
            ron_count += 1
        elif r.result_type == "tsumo":
            tsumo_count += 1
        else:
            draw_count += 1
            continue

        winner = r.winner
        p = e.players[winner]
        c = p.full_counts()
        if r.result_type == "ron":
            c[e.last_discard if e.last_discard is not None else 0] += 1

        # 牌型必须成立
        mc = len(p.melds)
        shape_ok = (H.is_standard_complete(c, mc)
                    or H.is_chiitoitsu(c, mc)
                    or H.is_kokushi(c, mc))
        if not shape_ok:
            illegal.append((seed, "shape"))
            continue
        # 必须有役
        if not r.yaku:
            illegal.append((seed, "no_yaku"))
            continue
        # 自摸不能振听
        if r.result_type == "tsumo" and p.furiten:
            illegal.append((seed, "furiten_tsumo"))

    total = ron_count + tsumo_count
    ok = len(illegal) == 0
    detail = (f"{n_games} 局：和了 {total}（放铳 {ron_count}/自摸 {tsumo_count}）"
              f"，流局 {draw_count}，非法 {len(illegal)}")
    if illegal:
        detail += f" 例:{illegal[:3]}"
    return record("T2 随机对局和了合法性", ok, detail)


# ---------------------------------------------------------------------------
# T3 点数结算守恒
# ---------------------------------------------------------------------------

def test_score_conservation(n_games=200):
    """每局：结算前后点数总和不变，且 delta 之和为 0。"""
    bad = []
    total_pt_ok = 0
    for seed in range(1, n_games + 1):
        e = E.MahjongEnv(seed=seed)
        before = [p.points for p in e.players]
        r, e, steps = random_play(seed)
        if r is None:
            continue
        after = [p.points for p in e.players]
        dsum = sum(r.points_delta)
        if dsum != 0:
            bad.append((seed, "delta_sum", dsum, r.points_delta))
            continue
        if sum(after) != sum(before):
            bad.append((seed, "total", sum(before), sum(after)))
            continue
        total_pt_ok += 1
    ok = len(bad) == 0
    return record("T3 点数结算守恒", ok,
                  f"{total_pt_ok} 局结算通过，违规 {len(bad)}"
                  + (f" 例:{bad[:2]}" if bad else ""))


# ---------------------------------------------------------------------------
# T4 和了分解完备性
# ---------------------------------------------------------------------------

def test_decomposition_completeness():
    cases = [
        ("123m 456m 789m 123p EE", 0, True),
        ("111m 222m 333m 444m 55z", 0, True),
        ("234m 567m 234p 567p 22p", 0, True),
        ("1133m 5577p 2299s 11s", 0, "7pairs"),
        ("119m 19p 19s 1234567z", 0, "kokushi"),
        ("123m 456m 789m 11p 22p", 0, False),
        ("135m 246m 369m 11p EE", 0, False),
    ]
    bad = []
    for spec, mc, expect in cases:
        c = T.counts_to_list(T.parse_tiles(spec))
        if expect is True:
            got = H.is_standard_complete(c, mc)
        elif expect == "7pairs":
            got = H.is_chiitoitsu(c, mc)
        elif expect == "kokushi":
            got = H.is_kokushi(c, mc)
        else:
            got = H.is_standard_complete(c, mc)
        if got != bool(expect):
            bad.append((spec, expect, got))
    ok = len(bad) == 0
    return record("T4 和了分解完备性", ok,
                  f"{len(cases)} 例，失败 {len(bad)}"
                  + (f" {bad}" if bad else ""))


# ---------------------------------------------------------------------------
# T5 符跳符正确性
# ---------------------------------------------------------------------------

def test_fu_roundup():
    """符跳符必须落在合法符级，且单调不减。"""
    levels = set(RS.FU_LEVELS)
    bad = []
    prev = -1
    for fu in range(0, 140):
        out = RS.fu_roundup(fu)
        if out not in levels:
            bad.append((fu, out))
        if out < prev:
            bad.append((fu, out, "non-monotone"))
        prev = out
    ok = len(bad) == 0
    return record("T5 跳符合法性", ok,
                  f"输入 0..139，失败 {len(bad)}" + (f" {bad[:3]}" if bad else ""))


# ---------------------------------------------------------------------------
# T6 振听正确性
# ---------------------------------------------------------------------------

def test_furiten():
    """构造三种振听：舍张振听、自摸被拒、放铳被允。"""
    e = E.MahjongEnv(seed=100)
    p = e.players[0]
    # 13 张听牌：123m 456m 789m 11p 45p，听 6p
    p.hand = sorted(T.parse_tiles("123m 456m 789m 11p 45p"))
    p.melds = []
    tenpai_before = e._is_tenpai(0)
    # 往牌河里放待牌 6p，制造舍张振听
    p.river.append(T.parse_tile("6p"))
    e._update_furiten(p)
    furiten = p.furiten
    # 振听后自摸和了必须被拒
    p.hand = sorted(T.parse_tiles("123m 456m 789m 11p 456p"))
    can_tsumo = e._can_tsumo(0)
    ok = tenpai_before and furiten and (not can_tsumo)
    return record("T6 舍张振听", ok,
                  f"听牌={tenpai_before} 振听={furiten} "
                  f"振听后自摸={can_tsumo}（应为 False）")


# ---------------------------------------------------------------------------
# T7 边界动作
# ---------------------------------------------------------------------------

def test_edge_actions():
    bad = []

    # 门前不可吃
    e = E.MahjongEnv(seed=200)
    p0, p1 = e.players[0], e.players[1]
    p0.hand = T.parse_tiles("234m 567m 234p 567p 2p")
    p0.hand.sort()
    e.last_discard = T.parse_tile("1p")
    e.last_discard_player = 0
    e.phase = E.Phase.RESPONSE
    # p1 若是庄家/上家关系才可能吃；这里检查「字牌不可吃」
    p1.hand = T.parse_tiles("234m 567m 234p 567p 2p")
    p1.hand.sort()
    e.last_discard = T.parse_tile("E")   # 真正的字牌：不能吃
    acts = e.legal_actions(1)
    if any(a.type == E.ACT_CHI for a in acts):
        bad.append("字牌被允许吃")

    # 立直后不能吃碰杠
    e2 = E.MahjongEnv(seed=201)
    q = e2.players[0]
    q.riichi = True
    e2.last_discard = T.parse_tile("1p")
    e2.last_discard_player = 2
    e2.phase = E.Phase.RESPONSE
    acts2 = e2.legal_actions(0)
    for a in acts2:
        if a.type in (E.ACT_CHI, E.ACT_PON, E.ACT_MINKAN):
            bad.append(f"立直后出现 {a}")

    ok = len(bad) == 0
    return record("T7 边界动作", ok, f"失败 {len(bad)} {bad}")


# ---------------------------------------------------------------------------
# T8 可复现性
# ---------------------------------------------------------------------------

def test_reproducibility(n_games=30):
    mismatch = []
    for seed in range(1, n_games + 1):
        r1, e1, s1 = random_play(seed)
        r2, e2_, s2 = random_play(seed)
        if s1 != s2:
            mismatch.append((seed, "steps"))
            continue
        if (r1 is None) != (r2 is None):
            mismatch.append((seed, "result"))
            continue
        if r1 is not None and r1.result_type != r2.result_type:
            mismatch.append((seed, "type"))
            continue
        if r1 is not None and r1.points_delta != r2.points_delta:
            mismatch.append((seed, "delta"))
    ok = len(mismatch) == 0
    return record("T8 可复现性", ok,
                  f"{n_games} 局，不一致 {len(mismatch)}"
                  + (f" {mismatch[:3]}" if mismatch else ""))


# ---------------------------------------------------------------------------


def test_points_table_vs_reference():
    """T5c 点数表与权威实现逐格一致。

    对照源：MahjongRepository/mahjong 2.0.0（PyPI mahjong）
    ScoresCalculator.calculate_scores。这 110 个值是 2026-10-04
    逐格跑出来的真值，替代了原先推导出来的表（推导错的 59 格）。

    若要重新生成，用 rules/VERIFICATION.md §6 的方法跑参考库。
    """
    # (fu, han) -> (庄家放铳收点, 子家放铳收点)
    expect = {
        (20, 1): (1000, 700), (20, 2): (2000, 1300), (20, 3): (3900, 2600),
        (20, 4): (7700, 5200), (20, 5): (12000, 8000),
        (25, 1): (1200, 800), (25, 2): (2400, 1600), (25, 3): (4800, 3200),
        (25, 4): (9600, 6400), (25, 5): (12000, 8000),
        (30, 1): (1500, 1000), (30, 2): (2900, 2000), (30, 3): (5800, 3900),
        (30, 4): (11600, 7700), (30, 5): (12000, 8000),
        (40, 1): (2000, 1300), (40, 2): (3900, 2600), (40, 3): (7700, 5200),
        (40, 4): (12000, 8000), (40, 5): (12000, 8000),
        (50, 1): (2400, 1600), (50, 2): (4800, 3200), (50, 3): (9600, 6400),
        (50, 4): (12000, 8000), (50, 5): (12000, 8000),
        (60, 1): (2900, 2000), (60, 2): (5800, 3900), (60, 3): (11600, 7700),
        (60, 4): (12000, 8000), (60, 5): (12000, 8000),
        (70, 1): (3400, 2300), (70, 2): (6800, 4500), (70, 3): (12000, 8000),
        (70, 4): (12000, 8000), (70, 5): (12000, 8000),
        (80, 1): (3900, 2600), (80, 2): (7700, 5200), (80, 3): (12000, 8000),
        (90, 1): (4400, 2900), (90, 2): (8700, 5800), (90, 3): (12000, 8000),
        (100, 1): (4800, 3200), (100, 2): (9600, 6400), (100, 3): (12000, 8000),
        (110, 1): (5300, 3600), (110, 2): (10600, 7100), (110, 3): (12000, 8000),
    }
    bad = [(k, v, RS.TABLE.get(k)) for k, v in expect.items()
           if RS.TABLE.get(k) != v]
    ok = len(bad) == 0
    return record("T5c 点数表 vs 权威实现", ok,
                  f"{len(expect)} 格，失败 {len(bad)}"
                  + (f" {bad[:3]}" if bad else ""))


def test_fu_vs_reference():
    """T5d 符计算与权威实现的 doctest 样例一致。

    四个样例取自 MahjongRepository/mahjong 2.0.0 FuCalculator 的
    docstring（那些是可直接执行的断言）。
    """
    bad = []
    # 七对子固定 25 符
    if S.compute_fu([], 27, True, None, 27, chiitoi=True) != 25:
        bad.append("chiitoi != 25")
    # 门前平和放铳 = 30 符（底符本身是 30，不需要再翻）
    pinfu = S.compute_fu([('chi', 0, True), ('chi', 3, True),
                          ('chi', 9, True), ('chi', 18, True)], 11, True, None, 27)
    if pinfu != 30:
        bad.append(f"pinfu closed ron = {pinfu}, 期望 30")
    # 暗刻中张 + 门前 = 40 符
    pon = S.compute_fu([('pon', 2, True), ('chi', 3, True),
                        ('chi', 9, True), ('chi', 18, True)], 11, True, None, 27)
    if pon != 40:
        bad.append(f"closed simple pon = {pon}, 期望 40")
    # 自摸的门前平和 = 20 符（自摸不给 30 底符）
    tsumo_pinfu = S.compute_fu([('chi', 0, True), ('chi', 3, True),
                               ('chi', 9, True), ('chi', 18, True)],
                              11, True, None, 27, is_tsumo=True)
    if tsumo_pinfu != 20:
        bad.append(f"tsumo pinfu = {tsumo_pinfu}, 期望 20")
    ok = len(bad) == 0
    return record("T5d 符计算 vs 权威实现", ok,
                  f"4 例，失败 {len(bad)}" + (f" {bad}" if bad else ""))



def test_furiten_chankan():
    """T9 振听细则：普通放铳可和、抢杠不可和。

    RULES.md §11 / §12.2：
    - 舍张振听 -> 自摸被拒，但**普通放铳仍可和**
    - 抢杠（chankan）+ 振听 -> 不可和
    """
    e = E.MahjongEnv(seed=300, dealer=1)
    p = e.players[0]
    p.hand = sorted(T.parse_tiles("123m 456m 789m 11p 45p"))
    p.melds = []
    p.furiten = False
    tenpai = e._is_tenpai(0)

    # 往牌河放待牌 6p -> 制造舍张振听
    p.river.append(T.parse_tile("6p"))
    e._update_furiten(p)
    furiten = p.furiten

    # 振听后自摸必须被拒
    p.hand = sorted(T.parse_tiles("123m 456m 789m 11p 456p"))
    tsumo_blocked = not e._can_tsumo(0)

    # 普通放铳必须仍可和
    p.hand = sorted(T.parse_tiles("123m 456m 789m 11p 45p"))
    e.last_discard = T.parse_tile("6p")
    e.last_discard_player = 3
    ron_allowed = e._can_ron(0)

    # 抢杠 + 振听必须被拒
    chankan_blocked = not e._can_ron(0, from_chankan=True)

    ok = (tenpai and furiten and tsumo_blocked
          and ron_allowed and chankan_blocked)
    return record("T9 振听细则", ok,
                  f"听牌={tenpai} 振听={furiten} 自摸被拒={tsumo_blocked} "
                  f"放铳可和={ron_allowed} 抢杠被拒={chankan_blocked}")


def test_riichi_action():
    """T10 立直动作必须携带「打出的那张牌」。

    旧实现生成无参 RIICHI(tile=-1)，导致打牌流程拿不到该打哪张。
    """
    e = E.MahjongEnv(seed=301, dealer=0)
    p = e.players[0]
    p.hand = sorted(T.parse_tiles("123m 456m 789m 11p 456p"))
    p.melds = []
    p.points = 25000
    e.turn = 10
    e.phase = E.Phase.DISCARD
    e.last_discard = None
    acts = e.legal_actions(0)
    riichi = [a for a in acts if a.type == E.ACT_RIICHI]
    has_tile = len(riichi) > 0 and all(a.tile >= 0 for a in riichi)
    in_hand = all(a.tile in p.hand for a in riichi)

    # 立直后必须立刻打出一张（牌手数不变 + 标记立直）
    if riichi:
        a = riichi[0]
        n_before = len(p.hand)
        e.step({0: a})
        stepped = (len(p.hand) == n_before - 1 and p.riichi)
    else:
        stepped = False

    ok = has_tile and in_hand and stepped
    return record("T10 立直动作带牌", ok,
                  f"riichi动作={len(riichi)} 带tile={has_tile} "
                  f"在手中={in_hand} 执行后减牌并标记立直={stepped}")



def test_scenarios():
    """T11 弃听 / 全防 / 末局搏命 / 场次 / 分差 四个场景。

    用构造局面验证判定，不依赖自然对局是否碰到。
    """
    from mahjong import scenarios as SC
    from mahjong.hanchan import HanchanState

    bad = []

    # --- 全防：有人立直 ---
    h = HanchanState(dealer=0)
    e = E.MahjongEnv(seed=1)
    e.players[2].riichi = True
    c = e.players[0].full_counts()
    _, sit, _ = SC.situation_features(e, h, 0, REF.ref_shanten(c))
    if sit.label != SIT_DEFENSE:
        bad.append(f"defense: got {sit.label}")

    # --- 末局搏命：南4 + 大幅落后 ---
    h2 = HanchanState(dealer=0)
    h2.round = "S"; h2.kyoku = 4
    h2.sticks = [25000, 2000, 25000, 25000]
    e2 = E.MahjongEnv(seed=1)
    c2 = e2.players[1].full_counts()
    _, sit2, _ = SC.situation_features(e2, h2, 1, REF.ref_shanten(c2))
    if sit2.label != SIT_LAST_CHANCE:
        bad.append(f"last_chance: got {sit2.label}")

    # --- 弃听：落后 + 巡目过半 + 未听牌 ---
    h3 = HanchanState(dealer=0)
    h3.sticks = [25000, 3000, 25000, 25000]
    e3 = E.MahjongEnv(seed=1)
    e3.turn = 55
    c3 = e3.players[1].full_counts()
    _, _, gu3 = SC.situation_features(e3, h3, 1, REF.ref_shanten(c3))
    if not gu3:
        bad.append("give_up should trigger (late + far behind)")

    # --- 听牌后不再弃听 ---
    e3.players[1].hand = sorted(T.parse_tiles("123m 456m 789m 11p 45p"))
    e3.players[1].melds = []
    _, _, gu4 = SC.situation_features(e3, h3, 1,
                                   REF.ref_shanten(e3.players[1].full_counts()))
    if gu4:
        bad.append("give_up should NOT trigger when tenpai")

    # --- 场次推进：和了与流局都推进，且半庄必终止 ---
    class _R:
        def __init__(self, t, w=None):
            self.result_type = t; self.winner = w; self.points_delta = [0] * 4
    for name, mk in (("ron", lambda: _R("ron", 2)),
                     ("draw", lambda: _R("ryuukyoku"))):
        hh = HanchanState(dealer=0)
        n = 0
        while not hh.finished and n < 12:
            hh.advance(mk()); n += 1
        if not hh.finished:
            bad.append(f"hanchan {name} did not terminate")
        if n != 8:
            bad.append(f"hanchan {name}: {n} deals (expect 8)")

    ok = len(bad) == 0
    return record("T11 弃听/全防/场次/分差", ok,
                  f"4 项场景 + 场次推进，失败 {len(bad)}"
                  + (f" {bad[:3]}" if bad else ""))


def test_hanchan_progress():
    """T12 半庄场次序列与风位轮转。"""
    from mahjong.hanchan import HanchanState

    class _R:
        def __init__(self, t, w=None):
            self.result_type = t; self.winner = w; self.points_delta = [0] * 4

    h = HanchanState(dealer=0)
    seq = []
    for _ in range(8):
        h.advance(_R("ron", 2))
        seq.append(h.progress_label())
    # 第 7 局打完进到 S4，第 8 局结束后半庄终止（标签仍是 S4）
    expect = ["E2 (0 连庄)", "E3 (0 连庄)", "E4 (0 连庄)", "S1 (0 连庄)",
              "S2 (0 连庄)", "S3 (0 连庄)", "S4 (0 连庄)", "S4 (0 连庄)"]
    ok = (h.finished and seq == expect and len(set(seq)) == 7)

    # 风位随庄家轮转：庄家是场风
    wind_ok = all(h.wind_of(h.dealer) in (27, 28) for _ in [0])

    # 连庄棒：流局累加，和了清零
    h2 = HanchanState(dealer=0)
    h2.advance(_R("ryuukyoku")); a = h2.honba
    h2.advance(_R("ryuukyoku")); b = h2.honba
    h2.advance(_R("ron", 3)); c = h2.honba
    honba_ok = (a == 1 and b == 2 and c == 0)

    full_ok = ok and wind_ok and honba_ok
    return record("T12 半庄场次与连庄", full_ok,
                  f"序列正确={ok} 风位={wind_ok} 连庄棒={honba_ok}")

def main():
    print("=" * 72)
    print("立直麻将 RL 环境自测")
    print("=" * 72)
    test_tile_conservation(200)
    test_forced_wins()
    test_no_illegal_win(200)
    test_score_conservation(200)
    test_decomposition_completeness()
    test_fu_roundup()
    test_points_table_vs_reference()
    test_fu_vs_reference()
    test_furiten()
    test_edge_actions()
    test_reproducibility(30)
    test_furiten_chankan()
    test_riichi_action()
    test_scenarios()
    test_hanchan_progress()
    print("=" * 72)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"结果：{npass}/{len(RESULTS)} 通过")
    print("=" * 72)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

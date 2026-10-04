"""牌的基础表示：34 种牌的编码、类别判定、赤牌管理。

对应规则文档：riichi-rl/rules/RULES.md §1

编码约定（与 RULES.md §1.1 一致）:
    0-8   万 1-9
    9-17  筒 1-9
    18-26 索 1-9
    27-30 东 南 西 北
    31-33 白 发 中
"""

from __future__ import annotations

# --- 基本分类 ---

SUIT_START = (0, 9, 18)
SUIT_END = (9, 18, 27)
EAST, SOUTH, WEST, NORTH = 27, 28, 29, 30
HAKU, HATSU, CHUN = 31, 32, 33

NUM_TILES = 34
NUM_COPIES = 4
NUM_TOTAL_TILES = NUM_TILES * NUM_COPIES  # 136

WIND_TILES = (EAST, SOUTH, WEST, NORTH)
DRAGON_TILES = (HAKU, HATSU, CHUN)
HONOR_TILES = WIND_TILES + DRAGON_TILES
YAO9_TILES = (0, 8, 9, 17, 18, 26)  # 1m9m 1p9p 1s9s

#: 十三幺所需�� 13 种幺九牌
TERMINALS_AND_HONORS = YAO9_TILES + HONOR_TILES

#: 番牌宝牌候选（RULES.md §1.2，本规范 1m 3p 5s）
AKA_TILES = (0, 11, 22)  # 1m, 3p, 5s

#: 名字表，供可读日志使用
_TILE_NAMES = (
    [f"{i}m" for i in range(1, 10)]
    + [f"{i}p" for i in range(1, 10)]
    + [f"{i}s" for i in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
)
TILE_NAMES = tuple(_TILE_NAMES)

_SUIT_CHAR = ("m", "p", "s", "z")


def suit_of(tile: int) -> int:
    """0=万 1=筒 2=索 3=字牌"""
    if tile < 9:
        return 0
    if tile < 18:
        return 1
    if tile < 27:
        return 2
    return 3


def rank_of(tile: int) -> int:
    """数牌的 1-9；字牌返回 1-7（东=1 南=2 西=3 北=4 白=5 发=6 中=7）"""
    if tile < 27:
        return tile % 9 + 1
    return tile - 26


def is_honor(tile: int) -> bool:
    return tile >= 27


def is_yao9(tile: int) -> bool:
    """幺九牌：数牌 1/9 或字牌"""
    if is_honor(tile):
        return True
    r = tile % 9 + 1
    return r == 1 or r == 9


def is_chunchan(tile: int) -> bool:
    """中张：2-8 的数牌（字牌不是中张）"""
    return (not is_honor(tile)) and 2 <= tile % 9 + 1 <= 8


def is_terminal(tile: int) -> bool:
    """老头牌：数牌 1/9"""
    return (not is_honor(tile)) and tile % 9 in (0, 8)


def is_simple(tile: int) -> bool:
    return not is_yao9(tile)


def tile_name(tile: int) -> str:
    return TILE_NAMES[tile]


def tiles_name(tiles) -> str:
    return "".join(TILE_NAMES[t] for t in sorted(tiles))


def parse_tile(s: str) -> int:
    """从 '3m' / 'E' 这样的字符串解析 tile id。用于测试构造牌型。"""
    s = s.strip()
    if not s:
        raise ValueError("empty tile spec")
    if s[-1].lower() in _SUIT_CHAR:
        num, suit = s[:-1], s[-1].lower()
    else:
        num, suit = s, "z"
    if s.upper() in ("E", "S", "W", "N", "P", "F", "C"):
        return {"E": EAST, "S": SOUTH, "W": WEST, "N": NORTH,
                "P": HAKU, "F": HATSU, "C": CHUN}[s.upper()]
    if suit == "z":
        if not num.isdigit():
            raise ValueError(f"bad tile spec {s!r}")
        idx = int(num)
        if not 1 <= idx <= 7:
            raise ValueError(f"bad honor index in {s!r}")
        return 26 + idx
    if not num.isdigit():
        raise ValueError(f"bad tile spec {s!r}")
    rank = int(num)
    if not 1 <= rank <= 9:
        raise ValueError(f"bad rank in {s!r}")
    return _SUIT_CHAR.index(suit) * 9 + (rank - 1)


def parse_tiles(s: str):
    """解析牌串，例如 '123m 456p 99s EE' -> [0,1,2,11,12,13,26,27]

    逐字符扫描：'数字串 + 花色字母' 展开成连续牌；单个字牌字母直接成牌。
    支持 '19m19p19s' 这种连写，以及 'z1z2z3' 这种字牌简写。
    """
    s = s.replace(",", " ").replace(" ", "")
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            digits = s[i:j]
            if j >= n or s[j].lower() not in _SUIT_CHAR:
                raise ValueError(f"digits {digits!r} at {i} need a suit letter")
            suit = s[j].lower()
            base = _SUIT_CHAR.index(suit) * 9
            for d in digits:
                rank = int(d)
                if not 1 <= rank <= 9:
                    raise ValueError(f"bad rank {rank} in {s!r}")
                out.append(base + rank - 1)
            i = j + 1
        elif ch.upper() in ("E", "S", "W", "N", "P", "F", "C"):
            out.append(parse_tile(ch))
            i += 1
        else:
            raise ValueError(f"unexpected char {ch!r} in {s!r}")
    return out


def dora_from_indicator(indicator: int) -> int:
    """宝牌指示牌 -> 宝牌（RULES.md §7.5）：下一个数牌种，9 的下一张是 1"""
    return (indicator + 1) % NUM_TILES


def next_tile(tile: int) -> int:
    """数牌的下一张（1->2 ... 8->9, 9->1），字牌不变。用于「同巡下一张」等判断。"""
    if is_honor(tile):
        return tile
    return tile % 9 + 1 if tile % 9 < 8 else tile - 8


def is_terminal_or_honor(tile: int) -> bool:
    return is_yao9(tile)


# --- 计数向量工具 ---
# 手牌内部用 34 维 list 表示，长度固定 34，元素 0..4。
# 这样做的好处：算番、去重、和了判定都是纯局部运算，不需要反复排序。


def counts_to_list(counts) -> list:
    """把任意可迭代的 tile id 序列转成 34 维计数向量。"""
    c = [0] * NUM_TILES
    for t in counts:
        c[t] += 1
    return c


def list_to_tiles(c: list) -> list:
    out = []
    for t in range(NUM_TILES):
        out.extend([t] * c[t])
    return out


def counts_total(c: list) -> int:
    return sum(c)


def counts_str(c: list) -> str:
    parts = []
    for s in range(4):
        if s < 3:
            letters = ""
            for r in range(9):
                if c[s * 9 + r]:
                    letters += str(r + 1) * c[s * 9 + r]
            if letters:
                parts.append(letters + "mpsz"[s])
        else:
            letters = "".join(TILE_NAMES[t] * c[t] for t in range(27, 34) if c[t])
            if letters:
                parts.append(letters)
    return " ".join(parts)


def empty_counts() -> list:
    return [0] * NUM_TILES

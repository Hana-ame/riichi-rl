"""把 make_game_log.py 导出的对局 JSON 渲染成自包含 HTML 牌谱游览器。

精简版：只存「本步动作事件」+ 初始状态，浏览器增量重放。HTML ~20KB。

座位/风位：log["initial"]["players"] 已按玩家索引 0,1,2,3 排列（players[i] 即索引 i），
每家的 seat_wind（27=东 28=南 29=西 30=北）就在该元素里，庄家标记 dealer 同理。
events[k].seat 是玩家索引 0-3，与 players[i].idx=i 一一对应。

用法：
    python render_game_html.py --log rules/game_real_7101.json --out rules/game_real_7101.html
"""
from __future__ import annotations
import argparse, json, os

# 牌名 -> 排序键。真值来自 mahjong/tiles.py：
#   0-8=1m..9m, 9-17=1p..9p, 18-26=1s..9s, 27-33=E S W N P F C
# 字符串排序是错的（"1m"<"1p"<"1s"<"2m" 会把花色打散），必须按真值索引排。
_NAME_ORDER = ([f"{i}m" for i in range(1, 10)]
               + [f"{i}p" for i in range(1, 10)]
               + [f"{i}s" for i in range(1, 10)]
               + ["E", "S", "W", "N", "P", "F", "C"])
NAME_KEY = {n: k for k, n in enumerate(_NAME_ORDER)}
def hand_key(name):
    return NAME_KEY.get(name, 99)


def compact(log):
    """只保留渲染必需字段。手牌按花色+点数真值排序，保留玩家索引顺序。"""
    init = log["initial"]
    players0 = []
    for i, p in enumerate(init["players"]):   # 数组顺序即玩家索引 0-3
        players0.append({
            "idx": i,
            "wind": p.get("seat"),                # seat_wind 27-30，定东/南/西/北
            "hand": sorted(p.get("hand", []), key=hand_key),
            "river": [],
            "melds": [],
            "points": p.get("points", 25000),
            "riichi": False,
            "dealer": p.get("dealer", False),
        })
    events = []
    for st in log["steps"]:
        for e in st["events"]:
            events.append({"seat": e["seat"], "kind": e["kind"], "tile": e.get("tile")})
    return {
        "meta": log["meta"],
        "result": log.get("result"),
        "dora0": init.get("dora", []),
        "wall0": init.get("wall"),
        "players0": players0,
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = json.load(open(a.log, encoding="utf-8"))
    data = compact(log)
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = open(os.path.join(here, "game_viewer_template.html"), encoding="utf-8").read()
    out = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    open(a.out, "w", encoding="utf-8").write(out)
    print(f"wrote {a.out}  events={len(data['events'])}  {len(out)/1024:.1f}KB")


if __name__ == "__main__":
    main()

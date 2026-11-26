"""把 make_game_log.py 导出的对局 JSON 渲染成自包含 HTML 牌谱游览器。

精简版：不内嵌每步全量快照（那样 54 步就有 ~50KB），而是只存「本步动作事件」，
由浏览器从初始快照增量重放。HTML 体积从 62K 降到 ~8K。

用法：
    python render_game_html.py --log rules/game_real_7101.json --out rules/game_real_7101.html
"""
from __future__ import annotations
import argparse, json, os

SEAT_ORDER = [27, 28, 29, 30]          # mahjong.tiles: 东27 南28 西29 北30
ACT = {0:"DISCARD",1:"TSUMO",2:"RIICHI",3:"ANKAN",4:"MINKAN",
       5:"CHI",6:"PON",7:"ADD_KAN",8:"RON",9:"RYUUKYOKU",11:"PASS"}
CALL = {"CHI","PON","ANKAN","MINKAN","ADD_KAN"}


def compact(log):
    """只保留渲染必需的字段，把每步压成轻量事件。"""
    init = log["initial"]
    players0 = []
    for s in SEAT_ORDER:
        p = next((pp for pp in init["players"] if pp["seat"] == s), {})
        players0.append({
            "seat": s,
            "hand": sorted(p.get("hand", [])),
            "river": [],
            "melds": [],
            "points": p.get("points", 25000),
            "riichi": False,
            "dealer": p.get("dealer", False),
        })
    events = []
    for st in log["steps"]:
        for e in st["events"]:
            events.append({
                "seat": e["seat"],
                "kind": e["kind"],
                "tile": e.get("tile"),
            })
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

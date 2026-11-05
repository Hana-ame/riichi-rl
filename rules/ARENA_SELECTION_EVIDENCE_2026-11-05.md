# 证据：arena_elo 的「较好一半 + 随机替换」确实在跑（2026-11-05）

用户/审阅反复问「对战模型=表现较好一半+随机几个替换是否实现」。这是**实跑日志**
（`arena_elo.py --games 200 --warmup-games 50 --reselect-every 50 --random-fill 2
--seed 1`，日志 `/tmp/arena_sel.log`）：

```
[arena] 30s 选员：保留 Elo 前 7 名 + 随机 2 个替换
   -> greedy_tenpai, damaten, riichi_needed, riichi_ev, yaochu_first,
      a2c_r10, dagger, hc, teacher47_v3
[arena] 56s 选员：保留 Elo 前 7 名 + 随机 2 个替换
   -> greedy_tenpai, damaten, riichi_needed, riichi_ev, call_first,
      yaochu_first, from_games_v2, dagger, bc_v1
[arena] 79s 选员：保留 Elo 前 7 名 + 随机 2 个替换
   -> greedy_tenpai, riichi_needed, riichi_ev, call_first, yaochu_first,
      riichi_net, a2c_r10, dagger, bc_v1
```

- 参数：`--warmup-games 50` 先热身；之后每 `--reselect-every 50` 局重选；
  `--random-fill 2` = 保留 Elo 前一半（7 名，共 14–15 参赛）+ 随机 2 个替换；
- 三次选员名单不同（daggers/a2c/hc/riichi_net/from_games 进出），说明机制生效；
- 参赛局数差异也印证：常驻者 100+ 局，被换下的明显少。
- 本次 200 局只作机制证据；**正式 ELO 记录仍以 600 局的 `rules/ELO.md` 为准**
  （已恢复，未被本次实验覆盖）。

## 用法（正式跑）

```bash
$PY arena_elo.py --games 600 --warmup-games 100 --reselect-every 100 \
    --random-fill 3 --seed 0
```

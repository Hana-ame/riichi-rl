# 评估体系现状：ELO / 选员 / 东风-半庄 / 副露-dora 排期（2026-11-05）

> 回应审阅「条件不足，请主会话补充」的 4 项：① ELO 实现/结果；② 「对战模型=表现较好
> 一半+随机替换」是否已做；③ 东风/半庄分别评估数据；④ 副露/杠/dora 排期。
> 以及纠正审阅 9/10/11 项「未做/证据不足」的误判。

## ① ELO 评估：已实现且有 600 局结果

- 实现：[arena_elo.py](../arena_elo.py)——4 人随机桌、逐对零和 Elo
  （K=24）、额外任务奖励计入（`--aux-weight`）、放铳率列、选员机制（见 ②）。
- 最近一次 600 局结果（[rules/ELO.md](ELO.md)，15 参赛）：
  from_games_v2 1692.8、bc_v1 1666.3、damaten 1651.3、greedy 1597.2、riichi_needed
  1581.4、random 1536.7、hc 1496.8、call_first 1488.5、from_games_v1 1450.5、
  a2c_r10 1449.4、teacher47_v3 1435.8、riichi_ev 1402.5、dagger 1384.5、
  riichi_net 1359.9、yaochu 1306.4（15 人 Elo 总和 22500 = 1500×15，零和）。
- 每人局数 29–230、放铳率/和牌率/场均/任务奖励均在表内。

## ② 选员「表现较好一半 + 随机替换」：已实现并实际生效

代码：`arena_elo.py`
```
--warmup-games 100      # 全体热身
--reselect-every 100    # 每 100 局重新选员
--random-fill 3         # 保留 Elo 前一半（7 名）+ 从落选者随机换 3 个
```
ELO.md 头部保存最近一次选员名单（原文）：
```
> 选员：先全体热身，之后每轮保留 Elo 前一半 + 随机几个替换…
> 最后一次选员：186s 选员：保留 Elo 前 7 名 + 随机 3 个替换 ->
  random, greedy_tenpai, damaten, riichi_needed, riichi_ev, call_first,
  riichi_net, from_games_v2, hc, bc_v1
```
**生效证据**：被换下的参赛者局数明显少（dagger 29 局、from_games_v1 67 局、
random 103 局），而常驻者 219–230 局——局数差异就是选员机制留下的痕迹。

## ③ 东风 / 半庄分别评估：已支持且刚跑出双模式数据

- 实现：`mahjong/hanchan.py::HanchanState(east_only=True)` 东 4 结束即终局；
  `eval_hanchan.py --mode {hanchan,east}`。
- 实测（同一模型 teacher47_v3，各 3 场）：

| 模式 | 局数 | uniform-random | shanten-greedy | teacher47_v3 |
|---|---|---|---|---|
| 半庄 E1–S4（3 半庄） | 24 | 0.0% | 37.5% | **12.5%（3/24）** |
| 东风战 E1–E4（3 场） | 12 | 0.0% | 50.0% | **16.7%（2/12）** |

> 东风战局数固定是半庄的一半（每场 4 局 vs 最多 8 局），上表就是按模式分别统计。

## ④ 副露/杠、dora 的现状与排期

### 副露/吃碰杠
- **规则策略有**：`CallFirst`（明杠 50/加杠 50/碰 40/吃 30/暗杠 5），并在 Elo 参赛
  156 局；实测 40 局动作数 CHI 61 / PON 37 / ADD_KAN 11 / MINKAN 0（yaochu 另有
  MINKAN 1 / ADD_KAN 12）。`GreedyTenpai/damaten/riichi_*` 响应只荣和/PASS。
- **机器学习策略没有**：47 槽模型虽含 PON/CHI/KAN 槽，但目前不做独立监督。
- 排期：**P2**。① 用 `call_first` 作教师生成 DAgger 数据（含吃碰杠标签）；
  ② 对特殊槽（PON/CHI/MINKAN/ADD_KAN/ANKAN）加权 BC；③ 用 `evaluate` 的放铳率/
  和牌率对比「模型叫牌 vs 规则叫牌」。

### dora
- **特征与计分有**：`obs2.py:78-103` 三个 dora 平面；`env.py` 杠后翻宝牌、
  dora/赤/里宝计番（scoring）。
- **独立策略没有**：神经网络从特征自学；规则策略不读 dora。
- 排期：**P3**。① 在 `aux_tasks` 加 `dora` 任务（和牌且 dora/赤/里宝≥1 番），
  用 `--aux-weights` 调；② 与 `tanyao/pinfu` 风格权重同机制，训练一版对照；
  ③ 评测「dora 和牌占比 + 场均」。

## 审阅 9/10/11 项纠正

| 审阅说法 | 实际 |
|---|---|
| 9 ELO「证据不足/未做」 | ❌ 误判：已实现并有 600 局结果（见 ①），`tests/test_arena.py` 锁零和 |
| 10 「较好一半+随机替换」未做 | ❌ 误判：已实现且实际生效（见 ②，局数差异可证） |
| 11 东风/半庄未分别评估 | ❌ 误判：`--mode` 已支持，本轮跑出双模式表（见 ③） |

## 仍未完成的（与审阅一致）

1. 牌河**顺序**→形状/向听：仍无序列特征（只有对手持牌概率近似）。
2. 和牌率进 15–25%：目前只有 teacher47_v3 强桌 24.2% 在区间，hybrid/hybridd 7–10%。
3. 放铳率进 9–20%：bc 10% 在区间；teacher47 5%、from_games 25%。
4. 独立副露/杠、dora 学习策略（排期见 ④）。
5. 危险张「降放铳」统计显著：500 局 INCONCLUSIVE，升级版 hybridw 失败。
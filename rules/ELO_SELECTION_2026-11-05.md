# Elo 选员 + 放铳率列 + 四个待确认项的答复（2026-11-05）

> 回应审阅的 ①~④。对应改动：`arena_elo.py` 加「表现较好一半 + 随机替换」选员、
> ELO.md 加**放铳率**列、把 `teacher47_v3` 加权 BC 模型放进竞技场；
> 600 局重跑结果见 `rules/ELO.md`。

## ① 危险张 / 和牌率：显式输出还是只做决策？

| 项 | 现状 | 证据 |
|---|---|---|
| **和牌率** | **有显式模型输出**（不是只做决策）：`hand_predictor.py` 三头 = 每局和牌概率（sigmoid）/ 若和了的预期点数 / 还几巡听牌；已训出 `checkpoints/hand_predictor.pt`（51,666 样本），并接入观战 `serve_game.hand_predictions()`（四家 win/pts/turns 每帧输出）。 | `hand_predictor.py`；`serve_game.py:get_hand_model/hand_predictions`；`rules/TEACHER_BC` 同批证据 |
| **危险张** | **没有显式输出**（只有统计与局面级防守）：`arena.py` 统计放铳率；`scenarios.py` 有放铳惩罚/弃听；`situation.py` 有 `SIT_DEFENSE`；`opponent_model.py` 可从牌河/副露推断对手手牌（原料，但未组装成单张危险度）。且 `RULES.md §15` 明文禁止引擎/策略里手写「安全牌/危险牌」。 | `rules/RULES.md §15`；`mahjong/scenarios.py`；`mahjong/situation.py` |
| 结论 | 若要求「每张牌危险度」显式输出，需**训练**一个危险度头（用放铳结果做标签，不违反 §15）；当前不输出。 |

## ② Elo 表补放铳率列（已完成）

`render_markdown` 增加「放铳率」列（`deal_in_rate`，每参赛者放铳局数/参赛局数）；
`ELO.json` 同步存字段。600 局读数（节选，全表见 `rules/ELO.md`）：

| 排名 | 参赛者 | Elo | 和牌率 | **放铳率** | 平均点数 | 任务奖励/局 |
|---|---|---|---|---|---|---|
| 1 | from_games_v2(蒸馏) | 1692.8 | 0.4% | **3.5%** | −179 | +250 |
| 2 | bc_v1(人类牌谱教师) | 1666.3 | 16.4% | **2.7%** | +1244 | +561 |
| 3 | damaten(默听) | 1651.3 | 9.9% | **3.1%** | +643 | +370 |
| 4 | greedy_tenpai(立直优先) | 1597.2 | 12.2% | **1.8%** | +787 | +540 |
| 5 | riichi_needed | 1581.4 | 13.0% | **2.2%** | +925 | +459 |
| 8 | call_first | 1488.5 | 0.0% | **10.9%** | −1111 | +196 |
| 11 | teacher47_v3(加权BC) | 1435.8 | 5.1% | **5.1%** | −179 | +328 |
| 15 | yaochu_first | 1306.4 | 0.0% | **7.7%** | −1447 | +199 |

> 注：当前引擎/对手池下，除 `call_first`（10.9%）外放铳率普遍低于「正常 9–20」；
> 原因是荒牌率仍高（68%）、荣和占和牌 84%，放铳机会本来就少。这一列先作为
> 可核对读数存在，不能当作「防守好」的证据。

## ③ Elo 选员：表现较好一半 + 随机几个替换（已完成）

按你的原始要求实现（`arena_elo.py`）：

- `--warmup-games 100`：先全体参赛热身 100 局，拿到初始 Elo；
- 之后每 `--reselect-every 100` 局重新选员：**保留 Elo 前一半（7 名）+
  从落选者里随机换 3 个**进参赛名单（`--random-fill 3`），其余冻结；
- 每局从当前参赛名单里随机抽 4 人。

600 局实际选员记录（ELO.md 头部保存最后一次；日志含全部四次）：

```
41s  保留前 7 + 随机 3 -> greedy/damaten/riichi_needed/riichi_ev/call_first/
     yaochu/from_games_v1/from_games_v2/teacher47_v3/bc_v1
77s  保留前 7 + 随机 3 -> … yaochu/riichi_net/from_games_v2/hc/…
118s 保留前 7 + 随机 3 -> … riichi_net/from_games_v2/a2c/hc/…
186s 保留前 7 + 随机 3 -> random/greedy/damaten/riichi_needed/riichi_ev/
     call_first/riichi_net/from_games_v2/hc/bc_v1
```

效果：`from_games_v2` 被反复保留到最后、`riichi_net` 中段被换下又被随机送回；
`dagger/from_games_v1/a2c` 等弱模型后段基本被换下（参赛局数只剩 29/67/120），
Elo 的局数不齐（表内已列局数，判读时注意）。

**连带发现**：选员后 Elo 第一是 `from_games_v2`（0.4% 和牌率、平均点 −179）——
它靠「不输」（放铳率 3.5%）苟名次，属于「和牌率极低但排名虚高」的典型；
`bc_v1`（16.4% 和牌率）Elo 第二、平均点 +1244。再次说明纯模型的**和牌率**
仍是最大瓶颈，选员规则只影响名次不代表变强。

## ④ 数据采集 / 数据添加管道现状（可工作，已实测）

| 数据 | 路径/体积 | 来源 → 用途 | 脚本 |
|---|---|---|---|
| 天鳳鳳凰室人类牌谱 | `data/houou/` 528M | 官方训练集下载（socks5）→ `data/cache/` 585M npz | `data/dl_houou.sh`、`build_cache.py`、`data/build_all.sh` |
| BC 教师（人类牌谱） | `checkpoints/bc_v1_last.pt` | cache npz 训练 → `bc_v1` | `train_bc.py` |
| 对局自采（selfplay） | `data/selfplay/` 11M，703 局 | `serve_game` 观战逐决策记录 `rec_step/flush_rec`（自动续号 `_next_game_seq`） | `serve_game.py` |
| 蒸馏模型 | `from_games_v1/v2.pt` | selfplay 703 局 × 75,982 决策 → 监督蒸馏 | `train_from_games.py --pol all` |
| 规则数据生成 | `data/bc2/` 1.4M、`gen_greedy_data.py` | 规则策略自对弈批量落 npz（可继续加） | `gen_greedy_data.py --games N --out-dir data/selfplay` |
| 教师 BC 数据（固定槽位） | `data/teacher47.npz` 4.7M，62,610 样本 | GreedyTenpai×4 30 半庄，带 47 槽 mask/标签 | `gen_teacher_data.py` |
| 增量「数据添加」 | 同上 | 观战每局自动追加、生成脚本可续号；`data/selfplay` 已 703 局 | `serve_game` 常驻时自动攒 |

**缺口**：没有「单张危险度」标签数据集（要放铳/被荣和结果反推）；蒸馏数据只来自
规则/模型的 54 槽记录（固定槽位版本的数据管道是本轮新加的 teacher47 线）。

## 复现

```bash
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
$PY -m tests.run_all                  # 80/80
$PY arena_elo.py --games 600 --seed 0 --aux-weight 1.0 \
    --warmup-games 100 --reselect-every 100 --random-fill 3
#   -> rules/ELO.md / rules/ELO.json（含放铳率列、任务奖励列、选员记录）
```
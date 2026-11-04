# 立直 / 默听策略对比：谁的点数最大（2026-11-05）

> 用户要求：「有不同的立直策略，默听策略，要求分最大」
> 做法：实现 4 套立直决策策略（立直优先 / 默听 / 没役才立直 / 按点数期望），
> 在同台对局 + Elo 竞技场里用**平均点数**和 **Elo** 双口径实测，不靠拍脑袋。
> 结论：**点估计上「立直优先」平均分最高**（+264/局），但四种策略两两 95% CI
> 全部跨 0（1600 局仍无法区分）；「默听」并没有比立直更赚。

## 1. 四套策略（`mahjong/policies.py`）

| 策略名 | 行为 |
|---|---|
| `greedy_tenpai` | 能立直就立直（旧默认；立直保底有役 + 加番 + 里宝机会） |
| `damaten` | **默听**：能立直也不宣告，保留门清、不押 1000 立直棒、不暴露 |
| `riichi_needed` | 只在「不立直就没役」时立直；有役时默听等荣和 |
| `riichi_ev` | 按点数期望：默听有役且已达满贯（≥8000）→ 默听，否则立直 |

默听判定都用**引擎算分器** `_best_score`（`和了这手算多少点/有没有役`），
不写手写牌效表（遵守 `RULES.md §15` 与 `policies.py` 铁律）。

## 2. 前置：先修 3 个计分 bug（否则「分最大」比较没有意义）

| # | 位置 | 问题 | 修正 |
|---|---|---|---|
| 1 | `ruleset.py` | `AKA_TILES=(0,11,22)` 注释写「1m 3p 5s」——非标准且 id 错（0 是 1m，11 是 3p） | 改 `(4,13,22)` = **5m 5p 5s**（标准赤牌），同步 `RULES.md §1.2/§14/§17-A2` |
| 2 | `env.py::_aka_count` | 把**四家手牌**赤牌一起数再 `min(...,3)` → 每家和牌白送 **+3 番**（任何 1 番牌变满贯） | 改 `_aka_count(p_idx)`：只数和牌者，每种赤牌最多 1 张 |
| 3 | `scoring.py::_detect_yakus` | `is_menzen` 就加「门前清」——门清荣和也白送 1 番（门前清不是独立役，只有**门前清自摸**才成役） | 加 `and is_tsumo` 条件 |

修复后 `tests/run_all` 里 3 个依赖旧假役的用例被修正（如「门清荣和 + 东单骑」
其实无役），全绿：**74/74**（新增赤牌/门前清计分回归 + 5 个立直策略用例）。

## 3. 实测一：同台 4 策略、1600 局（`arena.evaluate`，批次为独立单元）

复现：`arena.evaluate(["greedy_tenpai","damaten","riichi_needed","riichi_ev"], n_batches=400, seed0=20261105)`

| 策略 | 平均点差 | 和了率 | 放铳率 | 立直率 | 平均和了点 |
|---|---|---|---|---|---|
| **greedy_tenpai（立直优先）** | **+264** | 9.7% | 6.7% | 38.6% | 9476 |
| riichi_ev（按点数期望） | +182 | 8.9% | 6.3% | 34.8% | 8830 |
| riichi_needed（没役才立直） | −13 | 9.6% | 5.9% | 20.3% | 7301 |
| damaten（默听） | −68 | 7.6% | 6.9% | 0% | 8410 |

两两 95% CI（批次独立，n=400）**全部跨 0**，最接近显著的一对是
`greedy_tenpai vs damaten = +332 [−53, +717]`——方向支持立直，但仍不能下结论。

## 4. 实测二：600 局 Elo 竞技场（14 参赛，含模型）

`arena_elo.py --games 600 --seed 0`（每参赛 158–188 局；Elo 总和 21000 零和）：

| 策略 | Elo 排名 | Elo | 和牌率 | 平均点数 |
|---|---|---|---|---|
| greedy_tenpai(立直优先) | **1** | **1625.2** | 11.0% | +619 |
| damaten(默听) | 2 | 1582.7 | 12.5% | **+911** |
| riichi_needed(没役才立直) | 4 | 1573.2 | 11.3% | +761 |
| riichi_ev(按点数期望) | 8 | 1482.2 | 12.5% | +1076 |

注：Elo 按**名次**更新，平均点数按**净点数**统计，两者会不一致——
`riichi_ev` 平均点最高（+1076，方差也大/第 4 名更多）、Elo 却最低；
`greedy_tenpai` Elo 第一、平均点 +619 低于默听。600 局每人只有 ~170 局，
单人平均点标准误很大。

## 5. 结论（按用户目标「分最大」）

1. **点估计最优：`greedy_tenpai`（能立直就立直）** —— 1600 局同台 +264 最高；
   Elo 也第一。默听（−68）并不比立直赚，反而少了立直番/里宝/棒收入。
2. **`riichi_ev`（按点数选）** 同台第二（+182），在 600 局 Elo 里平均点最高
   但 Elo 最低 —— 属于高波动策略，不适合当“稳定最大化”。
3. **`riichi_needed`/`damaten` 都没有超过立直优先**；默听唯一优势是放铳率
   不低（反而 6.9% 最高），因为没有立直威慑、对手照常进攻。
4. 差异都不显著（CI 跨 0），要下定论需要 ≥5000 局或更强对手池。
   **当前默认保持「立直优先」**，默听/按点数策略作为可选项保留。

## 6. 怎么用

观战/对局里每席可随时切换策略（下拉框和 CLI 都加了）：

```bash
$PY serve_game.py --seed 1000 --opp damaten --p0 greedy_tenpai
# 浏览器打开 http://localhost:8776/mahjong_view.html，界面下拉可切：
#   立直优先 / 默听 / 没役才立直 / 按点数选立直-默听
```

```python
from mahjong import policies as POL
pol = POL.make_policy("riichi_ev", seed=0)   # 或 damaten / riichi_needed
```

## 7. 复现

```bash
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
$PY -m tests.run_all                    # 74/74
$PY arena_elo.py --games 600 --seed 0   # 14 参赛 Elo（约 9 min，2 线程）
$PY - <<'EOF'
import sys; sys.path.insert(0,'.')
import arena
print(arena.evaluate(["greedy_tenpai","damaten","riichi_needed","riichi_ev"],
                     n_batches=400, seed0=20261105).report())
EOF
```
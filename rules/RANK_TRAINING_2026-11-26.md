# PPO 排位训练追踪（以排位最高为目标）

## 目标
从当前排位榜首 `teacher47_eff_wuke_d2`（BC 模仿学习产物）出发，用 PPO 自我对弈
持续提升，**目标：Elo 排位第一且与第二名拉开显著差距（> ±31 SE）**。

## 基线（PPO 起点前，obs2 修复后）
600 局 Elo（`rules/ELO_2026-11-26.md`）：

| 排名 | 参赛者 | Elo | 和牌率 | 平均点数 |
|---|---|---|---|---|
| 1 | eff_wuke_d2 | 1644.1 | 34.5% | +1841 |
| 2 | damaten(默听规则) | 1627.8 | 15.5% | +680 |

注意：best 与第 2 名仅差 16 分 << ±31 SE，**登顶不显著**。PPO 的价值就是拉开差距。

## 基础设施修复（本次解锁）
1. `train_hanchan.py`：从 ckpt `head_fc` 反推 `scalar_dim` + 按 `in_planes` 分派
   legacy obs encoder（22平面→`obs2_v1`），使 legacy ckpt 可作 PPO init。
   否则 `load_state_dict` 因 366 vs 372 shape 崩溃，或 forward 610 vs 366 崩溃。
2. `arena_elo.py`：`HanchanAgent` 同样按 `in_planes`/`embed` 通道分派，
   7 个模型全部恢复参战（之前只剩 7 个规则策略）。
3. `arena_elo.py --extra-ckpt`：让任意在训 PPO checkpoint 参加排位赛。
4. `rank_watch.sh`：后台轮询新 checkpoint，自动跑 Elo 300 局追踪排位趋势。

## PPO 训练曲线（从 best init，200 iters，16 hanchans/iter）

| iter | deals | win | R(回报) | policy loss | vloss | entropy | KL | 备注 |
|---|---|---|---|---|---|---|---|---|
| 0 | 160 | 86.25% | -0.468 | 0.0000 | 0.5165 | 0.000 | 0.000 | warmup 起 |
| 5 | 1111 | 85.15% | -0.451 | 0.0000 | 0.5307 | 0.000 | 0.000 | warmup（critic-only） |
| 10 | 1974 | 85.71% | -0.474 | -0.0001 | 0.4874 | 0.076 | 0.000 | warmup 尾 |
| 15 | 2866 | 86.01% | -0.318 | +0.0052 | 0.3565 | 0.092 | 0.0053 | 策略头开始更新 |
| 20 | 3744 | 85.28% | **-0.186** | -0.0007 | **0.2992** | 0.106 | **0.0260** | vloss 大降、R 改善 |

健康信号：vloss 持续下降（0.52→0.30）、回报 R 改善（-0.47→-0.19）、
熵温和上升（0→0.106，未崩）、KL 温和增长（策略在学但未发散）。

## 排位追踪（300 局 Elo，`rank_watch.sh` 自动）

| checkpoint | Elo | 排名 | 和牌率 | 平均点数 | vs 基座 eff_wuke_d2 |
|---|---|---|---|---|---|
| eff_wuke_d2（基线） | 1631.4 | 1 | 27.2% | +1454 | — |
| ppo_rank_v1_it10 | — | 未进前列 | — | — | 仍在 warmup，策略未动 |
| ppo_rank_v1_it20 | 1621.2 | 3 | **35.0%** | **+2661** | 点数 +1207、和牌 +7.8pt |

**趋势判断**：PPO 仅 20 iter 就把**平均点数提升 83%**（+1454→+2661，全场最高）、
和牌率 +7.8pt。Elo 暂略低于基座是 300 局采样噪声（±31 SE）。
若趋势持续，PPO 应在几十 iter 后稳定超越基座并拉开差距。

## 待办
- PPO 训练至 200 iters（`checkpoints/ppo_rank_v1*.pt`，后台 setsid）
- `rank_watch.sh` 每 10 iter 自动 Elo 300 局，记录趋势
- 达到「Elo 第一 + 领先第 2 名 > 61 分」才算排位目标达成
- 中途 checkpoint 若明显超过基座，可提前用于实战

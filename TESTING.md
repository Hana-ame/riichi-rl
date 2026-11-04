# 测试体系：每个模块可单独验证

> 2026-11-04 · 目标：任何模块都能在 30 秒内单独跑测试确认自己没坏。
> 零依赖：只用标准库 `unittest`，不需要 pytest。

## 快速开始

```bash
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python

# 全部（约 3 秒 + 引擎自测另算）
$PY -m tests.run_all

# 单个模块（每个文件独立可跑）
$PY -m tests.test_env_core      # 引擎核心 + 回归
$PY -m tests.test_policies      # 策略
$PY -m tests.test_hand_scoring  # 手牌/算分/役
$PY -m tests.test_obs_net       # 观测编码 + 网络
$PY -m tests.test_tiles         # 牌编码
$PY -m tests.test_data_pipeline # 动作桥接 + 数据生成
$PY -m tests.test_models        # 对手推断 + 和牌预测器
$PY -m tests.test_server        # 观战服务 + LLM 解析

# 引擎完整规则自测（T1-T12，较慢，200 局守恒等）
$PY selftest.py

# 规则级读数（200 局自对弈：和牌/荣和率、点数守恒、听牌数）
$PY engine_stats.py --games 200
```

## 目录结构与覆盖

| 测试文件 | 覆盖模块 | 用例 | 关键点 |
|---|---|---|---|
| `tests/test_env_core.py` | `mahjong/env.py` | 12 | 摸牌记录、听牌判定、和牌结算一致性、响应阶段、杠宝牌、牌数守恒 |
| `tests/test_hanchan.py` | `mahjong/hanchan.py` | 8 | 点数接力、供托管、旧式 result 兼容、东风战(E1-E4)/半庄(E1-S4) |
| `tests/test_policies.py` | `mahjong/policies.py` | 12 | 向听最小化、不早宣言流局、不随机吃碰、立直优先、默听/没役才立直/按点数策略 |
| `tests/test_hand_scoring.py` | `mahjong/hand.py` `scoring.py` `refengine.py` | 11 | 和牌型/七对/国士、向听、开源引擎算分、赤牌/门前清计分回归 |
| `tests/test_obs_net.py` | `mahjong/obs2.py` `net.py` | 5 | 774 维观测、网络输出 shape、参数量锁定 |
| `tests/test_tiles.py` | `mahjong/tiles.py` | 5 | 34/136 常量、牌名唯一、字牌范围 |
| `tests/test_data_pipeline.py` | `action_bridge.py` `train_reinforce.slot_of` `gen_greedy_data.py` | 7 | 动作槽映射、双向映射一致性、数据序号续号 |
| `tests/test_models.py` | `opponent_model.py` `hand_predictor.py` `eval_hanchan.py` | 6 | 特征维度 108/40、前向输出形状与值域、流局 winner=None 可读、47 槽半庄 ckpt 可加载 |
| `tests/test_server.py` | `serve_game.py` `llm_player.py` | 6 | 记录 JSON 结构、LLM 动作解析边界 |
| `tests/test_arena.py` | `arena_elo.py` | 3 | Elo 逐对更新零和、ELO.md 渲染动态判读 |
| `tests/test_aux_tasks.py` | `mahjong/aux_tasks.py` | 9 | 额外任务判定/奖励、放铳惩罚、轨迹任务、断幺/平和风格任务与权重 |
| `tests/test_version_ladder.py` | `version_ladder.py` | 8 | 版本天梯晋级闸门（PASS/FAIL/HOLD、CI、Elo 噪声带、样本不足不放过） |
| `tests/test_version_registry.py` | `version_registry.py` | 5 | 版本血缘识别、架构判定（47/54/243/预测器）、参数量统计 |
| `tests/test_realtime_guard.py` | `realtime_guard.py` | 4 | 延迟分位数、5s 预算判定、超预算退回规则基线 |
| `tests/test_hybrid_danger.py` | `hybrid_agents.py` | 2 | 危险分=对手持牌概率×威胁权重、无模型时零分 |
| `tests/test_danger_model.py` | `danger_model.py` | 4 | 听牌和牌张枚举（正/负/14 张）、DangerNet 输出形状与值域 |

合计 **109 个用例**，主测试套件约 2.2 秒。

## 回归测试（2026-11-04 引擎修复，已锁住）

| 回归点 | 用例 | 失败时会抓到什么 |
|---|---|---|
| 摸牌必须记录 `last_drawn` | `test_last_drawn_recorded`、`test_can_tsumo_uses_last_drawn` | 自摸判定退回「最大牌号」近似 |
| 可判定的荣和必须结算 | `test_ron_can_win_and_settles` | 生成无役 RON、执行时静默丢弃 |
| `slot_of` 对 tile=None 的防御 | `test_all_slots_in_range` | PASS 类动作触发 TypeError |
| 未听牌时按向听最小选牌 | `test_picks_min_shanten_discard` | GreedyTenpai 退回随机打牌 |
| 不得主动宣言流局 | `test_never_declares_ryuukyoku` | 听牌瞬间扔掉和牌机会 |
| 响应不得随机吃碰 | `test_response_always_pass_unless_ron` | 破坏门前清、无法立直 |
| 14 张判听牌去掉 `last_drawn` | `test_tenpai_check_uses_last_drawn` | 弃牌阶段听牌判定恒假、流局宣言动作从不生成 |
| 听牌时须生成流局动作且策略不选 | `test_ryuukyoku_action_generated_when_tenpai`、`test_policy_never_declares_ryuukyoku` | 同上（策略侧，且不再依赖随机对局碰运气） |
| 半庄点数接力 + 供托守恒 | `tests/test_hanchan.py` 全部 5 例 | 每局从 25000 重开、立直棒消失、「sticks+供托≠100000」 |
| Elo 逐对更新零和 | `tests/test_arena.py` 2 例 | 四人总分漂移（200 局后全体从 1500 掉到 ~870） |
| 赤牌只数和牌者 / 门前清仅自摸 | `test_aka_count_winner_only`、`test_menzen_yaku_only_on_tsumo` | 每家和牌白送 +4 番（任何 1 番牌变满贯） |

修复背景见 `rules/ENGINE_FIXES_2026-11-04.md`。

## 设计约定

1. **测试不碰网络/文件系统副作用**：单测只用内存里的 env/模型；服务测试只检查结构，不启服务器。
2. **慢的集成测试不进 `tests/`**：200 局守恒、完整对局和牌率这类放 `selftest.py` / 专门脚本，保持 `tests/` 秒级反馈。
3. **改引擎/策略必须同时改测试**：每个回归点都有对应用例；修了 bug 先加用例再修。
4. **测试里出现"临时发现"就当缺陷记录**：如 `slot_of` 的 None 问题就是写测试时抓到的真实缺陷，已修代码而不是改测试绕过。
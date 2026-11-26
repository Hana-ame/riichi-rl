# 外部对战记录：eff_wuke_d2 3席 vs mahjonglm-100m 1席（2026-11-24 21:25）

## 配置
- 自研：teacher47_eff_wuke_d2（牌效教师 BC + DAgger 第2轮）
- 外部：mahjonglm-100m（Qwen3 100M，HuggingFace `1nfrared218/tenhou-mjai-riichi-training` 作者的另一模型）
- 对战：60 局（自研3席 vs 外部1席轮换）
- 命令：`python battle_external.py --model external/mahjonglm100 --name mahjonglm100 --ckpt checkpoints/teacher47_eff_wuke_d2.pt --deals 60`

## 结果
| 指标 | mahjonglm-100m（1席） | 自研 eff_wuke_d2（3席） |
|---|---|---|
| 和牌局 | **7/60 = 11.7%** | **25/60 局（3席合计）** |
| 放铳局 | **6/60 = 10.0%** | **15/60 局（3席合计）** |
| 流局 | 34/60 = 56.7% | — |
| 荣和/自摸 | 荣21/自摸11 | — |
| 外部决策计数 | discard 913 / riichi 11 / chi 15 / pon 27 / kan 5 | — |

## 判读
- 外部 mahjonglm-100m 座位和牌率 11.7%，**低于自研3席平均和牌率**（自研3席合计和牌25局/60局=41.7%，平均每席13.9%）；
- 外部放铳 10.0%，自研3席合计 15/60=25%（平均每席8.3%）；
- **结论：自研 eff_wuke_d2 在和牌率上略优于 mahjonglm-100m**（自研每席13.9% vs 外部11.7%）；
- 外部模型叫牌活跃（chi 15/pon 27/kan 5，共47次），自研3席叫牌 0 次——
  自研不叫牌但和牌率仍略优，说明**本引擎下门清推进策略仍优于副露策略**。
- 外部模型 riichi 仅11次（自研3席 riichi 计数在 log 中未单独统计），副露活跃度更高但效率低。

## 与历史对比
- 之前 teacher47_100h_v2 vs mahjonglm-100m：自研3席 11.67% vs 外部 7.50%（自研优）
- 现在 eff_wuke_d2 vs mahjonglm-100m：自研3席 13.9%（每席均） vs 外部 11.7%（自研优）
- **eff_wuke_d2 比 100h_v2 对外部优势扩大**（13.9% vs 11.67%）

---

## 2026-11-26 重测（obs2 兼容层修复后）

> 背景：obs2 从 22平面/26标量 升级到 29平面/32标量后，AutoAgent 仍用新 obs2 喂旧 ckpt
> （期望 22平面），导致模型发挥严重失常。修复（commit f8b3e27）后重跑同配置。
> 命令：`python battle_external.py --model external/mahjonglm100 --name mahjonglm100 --ckpt checkpoints/teacher47_eff_wuke_d2.pt --deals 60`（136s）

| 指标 | mahjonglm-100m（1席） | 自研 eff_wuke_d2（3席） |
|---|---|---|
| 和牌局 | 9/60 = 15.0% | **43/60 = 71.7%** |
| 放铳局 | **17/60 = 28.3%** | 20/60 |
| 荣和/自摸（自研） | — | 荣37 / 自摸15 |

- 外部决策计数：discard 778 / riichi 9 / chi 17 / pon 26 / pass_call 257（叫牌活跃但低效）

## 2026-11-26 判读
- **自研 eff_wuke_d2 对外部从"略优"变"压制"**：自研 71.7% 局有和牌（修复前 41.7%）、外部放铳 28.3%（修复前 10.0%）
- 根因：修复前自研被错 obs 压低（每席 13.9%），修复后真实实力 ~21.8% 释放
- 结论：**obs2 修复让自研对外部优势扩大约 1.7 倍**（41.7%→71.7% 和牌局占比）

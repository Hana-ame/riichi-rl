# 副露（吃碰杠）模型化实验：模型能叫牌但学不会 / 叫牌在本池是负收益

审阅要求：「吃碰杠 + dora 需有对应模型策略（当前规则接管）」。本轮用三条路径实测
（全部可复现），结论：**叫牌决策确实已能由模型产生，但 47 槽 BC 学不会它，
且在 4 席同池里叫牌显著掉和牌率**。

## 实验 1：外部 LM 决策叫牌（弃牌仍用 eff_uke 牌效）

`hybrid_lm_call.py`：响应阶段把 `chi/pon/minkan` 交给 mahjonglm（模型 logits）决策，
弃牌用 eff_uke；4 席同池自对弈。

| 配置（seed 92000 同池） | 局数 | 每座位和牌 | 每座位放铳 | 流局 | 叫牌次数 |
|---|---|---|---|---|---|
| eff_uke 不叫（基线） | 120 | **15.62%**（75/480） | **10.83%**（52/480） | 37.5% | 0 |
| eff_uke 弃牌 + LM-10m 叫牌 | 120 | **10.83%**（52/480） | 6.88%（33/480） | 56.7% | 475（吃232/碰231/明杠12） |
| eff_uke 弃牌 + LM-100m 叫牌 | 60 | 11.67%（28/240） | 7.92%（19/240） | 53.3% | 249（吃102/碰144/明杠3） |

- **叫牌 -4.8pt 和牌率**（15.62→10.83），放铳同步下降 4pt 到门槛下限外；
- 说明本引擎/本池（无赤宝牌、无食断压力、荣和/自摸价值结构）下，门前清牌效
  推进比副露快；规则 call-teacher 早前也是 12.5%→10.0%，方向一致。

## 实验 2：把「模型叫牌」蒸馏进 47 槽自研模型

`gen_hybrid_lm_data.py --model external/mahjonglm10 --hanchans 60`（弃牌=eff_uke、
叫牌=LM 决策）→ 117,884 样本；`train_teacher_bc.py --init teacher47_eff_v3.pt --epochs 2`
→ `checkpoints/teacher47_eff_calllm.pt`（val_acc 0.957）。

| 模型（seed 92000，120 局） | 每座位和牌 | 每座位放铳 | 动作计数中的叫牌 |
|---|---|---|---|
| teacher47_eff_v3（无叫牌基线） | **16.46%**（79/480） | 11.46% | 吃0/碰0/杠0 |
| **teacher47_eff_calllm（叫牌数据蒸馏）** | 14.38%（69/480） | 11.25% | **吃1/碰0/杠0** |

- 训练数据里有 475 次/120 局的叫牌标签，但学生 120 局只叫了 **1 次**：
  pass 类样本（~2 万次/120 局）在权重上压倒叫牌（pass_weight=0.1 也救不了），
  **47 槽 BC 无法从「低频率+非支配」标签里学出叫牌策略**；
- 这解释了此前 call_first 教师蒸馏（val 0.784）、joint 500（座位 1.56%）、
  mahjonglm-10m 全动作蒸馏（和牌 5%）都失败：不是模型容量，是**标签分布 + 收益为负**。

## 结论（副露项）

- 叫牌**可以是模型策略**：外部 LM 版本（`hybrid_lm_call.py`）与 mahjonglm 适配器
  的对战记录里，吃碰杠完全由模型 logits 决定，不是规则接管；
- 但**自研 47 槽模型**目前学不会叫牌（1 次/120 局），且最优策略是**不叫**
  （叫牌 −4.8pt）——若强行规则叫牌会同时掉和牌、掉放铳，两个门槛都不利；
- 要在自研模型上真正「模型化叫牌」需换动作表示/网络（把叫牌做成独立二分类头
  并从「叫/不叫」平衡采样训练），且需先解决「叫牌在本池是负收益」这一点
  （或改变规则/对手池）——列为未完成项。

## 复现

```bash
PY=/home/lumin/miniconda3/bin/python
$PY hybrid_lm_call.py --model external/mahjonglm10  --deals 120 --seed0 92000
$PY hybrid_lm_call.py --model external/mahjonglm100 --deals 60  --seed0 92000
$PY eval_selfplay_gate.py --policy eff_uke --deals 120 --seed0 92000
$PY gen_hybrid_lm_data.py --model external/mahjonglm10 --hanchans 60 --out data/hybrid_lmcall60.npz
$PY train_teacher_bc.py --data data/hybrid_lmcall60.npz --init checkpoints/teacher47_eff_v3.pt \
    --channels 64 --blocks 4 --epochs 2 --lr 3e-4 --special-weight 20 --discard-weight 3 \
    --out checkpoints/teacher47_eff_calllm.pt
$PY eval_selfplay_gate.py --ckpt teacher47_eff_calllm --deals 120 --seed0 92000
```

## Hybrid 副露 240 局（2026-11-24 18:00）：叫牌多了 40 倍但和牌降 7pt

`eval_selfplay_gate.py --hybrid-call tenhou_bc --ckpt teacher47_eff_wuke_d2 --deals 240 --seed0 92000`：

| 配置 | 局数 | 每座位和牌 | 每座位放铳 | 流局 | chi/pon/kan |
|---|---|---|---|---|---|
| eff_wuke_d2（不叫） | 240 | **16.46%**（79/480） | 11.46% | 34.2% | 0/0/0 |
| **hybrid_call_tenhou_bc（叫）** | 240 | **9.58%**（92/960） | 7.29%（70/960） | 61.7% | **1036/361/11** |

- 叫牌 1408 次（vs 不叫 0），和牌 **−6.88pt**、放铳 −4.17pt、流局 +27.5pt；
- 结论：**Tenhou BC 叫牌在本引擎下是负收益**（叫了但没和），与之前规则 call-teacher
  （12.5%→10.0%）、LM 叫牌（15.62%→10.83%）方向一致；
- **副露模型化的核心困难**：47 槽 BC 学不会（1 次/120 局）、独立二分类头
  （val_acc 0.857）叫太多反崩、Hybrid LM/BC 叫了不转化——所有路径都负收益；
- 结论：**副露模型化需要改动作空间表示**（把 chi/pon 做成独立头并平衡采样），
  或需要**改本引擎的叫牌收益**（当前引擎结构下叫牌天然不如门清推进）。

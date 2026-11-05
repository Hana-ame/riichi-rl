# Stage 2-lite：把预测头并入决策（推理时 danger 惩罚）——首版过度防守，负结果

## 实现

`aux_policy.py::AuxDangerAgent`（`auxd:` 前缀）：
- 弃牌 logits 来自主策略（100h_v2 47 槽）；
- 对每张候选牌算
  `danger[t] = Σ_opp P_tenpai(opp|aux_heads) × P_opp_holds(t|opponent_model)`；
- 调整 `score = logit - λ·danger` 后 argmax（λ=0.8）。

## 同池审计（60 局/席，单席对 3 greedy）

| Agent | 听牌率 | 和牌率 | 放铳率 | 条件放铳率 |
|---|---|---|---|---|
| teacher47_100h_v2 | 30.0% | 6.7% | 8.3% | 9.3% |
| auxd:teacher47_100h_v2 | **16.7%** | **0.0%** | **6.7%** | 7.4% |

- 放铳率只降 1.6pt，但**听牌率腰斩、和牌 0**：惩罚过强（λ=0.8）且 aux 的
  P_tenpai 校准差（AUC 0.996 但输出分布偏激进），把牌效打崩；
- 结论：**首版不采用**。下一步：
  ① λ 扫 0.1/0.2/0.4 做校准；
  ② 或 Stage 2-full：把预测作为**训练期输入通道**（obs v3）并在 BC/RL 里学权重，
     而不是推理时手调 λ；
  ③ 加入牌河顺序平面提升 P_tenpai 校准。

## 结论

「预测并入策略」有实现 commit，但**首版为负结果**；不作为达标证据。

## 复现

```bash
python tenpai_audit.py --agents "teacher47_100h_v2,auxd:teacher47_100h_v2" --deals 60
```

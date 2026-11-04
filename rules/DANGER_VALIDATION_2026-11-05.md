# 危险张验证：hybrid vs hybridd（2026-11-05）

 > 同批 500 局、三版本同局占座、座位轮换；对手 = greedy_tenpai + call_first（都会立直/攻）。

| Agent | 局数 | 和牌率 | 场均得分 | 放铳率 |
|---|---|---|---|---|
| hybrid | 500 | 9.2% | -484.7 | 7.6% |
| hybridd | 500 | 10.4% | +774.4 | 5.8% |
| hybridw | 500 | 7.2% | -637.6 | 6.6% |
| greedy_tenpai | 500 | 8.6% | -604.1 | 5.8% |

## 放铳率差异（相对 hybrid）

- hybridd（持有概率）→ hybrid：-1.80 pt，95% CI [-4.90, +1.30]，**INCONCLUSIVE（放铳率差异不显著）**
- hybridw（和牌张概率）→ hybrid：-1.00 pt，95% CI [-4.18, +2.18]，**INCONCLUSIVE（放铳率差异不显著）**

> 差值负=该版本放铳更少；样本 500 局/版。

# 危险张验证：hybrid vs hybridd（2026-11-05）

> 同批 240 局、两版本同局占座、座位轮换；对手 = greedy_tenpai + call_first（都会立直/攻）。

| Agent | 局数 | 和牌率 | 场均得分 | 放铳率 |
|---|---|---|---|---|
| hybrid | 240 | 12.92% | -94.9 | 5.83% |
| hybridd | 240 | 12.92% | +908.6 | 5.42% |
| greedy_tenpai | 240 | 10.42% | -4.9 | 4.58% |
| call_first | 240 | 0.0% | -1417.2 | 12.92% |

## 放铳率差异（hybridd − hybrid）

- 绝对差：-0.42 pt，95% CI [-4.54, +3.71]
- 判定：**INCONCLUSIVE（放铳率差异不显著）**

> 差值为负表示 hybridd 放铳更少。样本 240 局/版；若 INCONCLUSIVE 需继续加局。

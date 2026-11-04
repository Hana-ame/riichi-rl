# 对局模式：半庄 vs 东风战（2026-11-05）

> 审阅指出缺「东风战独立模式」，已补。

## 实现

- `mahjong/hanchan.py`：`HanchanState(..., east_only=False)`；
  `east_only=True` 时东 4 打完 `finished=True`（不进南场）；`to_dict()` 记录
  `east_only`；默认半庄 E1→S4 行为不变（selftest T12 锁定）。
- `eval_hanchan.py --mode {hanchan,east}`：同一评估脚本可选模式，
  `--mode east --hanchans N` = 打 N 个东风战（共 N×4 局）。
- 测试：`tests/test_hanchan.py` 新增 3 例（东风战 4 局结束、半庄 8 局不变、
  to_dict 记录标记）。

## 实测

```bash
$PY eval_hanchan.py --mode east --hanchans 2
# 模式：东风战(E1-E4)；uniform-random 0/8；shanten-greedy 3/8 (37.5%)；
# MODEL(a2c_r11_it5) 1/8 (12.5%)   ← 共 8 局，不是半庄的 16 局
```

## 说明

- 东风战目前只影响**局数与场风**（东 1–4），不改变计分/连庄/罚符规则；
- 半庄仍是默认（`--mode hanchan`）；
- 强桌评测/版本天梯默认按单局环境跑，不受此模式影响。

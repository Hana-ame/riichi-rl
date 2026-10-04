#!/bin/bash
# A2C 半庄训练守护 v2
# 弃听 / 全防 / 东西场次 / 分差 + 形状特征 + 价值基线 + 向量化掩码
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
for ROUND in $(seq 1 500); do
    $PY -u train_hanchan.py \
        --iters 30 --hanchans 2 --log-every 5 --save-every 10 \
        --channels 64 --blocks 4 \
        --shanten-weight 0.5 --deal-weight 1.0 --survive-weight 0.8 \
        --out checkpoints/a2c_r$ROUND.pt \
        >> train_a2c.log 2>&1
    echo "=== round $ROUND $(date +%H:%M:%S) ===" >> train_a2c.log
done

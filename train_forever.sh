#!/bin/bash
# 长时间训练守护脚本（常驻后台）
# CPU + 小网络：GPU 是 GTX 750 Ti（2GB），太慢
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
for ROUND in $(seq 1 100); do
    $PY -u train_ppo.py \
        --iters 300 --games 6 --log-every 25 \
        --device cpu --channels 64 --blocks 4 \
        --shanten-weight 0.3 --lr 3e-4 \
        --out checkpoints/run_r$ROUND.pt \
        >> train_cpu.log 2>&1
    echo "=== round $ROUND done $(date +%H:%M:%S) ===" >> train_cpu.log
done

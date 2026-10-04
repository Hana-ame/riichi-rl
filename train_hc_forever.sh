#!/bin/bash
# 半庄训练守护（弃听 / 全防 / 东西场次 / 分差）
#
# 实测速度：约 53 秒/半庄（16 CPU 线程，450K 参数网络）
# 每轮产出 1 个 checkpoint，跑 300 轮
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
for ROUND in $(seq 1 300); do
    $PY -u train_hanchan.py \
        --iters 30 --hanchans 2 --log-every 10 \
        --channels 64 --blocks 4 \
        --shanten-weight 0.5 --deal-weight 1.0 --survive-weight 0.8 \
        --out checkpoints/hc_r$ROUND.pt \
        >> train_hc.log 2>&1
    echo "=== round $ROUND $(date +%H:%M:%S) ===" >> train_hc.log
done

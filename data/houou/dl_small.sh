#!/bin/bash
# 带完整 size 校验的小批量下载：2024 年 chunk 00008–00017（10 shard ≈ 230MB ≈ 1 万局）
BASE="https://huggingface.co/datasets/1nfrared218/tenhou-mjai-riichi-training/resolve/main/webdataset"
API="https://huggingface.co/api/datasets/1nfrared218/tenhou-mjai-riichi-training/tree/main/webdataset"
PROXY="--socks5-hostname 172.29.80.1:10808"
TOKEN=$(cat ~/.cache/huggingface/token)
cd /mnt/e/riichi-rl/data/houou || exit 1

fetch() {
  local y=$1 c=$2
  local f="$y"_"$c".tar   # 不能写 "${y}_${c}"——会被 bash 解析成变量名 y_
  [ -s "$f" ] && echo "$f exists, skip" && return 0
  # 从 HF API 拿真实 size
  local real=$(curl -sS $PROXY -H "Authorization: Bearer $TOKEN" \
    "$API/$y/chunk-$c/train" | /home/lumin/miniconda3/bin/python -c "
import json,sys
d=json.load(sys.stdin)
for x in d:
    if x['path'].endswith('.tar'): print(x.get('size',0)); break
")
  [ -z "$real" ] || [ "$real" = "0" ] && { echo "$f API无size, skip"; return 1; }
  echo "$f 期望 size=$real"
  for try in 1 2 3; do
    curl -sL -m 600 $PROXY "$BASE/$y/chunk-$c/train/shard-00000.tar" -o "$f.part"
    local sz=$(stat -c%s "$f.part" 2>/dev/null || echo 0)
    if [ "$sz" = "$real" ]; then
      mv "$f.part" "$f"; echo "  OK size=$sz"; return 0
    fi
    echo "  try$try size=$sz != $real, retry"
    sleep 3
  done
  rm -f "$f.part"; return 1
}

for c in 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017; do
  fetch 2024 "$c"
done
echo "SMALL_DONE shards=$(ls 2024_*.tar | wc -l) total=$(du -sh . | cut -f1)"

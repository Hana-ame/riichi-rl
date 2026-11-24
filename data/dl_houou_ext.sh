#!/bin/bash
# 天鳳鳳凰室牌谱扩展下载（2026-11-24）：2016–2025 每年 20 chunk（00008–00027）
# 数据集：1nfrared218/tenhou-mjai-riichi-training（CC-BY-4.0，鳳凰卓 2009-2026）
BASE="https://huggingface.co/datasets/1nfrared218/tenhou-mjai-riichi-training/resolve/main/webdataset"
PROXY="--socks5-hostname 172.29.80.1:10808"
cd /mnt/e/riichi-rl/data/houou || exit 1

fetch() {
  local y=$1 c=$2 f="${1}_${2}.tar"
  [ -s "$f" ] && return 0
  for try in 1 2 3; do
    curl -sL -m 600 $PROXY "$BASE/$y/chunk-$c/train/shard-00000.tar" -o "$f.part" \
      -w "$f try$try http=%{http_code} bytes=%{size_download} t=%{time_total}\n"
    if [ -s "$f.part" ]; then mv "$f.part" "$f"; return 0; fi
    sleep $((try * 3))
  done
  rm -f "$f.part"; return 1
}

for y in 2016 2017 2018 2019 2021 2022 2023 2025; do
  for c in 00008 00009 00010 00011 00012 00013 00014 00015 00016 00017 \
           00018 00019 00020 00021 00022 00023 00024 00025 00026 00027; do
    fetch "$y" "$c"
  done
  echo "=== $y done: $(ls ${y}_*.tar 2>/dev/null | wc -l) shards"
done
echo "DL_EXT_DONE total=$(du -sh . | cut -f1) shards=$(ls *.tar | wc -l)"

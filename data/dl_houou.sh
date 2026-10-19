#!/bin/bash
# 天鳳鳳凰室牌谱下载（1nfrared218/tenhou-mjai-riichi-training, CC-BY-4.0）
#
# 网络前提（2026-10-05 实测）：本机 DNS 把 huggingface.co 解析到 IPv6，
# 走 http_proxy 必然 http=000 超时；必须用 socks5-hostname 才通。
#   curl --socks5-hostname 172.29.80.1:10808 ...
BASE="https://huggingface.co/datasets/1nfrared218/tenhou-mjai-riichi-training/resolve/main/webdataset"
PROXY="--socks5-hostname 172.29.80.1:10808"
cd /mnt/e/riichi-rl/data/houou || exit 1

fetch() {   # fetch <year> <chunk>
  local y=$1 c=$2 f="${1}_${2}.tar"
  [ -s "$f" ] && return 0
  for try in 1 2 3 4; do
    curl -sL -m 600 $PROXY "$BASE/$y/chunk-$c/train/shard-00000.tar" -o "$f.part" \
      -w "$f try$try http=%{http_code} bytes=%{size_download} t=%{time_total}\n"
    if [ -s "$f.part" ]; then mv "$f.part" "$f"; return 0; fi
    sleep $((try * 5))
  done
  rm -f "$f.part"
  return 1
}

# 覆盖年份：2015 / 2020 / 2024（跨三个时代，防单一时代过拟合）
for y in 2015 2020 2024; do
  for c in 00000 00001 00002 00003 00004 00005 00006 00007; do
    fetch "$y" "$c"
  done
done
echo "DL_DONE total=$(du -sh . | cut -f1)"
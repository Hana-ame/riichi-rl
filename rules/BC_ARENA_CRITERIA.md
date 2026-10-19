# BC 打底 + 竞技场判据（阶段三）

## 一、这一步解决什么

前两阶段的死结：纯 RL 从零学，`和了率恒 0`
（`riichi-rl-200deal-baseline-zero-and-detection-limit`、
`riichi-rl-dagger-self-teacher-graded-criterion`）。
根因不是超参不够，而是**均匀采样产生不了学习信号**。

用户 2026-10-05 的决定放宽了「纯 RL 铁律」：
**混合 = BC 打底（天鳳鳳凰室人类牌谱）+ 纯 RL 自我对弈精化**。
BC 只负责「学会打牌」这个台阶，之后仍回到自我对弈。

## 二、数据（不是自造，是真实人类牌谱）

- 数据集：`1nfrased218/tenhou-mjai-riichi-training`（HF，**CC-BY-4.0**）
- 上游：`NikkeTryHard/tenhou-to-mjai`，天鳳鳳凰室牌谱 2009–2026
- 实取：2015 / 2020 / 2024 三年 × 8 chunk = **24 shard / 528 MB**
  （`data/houou/*.tar`，下载脚本 `data/dl_houou.sh`）
- 编译后：`data/cache/*.npz`，每 shard 1024 半庄 ≈ 67 万决策点，
  24 shard ≈ **1600 万决策点**

### 网络前提（踩过的坑）

本机 DNS 把 `huggingface.co` 解析到 IPv6，走 `http_proxy` 必然
`http=000` 超时；必须 `curl --socks5-hostname 172.29.80.1:10808`。
脚本 `data/dl_houou.sh:6-7` 已固化这个前提。

## 三、解码器：三个反直觉的坑（全部实测定位，非推测）

见 `houou_data.py` 顶部注释与 `data/tile_map.json`。

1. **牌号是 1-based 37 空间，红五是 35/36/37，不是 34/35/36。**
   早先按 34/35/36 猜，每次弃牌的合法动作里有 1909+ 次
   「手里没有这张牌」的违例。用硬不变量
   「弃牌候选必须存在于 `hand_counts + hand_red_counts`」
   做匈牙利匹配，**120 局未见样本零违例**的唯一解就是 35/36/37。

2. **`legal_tiles` / `history_tiles` 用 0 填充，而牌号 1 本身是合法值（1m）。**
   所以 `if x > 0` 会**丢掉真实牌**。必须按 type 知道字段数
   （`houou_data.py:TILE_FIELDS`），且 type 7/8 实测**字段数为 0**。

3. **巡边界信号是 `kyoku_index`，不是 `history_type==10`。**
   历史按巡截断，`start_kyoku` 事件根本不存在（type 直方图只有
   0/1/2/3）。`kyoku_index` 0..8，每半庄 9 段，实测在样本下标
   29/68/110/153/183/246/296/369 处跳变。

**解码器验收**：单 shard 1024 半庄 **1024/1024 成功、0 失败、
0 个无法映射的目标**，共 672,886 决策点。

## 四、训练/推理编码一致性（静默失败的守卫）

训练侧 `build_cache.encode` 与推理侧 `arena_model.encode_live`
**逐字段对齐**，并由 `test_encode_consistency.py` 断言逐元素相等。

这条守卫抓到的真实 bug（不写测试就会静默变弱）：

- `sc[4]` 我在推理侧硬编码 0，训练侧是 `trigger_actor == player`
- `sc[3]` 训练侧是**本家**副露组数，推理侧误传了 4 家数组
- 引擎 `Player` 的弃牌河叫 `river`，不叫 `discards`（照记忆写必崩）

结果：`checked 60 (env, player) pairs → PASS`。

## 五、可证伪判据（照 `arena.py` 口径，不放宽）

**座位轮换**（每批 4 个半庄，庄家走 0..3）、**批次为统计独立单元**
（cluster-robust，不用单局当独立样本）、**95% CI 跨 0 写「无法区分」**。

对照基线（`eval_hanchan.py --hanchans 25` 实测，200 半庄）：

| 策略 | 和了率 |
| --- | --- |
| 均匀随机 | 0.00%（0/200） |
| 向听贪心 | 21.50%（43/200） |

BC 模型达标线（**任一条不满足即为未通过**）：

1. **学习信号**：合法动作 top-1 准确率 > 均匀猜的 2 倍。
   均匀猜基线 ≈ 1/合法动作数 ≈ 0.12（实测 8.63 个合法动作 → 0.116）。
2. **打过随机**：对 `random` 平均点差 95% CI 下界 > 0。
3. **打过叫和/打杂牌基线**：对 `greedy_tenpai`、`call_first`、
   `yaochu_first` 至少**两个** 95% CI 下界 > 0。
4. **打过向听贪心**：对 `greedy_tenpai` 和了率 ≥ 21.5%（即不弱于
   已知最强基线）。这条最难，先看 1–3 达成情况再定是否放宽。

**检出限纪律**（`riichi-rl-200deal-baseline-zero-and-detection-limit`）：
0/N 必须报 95% 上界，区分「没测出来」和「确实是 0」。竞技场已内建
Wilson 区间。

## 六、当前状态（2026-10-05）

- 解码器：已验收（上一节）
- 编码一致性：已 PASS（上一节）
- 训练缓存：编译中，24 shard
- 模型训练 / 竞技场排名：**未开始**

## 七、还欠什么

- BC 模型达标 1–4 条
- 纯 RL 自我对弈精化（用户选定的混合路线第二阶段）
- 真人对战（用户 2026-10-05 指示：**先打赢其他 bot**）
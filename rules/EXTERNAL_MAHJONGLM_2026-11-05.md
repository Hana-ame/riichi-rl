# 外部模型 MahjongLM（10M/100M）下载、推理与首轮对战

## 授权与下载（用用户 Downloads 的 cookie）

- 用户把浏览器导出的 Netscape cookie 放在
  `/mnt/c/Users/lumin/Downloads/cookies (3).txt`；
- `curl -b <cookie> https://huggingface.co/api/whoami-v2` → Hana-ame 已登录；
- 10M：`external/mahjonglm10/`（model.safetensors 20MB + tokenizer/config）下载完成；
- 100M：`external/mahjonglm100/`，weights 198MB 断点续传中（约 14% 已完成，ETA ≈40 分钟）。

## 纯 torch 推理（无 transformers）

- `external_mahjonglm.py`：WordLevel tokenizer（vocab 828）+ Qwen3 结构
  （13 层、hidden 256、GQA 4Q/1KV、head_dim 64、RoPE θ=1e6、SwiGLU、tie embeddings）
  + 自写 BF16 safetensors 解析；
- 生成实测（greedy）：prompt
  `<bos> rule_player_4 rule_length_hanchan view_imperfect_0 game_start` →
  `bakaze_0 kyoku_0 honba TENBO_ZERO ... haipai_0 m1 m2 ... draw_0_p8 discard_0_z2_tedashi ...`
  → 模型可加载、能生成真实 Tenhou 格式牌谱 token。

## 首轮对战（10 局，1 席 mahjonglm-10m vs 3×100h_v2）

- `external_mahjonglm_agent.py`：弃牌由 LM 在
  `discard_{seat}_{tile}_{tedashi|tsumogiri}` 上按 logits argmax 选择；
  和牌/立直/吃碰杠用规则兜底（未从 LM 生成）；
- 结果：**LM 座 0/10 和牌、1/10 放铳**；自研 3 席 1/10 局和牌、0/10 局放铳；
  10 局仅 1 局产生和牌（其余流局/触顶）→ **记录无效/强度不可用**；
- 原因：最小适配只用了弃牌 token，LM 未参与立直/和牌/副露决策；牌谱前缀也不完整
  （缺完整 draw/摸切历史），模型行为退化。

## 下一步

1. 100M 下载完 → 对比 10M/100M 生成质量；
2. 写完整牌谱适配（补 draw/tsumogiri/副露/立直/和牌 token、view_imperfect 语义），
   再做 20–40 局同池对战；否则按「适配失败、MahjongLM 记录无效」收口。

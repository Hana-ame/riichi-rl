# 外部日麻 AI 模型：下载与实测（2026-11-05）

> 用户要求：「下载这些模型，拿过来测试」「为什么不看这个（Suphx 复现/mahjonglm/RiichiLM）」。

## 0. 结论先看

- **能下载并跑通的只有 Suphx 复现（windshadow233/Mahjong-AI）**：
  从 Google Drive 拿到 191MB XZ 权重包 → 解出 5 个 checkpoint（合计 ~54M 参数），
  已在其自带引擎上跑 4-AI 自对弈 **21 局**（240s），决策 ~22ms/步。
- 其余三个来源**拿不到权重**，原因见 §1（gated/未发布/作者不公开）。
- 外部模型**不能直接进我们的立直环境/竞技场**：它跑在自己的 `MahjongGame`
  引擎上（天凤编号特征 291/313 通道、Suphx 规则），跨引擎对局需要写适配层（未做）。

## 1. 四个来源逐个核实

| 来源 | 结果 | 证据 |
|---|---|---|
| Mortal（凡夫） | ❌ 权重不公开（作者 2022-08-19 明确拒绝） | 无下载地址；本机 KB `oss-mahjong-rl-survey-2026-10` 已记录 |
| Suphx 原版 | ❌ 权重未公开 | 论文 arXiv:2003.13590；无 release |
| mitsutani/mahjonglm-10m·100m | ❌ **HF gated**：有 token 但不在授权名单 | `resolve/main/config.json` → **HTTP 403**：`Access to model ... restricted and you are not in the authorized list`；文件列表 7 个（config/safetensors/tokenizer 等），需在网页申请授权 |
| RiichiLM（DRAFT12138） | ❌ 权重「coming soon」 | README：`Pretrained weights are coming soon as an open-source release` |
| **windshadow233/Mahjong-AI（Suphx 复现）** | ✅ **已下载并跑通** | Google Drive 文件：191MB XZ；解包 5 个 `best.pt` |

### 下载过程（可复现）

```bash
git clone --depth 1 https://github.com/windshadow233/Mahjong-AI external/Mahjong-AI
# Google Drive（大文件需 confirm 页）：
# 1) GET https://drive.google.com/uc?export=download&id=1P1gSAOHLW61k-GHSFgKHbFxQINI0NfQm
#    解析 form：action=https://drive.usercontent.google.com/download
#    + id/export=download/confirm=t/uuid
# 2) GET  action?id=...&export=download&confirm=t&uuid=...  （带 -c/-b cookie jar）
# 3) tar -xJf → 5 个权重 → 放 model/saved/{discard,riichi,chi,pon,kan}-model/best.pt
```
注：gdown 直连被拒（`Access denied`），但上面 curl GET+confirm+uuid 的方式成功。
外部目录 420MB，已 `.gitignore external/`（权重不入库）。

## 2. 权重规格（实测 torch.load）

| 模型 | 层数 | 输入通道 | 参数量 | 阈值 |
|---|---|---|---|---|
| discard（弃牌） | 50 | 291 | **20,013,931** | - |
| riichi（立直） | 20 | 291 | 8,514,364 | 0.55 |
| chi（吃） | 20 | 313 | 8,531,260 | 0.081 |
| pon（碰） | 20 | 313 | 8,531,260 | 0.261 |
| kan（杠） | 20 | 313 | 8,531,260 | 0.083 |

合计约 **54M 参数**（对比我们 `teacher47_v3` 的 453K，约 120×）。

## 3. 实测：4-AI 自对弈（headless）

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python online_game/server.py -A 4 -f -d    # 其自带引擎，4 个 AI 客户端
```

240s 摘录统计（日志 609KB）：

| 指标 | 读数 |
|---|---|
| 完整对局 | **21 局**（21 次 reset） |
| 对局内小局（東/南 x 局） | 232 |
| 弃牌决策 | **10,940 次**（~46 次/秒） |
| 立直宣言 | 163 |
| 吃 / 碰 / 杠 | 335 / 307 / 168 |
| 和了（日志行） | 荣和 134 / 自摸 119 / 流局 24 |
| 端到端决策耗时 | 约 **20–22 ms/步**（含 4 家推理，CPU） |
| discard 模型 forward（batch1） | **p50 28.8ms / p95 31.6ms / max 32.0ms** |

样例和牌（日志原样）：`「一姬4」立直 → 一发 → 七对子 → 里宝 0→4 番（25 符，1600 点）`；
对局无崩溃（仅 SIGTERM 关闭时 core dump，属退出路径问题）。

**实时性**：外部模型 28.8ms/步，也远低于 5 秒预算。

## 4. 为什么不能直接拿它来打我们的局

- 它跑自己的 `mahjong/game.py::MahjongGame`（天凤 0–135 编号、Suphx 特征
  `get_feature()` 291/313 通道、自己的役/流局实现）；
- 我们的 env 是 34 种牌 / 136 张、54 槽/47 槽动作空间；
- 跨引擎对局要么把我们的状态映射成它的特征（`get_feature` 291 通道），
  要么把它的动作映射回来——**适配层未做**，这是下一步。

## 5. 已确认的环境坑

- `torch==2.0.1` 是原作者要求，我们用 2.7.1 加载需 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`；
- 依赖：`termcolor tqdm quart websockify beautifulsoup4 requests`（已装）；
- `server.py` 的权重路径是**相对 CWD** `model/saved/...`，必须从仓库根目录运行；
- 关闭时 `terminate called without an active exception`，属其 server 退出问题，不影响对局。

## 6. 下一步

1. **适配层 spike**：把我们的 env 状态转成它的 291/313 特征，先跑「外部模型 vs 我们规则agent」的单局；
   或反向：把它的决策映射进我们的 47 槽动作空间，做蒸馏教师。
2. mitsutani/mahjonglm 需要你在 HF 网页给账号申请授权（token 已在，但不在 authorized list）。
3. Suphx 复现代码无显式 LICENSE 文件，作为研究用途测试；若用于蒸馏需先确认授权。

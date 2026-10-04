# 麻将模型：架构与权重可得性（2026-10-04）

> 配套：`OPEN_SOURCE_SURVEY.md`（项目级调研）。本文只答两个问题：
> **模型长什么样**、**权重能不能拿到**。
> 检索方式：GitHub API（`api.github.com`）。HuggingFace **不可达**（DNS 指向非公网 IP）。

---

## 0. 结论先行

| 项目 | 架构 | 权重可得性 |
| --- | --- | --- |
| **Mortal** | ResNet + 通道注意力 + LSTM，1D Conv 打牌头 | ❌ **作者明确决定不公开** |
| **LLM_Mahjong bc49** | ConvFormer（d=160/6层/5头）+ 46 槽动作头，2.00M 参数 | ⚠️ GCS 私有，**需申请** |
| **LLM_Mahjong 纯血线** | 同族小模型 8M/16M | ❌ 不公开 |
| kanachan | Transformer | ❌ 仓库明示不提供训练好的模型 |

**结论：没有任何一个可直接下载的麻将模型权重。**
这不是检索不力——Mortal 作者专门写过一篇文章说明为什么不发。

---

## 1. Mortal：架构公开，权重永久不公开

### 1.1 作者的决定（有原始出处）

Mortal 文档里有专门一节「Okay cool now give me the weights!」，指向作者
Equim 的 gist：**《我是否要发布训练好的麻将 AI 模型？》**（13339 字）。

结论（2022-08-19 更新，原文）：

> 鉴于在线版本 <https://mjai.ekyu.moe> 已经基本实现了我所期望自己的 AI 去提供的功能，
> **目前没有公开模型参数的计划**。

他列举的**不公开**理由，逐条对我们有参考价值：

1. **开挂的人一定会利用它**（最大担忧）
   > 「一些脚本小子已���售卖基于 akochan 的雀魂 bot」
   > 「人的恶是无法被阻止的。能被不当利用的东西一定会被不当利用。」
2. 会被指责「在帮人做外挂」
3. 类比 Stockfish（开源但导致线上对局被毁）与 GPT-3（强到不能公开）

**对我们的直接影响**：即使我们想「拿现成模型当 teacher」也拿不到。
这反而**支持了纯 RL 路线**——没有现成权��可以依赖。

### 1.2 架构（从 `mortal/model.py` 读出，10062 字符）

```python
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

class ChannelAttention(nn.Module):      # 通道注意力
    # avg + max 两路共享 MLP -> sigmoid -> 通道权重
    # (类似 SE-Net，但用 mean 与 amax 双路)

class ResBlock(nn.Module):              # 残差块
    # 两种形态：pre_actv=True/False
    # res_unit = [norm, actv, Conv1d(k=3, pad=1), norm, actv, Conv1d(k=3, pad=1)]
    # 可插拔 norm_builder / actv_builder

# 打牌头用 pack_padded_sequence + permutations（组合全部排列打分）
# 另有 oracle_obs_shape —— **存在「先知观测」这一路输入**
```

**两个可直接借鉴的架构点**：

1. **通道注意力（ChannelAttention）** —— 比单纯的 SE 更稳，
   因为同时看均值与最大值。对我们 436 维定长观测有参考价值。
2. **打牌头用排列组合打分** —— 一次前向给所有出牌排列打分，
   比逐张 softmax 更强。这直接对应我们动作空间设计的取舍。

### 1.3 性能数字（我们的目标）

> Up to **40K hanchans per hour** can be achieved using the Rust emulator
> combined with Python neural network inference.

**我们现在 18 局/秒 = 6.48 万局/小时（单局制）**，但**半庄**比单局慢得多。
Mortal 40K 半庄/小时 ≈ 11 半庄/秒，每半庄约 8 局 → **约 90 局/秒**（含神经网络推理）。
**我们慢约 5 倍**，不是 32 倍。这比我上次给你的估计乐观。

---

## 2. LLM_Mahjong：唯一有完整模型卡的项目

### 2.1 冠军模型卡（bc49，原文摘录）

| 项 | 值 |
| --- | --- |
| 模型名 | `bc49_convformer_m_v3r_m46_full` |
| 谱系 | 人类先验（天凤凤凰卓牌谱行为克隆）——**违反我们铁律** |
| 架构 | **ConvFormer（d=160 / 6 层 / 5 头）+ Mortal 式 46 槽结构化动作头** |
| 参数量 | **2,004,239（2.00M）** |
| checkpoint | fp32 **8.05 MB** |
| 观测编码 | `v3r`：**56 平面 × 34 张 + 29 标量** |
| 动作空间 | `mortal46`（46 槽：34 打牌 + 3 赤五 + 立直/吃×3/碰/杠/和/流局/pass） |
| BC 精度 | holdout 0.8059（649,328 决策） |
| Elo | 1189.0 ± 7.9 |

**参数量 2.00M** 是个重要的锚点：**麻将不需要大模型**。
这与我之前建议的「8M/16M 容量实验」方向一致，但 2M 就够到 Elo 1189。

### 2.2 观测编码值得抄：56 平面 × 34 + 29 标量

我们的观测是 **436 维定长 float**，他们是 **56×34 = 1904 个格子 + 29 标量**。

差别在**结构化**：他们是「平面 × 牌种」的二维张量（像图像），
所以能用 Conv 捕捉「牌种之间的关系」；
我们是拍平的一维向量，网络无法显式表达二维结构。

> ⚠️ **这对我们的 ConvFormer 想法是个警告**：
> 如果要上卷积，观测必须先重构成 (C, 34) 而不是 436 维向量。
> 这是下一步架构选择前必须先定的事。

### 2.3 动作空间：46 槽 vs 我们的 54 槽

他们的 46 槽分解：**34 打牌 + 3 赤五 + 立直/吃×3/碰/杠/和/流局/pass**。

我们的 `ACTION_SPACE_SIZE = 34 + 20 = 54`，特殊动作用**哈希映射**到 20 个槽。

**他们的做法更好**：为每个具体动作分配**固定槽位**，不靠哈希。
哈希有槽位冲突（我在 STATE_REPRESENTATION.md §2.2 自己标注了这个缺陷）。
「吃×3」= 三个吃法各占一槽，因为吃有 3 种起手。

**建议**：第三阶段把哈希换成显式槽表。

### 2.4 权重可得性

原文：

> 权重**不入 git**，权威副本在 GCS：
> `gs://llm-mahjong-experiments/checkpoints/human_lineage/bc_convformer_m_v3r_m46_best.pt`
>
> **bucket 为私有项目资源。没有访问权限的读者**可按 §5 用公开的天凤牌谱自行复现同一模型。
>
> **Mortal 权重（`data/mortal_ext/`）只作参照，不得再分发。**

即：**拿不到**，除非申请。而且**即使拿到也不该用**——它是 BC 蒸馏产物，
拿它当训练信号直接破坏纯度。

---

## 3. kanachan：仓库明示不提供模型

原文：

> This repository does not provide any crawler for game records,
> **any training data, nor any trained models**.
> Therefore, users are assumed to prepare their own training data and computation resources.

它还要求用户自己用 mitmproxy/Wireshark 抓雀魂 WebSocket 牌谱——
**这本身就是「依赖人类打谱」，与我们的铁律冲突**。

---

## 4. 对我们的三条具体改动建议

| # | 建议 | 依据 |
| --- | --- | --- |
| 1 | **参数量锚点定在 2M 左右**，不要一上来 8M/16M | bc49 用 2.00M 到 Elo 1189 |
| 2 | **观测重构成 (C, 34) 二维**，而不是 436 维向量 | 他们的 v3r 是 56×34，用 Conv 才吃得到结构 |
| 3 | **动作槽位改显式分配**，不用哈希 | 46 槽固定表 vs 我们 54 槽哈希（有冲突） |

---

## 5. 没做的事（诚实标注）

- **没有下载任何一个权重**（也下不到，见上）
- **没有跑 Mortal 的代码**，架构结论来自读 `mortal/model.py` 源码
- 他们的 BC 精度 0.8059、ELO 1189 等数字**未独立验证**
- 「Mortal 90 局/秒」是我从「40K 半庄/小时」换算的**推算值**，非官方数字

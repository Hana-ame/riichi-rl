"""双流 Transformer 策略价值网络（轻量收拢版，支持一键扩缩至 4M–8M）。

架构流水线：
1. 牌面 + 场况查询流（Query Stream -> [B, 35, D]）：
   - 数牌流 [B, 27, C_in] -> 重排 [B*3, 9, C_in] -> 共享 1D 残差编码器 + 花色 Emb -> [B, 27, D]
   - 字牌流 [B, 7, C_in]  -> 独立 1×1 / MLP 编码器 -> [B, 7, D]
   - 场况流 [B, S]        -> 投影为 1 个全局场况词元 -> [B, 1, D]
   - 拼接为 Q_attn: [B, 35, D]（前 34 行严格对应 34 种牌，第 35 行为全局枢纽）

2. 统一历史事件流（Key/Value Stream -> [B, T, D]）：
   - events: int64 [B, T, 4]（相对座位, 牌号, 动作类型, 巡目）
   - 4 个离散 Embedding 相加 + 2 层 Pre-LN 因果 Self-Attention -> K, V_attn: [B, T, D]

3. 融合主干（n_layers × [Cross-Attn + Self-Attn + FFN]）：
   - 让每张手牌主动检索全部 T 步牌河/副露/立直事件，再做 35-Token 全局自注意力交互。

4. 统一输出 + 训练期旁路防守头：
   - 前 34 行 -> 共享线性层 -> 常规切牌 logits [B, 34]
   - 第 35 行 -> 特殊动作头 [B, 13]（兼容 47 槽位）+ 独立 Value 价值塔 V(s) [B, 1]
   - 旁路 Aux 防守头（仅 forward_aux 时触发）：前 34 行 -> 危险张概率 logits [B, 34]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mahjong import event_stream as ES
from mahjong import obs2 as OBS
from mahjong.tiles import NUM_TILES

N_PLANES = OBS.N_PLANES
PLANE_SIZE = OBS.PLANE


class _SuitResBlock1D(nn.Module):
    """数牌专用 1D 残差块（仅在长度 9 的同花色内卷积，绝不跨花色污染）。"""

    def __init__(self, dim: int):
        super().__init__()
        groups = 4 if dim % 4 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class _FusionBlock(nn.Module):
    """Pre-LN 融合块：Cross-Attn(Q=牌面35, K/V=事件T) -> Self-Attn(35) -> FFN。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q_cross = nn.LayerNorm(d_model)
        self.norm_kv_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Cross-Attn：每张牌主动检索历史事件流
        qn = self.norm_q_cross(q)
        kvn = self.norm_kv_cross(kv)
        c_out, _ = self.cross_attn(qn, kvn, kvn, key_padding_mask=kv_pad_mask, need_weights=False)
        q = q + c_out

        # 2. Self-Attn：34 张牌 + 1 全局场况词元内部交互
        sn = self.norm_self(q)
        s_out, _ = self.self_attn(sn, sn, sn, need_weights=False)
        q = q + s_out

        # 3. FFN
        q = q + self.ffn(self.norm_ffn(q))
        return q


class DualStreamMahjongNet(nn.Module):
    """轻量收拢版双流 Transformer 网络。

    默认配置 (d_model=128, n_layers=3, n_heads=4) 约 1.15M 参数，适合单卡/CPU 快速验证；
    将 d_model=256, n_layers=6, n_heads=8 即可平滑扩展到 ~6.5M 参数规模。
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        n_actions: int = 47,
        touch_dim: int = 0,
        extra_scalar: int = 0,
        in_planes: int | None = None,
        aux_dim: int = 0,
        max_event_len: int = ES.DEFAULT_EVENT_LEN,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_actions = int(n_actions)
        self.in_planes = int(in_planes) if in_planes else N_PLANES
        self.n_planes = self.in_planes
        self.scalar_dim = OBS.SCALAR_DIM
        self.touch_dim = int(touch_dim)
        self.extra_scalar = int(extra_scalar)
        self.in_scalar = self.scalar_dim + self.touch_dim + self.extra_scalar
        self.aux_dim = int(aux_dim or 0)
        self.max_event_len = int(max_event_len)

        # --- 流 1a：万/筒/索花色共享编码器 ---
        self.suit_in = nn.Conv1d(self.in_planes, d_model, kernel_size=1)
        self.suit_blocks = nn.Sequential(_SuitResBlock1D(d_model), _SuitResBlock1D(d_model))
        self.suit_emb = nn.Parameter(torch.zeros(3, 1, d_model))
        nn.init.normal_(self.suit_emb, std=0.02)

        # --- 流 1b：7 种字牌独立编码器 ---
        self.honor_proj = nn.Sequential(
            nn.Linear(self.in_planes, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- 流 1c：全局场况标量 -> 第 35 号词元 ---
        self.sit_proj = nn.Sequential(
            nn.Linear(self.in_scalar, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 35 词元位置嵌入（保留 0..33 牌种编号与 34 全局词元身份）
        self.pos_emb_35 = nn.Parameter(torch.zeros(1, NUM_TILES + 1, d_model))
        nn.init.normal_(self.pos_emb_35, std=0.02)

        # --- 流 2：统一历史事件流编码器 [B, T, 4] ---
        self.emb_seat = nn.Embedding(ES.SEAT_VOCAB, d_model)
        self.emb_tile = nn.Embedding(ES.TILE_VOCAB, d_model)
        self.emb_act = nn.Embedding(ES.ACT_VOCAB, d_model)
        self.emb_turn = nn.Embedding(ES.TURN_VOCAB, d_model)
        ev_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(ev_layer, num_layers=2)
        self.event_norm = nn.LayerNorm(d_model)

        # --- 融合主干 ---
        self.fusion_layers = nn.ModuleList(
            [_FusionBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)

        # --- 统一输出头 ---
        # 前 34 行逐牌切牌打分（完全共享权重，保留 34 牌空间位置）
        self.discard_head = nn.Linear(d_model, 1)
        # 第 35 行负责全局特殊动作（和牌/立直/吃碰杠/过牌）
        self.special_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_actions - NUM_TILES),
        )
        # 独立 Value 价值塔（Critic，与策略头解耦便于 BC->PPO Warmup）
        self.value_head = nn.Sequential(
            nn.Linear(d_model + self.in_scalar, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # 训练期旁路危险张预测头（推理默认不计算，零额外耗时）
        self.aux_danger_head = nn.Linear(d_model, 1)

    def _encode_query_stream(self, obs: torch.Tensor, touch: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        planes = obs[:, : self.n_planes * PLANE_SIZE].view(B, self.n_planes, PLANE_SIZE)
        scalars = obs[:, self.n_planes * PLANE_SIZE :]
        if touch is not None and self.touch_dim > 0:
            scalars = torch.cat([scalars, touch], dim=1)

        # 1. 万/筒/索折叠进 batch 维共享编码：[B, C, 27] -> [B*3, C, 9]
        suits_raw = planes[:, :, :27].view(B, self.in_planes, 3, 9)
        suits_fold = suits_raw.permute(2, 0, 1, 3).reshape(3 * B, self.in_planes, 9)
        h_suits = self.suit_blocks(self.suit_in(suits_fold))          # (3B, D, 9)
        h_suits = h_suits.view(3, B, self.d_model, 9).permute(1, 0, 3, 2)  # (B, 3, 9, D)
        h_suits = (h_suits + self.suit_emb.unsqueeze(0)).reshape(B, 27, self.d_model)

        # 2. 字牌独立编码：[B, 7, C_in] -> [B, 7, D]
        honors_raw = planes[:, :, 27:34].transpose(1, 2)             # (B, 7, C_in)
        h_honors = self.honor_proj(honors_raw)                       # (B, 7, D)

        # 3. 场况标量 -> 第 35 号全局词元 [B, 1, D]
        h_sit = self.sit_proj(scalars).unsqueeze(1)                  # (B, 1, D)

        q35 = torch.cat([h_suits, h_honors, h_sit], dim=1) + self.pos_emb_35
        return q35, scalars

    def _encode_event_stream(self, events: torch.Tensor | None, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if events is None:
            # 未传入历史事件时自动构造 BOS 单步序列，保持向后兼容
            events = torch.zeros((batch_size, 1, 4), dtype=torch.long, device=device)
            events[:, 0, 0] = 1
            events[:, 0, 1] = 34
            events[:, 0, 3] = 1

        seat = events[:, :, 0].clamp(0, ES.SEAT_VOCAB - 1)
        tile = events[:, :, 1].clamp(0, ES.TILE_VOCAB - 1)
        act = events[:, :, 2].clamp(0, ES.ACT_VOCAB - 1)
        turn = events[:, :, 3].clamp(0, ES.TURN_VOCAB - 1)

        x_ev = (
            self.emb_seat(seat)
            + self.emb_tile(tile)
            + self.emb_act(act)
            + self.emb_turn(turn)
        )
        T_len = x_ev.shape[1]
        causal_mask = torch.triu(
            torch.ones((T_len, T_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        pad_mask = seat == 0
        # 强制第 0 位永不被 mask，彻底防止全 padding 产生 NaN
        if T_len > 0:
            pad_mask = pad_mask.clone()
            pad_mask[:, 0] = False

        kv = self.event_encoder(x_ev, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.event_norm(kv), pad_mask

    def _forward_backbone(
        self,
        obs: torch.Tensor,
        touch: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q35, scalars = self._encode_query_stream(obs, touch)
        kv, pad_mask = self._encode_event_stream(events, obs.shape[0], obs.device)
        for layer in self.fusion_layers:
            q35 = layer(q35, kv, kv_pad_mask=pad_mask)
        return self.out_norm(q35), scalars

    def forward(
        self,
        obs: torch.Tensor,
        touch: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """标准前向推理：返回 (policy_logits [B, 47], value [B])。"""
        h35, scalars = self._forward_backbone(obs, touch, events)
        discard_logits = self.discard_head(h35[:, :NUM_TILES, :]).squeeze(-1)
        special_logits = self.special_head(h35[:, NUM_TILES, :])
        policy_logits = torch.cat([discard_logits, special_logits], dim=-1)
        value = self.value_head(torch.cat([h35[:, NUM_TILES, :], scalars], dim=-1)).squeeze(-1)
        return policy_logits, value

    def forward_aux(
        self,
        obs: torch.Tensor,
        touch: torch.Tensor | None = None,
        events: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练期前向：额外返回危险张辅助监督 logits [B, 34]。"""
        h35, scalars = self._forward_backbone(obs, touch, events)
        discard_logits = self.discard_head(h35[:, :NUM_TILES, :]).squeeze(-1)
        special_logits = self.special_head(h35[:, NUM_TILES, :])
        policy_logits = torch.cat([discard_logits, special_logits], dim=-1)
        value = self.value_head(torch.cat([h35[:, NUM_TILES, :], scalars], dim=-1)).squeeze(-1)
        danger_logits = self.aux_danger_head(h35[:, :NUM_TILES, :]).squeeze(-1)
        return policy_logits, value, danger_logits

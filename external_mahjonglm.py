"""MahjongLM (mitsutani/mahjonglm-10m/100m) 纯 torch 推理。

WordLevel tokenizer（vocab 828，空格分词）；Qwen3 结构（13/16 层、hidden256、
GQA 4Q/1KV、head_dim64、RoPE theta 1e6、SwiGLU、tie embeddings）。
没有 transformers 依赖：自己解 safetensors（BF16→FP32）并实现前向。
"""
from __future__ import annotations
import json, struct, math
import torch, torch.nn as nn, torch.nn.functional as F
from mahjong.tiles import NUM_TILES

TILE_TOKENS = [
    *("m%d" % i for i in range(1, 10)),
    *("p%d" % i for i in range(1, 10)),
    *("s%d" % i for i in range(1, 10)),
    "m0", "p0", "s0",
    "E", "S", "W", "N", "P", "F", "C",
]

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, meta in hdr.items():
        if k == "__metadata__":
            continue
        dt = meta["dtype"]; shape = tuple(meta["shape"])
        off0, off1 = meta["data_offsets"]
        raw = data[off0:off1]
        if dt == "BF16":
            u = torch.frombuffer(bytearray(raw), dtype=torch.int16)
            u = u.to(torch.int32) << 16
            t = u.view(torch.float32).reshape(shape).clone()
        elif dt == "F32":
            t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()
        else:
            raise ValueError(f"unsupported dtype {dt}")
        out[k] = t
    return out


def load_tokenizer(path):
    d = json.load(open(path))
    vocab = d["model"]["vocab"]; inv = {v: k for k, v in vocab.items()}
    def encode(text):
        toks = text.split()
        return [vocab.get(t, 1) for t in toks]
    def decode(ids):
        return " ".join(inv.get(int(i), "<unk>") for i in ids)
    return encode, decode, vocab, inv


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return self.w * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden_size"]; self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]; self.hd = cfg["head_dim"]
        self.q = nn.Linear(h, self.nh * self.hd, bias=False)
        self.k = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.v = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, h, bias=False)
        self.qn = RMSNorm(self.hd); self.kn = RMSNorm(self.hd)
    def forward(self, x, cos, sin, causal):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.nh, self.hd)).transpose(1, 2)   # B,H,T,D
        k = self.kn(self.k(x).view(B, T, self.nkv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k = k.repeat_interleave(self.nh // self.nkv, dim=1)
        v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        if causal:
            m = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            a = a.masked_fill(m, float("-inf"))
        a = a.softmax(-1)
        y = (a @ v).transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o(y)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden_size"]; inter = cfg["intermediate_size"]
        self.gate = nn.Linear(h, inter, bias=False); self.up = nn.Linear(h, inter, bias=False)
        self.down = nn.Linear(inter, h, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"]); self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg["hidden_size"]); self.mlp = MLP(cfg)
    def forward(self, x, cos, sin, causal):
        x = x + self.attn(self.ln1(x), cos, sin, causal)
        x = x + self.mlp(self.ln2(x))
        return x


class MahjongLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.embed.weight  # tie
        hd = cfg["head_dim"]; theta = cfg["rope_parameters"]["rope_theta"]
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("inv_freq", inv)
    def forward(self, ids):
        B, T = ids.shape
        t = torch.arange(T).float()
        freqs = torch.outer(t, self.inv_freq)          # T, D/2
        emb = torch.cat([freqs, freqs], dim=-1)        # T, D
        cos = emb.cos()[None, None, :, :]; sin = emb.sin()[None, None, :, :]
        x = self.embed(ids)
        for blk in self.layers:
            x = blk(x, cos, sin, causal=True)
        return self.lm_head(self.norm(x))
    def load(self, path):
        sd = load_safetensors(path)
        mapping = {"model.embed_tokens.weight": "embed.weight", "model.norm.weight": "norm.w"}
        for k, v in sd.items():
            if not k.startswith("model.layers."):
                continue
            _, _, i, *rest = k.split(".")
            r = ".".join(rest)
            r = r.replace("self_attn.q_norm.weight", "attn.qn.w")
            r = r.replace("self_attn.k_norm.weight", "attn.kn.w")
            r = r.replace("self_attn.q_proj.weight", "attn.q.weight")
            r = r.replace("self_attn.k_proj.weight", "attn.k.weight")
            r = r.replace("self_attn.v_proj.weight", "attn.v.weight")
            r = r.replace("self_attn.o_proj.weight", "attn.o.weight")
            r = r.replace("input_layernorm.weight", "ln1.w")
            r = r.replace("post_attention_layernorm.weight", "ln2.w")
            r = r.replace("mlp.gate_proj.weight", "mlp.gate.weight")
            r = r.replace("mlp.up_proj.weight", "mlp.up.weight")
            r = r.replace("mlp.down_proj.weight", "mlp.down.weight")
            mapping[k] = f"layers.{i}.{r}"
        own = self.state_dict()
        for bk, mk in mapping.items():
            if bk in sd and mk in own:
                with torch.no_grad():
                    own[mk].copy_(sd[bk])
        missing = [k for k in own if k not in mapping.values()
                   and k not in ("inv_freq", "lm_head.weight")]
        if missing:
            raise RuntimeError(f"missing weights: {missing[:5]}")
        self.load_state_dict(own)
        self.lm_head.weight = self.embed.weight
        return self


def load_model(ckpt_dir="external/mahjonglm10"):
    cfg = json.load(open(f"{ckpt_dir}/config.json"))
    import os
    m = MahjongLM(cfg)
    path = f"{ckpt_dir}/model.safetensors"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m.load(path)
    m.eval()
    return m

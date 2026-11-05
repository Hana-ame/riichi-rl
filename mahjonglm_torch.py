"""MahjongLM (mitsutani/mahjonglm-10m/100m) 纯 torch 推理（含 KV cache）。

WordLevel tokenizer（vocab 828，空格分词）；Qwen3 结构：GQA、head_dim=64、
RoPE(default, theta=1e6)、SwiGLU、tie embeddings、q/k RMSNorm。
不依赖 transformers：自行解析 safetensors（BF16→FP32）并实现前向。
"""
from __future__ import annotations
import json, struct, math, os
import torch, torch.nn as nn, torch.nn.functional as F


# ---------------- safetensors ----------------
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
            u = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.int32) << 16
            t = u.view(torch.float32).reshape(shape).clone()
        elif dt == "F32":
            t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()
        elif dt == "F16":
            t = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape).to(torch.float32)
        else:
            raise ValueError(f"unsupported dtype {dt}")
        out[k] = t
    return out


def load_tokenizer(path):
    d = json.load(open(path))
    vocab = d["model"]["vocab"]; inv = {v: k for k, v in vocab.items()}

    def encode(text):
        return [vocab.get(t, 1) for t in text.split()]

    def decode(ids):
        return " ".join(inv.get(int(i), "<unk>") for i in ids)

    return encode, decode, vocab, inv


# ---------------- model ----------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(dim)); self.eps = eps

    def forward(self, x):
        return self.w * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


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

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.qn(self.q(x).view(B, T, self.nh, self.hd)).transpose(1, 2)
        k = self.kn(self.k(x).view(B, T, self.nkv, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        kk = k.repeat_interleave(self.nh // self.nkv, dim=1)
        vv = v.repeat_interleave(self.nh // self.nkv, dim=1)
        a = (q @ kk.transpose(-1, -2)) / math.sqrt(self.hd)
        Tall = kk.shape[2]
        if T > 1 or cache is not None:
            pos = torch.arange(Tall - T, Tall)
            mask = torch.arange(Tall)[None, :] > pos[:, None]
            a = a.masked_fill(mask, float("-inf"))
        a = a.softmax(-1)
        y = (a @ vv).transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o(y), (k, v)


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

    def forward(self, x, cos, sin, cache=None):
        y, c = self.attn(self.ln1(x), cos, sin, cache)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, c


class MahjongLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.embed.weight
        hd = cfg["head_dim"]
        theta = (cfg.get("rope_parameters") or {}).get("rope_theta", 1000000.0)
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("inv_freq", inv)

    def forward(self, ids, cache=None):
        """ids: (B,T)。cache: list[tuple(k,v)] 每层。返回 logits (B,T,V) 与新 cache。"""
        B, T = ids.shape
        pos0 = cache[0][0].shape[2] if cache else 0
        t = torch.arange(pos0, pos0 + T).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]; sin = emb.sin()[None, None, :, :]
        x = self.embed(ids)
        new_cache = []
        for i, blk in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            x, nc = blk(x, cos, sin, c)
            new_cache.append(nc)
        logits = self.lm_head(self.norm(x))
        return logits, new_cache

    def load(self, path):
        sd = load_safetensors(path)
        mapping = {"model.embed_tokens.weight": "embed.weight", "model.norm.weight": "norm.w"}
        for k in sd:
            if not k.startswith("model.layers."):
                continue
            parts = k.split(".")
            i = parts[2]; r = ".".join(parts[3:])
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
        hit = 0
        for bk, mk in mapping.items():
            if bk in sd and mk in own:
                with torch.no_grad():
                    own[mk].copy_(sd[bk])
                hit += 1
        missing = [k for k in own if k not in mapping.values()
                   and k not in ("inv_freq", "lm_head.weight")]
        if missing:
            raise RuntimeError(f"missing weights: {missing[:5]}")
        self.lm_head.weight = self.embed.weight
        return self


def load_model(ckpt_dir="external/mahjonglm10"):
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    m = MahjongLM(cfg)
    m.load(os.path.join(ckpt_dir, "model.safetensors"))
    m.eval()
    return m


@torch.no_grad()
def generate(model, ids, max_new=200, topk=1, temperature=1.0):
    """ids: list[int] 前缀；返回新 token id 列表。"""
    logits, cache = model(torch.tensor([ids]))
    out = []
    cur = logits[0, -1]
    for _ in range(max_new):
        if topk and topk > 1:
            v, idx = torch.topk(cur / max(1e-6, temperature), topk)
            p = torch.softmax(v, -1)
            j = int(torch.multinomial(p, 1))
            nxt = int(idx[j])
        else:
            nxt = int(cur.argmax())
        out.append(nxt)
        logits, cache = model(torch.tensor([[nxt]]), cache)
        cur = logits[0, -1]
    return out

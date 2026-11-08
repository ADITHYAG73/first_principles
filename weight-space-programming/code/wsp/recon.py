"""Local (CPU) re-computation of Qwen3.5 full-attention layers from recorded residual streams.

Follows transformers 5.x `modeling_qwen3_5.py` (Qwen3_5Attention / Qwen3_5RMSNorm / Qwen3_5MLP):
  * RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)          (zero-centred weight)
  * q_proj -> per head [query(256) | gate(256)]; q_norm/k_norm per head; partial RoPE on the first 64 dims
    (partial_rotary_factor 0.25, theta 1e7; text-only so the 3 mrope sections share the same positions)
  * causal softmax(q k^T / 16) v, GQA 8 query heads on 2 KV heads, output * sigmoid(gate), then o_proj.
Used only for analysis: inputs are the residual streams returned by a teacher-forced pass, and the
reconstruction is checked against the next recorded residual (see `check_layer`).
"""
from __future__ import annotations

import torch

EPS = 1e-6
N_HEADS, N_KV, HD, ROT = 8, 2, 256, 64
THETA = 10_000_000.0
P = "model.language_model.layers.{}."


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.float())


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x: (heads, T, 256); rotate the first ROT dims (rotate_half convention)."""
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float) / ROT))
    f = pos[:, None].float() * inv[None]
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
    xr, xp = x[..., :ROT], x[..., ROT:]
    x1, x2 = xr[..., : ROT // 2], xr[..., ROT // 2:]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, xp], -1)


class AttnLayer:
    def __init__(self, load, L: int):
        g = lambda n: load(P.format(L) + n).float()
        self.L = L
        self.ln = g("input_layernorm.weight")
        self.q, self.k, self.v, self.o = (g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                          g("self_attn.v_proj.weight"), g("self_attn.o_proj.weight"))
        self.qn, self.kn = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")
        self.pln = g("post_attention_layernorm.weight")
        self.gate_w, self.up_w, self.down_w = g("mlp.gate_proj.weight"), g("mlp.up_proj.weight"), g("mlp.down_proj.weight")

    def qkv(self, r: torch.Tensor):
        """r: (T, 1024) residual entering the layer. Returns q (8,T,256), k,v (2,T,256), gate (T,8,256)."""
        T = r.shape[0]
        x = rms(r, self.ln)
        qg = (x @ self.q.T).view(T, N_HEADS, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:]
        q = rms(q, self.qn).transpose(0, 1)
        k = rms((x @ self.k.T).view(T, N_KV, HD), self.kn).transpose(0, 1)
        v = (x @ self.v.T).view(T, N_KV, HD).transpose(0, 1)
        pos = torch.arange(T)
        return rope(q, pos), rope(k, pos), v, gate

    def heads(self, r: torch.Tensor, keep_attn_rows: torch.Tensor | None = None, chunk: int = 1024):
        """Per-head gated outputs z (T,8,256) and per-head residual contributions c (T,8,1024).
        keep_attn_rows: optional 1-D LongTensor of query positions whose attention rows (8, len, T) are returned."""
        q, k, v, gate = self.qkv(r)
        T = r.shape[0]
        z = torch.empty(T, N_HEADS, HD)
        rows = None if keep_attn_rows is None else torch.empty(N_HEADS, len(keep_attn_rows), T)
        for h in range(N_HEADS):
            kv = h // (N_HEADS // N_KV)
            for a in range(0, T, chunk):
                s = q[h, a:a + chunk] @ k[kv].T / HD ** 0.5          # (c, T)
                idx = torch.arange(a, min(a + chunk, T))
                s.masked_fill_(torch.arange(T)[None] > idx[:, None], float("-inf"))
                p = s.softmax(-1)
                z[a:a + chunk, h] = p @ v[kv]
                if rows is not None:
                    sel = (keep_attn_rows >= a) & (keep_attn_rows < a + chunk)
                    if sel.any():
                        rows[h, sel] = p[keep_attn_rows[sel] - a]
        z = z * torch.sigmoid(gate)
        c = torch.einsum("thd,ohd->tho", z, self.o.view(1024, N_HEADS, HD))
        return z, c, rows

    def mlp(self, h: torch.Tensor) -> torch.Tensor:
        x = rms(h, self.pln)
        return (torch.nn.functional.silu(x @ self.gate_w.T) * (x @ self.up_w.T)) @ self.down_w.T


def check_layer(layer: AttnLayer, r_in: torch.Tensor, r_out: torch.Tensor) -> dict:
    """Reconstruct the whole decoder layer and compare with the recorded output residual."""
    _, c, _ = layer.heads(r_in)
    h = r_in.float() + c.sum(1)
    pred = h + layer.mlp(h)
    err = (pred - r_out.float()).norm(dim=-1) / (r_out.float() - r_in.float()).norm(dim=-1).clamp_min(1e-6)
    return dict(rel_err_median=float(err.median()), rel_err_p95=float(err.quantile(0.95)))

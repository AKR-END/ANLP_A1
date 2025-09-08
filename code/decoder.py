from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope(q: torch.Tensor, k: torch.Tensor):
    Dh = q.size(-1)
    half = Dh // 2
    sinusoid = torch.arange(half, device=q.device).float()
    theta = 10000 ** (-2 * sinusoid / Dh)
    T = q.size(-2)
    pos = torch.arange(T, device=q.device).float().unsqueeze(1)
    angles = pos * theta.unsqueeze(0)
    sin = angles.sin()[None,None,:,:]
    cos = angles.cos()[None,None,:,:]
    def rope(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rope(q), rope(k)

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Embedding(2*max_distance+1, num_heads)
    def forward(self, qlen:int, klen:int, device=None):
        ctx = torch.arange(qlen, device=device)[:, None] - torch.arange(klen, device=device)[None, :]
        ctx = ctx.clamp(-self.max_distance, self.max_distance) + self.max_distance
        out = self.bias(ctx).permute(2,0,1).unsqueeze(0)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: bool = False, rel_bias: bool = False, max_distance: int = 128, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope
        self.rel_bias = rel_bias

        # Separate projections for Q, K, V (no concat on time axis)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer("mask_cache", torch.empty(0), persistent=False)
        if rel_bias:
            self.rel = RelativePositionBias(num_heads=num_heads, max_distance=max_distance)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        # x_q: (B, Tq, C), x_kv: (B, Tk, C)
        B, Tq, _ = x_q.size()
        _, Tk, _ = x_kv.size()

        q = self.q_proj(x_q)  # (B, Tq, C)
        k = self.k_proj(x_kv) # (B, Tk, C)
        v = self.v_proj(x_kv) # (B, Tk, C)

        # reshape to heads
        q = q.view(B, Tq, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, Tq, Dh)
        k = k.view(B, Tk, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, Tk, Dh)
        v = v.view(B, Tk, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, Tk, Dh)

        # optional RoPE
        if self.rope:
            q, k = apply_rope(q, k)

        # attention scores
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,Tq,Tk)

        # optional relative position bias
        if self.rel_bias:
            att = att + self.rel(Tq, Tk, device=att.device)  # (1,H,Tq,Tk) broadcast

        # optional causal mask (for decoder self-attn)
        if is_causal:
            if self.mask_cache.size(0) != Tq or self.mask_cache.size(1) != Tk:
                if Tk == Tq:
                    m = torch.triu(torch.ones(Tq, Tk, dtype=torch.bool, device=att.device), diagonal=1)
                else:
                    m = torch.ones(Tq, Tk, dtype=torch.bool, device=att.device).triu(diagonal=1 + Tk - Tq)
                self.mask_cache = m
            att = att.masked_fill(self.mask_cache, float('-inf'))

        # additive padding masks (broadcastable to B,1,Tq,Tk)
        if mask is not None:
            att = att + mask

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = torch.matmul(att, v)  # (B,H,Tq,Dh)
        y = y.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        y = self.proj_drop(self.out(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.drop(F.relu(self.w1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, rope=False, rel_bias=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, rope=rope, rel_bias=rel_bias, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, rope=False, rel_bias=rel_bias, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
    def forward(self, x, mem, tgt_mask=None, mem_mask=None):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), mask=tgt_mask, is_causal=True)
        x = x + self.cross_attn(self.ln2(x), mem, mask=mem_mask, is_causal=False)
        x = x + self.ff(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, rope=False, rel_bias=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, rope, rel_bias) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, y, mem, tgt_mask=None, mem_mask=None):
        h = self.emb(y)
        for layer in self.layers:
            h = layer(h, mem, tgt_mask=tgt_mask, mem_mask=mem_mask)
        h = self.ln(h)
        return self.proj(h)
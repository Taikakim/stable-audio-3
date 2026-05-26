import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        # t: [seq_len]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

def apply_rotary_emb(x, cos, sin):
    # x: [B, seq_len, num_heads, head_dim]
    b, s, h, d = x.shape
    d_half = d // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    x_rotated = torch.cat((-x2, x1), dim=-1)

    # Pre-reshape cos and sin to broadcast over num_heads
    # cos, sin are [seq_len, dim] -> [1, seq_len, 1, dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (x_rotated * sin)

class LatCHAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, T, num_heads, head_dim]

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # PyTorch scaled dot-product attention
        q = q.transpose(1, 2) # [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

class LatCHBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LatCHAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim)
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        # t: [B]
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)

class LatCH(nn.Module):
    """
    Latent-Control Head (LatCH) Model
    Bidirectional Transformer operating on latents, predicting control features.
    Approximately 7M parameters:
    For dim=256, depth=6, num_heads=8:
    Each block: 256*256*4 + 256*1024*2 = 262K + 524K = 786K.
    6 blocks = 4.7M.
    Total with embeddings ~5-7M.
    """
    def __init__(
        self,
        in_channels=256,
        out_channels=1,
        dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.depth = depth

        self.latent_proj = nn.Linear(in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)

        self.blocks = nn.ModuleList([
            LatCHBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_channels)
        self.rotary_emb = RotaryEmbedding(dim // num_heads)

    def forward(self, x, t):
        """
        x: [B, in_channels, T] (noisy latents)
        t: [B] (timesteps)
        Returns: [B, out_channels, T]
        """
        B, C, T_seq = x.shape
        # Move channels to last dimension: [B, T, C]
        x = x.transpose(1, 2)

        # Project inputs
        x = self.latent_proj(x)

        # Add timestep embedding
        t_emb = self.t_embedder(t).unsqueeze(1) # [B, 1, dim]
        x = torch.cat([t_emb, x], dim=1) # Sequence concatenation: [B, T+1, dim]

        # Get rotary embeddings for sequence length
        pos = torch.arange(T_seq + 1, device=x.device).float()
        cos, sin = self.rotary_emb(pos)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm_final(x)

        # Strip timestep token
        x = x[:, 1:, :]

        out = self.out_proj(x) # [B, T_seq, out_channels]
        return out.transpose(1, 2) # [B, out_channels, T_seq]

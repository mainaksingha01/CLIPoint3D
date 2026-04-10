import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_kv):
            x_kv = self.norm_context(x_kv)
        else:
            x_kv = x_q

        return self.fn(x_q, x_kv, x_kv, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            latent_dim,
            kv_dim,
            cross_heads=4,
            seq_dropout_prob=0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim,
                    nn.MultiheadAttention(latent_dim, num_heads=cross_heads, kdim=kv_dim, vdim=kv_dim,
                                          dropout=seq_dropout_prob, batch_first=True),
                    context_dim=kv_dim),
            FeedForward(latent_dim)])

    def forward(
            self,
            data,
            soft_prompt,
            mask=None,
    ):
        b, *_, device = *data.shape, data.device
        # x = repeat(soft_prompt, 'n d -> b n d', b=b)
        x = soft_prompt
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x)+x

        return x


class SelfAttention(nn.Module):
    def __init__(
            self,
            depth,
            latent_dim,
            latent_heads=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, nn.MultiheadAttention(latent_dim, num_heads=latent_heads, batch_first=True)),
                FeedForward(latent_dim)
            ]))

    def forward(
            self,
            x,
            mask=None
    ):
        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x, key_padding_mask=mask)[0] + x
            x = self_ff(x) + x
        return x

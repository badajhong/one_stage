"""Small depth CNN and finite causal-window Transformer for FlashSAC.

Frames are encoded once per sampled chunk. Independent windows prevent information
outside the configured history (or across an episode boundary) from reaching a
prediction, even with multiple attention layers. Only supervised losses train it.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..ppo.common import FlattenBatch, make_mlp
from .one_stage_replay import encode_depth, decode_depth


class CausalWindowBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim):
        super().__init__()
        self.num_heads = num_heads
        self.norm_attention = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.projection = nn.Linear(model_dim, model_dim)
        self.norm_ffn = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(nn.Linear(model_dim, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, model_dim))

    def forward(self, x, valid, *, last_only=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm_attention(x)).reshape(
            batch, length, 3, self.num_heads, width // self.num_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        if last_only:
            q = q[:, :, -1:]
            # Every valid key precedes this sole query. Give an invalid query a
            # finite self-attention fallback; the final prediction is zeroed.
            self_key = torch.arange(length, device=x.device) == length - 1
            allowed = valid | (~valid[:, -1:] & self_key)
            mask = allowed[:, None, None, :]
            x = x[:, -1:]
        else:
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            diagonal = torch.eye(length, dtype=torch.bool, device=x.device)
            mask = (valid[:, None, :] & causal) | (~valid[:, :, None] & diagonal)
            mask = mask[:, None]
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        x = x + self.projection(attention.transpose(1, 2).reshape(batch, x.shape[1], width))
        return x + self.ffn(self.norm_ffn(x))


class WindowTransformerPerception(nn.Module):
    def __init__(self, context_dim, depth_shape, object_dim, geometry,
                 latent_dim=256, window_size=16, model_dim=128, num_layers=1,
                 num_heads=4, ffn_dim=256, depth_feature_dim=64):
        super().__init__()
        if min(window_size, model_dim, num_layers, num_heads, ffn_dim) < 1:
            raise ValueError("Transformer dimensions and window_size must be positive")
        if model_dim % num_heads:
            raise ValueError("perception_model_dim must be divisible by perception_num_heads")
        self.window_size = window_size
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.depth_shape = tuple(depth_shape)
        self.context_dim = context_dim
        self.register_buffer("geometry", geometry.detach().float().flatten().clone())
        self.depth_cnn = nn.Sequential(
            FlattenBatch(nn.Sequential(
                nn.Conv2d(depth_shape[0], 8, 5, stride=4, padding=2), nn.Mish(),
                nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.Mish(), nn.Flatten(),
            ), data_dim=3),
            nn.LazyLinear(depth_feature_dim), nn.LayerNorm(depth_feature_dim),
        )
        self.visual = nn.Linear(depth_feature_dim, 256)
        self.context = make_mlp([256, 256])
        self.token = nn.Linear(512, model_dim)
        self.position = nn.Parameter(torch.randn(1, window_size, model_dim) * .02)
        self.blocks = nn.ModuleList([
            CausalWindowBlock(model_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)
        self.object_head = nn.Linear(model_dim, object_dim)
        self.priv_head = nn.Linear(model_dim, latent_dim)
        self.to(geometry.device)
        with torch.no_grad():
            self.encode(torch.zeros(2, *depth_shape, device=geometry.device),
                        torch.zeros(2, context_dim, device=geometry.device))

    def encode(self, depth, context):
        geometry = self.geometry.expand(*context.shape[:-1], -1)
        p = self.context(torch.cat([context, geometry], -1))
        v = self.visual(self.depth_cnn(depth))
        return self.token(torch.cat([p, v], -1))

    @staticmethod
    def window_valid(is_init, valid):
        """Only valid frames in the final query's episode may supply keys."""
        episode = (is_init.bool() | ~valid.bool()).long().cumsum(-1)
        return valid.bool() & (episode == episode[..., -1:])

    def window(self, tokens, valid):
        """Predict the final frame from [batch,L,D] independent windows."""
        if tokens.shape[1] != self.window_size:
            raise ValueError("perception window has the wrong length")
        x = torch.where(valid.unsqueeze(-1), tokens, 0.) + self.position
        for index, block in enumerate(self.blocks):
            x = block(x, valid, last_only=index == len(self.blocks) - 1)
        x = self.norm(x[:, -1])
        current_valid = valid[:, -1:]
        return (torch.where(current_valid, self.priv_head(x), 0.),
                torch.where(current_valid, self.object_head(x), 0.))

    def forward(self, depth, context, is_init, valid):
        return self.window(self.encode(depth, context), self.window_valid(is_init, valid))

    def sequence_from_tokens(self, tokens, is_init, valid):
        """Return predictions for indices L-1..T-1, with exact L-frame context."""
        length = self.window_size
        if tokens.shape[1] < length:
            raise ValueError("sequence must contain at least one full perception window")
        windows = tokens.unfold(1, length, 1).permute(0, 1, 3, 2)
        batch, count, _, width = windows.shape
        # Compute episode IDs once, before extracting overlapping windows.
        episode = (is_init.bool() | ~valid.bool()).long().cumsum(-1)
        episodes = episode.unfold(1, length, 1)
        keep = valid.unfold(1, length, 1) & (episodes == episodes[..., -1:])
        z, objects = self.window(windows.reshape(batch * count, length, width),
                                 keep.reshape(batch * count, length))
        return z.reshape(batch, count, -1), objects.reshape(batch, count, -1)

    def sequence(self, depth, context, is_init, valid):
        return self.sequence_from_tokens(self.encode(depth, context), is_init, valid)


class WindowHistory:
    """Online raw-frame ring with versioned CNN tokens, never recurrent state.

    A perception optimizer step invalidates all tokens. The next rollout rebuilds
    them from retained raw frames, so a window always uses one set of weights.
    """
    def __init__(self, perception, batch_size, num_depth_bins):
        device = perception.geometry.device
        length = perception.window_size
        self.num_depth_bins = num_depth_bins
        self.depth = torch.zeros(batch_size, length, *perception.depth_shape,
                                 dtype=torch.uint8, device=device)
        self.context = torch.zeros(batch_size, length, perception.context_dim, device=device)
        self.tokens = torch.zeros(batch_size, length, perception.model_dim, device=device)
        self.valid = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        offsets = torch.arange(length, device=device)
        self.orders = (offsets[:, None] + 1 + offsets[None]) % length
        self.cursor = 0
        self.version = -1

    @torch.no_grad()
    def step(self, perception, depth, context, is_init, valid, version):
        reset = is_init.reshape(-1).bool() | ~valid.reshape(-1).bool()
        self.valid &= ~reset[:, None]
        index = self.cursor
        self.depth[:, index].copy_(encode_depth(depth, self.num_depth_bins))
        self.context[:, index].copy_(context)
        self.valid[:, index].copy_(valid.reshape(-1))
        if self.version != version:
            self.tokens.copy_(perception.encode(decode_depth(self.depth, self.num_depth_bins), self.context))
            self.version = version
        else:
            self.tokens[:, index].copy_(perception.encode(
                decode_depth(self.depth[:, index], self.num_depth_bins), self.context[:, index]))
        order = self.orders[index]
        self.cursor = (index + 1) % perception.window_size
        return perception.window(self.tokens.index_select(1, order), self.valid.index_select(1, order))

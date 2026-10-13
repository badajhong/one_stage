"""Gated visual/proprioceptive belief used by the joint FlashSAC student.

Only the supervised latent/object objective trains this module. The actor consumes
detached predictions. Object geometry is the fixed object-frame template, never a
point cloud transformed by a ground-truth pose.
"""

import torch
from torch import nn

from ..ppo.common import make_conv, make_mlp


class GatedBeliefPerception(nn.Module):
    def __init__(self, context_dim, depth_shape, object_dim, geometry,
                 latent_dim=256, hidden_dim=256, depth_feature_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.depth_shape = tuple(depth_shape)
        self.register_buffer("geometry", geometry.detach().float().flatten().clone())
        self.depth_cnn = nn.Sequential(
            make_conv([8, 8, 8], activation=nn.Mish, kernel_sizes=5),
            nn.LazyLinear(depth_feature_dim), nn.LayerNorm(depth_feature_dim),
        )
        self.visual = nn.Linear(depth_feature_dim, hidden_dim)
        self.context = make_mlp([hidden_dim, hidden_dim])
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.object_head = nn.Linear(hidden_dim, object_dim)
        self.priv_head = nn.Linear(hidden_dim, latent_dim)
        # Begin near an open/closed balance; avoid starving the visual branch.
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.zeros_(self.gate.bias)
        self.to(geometry.device)
        with torch.no_grad():
            self.encode(torch.zeros(2, *depth_shape, device=geometry.device),
                        torch.zeros(2, context_dim, device=geometry.device))

    def encode(self, depth, context):
        """Batch the image/context encoders independently of the recurrent loop."""
        geometry = self.geometry.expand(*context.shape[:-1], -1)
        p = self.context(torch.cat([context, geometry], -1))
        v = self.visual(self.depth_cnn(depth))
        return p, v

    def step(self, p, v, hidden, is_init, valid=None):
        # The previous episode must not affect even the visual gate.
        reset = is_init.reshape(*hidden.shape[:-1], 1).bool()
        if valid is not None:
            reset = reset | ~valid.reshape_as(reset).bool()
        hidden = torch.where(reset, torch.zeros_like(hidden), hidden)
        gate = torch.sigmoid(self.gate(torch.cat([p, v, hidden], -1)))
        hidden = self.gru(torch.cat([p, gate * v], -1), hidden)
        if valid is not None:
            hidden = torch.where(valid.reshape_as(reset).bool(), hidden, 0.)
        return self.priv_head(hidden), self.object_head(hidden), hidden, gate

    def forward(self, depth, context, hidden, is_init, valid=None):
        return self.step(*self.encode(depth, context), hidden, is_init, valid)

    def sequence(self, depth, context, hidden, is_init, valid, learning_start=None):
        """Unroll [batch,time,...], optionally truncating gradients at each burn-in.

        learning_start=None computes the whole sequence in the caller's grad mode.
        Otherwise the common prefix is encoded without a backward graph, and each
        row's hidden is detached until its own learning window begins. A saved
        old-weight hidden followed by burn-in remains an approximation.
        """
        prefix = 0 if learning_start is None else int(learning_start.min())
        contexts, visuals = [], []
        if prefix:
            with torch.no_grad():
                p, v = self.encode(depth[:, :prefix], context[:, :prefix])
            contexts.append(p)
            visuals.append(v)
        if prefix < depth.shape[1]:
            p, v = self.encode(depth[:, prefix:], context[:, prefix:])
            contexts.append(p)
            visuals.append(v)
        p, v = torch.cat(contexts, 1), torch.cat(visuals, 1)
        latents, objects, states, gates = [], [], [], []
        for t in range(depth.shape[1]):
            if t < prefix:
                with torch.no_grad():
                    z, obj, hidden, gate = self.step(p[:, t], v[:, t], hidden,
                                                    is_init[:, t], valid[:, t])
            else:
                if learning_start is not None:
                    learning = (t >= learning_start).unsqueeze(-1)
                    hidden = torch.where(learning, hidden, hidden.detach())
                    pt = torch.where(learning, p[:, t], p[:, t].detach())
                    vt = torch.where(learning, v[:, t], v[:, t].detach())
                else:
                    learning, pt, vt = None, p[:, t], v[:, t]
                z, obj, hidden, gate = self.step(pt, vt, hidden,
                                                is_init[:, t], valid[:, t])
                if learning is not None:
                    hidden = torch.where(learning, hidden, hidden.detach())
            latents.append(z)
            objects.append(obj)
            states.append(hidden)
            gates.append(gate)
        return tuple(torch.stack(values, 1) for values in (latents, objects, states, gates))

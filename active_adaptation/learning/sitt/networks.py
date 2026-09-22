"""Training-only proxy; original VAIC perception modules are reused unchanged."""

import torch
from torch import nn
from tensordict.nn import TensorDictModuleBase

from ..ppo.common import make_mlp


class ProxyStudent(TensorDictModuleBase):
    """Predict the real student's latent from privileged observations.

    It has exactly the privileged MLP's input contract and layer structure, but
    independently initialized parameters. It is not a copy of the student's
    CNN/object predictor/GRU. GT objects are inputs; the target is the real
    student's latent and decoded mean, including its current perception errors.
    There is no object MSE here. The shared action decoder is frozen during
    proxy alignment. This encoder has no depth input or recurrent state.
    """

    def __init__(self, in_keys, latent_dim):
        super().__init__()
        self.in_keys = list(in_keys)
        self.out_keys = ["proxy_feature"]
        # Keep this identical to PPOVEL.encoder_priv's MLP. Do not share or
        # copy its weights: teacher and proxy must remain separate encoders.
        self.net = nn.Sequential(make_mlp([latent_dim]), nn.LazyLinear(latent_dim))

    def forward(self, td):
        inputs = [td[key].flatten(start_dim=td.ndim) for key in self.in_keys]
        td["proxy_feature"] = self.net(torch.cat(inputs, dim=-1))
        return td

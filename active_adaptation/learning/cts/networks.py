"""CTS inference paths and the shared latent-conditioned critic."""

import weakref

import torch.nn.functional as F
from tensordict.nn import TensorDictModuleBase

from ..modules.rnn import recurrent_mode


class StudentRollout(TensorDictModuleBase):
    """Self-contained deployable modules: no teacher, critic, or owner reference.

Can be copied, moved to CPU or serialized separately from the training policy.
Inputs must use the observation normalizer saved with the checkpoint.
    """

    def __init__(self, policy):
        super().__init__()
        self.in_keys = ["policy", "vel_command", "depth", "object_geo_",
                        "is_init", "adapt_hx", "depth_hx"]
        self.out_keys = ["action", "sample_log_prob", "loc", "scale", "cts_latent",
                         ("next", "adapt_hx"), ("next", "depth_hx")]
        self.temporal_depth_gru = policy.temporal_depth_gru
        self.object_adapt = policy.object_adapt
        self.object_pred_transform = policy.object_pred_transform
        self.adapt_module = policy.adapt_module
        self.shared_actor = policy.shared_actor
        self.normalize_latent = policy.cfg.cts.normalize_latent

    def forward(self, td):
        if not recurrent_mode():
            for key in ("adapt_hx", "depth_hx"):
                td[key] = td[key] * (~td["is_init"]).to(td[key].dtype)
        self.temporal_depth_gru(td)
        self.object_adapt(td)
        self.object_pred_transform(td)
        self.adapt_module(td)
        if self.normalize_latent:
            td["priv_pred"] = F.normalize(td["priv_pred"], p=2, dim=-1, eps=1e-6)
        td["cts_latent"] = td["priv_pred"]
        return self.shared_actor(td)


class LatentCritic(TensorDictModuleBase):
    """Value regression trains the critic only, never either encoder."""

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.in_keys = list(network.in_keys)
        self.out_keys = ["state_value"]

    def forward(self, td):
        local = td.copy()
        local["cts_latent"] = td["cts_latent"].detach()
        td["state_value"] = self.network(local)["state_value"]
        return td


class CTSRollout(TensorDictModuleBase):
    """Fixed training groups; explicit student/teacher modes use every env."""

    def __init__(self, policy, role):
        super().__init__()
        self._owner = weakref.ref(policy)
        self.role = role
        if role == "student":
            self.in_keys = ["policy", "vel_command", "depth", "object_geo_",
                            "is_init", "adapt_hx", "depth_hx"]
        else:
            self.in_keys = list(policy.observation_spec.keys()) + [
                "is_init", "adapt_hx", "depth_hx"]
        self.out_keys = ["action", "sample_log_prob", "loc", "scale", "cts_latent",
                         ("next", "adapt_hx"), ("next", "depth_hx")]

    def forward(self, td):
        policy = self._owner()
        if policy is None:
            raise RuntimeError("CTS rollout requires its owning policy to remain alive.")
        encoded = policy.encode_role(td.copy(), self.role)
        policy.shared_actor(encoded)
        for key in self.out_keys:
            td[key] = encoded[key]
        return td

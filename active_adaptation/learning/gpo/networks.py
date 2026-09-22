"""Guider collection, learner deployment, and detached-latent value regression."""

import weakref

import torch.nn.functional as F
from tensordict.nn import TensorDictModuleBase

from ..modules.rnn import recurrent_mode


class LearnerRollout(TensorDictModuleBase):
    """Self-contained CTS student modules; normalized observations required."""

    def __init__(self, policy):
        super().__init__()
        self.in_keys = ["policy", "vel_command", "depth", "object_geo_",
                        "is_init", "adapt_hx", "depth_hx"]
        self.out_keys = ["action", "sample_log_prob", "loc", "scale", "gpo_latent",
                         ("next", "adapt_hx"), ("next", "depth_hx")]
        self.temporal_depth_gru = policy.temporal_depth_gru
        self.object_adapt = policy.object_adapt
        self.object_pred_transform = policy.object_pred_transform
        self.adapt_module = policy.adapt_module
        self.shared_actor = policy.shared_actor
        self.normalize_latent = policy.cfg.gpo.normalize_latent

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
        td["gpo_latent"] = td["priv_pred"]
        return self.shared_actor(td)


class GuiderRollout(TensorDictModuleBase):
    """Only guider actions are executed; carry the learner history alongside."""

    def __init__(self, policy):
        super().__init__()
        self._owner = weakref.ref(policy)
        self.in_keys = list(policy.observation_spec.keys()) + ["is_init", "adapt_hx", "depth_hx"]
        self.out_keys = ["action", "sample_log_prob", "loc", "scale", "gpo_latent",
                         ("next", "adapt_hx"), ("next", "depth_hx")]

    def forward(self, td):
        policy = self._owner()
        if policy is None:
            raise RuntimeError("Guider rollout requires its owning policy.")
        learner = policy.encode_learner(td.copy())
        guider = policy.encode_guider(td.copy())
        policy.shared_actor(guider)
        for key in ("action", "sample_log_prob", "loc", "scale", "gpo_latent"):
            td[key] = guider[key]
        for key in ("adapt_hx", "depth_hx"):
            td["next", key] = learner["next", key]
        return td


class LatentCritic(TensorDictModuleBase):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.in_keys = list(network.in_keys)
        self.out_keys = ["state_value"]

    def forward(self, td):
        local = td.copy()
        local["gpo_latent"] = td["gpo_latent"].detach()
        td["state_value"] = self.network(local)["state_value"]
        return td

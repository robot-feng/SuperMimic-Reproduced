"""Research-only imagined step: noise in dynamics, clean command in history/reward."""
from __future__ import annotations

import torch

from supermimic.imagination.compiled import REWARD_TERM_NAMES
from supermimic.world_model.heads import analytic_observation


class NoisyImagStep:
    def __init__(self, wm, default_joint_pos, executed_noise):
        """executed_noise: fixed samples [horizon, batch, action_dim]."""
        self.wm = wm
        self.q0 = default_joint_pos
        self.noise = executed_noise
        self.index = 0

    def __call__(self, state, clean, prev_clean, xi, ref):
        if self.index >= len(self.noise) or self.noise[self.index].shape != clean.shape:
            raise ValueError('noise tape does not match the imagined trajectory')
        executed = clean + self.noise[self.index]
        self.index += 1
        s_next = self.wm.step(state, executed, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        terms = self.wm.heads.reward(s_next, clean, prev_clean, ref, bodies)
        logits = self.wm.termination_logits(s_next, ref)
        observation = analytic_observation(s_next, clean, xi, self.q0)
        ai = self.wm.heads.anchor_index
        anchor_pose = torch.cat([bodies['pos'][..., ai, :], bodies['quat'][..., ai, :]], dim=-1)
        return s_next, terms['total'], torch.stack([terms[k] for k in REWARD_TERM_NAMES]), logits, observation, anchor_pose

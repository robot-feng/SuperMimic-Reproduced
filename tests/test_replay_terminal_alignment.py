"""Termination labels must follow the transition that produced the next state."""

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer


@pytest.mark.parametrize("terminated", [False, True])
def test_window_done_matches_current_transition(monkeypatch, terminated):
    buffer = ReplayBuffer(num_envs=1, capacity=8, state_dim=1, act_dim=1, xi_dim=1, device="cpu")
    ids = torch.tensor([0])
    for step in range(4):
        terminal_row = step == 3
        transition_done = bool(terminated and step in (2, 3))
        buffer.add(ids, s=torch.tensor([[float(step)]]), a=torch.zeros(1, 1), a_exec=torch.zeros(1, 1),
                   xi=torch.zeros(1, 1), r=torch.zeros(1), done=torch.tensor([float(transition_done)]),
                   frame=torch.tensor([step]), clip=torch.tensor([0]), terminal=torch.tensor([terminal_row]),
                   ep_start=torch.tensor([step == 0]))
    buffer.end_episode(ids)
    assert buffer.window_mask(3)[0, 0]
    monkeypatch.setattr(buffer, "_sample_from_mask", lambda mask, n, **kw: (torch.tensor([0]), torch.tensor([0])))
    window = buffer.sample_windows(1, 3)
    assert window.states[:, 0, 0].tolist() == [0., 1., 2., 3.]
    assert window.done[:, 0].tolist() == [0., 0., float(terminated)]

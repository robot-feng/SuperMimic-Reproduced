"""Research-only capture of fresh G1 transitions and valid model windows.

The caller supplies the terminal capture installed for ``collect_real``.  The
wrapper records pre-reset terminal states and restores env.step even on error.
This module never samples old replay for an on-policy score estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import WindowBatch


@dataclass
class FreshTransitions:
    states: torch.Tensor       # H,N,S, before executing the action
    next_states: torch.Tensor  # H,N,S, terminal states are before reset
    contexts: torch.Tensor     # H,N,X
    frames: torch.Tensor       # H,N
    clean: torch.Tensor        # H,N,A
    executed: torch.Tensor     # H,N,A
    rewards: torch.Tensor      # H,N
    done: torch.Tensor         # H,N, post-action termination
    active: torch.Tensor       # H,N, before executing action

    def windows(self, length: int) -> WindowBatch:
        """Enumerate windows ending at most on the first terminal transition.

        All other transitions after an environment's reset are excluded. The
        next-state/time and constant-context assertions guard model supervision.
        """
        h, n = self.active.shape
        if length < 1 or length > h:
            raise ValueError('invalid window length')
        if self.done.shape != (h, n) or self.frames.shape != (h, n):
            raise ValueError('inconsistent transition shapes')
        if not torch.equal(self.active[1:], self.active[:-1] & ~self.done[:-1]):
            raise ValueError('invalid active/termination sequence')
        valid = []
        for t in range(h - length + 1):
            good = self.active[t:t + length].all(0)
            if length > 1:
                continuous = (self.next_states[t:t+length-1] - self.states[t+1:t+length]).abs().amax(-1)
                if (continuous[:, good] > 1e-5).any():
                    raise ValueError('non-contiguous state sequence')
                context_delta = (self.contexts[t:t+length] - self.contexts[t:t+1]).abs().amax(-1)
                if (context_delta[:, good] > 1e-5).any():
                    raise ValueError('context changed within window')
            for e in good.nonzero().flatten().tolist():
                valid.append((t, e))
        if not valid:
            raise ValueError('no complete active windows')
        ts = torch.tensor([t for t, _ in valid], dtype=torch.long, device=self.states.device)
        es = torch.tensor([e for _, e in valid], dtype=torch.long, device=self.states.device)
        offsets = torch.arange(length, device=ts.device)[:, None]
        idx = ts[None, :] + offsets
        env = es[None, :].expand_as(idx)
        state_seq = torch.cat((self.states[idx, env], self.next_states[ts+length-1, es][None]), dim=0)
        return WindowBatch(states=state_seq, a_exec=self.executed[idx, env], a_clean=self.clean[idx, env],
                           xi=self.contexts[ts, es], reward=self.rewards[idx, env],
                           done=self.done[idx, env].to(self.rewards.dtype), frame=self.frames[idx, env])


def capture_real(collect, env, io, ctx, cmd, term, injection, *args, **kwargs):
    """Run the existing on-policy collector while capturing its transitions.

    ``collect`` is called with *args/**kwargs; it is responsible for setting
    capture_terminal around each env.step and returning its active mask.
    """
    rows = []
    original_step = env.step

    def step(clean):
        before = io.read().clone()
        context = ctx.read().clone()
        frame = cmd.frame.clone()
        noise = injection['noise'].clone()
        result = original_step(clean)
        _, reward, done, timeout, _ = result
        after = io.read().clone()
        ended = done | timeout
        if ended.any():
            captured = injection.get('pre_reset')
            if captured is None or not torch.equal(captured['ids'], ended.nonzero().flatten()):
                raise ValueError('missing or inconsistent pre-reset terminal state')
            after[captured['ids']] = captured['state']
        executed = term.executed_raw.clone()
        # Auto-reset can reset the action term, so reconstruct executed actions
        # for ended environments from the injected, audited clean command.
        executed[ended] = (clean + noise)[ended]
        if not torch.allclose(executed, clean + noise, atol=1e-6, rtol=0):
            raise ValueError('executed command differs from injected noise')
        rows.append((before, after, context, frame, clean.clone(), executed,
                     reward.clone(), done.clone()))
        return result

    env.step = step
    try:
        trajectory = collect(*args, **kwargs)
    finally:
        env.step = original_step
    if len(rows) != len(trajectory['active']):
        raise ValueError('capture length mismatch')
    columns = [torch.stack(items) for items in zip(*rows)]
    result = FreshTransitions(*columns, active=trajectory['active'].clone())
    if not torch.allclose(result.clean, trajectory['actions'], atol=1e-6, rtol=0):
        raise ValueError('captured and score actions differ')
    if not torch.equal(result.rewards, trajectory['rewards']):
        raise ValueError('captured and score rewards differ')
    return trajectory, result

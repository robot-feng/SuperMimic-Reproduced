"""Research-only execution perturbation after a clean action has been processed."""
import torch


def replace_execution(term, clean_action, noise):
    """Keep the term's existing raw action, replace only the executed setpoint."""
    if clean_action.shape != noise.shape:
        raise ValueError('execution noise must match action shape')
    term._executed_raw = clean_action + noise
    term._processed_actions = term._executed_raw * term._scale + term._offset
    if term.cfg.clip is not None:
        term._processed_actions = torch.clamp(term._processed_actions, min=term._clip[:, :, 0], max=term._clip[:, :, 1])

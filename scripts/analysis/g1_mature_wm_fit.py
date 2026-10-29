"""Conservative offline fine-tune of the 2000-step G1 world model, CPU only."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

from g1_tail_policy_switch_probe import available_memory, sha


def select_crop(states, actions, indices, offsets, horizon, arange):
    """Select independently aligned subwindows from time-major arrays."""
    if horizon < 1 or states.shape[0] != actions.shape[0] + 1:
        raise ValueError('incompatible window length')
    if indices.ndim != 1 or offsets.shape != indices.shape:
        raise ValueError('indices and offsets must be aligned vectors')
    if bool((offsets < 0).any()) or bool((offsets + horizon > actions.shape[0]).any()):
        raise ValueError('crop outside recorded window')
    state_t = offsets[None, :] + arange(horizon + 1)[:, None]
    action_t = offsets[None, :] + arange(horizon)[:, None]
    return states[state_t, indices[None, :]], actions[action_t, indices[None, :]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dynamics-output', type=Path, required=True)
    p.add_argument('--crop', choices=('prefix', 'random'), default='prefix',
                   help='Use the first 8 transitions or a random 8-step subwindow.')
    args = p.parse_args()
    if args.output.exists() or args.dynamics_output.exists():
        raise FileExistsError('refusing to overwrite output')
    free = available_memory()
    if free is not None and free < 1800 * 1024 * 1024:
        raise RuntimeError(f'insufficient available physical memory: {free} bytes; fitting not started')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    from g1_directional_probe import config_from_checkpoint
    from g1_wm_horizon_fit import state_window_loss
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel
    torch.set_num_threads(1)
    reports, data = {}, {}
    for seed in (5501, 5502, 5510, 5511):
        path = Path(f'docs/analysis/g1_mature_windows_seed{seed}.json')
        report = json.loads(path.read_text(encoding='utf8'))
        window_path = Path(report['window_path'])
        if report['seed'] != seed or report['horizon'] != 32 or sha(window_path) != report['windows_sha256']:
            raise ValueError('window provenance mismatch')
        reports[seed] = report
        data[seed] = torch.load(window_path, map_location='cpu', weights_only=True)
    ckpath = Path(reports[5501]['checkpoint'])
    if len({r['checkpoint_sha256'] for r in reports.values()}) != 1 or sha(ckpath) != reports[5501]['checkpoint_sha256']:
        raise ValueError('checkpoint mismatch')
    ck = torch.load(ckpath, map_location='cpu', weights_only=False, mmap=True)
    _, cfg = config_from_checkpoint(ck)
    wm = WorldModel(cfg, AnalyticHeads, 'cpu'); wm.load_state_dict(ck['wm']); wm.eval()
    del ck
    gc.collect()
    print('Checkpoint and replay loaded; starting baseline evaluation.', flush=True)
    train = {k: torch.cat([data[s][k] for s in (5501, 5502)], dim=1 if k in ('states', 'a_exec') else 0)
             for k in ('states', 'a_exec', 'xi')}
    generator = torch.Generator().manual_seed(5530)
    indices = torch.randint(train['xi'].shape[0], (60, 16), generator=generator)
    crop_generator = torch.Generator().manual_seed(5531)
    offsets = (torch.randint(25, (60, 16), generator=crop_generator)
               if args.crop == 'random' else torch.zeros((60, 16), dtype=torch.long))
    opt = torch.optim.Adam(wm.dynamics.parameters(), lr=2e-5)

    @torch.no_grad()
    def evaluate(dataset):
        errors = {str(h): [] for h in (1, 8, 16, 32)}
        for ix in torch.arange(dataset['xi'].shape[0]).split(16):
            pred = dataset['states'][0, ix]
            for t in range(32):
                pred = wm.step(pred, dataset['a_exec'][t, ix], dataset['xi'][ix])
                if t + 1 in (1, 8, 16, 32):
                    errors[str(t + 1)].append(wm.spec.normalized_sq_error(pred, dataset['states'][t + 1, ix]))
        results = {}
        for h, chunks in errors.items():
            e = torch.cat(chunks)
            if not torch.isfinite(e).all():
                raise ValueError('nonfinite evaluation')
            results[h] = {'mean': float(e.mean()), 'median': float(e.median()), 'p90': float(torch.quantile(e, .9))}
        return results

    evaluations = {'0': {str(seed): evaluate(data[seed]) for seed in (5510, 5511)}}
    print('Baseline evaluation complete; starting dynamics fitting.', flush=True)
    training_seconds = 0.
    for i, ix in enumerate(indices, 1):
        started = time.perf_counter()
        offset = offsets[i - 1]
        states, actions = select_crop(train['states'], train['a_exec'], ix, offset, 8, torch.arange)
        loss = state_window_loss(wm, states, actions, train['xi'][ix], 8)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.dynamics.parameters(), 1.)
        opt.step()
        training_seconds += time.perf_counter() - started
        if i in (20, 60):
            print(f'Completed {i} model updates; evaluating held-out windows.', flush=True)
            evaluations[str(i)] = {str(seed): evaluate(data[seed]) for seed in (5510, 5511)}
    torch.save(wm.dynamics.state_dict(), args.dynamics_output)
    out = {'checkpoint': str(ckpath), 'checkpoint_sha256': sha(ckpath), 'train_seeds': [5501, 5502],
           'eval_seeds': [5510, 5511], 'window_report_sha256': {str(s): sha(Path(f'docs/analysis/g1_mature_windows_seed{s}.json')) for s in reports},
           'dynamics_sha256': sha(args.dynamics_output), 'minibatch_seed': 5530, 'batch_size': 16,
           'crop': args.crop, 'crop_seed': 5531 if args.crop == 'random' else None,
           'horizon': 8, 'updates': 60, 'lr': 2e-5, 'optimizer': 'Adam cold start', 'grad_clip': 1.,
           'training_model_steps': 60*16*8, 'training_seconds': training_seconds, 'last_minibatch_loss': float(loss.detach()),
           'evaluations': evaluations,
           'notes': 'Frozen policy/terminal head; offline state loss only. Full-32 window selection and three clips; held-out seeds are not independent training. Evaluation open-loop using recorded actions.'}
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

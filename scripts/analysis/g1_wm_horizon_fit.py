"""Offline paired G1 dynamics fit: K=8 versus K=32 and matched model steps."""
import argparse
import json
import os
import time
from pathlib import Path

from g1_tail_policy_switch_probe import available_memory, sha


def state_window_loss(wm, states, actions, xi, horizon):
    if horizon < 1 or states.shape[0] < horizon + 1 or actions.shape[0] < horizon:
        raise ValueError('incomplete training window')
    state = states[0]
    loss = 0.
    for t in range(horizon):
        state = wm.step(state, actions[t], xi)
        loss = loss + wm.spec.error_vector(state, states[t + 1]).square().sum(-1).mean() / horizon
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights-output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.weights_output.exists():
        raise FileExistsError('refusing to overwrite outputs')
    free = available_memory()
    if free is not None and free < 650 * 1024 * 1024:
        raise RuntimeError(f'insufficient available memory: {free} bytes')
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    from g1_directional_probe import config_from_checkpoint
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel
    torch.set_num_threads(1)
    seeds = [5301, 5302, 5310, 5311]
    data = {}
    reports = {}
    for seed in seeds:
        path = Path(f'docs/analysis/g1_wm_horizon_seed{seed}.json')
        report = json.loads(path.read_text(encoding='utf8'))
        windows = Path(report['window_path'])
        if report['seed'] != seed or report['horizon'] != 32 or sha(windows) != report['windows_sha256']:
            raise ValueError('window provenance mismatch')
        reports[seed] = report
        data[seed] = torch.load(windows, map_location='cpu', weights_only=True)
    ckpath = Path(reports[5301]['checkpoint'])
    if len({r['checkpoint_sha256'] for r in reports.values()}) != 1 or sha(ckpath) != reports[5301]['checkpoint_sha256']:
        raise ValueError('checkpoint mismatch')
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    _, cfg = config_from_checkpoint(ck)
    initial = ck['wm']
    train = {key: torch.cat([data[seed][key] for seed in (5301, 5302)], dim=1 if key in ('states', 'a_exec') else 0)
             for key in ('states', 'a_exec', 'xi')}
    rng = torch.Generator().manual_seed(5330)
    indices = torch.randint(train['xi'].shape[0], (240, 16), generator=rng)

    @torch.no_grad()
    def evaluate(wm, dataset):
        pred = dataset['states'][0]
        output = {}
        for t in range(32):
            pred = wm.step(pred, dataset['a_exec'][t], dataset['xi'])
            if t + 1 in (1, 8, 16, 32):
                e = wm.spec.normalized_sq_error(pred, dataset['states'][t + 1])
                output[str(t + 1)] = {'mean': float(e.mean()), 'median': float(e.median()),
                                      'p90': float(torch.quantile(e, .9))}
        return output

    result, weights = {}, {}
    for name, horizon, updates in [('k8_60', 8, 60), ('k32_60', 32, 60), ('k32_15', 32, 15), ('k8_240', 8, 240)]:
        wm = WorldModel(cfg, AnalyticHeads, 'cpu')
        wm.load_state_dict(initial)
        wm.eval()
        opt = torch.optim.Adam(wm.dynamics.parameters(), lr=.0002)
        before = {str(seed): evaluate(wm, data[seed]) for seed in (5310, 5311)}
        start = time.perf_counter()
        for ix in indices[:updates]:
            loss = state_window_loss(wm, train['states'][:, ix], train['a_exec'][:, ix], train['xi'][ix], horizon)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.dynamics.parameters(), 1.)
            opt.step()
        seconds = time.perf_counter() - start
        after = {str(seed): evaluate(wm, data[seed]) for seed in (5310, 5311)}
        result[name] = {'horizon': horizon, 'updates': updates, 'batch': 16,
                        'model_training_step_calls': horizon * updates * 16,
                        'training_seconds': seconds, 'last_batch_state_loss': float(loss.detach()),
                        'evaluation_before': before, 'evaluation_after': after}
        weights[name] = wm.state_dict()
    torch.save(weights, args.weights_output)
    out = {'checkpoint_sha256': sha(ckpath), 'train_seeds': [5301, 5302], 'evaluation_seeds': [5310, 5311],
           'report_hashes': {str(k): sha(Path(f'docs/analysis/g1_wm_horizon_seed{k}.json')) for k in seeds},
           'weights_sha256': sha(args.weights_output), 'minibatch_seed': 5330,
           'real_transitions': sum(reports[s]['counted_simulator_transitions'] for s in seeds),
           'results': result,
           'notes': 'Offline dynamics-only state loss, cold Adam state; common 32-step eligible windows and minibatch indices. 32-step survival selection and model compute differences; not actor training or policy-performance evaluation.'}
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({name: {'seconds': v['training_seconds'], 'after': v['evaluation_after']} for name, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()

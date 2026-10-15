"""Test the existing SPE gate, keeping Kuka regularization and all other settings.

The reference is a completed benchmark_kuka_smoothness.py study with SPE off.
This enables the existing combined reward AND predicted-cost gate, without
changing its semantics or enabling the separate per-trajectory RN heuristic.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def summarize(args):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('TkAgg' if args.show else 'Agg')
    import matplotlib.pyplot as plt
    from icrl.constraint_net import ConstraintNet
    from icrl.utils import load_expert_data
    torch.set_num_threads(1)
    (obs, acs, _), _, _ = load_expert_data(str(ROOT / 'icrl/expert_data/KukaCup1'), 10, selection_seed=23)
    baseline = json.loads((args.baseline_dir / 'summary.json').read_text())
    records = [dict(r, spe=False) for r in baseline['records'] if r['weight'] == .01 and r['seed'] in args.seeds]
    samples = []
    paired_config_checks = []
    for seed in args.seeds:
        run = args.output_dir / f'seed-{seed}_lambda-0.01'
        assert (run / 'complete.json').exists()
        reference = args.baseline_dir / run.name
        cfg = json.loads((run / 'files/config.json').read_text())
        original = json.loads((reference / 'files/config.json').read_text())
        changed = {k for k in cfg if cfg[k] != original.get(k)}
        assert changed == {'save_dir', 'select_policy_excerpts'}, changed
        assert cfg['select_policy_excerpts'] and not cfg['add_rn_each_traj']
        assert cfg['cn_smoothness_coeff'] == .01 and cfg['kNN_thresh'] == .03
        for filename in ('task_config.json', 'evaluation_0.json'):
            assert json.loads((run / 'files' / filename).read_text()) == json.loads((reference / 'files' / filename).read_text())
        paired_config_checks.append(seed)
        for iteration in range(11):
            sampling = json.loads((run / f'files/sampling_{iteration}.json').read_text())
            samples.append(dict(seed=seed, **sampling))
            if iteration == 0:
                continue
            evaluation = json.loads((run / f'files/evaluation_{iteration}.json').read_text())
            model = ConstraintNet.load(str(run / f'files/models/icrl_{iteration}_itrs/cn.pt'), device='cpu')
            model.manual_threshold = cfg['cn_manual_threshold']
            model.eps = cfg['cn_eps']
            model.network.eval()
            row = dict(seed=seed, spe=True, weight=.01, iteration=iteration,
                       goal=evaluation['goal_reached_portion'], safe_goal=evaluation['safe_goal_reached_portion'],
                       path=evaluation['average_path_length'],
                       expert_feasible=float(np.mean(model.cost_function(obs, acs) == 0)))
            if iteration in (3, 10):
                result = json.loads((run / f'files/perturbed_{iteration}.json').read_text())
                previous = json.loads((reference / f'files/perturbed_{iteration}.json').read_text())
                for key in ('start', 'goal'):
                    np.testing.assert_array_equal([r[key] for r in result['episodes']], [r[key] for r in previous['episodes']])
                row.update(perturbed_goal=result['goal_reached_portion'], perturbed_safe_goal=result['safe_goal_reached_portion'])
            records.append(row)
    groups = []
    for spe in (False, True):
        for iteration in (3, 10):
            group = [r for r in records if r['spe'] == spe and r['iteration'] == iteration]
            groups.append(dict(spe=spe, iteration=iteration, metrics={
                key: dict(mean=float(np.mean([r[key] for r in group])), std=float(np.std([r[key] for r in group], ddof=1)))
                for key in ('goal', 'safe_goal', 'perturbed_goal', 'perturbed_safe_goal', 'expert_feasible')}))
    protocol = dict(seeds=args.seeds, weight=.01, kNN_threshold=.03, add_rn_each_traj=False,
                    iterations=10, inner_epochs=300, horizon=200, baseline_dir=str(args.baseline_dir),
                    spe_condition='nominal_return > corresponding_demo_return AND mean_predicted_continuous_cost < 0.5',
                    policy_excerpts_weight=1.0, true_geometry_in_training=False,
                    notes=['Same training seeds, demonstrations, dynamics and perturbation tests as SPE-off baseline.',
                           'SPE may retain zero new trajectories; existing memory remains available.',
                           'Initial evaluation uses the unmodulated DS; checkpoint i matches evaluation i.',
                           'Training-demo feasibility is not held-out constraint accuracy.',
                           'Safety is measured using the temporary cylinder only.'])
    report = dict(protocol=protocol, paired_configs_verified=paired_config_checks,
                  groups=groups, records=records, sampling=samples)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    print('GROUPS', json.dumps(groups, indent=2), flush=True)
    print('SPE retained trajectories by seed:', flush=True)
    for seed in args.seeds:
        print(seed, [s['retained_trajectories'] for s in samples if s['seed'] == seed], flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for spe, color in [(False, '#2463bd'), (True, '#087b3c')]:
        for ax, key, title in zip(axes.ravel()[:2], ('goal', 'expert_feasible'),
                                  ('Goal reached: original starts', 'Training demos predicted feasible')):
            values = np.array([[next(r[key] for r in records if r['spe'] == spe and r['seed'] == seed and r['iteration'] == i)
                                for i in range(1, 11)] for seed in args.seeds])*100
            mean, std = values.mean(0), values.std(0, ddof=1)
            ax.plot(range(1, 11), mean, 'o-', color=color, label=f'SPE {"on" if spe else "off"}')
            ax.fill_between(range(1, 11), mean-std, mean+std, color=color, alpha=.15)
            ax.set(title=title, xlabel='Training updates', ylabel='Percent', ylim=(0, 105))
    for seed in args.seeds:
        row = [s for s in samples if s['seed'] == seed]
        axes[1, 0].plot(range(11), [s['retained_trajectories'] for s in row], 'o-', label=f'seed {seed}')
        axes[1, 1].plot(range(10), [s['reliable_infeasible_points'] for s in row[:10]], 'o-', label=f'seed {seed}')
    axes[1, 0].axhline(10, linestyle='--', color='gray', label='SPE off: all 10 retained')
    axes[1, 0].set(title='New nominal trajectories retained (SPE on)', xlabel='Sampling round (0 = initial DS)', ylabel='Trajectories', ylim=(-.5, 10.5))
    axes[1, 1].set(title='Reliable infeasible points in memory (SPE on)', xlabel='Backward update index (0-based)', ylabel='Points')
    for ax in axes.ravel():
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'KukaCup1: SPE on/off; lambda=0.01; {len(args.seeds)} paired seeds (bands = seed SD)')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'spe_comparison.png', dpi=160)
    if args.show:
        fig.canvas.manager.set_window_title('KukaCup1 - SPE filtering comparison')
        print('Opening interactive SPE comparison.', flush=True)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 23, 42, 101])
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    if args.worker:
        from icrl.tests.benchmark_kuka_smoothness import run_worker
        args.weight, args.spe = .01, True
        run_worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        files = [ROOT / p for p in ('icrl/constraint_net.py', 'icrl/dscl.py', 'icrl/utils.py', 'icrl/ds_policy.py',
                                    'gym_panda/envs/KukaCup1.py', 'icrl/tests/benchmark_kuka_smoothness.py',
                                    'icrl/tests/benchmark_kuka_spe.py')]
        files += sorted((ROOT / 'icrl/expert_data/KukaCup1/files/EXPERT/rollouts').glob('*.pkl'))
        (args.output_dir / 'input_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
        def launch(seed):
            if (args.output_dir / f'seed-{seed}_lambda-0.01/complete.json').exists():
                return seed
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
            with (args.output_dir / f'seed-{seed}.log').open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--seed', str(seed),
                                         '--output-dir', str(args.output_dir), '--baseline-dir', str(args.baseline_dir)],
                                        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Seed {seed} failed; inspect its log')
            return seed
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for future in as_completed([executor.submit(launch, seed) for seed in args.seeds]):
                print('Completed SPE seed:', future.result(), flush=True)
    summarize(args)


if __name__ == '__main__':
    main()

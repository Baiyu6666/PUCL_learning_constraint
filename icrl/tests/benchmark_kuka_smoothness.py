"""Paired, fixed-protocol KukaCup1 smoothness experiments, without online logging.

Run with --output-dir icrl/wandb/run-<name>. Use --summarize-only --show
afterwards for interactive figures. True geometry is used only for evaluation.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def dscl_parser():
    parser = argparse.ArgumentParser()
    tree = ast.parse((ROOT / 'icrl/dscl.py').read_text())
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    for node in entry.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'add_argument'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<dscl defaults>', 'exec'), {'parser': parser})
    return parser


def run_worker(args):
    import numpy as np
    import torch
    from icrl.dscl import dscl
    from icrl.utils import load_expert_data
    torch.set_num_threads(1)
    start_time = time.monotonic()
    run = args.output_dir / f'seed-{args.seed}_lambda-{args.weight:g}'
    run.mkdir(parents=True, exist_ok=False)
    task = json.loads((ROOT / 'icrl/expert_data/KukaCup1/task_config.json').read_text())
    task.update(obstacle_center=[0.6, -0.08], obstacle_radius=0.04, max_steps=200)
    task_file = run / 'synthetic_task_config.json'
    task_file.write_text(json.dumps(task, indent=2))
    os.environ['KUKACUP1_CONFIG'] = str(task_file.resolve())
    config = vars(dscl_parser().parse_args([
        'dscl', '-tei', 'KukaCup1-v0', '-eei', 'KukaCup1-v0',
        '-ep', str(ROOT / 'icrl/expert_data/KukaCup1'), '-er', '10',
        '-cosd', '0', '1', '2', '-casd', '-1', '-dno', '-dnr', '-dnc',
        '-rg', '1', '-cl', '32', '32', '-clr', '0.005', '-ni', '10', '-bi', '300',
        '-cbs', '256', '-een', '10', '-twm', '-kNNt', '0.03', '-dmr', '0.2',
        '-d', 'cpu', '-s', str(args.seed), '--cn_smoothness_coeff', str(args.weight)]))
    config.update(use_wandb=False, wandb_sweep=True, save_dir=str(run / 'files'),
                  cpu_threads=1, demonstration_points=100,
                  select_policy_excerpts=getattr(args, 'spe', False),
                  add_rn_each_traj=getattr(args, 'aret', False))
    reference = getattr(args, 'reference_point', None)
    if reference is not None:
        config.update(ds_modulation_with_refer_point=True, ds_reference_point=list(reference),
                      ds_modulation_rho=args.rho)
    assert not config['train_constraint_with_true_data'] and not config['use_true_constraint']
    assert not config['cn_normalize']
    (run / 'files').mkdir()
    (run / 'files/config.json').write_text(json.dumps(config, indent=2))
    dscl(SimpleNamespace(**config))

    (obs, acs, _), lengths, _ = load_expert_data(config['expert_path'], 10, selection_seed=args.seed)
    starts = obs[np.r_[0, np.cumsum(lengths[:-1])]]
    for iteration in range(11):
        result = json.loads((run / f'files/evaluation_{iteration}.json').read_text())
        np.testing.assert_allclose([r['goal'] for r in result['episodes']], starts[:, :3] + starts[:, 3:6], atol=1e-6)
        assert len(result['episodes']) == 10
        for episode in result['episodes']:
            assert episode['steps'] <= 200
            penalty = 0 if episode['goal_reached'] else 200 * 0.008 * np.sqrt(3) + 1
            assert abs(episode['return'] + episode['path_length'] + penalty) < 1e-5
    # New starts only; keep every original goal and the previous displacement zero.
    # Fixed dataset order and perturbations are shared by all seeds and weights.
    (fixed_obs, _, _), fixed_lengths, _ = load_expert_data(config['expert_path'], 10, selection_seed=23)
    fixed_starts = fixed_obs[np.r_[0, np.cumsum(fixed_lengths[:-1])]]
    rng = np.random.RandomState(51015)
    perturbed = np.repeat(fixed_starts.copy(), 2, axis=0)
    noise = rng.randn(len(perturbed), 3)
    noise *= 0.01 / np.linalg.norm(noise, axis=1, keepdims=True)
    perturbed[:, :3] += noise
    perturbed[:, 3:6] -= noise
    perturbed[:, 6:] = 0
    np.testing.assert_allclose(perturbed[:, :3] + perturbed[:, 3:6],
                               np.repeat(fixed_starts[:, :3] + fixed_starts[:, 3:6], 2, axis=0), atol=1e-6)

    import gym
    import gym_panda
    from icrl.constraint_net import ConstraintNet
    from icrl.ds_policy import DS_Policy
    from icrl.goal_reaching_evaluation import evaluate_goal_reaching_policy
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalizeWithCost
    env = VecNormalizeWithCost(DummyVecEnv([lambda: gym.make('KukaCup1-v0', config_path=str(task_file))]),
                               norm_obs=False, norm_reward=False, norm_cost=False)
    try:
        for iteration in (3, 10):
            model = ConstraintNet.load(str(run / f'files/models/icrl_{iteration}_itrs/cn.pt'), device='cpu')
            model.manual_threshold = config['cn_manual_threshold']
            model.eps = config['cn_eps']
            model.network.eval()
            policy = DS_Policy(3, lambda x: 2*x[3:6], stable_modulation_basis=True)
            if config['ds_modulation_with_refer_point']:
                policy.modulate_with_NN_with_refer_point(model.build_gamma_with_grad_for_ds,
                    refer_point=config['ds_reference_point'], rho=config['ds_modulation_rho'])
            else:
                policy.modulate_with_NN(model.build_gamma_with_grad_for_ds, config['ds_modulation_rho'])
            env.env_method('set_starting_point', perturbed)
            result = evaluate_goal_reaching_policy(policy, env, env.get_attr('true_cost_function')[0], 20)
            (run / f'files/perturbed_{iteration}.json').write_text(json.dumps(result, indent=2))
    finally:
        env.close()
    (run / 'complete.json').write_text(json.dumps(dict(seconds=time.monotonic()-start_time, seed=args.seed, weight=args.weight)))


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
    rng = np.random.RandomState(8674)
    mix = rng.rand(4096, 1)
    anchors = torch.tensor(mix*obs[rng.randint(len(obs), size=4096), :3]
                           + (1-mix)*obs[rng.randint(len(obs), size=4096), :3], dtype=torch.float32)
    directions = torch.tensor(rng.randn(4096, 3), dtype=torch.float32)
    rows = []
    for seed in args.seeds:
        for weight in (0.0, 0.01):
            run = args.output_dir / f'seed-{seed}_lambda-{weight:g}'
            assert (run / 'complete.json').exists(), str(run)
            cfg = json.loads((run / 'files/config.json').read_text())
            assert cfg['seed'] == seed and cfg['cn_smoothness_coeff'] == weight
            for iteration in range(1, 11):
                result = json.loads((run / f'files/evaluation_{iteration}.json').read_text())
                model = ConstraintNet.load(str(run / f'files/models/icrl_{iteration}_itrs/cn.pt'), device='cpu')
                model.manual_threshold = cfg['cn_manual_threshold']
                model.eps = cfg['cn_eps']
                model.network.eval()
                with torch.no_grad():
                    curvatures = {f'curvature_{radius:g}': float(model.local_curvature_loss(anchors, radius, directions=directions))
                                  for radius in (0.005, 0.01, 0.02)}
                row = dict(seed=seed, weight=weight, iteration=iteration,
                           goal=result['goal_reached_portion'], safe_goal=result['safe_goal_reached_portion'],
                           path=result['average_path_length'],
                           expert_feasible=float(np.mean(model.cost_function(obs, acs) == 0)), **curvatures)
                if iteration in (3, 10):
                    perturbed = json.loads((run / f'files/perturbed_{iteration}.json').read_text())
                    row.update(perturbed_goal=perturbed['goal_reached_portion'],
                               perturbed_safe_goal=perturbed['safe_goal_reached_portion'])
                rows.append(row)
    groups = []
    for weight in (0.0, 0.01):
        for iteration in (3, 10):
            group = [r for r in rows if r['weight'] == weight and r['iteration'] == iteration]
            keys = ['goal', 'safe_goal', 'perturbed_goal', 'perturbed_safe_goal', 'expert_feasible', 'curvature_0.01']
            groups.append(dict(weight=weight, iteration=iteration,
                               metrics={key: dict(mean=float(np.mean([r[key] for r in group])),
                                                  std=float(np.std([r[key] for r in group], ddof=1))) for key in keys}))
    pairs = []
    for seed in args.seeds:
        for iteration in (3, 10):
            a = next(r for r in rows if r['seed'] == seed and r['iteration'] == iteration and r['weight'] == 0)
            b = next(r for r in rows if r['seed'] == seed and r['iteration'] == iteration and r['weight'] == 0.01)
            pairs.append(dict(seed=seed, iteration=iteration,
                              goal_difference=b['goal']-a['goal'], perturbed_difference=b['perturbed_goal']-a['perturbed_goal'],
                              expert_feasible_difference=b['expert_feasible']-a['expert_feasible'],
                              curvature_ratio=b['curvature_0.01']/a['curvature_0.01']))
    report = dict(protocol=dict(seeds=args.seeds, weights=[0, 0.01], updates=10, inner_epochs=300,
                                horizon=200, regularizer_radius=0.01, anchor_batch_size=64,
                                evaluation_updates=[3, 10], perturbation_seed=51015,
                                start_perturbation_radius_m=0.01, perturbations_per_demo=2,
                                true_geometry_in_training=False,
                                notes=['Five paired training seeds; episodes share the same demonstrations.',
                                       'Training-demo feasibility is not held-out constraint accuracy.',
                                       'Perturbed starts keep the original goals; no test data enters training.',
                                       'Safety refers only to the temporary test cylinder, not known real geometry.',
                                       'Fixed weights and checkpoints, no best-checkpoint selection.']),
                  groups=groups, paired_results=pairs, records=rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    print('GROUP SUMMARY', json.dumps(groups, indent=2), flush=True)
    print('PAIRED RESULTS', json.dumps(pairs, indent=2), flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for weight, color in [(0, '#2463bd'), (0.01, '#087b3c')]:
        for ax, key, title, scale in zip(axes.ravel(), ['goal', 'expert_feasible', 'curvature_0.01', 'perturbed_goal'],
                                        ['Goal reached: original starts', 'Training demos predicted feasible',
                                         'Local curvature: fixed isotropic probes', 'Goal reached: starts perturbed by 1 cm'], [100, 100, 1, 100]):
            iterations = [3, 10] if key == 'perturbed_goal' else list(range(1, 11))
            values = np.array([[next(r[key] for r in rows if r['weight'] == weight and r['seed'] == seed and r['iteration'] == i)
                                for i in iterations] for seed in args.seeds])*scale
            mean, std = values.mean(0), values.std(0, ddof=1)
            ax.plot(iterations, mean, 'o-', color=color, label=f'lambda={weight:g}')
            ax.fill_between(iterations, mean-std, mean+std, color=color, alpha=0.15)
            ax.set(title=title, xlabel='Training updates')
            ax.grid(alpha=0.25)
            if scale == 100:
                ax.set_ylim(0, 105)
                ax.set_ylabel('Percent')
            else:
                ax.set_yscale('log')
    axes[0, 0].legend()
    fig.suptitle(f'KukaCup1: {len(args.seeds)} paired seeds; mean +/- seed standard deviation')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'learning_curves.png', dpi=160)
    if args.show:
        fig.canvas.manager.set_window_title('KukaCup1 - multi-seed smoothness benchmark')
        print('Opening interactive learning curves.', flush=True)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 23, 42, 101])
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--weight', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        run_worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        # Save data/code hashes alongside results for reproducibility.
        inputs = [ROOT / path for path in ['icrl/constraint_net.py', 'icrl/dscl.py', 'icrl/ds_policy.py',
                                           'gym_panda/envs/KukaCup1.py', 'icrl/tests/benchmark_kuka_smoothness.py']]
        inputs += sorted((ROOT / 'icrl/expert_data/KukaCup1/files/EXPERT/rollouts').glob('*.pkl'))
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        (args.output_dir / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
        def launch(seed, weight):
            name = f'seed-{seed}_lambda-{weight:g}'
            if (args.output_dir / name / 'complete.json').exists():
                return name + ' already complete'
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
            with (args.output_dir / f'{name}.log').open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--output-dir', str(args.output_dir),
                                         '--seed', str(seed), '--weight', str(weight)], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} failed; inspect its log')
            return name + ' complete'
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(launch, seed, weight) for seed in args.seeds for weight in (0, 0.01)]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    summarize(args)


if __name__ == '__main__':
    main()

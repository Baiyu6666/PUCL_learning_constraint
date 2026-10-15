"""Compare fixed runs without using any obstacle geometry for probe sampling."""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('TkAgg' if '--show' in sys.argv else 'Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from icrl.constraint_net import ConstraintNet
from icrl.tests.plot_kuka_constraint_3d import boundary_faces
from icrl.utils import load_expert_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dirs[0] / 'files/config.json').read_text())
    (obs, acs, _), _, _ = load_expert_data(config['expert_path'], 10, selection_seed=config['seed'])
    # Identical label-free probes for all runs, independent of true geometry.
    rng = np.random.RandomState(8674)
    mix = rng.rand(4096, 1)
    anchors = mix * obs[rng.randint(len(obs), size=4096), :3] + (1-mix) * obs[rng.randint(len(obs), size=4096), :3]
    directions = torch.tensor(rng.randn(4096, 3), dtype=torch.float32)
    anchors = torch.tensor(anchors, dtype=torch.float32)
    radii = (0.005, 0.01, 0.02)
    trajectories = []
    for file in sorted((Path(config['expert_path']) / 'files/EXPERT/rollouts').glob('*.pkl')):
        with file.open('rb') as stream:
            data = pickle.load(stream)
        trajectories.append(np.vstack([data['observations'][:, :3], data['terminal_observation'][:3]]))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    volume_figure = plt.figure(figsize=(16, 10))
    # Visualization bounds only; the regularizer never sees these bounds.
    low = np.array([0.3, -0.4, 0.1])
    high = np.array([0.9, 0.3, 1.0])
    edges = [np.linspace(a, b, 41) for a, b in zip(low, high)]
    grid_axes = [(edge[:-1] + edge[1:])/2 for edge in edges]
    grid = np.column_stack([v.ravel() for v in np.meshgrid(*grid_axes, indexing='ij')]).astype(np.float32)
    summary = []
    for column, run in enumerate(args.run_dirs):
        cfg = json.loads((run / 'files/config.json').read_text())
        weight = cfg.get('cn_smoothness_coeff', 0.0)
        label = f'lambda={weight:g}'
        records = []
        for iteration in range(1, cfg['n_iters'] + 1):
            model = ConstraintNet.load(str(run / f'files/models/icrl_{iteration}_itrs/cn.pt'), device='cpu')
            model.manual_threshold = cfg['cn_manual_threshold']
            model.network.eval()
            result = json.loads((run / f'files/evaluation_{iteration}.json').read_text())
            record = dict(iteration=iteration, goal_rate=result['goal_reached_portion'],
                          safe_goal_rate=result['safe_goal_reached_portion'],
                          mean_path=result['average_path_length'],
                          expert_feasible_fraction=float(np.mean(model.cost_function(obs, acs) == 0)))
            with torch.no_grad():
                for radius in radii:
                    record[f'curvature_{radius:g}'] = float(model.local_curvature_loss(anchors, radius, directions=directions))
            records.append(record)
            if iteration in (3, 10):
                with torch.no_grad():
                    cost = (1-model(torch.tensor(grid))).numpy().reshape(40, 40, 40)
                faces = boundary_faces(cost > model.manual_threshold, grid_axes)
                ax = volume_figure.add_subplot(2, len(args.run_dirs), (0 if iteration == 3 else len(args.run_dirs)) + column + 1, projection='3d')
                if len(faces):
                    ax.add_collection3d(Poly3DCollection(faces, facecolors='#d84a4a', edgecolors='none', alpha=0.22))
                for trajectory in trajectories:
                    ax.plot(*trajectory.T, color='#087b3c', linewidth=1.1)
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title=f'{label}, update {iteration}')
                ax.set_box_aspect(high-low)
                ax.view_init(elev=24, azim=-60)
        updates = [r['iteration'] for r in records]
        axes[0].plot(updates, [100*r['goal_rate'] for r in records], 'o-', label=label)
        axes[1].plot(updates, [100*r['expert_feasible_fraction'] for r in records], 'o-', label=label)
        axes[2].semilogy(updates, [r['curvature_0.01'] for r in records], 'o-', label=label)
        summary.append(dict(run=str(run.resolve()), weight=weight, records=records))
    titles = ['Goal reached (200-step limit)', 'Training demos predicted feasible', 'Isotropic local curvature (r=1 cm)']
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel='Training updates')
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Episodes (%)')
    axes[1].set_ylabel('Demonstration points (%)')
    figure.tight_layout()
    figure.savefig(args.output_dir / 'learning_comparison.png', dpi=160)
    volume_figure.suptitle('Red: learned cost > 0.5; green: demonstrations. Above z=0.6 m is extrapolation.')
    volume_figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92, hspace=0.28, wspace=0.08)
    volume_figure.savefig(args.output_dir / 'volume_comparison.png', dpi=140)
    report = dict(
        protocol=dict(seed=config['seed'], updates=config['n_iters'],
                      horizon=200, demo_points_per_trajectory=100,
                      regularizer_radius=0.01, anchor_batch_size=64,
                      probe_seed=8674, probe_count=4096, probe_radii=radii,
                      visualization_updates=[3, 10],
                      geometry_in_regularizer=False,
                      notes=['Fixed weights; no cylinder-fit metric used to set weights.',
                             'Temporary cylinder used only for post-hoc safety evaluation.',
                             'Feasibility is measured on training demonstrations.',
                             'One seed; comparisons are exploratory.',
                             'Plotting above workspace z=0.6 m is extrapolation.']),
        runs=summary)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2))
    print('Saved comparison:', args.output_dir, flush=True)
    for run in summary:
        print('WEIGHT', run['weight'], 'UPDATE 3', run['records'][2], 'FINAL', run['records'][-1], flush=True)
    if args.show:
        figure.canvas.manager.set_window_title('KukaCup1 - smoothness learning curves')
        volume_figure.canvas.manager.set_window_title('KukaCup1 - smoothness 3D comparison (drag to rotate)')
        plt.show()


if __name__ == '__main__':
    main()

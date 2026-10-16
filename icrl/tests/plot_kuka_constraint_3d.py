"""Plot a saved KukaCup1 learned infeasible volume and the expert trajectories."""

import argparse
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('TkAgg' if '--show' in sys.argv else 'Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from icrl.constraint_net import ConstraintNet


def boundary_faces(mask, axes):
    """Expose the boundary of occupied cells without an extra mesh dependency."""
    widths = [axis[1] - axis[0] for axis in axes]
    faces = []
    for dimension in range(3):
        other = [i for i in range(3) if i != dimension]
        for side in (-1, 1):
            neighbor = np.zeros_like(mask)
            destination, source = [slice(None)] * 3, [slice(None)] * 3
            destination[dimension] = slice(0, -1) if side == 1 else slice(1, None)
            source[dimension] = slice(1, None) if side == 1 else slice(0, -1)
            neighbor[tuple(destination)] = mask[tuple(source)]
            indices = np.argwhere(mask & ~neighbor)
            if not len(indices):
                continue
            centers = np.column_stack([axes[d][indices[:, d]] for d in range(3)])
            corners = []
            for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                vertex = centers.copy()
                vertex[:, dimension] += side * widths[dimension] / 2
                vertex[:, other[0]] += a * widths[other[0]] / 2
                vertex[:, other[1]] += b * widths[other[1]] / 2
                corners.append(vertex)
            faces.append(np.stack(corners, axis=1))
    return np.concatenate(faces) if faces else np.empty((0, 4, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='experiment directory containing files/config.json')
    parser.add_argument('--resolution', type=int, default=70)
    parser.add_argument('--checkpoint-index', type=int, help='checkpoint index under models/icrl_<index>_itrs')
    parser.add_argument('--z-max', type=float, help='upper z limit for visualizing height extrapolation')
    parser.add_argument('--show', action='store_true', help='open an interactive, rotatable Matplotlib window')
    args = parser.parse_args()
    if args.resolution < 8:
        parser.error('resolution must be at least 8')
    files = args.run_dir / 'files'
    config = json.loads((files / 'config.json').read_text())
    task = json.loads((files / 'task_config.json').read_text())
    checkpoint_index = args.checkpoint_index or config['n_iters'] + 1
    checkpoint = files / f"models/icrl_{checkpoint_index}_itrs/cn.pt"
    model = ConstraintNet.load(str(checkpoint), device='cpu')
    model.manual_threshold = config['cn_manual_threshold']
    model.eps = config['cn_eps']
    # ConstraintNet overrides nn.Module.train() with its training algorithm.
    model.network.eval()
    low = np.asarray(task.get('plot_workspace_low', task['workspace_low']))
    high = np.asarray(task.get('plot_workspace_high', task['workspace_high']))
    if args.z_max is not None:
        if args.z_max <= low[2]:
            parser.error('z-max must exceed the workspace lower z bound')
        high[2] = args.z_max
    edges = [np.linspace(a, b, args.resolution + 1) for a, b in zip(low, high)]
    axes = [(axis[:-1] + axis[1:]) / 2 for axis in edges]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    observations = np.zeros((len(positions), model.obs_dim), dtype=np.float32)
    observations[:, :3] = positions
    cost = np.concatenate([
        model.cost_function_non_binary(observations[i:i + 8192],
                                       np.zeros((len(observations[i:i + 8192]), model.acs_dim)))
        for i in range(0, len(observations), 8192)
    ]).reshape(X.shape)
    infeasible = cost > model.manual_threshold
    faces = boundary_faces(infeasible, axes)
    dataset = Path(config['expert_path'])
    if not dataset.is_absolute():
        dataset = Path(__file__).resolve().parents[2] / dataset
    manifest = json.loads((dataset / 'manifest.json').read_text())
    trajectories = []
    goals = []
    if config.get('demonstration_points'):
        for entry in manifest['converted_rollouts']:
            with (dataset / 'files/EXPERT/rollouts' / entry['rollout_file']).open('rb') as stream:
                rollout = pickle.load(stream)
            trajectories.append(np.vstack([rollout['observations'][:, :3], rollout['terminal_observation'][:3]]))
            goals.append(rollout['goal'])
    else:
        for name in manifest['selected_demo_ids']:
            with np.load(dataset / 'raw' / (name + '.npz'), allow_pickle=False) as data:
                trajectories.append(data['poses'][:, :3].copy())
                goals.append(data['poses'][-1, :3].copy())
    goals = np.asarray(goals)

    figure = plt.figure(figsize=(14, 6), dpi=160)
    for index, azimuth in enumerate([-60, -145], start=1):
        ax = figure.add_subplot(1, 2, index, projection='3d')
        if len(faces):
            surface = Poly3DCollection(faces, facecolors='#df4b4b', edgecolors='none', alpha=0.22,
                                       zsort='average')
            ax.add_collection3d(surface)
        for trajectory in trajectories:
            ax.plot(*trajectory.T, color='#087b3c', linewidth=1.5, alpha=0.95)
        starts = np.array([v[0] for v in trajectories])
        ax.scatter(*starts.T, color='#2463bd', s=18, depthshade=False)
        ax.scatter(*goals.T, color='#6f3293', marker='*', s=42, depthshade=False)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
               xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
        ax.set_box_aspect(high - low)
        ax.view_init(elev=24, azim=azimuth)
        ax.set_title('Learned infeasible region + demonstrations', fontsize=11)
    figure.legend(handles=[Patch(facecolor='#df4b4b', alpha=0.4, label=f'Learned cost > {model.manual_threshold:g}'),
                           Line2D([], [], color='#087b3c', label=f'Expert demonstrations ({len(trajectories)})'),
                           Line2D([], [], marker='o', linestyle='', color='#2463bd', label='Start'),
                           Line2D([], [], marker='*', linestyle='', color='#6f3293', label='Goal')],
                  loc='lower center', ncol=4, frameon=False)
    updates = min(checkpoint_index, config['n_iters'])
    figure.suptitle(f'KukaCup1: 3D network prediction ({updates} training updates)', fontsize=14)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.87, wspace=0.06)
    name = f'learned_constraint_3d_iter_{updates:02d}.png' if args.checkpoint_index else 'learned_constraint_3d.png'
    output = files / 'constraint_net' / name
    figure.savefig(output, dpi=160)
    np.savez_compressed(output.with_suffix('.npz'), x=axes[0], y=axes[1], z=axes[2],
                        cost=cost, threshold=model.manual_threshold)
    print('Saved:', output.resolve())
    print('Grid:', cost.shape, 'infeasible cells:', int(infeasible.sum()), 'boundary faces:', len(faces))
    print('Prediction range:', float(cost.min()), float(cost.max()))
    if args.show:
        figure.canvas.manager.set_window_title('KukaCup1 — learned 3D constraint and demonstrations')
        print('Opening interactive plot; drag with the left mouse button to rotate.', flush=True)
        plt.show()
    else:
        plt.close(figure)


if __name__ == '__main__':
    main()

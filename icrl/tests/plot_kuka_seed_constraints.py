"""Show the same fixed-update constraint across paired training seeds."""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('TkAgg' if '--show' in sys.argv else 'Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from icrl.constraint_net import ConstraintNet
from icrl.tests.plot_kuka_constraint_3d import boundary_faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=3)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.benchmark_dir / 'summary.json').read_text())
    seeds = summary['protocol']['seeds']
    trajectories = []
    for file in sorted((ROOT / 'icrl/expert_data/KukaCup1/files/EXPERT/rollouts').glob('*.pkl')):
        with file.open('rb') as stream:
            data = pickle.load(stream)
        trajectories.append(np.vstack([data['observations'][:, :3], data['terminal_observation'][:3]]))
    task = json.loads((args.benchmark_dir / f'seed-{seeds[0]}_lambda-0/files/task_config.json').read_text())
    low, high = np.array(task['workspace_low']), np.array(task['workspace_high'])
    high[2] = 1.0
    edges = [np.linspace(a, b, 41) for a, b in zip(low, high)]
    axes = [(e[:-1]+e[1:])/2 for e in edges]
    positions = torch.tensor(np.column_stack([v.ravel() for v in np.meshgrid(*axes, indexing='ij')]), dtype=torch.float32)
    fig = plt.figure(figsize=(18, 8))
    for row, weight in enumerate((0, 0.01)):
        for column, seed in enumerate(seeds):
            run = args.benchmark_dir / f'seed-{seed}_lambda-{weight:g}'
            config = json.loads((run / 'files/config.json').read_text())
            model = ConstraintNet.load(str(run / f'files/models/icrl_{args.iteration}_itrs/cn.pt'), device='cpu')
            model.network.eval()
            with torch.no_grad():
                mask = ((1-model(positions)).numpy() > config['cn_manual_threshold']).reshape(40, 40, 40)
            faces = boundary_faces(mask, axes)
            ax = fig.add_subplot(2, len(seeds), row*len(seeds)+column+1, projection='3d')
            if len(faces):
                ax.add_collection3d(Poly3DCollection(faces, facecolors='#d84a4a', edgecolors='none', alpha=0.22))
            for trajectory in trajectories:
                ax.plot(*trajectory.T, color='#087b3c', linewidth=0.8)
            ax.set(xlim=(low[0],high[0]), ylim=(low[1],high[1]), zlim=(low[2],high[2]),
                   title=f'seed {seed}, lambda={weight:g}', xlabel='x', ylabel='y', zlabel='z')
            ax.tick_params(labelsize=7)
            ax.set_box_aspect(high-low)
            ax.view_init(elev=24, azim=-60)
    fig.suptitle(f'KukaCup1, fixed update {args.iteration}: red learned infeasible volume; green demos. z > 0.6 m is extrapolation.')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.91, hspace=0.26, wspace=0.05)
    output = args.benchmark_dir / f'constraint_volumes_update_{args.iteration}.png'
    fig.savefig(output, dpi=140)
    print('Saved:', output, flush=True)
    if args.show:
        fig.canvas.manager.set_window_title(f'KukaCup1 - five seeds at update {args.iteration}, drag to rotate')
        plt.show()


if __name__ == '__main__':
    main()

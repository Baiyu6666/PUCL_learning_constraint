"""Summarize a KukaCup1 DSCL run and plot a small learning curve."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('TkAgg' if '--show' in sys.argv else 'Agg')
import matplotlib.pyplot as plt
import numpy as np

from icrl.constraint_net import ConstraintNet
from icrl.utils import load_expert_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    files = args.run_dir / 'files'
    config = json.loads((files / 'config.json').read_text())
    task = json.loads((files / 'task_config.json').read_text())
    (expert_obs, expert_acs, _), _, _ = load_expert_data(
        config['expert_path'], config['expert_rollouts'], selection_seed=config['seed'])
    probe_z = np.linspace(task['workspace_low'][2], 1.0, 901)
    probe = np.zeros((len(probe_z), 9), dtype=np.float32)
    probe[:, :2] = task['obstacle_center']
    probe[:, 2] = probe_z
    records = []
    for iteration in range(config['n_iters'] + 1):
        result = json.loads((files / f'evaluation_{iteration}.json').read_text())
        record = {key: result[key] for key in ['goal_reached_portion', 'safe_goal_reached_portion',
                                             'average_path_length', 'average_reward', 'violation_steps_portion']}
        record['iteration'] = iteration
        record['expert_feasible_fraction'] = np.nan
        record['highest_infeasible_z_at_test_xy'] = np.nan
        if iteration > 0:
            # Evaluation i uses the frozen model trained by update i, not the
            # just-trained update i+1. The last checkpoint is a duplicate.
            checkpoint = files / f'models/icrl_{iteration}_itrs/cn.pt'
            model = ConstraintNet.load(str(checkpoint), device='cpu')
            model.manual_threshold = config['cn_manual_threshold']
            model.eps = config['cn_eps']
            model.network.eval()
            expert_cost = np.concatenate([
                model.cost_function_non_binary(expert_obs[i:i + 8192], expert_acs[i:i + 8192])
                for i in range(0, len(expert_obs), 8192)
            ])
            record['expert_feasible_fraction'] = float(np.mean(expert_cost <= model.manual_threshold))
            vertical_cost = model.cost_function_non_binary(probe, np.zeros((len(probe), 3)))
            infeasible_z = probe_z[vertical_cost > model.manual_threshold]
            if len(infeasible_z):
                record['highest_infeasible_z_at_test_xy'] = float(infeasible_z.max())
        records.append(record)

    best = max(records[1:], key=lambda r: (r['safe_goal_reached_portion'], -r['average_path_length']))
    summary = {
        'best_iteration': best['iteration'],
        'best_checkpoint': str((files / f"models/icrl_{best['iteration']}_itrs/cn.pt").resolve()),
        'criterion': 'highest safe goal-reaching rate, then shortest mean path; safety uses temporary test cylinder',
        'test_obstacle_center': task['obstacle_center'],
        'test_obstacle_radius': task['obstacle_radius'],
        'best_metrics': best,
        'last_metrics': records[-1],
        'notes': ['Expert feasibility is measured on the training demonstrations, not a held-out set.',
                  'Height probes use the test XY only; predictions above workspace z are extrapolations.'],
    }
    (files / 'learning_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (files / 'learning_curve.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    iterations = [r['iteration'] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=150)
    axes[0].plot(iterations, [100 * r['goal_reached_portion'] for r in records], 'o-', label='Goal reached')
    axes[0].plot(iterations, [100 * r['safe_goal_reached_portion'] for r in records], 's-', label='Safe goal reached')
    axes[0].set(ylabel='Episodes (%)', ylim=(-5, 105), title='Reaching (test cylinder)')
    axes[0].legend()
    axes[1].plot(iterations, [r['average_path_length'] for r in records], 'o-', color='#087b3c')
    axes[1].set(ylabel='Mean path length (m)', title='Path length')
    axes[2].plot(iterations[1:], [r['highest_infeasible_z_at_test_xy'] for r in records[1:]], 'o-', color='#c74343')
    axes[2].axhspan(float(expert_obs[:, 2].min()), float(expert_obs[:, 2].max()), color='green', alpha=0.12,
                   label='Demo height range')
    axes[2].axhline(task['workspace_high'][2], color='gray', linestyle=':', label='Workspace z limit')
    axes[2].set(ylabel='Highest predicted infeasible z (m)',
                title=f"Height at test XY {tuple(task['obstacle_center'])}")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('DS updates (0 = initial straight DS)')
        ax.set_xticks(iterations)
        ax.grid(alpha=0.25)
        ax.axvline(best['iteration'], color='#6f3293', linestyle='--', alpha=0.65)
    fig.suptitle(f"KukaCup1: {config['n_iters']} updates; selected update {best['iteration']} (temporary geometry)")
    fig.tight_layout()
    output = files / 'learning_curve.png'
    fig.savefig(output)
    print(json.dumps(summary, indent=2), flush=True)
    print('Curve:', output.resolve(), flush=True)
    if args.show:
        fig.canvas.manager.set_window_title('KukaCup1 — learning curve')
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()

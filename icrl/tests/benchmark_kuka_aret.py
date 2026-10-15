"""Paired ARET test with SPE and the Kuka smoothness regularizer enabled."""
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
    previous = json.loads((args.baseline_dir / 'summary.json').read_text())
    records = [dict(r, aret=False) for r in previous['records'] if r['spe'] and r['seed'] in args.seeds]
    samples, diagnostics = [], []
    for seed in args.seeds:
        run = args.output_dir / f'seed-{seed}_lambda-0.01'
        reference = args.baseline_dir / run.name
        assert (run / 'complete.json').exists()
        cfg = json.loads((run / 'files/config.json').read_text())
        old = json.loads((reference / 'files/config.json').read_text())
        assert {k for k in cfg if cfg[k] != old.get(k)} == {'save_dir', 'add_rn_each_traj'}
        assert cfg['select_policy_excerpts'] and cfg['add_rn_each_traj'] and cfg['cn_smoothness_coeff'] == .01
        for filename in ('task_config.json', 'evaluation_0.json'):
            assert json.loads((run / 'files' / filename).read_text()) == json.loads((reference / 'files' / filename).read_text())
        for iteration in range(11):
            for enabled, folder in [(False, reference), (True, run)]:
                samples.append(dict(seed=seed, aret=enabled, **json.loads((folder / f'files/sampling_{iteration}.json').read_text())))
            if iteration < 10:
                training = json.loads((run / f'files/training_{iteration}.json').read_text())
                diagnostics.append(dict(seed=seed, iteration=iteration, **{k:v for k,v in training.items() if 'aret' in k}))
            if iteration == 0:
                continue
            result = json.loads((run / f'files/evaluation_{iteration}.json').read_text())
            model = ConstraintNet.load(str(run / f'files/models/icrl_{iteration}_itrs/cn.pt'), device='cpu')
            model.manual_threshold = cfg['cn_manual_threshold']
            model.eps = cfg['cn_eps']
            model.network.eval()
            row = dict(seed=seed, aret=True, iteration=iteration, goal=result['goal_reached_portion'],
                       safe_goal=result['safe_goal_reached_portion'], path=result['average_path_length'],
                       expert_feasible=float(np.mean(model.cost_function(obs, acs) == 0)))
            if iteration in (3, 10):
                perturbed = json.loads((run / f'files/perturbed_{iteration}.json').read_text())
                original = json.loads((reference / f'files/perturbed_{iteration}.json').read_text())
                for key in ('start', 'goal'):
                    np.testing.assert_array_equal([r[key] for r in perturbed['episodes']], [r[key] for r in original['episodes']])
                row.update(perturbed_goal=perturbed['goal_reached_portion'], perturbed_safe_goal=perturbed['safe_goal_reached_portion'])
            records.append(row)
    groups = []
    for enabled in (False, True):
        for iteration in (3, 10):
            group = [r for r in records if r['aret'] == enabled and r['iteration'] == iteration]
            groups.append(dict(aret=enabled, iteration=iteration, metrics={
                key: dict(mean=float(np.mean([r[key] for r in group])), std=float(np.std([r[key] for r in group], ddof=1)))
                for key in ('goal', 'safe_goal', 'perturbed_goal', 'perturbed_safe_goal', 'expert_feasible')}))
    report = dict(protocol=dict(seeds=args.seeds, spe=True, smoothness_coeff=.01, kNN_threshold=.03,
                                iterations=10, inner_epochs=300, horizon=200, baseline_dir=str(args.baseline_dir),
                                true_geometry_in_training=False,
                                notes=['Original ARET heuristic unchanged, including prefix label removal.',
                                       'Original SPE reward AND predicted-cost gate unchanged.',
                                       'Fixed updates 3 and 10, identical perturbed test starts and goals.',
                                       'Demo feasibility is a training metric, not held-out accuracy.']),
                  groups=groups, records=records, sampling=samples, aret_diagnostics=diagnostics)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    print('GROUPS', json.dumps(groups, indent=2), flush=True)
    for seed in args.seeds:
        print('SEED', seed, 'goals off/on', [(i, *[next(r['goal'] for r in records if r['seed']==seed and r['iteration']==i and r['aret']==enabled) for enabled in (False,True)]) for i in (3,10)], flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for enabled, color in [(False, '#2463bd'), (True, '#087b3c')]:
        for ax, key, title in zip(axes.ravel()[:2], ('goal', 'expert_feasible'),
                                  ('Goal reached: original starts', 'Training demos predicted feasible')):
            values = np.array([[next(r[key] for r in records if r['aret']==enabled and r['seed']==seed and r['iteration']==i)
                                for i in range(1,11)] for seed in args.seeds])*100
            mean, std = values.mean(0), values.std(0, ddof=1)
            ax.plot(range(1,11), mean, 'o-', color=color, label=f'ARET {"on" if enabled else "off"}')
            ax.fill_between(range(1,11),mean-std,mean+std,color=color,alpha=.15)
            ax.set(title=title,xlabel='Training updates',ylabel='Percent',ylim=(0,105))
        values = np.array([[next(s['reliable_infeasible_points'] for s in samples if s['aret']==enabled and s['seed']==seed and s['iteration']==i)
                            for i in range(10)] for seed in args.seeds])
        axes[1,0].plot(range(1,11),values.mean(0),'o-',color=color,label=f'ARET {"on" if enabled else "off"}')
        values = np.array([[next(r['perturbed_goal'] for r in records if r['aret']==enabled and r['seed']==seed and r['iteration']==i)
                            for i in (3,10)] for seed in args.seeds])*100
        axes[1,1].errorbar([3,10], values.mean(0), yerr=values.std(0,ddof=1), fmt='o-', color=color, capsize=4, label=f'ARET {"on" if enabled else "off"}')
    axes[1,0].set(title='Reliable infeasible points used for training',xlabel='Training updates',ylabel='Mean points')
    axes[1,1].set(title='Goal reached: starts perturbed by 1 cm',xlabel='Training updates',ylabel='Percent',ylim=(0,105))
    for ax in axes.ravel():
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle(f'KukaCup1: ARET on/off, SPE on, lambda=0.01; {len(args.seeds)} paired seeds (bands = SD)')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'aret_comparison.png',dpi=160)
    if args.show:
        fig.canvas.manager.set_window_title('KukaCup1 - ARET comparison with SPE on')
        print('Opening interactive ARET comparison.',flush=True)
        plt.show()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,19,23,42,101])
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve();args.baseline_dir=args.baseline_dir.resolve()
    if args.worker:
        from icrl.tests.benchmark_kuka_smoothness import run_worker
        args.weight,args.spe,args.aret=.01,True,True
        run_worker(args)
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.summarize_only:
        files=[ROOT/p for p in ('icrl/constraint_net.py','icrl/dscl.py','icrl/ds_policy.py','icrl/utils.py',
                                'gym_panda/envs/KukaCup1.py','icrl/tests/benchmark_kuka_smoothness.py','icrl/tests/benchmark_kuka_aret.py')]
        files+=sorted((ROOT/'icrl/expert_data/KukaCup1/files/EXPERT/rollouts').glob('*.pkl'))
        (args.output_dir/'input_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
        def launch(seed):
            if (args.output_dir/f'seed-{seed}_lambda-0.01/complete.json').exists():return seed
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
            with (args.output_dir/f'seed-{seed}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--seed',str(seed),
                                       '--output-dir',str(args.output_dir),'--baseline-dir',str(args.baseline_dir)],
                                      cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Seed {seed} failed; inspect its log')
            return seed
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for future in as_completed([executor.submit(launch,seed) for seed in args.seeds]):
                print('Completed ARET seed:',future.result(),flush=True)
    summarize(args)


if __name__=='__main__':
    main()

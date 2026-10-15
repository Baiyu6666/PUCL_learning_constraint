"""Compare ordinary and vertical-reference-axis DS modulation on KukaCup1.

The manually fixed XY reference is controller prior information, not a label
or input for constraint training. SPE on, ARET off and smoothness 0.01 match
the completed SPE-on baseline. Test rho 0.2 and the ReachConcave example's 2.
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
    (obs, acs, _), _, _ = load_expert_data(str(ROOT/'icrl/expert_data/KukaCup1'),10,selection_seed=23)
    old_report=json.loads((args.baseline_dir/'summary.json').read_text())
    records=[dict(r,method='ordinary rho=0.2') for r in old_report['records'] if r['spe'] and r['seed'] in args.seeds]
    sampling=[]
    for rho in (.2,2):
        method=f'reference rho={rho:g}'
        for seed in args.seeds:
            run=args.output_dir/f'rho-{rho:g}'/f'seed-{seed}_lambda-0.01'
            old=args.baseline_dir/run.name
            assert (run/'complete.json').exists()
            cfg=json.loads((run/'files/config.json').read_text());base=json.loads((old/'files/config.json').read_text())
            expected={'save_dir','ds_modulation_with_refer_point','ds_reference_point'}
            if rho!=.2:expected.add('ds_modulation_rho')
            assert {k for k in cfg if cfg[k]!=base.get(k)}==expected
            assert cfg['select_policy_excerpts'] and not cfg['add_rn_each_traj']
            assert cfg['ds_reference_point']==args.reference_point and cfg['ds_modulation_rho']==rho
            for filename in ('task_config.json','evaluation_0.json'):
                assert json.loads((run/'files'/filename).read_text())==json.loads((old/'files'/filename).read_text())
            for iteration in range(11):
                sampling.append(dict(seed=seed,method=method,**json.loads((run/f'files/sampling_{iteration}.json').read_text())))
                if iteration==0:continue
                result=json.loads((run/f'files/evaluation_{iteration}.json').read_text())
                model=ConstraintNet.load(str(run/f'files/models/icrl_{iteration}_itrs/cn.pt'),device='cpu')
                model.manual_threshold=cfg['cn_manual_threshold'];model.eps=cfg['cn_eps'];model.network.eval()
                row=dict(seed=seed,method=method,iteration=iteration,goal=result['goal_reached_portion'],
                         safe_goal=result['safe_goal_reached_portion'],path=result['average_path_length'],
                         expert_feasible=float(np.mean(model.cost_function(obs,acs)==0)))
                if iteration in (3,10):
                    perturbed=json.loads((run/f'files/perturbed_{iteration}.json').read_text())
                    prior=json.loads((old/f'files/perturbed_{iteration}.json').read_text())
                    for key in ('start','goal'):
                        np.testing.assert_array_equal([r[key] for r in perturbed['episodes']],[r[key] for r in prior['episodes']])
                    row.update(perturbed_goal=perturbed['goal_reached_portion'],perturbed_safe_goal=perturbed['safe_goal_reached_portion'])
                records.append(row)
    methods=['ordinary rho=0.2','reference rho=0.2','reference rho=2']
    groups=[]
    for method in methods:
        for iteration in (3,10):
            rows=[r for r in records if r['method']==method and r['iteration']==iteration]
            groups.append(dict(method=method,iteration=iteration,metrics={
                key:dict(mean=float(np.mean([r[key] for r in rows])),std=float(np.std([r[key] for r in rows],ddof=1)))
                for key in ('goal','safe_goal','perturbed_goal','perturbed_safe_goal','expert_feasible')}))
    all_updates={method:float(np.mean([r['goal'] for r in records if r['method']==method])) for method in methods}
    report=dict(protocol=dict(seeds=args.seeds,reference_xy=args.reference_point,reference_z='current observation z',
                              center_selection='Manually rounded XY midpoint of the demonstration position envelope.',
                              rho=[.2,2],spe=True,aret=False,smoothness_coeff=.01,kNN_threshold=.03,
                              updates=10,inner_epochs=300,horizon=200,
                              notes=['Reference axis is controller prior information, never supplied as a network input or training label.',
                                     'Network retains all three position inputs and full XYZ gradient.',
                                     'Equal tangent eigenvalues can change z velocity; z is not forcibly frozen.',
                                     'Kuka uses stable tangent basis and an ordinary-modulation fallback for singular reference geometry.',
                                     'New Kuka path passes rho by keyword; legacy ReachConcave call is unchanged.',
                                     'Safety uses temporary test geometry only.']),groups=groups,
                all_updates_mean_goal=all_updates,records=records,sampling=sampling)
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2))
    print('GROUPS',json.dumps(groups,indent=2),flush=True)
    print('ALL UPDATES MEAN',all_updates,flush=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for method in methods:
        for ax,key,title,iterations in zip(axes.ravel(),('goal','expert_feasible','perturbed_goal','safe_goal'),
                    ('Goal reached: original starts','Training demos predicted feasible','Goal reached: starts perturbed by 1 cm','Safe goal reached: temporary cylinder'),
                    (list(range(1,11)),list(range(1,11)),[3,10],list(range(1,11)))):
            values=np.array([[next(r[key] for r in records if r['method']==method and r['seed']==seed and r['iteration']==i)
                              for i in iterations] for seed in args.seeds])*100
            mean,std=values.mean(0),values.std(0,ddof=1)
            line=ax.plot(iterations,mean,'o-',label=method)[0]
            ax.fill_between(iterations,mean-std,mean+std,color=line.get_color(),alpha=.12)
            ax.set(title=title,xlabel='Training updates',ylabel='Percent',ylim=(0,105))
    for ax in axes.ravel():ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f'KukaCup1: reference XY {args.reference_point}, follows z; {len(args.seeds)} paired seeds (bands = SD)')
    fig.tight_layout();fig.savefig(args.output_dir/'reference_comparison.png',dpi=160)
    if args.show:
        fig.canvas.manager.set_window_title('KukaCup1 - vertical reference axis modulation comparison')
        print('Opening interactive reference comparison.',flush=True);plt.show()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,19,23,42,101])
    parser.add_argument('--reference-point',type=float,nargs=2,default=[.60,-.06])
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--rho',type=float,help=argparse.SUPPRESS)
    args=parser.parse_args();args.output_dir=args.output_dir.resolve();args.baseline_dir=args.baseline_dir.resolve()
    if args.worker:
        from icrl.tests.benchmark_kuka_smoothness import run_worker
        args.weight,args.spe,args.aret=.01,True,False
        run_worker(args);return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.summarize_only:
        files=[ROOT/p for p in ('icrl/ds_policy.py','icrl/dscl.py','icrl/constraint_net.py','icrl/utils.py',
                                'gym_panda/envs/KukaCup1.py','icrl/tests/benchmark_kuka_smoothness.py','icrl/tests/benchmark_kuka_reference.py')]
        files+=sorted((ROOT/'icrl/expert_data/KukaCup1/files/EXPERT/rollouts').glob('*.pkl'))
        (args.output_dir/'input_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
        def launch(seed,rho):
            folder=args.output_dir/f'rho-{rho:g}';folder.mkdir(exist_ok=True)
            if (folder/f'seed-{seed}_lambda-0.01/complete.json').exists():return seed,rho
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
            with (folder/f'seed-{seed}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--seed',str(seed),'--rho',str(rho),
                    '--reference-point',*[str(v) for v in args.reference_point],
                    '--output-dir',str(folder),'--baseline-dir',str(args.baseline_dir)],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Seed {seed}, rho {rho} failed; inspect its log')
            return seed,rho
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for future in as_completed([executor.submit(launch,seed,rho) for rho in (.2,2) for seed in args.seeds]):
                print('Completed reference seed/rho:',future.result(),flush=True)
    summarize(args)


if __name__=='__main__':main()

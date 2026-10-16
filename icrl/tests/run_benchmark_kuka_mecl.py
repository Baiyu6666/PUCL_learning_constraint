"""Single-seed MECL baseline with the established KukaCup1 reference-DS protocol."""
import argparse
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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from icrl.constraint_net import ConstraintNet
    from icrl.utils import load_expert_data
    torch.set_num_threads(1)
    run = args.output_dir / f'seed-{args.seed}_lambda-0'
    assert (run / 'complete.json').is_file()
    config = json.loads((run / 'files/config.json').read_text())
    assert config['constraint_learning_method'] == 'mecl' and config['cn_loss_type'] == 'ml'
    assert not config['select_policy_excerpts'] and config['cn_smoothness_coeff'] == 0
    assert not config['train_constraint_with_true_data'] and not config['use_true_constraint']
    (obs, acs, _), _, _ = load_expert_data(config['expert_path'], 10, selection_seed=args.seed)
    reference = ROOT / 'icrl/runs/KukaCup1-v0-20261015_133854-bnllihpv/files'
    records = []
    for method, files in [('MECL', run/'files'), ('PUCL', reference)]:
        cfg = json.loads((files/'config.json').read_text())
        for key in ('seed','cn_layers','cn_learning_rate','cn_batch_size','backward_iters',
                    'expert_rollouts','ds_modulation_rho','ds_modulation_with_refer_point','ds_reference_point'):
            assert cfg[key] == config[key], (method,key,cfg[key],config[key])
        task = json.loads((files/'task_config.json').read_text())
        other = json.loads((run/'files/task_config.json').read_text())
        assert task['tasks'] == other['tasks']
        for iteration in range(1, 11):
            result = json.loads((files/f'evaluation_{iteration}.json').read_text())
            model = ConstraintNet.load(str(files/f'models/icrl_{iteration}_itrs/cn.pt'), device='cpu')
            model.manual_threshold=.5; model.eps=cfg['cn_eps']; model.network.eval()
            record=dict(method=method,iteration=iteration,goal=result['goal_reached_portion'],
                        safe_goal=result['safe_goal_reached_portion'],path=result['average_path_length'],
                        expert_feasible=float(np.mean(model.cost_function(obs,acs)==0)))
            if iteration in (3,10) and (files/f'perturbed_{iteration}.json').exists():
                perturbed=json.loads((files/f'perturbed_{iteration}.json').read_text())
                record.update(perturbed_goal=perturbed['goal_reached_portion'],
                              perturbed_safe_goal=perturbed['safe_goal_reached_portion'])
            records.append(record)
    report=dict(protocol=dict(seed=args.seed,updates=10,inner_epochs=300,expert_rollouts=10,
        network=[32,32],reference_xy=[.6,-.06],rho=.2,threshold=.5,regularizer=args.reg_coeff,
        method='MECL maximum-likelihood loss; no PU selection, SPE, ARET or PUCL smoothness',
        safety_geometry='Estimated evaluation cylinder only, never training labels',
        comparison='Historical PUCL uses SPE and smoothness 0.01; this compares methods, not a loss-only ablation',
        validation='Offline DS training/evaluation only; no joint IK or hardware execution'),records=records,
        final=[r for r in records if r['iteration']==10],checkpoint=str(run/'files/models/icrl_10_itrs/cn.pt'))
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for method in ('PUCL','MECL'):
        rows=[r for r in records if r['method']==method]
        for ax,key,title in zip(axes,('goal','safe_goal','expert_feasible'),
                ('Goal reached','Safe goal: estimated cylinder','Training demos predicted feasible')):
            ax.plot([r['iteration'] for r in rows],[100*r[key] for r in rows],'-o',label=method)
            ax.set(title=title,xlabel='Updates',ylabel='Percent',ylim=(-3,103));ax.grid(alpha=.25)
    axes[0].legend();fig.suptitle(f'KukaCup1 baseline: seed {args.seed}, reference DS')
    fig.tight_layout();fig.savefig(args.output_dir/'learning_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report['final'],indent=2),flush=True)
    print('Checkpoint:',report['checkpoint'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=23)
    parser.add_argument('--reg-coeff',type=float,default=.01)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.output_dir=args.output_dir.resolve()
    if args.worker:
        from icrl.tests.benchmark_kuka_smoothness import run_worker
        args.weight,args.spe,args.aret,args.method=0.,False,False,'mecl'
        args.reference_point,args.rho=[.6,-.06],.2
        run_worker(args);return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    run=args.output_dir/f'seed-{args.seed}_lambda-0'
    if not (run/'complete.json').exists():
        paths=[ROOT/name for name in ('icrl/dscl.py','icrl/constraint_net.py','icrl/ds_policy.py',
            'icrl/tests/benchmark_kuka_smoothness.py','icrl/tests/run_benchmark_kuka_mecl.py',
            'gym_panda/envs/KukaCup1.py','icrl/expert_data/KukaCup1/task_config.json')]
        paths+=sorted((ROOT/'icrl/expert_data/KukaCup1/files/EXPERT/rollouts').glob('*.pkl'))
        (args.output_dir/'input_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
        with (args.output_dir/'training.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--seed',str(args.seed),
                '--reg-coeff',str(args.reg_coeff),'--output-dir',str(args.output_dir)],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    summarize(args)


if __name__ == '__main__':
    main()

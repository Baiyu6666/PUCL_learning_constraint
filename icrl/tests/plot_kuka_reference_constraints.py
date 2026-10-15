"""Compare learned constraint geometry for the three DS modulation experiments."""
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
import pyvista as pv
import torch
from icrl.constraint_net import ConstraintNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--resolution', type=int, default=75)
    parser.add_argument('--z-max', type=float, default=1.0)
    parser.add_argument('--slice-z', type=float, default=.35)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.resolution < 10:
        parser.error('resolution must be at least 10')
    torch.set_num_threads(1)
    runs = [('Ordinary, rho=0.2', args.baseline_dir/f'seed-{args.seed}_lambda-0.01'),
            ('Reference axis, rho=0.2', args.reference_dir/f'rho-0.2/seed-{args.seed}_lambda-0.01'),
            ('Reference axis, rho=2', args.reference_dir/f'rho-2/seed-{args.seed}_lambda-0.01')]
    configs = [json.loads((run/'files/config.json').read_text()) for _,run in runs]
    task = json.loads((runs[0][1]/'files/task_config.json').read_text())
    low, high = np.array(task['workspace_low']), np.array(task['workspace_high'])
    workspace_top = high[2]
    high[2] = args.z_max
    if np.any(high <= low):
        parser.error('z-max must exceed the lower workspace z bound')
    axes = [np.linspace(a,b,args.resolution) for a,b in zip(low,high)]
    positions = np.column_stack([v.ravel() for v in np.meshgrid(*axes,indexing='ij')]).astype(np.float32)
    trajectories = []
    for file in sorted((Path(configs[0]['expert_path'])/'files/EXPERT/rollouts').glob('*.pkl')):
        with file.open('rb') as stream:
            data = pickle.load(stream)
        trajectories.append(np.vstack([data['observations'][:,:3], data['terminal_observation'][:3]]))
    demos = np.concatenate(trajectories)
    slice_x, slice_y = np.linspace(low[0],high[0],180), np.linspace(low[1],high[1],180)
    sx, sy = np.meshgrid(slice_x,slice_y,indexing='ij')
    slice_points = np.column_stack([sx.ravel(),sy.ravel(),np.full(sx.size,args.slice_z)]).astype(np.float32)
    figures = [plt.figure(figsize=(15,10)),plt.figure(figsize=(15,9))]
    fields, metrics = {}, []
    for row,iteration in enumerate((3,10)):
        for column,((label,run),config) in enumerate(zip(runs,configs)):
            assert config['cn_obs_select_dim']==[0,1,2] and config['cn_acs_select_dim']==[-1]
            assert config['cn_smoothness_coeff']==.01 and config['select_policy_excerpts'] and not config['add_rn_each_traj']
            checkpoint = run/f'files/models/icrl_{iteration}_itrs/cn.pt'
            model = ConstraintNet.load(str(checkpoint),device='cpu')
            model.network.eval()
            threshold = config['cn_manual_threshold']
            with torch.no_grad():
                cost = np.concatenate([(1-model(torch.tensor(batch))).numpy().ravel()
                                       for batch in np.array_split(positions,24)]).reshape((args.resolution,)*3)
                slice_cost = (1-model(torch.tensor(slice_points))).numpy().reshape(sx.shape)
                demo_cost = (1-model(torch.tensor(demos,dtype=torch.float32))).numpy().ravel()
            mask = cost>threshold
            grid = pv.RectilinearGrid(*axes)
            grid.point_data['cost'] = cost.ravel(order='F')
            surface = grid.contour([threshold],scalars='cost').triangulate()
            ax = figures[0].add_subplot(2,3,row*3+column+1,projection='3d')
            if surface.n_cells:
                faces = surface.points[surface.faces.reshape(-1,4)[:,1:]]
                ax.add_collection3d(Poly3DCollection(faces,facecolors='#d84a4a',edgecolors='none',alpha=.3))
            for trajectory in trajectories:
                ax.plot(*trajectory.T,color='#087b3c',linewidth=1.2)
            # Outline of the training workspace ceiling, not a constraint surface.
            ax.plot([low[0],high[0],high[0],low[0],low[0]],
                    [low[1],low[1],high[1],high[1],low[1]],
                    [workspace_top]*5,color='gray',linestyle=':',linewidth=.7,alpha=.65)
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),
                   xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f'{label}; update {iteration}')
            ax.set_box_aspect(high-low);ax.view_init(elev=24,azim=-60)
            slice_ax = figures[1].add_subplot(2,3,row*3+column+1)
            heatmap = slice_ax.pcolormesh(slice_x,slice_y,slice_cost.T,vmin=0,vmax=1,cmap='RdYlBu_r',shading='auto')
            if slice_cost.min()<threshold<slice_cost.max():
                slice_ax.contour(slice_x,slice_y,slice_cost.T,levels=[threshold],colors='black',linewidths=1)
            for trajectory in trajectories:
                slice_ax.plot(*trajectory[:,:2].T,color='#087b3c',linewidth=1)
            slice_ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='x (m)',ylabel='y (m)',
                         title=f'{label}; update {iteration}',aspect='equal')
            in_workspace = np.broadcast_to((axes[2]<=workspace_top)[None,None,:],mask.shape)
            occupied_z = axes[2][np.any(mask,axis=(0,1))]
            metric = dict(method=label,iteration=iteration,checkpoint=str(checkpoint.resolve()),
                          workspace_infeasible_grid_fraction=float(mask[in_workspace].mean()),
                          demo_feasible_fraction=float((demo_cost<=threshold).mean()),
                          occupied_z_min=float(occupied_z.min()) if len(occupied_z) else None,
                          occupied_z_max=float(occupied_z.max()) if len(occupied_z) else None,
                          touches_upper_plot_bound=bool(mask[:,:,-1].any()))
            metrics.append(metric);print(metric,flush=True)
            fields[f'cost_iter{iteration}_method{column}'] = cost
    figures[0].suptitle(f'KukaCup1, seed {args.seed}: learned cost=0.5 surface (red), demos (green). Above z={workspace_top:g} m is extrapolation.')
    figures[0].subplots_adjust(left=.025,right=.975,bottom=.075,top=.91,hspace=.27,wspace=.12)
    figures[1].suptitle(f'Seed {args.seed}: XY cost slice at z={args.slice_z:g} m; black cost=0.5; green demos projected to XY')
    figures[1].subplots_adjust(left=.05,right=.88,bottom=.07,top=.90,hspace=.32,wspace=.32)
    figures[1].colorbar(heatmap,cax=figures[1].add_axes([.91,.15,.018,.65]),label='Learned continuous cost')
    output = args.reference_dir/f'constraint_comparison_seed_{args.seed}'
    output.mkdir(exist_ok=True)
    figures[0].savefig(output/'learned_volumes.png',dpi=160)
    figures[1].savefig(output/'xy_slices.png',dpi=160)
    (output/'metrics.json').write_text(json.dumps(dict(seed=args.seed,threshold=.5,workspace_ceiling=workspace_top,
        plot_z_max=args.z_max,metrics=metrics,notes=['No true obstacle surface or cost labels are used for these plots.',
            'Geometry is shown for one preselected seed; it is not a five-seed aggregate.',
            'Grid occupancy is descriptive volume, not constraint accuracy.']),indent=2))
    np.savez_compressed(output/'cost_grids.npz',x=axes[0],y=axes[1],z=axes[2],**fields)
    print('Saved comparison:',output,flush=True)
    if args.show:
        figures[0].canvas.manager.set_window_title(f'KukaCup1 - learned constraint surfaces, seed {args.seed} (drag to rotate)')
        figures[1].canvas.manager.set_window_title(f'KukaCup1 - learned constraint XY slices, seed {args.seed}')
        print('Opening rotatable 3D comparison and XY slices.',flush=True)
        plt.show()


if __name__=='__main__':main()

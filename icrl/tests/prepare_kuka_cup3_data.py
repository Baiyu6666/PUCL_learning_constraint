"""Import the Friday session and normalize exported pickles for Python 3.8/NumPy 1.

For a new session import, the exporter rollouts.json defines the subset.
--refresh-metadata indexes the existing curated EXPERT/FAILED pickles without
rewriting them. Preserve the full source session under raw/ in both cases.
"""

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


class ExportUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 writes numpy._core names; this project uses NumPy 1 on Python 3.8.
        return super().find_class(module.replace('numpy._core', 'numpy.core'), name)


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    export = source / 'analysis/pucl_rollouts'
    mapping = json.loads((export / 'rollouts.json').read_text())
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {destination}')
    shutil.copytree(source, destination / 'raw')
    rollouts = destination / 'files/EXPERT/rollouts'
    rollouts.mkdir(parents=True)
    tasks, demonstrations, position_bounds = [], [], []
    for filename, demo in mapping.items():
        source_pickle = export / 'files/EXPERT/rollouts' / filename
        with source_pickle.open('rb') as stream:
            rollout = ExportUnpickler(stream).load()
        obs, actions = rollout['observations'], rollout['actions']
        assert rollout['source_demo'] == demo
        assert obs.shape == (99, 9) and actions.shape == (99, 3)
        assert int(rollout['lengths'][0]) == len(actions)
        assert np.isfinite(obs).all() and np.isfinite(actions).all()
        assert np.max(np.abs(actions)) <= 1
        positions = np.vstack([obs[:, :3], rollout['terminal_observation'][:3]])
        np.testing.assert_allclose(np.diff(positions, axis=0), actions * 0.008, atol=1e-7)
        np.testing.assert_allclose(obs[:, :3] + obs[:, 3:6],
                                   np.broadcast_to(rollout['goal'], (99, 3)), atol=1e-7)
        position_bounds.extend([positions.min(axis=0), positions.max(axis=0), rollout['goal']])
        assert np.linalg.norm(positions[-1] - rollout['goal']) <= 0.01
        np.testing.assert_allclose(rollout['rewards'][0],
                                   -np.linalg.norm(np.diff(positions, axis=0), axis=1).sum(), atol=1e-7)
        with (rollouts / filename).open('wb') as stream:
            pickle.dump(rollout, stream, protocol=4)
        tasks.append(dict(demo_id=demo, rollout_file=filename, start=obs[0, :3].tolist(),
                          goal=rollout['goal'].tolist(), terminal_position=positions[-1].tolist(),
                          num_transitions=len(actions), num_positions=len(positions),
                          path_length=-float(rollout['rewards'][0])))
        raw_file = source / 'analysis/segments' / (demo + '.npz')
        with np.load(raw_file, allow_pickle=False) as arrays:
            samples = len(arrays['poses'])
        demonstrations.append(dict(demo_id=demo, file=f'raw/analysis/segments/{demo}.npz',
                                   sha256=hashlib.sha256(raw_file.read_bytes()).hexdigest(),
                                   num_samples=samples, source_rollout_sha256=hashlib.sha256(source_pickle.read_bytes()).hexdigest(),
                                   rollout_file=f'files/EXPERT/rollouts/{filename}'))
    # Demonstration bounds only select the plotting region, never the rollout limits.
    plot_low = np.floor((np.min(position_bounds, axis=0) - 0.05) * 100) / 100
    plot_high = np.ceil((np.max(position_bounds, axis=0) + 0.05) * 100) / 100
    task_config = dict(coordinate_frame='iiwa7_link_0', units='metres',
                       true_constraint_available=False, obstacle_center=None, obstacle_radius=None,
                       geometry_source='Unknown: source snapshot constraint and scene are null. No KukaCup1 cylinder is assumed.',
                       workspace_low=[-10.0] * 3, workspace_high=[10.0] * 3,
                       workspace_source='Temporary broad simulation bounds; not measured robot workspace.',
                       plot_workspace_low=plot_low.tolist(), plot_workspace_high=plot_high.tolist(),
                       step_size=0.008, goal_tolerance=0.01, max_steps=200, tasks=tasks)
    manifest = dict(schema_version=1, dataset_name='KukaCup3', environment_name='KukaCup1',
                    data_role='expert_demonstrations', source_session=source.name, source_directory=str(source),
                    selected_demo_ids=list(mapping.values()), num_demonstrations=len(mapping),
                    total_samples=sum(d['num_samples'] for d in demonstrations),
                    total_training_observations=sum(t['num_transitions'] for t in tasks),
                    conversion_status='pucl_rollouts_ready', demonstrations=demonstrations,
                    downsampling=dict(method='uniform_arc_length', num_positions_per_demo=100),
                    notes=[f'Imported {len(mapping)} demonstrations from raw/analysis/pucl_rollouts/rollouts.json.',
                           'raw/ is an unchanged copy of the entire source session, including the 25-entry selected_demos.txt.',
                           'Training pickles contain the same exported values, reserialized with NumPy 1 and protocol 4.',
                           'Observations are [XYZ, goal-XYZ, previous displacement]; actions are displacement/0.008.',
                           'True constraint geometry is unavailable; safety and constraint accuracy are not evaluated.'])
    for filename, data in [('task_config.json', task_config), ('manifest.json', manifest)]:
        (destination / filename).write_text(json.dumps(data, indent=2) + '\n')
    return tasks


def refresh_metadata(dataset_dir):
    """Index current rollouts and update task metadata, never trajectory values."""
    root = Path(dataset_dir).resolve()
    config_file, manifest_file = root / 'task_config.json', root / 'manifest.json'
    config = json.loads(config_file.read_text())
    manifest = json.loads(manifest_file.read_text())
    demonstrations, failed, tasks, bounds = [], [], [], []
    for role, records in [('EXPERT', demonstrations), ('FAILED', failed)]:
        for path in sorted((root / f'files/{role}/rollouts').glob('*.pkl'), key=lambda p: int(p.stem)):
            with path.open('rb') as stream:
                rollout = ExportUnpickler(stream).load()
            obs, actions = rollout['observations'], rollout['actions']
            positions = np.vstack([obs[:, :3], rollout['terminal_observation'][:3]])
            if rollout['save_scheme'] != role or obs.shape != (99, 9) or actions.shape != (99, 3):
                raise ValueError(f'{path}: unexpected role or rollout shape')
            if not np.isfinite(obs).all() or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                raise ValueError(f'{path}: invalid observations or actions')
            np.testing.assert_allclose(np.diff(positions, axis=0), actions * config['step_size'], atol=1e-7)
            np.testing.assert_allclose(obs[:, :3] + obs[:, 3:6],
                                       np.broadcast_to(rollout['goal'], (len(obs), 3)), atol=1e-7)
            np.testing.assert_allclose(rollout['rewards'][0],
                                       -np.linalg.norm(np.diff(positions, axis=0), axis=1).sum(), atol=1e-7)
            if int(rollout['lengths'][0]) != len(obs):
                raise ValueError(f'{path}: inconsistent trajectory length')
            if np.linalg.norm(positions[-1] - rollout['goal']) > config['goal_tolerance']:
                raise ValueError(f'{path}: terminal position does not reach the goal')
            demo = rollout['source_demo']
            record = dict(demo_id=demo, rollout_file=str(path.relative_to(root)),
                          rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          num_transitions=len(actions), num_positions=len(positions),
                          label_source=rollout.get('label_source'))
            if role == 'EXPERT':
                raw_file = root / 'raw/analysis/segments' / (demo + '.npz')
                with np.load(raw_file, allow_pickle=False) as arrays:
                    raw_positions = arrays['poses'][:, :3]
                    np.testing.assert_allclose(raw_positions[0], positions[0], atol=1e-7)
                    np.testing.assert_allclose(raw_positions[-1], rollout['goal'], atol=1e-7)
                    record['num_samples'] = len(raw_positions)
                record.update(file=str(raw_file.relative_to(root)),
                              sha256=hashlib.sha256(raw_file.read_bytes()).hexdigest())
                tasks.append(dict(demo_id=demo, rollout_file=path.name, start=obs[0, :3].tolist(),
                                  goal=rollout['goal'].tolist(), terminal_position=positions[-1].tolist(),
                                  num_transitions=len(actions), num_positions=len(positions),
                                  path_length=-float(rollout['rewards'][0])))
                bounds.extend([positions.min(axis=0), positions.max(axis=0), rollout['goal']])
            # FAILED labels are preserved as provided; their recording-session
            # provenance is not inferred from matching demo names in the raw archive.
            records.append(record)
    if not tasks:
        raise ValueError(f'No EXPERT rollouts found in {root}')
    config['tasks'] = tasks
    config.setdefault('plot_workspace_low', (np.floor((np.min(bounds, axis=0) - 0.05) * 100) / 100).tolist())
    config.setdefault('plot_workspace_high', (np.ceil((np.max(bounds, axis=0) + 0.05) * 100) / 100).tolist())
    manifest.update(selected_demo_ids=[task['demo_id'] for task in tasks], num_demonstrations=len(tasks),
                    demonstrations=demonstrations, total_samples=sum(d['num_samples'] for d in demonstrations),
                    total_training_observations=sum(d['num_transitions'] for d in demonstrations),
                    num_failed_demonstrations=len(failed), failed_demonstrations=failed,
                    total_failed_observations=sum(d['num_transitions'] for d in failed),
                    selection_source='Current files/EXPERT/rollouts; curated selection supersedes archived export mapping.',
                    failed_source_session=None,
                    notes=['raw/ preserves the original session, including its historical 15-rollout export.',
                           'Current EXPERT and FAILED pickle files define the curated dataset roles.',
                           'Metadata refresh validates rollouts but never rewrites, resamples or relabels them.',
                           'Each expert raw segment is checked against the current rollout start and goal.',
                           'FAILED recording-session provenance is unspecified; labels are trajectory-level only.',
                           'True geometry is unavailable; safety and constraint accuracy are not evaluated.'])
    config_file.write_text(json.dumps(config, indent=2) + '\n')
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
    return tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='?')
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'expert_data/KukaCup3')
    parser.add_argument('--refresh-metadata', action='store_true',
                        help='Update manifest and task list from existing pickles without changing them.')
    args = parser.parse_args()
    if args.refresh_metadata:
        tasks = refresh_metadata(args.destination)
        print(f'Indexed {len(tasks)} current KukaCup3 expert demonstrations in {args.destination}')
    else:
        if args.source is None:
            parser.error('source is required unless --refresh-metadata is specified')
        tasks = prepare(args.source, args.destination)
        print(f'Imported {len(tasks)} KukaCup3 demonstrations into {args.destination}')


This repository contains the code for the paper "Positive-Unlabeled Constraint Learning (PUCL) for Inferring Nonlinear Continuous Constraint Functions from Expert Demonstrations" [[link]](https://arxiv.org/abs/2408.01622). This method leverages Positive-Unlabeled learning techniques to infer implicit constraints from demonstrations, allowing robust and efficient learning of constraint functions in continuous state-action spaces. This repository is based on the code of Inverse Constrained Reinforcement Learning (ICML 2021).
## Code Dependency

It's highly recommended to use Python 3.8. 
```bash
pip install -r requirements.txt
pip install -e ./custom_envs 
```

MuJoCo is optional. Non-MuJoCo tasks use the base requirements above.
For MuJoCo-backed tasks (such as Ant, HalfCheetah, Walker, and UR5),
install `requirements-mujoco.txt` instead
of `requirements.txt`, and set up the compatible MuJoCo runtime. Environment
implementations are imported only when selected, so a missing MuJoCo installation
does not prevent other tasks from starting.

Training saves results locally by default; no WandB account is needed.

To run any experiment present in the code, go to the main directory and run a command from the following list. Experiments typically take 10-30 minutes to complete.

## Running Constraint Learning Experiments 

### KukaCup1: iiwa7 Cartesian reaching

A 3D point-reaching task with a different start and goal for each demonstration.
The constraint network learns **XYZ position only**. Reward is negative path
length, with a timeout penalty; episodes last at most **200 steps**.
The 10 demonstrations in `icrl/expert_data/KukaCup1` are already resampled to
100 positions (99 transitions) each.

The default evaluation obstacle is an estimated vertical cylinder with XY center
`[0.60, -0.08]` and radius 0.04 m. Its geometry is not used as training labels.

Run from the repository root with the `pucl-public` Conda environment activated:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python run_me.py dscl -g KukaCup1-reference \
  -tei KukaCup1-v0 -eei KukaCup1-v0 \
  -ep icrl/expert_data/KukaCup1 -er 10 -cosd 0 1 2 \
  -dno -dnr -dnc -rg 1 -cl 32 32 -clr 0.005 \
  -ni 10 -bi 300 -cbs 256 -een 10 -cpe 1 \
  -twm -kNNt 0.03 -dmwr --ds_reference_point 0.60 -0.06
```

This uses 10 PUCL updates, 300 constraint-training epochs per update, SPE,
shape-independent smoothness regularization, and reference DS modulation.
Defaults supply seed 23, CPU, threshold 0.5, rho 0.2, evaluation/checkpointing
every update, and (for KukaCup1) SPE and smoothness coefficient 0.01.
The reference is `[0.60, -0.06, current_z]`, a manually chosen controller prior.
ARET is off. For ablations, use `--no-spe`, `--cn_smoothness_coeff 0`, or remove
`-dmwr --ds_reference_point 0.60 -0.06` for ordinary modulation.

Results are saved in a new `icrl/runs/<environment>-<timestamp>-*/files/` directory
for each run: `config.json`, `metrics.jsonl`,
per-iteration evaluation records, and `models/icrl_*_itrs/cn.pt` checkpoints.
To rebuild the expert rollouts, run `python icrl/tests/prepare_kuka_cup1_data.py`.

To save a single pass of initial, unmodulated nominal trajectories whose episode
returns exceed their corresponding expert returns, run
`python icrl/tests/generate_kuka_cup1_failed_data.py`. This performs no constraint
updates and saves rollouts to `icrl/expert_data/KukaCup1/files/FAILED/rollouts/`,
with a generation summary in `files/FAILED/summary.json`.

### KukaCup3: Friday demonstrations, shared KukaCup1 environment

`icrl/expert_data/KukaCup3/raw/` preserves the complete
`20261016T094141_129931Z_demo_friday` session. The current curated dataset has
**25 EXPERT rollouts**, each with 100 positions and 99 transitions (2,475 training
observations), plus **15 FAILED rollouts** with trajectory-level labels.
Training reads `files/EXPERT/rollouts`; the archived `raw/analysis/pucl_rollouts/`
contains a historical 15-rollout export. Both the original export and the
25-entry `selected_demos.txt` remain unchanged as provenance. The current
training pickles use a NumPy 1 compatible format. FAILED recording-session
provenance is not specified in the current bundle.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python run_me.py dscl -g KukaCup3 -n KukaCup3 \
  -tei KukaCup1-v0 -eei KukaCup1-v0 \
  --task_config_path icrl/expert_data/KukaCup3/task_config.json \
  -ep icrl/expert_data/KukaCup3 -er 25 -cosd 0 1 2 -casd -1 \
  -dno -dnr -dnc -rg 1 -cl 32 32 -clr 0.005 \
  -ni 10 -bi 300 -cbs 256 -een 25 -cpe 1 \
  -twm -kNNt 0.03 -spe --cn_smoothness_coeff 0.01
```

Both environments use `KukaCup1-v0`, with KukaCup3's own `task_config.json`
passed to training, sampling, evaluation and plotting. This command uses
10 updates, 300 epochs per update, SPE and smoothness 0.01. It uses ordinary DS
modulation because no reference point has been established for this setup.
For a quick check, change `-ni 10 -bi 300` to `-ni 1 -bi 1`.

KukaCup3 temporarily uses broad simulation bounds of **[-10, 10] m on each axis**.
These are not measured robot limits. With actions bounded by 1, a step size of
0.008 m and at most 200 steps, each coordinate can move at most 1.6 m from its
start; the current demonstration starts cannot reach these bounds during an
episode. This prevents workspace clipping from altering training rollouts.
The independent `plot_workspace_low` / `plot_workspace_high` settings select
the constraint plotting region only, currently `[0.34, -0.52, 0.24]` to
`[0.74, 0.06, 0.64]` m. They do not constrain motion or training samples.

The session snapshot has no true constraint or scene geometry. KukaCup3 therefore
sets `true_constraint_available: false`; goal-reaching, path length and returns
are evaluated, while safety fields are `null` and constraint accuracy is omitted.
To evaluate a measured cylinder later, set this flag to `true` and fill in
`obstacle_center` and `obstacle_radius` in KukaCup3's task configuration.

To update the manifest and task list after changing the curated pickle selection,
without rewriting or resampling the trajectories:

```bash
python icrl/tests/prepare_kuka_cup3_data.py --refresh-metadata
```

To import the original session's archived export into a fresh destination
(its 15-rollout subset, rather than the current curated 25-rollout selection):

```bash
python icrl/tests/prepare_kuka_cup3_data.py /path/to/20261016T094141_129931Z_demo_friday \
  --destination /path/to/new/KukaCup3
```

### 2D reach with position constraint
```bash
# PUCL
python run_me.py pucl -g test -tei PointEllip-v0 -eei PointEllipTest-v0 -ep icrl/expert_data/PointEllip -nis -lr 3e-4 -piv 10 -ft 3e4 -ni 40 -clr 0.003 -ec 0.01 -kp 1 -ki 0.1 -upid -lpd run-20240523_190205-8z00tfj3 -lpi 15 -lp -dnr -dnc -dno -spe -um -cl 32 32 -ee 3 -kNNt 0.12

# MECL
python run_me.py pucl -g test -lr 3e-4 -piv 10 -ft 3e4 -ni 40 -clr 0.003 -ec 0.0 -dno -dnc -dnr -tei PointEllip-v0 -eei PointEllipTest-v0 -lp -lpd run-20240523_190205-8z00tfj3 -lpi 15 -ep icrl/expert_data/PointEllip -upid -clt ml -crc 0.1 -cl 32 32 -kp 20 -ki 0.1 -ee 3

# BC
python run_me.py pucl -g test -lr 3e-4 -piv 10 -ft 3e4 -ni 40 -clr 0.003 -ec 0.01 -dno -dnc -dnr -tei PointEllip-v0 -eei PointEllipTest-v0 -lp -lpd run-20240523_190205-8z00tfj3 -lpi 15 -ep icrl/expert_data/PointEllip -upid -clt bce -cl 32 32 -kp 10 -ki 0.2 -ee 3 -crc 0.05

# GPUCL
python run_me.py pucl -g test -tei PointEllip-v0 -eei PointEllipTest-v0 -ep icrl/expert_data/PointEllip -nis -lr 3e-4 -piv 10 -ft 3e4 -ni 40 -clr 0.003 -ec 0.01 -kp 1 -ki 0.1 -upid -lpd run-20240523_190205-8z00tfj3 -lpi 15 -lp -dnr -dnc -dno -spe -um -cl 32 32 -ee 3 -rdm GPU -GPUlt -6 -GPUng 7

# DSCL
python  run_me.py dscl -g test -cl 32 32 -ni 15 -clr 0.005 -dno -dnc -dnr -tei PointEllip-v0 -eei PointEllipTest-v0 -ep icrl/expert_data/PointEllip -een 30 -cosd 0 1 -bi 1000 -cpe 1 -ee 999 -er 4 -d cuda:0 -cbs 256 -twm -kNNt 0.2 -aret -sf

# Note that to save training time, all four algorithms starts from a pre-trained unconstrained policy. 
```



### 3D reach with position constraint 

```bash
# CRL-PUCL
python run_me.py pucl -g test -tei ReachObs-v0 -eei ReachObs-v0 -ep icrl/expert_data/ReachObs -lr 3e-4 -piv 0.0 -ft 3e4 -ni 30 -clr 0.003 -ec 0.01 -tk 0.02 -bs 128 -ne 20 -dnr -dnc -kNNt 0.03 -cosd 0 1 2 -spe -um -upid -d cuda:0 -cpe 3 -aret -ee 3 -kp 20 -ki 0.5 -pew 0.97

# DSM-PUCL
python run_me.py dscl -g 3d_ds -cl 32 32 -ni 10 -clr 0.005 -dno -dnc -dnr -tei ReachObs-v0 -eei ReachObs-v0 -ep icrl/expert_data/ReachObsDS -cosd 0 1 2 -bi 300 -cpe 1 -er 44 -d cuda:0 -cbs 256 -twm -kNNt 0.031 -aret

# MECL
python run_me.py pucl -g test -tei ReachObs-v0 -eei ReachObs-v0 -ep icrl/expert_data/ReachObs  -lr 3e-4 -piv 0. -ft 3e4 -ni 30 -clr 0.005 -ec 0.01 -tk 0.02 -bs 256 -ne 10 -dnr -dnc -cosd 0 1 2 -upid -d cuda:0 -cpe 3 -clt ml -ee 3 -crc 0.01 -kp 0.05 -ki 0.005 -bi 10 -cl 32 32 32 -nis 

# BC
python run_me.py pucl -g test -tei ReachObs-v0 -eei ReachObs-v0 -ep icrl/expert_data/ReachObs -lr 3e-4 -piv 0.0 -ft 3e4 -ni 30 -clr 0.003 -ec 0.01 -tk 0.02 -bs 128 -ne 20 -dnr -dnc -kNNt 0.03 -cosd 0 1 2 -upid -d cuda:0 -cpe 3 -ee 3 -kp 10 -ki 0.1  -bi 200 -clt bce -crc 0.1 

# GPUCL
python run_me.py pucl -g test -tei ReachObs-v0 -eei ReachObs-v0 -ep icrl/expert_data/ReachObs -lr 3e-4 -piv 0.0 -ft 3e4 -ni 30 -clr 0.003 -ec 0.01 -tk 0.02 -bs 128 -ne 20 -dnr -dnc -kNNt 0.03 -cosd 0 1 2 -spe -um -upid -d cuda:0 -cpe 3 -aret -ee 3 -kp 30 -ki 0.2 -pew 0.97 -rdm GPU -GPUlt -7 
```

### 3D reach with velocity constraint

```bash
# PUCL
python run_me.py pucl -cl 4 4 -g ReachVel_ml_new2 -tei ReachVel-v0 -eei ReachVel-v0 -ep icrl/expert_data/ReachVel_xz -lr 3e-4 -piv 0. -ft 10e4 -ni 30 -clr 0.003 -ec 0.0 -tk 0.02 -bs 128 -ne 20 -dnr -dnc -cosd 6 7 8 -d cuda:0 -upid -kp 5 -ki 0.2  -ee 2 -clt ml -crc 0.05  -bi 500

# MECL
python run_me.py pucl -cl 4 4 -g ReachVel_pu_new -tei ReachVel-v0 -eei ReachVel-v0 -ep icrl/expert_data/ReachVel_xz -lr 3e-4 -piv 0. -ft 10e4 -ni 30 -clr 0.003 -ec 0.0 -tk 0.02 -bs 128 -ne 20 -dnr -dnc -kNNt 0.02 -cosd 6 7 8 -d cuda:0 -spe -um -upid -kp 20 -ki 0.1 -aret -ee 2 
```
### Blocked half cheetah 

```bash
# PUCL
python run_me.py pucl -g HC_pu -er 30 -ni 30 -clr 0.003 -ec 0.0 -dnc -dnr -tei HCWithPos-v0 -eei HCWithPosTest-v0 -ep icrl/expert_data/HCWithPos-New -cl 4 4 -piv 0.9 -ft 5e4 -bs 128 -kNNt 2.6 -kNNn -cosd 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 -d cuda:0 -upid -kp 0 -ki 0 -kNNm euclidean -spe -um -pew 1 

# MECL
python run_me.py pucl -g HC_ml -ft 5e4 -ni 30 -clr 0.005 -ec 0.0 -dnc -dnr -er 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -ep icrl/expert_data/HCWithPos-New -cosd 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 -bs 128 -clt ml -crc 0.2 -cl 4 4 -upid -kp 0. -ki 0. -piv 0.37 -d cuda:0 -bi 50 

# BC
python run_me.py pucl -g HC_bce -piv 0.36 -ft 5e4 -ni 30 -er 30 -bs 128 -clr 0.003 -ec 0.0 -dnc -dnr -tei HCWithPos-v0 -eei HCWithPosTest-v0 -ep icrl/expert_data/HCWithPos-New -cosd 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 -d cuda:0 -upid -clt bce -cl 4 4 -kp 0 -ki 0. -crc 0.2 

# GPUCL
python run_me.py pucl -g HC_gpu -tei HCWithPos-v0 -eei HCWithPosTest-v0 -ep icrl/expert_data/HCWithPos-New -piv 0.9 -ft 5e4 -ni 30 -clr 0.003 -ec 0.0 -bs 128 -dnr -dnc -cosd 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 -spe -um -upid -d cuda:0 -kp 0 -ki 0. -pew 1 -rdm GPU -GPUlt -5.5 -GPUng 9 -er 30 -cl 4 4 
```

### Constraint Transfer (Section 4.C; Figure 1)
```bash
# Learn constraint 
python run_me.py dscl -g test -cl 16 16 -ni 18 -clr 0.005 -dno -dnc -dnr -tei ReachConcaveObs-v0 -eei ReachConcaveObs-v0 -ep icrl/expert_data/ReachConcaveObsDS -cosd 0 1 2 -bi 400 -cpe 1 -er 44 -d cuda:0 -cbs 256 -twm -kNNt 0.029 -dmwr -spe -aret -dmr 2

# Transfer learned constraint network and generate policy rollouts
# Please check ds_policy.py and plots/render_reach_env_bullet.py

``` 

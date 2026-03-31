
This repository contains the code for the paper "Positive-Unlabeled Constraint Learning (PUCL) for Inferring Nonlinear Continuous Constraint Functions from Expert Demonstrations" [[link]](https://arxiv.org/abs/2408.01622). This method leverages Positive-Unlabeled learning techniques to infer implicit constraints from demonstrations, allowing robust and efficient learning of constraint functions in continuous state-action spaces. This repository is based on the code of Inverse Constrained Reinforcement Learning (ICML 2021).
## Code Dependency

It's highly recommended to use Python 3.8. 
```bash
pip install -r requirements.txt
pip install -e ./custom_envs 
```

In addition, you will need to setup a (free) [wandb](www.wandb.ai) account. 

To run any experiment present in the code, go to the main directory and run a command from the following list. Experiments typically take 10-30 minutes to complete.

## Running Constraint Learning Experiments 

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

python run_me.py dscl   -g test   -ctm mil   -tei ReachConcaveObs-v0   -eei ReachConcaveObs-v0   -ep ../icrl-master/icrl/expert_data/ReachConcaveObsDS   -fdp ../icrl-master/icrl/expert_data/ReachConcaveObsDS   -er 19   -fr 15   -cl 32 32   -clr 0.001   -bi 2000   -cpe 1   -cbs 256   -cmp lse   -cmtn 0.05   -crc 0.053   -cosd 0 1 2   -dno -dnc -dnr   -d cuda:0   -s 65 -sf -spe -twm -pew 1.05
# Transfer learned constraint network and generate policy rollouts
# Please check ds_policy.py and plots/render_reach_env_bullet.py

``` 







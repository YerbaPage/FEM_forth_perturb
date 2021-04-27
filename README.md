# Finite element method

### Desc:

Codes for paper: **A Morley-Wang-Xu element method for a fourth order elliptic singular perturbation problem** https://arxiv.org/abs/2011.14064 

### Files:

- `run.py` in each folder contains main codes for each example 
- `derivatives.ipynb` contains matlab codes used for computing derivatives to define different solution `u` for different examples

### Requirements:

- numpy == 1.18.1
- scipy == 1.4.1
- matplotlib == 3.1.1
- scikit-fem == 2.1.1
- pyamg == 4.0.0

Some changes are necessary in `scipy` and `skfem` to reproduce the experiments:

- return num of iters in scipy solvers (cg and gmres)
- wrapped mgcg solver (you can install from https://github.com/YerbaPage/scikit-fem)

Much thanks to packages : @ifem @skfem @pyamg


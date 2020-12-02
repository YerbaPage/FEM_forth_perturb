# Finite element method

### Desc:

Codes for paper: **A Morley-Wang-Xu element method for a fourth order elliptic singular perturbation problem** https://arxiv.org/abs/2011.14064 

### Reminder:

Some changes are necessary in @scipy and @skfem to reproduce the experiments:

- return num of iters in scipy solvers
- built in mgcg solver (you can install from https://github.com/YerbaPage/scikit-fem)

Much thanks to packages : @ifem @skfem @pyamg @distmesh


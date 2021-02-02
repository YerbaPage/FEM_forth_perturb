from skfem import *
import numpy as np
from utils import *
from skfem.helpers import d, dd, ddd, dot, ddot, grad, dddot, prod
from scipy.sparse.linalg import LinearOperator, minres
from skfem.models.poisson import *
from skfem.assembly import BilinearForm, LinearForm
import datetime
import pandas as pd
import sys
import time


tol = 1e-8
intorder = 6
solver_type = 'mgcg'
refine_time = 7
element_type = 'P1'
sigma = 5 
penalty = False
epsilon = 1e-6
ep = epsilon
example = 'ex3'

test_order = 3
test_path = 'uh0_{}.npy'.format(test_order)
m = MeshTri()
m.refine(test_order)
test_basis, test_fbasis = solve_problem2(m, element_type, intorder=3, basis_only=True)
test_uh0 = np.load(test_path)


base_order = 7
base_path = 'uh0_{}.npy'.format(base_order)
m = MeshTri()
m.refine(base_order)
base_basis, base_fbasis = solve_problem2(m, element_type, intorder=3, basis_only=True)
base_uh0 = np.load(base_path)

asm(mass, test_basis['u'], base_basis['u'])
# from skfem import *
import numpy as np
from utils import *
from skfem.helpers import d, dd, ddd, dot, ddot, grad, dddot, prod
from scipy.sparse.linalg import LinearOperator, minres
# from skfem.models.poisson import *
from skfem.assembly import BilinearForm, LinearForm
import datetime
import pandas as pd
import sys
import time
from tqdm import tqdm


tol = 1e-8
intorder = 6
solver_type = 'mgcg'
refine_time = 7
element_type = 'P1'
sigma = 5
penalty = True
epsilon = 1e-6
ep = epsilon
example = 'ex3'
save_path = 'log/' + example + '_' + element_type + '_' + ('pen' if penalty else 'nopen') + '_' +'{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Save $u_{h0}$ with different $h$ 

for i in range(8, 10):
    m = MeshTri()
    base_order = i
    base_path = 'solutions/{}_{}/uh0_{}.npy'.format(element_type, ('pen' if penalty else 'nopen'), base_order)
    m.refine(base_order)

    if penalty:
        uh0, basis, fbasis = solve_problem2(m, element_type, solver_type, intorder=intorder, tol=1e-8, epsilon=1e-6)
    else:
        uh0, basis = solve_problem1(m, element_type, solver_type, intorder=intorder, tol=1e-8, epsilon=1e-6)

    np.save(base_path, uh0)
    print('{} th saved'.format(i))
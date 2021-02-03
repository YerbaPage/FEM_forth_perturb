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
intorder = 3 # 6
solver_type = 'mgcg'
refine_time = 7
element_type = 'P1'
sigma = 5 
penalty = False
epsilon = 1e-6
ep = epsilon
example = 'ex3'
save_path = 'log/' + example + '_' + element_type + '_' + ('pen' if penalty else 'nopen') + '_' +'{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Save $u_{h0}$ with different $h$ 

# %%time
# for i in range(7, 10):
#     m = MeshTri()
#     base_order = i
#     base_path = 'solutions/uh0_{}.npy'.format(base_order)
#     m.refine(base_order)

#     if penalty:
#         uh0, basis, fbasis = solve_problem2(m, element_type, solver_type, intorder=6, tol=1e-8, epsilon=1e-6)
#     else:
#         uh0, basis = solve_problem1(m, element_type, solver_type, intorder=6, tol=1e-8, epsilon=1e-6)

#     np.save(base_path, uh0)
#     print('{} th saved'.format(i))

test_order = 3
test_path = 'solutions/uh0_{}.npy'.format(test_order)
coarse_m = MeshTri()
coarse_m.refine(test_order)
test_basis, test_fbasis = solve_problem2(coarse_m, element_type, intorder=3, basis_only=True)
test_uh0 = np.load(test_path)

base_order = 6
base_path = 'solutions/uh0_{}.npy'.format(base_order)
fine_m = MeshTri()
fine_m.refine(base_order)
base_basis, base_fbasis = solve_problem2(fine_m, element_type, intorder=3, basis_only=True)
base_uh0 = np.load(base_path)

coordinates = base_basis['u'].global_coordinates().value

base_test_basis = np.zeros_like(base_basis['u'].interpolate(base_uh0).value)

import concurrent.futures

def interpolator_parallel(j):
    for i in tqdm(range(base_test_basis.shape[0])):
        base_test_basis[i][j] = test_basis['u'].interpolator(test_uh0)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))
    print(j, 'done')

import time

if __name__ == '__main__':

    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(interpolator_parallel, range(base_test_basis.shape[1]))

    end = time.time()

    print('time: ', end - start)

    start = time.time()

    for i in tqdm(range(base_test_basis.shape[0])):
        for j in range(base_test_basis.shape[1]):
            base_test_basis[i][j] = test_basis['u'].interpolator(test_uh0)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))

    end = time.time()

    print('time: ', end - start)
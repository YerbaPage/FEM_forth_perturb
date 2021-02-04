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
from concurrent.futures import ProcessPoolExecutor

def interpolator_parallel(j):
    global base_test_basis
    for i in tqdm(range(base_test_basis.shape[0])):
        base_test_basis[i][j] = test_basis['u'].interpolator(test_uh0)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))
    # print(j, 'done')
    return base_test_basis

def paralleled_interpolator(test_basis, base_test_basis, coordinates):
    print('\n \n', base_test_basis)
    with ProcessPoolExecutor() as executor:
        result_list = list(executor.map(interpolator_parallel, tqdm(range(base_test_basis.shape[1]))))

    for k in tqdm(result_list):
        base_test_basis += k
    
    return base_test_basis

if __name__ == '__main__':
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

    print('Before: ', base_test_basis)
    start = time.time()

    base_test_basis = paralleled_interpolator(test_basis=test_basis, base_test_basis=base_test_basis, coordinates=coordinates)

    end = time.time()
    print('time: ', end - start)
    print('After: ', base_test_basis)
    
    # start = time.time()

    # for i in tqdm(range(base_test_basis.shape[0])):
    #     for j in range(base_test_basis.shape[1]):
    #         base_test_basis[i][j] = test_basis['u'].interpolator(test_uh0)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))

    # end = time.time()

    # print('time: ', end - start)
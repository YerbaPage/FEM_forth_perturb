# from skfem import *
import numpy as np
from utils import *
from skfem.helpers import d, dd, ddd, dot, ddot, grad, dddot, prod
from scipy.sparse.linalg import LinearOperator, minres, cg
from skfem.assembly import BilinearForm, LinearForm
import datetime
import pandas as pd
import sys
import time
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

tol = 1e-8
intorder = 3 # 6
solver_type = 'mgcg'
refine_time = 2
base_order = 8
element_type = 'P1'
sigma = 5
penalty = False
epsilon = 1e-6
ep = epsilon
example = 'ex3'
save_path = 'log/' + example + '_' + element_type + '_' + ('pen' if penalty else 'nopen') + '_' +'{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

class Logger(object):
    def __init__(self, filename=save_path+'.txt', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

sys.stdout = Logger(save_path+'.txt', sys.stdout)

base_basis, base_fbasis, base_uh0, fine_m = load_solution(test_order=base_order, element_type=element_type, penalty=penalty, intorder=intorder)

coordinates = base_basis['u'].global_coordinates().value
# print(coordinates)

# solving 

print('=======Arguments=======')
print('penalty:\t{}'.format(penalty))
print('element_type:\t{}'.format(element_type))
print('base_order:\t{}'.format(base_order))
print('solver_type:\t{}'.format(solver_type))
print('tol:\t{}'.format(tol))
print('intorder:\t{}'.format(intorder))
print('refine_time:\t{}'.format(refine_time))
print('epsilon:\t{}'.format(epsilon))
print('sigma:\t{}'.format(sigma))
print('=======Results=======')

time_start = time.time()

ep = epsilon
df_list = []
L2_list = []
Du_list = []
D2u_list = []
h_list = []
epu_list = []

def interpolator_parallel(j):
    # global base_test_basis
    for i in tqdm(range(base_test_basis.shape[0])):
        base_test_basis[i][j] = test_basis['u'].interpolator(test_uh0)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))
    # print(j, 'done')
    return base_test_basis

if __name__ == '__main__':
        
    for i in range(1, refine_time+1):

        m = MeshTri()
        m.refine(i)
        test_order = i
        print('Testing order: ', test_order)
        test_basis, test_fbasis, test_uh0, coarse_m = load_solution(test_order=test_order, element_type=element_type, penalty=penalty, intorder=intorder)

        uh0 = test_uh0
        if penalty:
            basis, fbasis = solve_problem2(m, element_type, solver_type, intorder=intorder, tol=1e-8, epsilon=1e-6, basis_only=True)
        else:
            basis = solve_problem1(m, element_type, solver_type, intorder=intorder, tol=1e-8, epsilon=1e-6, basis_only=True)

        base_test_basis = np.zeros_like(base_basis['u'].interpolate(base_uh0).value)

        # with ProcessPoolExecutor() as executor:
        #     func = interpolator_parallel
        #     result_list = list(executor.map(func, tqdm(range(base_test_basis.shape[1]))))

        # for k in tqdm(result_list):
        #     base_test_basis += k

        for i in tqdm(range(base_test_basis.shape[0])):
            for j in range(base_test_basis.shape[1]):
                base_test_basis[i][j] = test_basis['u'].interpolator(test_uh0)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))

        L2u = np.sqrt(np.sum(base_basis['u'].dx * (base_basis['u'].interpolate(base_uh0).value - base_test_basis)**2))

        dbasis = InteriorBasis(coarse_m, ElementTriDG(ElementTriP1()), intorder=intorder)
        dx = project(test_uh0, basis_from=test_basis['u'], basis_to=dbasis, diff=0)
        dy = project(test_uh0, basis_from=test_basis['u'], basis_to=dbasis, diff=1)


        dux = np.zeros_like(base_basis['u'].interpolate(base_uh0).value)

        for i in tqdm(range(base_test_basis.shape[0])):
            for j in range(base_test_basis.shape[1]):
                dux[i][j] = dbasis.interpolator(dx)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))
                
        duy = np.zeros_like(base_basis['u'].interpolate(base_uh0).value)

        for i in tqdm(range(base_test_basis.shape[0])):
            for j in range(base_test_basis.shape[1]):
                duy[i][j] = dbasis.interpolator(dy)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))

                
        def get_DuError_N(basis, u):
            duh = basis.interpolate(u).grad
            x = basis.global_coordinates().value
            dx = basis.dx
            return np.sqrt(np.sum(((duh[0] - dux)**2 + (duh[1] - duy)**2) * dx))

        Du = get_DuError_N(base_basis['u'], base_uh0)
        H1u = Du + L2u
        
        ddbasis = InteriorBasis(coarse_m, ElementTriDG(ElementTriP0()), intorder=intorder)
        dxx = project(dx, basis_from=dbasis, basis_to=ddbasis, diff=0)
        dxy = project(dx, basis_from=dbasis, basis_to=ddbasis, diff=1)
        dyx = dxy
        # dyx = project(dy, basis_from=dbasis, basis_to=ddbasis, diff=0)
        dyy = project(dy, basis_from=dbasis, basis_to=ddbasis, diff=1)

        duxx = np.zeros_like(base_basis['u'].interpolate(base_uh0).value)

        for i in tqdm(range(base_test_basis.shape[0])):
            for j in range(base_test_basis.shape[1]):
                duxx[i][j] = ddbasis.interpolator(dxx)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))
                
        duxy = np.zeros_like(base_basis['u'].interpolate(base_uh0).value)

        for i in tqdm(range(base_test_basis.shape[0])):
            for j in range(base_test_basis.shape[1]):
                duxy[i][j] = ddbasis.interpolator(dxy)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))
                
        duyx = duxy
                
        duyy = np.zeros_like(base_basis['u'].interpolate(base_uh0).value)

        for i in tqdm(range(base_test_basis.shape[0])):
            for j in range(base_test_basis.shape[1]):
                duyy[i][j] = ddbasis.interpolator(dyy)(np.array([[coordinates[0][i][j]], [coordinates[1][i][j]]]))

                
        def get_D2uError_N(basis, u):
            dduh = basis.interpolate(u).hess
            x = basis.global_coordinates().value
            dx = basis.dx
            return np.sqrt(np.sum(((dduh[0][0] - duxx)**2 + (dduh[0][1] - duxy)**2 +
                        (dduh[1][1] - duyy)**2 + (dduh[1][0] - duyx)**2) * dx))

        if penalty:
            D2u = np.sqrt(get_D2uError_N(base_basis['u'], base_uh0)**2 + L2pnvError.assemble(fbasis, w=fbasis.interpolate(uh0)))
        else:
            D2u = get_D2uError_N(base_basis['u'], base_uh0)
            
        epu = np.sqrt(epsilon**2 * D2u**2 + Du**2)
        h_list.append(m.param())
        Du_list.append(Du)
        L2_list.append(L2u)
        D2u_list.append(D2u)
        epu_list.append(epu)

    hs = np.array(h_list)
    L2s = np.array(L2_list)
    Dus = np.array(Du_list)
    D2us = np.array(D2u_list)
    epus = np.array(epu_list)
    H1s = L2s + Dus
    H2s = H1s + D2us

    # store data
    data = np.array([L2s, H1s, H2s, epus])
    df = pd.DataFrame(data.T, columns=['L2', 'H1', 'H2', 'Energy'])
    df_list.append(df)

    show_result(L2s, H1s, H2s, epus)

    df.to_csv(save_path+'.csv')
    time_end = time.time()
    print('Total Time Cost {:.2f} s'.format(time_end-time_start))
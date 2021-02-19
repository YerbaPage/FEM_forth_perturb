import numpy as np
from utils import *
from skfem.helpers import *
from scipy.sparse.linalg import LinearOperator, minres, cg
from skfem.assembly import BilinearForm, LinearForm
import datetime
import pandas as pd
from skfem.visuals.matplotlib import draw, plot, show
import sys
import time

tol = 1e-8
intorder = 5
solver_type = 'mgcg'
refine_time = 5
epsilon_range = 6
element_type = 'P1'
sigma = 5
penalty = False
# epsilon = 1e-5
example = 'ex4'
save_path = 'log/' + example + '_' + element_type + '_' + ('pen' if penalty else 'nopen') + '_' +'{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# def dirichlet(w):
#     """return a harmonic function"""
#     x, y = w
#     theta = arctan3(y, x)
#     return (x**2 + y**2)**(5/6) * sin(5*theta/3)

def solve_problem1(m, element_type='P1', solver_type='pcg', intorder=6, tol=1e-8, epsilon=1e-6):
    '''
    switching to mgcg solver for problem 1
    '''
    if element_type == 'P1':
        element = {'w': ElementTriP1(), 'u': ElementTriMorley()}
    elif element_type == 'P2':
        element = {'w': ElementTriP2(), 'u': ElementTriMorley()}
    else:
        raise Exception("Element not supported")

    basis = {
        variable: InteriorBasis(m, e, intorder=intorder)
        for variable, e in element.items()
    }
    
    K1 = asm(laplace, basis['w'])
    f1 = asm(f_load, basis['w'])
    wh = solve(*condense(K1, f1, D=basis['w'].find_dofs()), solver=solver_iter_mgcg(tol=tol))
    
    global K2, f2, uh0, boundary_dofs, boundary_basis, boundary_dofs_un
    # K2 = asm(b_load, basis['u'])
    K2 = epsilon**2 * asm(a_load, basis['u']) + asm(b_load, basis['u'])
    # f2 = asm(wv_load, basis['w'], basis['u']) * wh + asm(boundary_load_un, fbasis) + epsilon**2 * asm(boundary_load_gradun, fbasis)
    f2 = asm(wv_load, basis['w'], basis['u']) * wh
    # f2 = np.zeros(basis['u'].N)
    boundary_basis = FacetBasis(m, ElementTriMorley())
    boundary_dofs = boundary_basis.find_dofs()['all'].all()
    boundary_dofs_u = np.array([i for i in boundary_dofs if i in basis['u'].nodal_dofs[0]])
    boundary_dofs_un = np.array([i for i in boundary_dofs if i in basis['u'].facet_dofs[0]])
    # print(boundary_dofs_u)
    # print(boundary_dofs_un)
    uh0 = np.zeros(basis['u'].N)
    # print(boundary_dofs)

    # uh0[boundary_dofs] = pproject(dirichlet, basis_to=boundary_basis, I=boundary_dofs, solver=minres)
    uh0[boundary_dofs_u] = exact_u(boundary_basis.doflocs[0][boundary_dofs_u], boundary_basis.doflocs[1][boundary_dofs_u])
    uh0[boundary_dofs_un] = exact_un(boundary_basis.doflocs[0][boundary_dofs_un], boundary_basis.doflocs[1][boundary_dofs_un])
    uh0 = solve(*condense(K2, f2, uh0, D=boundary_dofs), solver=solver_iter_mgcg(tol=tol))
    # uh0 = solve(*condense(K2, f2, D=easy_boundary(m, basis['u'])), solver=solver_iter_mgcg(tol=tol))
    return uh0, basis

def exact_un(x, y):
    # nx = -1 * (x == -1) + 1 * ((x == 1) + (x == 0) * (y > 0))
    # ny = -1 * (y == -1) + 1 * ((y == 1) + (y == 0) * (x > 0))
    nx = -1 * (x == 0) + 1 * (x == 1)
    ny = -1 * (y == 0) + 1 * (y == 1)
    dux, duy = dexact_u(x, y)
    out = nx * dux + ny * duy
    # out[np.isnan(out)] = 0
    return out

@LinearForm
def f_load(v, w):
    '''
    for $(f, x_{h})$
    '''
    x, y = w.x
    # lu = 0
    # llu = 0
    # return (epsilon**2 * llu - lu) * v
    return (24*ep**2*x**4 - 48*ep**2*x**3 + 288*ep**2*x**2*y**2 - 288*ep**2*x**2*y + 72*ep**2*x**2 - 288*ep**2*x*y**2 + 288*ep**2*x*y - 48*ep**2*x + 24*ep**2*y**4 - 48*ep**2*y**3 + 72*ep**2*y**2 - 48*ep**2*y + 8*ep**2 - 12*x**4*y**2 + 12*x**4*y - 2*x**4 + 24*x**3*y**2 - 24*x**3*y + 4*x**3 - 12*x**2*y**4 + 24*x**2*y**3 - 24*x**2*y**2 + 12*x**2*y - 2*x**2 + 12*x*y**4 - 24*x*y**3 + 12*x*y**2 - 2*y**4 + 4*y**3 - 2*y**2) * v

def exact_u(x, y):
    return x**2*y**2*(x - 1)**2*(y - 1)**2

def dexact_u(x, y):
    dux = 2*x*y**2*(y - 1)**2*(2*x**2 - 3*x + 1)
    duy = 2*x**2*y*(x - 1)**2*(2*y**2 - 3*y + 1)
    return dux, duy

# m = MeshTri().init_lshaped()
m = MeshTri()
# m = MeshTri().init_symmetric()
m.refine(4)
# draw(m)

epsilon = 0
ep = epsilon

uh0, basis = solve_problem1(m, element_type, solver_type, intorder, tol, epsilon)

x = basis['u'].doflocs[0]
y = basis['u'].doflocs[1]
u = exact_u(x, y)
plot(basis['u'], u-uh0, colorbar=True)
# plot(basis['u'], u, colorbar=True)
show()
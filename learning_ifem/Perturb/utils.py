from skfem import *
import numpy as np
from skfem.visuals.matplotlib import draw, plot
from skfem.utils import solver_iter_krylov
from skfem.helpers import d, dd, ddd, dot, ddot, grad, dddot, prod
from scipy.sparse.linalg import LinearOperator, minres
from skfem import *
from skfem.models.poisson import *
from skfem.assembly import BilinearForm, LinearForm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp



@Functional
def L2uError(w):
    x, y = w.x
    return (w.w - exact_u(x, y))**2


def get_DuError(basis, u):
    duh = basis.interpolate(u).grad
    x = basis.global_coordinates().value
    dx = basis.dx  # quadrature weights
    dux, duy = dexact_u(x[0], x[1])
    return np.sqrt(np.sum(((duh[0] - dux)**2 + (duh[1] - duy)**2) * dx))


def get_D2uError(basis, u):
    dduh = basis.interpolate(u).hess
    x = basis.global_coordinates(
    ).value  # coordinates of quadrature points [x, y]
    dx = basis.dx  # quadrature weights
    duxx, duxy, duyx, duyy = ddexact(x[0], x[1])
    return np.sqrt(
        np.sum(((dduh[0][0] - duxx)**2 + (dduh[0][1] - duxy)**2 +
                (dduh[1][1] - duyy)**2 + (dduh[1][0] - duyx)**2) * dx))


def easy_boundary_penalty(basis):
    '''
    Input basis
    ----------------
    Return D for boundary conditions
    '''

    dofs = basis.find_dofs({
        'left': m.facets_satisfying(lambda x: x[0] == 0),
        'right': m.facets_satisfying(lambda x: x[0] == 1),
        'top': m.facets_satisfying(lambda x: x[1] == 1),
        'buttom': m.facets_satisfying(lambda x: x[1] == 0)
    })

    D = np.concatenate((dofs['left'].nodal['u'], dofs['right'].nodal['u'],
                        dofs['top'].nodal['u'], dofs['buttom'].nodal['u']))
    return D

def solve_problem1(m, element_type='P1'):
    '''
    solver for problem 1 to get loads (todo)
    '''
    if element_type == 'P1':
        element = {'w': ElementTriP1(), 'u': ElementTriMorley()}
    elif element_type == 'P2':
        element = {'w': ElementTriP2(), 'u': ElementTriMorley()}
    else:
        raise Exception("The element not supported")
        
    basis = {
        variable: InteriorBasis(m, e, intorder=5)
        for variable, e in element.items()
    }  # intorder: integration order for quadrature

    K1 = asm(laplace, basis['w'])
    f1 = asm(f_load, basis['w'])

    wh = solve(*condense(K1, f1, D=basis['w'].find_dofs()),
               solver=solver_iter_krylov(Precondition=True, tol=1e-8))

    K2 = epsilon**2 * asm(a_load, basis['u']) + asm(b_load, basis['u'])
    f2 = asm(wv_load, basis['w'], basis['u']) * wh
    uh0 = solve(*condense(K2, f2, D=easy_boundary(basis['u'])),
                solver=solver_iter_krylov(Precondition=True, tol=1e-8))  # cg
    return uh0, basis

def solve_problem1_v1(m, element_type='P1'):
    '''
    old solver for problem 1, can't read f 
    '''
    if element_type == 'P1':
        element = {'w': ElementTriP1(), 'u': ElementTriMorley()}
    elif element_type == 'P2':
        element = {'w': ElementTriP2(), 'u': ElementTriMorley()}
    else:
        raise Exception("The element not supported")
        
    basis = {
        variable: InteriorBasis(m, e, intorder=5)
        for variable, e in element.items()
    }  # intorder: integration order for quadrature

    K1 = asm(laplace, basis['w'])
    f1 = asm(f_load, basis['w'])

    wh = solve(*condense(K1, f1, D=basis['w'].find_dofs()),
               solver=solver_iter_krylov(Precondition=True, tol=1e-8))

    K2 = epsilon**2 * asm(a_load, basis['u']) + asm(b_load, basis['u'])
    f2 = asm(wv_load, basis['w'], basis['u']) * wh
    uh0 = solve(*condense(K2, f2, D=easy_boundary(basis['u'])),
                solver=solver_iter_krylov(Precondition=True, tol=1e-8))  # cg
    return uh0, basis

def solve_problem2(m, element_type='P1'):
    '''
    adding penalty inside (todo)
    '''
    if element_type == 'P1':
        element = {'w': ElementTriP1(), 'u': ElementTriMorley()}
    elif element_type == 'P2':
        element = {'w': ElementTriP2(), 'u': ElementTriMorley()}
    else:
        raise Exception("The element not supported")
        
    basis = {
        variable: InteriorBasis(m, e, intorder=5)
        for variable, e in element.items()
    }  

    @BilinearForm
    def penalty_1(u, v, w):
        return ddot(-dd(u), prod(w.n, w.n)) * dot(grad(v), w.n)

    @BilinearForm
    def penalty_2(u, v, w):
        return ddot(-dd(v), prod(w.n, w.n)) * dot(grad(u), w.n) 


    @BilinearForm
    def penalty_3(u, v, w):
        global mem
        global nn
        global memu
        nn = prod(w.n, w.n)
        mem = w
        memu = u
        return (sigma / w.h) * dot(grad(u), w.n) * dot(grad(v), w.n) 

    K1 = asm(laplace, basis['w'])
    f1 = asm(f_load, basis['w'])

    wh = solve(*condense(K1, f1, D=basis['w'].find_dofs()),
               solver=solver_iter_krylov(Precondition=True, tol=1e-8))
    
    fbasis = FacetBasis(m, element['u'])
    
    p1 = asm(penalty_1, fbasis)
    p2 = asm(penalty_2, fbasis)
    p3 = asm(penalty_3, fbasis)
    P = p1 + p2 + p3

    K2 = epsilon**2 * asm(a_load, basis['u']) + epsilon**2 * P + asm(b_load, basis['u'])
    f2 = asm(wv_load, basis['w'], basis['u']) * wh
    uh0 = solve(*condense(K2, f2, D=easy_boundary_penalty(basis['u'])), solver=solver_iter_krylov(Precondition=True, tol=1e-8))  
    # uh0 = solve(*condense(K2 + P, f2, D=m.boundary_nodes()), solver=solver_iter_krylov(Precondition=True)) 
    return uh0, basis

def solve_problem2_v1(m, element_type='P1'):
    '''
    old solver
    '''
    if element_type == 'P1':
        element = {'w': ElementTriP1(), 'u': ElementTriMorley()}
    elif element_type == 'P2':
        element = {'w': ElementTriP2(), 'u': ElementTriMorley()}
    else:
        raise Exception("The element not supported")
        
    basis = {
        variable: InteriorBasis(m, e, intorder=5)
        for variable, e in element.items()
    }

    K1 = asm(laplace, basis['w'])
    f1 = asm(f_load, basis['w'])

    wh = solve(*condense(K1, f1, D=basis['w'].find_dofs()),
               solver=solver_iter_krylov(Precondition=True, tol=1e-8))
    
    fbasis = FacetBasis(m, element['u'])
    
    p1 = asm(penalty_1, fbasis)
    p2 = asm(penalty_2, fbasis)
    p3 = asm(penalty_3, fbasis)
    P = p1 + p2 + p3

    K2 = epsilon**2 * asm(a_load, basis['u']) + epsilon**2 * P + asm(b_load, basis['u'])
    f2 = asm(wv_load, basis['w'], basis['u']) * wh
    uh0 = solve(*condense(K2, f2, D=easy_boundary_penalty(basis['u'])), solver=solver_iter_krylov(Precondition=True, tol=1e-8))  
    # uh0 = solve(*condense(K2 + P, f2, D=m.boundary_nodes()), solver=solver_iter_krylov(Precondition=True)) 
    return uh0, basis


"""This module contains utility functions such as convenient access to
SciPy linear solvers."""

import sys
from skfem.assembly import BilinearForm, LinearForm
from skfem.models.poisson import *
from scipy.sparse.linalg import LinearOperator, minres
from skfem.helpers import d, dd, ddd, dot, ddot, grad, dddot, prod
from skfem import *
import warnings
from typing import Optional, Union, Tuple, Callable, Dict
from inspect import signature
from skfem.visuals.matplotlib import draw, plot
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as spg
import scipy.sparse.linalg as spl
from numpy import ndarray
from scipy.sparse import spmatrix

import pyamg

from skfem.assembly import asm, BilinearForm, LinearForm, DofsView
from skfem.assembly.basis import Basis
from skfem.element import ElementVectorH1


# custom types for describing input and output values

LinearSolver = Callable[[spmatrix, ndarray], ndarray]
EigenSolver = Callable[[spmatrix, spmatrix], Tuple[ndarray, ndarray]]
CondensedSystem = Union[spmatrix,
                        Tuple[spmatrix, ndarray],
                        Tuple[spmatrix, spmatrix],
                        Tuple[spmatrix, ndarray, ndarray],
                        Tuple[spmatrix, ndarray, ndarray, ndarray],
                        Tuple[spmatrix, spmatrix, ndarray, ndarray]]
DofsCollection = Union[ndarray, DofsView, Dict[str, DofsView]]


# preconditioners, e.g. for :func:`skfem.utils.solver_iter_krylov`


def build_pc_ilu(A: spmatrix,
                 drop_tol: Optional[float] = 1e-4,
                 fill_factor: Optional[float] = 20) -> spl.LinearOperator:
    """Incomplete LU preconditioner."""
    P = spl.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    M = spl.LinearOperator(A.shape, matvec=P.solve)
    return M


def build_pc_diag(A: spmatrix) -> spmatrix:
    """Diagonal preconditioner."""
    return sp.spdiags(1.0/A.diagonal(), 0, A.shape[0], A.shape[0])


# solvers for :func:`skfem.utils.solve`


def solver_eigen_scipy(**kwargs) -> EigenSolver:
    """Solve generalized eigenproblem using SciPy (ARPACK).

    Returns
    -------
    EigenSolver
        A solver function that can be passed to :func:`solve`.

    """
    params = {
        'sigma': 10,
        'k': 5,
        'mode': 'normal',
    }
    params.update(kwargs)

    def solver(K, M, **solve_time_kwargs):
        params.update(solve_time_kwargs)
        from scipy.sparse.linalg import eigsh
        return eigsh(K, M=M, **params)

    return solver


def solver_direct_scipy(**kwargs) -> LinearSolver:
    """The default linear solver of SciPy."""
    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        return spl.spsolve(A, b, **kwargs)
    return solver


def solver_iter_mgcg_iter(krylov: Optional[LinearSolver] = spl.cg, verbose: Optional[bool] = False, **kwargs) -> LinearSolver:
    """MGCG iterative linear solver.

    Parameters
    ----------
    krylov
        A Krylov iterative linear solver, like, and by default,
        :func:`scipy.sparse.linalg.cg`
    verbose
        If True, print the norm of the iterate.

    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.
    And prints num of iterations
    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)

        import pyamg
        ml = pyamg.ruge_stuben_solver(A)
        kwargs['M'] = ml.aspreconditioner()  # params to be developed

        sol, info, iter = krylov(A, b, **{'callback': callback, **kwargs})
        print('mgcg total interation steps:', iter)
        if info > 0:
            warnings.warn("Convergence not achieved!")
        elif info == 0 and verbose:
            print(f"{krylov.__name__} converged to "
                  + f"tol={kwargs.get('tol', 'default')} and "
                  + f"atol={kwargs.get('atol', 'default')}")
        return sol

    return solver


def solver_iter_mgcg(krylov: Optional[LinearSolver] = spl.cg,
                     verbose: Optional[bool] = False,
                     **kwargs) -> LinearSolver:
    """MGCG iterative linear solver.

    Parameters
    ----------
    krylov
        A Krylov iterative linear solver, like, and by default,
        :func:`scipy.sparse.linalg.cg`
    verbose
        If True, print the norm of the iterate.

    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.

    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)

        import pyamg
        ml = pyamg.ruge_stuben_solver(A)
        kwargs['M'] = ml.aspreconditioner()  # params to be developed

        try:
            sol, info, _ = krylov(A, b, **{'callback': callback, **kwargs})
        except:
            sol, info = krylov(A, b, **{'callback': callback, **kwargs})
        if info > 0:
            warnings.warn("Convergence not achieved!")
        elif info == 0 and verbose:
            print(f"{krylov.__name__} converged to "
                  + f"tol={kwargs.get('tol', 'default')} and "
                  + f"atol={kwargs.get('atol', 'default')}")
        return sol

    return solver


def solver_iter_pyamg(verbose: Optional[bool] = False,
                      **kwargs) -> LinearSolver:
    """Pyamg iterative linear solver.

    Parameters
    ----------
    verbose
        If True, print the norm of the iterate.

    Any remaining keyword arguments are passed on to the solver, in particular
    tol and atol, the tolerances, maxiter, and M, the preconditioner.  If the
    last is omitted, a diagonal preconditioner is supplied using
    :func:`skfem.utils.build_pc_diag`.

    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.

    """

    def my_pyamg(A, b, **kwargs):
        '''
        solver for pyamg
        '''
        import pyamg
        ml = pyamg.ruge_stuben_solver(A)
        # print(ml)
        x = ml.solve(b, tol=1e-10, maxiter=10000, callback=callback)
        return x, np.linalg.norm(b-A*x)

    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        sol, info = my_pyamg(A, b, **{'callback': callback, **kwargs})
        if info > 1 and verbose:
            print('Warning: residual norm =', info)
        return sol

    return solver


def solver_iter_krylov_iter(krylov: Optional[LinearSolver] = spl.cg,
                            verbose: Optional[bool] = False,
                            **kwargs) -> LinearSolver:
    """Krylov-subspace iterative linear solver.

    Parameters
    ----------
    krylov
        A Krylov iterative linear solver, like, and by default,
        :func:`scipy.sparse.linalg.cg`
    verbose
        If True, print the norm of the iterate.

    Any remaining keyword arguments are passed on to the solver, in particular
    tol and atol, the tolerances, maxiter, and M, the preconditioner.  If the
    last is omitted, a diagonal preconditioner is supplied using
    :func:`skfem.utils.build_pc_diag`.

    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.
    And prints num of iters
    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        pre = False
        if 'Precondition' in kwargs:
            if kwargs['Precondition'] == True:
                pre = True
            kwargs.pop('Precondition')
        if 'M' not in kwargs and pre:
            # print('build_pc_diag(A) enabled')
            # pass
            kwargs['M'] = build_pc_diag(A)
        # print(kwargs['M'])
        sol, info, iter = krylov(A, b, **{'callback': callback, **kwargs})
        print(krylov.__name__, 'total interation steps:', iter)
        if info > 0:
            warnings.warn("Convergence not achieved!")
        elif info == 0 and verbose:
            # print(info)
            print(f"{krylov.__name__} converged to "
                  + f"tol={kwargs.get('tol', 'default')} and "
                  + f"atol={kwargs.get('atol', 'default')}")
        return sol

    return solver


def solver_iter_krylov(krylov: Optional[LinearSolver] = spl.cg,
                       verbose: Optional[bool] = False,
                       **kwargs) -> LinearSolver:
    """Krylov-subspace iterative linear solver.

    Parameters
    ----------
    krylov
        A Krylov iterative linear solver, like, and by default,
        :func:`scipy.sparse.linalg.cg`
    verbose
        If True, print the norm of the iterate.

    Any remaining keyword arguments are passed on to the solver, in particular
    tol and atol, the tolerances, maxiter, and M, the preconditioner.  If the
    last is omitted, a diagonal preconditioner is supplied using
    :func:`skfem.utils.build_pc_diag`.

    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.

    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        pre = False
        if 'Precondition' in kwargs:
            if kwargs['Precondition'] == True:
                pre = True
            kwargs.pop('Precondition')
        if 'M' not in kwargs and pre:
            # print('build_pc_diag(A) enabled')
            # pass
            kwargs['M'] = build_pc_diag(A)
        # print(kwargs['M'])
        sol, info, _ = krylov(A, b, **{'callback': callback, **kwargs})
        if info > 0:
            warnings.warn("Convergence not achieved!")
        elif info == 0 and verbose:
            # print(info)
            print(f"{krylov.__name__} converged to "
                  + f"tol={kwargs.get('tol', 'default')} and "
                  + f"atol={kwargs.get('atol', 'default')}")
        return sol

    return solver


def solver_iter_pcg(**kwargs) -> LinearSolver:
    """Conjugate gradient solver, specialized from solver_iter_krylov"""
    return solver_iter_krylov(**kwargs)


# solve and condense


def solve(A: spmatrix,
          b: Union[ndarray, spmatrix],
          x: Optional[ndarray] = None,
          I: Optional[ndarray] = None,
          solver: Optional[Union[LinearSolver, EigenSolver]] = None,
          **kwargs) -> ndarray:
    """Solve a linear system or a generalized eigenvalue problem.

    The remaining keyword arguments are passed to the solver.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector or the mass matrix of a generalized
        eigenvalue problem.
    solver
        Choose one of the following solvers:
        :func:`skfem.utils.solver_direct_scipy` (default),
        :func:`skfem.utils.solver_eigen_scipy` (default),
        :func:`skfem.utils.solver_iter_pcg`,
        :func:`skfem.utils.solver_iter_krylov`.

    """
    if solver is None:
        if isinstance(b, spmatrix):
            solver = solver_eigen_scipy(**kwargs)
        elif isinstance(b, ndarray):
            solver = solver_direct_scipy(**kwargs)

    if x is not None and I is not None:
        if isinstance(b, spmatrix):
            L, X = solver(A, b, **kwargs)
            y = np.tile(x.copy()[:, None], (1, X.shape[1]))
            y[I] = X
            return L, y
        else:
            y = x.copy()
            y[I] = solver(A, b, **kwargs)
            return y
    else:
        return solver(A, b, **kwargs)


def _flatten_dofs(S: DofsCollection) -> ndarray:
    if S is None:
        return None
    else:
        if isinstance(S, ndarray):
            return S
        elif isinstance(S, DofsView):
            return S.flatten()
        elif isinstance(S, dict):
            return np.unique(np.concatenate([S[key].flatten() for key in S]))
        raise NotImplementedError("Unable to flatten the given set of DOFs.")


def condense(A: spmatrix,
             b: Union[ndarray, spmatrix] = None,
             x: ndarray = None,
             I: DofsCollection = None,
             D: DofsCollection = None,
             expand: bool = True) -> CondensedSystem:
    """Eliminate degrees-of-freedom from a linear system.

    The user should provide the linear system ``A`` and ``b``
    and either the set of DOFs to eliminate (``D``) or the set
    of DOFs to keep (``I``).  Optionally, nonzero values for
    the eliminated DOFs can be supplied via ``x``.

    .. note::

        Supports also generalized eigenvalue problems
        where ``b`` is a matrix.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector or the mass matrix for generalized
        eigenvalue problems.
    x
        The values of the condensed degrees-of-freedom. If not given, assumed
        to be zero.
    I
        The set of degree-of-freedom indices to keep.
    D
        The set of degree-of-freedom indices to dismiss.
    expand
        If `True` (default), returns also `x` and `I`. As a consequence,
        :func:`skfem.utils.solve` will expand the solution vector
        automatically.

    Returns
    -------
    CondensedSystem
        The condensed linear system and (optionally) information about
        the boundary values.

    """
    D = _flatten_dofs(D)
    I = _flatten_dofs(I)

    if x is None:
        x = np.zeros(A.shape[0])

    if I is None and D is None:
        raise Exception("Either I or D must be given!")
    elif I is None and D is not None:
        I = np.setdiff1d(np.arange(A.shape[0]), D)
    elif D is None and I is not None:
        D = np.setdiff1d(np.arange(A.shape[0]), I)
    else:
        raise Exception("Give only I or only D!")

    if b is None:
        ret_value = (A[I].T[I].T,)
    else:
        if isinstance(b, spmatrix):
            # generalized eigenvalue problem: don't modify rhs
            Aout = A[I].T[I].T
            bout = b[I].T[I].T
        elif isinstance(b, ndarray):
            Aout = A[I].T[I].T
            bout = b[I] - A[I].T[D].T @ x[D]
        else:
            raise Exception("The second arg type not supported.")
        ret_value = (Aout, bout)

    if expand:
        ret_value += (x, I)

    return ret_value if len(ret_value) > 1 else ret_value[0]


# additional utilities


def rcm(A: spmatrix,
        b: ndarray) -> Tuple[spmatrix, ndarray, ndarray]:
    """Reverse Cuthill-McKee ordering."""
    p = spg.reverse_cuthill_mckee(A, symmetric_mode=False)
    return A[p].T[p].T, b[p], p


def adaptive_theta(est, theta=0.5, max=None):
    """For choosing which elements to refine in an adaptive strategy."""
    if max is None:
        return np.nonzero(theta * np.max(est) < est)[0]
    else:
        return np.nonzero(theta * max < est)[0]


def project(fun,
            basis_from: Basis = None,
            basis_to: Basis = None,
            diff: int = None,
            I: ndarray = None,
            expand: bool = False,
            solver: Optional[Union[LinearSolver, EigenSolver]] = None) -> ndarray:
    """Projection from one basis to another.

    Parameters
    ----------
    fun
        A solution vector or a function handle.
    basis_from
        The finite element basis to project from.
    basis_to
        The finite element basis to project to.
    diff
        Differentiate with respect to the given dimension.
    I
        Index set for limiting the projection to a subset.
    expand
        Passed to :func:`skfem.utils.condense`.

    Returns
    -------
    ndarray
        The projected solution vector.

    """

    @BilinearForm
    def mass(u, v, w):
        p = u * v
        return sum(p) if isinstance(basis_to.elem, ElementVectorH1) else p

    @LinearForm
    def funv(v, w):
        if len(signature(fun).parameters) == 1:
            p = fun(w.x) * v
        else:
            warnings.warn("The function provided to 'project' should "
                          "take only one argument in the future.",
                          DeprecationWarning)
            p = fun(*w.x) * v
        return sum(p) if isinstance(basis_to.elem, ElementVectorH1) else p

    @BilinearForm
    def deriv(u, v, w):
        from skfem.helpers import grad
        du = grad(u)
        return du[diff] * v

    M = asm(mass, basis_to)

    if not isinstance(fun, ndarray):
        f = asm(funv, basis_to)
    else:
        if diff is not None:
            f = asm(deriv, basis_from, basis_to) @ fun
        else:
            # print(asm(mass, basis_from, basis_to).shape)
            # print(fun.shape)
            f = asm(mass, basis_from, basis_to) @ fun

    if I is not None:
        # print('aaa')
        if solver is None:
            return solve(*condense(M, f, I=I, expand=expand))
        else:
            return solve(*condense(M, f, I=I, expand=expand), solver=solver_iter_krylov(solver, tol=1e-11))
    return solve(M, f)


def pproject(fun,
             basis_from: Basis = None,
             basis_to: Basis = None,
             diff: int = None,
             I: ndarray = None,
             expand: bool = False,
             solver: Optional[Union[LinearSolver, EigenSolver]] = None) -> ndarray:
    """Projection from one basis to another.

    Parameters
    ----------
    fun
        A solution vector or a function handle.
    basis_from
        The finite element basis to project from.
    basis_to
        The finite element basis to project to.
    diff
        Differentiate with respect to the given dimension.
    I
        Index set for limiting the projection to a subset.
    expand
        Passed to :func:`skfem.utils.condense`.

    Returns
    -------
    ndarray
        The projected solution vector.

    """

    @BilinearForm
    def mass(u, v, w):
        p = u * v
        return sum(p) if isinstance(basis_to.elem, ElementVectorH1) else p

    @LinearForm
    def funv(v, w):
        if len(signature(fun).parameters) == 1:
            p = fun(w.x) * v
        else:
            warnings.warn("The function provided to 'project' should "
                          "take only one argument in the future.",
                          DeprecationWarning)
            p = fun(*w.x) * v
        return sum(p) if isinstance(basis_to.elem, ElementVectorH1) else p

    @BilinearForm
    def deriv(u, v, w):
        from skfem.helpers import grad
        du = grad(u)
        return du[diff] * v

    M = asm(mass, basis_to)

    if not isinstance(fun, ndarray):
        f = asm(funv, basis_to)
    else:
        if diff is not None:
            f = asm(deriv, basis_from, basis_to) @ fun
        else:
            f = asm(mass, basis_from, basis_to) @ fun

    if I is not None:
        # print('aaa')
        if solver is None:
            return solve(*condense(M, f, I=I, expand=expand))
        else:
            return solve(*condense(M, f, I=I, expand=expand), solver=solver_iter_krylov(solver, tol=1e-11))
    return solve(M, f)


# for backwards compatibility
def L2_projection(a, b, c=None):
    """Superseded by :func:`skfem.utils.project`."""
    return project(a, basis_to=b, I=c)


def derivative(a, b, c, d=0):
    """Superseded by :func:`skfem.utils.project`."""
    return project(a, basis_from=b, basis_to=c, diff=d)


# functions for loads and boundary

pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
# atan = np.arctan2

# parameters

# end of parameters

# print parameters

# functions


def arctan3(x, y):
    theta = np.arctan2(y, x)
    theta[theta < 0] += 2 * pi
    return theta

#############################
def exact_u(x, y):
    theta = arctan3(y, x)
    return (x**2 + y**2)**(5/6) * sin(5*theta/3)


def dexact_u(x, y):
    theta = arctan3(y, x)
    dux = (5*x*sin((5*theta)/3))/(3*(x**2 + y**2)**(1/6)) - \
        (5*y*cos((5*theta)/3)*(x**2 + y**2)**(5/6))/(3*(y**2 + x**2))
    duy = (5*y*sin((5*theta)/3))/(3*(x**2 + y**2)**(1/6)) + \
        (5*x*cos((5*theta)/3)*(x**2 + y**2)**(5/6))/(3*(y**2 + x**2))
    return dux, duy


def ddexact(x, y):
    theta = arctan3(y, x)
    duxx = -(10*(y**2*sin((5*theta)/3) - x**2*sin((5*theta)/3) +
                 2*x*y*cos((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
    duxy = (10*(x**2*cos((5*theta)/3) - y**2*cos((5*theta)/3) +
                2*x*y*sin((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
    duyx = duxy
    duyy = (10*(y**2*sin((5*theta)/3) - x**2*sin((5*theta)/3) +
                2*x*y*cos((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
    return duxx, duxy, duyx, duyy

# @LinearForm
# def f_load(v, w):
#     '''
#     for $(f, x_{h})$
#     '''
#     return 0
########################################


# @LinearForm
# def f_load(v, w):
#     '''
#     for $(f, x_{h})$
#     '''
#     lu = 0
#     llu = 0
#     return (epsilon**2 * llu - lu) * v

# def exact_u(x, y):
#     return x*y

# def dexact_u(x, y):
#     dux = y
#     duy = x
#     return dux, duy

# def ddexact(x, y):
#     duxx = 0
#     duxy = 1
#     duyx = duxy
#     duyy = 0
#     return duxx, duxy, duyx, duyy

# ###################################x**2*y**2*(x - 1)**2*(y - 1)**2
# @LinearForm
# def f_load(v, w):
#     '''
#     for $(f, x_{h})$
#     '''
#     x, y = w.x
#     # lu = 0
#     # llu = 0
#     # return (epsilon**2 * llu - lu) * v
#     return (24*ep**2*x**4 - 48*ep**2*x**3 + 288*ep**2*x**2*y**2 - 288*ep**2*x**2*y + 72*ep**2*x**2 - 288*ep**2*x*y**2 + 288*ep**2*x*y - 48*ep**2*x + 24*ep**2*y**4 - 48*ep**2*y**3 + 72*ep**2*y**2 - 48*ep**2*y + 8*ep**2 - 12*x**4*y**2 + 12*x**4*y - 2*x**4 + 24*x**3*y**2 - 24*x**3*y + 4*x**3 - 12*x**2*y**4 + 24*x**2*y**3 - 24*x**2*y**2 + 12*x**2*y - 2*x**2 + 12*x*y**4 - 24*x*y**3 + 12*x*y**2 - 2*y**4 + 4*y**3 - 2*y**2) * v

# def exact_u(x, y):
#     return x**2*y**2*(x - 1)**2*(y - 1)**2

# def dexact_u(x, y):
#     dux = 2*x*y**2*(y - 1)**2*(2*x**2 - 3*x + 1)
#     duy = 2*x**2*y*(x - 1)**2*(2*y**2 - 3*y + 1)
#     return dux, duy

# def ddexact(x, y):
#     duxx = 2*y**2*(y - 1)**2*(6*x**2 - 6*x + 1)
#     duxy = 4*x*y*(2*x**2 - 3*x + 1)*(2*y**2 - 3*y + 1)
#     duyx = duxy
#     duyy = 2*x**2*(x - 1)**2*(6*y**2 - 6*y + 1)
#     return duxx, duxy, duyx, duyy
###################################

###################################x*y*(x - 1)*(y - 1)
# @LinearForm
# def f_load(v, w):
#     '''
#     for $(f, x_{h})$
#     '''
#     x, y = w.x
#     # lu = 0
#     # llu = 0
#     # return (epsilon**2 * llu - lu) * v
#     return (8*ep**2 - 2*x*(x - 1) - 2*y*(y - 1)) * v

# def exact_u(x, y):
#     return x*y*(x - 1)*(y - 1)

# def dexact_u(x, y):
#     dux = y*(2*x - 1)*(y - 1)
#     duy = x*(2*y - 1)*(x - 1)
#     return dux, duy

# def ddexact(x, y):
#     duxx = 2*y**2
#     duxy = 4*x*y
#     duyx = duxy
#     duyy = 2*x**2
#     return duxx, duxy, duyx, duyy
# ###################################

###################################
# @LinearForm
# def f_load(v, w):
#     '''
#     for $(f, x_{h})$
#     '''
#     x, y = w.x
#     # lu = 0
#     # llu = 0
#     # return (epsilon**2 * llu - lu) * v
#     return ((24*x**2*(x - 1)**2 + 24*y**2*(y - 1)**2 + 2*(4*x*(2*x - 2) + 2*(x - 1)**2 + 2*x**2)*(4*y*(2*y - 2) + 2*(y - 1)**2 + 2*y**2) + 72)*ep**2 - (y**2*(y - 1)**2 + 2)*(4*x*(2*x - 2) + 2*(x - 1)**2 + 2*x**2) - (x**2*(x - 1)**2 + 1)*(4*y*(2*y - 2) + 2*(y - 1)**2 + 2*y**2)) * v

# def exact_u(x, y):
#     return (x**2*(x - 1)**2 + 1)*(y**2*(y - 1)**2 + 2)

# def dexact_u(x, y):
#     dux = (y**2*(y - 1)**2 + 2)*(2*x*(x - 1)**2 + x**2*(2*x - 2))
#     duy = (x**2*(x - 1)**2 + 1)*(2*y*(y - 1)**2 + y**2*(2*y - 2))
#     return dux, duy

# def ddexact(x, y):
#     duxx = (12*x**2 - 12*x + 2)*(y**4 - 2*y**3 + y**2 + 2)
#     duxy = 4*x*y*(2*x**2 - 3*x + 1)*(2*y**2 - 3*y + 1)
#     duyx = duxy
#     duyy = (12*y**2 - 12*y + 2)*(x**4 - 2*x**3 + x**2 + 1)
#     return duxx, duxy, duyx, duyy
###################################


# def exact_u(x, y):
#     out = np.zeros_like(x)
#     if x.ndim == 1:
#         for i in range(x.shape[0]):
#             if x[i] != 0:
#                 out[i] = (x[i]**2 + y[i]**2)**(5 / 6) * sin(5 * atan(y[i] / x[i]) / 3)
#             elif y[i] > 0:
#                 out[i] = (x[i]**2 + y[i]**2)**(5 / 6) * sin(pi/2 * 5 / 3)
#             else:
#                 out[i] = (x[i]**2 + y[i]**2)**(5 / 6) * sin(3/2 * pi * 5 / 3)
#     else:
#         out = (x**2 + y**2)**(5/6) * sin(5*atan(y / x)/3)
#     return out

# def get_theta(x, y):
#     import math
#     import cmath
#     theta = np.zeros_like(x)
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             # print(x)
#             r, rad = cmath.polar(complex(x[i, j], y[i, j]))
#             theta[i, j] = math.degrees(rad)
#     return theta


# def dexact_u(x, y):
#     theta = get_theta(x, y)
#     dux = (5*x*sin((5*theta)/3))/(3*(x**2 + y**2)**(1/6)) - (5*y*cos((5*theta)/3)*(x**2 + y**2)**(5/6))/(3*(y**2 + x**2))
#     duy = (5*y*sin((5*theta)/3))/(3*(x**2 + y**2)**(1/6)) + (5*x*cos((5*theta)/3)*(x**2 + y**2)**(5/6))/(3*(y**2 + x**2))
#     return dux, duy


# def ddexact(x, y):
#     theta = get_theta(x, y)
#     duxx = -(10*(y**2*sin((5*theta)/3) - x**2*sin((5*theta)/3) + 2*x*y*cos((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
#     duxy = (10*(x**2*cos((5*theta)/3) - y**2*cos((5*theta)/3) + 2*x*y*sin((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
#     duyx = duxy
#     duyy = (10*(y**2*sin((5*theta)/3) - x**2*sin((5*theta)/3) + 2*x*y*cos((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
#     return duxx, duxy, duyx, duyy


def easy_boundary_penalty(m, basis):
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


def easy_boundary(m, basis):
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
                        dofs['top'].nodal['u'], dofs['buttom'].nodal['u'],
                        dofs['left'].facet['u_n'], dofs['right'].facet['u_n'],
                        dofs['top'].facet['u_n'], dofs['buttom'].facet['u_n']))
    return D


def solve_problem1__(m, element_type='P1', solver_type='pcg', intorder=6, tol=1e-8, epsilon=1e-6):
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
    }  # intorder: integration order for quadrature

    K1 = asm(laplace, basis['w'])
    f1 = asm(f_load, basis['w'])

    if solver_type == 'amg':
        wh = solve(
            *condense(K1, f1, D=basis['w'].find_dofs()), solver=solver_iter_pyamg(tol=tol))
    elif solver_type == 'pcg':
        wh = solve(*condense(K1, f1, D=basis['w'].find_dofs()),
                   solver=solver_iter_krylov(Precondition=True, tol=tol))
    elif solver_type == 'mgcg':
        wh = solve(
            *condense(K1, f1, D=basis['w'].find_dofs()), solver=solver_iter_mgcg(tol=tol))
    else:
        raise Exception("Solver not supported")

    K2 = epsilon**2 * asm(a_load, basis['u']) + asm(b_load, basis['u'])
    f2 = asm(wv_load, basis['w'], basis['u']) * wh

    if solver_type == 'amg':
        uh0 = solve(*condense(K2, f2, D=easy_boundary(m,
                                                      basis['u'])), solver=solver_iter_pyamg(tol=tol))
    elif solver_type == 'pcg':
        uh0 = solve(*condense(K2, f2, D=easy_boundary(m,
                                                      basis['u'])), solver=solver_iter_krylov(Precondition=True, tol=tol))
    elif solver_type == 'mgcg':
        uh0 = solve(*condense(K2, f2, D=easy_boundary(m,
                                                      basis['u'])), solver=solver_iter_mgcg(tol=tol))
    else:
        raise Exception("Solver not supported")

    return uh0, basis


def solve_problem2(m, element_type='P1', solver_type='pcg', intorder=6, tol=1e-8, epsilon=1e-6):
    '''
    adding mgcg solver for problem 2
    '''
    if element_type == 'P1':
        element = {'w': ElementTriP1(), 'u': ElementTriMorley()}
    elif element_type == 'P2':
        element = {'w': ElementTriP2(), 'u': ElementTriMorley()}
    else:
        raise Exception("The element not supported")

    basis = {
        variable: InteriorBasis(m, e, intorder=intorder)
        for variable, e in element.items()
    }

    K1 = asm(laplace, basis['w'])
    f1 = asm(f_load, basis['w'])

    if solver_type == 'amg':
        wh = solve(
            *condense(K1, f1, D=basis['w'].find_dofs()), solver=solver_iter_pyamg(tol=tol))
    elif solver_type == 'pcg':
        wh = solve(*condense(K1, f1, D=basis['w'].find_dofs()),
                   solver=solver_iter_krylov(Precondition=True, tol=tol))
    elif solver_type == 'mgcg':
        wh = solve(
            *condense(K1, f1, D=basis['w'].find_dofs()), solver=solver_iter_mgcg(tol=tol))
    else:
        raise Exception("Solver not supported")

    fbasis = FacetBasis(m, element['u'])

    p1 = asm(penalty_1, fbasis)
    p2 = asm(penalty_2, fbasis)
    p3 = asm(penalty_3, fbasis)
    P = p1 + p2 + p3

    K2 = epsilon**2 * asm(a_load, basis['u']) + \
        epsilon**2 * P + asm(b_load, basis['u'])
    f2 = asm(wv_load, basis['w'], basis['u']) * wh

    if solver_type == 'amg':
        uh0 = solve(*condense(K2, f2, D=easy_boundary_penalty(m,
                                                              basis['u'])), solver=solver_iter_pyamg(tol=tol))
    elif solver_type == 'pcg':
        uh0 = solve(*condense(K2, f2, D=easy_boundary_penalty(m,
                                                              basis['u'])), solver=solver_iter_krylov(Precondition=True, tol=tol))
    elif solver_type == 'mgcg':
        uh0 = solve(*condense(K2, f2, D=easy_boundary_penalty(m,
                                                              basis['u'])), solver=solver_iter_mgcg(tol=tol))
    else:
        raise Exception("Solver not supported")

    return uh0, basis, fbasis


@Functional
def L2pnvError(w):
    return (w.h * dot(w['n'].value, w['w'].grad))**2


@BilinearForm
def a_load(u, v, w):
    '''
    for $a_{h}$
    '''
    return ddot(dd(u), dd(v))


@BilinearForm
def b_load(u, v, w):
    '''
    for $b_{h}$
    '''
    return dot(grad(u), grad(v))


@BilinearForm
def wv_load(u, v, w):
    '''
    for $(\nabla \chi_{h}, \nabla_{h} v_{h})$
    '''
    return dot(grad(u), grad(v))


@BilinearForm
def penalty_1(u, v, w):
    return ddot(-dd(u), prod(w.n, w.n)) * dot(grad(v), w.n)


@BilinearForm
def penalty_2(u, v, w):
    return ddot(-dd(v), prod(w.n, w.n)) * dot(grad(u), w.n)


@BilinearForm
def penalty_3(u, v, w):
    return (sigma / w.h) * dot(grad(u), w.n) * dot(grad(v), w.n)


@BilinearForm
def laplace(u, v, w):
    '''
    for $(\nabla w_{h}, \nabla \chi_{h})$
    '''
    return dot(grad(u), grad(v))


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


# def exact_u(x, y):
#     return (sin(pi * x) * sin(pi * y))**2


# def dexact_u(x, y):
#     dux = 2 * pi * cos(pi * x) * sin(pi * x) * sin(pi * y)**2
#     duy = 2 * pi * cos(pi * y) * sin(pi * x)**2 * sin(pi * y)
#     return dux, duy

# def ddexact(x, y):
#     duxx = 2 * pi**2 * cos(pi * x)**2 * sin(pi * y)**2 - 2 * pi**2 * sin(
#         pi * x)**2 * sin(pi * y)**2
#     duxy = 2 * pi * cos(pi * x) * sin(pi * x) * 2 * pi * cos(pi * y) * sin(
#         pi * y)
#     duyx = duxy
#     duyy = 2 * pi**2 * cos(pi * y)**2 * sin(pi * x)**2 - 2 * pi**2 * sin(
#         pi * y)**2 * sin(pi * x)**2
#     return duxx, duxy, duyx, duyy

def show_result(L2s, H1s, H2s, epus):

    print('  h    L2u   H1u   H2u   epu')
    for i in range(H2s.shape[0] - 1):
        print(
            '2^-' + str(i + 2), ' {:.2f}  {:.2f}  {:.2f}  {:.2f}'.format(
                -np.log2(L2s[i + 1] / L2s[i]), -np.log2(H1s[i + 1] / H1s[i]),
                -np.log2(H2s[i + 1] / H2s[i]),
                -np.log2(epus[i + 1] / epus[i])))
        print(
            '2^-' + str(i + 2), ' {:.3e}  {:.3e}  {:.3e}  {:.3e}'.format(
                L2s[i + 1], H1s[i + 1],
                H2s[i + 1],
                epus[i + 1]))

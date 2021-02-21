import numpy as np
from skfem import *
from skfem.helpers import *
from skfem.assembly import BilinearForm, LinearForm
from skfem.visuals.matplotlib import draw, plot, show

##############################
def exact_u(x, y):
    return x*y

def dexact_u(x, y):
    dux = y
    duy = x
    return dux, duy
##############################

# def exact_u(x, y):
#     return (x * y * (1 - x) * (1 - y))**2 + 1


# def dexact_u(x, y):
#     dux = (y * (1 - y))**2 * (4 * x**3 - 6 * x**2 + 2 * x)
#     duy = (x * (1 - x))**2 * (4 * y**3 - 6 * y**2 + 2 * y)
#     return dux, duy


def exact_un(x, y):
    nx = -1 * (x == 0) + 1 * (x == 1)
    ny = -1 * (y == 0) + 1 * (y == 1)
    dux, duy = dexact_u(x, y)
    return nx * dux + ny * duy



m = MeshTri()
m.refine(6)
element = ElementTriMorley()

basis = InteriorBasis(m, element)
K = asm(BilinearForm(lambda u, v, w: ddot(dd(u), dd(v))), basis)
f = asm(LinearForm(lambda v, w: 0 * v), basis)

#######################
# @LinearForm
# def mybih(v, w):
#     x, y = w.x
#     return (24 * ((x**2 - x + 1)**2 + (y**2 - y + 1)**2 + 12 * (x - 1) *
#                   (y - 1) * x * y) - 40) * v

# f = asm(mybih, basis)
#######################


# all boundary dofs
boundary_dofs = basis.find_dofs()['all'].all()
# nodal dofs on boundary
boundary_dofs_u = np.array([i for i in boundary_dofs if i in basis.nodal_dofs[0]])
# facet dofs on boundary
boundary_dofs_un = np.array([i for i in boundary_dofs if i in basis.facet_dofs[0]])

# uh0 = np.zeros(basis.N)
# uh0 = project(lambda x, y: x * y, basis_to=basis)
uh0 = project(exact_u, basis_to=basis)
# boundary condition for 'u'
# uh0[boundary_dofs_u] = exact_u(basis.doflocs[0][boundary_dofs_u], basis.doflocs[1][boundary_dofs_u])
# boundary condition for 'u_n'
# uh0[boundary_dofs_un] = exact_un(basis.doflocs[0][boundary_dofs_un], basis.doflocs[1][boundary_dofs_un])
# print(basis.get_dofs().nodal_ix)
# print(basis.get_dofs().facet_ix)
# print(basis.get_dofs().all())
# print(boundary_dofs_u)
# print(boundary_dofs_un)
# print(boundary_dofs)
uh0 = solve(*condense(K, f, uh0, D=boundary_dofs))

x = basis.doflocs[0]
y = basis.doflocs[1]
u = exact_u(x, y)
u[basis.facet_dofs[0]] = exact_un(x[basis.facet_dofs[0]], y[basis.facet_dofs[0]])
# print(uh0)

plot(basis, u-uh0, colorbar=True)
show()

L2_Error = np.sqrt(Functional(lambda w: (w.w - exact_u(w.x[0], w.x[1]))**2).assemble(basis, w=basis.interpolate(uh0).value))
print(L2_Error)
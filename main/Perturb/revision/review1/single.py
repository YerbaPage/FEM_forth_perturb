import numpy as np
from skfem import *
from skfem.helpers import *
from skfem.assembly import BilinearForm, LinearForm
from skfem.visuals.matplotlib import draw, plot, show


def exact_u(x, y):
    return x*y

def dexact_u(x, y):
    dux = y
    duy = x
    return dux, duy

def exact_un(x, y):
    nx = -1 * (x == 0) + 1 * (x == 1)
    ny = -1 * (y == 0) + 1 * (y == 1)
    dux, duy = dexact_u(x, y)
    return nx * dux + ny * duy

m = MeshTri()
m.refine(1)
element = ElementTriMorley()

basis = InteriorBasis(m, element)
K = asm(BilinearForm(lambda u, v, w: ddot(dd(u), dd(v))), basis)
f = asm(LinearForm(lambda v, w: 0 * v), basis)

# all boundary dofs
boundary_dofs = basis.find_dofs()['all'].all()
# nodal dofs on boundary
boundary_dofs_u = np.array([i for i in boundary_dofs if i in basis.nodal_dofs[0]])
# facet dofs on boundary
boundary_dofs_un = np.array([i for i in boundary_dofs if i in basis.facet_dofs[0]])

uh0 = np.zeros(basis.N)
# boundary condition for 'u'
uh0[boundary_dofs_u] = exact_u(basis.doflocs[0][boundary_dofs_u], basis.doflocs[1][boundary_dofs_u])
# boundary condition for 'u_n'
uh0[boundary_dofs_un] = exact_un(basis.doflocs[0][boundary_dofs_un], basis.doflocs[1][boundary_dofs_un])
print(basis.get_dofs().nodal_ix)
print(basis.get_dofs().facet_ix)
print(basis.get_dofs().all())
print(boundary_dofs_u)
print(boundary_dofs_un)
print(boundary_dofs)
uh0 = solve(*condense(K, f, uh0, D=boundary_dofs_u))

x = basis.doflocs[0]
y = basis.doflocs[1]
u = exact_u(x, y)
# print(uh0)
plot(basis, u-uh0, colorbar=True)
# show()

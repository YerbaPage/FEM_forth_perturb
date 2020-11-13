# Krichholf plate blending problem

$ \frac{Ed^3}{12(1-\nu^2)} \Delta^2 u = f \quad \text{in $\Omega$} $ 
where $ \Omega = (0,1)^2$ , $ f $ is a perpendicular force,
$E$  and $\nu$  are material parameters.
In this example, we analyse a $ 1\,\text{m}^2$ plate of steel with thickness $d=0.1 {m}$ 
The Young's modulus of steel is $ E = 200 \cdot 10^9\,\text{Pa}$  and Poisson
ratio $\nu = 0.3$ .

In reality, the operator  $\frac{Ed^3}{12(1-\nu^2)} \Delta^2 $ is a combination of multiple first-order operators:

$\boldsymbol{K}(u) = - \boldsymbol{\varepsilon}(\nabla u), \quad \boldsymbol{\varepsilon}(\boldsymbol{w}) = \frac12(\nabla \boldsymbol{w} + \nabla \boldsymbol{w}^T)$ 

$\boldsymbol{M}(u) = \frac{d^3}{12} \mathbb{C} \boldsymbol{K}(u), \quad \mathbb{C} \boldsymbol{T} = \frac{E}{1+\nu}\left( \boldsymbol{T} + \frac{\nu}{1-\nu}(\text{tr}\,\boldsymbol{T})\boldsymbol{I}\right)$ 

where $\boldsymbol{I}$ is the identity matrix. In particular

$  \frac{Ed^3}{12(1-\nu^2)} \Delta^2 u = - \text{div}\,\textbf{div}\,\boldsymbol{M}(u) $

There are several boundary conditions that the problem can take.

The *fully clamped* boundary condition reads

$  u = \frac{\partial u}{\partial \boldsymbol{n}} = 0,$

where $ \boldsymbol{n}$  is the outward normal.

Moreover, the *simply supported* boundary condition reads

$  u = 0, \quad M_{nn}(u)=0,$

where  $M_{nn} = \boldsymbol{n} \cdot (\boldsymbol{M} \boldsymbol{n})$ .

Finally, the *free* boundary condition reads

  $ M_{nn}(u)=0, \quad V_{n}(u)=0 $

where $V_n$ is the  Kirchhoff shear force <https://arxiv.org/pdf/1707.08396.pdf>_. The exact definition is not needed here as this boundary condition is a natural one.

The correct weak formulation for the problem is: find  $u \in V$ such that

  $ \int_\Omega \boldsymbol{M}(u) : \boldsymbol{K}(v) \,\mathrm{d}x = \int_\Omega fv \,\mathrm{d}x \quad \forall v \in V $

where  $V$  is now a subspace of $ H^2$  with the essential boundary

conditions for $u$  and  $\frac{\partial u}{\partial \boldsymbol{n}}$ .



```
>>> from skfem import *
>>> from skfem.models.poisson import laplace, unit_load
>>> m = MeshTri()
>>> m.refine(2)
>>> basis = InteriorBasis(m, ElementTriP1())
>>> A = laplace.assemble(basis) # attention 
>>> b = unit_load.assemble(basis)
```
# Stokes equations.



This solves for the creeping flow problem in the primitive variables,i.e. velocity and pressure instead of the stream-function. These are governed

$$
- \nu\Delta\boldsymbol{u} + \rho^{-1}\nabla p = \boldsymbol{f} \ in \  \Omega
$$

$$
\nabla\cdot\boldsymbol{u} = 0 \ in \  \Omega
$$

$$
u = 0 \ on \ \partial \Omega
$$

One of the simplest workable choices is the Taylor--Hood element:  $P_2$ for velocity and   $P_1$ for pressure.

Once the velocity has been found, the stream-function  $\psi$ can be calculated by solving the Poisson problem

  $ -\Delta\psi = \nabla \times\ \boldsymbol{u} $

where  $\mathrm{rot}\,\boldsymbol{u} \equiv \partial u_y/\partial x - \partial u_x/\partial y$.

In the weak formulation

  $ \left(\nabla\phi, \nabla\psi\right) = \left(\phi, \nabla \times\ \boldsymbol{u}\right) \quad \forall \phi \in H^1_0(\Omega) $ 

the right-hand side can be converted using Green's theorem and the no-slip condition to not involve the derivatives of the velocity:

$ \left(\phi, \nabla \times \boldsymbol{u}\right) = \left(\nabla \times \phi, \boldsymbol{u}\right) $ 

where 

$ \nabla \times \phi \equiv \frac{\partial\phi}{\partial y}\hat{i} - \frac{\partial\phi}{\partial x}\hat{j} $

$ \mathrm{M}=\mathrm{L}_{0}^{2}(\Omega)=\left\{\mathrm{q} \in \mathrm{L}^{2}(\Omega): \int_{\Omega} \mathrm{q} \mathrm{d} \mathrm{x}=0\right\} $ 



> copied from https://kinnala.github.io/scikit-fem-docs/examples/ex18.html
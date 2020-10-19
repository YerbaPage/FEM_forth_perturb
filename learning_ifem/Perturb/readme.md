# A FOURTH ORDER ELLIPTIC SINGULAR PERTURBATION PROBLEM

$$
\left\{\begin{array}{lr}
\varepsilon^{2} \Delta^{2} u-\Delta u=f & \text { in } \Omega \\
u=\partial_{n} u=0 & \text { on } \partial \Omega
\end{array}\right.
$$

For any $v \in H^{s}\left(\mathcal{T}_{h}\right),$ define the broken $H^{s}$ norm and seminorm
$$
\|v\|_{s, h}^{2}:=\sum_{K \in \mathcal{T}_{h}}\|v\|_{s, K}^{2}, \quad|v|_{s, h}^{2}:=\sum_{K \in \mathcal{T}_{h}}|v|_{s, K}^{2}
$$
For any $v \in H^{2}\left(\mathcal{T}_{h}\right),$ define some other discrete norms
$$
\begin{array}{c}
\|v\|_{2, h}^{2}:=|v|_{2, h}^{2}+\sum_{F \in \mathcal{F}_{h}^{\partial}} h_{F}^{-1}\left\|\partial_{n} v\right\|_{0, F}^{2} \\
\|v\|_{\varepsilon, h}^{2}:=\varepsilon^{2}|v|_{2, h}^{2}+|v|_{1, h}^{2}, \quad\|v\|_{\varepsilon, h}^{2}:=\varepsilon^{2}\|v\|_{2, h}^{2}+|v|_{1, h}^{2}
\end{array}
$$
find 
$$
\varepsilon^{2} a_{h}\left(u_{h 0}, v_{h}\right)+b_{h}\left(u_{h 0}, v_{h}\right)=\left(f, P_{h} v_{h}\right) \quad \forall v_{h} \in V_{h 0}
$$
where
$$
a_{h}\left(u_{h 0}, v_{h}\right):=\left(\nabla_{h}^{2} u_{h 0}, \nabla_{h}^{2} v_{h}\right), \quad b_{h}\left(u_{h 0}, v_{h}\right):=\left(\nabla_{h} u_{h 0}, \nabla_{h} v_{h}\right)
$$

$$
\begin{aligned}
\left(\nabla w_{h}, \nabla \chi_{h}\right) &=\left(f, \chi_{h}\right) & & \forall \chi_{h} \in W_{h} \\
\varepsilon^{2} \tilde{a}_{h}\left(u_{h}, v_{h}\right)+b_{h}\left(u_{h}, v_{h}\right) &=\left(\nabla w_{h}, \nabla_{h} v_{h}\right) & & \forall v_{h} \in V_{h}
\end{aligned}
$$

Lemma $5.4 .$ In two dimensions, the discrete method (5.4) can be decoupled into two Morley element methods of Poisson equation and one nonconforming $P_{1}-P_{0}$ element method of Brinkman problem, i.e., find $\left(z_{h}, \phi_{h}, p_{h}, w_{h}\right) \in V_{h} \times V_{h}^{C R} \times Q_{h} \times$ $V_{h}$ such that
$$
\begin{aligned}
\left(\operatorname{curl}_{h} z_{h}, \operatorname{curl}_{h} v_{h}\right) &=\left(\nabla w_{h}, \nabla_{h} v_{h}\right) & & \forall v_{h} \in V_{h} \\
\left(\phi_{h}, \psi_{h}\right)+\varepsilon^{2} c_{h}\left(\phi_{h}, \psi_{h}\right)+\left(\operatorname{div}_{h} \psi_{h}, p_{h}\right) &=\left(\operatorname{curl}_{h} z_{h}, \psi_{h}\right) & & \forall \psi_{h} \in V_{h}^{C R} \\
\left(\operatorname{div}_{h} \phi_{h}, q_{h}\right) &=0 & & \forall q_{h} \in Q_{h} \\
\left(\operatorname{curl}_{h} u_{h}, \operatorname{curl}_{h} \chi_{h}\right) &=\left(\phi_{h}, \operatorname{curl}_{h} \chi_{h}\right) & & \forall \chi_{h} \in V_{h}
\end{aligned}
$$
where 
$$
c_{h}\left(\phi_{h}, \psi_{h}\right):=\left(\nabla_{h} \phi_{h}, \nabla_{h} \psi_{h}\right)-\sum_{F \in \mathcal{F}_{h}^{\partial}}\left(\partial_{n}\left(\phi_{h} \cdot t\right), \psi_{h} \cdot t\right)_{F}-\sum_{F \in \mathcal{F}_{h}^{\partial}}\left(\phi_{h} \cdot t, \partial_{n}\left(\psi_{h} \cdot t\right)\right)_{F}+\sum_{F \in \mathcal{F}_{h}^{\partial}} \frac{\sigma}{h_{F}}\left(\phi_{h} \cdot t, \psi_{h} \cdot t\right)_{F}
$$

### v1.1

Finished example 1 

### v1.2

Fdded four examples to test

### v1.3

Tried penalty

### v1.4

Modified boundary conditions fot penalty and computed results of problem1 and 2

### v1.5

Fixed penalty and showed $\partial u$ on boundary (the result doesn't match result in v1.3, looking into this problem)
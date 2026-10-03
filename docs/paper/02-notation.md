# 2. Notation

The symbols below are used throughout. Vectors are bold; $\odot$ and
$\oslash$ are the product and the quotient of two vectors element by
element.

| Symbol | Meaning |
|---|---|
| $N$ | The number of particles |
| $\mathbf x_i,\ \mathbf v_i,\ m_i$ | Position, velocity, and mass of particle $i$ |
| $\mathbf F_i = -\partial U/\partial\mathbf x_i$ | Force on particle $i$; $U$ the potential energy |
| $\mathbf a_i = \mathbf F_i / m_i$ | Acceleration; $0$ for a particle without mass (a virtual site) |
| $K = \tfrac12\sum_i m_i\lVert\mathbf v_i\rVert^2$ | Kinetic energy; $K_a$ its part along axis $a$ (Section 6) |
| $\mathbf L = (L_x, L_y, L_z)$, $V = L_xL_yL_z$ | Edges and volume of the orthorhombic periodic cell |
| $H$, $\mathbf a = (a_x, 0, 0)$, $\mathbf b = (b_x, b_y, 0)$, $\mathbf c = (c_x, c_y, c_z)$ | A triclinic cell: the lower-triangular matrix whose rows are the cell vectors, reduced so that $\lvert b_x\rvert \le a_x/2$, $\lvert c_x\rvert \le a_x/2$, $\lvert c_y\rvert \le b_y/2$; $V = a_xb_yc_z$; an orthorhombic cell is $H = \operatorname{diag}(\mathbf L)$ (D123) |
| $A = \lVert \mathbf a \times \mathbf b \rVert$ | Area of the face spanned by the first two cell vectors; $A = L_x L_y$ in the reduced triangular frame (D[cell-area]) |
| $\mathbf d_{ij} = \mathbf x_i - \mathbf x_j - \mathbf L\odot\operatorname{round}((\mathbf x_i - \mathbf x_j)\oslash\mathbf L)$ | Displacement in the minimum image; $r_{ij} = \lVert\mathbf d_{ij}\rVert$. In a triclinic cell the image is taken in one pass along $\mathbf c$, $\mathbf b$, and $\mathbf a$, exact within half of the least of $a_x, b_y, c_z$ (Section 4.4) |
| $r_c$ | Cutoff of the pair terms and of the direct sum of Ewald |
| $s,\ R = r_c + s$ | Skin and reach of a neighbor structure |
| $s_\text{in},\ R_\text{in} = r_c + s_\text{in}$ | Skin and reach of the inner list of a dual list |
| $\mathbf x^\text{ref},\ \mathbf L^\text{ref}$ | Configuration and cell of the last build of a structure |
| $\mathbf x^p,\ \mathbf L^p$ | Configuration and cell of the last pruning of an inner list |
| $\mathbf m = \mathbf L\oslash\mathbf L^\text{ref}$ | Scale of each axis since the build |
| $\mathbf L_0,\ \mathbf m_0 = \mathbf L\oslash\mathbf L_0$ | Cell of the file of coordinates, and the scale of each axis since it |
| $\mathbf x^\text{ref} = \mathbf m_0\odot\mathbf x^\text{c} + \mathbf o$ | Reference of a positional restraint: a center that scales with the cell and an offset that does not (Section 6.4) |
| $\mathbf u_i = \mathbf x_i\oslash\mathbf L$ | Coordinates of particle $i$ in the frame of the cell |
| $\varepsilon = \ln V$, $\varepsilon_a = \ln L_a$; $\mu_a$ | Strain of the cell and of axis $a$; $\mu_a$ the factor of a step of the barostat along $a$ |
| $\delta_k,\ \tau_k$ | Spacing of the points of a tabulated function along its argument $k$, and the place of an argument in its cell, from 0 to 1 (Section 3.1) |
| $\mathbf X_g,\ \mathbf F_g$ | Center of mass of a group $g$ that moves as a whole, and the total force on it |
| $P,\ P_0,\ P_{aa}$ | Pressure, its target, and the pressure of axis $a$ |
| $Z,\ F$ | Canonical partition function and Helmholtz free energy (a force is bold, $\mathbf F_i$) |
| $\Delta t$ | Time step |
| $\beta$ | Ewald splitting parameter; $\operatorname{erfc}(\beta r_c)$ is the tolerance of the direct sum |
| $q_i$ | Partial charge; $f = 1/(4\pi\varepsilon_0)$ the Coulomb constant in the units of the run |
| $c_i,\ \gamma(x)$ | Coefficient of the dispersion of particle $i$, $2\sqrt{\varepsilon}\,\sigma^3$ of its type, so that $C_{6,ij} = c_ic_j$; and $\gamma(x) = e^{-x^2}(1 + x^2 + x^4/2)$, the share of $-c_ic_j/r^6$ at $x = \beta r$ that the direct terms of the mesh of the dispersion keep (Section 5.5) |
| $K_a$ | In Section 5, the number of grid points of PME along axis $a$; $n$ the order of the B-splines (4) |
| $\mathsf W$ | Virial tensor; $\mathsf W_{aa} = -\partial U/\partial\varepsilon_a$ along the scaling of positions and cell (Sections 6.4 and 6.5) |
| $T,\ k_B$ | Temperature and Boltzmann's constant |
| $\gamma$; $D_i = k_BT/(m_i\gamma)$ | Friction of Langevin and Brownian dynamics, in 1/ps (in Section 6.4, $\gamma$ is the surface tension); the diffusion constant of particle $i$ in Brownian dynamics (Section 6.1) |
| $\eta_j,\ p_{\eta_j},\ Q_j$; $H'$ | Position, momentum, and mass of thermostat $j$ of a Nosé–Hoover chain of $M$, and the energy that the chain conserves (Section 6.3) |

**Units.** Inside a compiled program MDIR computes in nm, ps, amu, and
kJ/mol, with $k_B = 0.0083144626181532$ kJ/(mol K) [[Tiesinga2021]](references.md#tiesinga2021). The
control file and the logs use Å, ps, and kcal/mol, the units of the Amber
inputs that the suite of Section 10 takes; the driver converts at the
boundary (1 kcal/mol = 4.184 kJ/mol; 1 kJ/(mol nm³) =
$16.6053906717/1.01325 \approx 16.388$ atm).

**Precision.** f32 and f64 are the IEEE-754 binary32 and binary64 types;
an *ulp* is a unit in the last place. A *mode* (`single`, `mixed`,
`double`) assigns a type to each role of a value (Section 7).

**Decisions.** The design records each choice as a numbered decision
(D1, D2, ...) in `docs/decisions.md`, with the measurement that motivated
it; the paper cites them as "(D95)". File names are relative to the root
of the repository.

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
| $K = \tfrac12\sum_i m_i\lVert\mathbf v_i\rVert^2$ | Kinetic energy |
| $\mathbf L = (L_x, L_y, L_z)$, $V = L_xL_yL_z$ | Edges and volume of the orthorhombic periodic cell |
| $\mathbf d_{ij} = \mathbf x_i - \mathbf x_j - \mathbf L\odot\operatorname{round}((\mathbf x_i - \mathbf x_j)\oslash\mathbf L)$ | Displacement in the minimum image; $r_{ij} = \lVert\mathbf d_{ij}\rVert$ |
| $r_c$ | Cutoff of the pair terms and of the direct sum of Ewald |
| $s,\ R = r_c + s$ | Skin and reach of a neighbor structure |
| $s_\text{in},\ R_\text{in} = r_c + s_\text{in}$ | Skin and reach of the inner list of a dual list |
| $\mathbf x^\text{ref},\ \mathbf L^\text{ref}$ | Configuration and cell of the last build of a structure |
| $\mathbf x^p,\ \mathbf L^p$ | Configuration and cell of the last pruning of an inner list |
| $\mathbf m = \mathbf L\oslash\mathbf L^\text{ref}$ | Scale of each axis since the build |
| $\Delta t$ | Time step |
| $\beta$ | Ewald splitting parameter; $\operatorname{erfc}(\beta r_c)$ is the tolerance of the direct sum |
| $q_i$ | Partial charge; $f = 1/(4\pi\varepsilon_0)$ the Coulomb constant in the units of the run |
| $K_a$ | Number of grid points of PME along axis $a$; $n$ the order of the B-splines (4) |
| $\mathsf W$ | Virial tensor (Section 6.5 states its sign) |
| $T,\ k_B$ | Temperature and Boltzmann's constant |

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
(D1 to D114) in `docs/decisions.md`, with the measurement that motivated
it; the paper cites them as "(D95)". File names are relative to the root
of the repository.

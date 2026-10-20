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
| $K_\text{half},\ K_T,\ K_P$ | The mean of the kinetic energies of the half steps before and after a step; the kinetic energies of the temperature, $\tfrac13K + \tfrac23K_\text{half}$, which the thermostats control, and of the pressure, $K_{\text{half},g}$, of the log (Section 6.5) |
| $M_g,\ \mathbf P_g;\ K_g,\ K_{\text{half},g};\ \mathsf W_g$ | Mass and momentum of a group $g$ that the barostat moves as a whole (a rigid group, or a particle alone); the kinetic energy of the centers of the groups at the step and the mean of those at the half steps; the virial of the groups, $\sum_g\mathbf X_g\otimes\mathbf F_g$ (Section 6.4) |
| $\mathbf L = (L_x, L_y, L_z)$, $V = L_xL_yL_z$ | Edges and volume of the orthorhombic periodic cell |
| $H$, $\mathbf a = (a_x, 0, 0)$, $\mathbf b = (b_x, b_y, 0)$, $\mathbf c = (c_x, c_y, c_z)$ | A triclinic cell: the lower-triangular matrix whose rows are the cell vectors, reduced so that $\lvert b_x\rvert \le a_x/2$, $\lvert c_x\rvert \le a_x/2$, $\lvert c_y\rvert \le b_y/2$; $V = a_xb_yc_z$; an orthorhombic cell is $H = \mathrm{diag}(\mathbf L)$ (D123) |
| $A = \lVert \mathbf a \times \mathbf b \rVert$ | Area of the face spanned by the first two cell vectors; $A = L_x L_y$ in the reduced triangular frame (D170) |
| $u_{ij}(r),\ I_{ij};\ \nu$ | The energy of a pair term at distance $r$ for the pair $\{i,j\}$, and its tail $\int_{r_c}^\infty r^2u_{ij}\,dr$; $\nu = N^2/(N(N-1) - 2N_\text{excluded})$, the factor of the correction for the dispersion of a topology (Section 5.4) |
| $u_\text{shift},\ U_\text{shift};\ N_\text{in},\ E_\text{sh}$ | A pair energy shifted to 0 at $r_c$, $u - u(r_c)$ within it, and its sum over the pairs; the number of pairs within $r_c$, and the estimate at a uniform density of what the shift takes from them (Sections 5.4, 6.8) |
| $\xi;\ \Pi$ | A constant of a term given by an expression, whose derivative `observe` writes (Section 6.8); the osmotic pressure of a solution held between walls |
| $\mathbf d_{ij} = \mathbf x_i - \mathbf x_j - \mathbf L\odot\mathrm{round}((\mathbf x_i - \mathbf x_j)\oslash\mathbf L)$ | Displacement in the minimum image; $r_{ij} = \lVert\mathbf d_{ij}\rVert$. In a triclinic cell the image is taken in one pass along $\mathbf c$, $\mathbf b$, and $\mathbf a$, exact within half of the least of $a_x, b_y, c_z$ (Sections 4.3 and 4.4) |
| $\mathbf n = (n_a, n_b, n_c)$ | Lattice indices of an image of a displacement, $\mathbf d - \mathbf n H$ (Sections 4.3 and 4.4) |
| $\mathcal E;\ C_i,\ E_i,\ O_i$ | In Section 4.6, the pairs of images within $r_c$ that a learned potential takes, $(i, j, \mathbf n)$ with $\lVert\mathbf x_j - \mathbf x_i + \mathbf n H\rVert < r_c$; the candidates of particle $i$ (its pairs of images within $R$ at the last build), those of them within $r_c$ at the step, and the offset of its slots, $O_i = \sum_{k<i} C_k$ |
| $N_\text{layer};\ \mathbf h_i^{(k)}$ | The number of message-passing layers of a learned potential, and the features of atom $i$ after layer $k$ (Section 13) |
| $\phi_k,\ \psi_k,\ \eta,\ \mathcal R;\ \mathbf A_i^{(k)};\ z_i$ | In Section 3.3, the edge stage and the node stage of layer $k$ of a message-passing model, its embedding of the species, and its readout; the sum of the edge stage over the neighbors of $i$, $\mathbf A_i^{(k)} = \sum_j\phi_k(i,j)$; the species of atom $i$ |
| $\bar v = \partial U/\partial v;\ \bar r_{ij},\ \bar{\mathbf d}_{ij},\ \bar{\mathbf h}^\mathrm c_{ij},\ \bar{\mathbf h}^\mathrm n_{ij}$ | The adjoint of a value $v$ that the computation of $U$ forms; in Section 3.3, the cotangents of the distance, the displacement, the center's features, and the neighbor's features that the vector-Jacobian product of an edge stage gives at the directed pair $(i, j)$ for the cotangent $\bar{\mathbf A}_i$ of its value |
| $r_c$ | Cutoff of the pair terms and of the direct sum of Ewald |
| $r_s$, $S_2(r)$ | Onset and squared-distance potential switch of a topology Lennard-Jones term (Section 3.3) |
| $s,\ R = r_c + s$ | Skin and reach of a neighbor structure |
| $s_\text{in},\ R_\text{in} = r_c + s_\text{in}$ | Skin and reach of the inner list of a dual list |
| $n_\text{build},\ n_\text{prune};\ t_\text{step};\ \rho;\ v_\text{max}$ | In Section 4.5, the mean number of steps between builds of a structure and between prunings of its inner list; the time of a step; the number density of the particles; the largest speed among the particles |
| $t_\text{loop},\ t_\text{prune},\ t_\text{fused}$ | In Section 4.5, the time of a call of the loop over pairs over the inner list, of a pruning, and of the loop that prunes |
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
| $F_\text{tol};\ k_\text{h};\ \chi_k,\ \Lambda_k$ | The force tolerance of a minimization, compared with the largest force $\max_i\lVert m_i\mathbf g_i\rVert$ without its parts along the constraints; the force constant of a harmonic mode; a holonomic constraint $\chi_k(\mathbf x) = 0$ and its Lagrange multiplier (Section 6.7) |
| $\ell,\ \ell_\text{min}$ | The step length of steepest descent, the farthest a trial moves a particle, at most 0.1 Å; the floor below which a step that has shrunk ends the minimization (Section 6.7) |
| $\beta$ | Ewald splitting parameter; $\mathrm{erfc}(\beta r_c)$ is the tolerance of the direct sum |
| $q_i$ | Partial charge; $f = 1/(4\pi\varepsilon_0)$ the Coulomb constant in the units of the run |
| $c_i,\ \gamma(x)$ | Coefficient of the dispersion of particle $i$, $2\sqrt{\varepsilon}\,\sigma^3$ of its type, so that $C_{6,ij} = c_ic_j$; and $\gamma(x) = e^{-x^2}(1 + x^2 + x^4/2)$, the share of $-c_ic_j/r^6$ at $x = \beta r$ that the direct terms of the mesh of the dispersion keep (Section 5.5) |
| $\mathcal J_q,\ \mathcal J_c$ | In Sections 5.3 and 5.6, the particles that the meshes spread and gather: those of a nonzero charge, $\{j : q_j \ne 0\}$, and of a nonzero coefficient of the dispersion, $\{j : c_j \ne 0\}$ |
| $K_a$ | In Section 5, the number of grid points of PME along axis $a$; $n$ the order of the B-splines (4) |
| $\mathcal F,\ \mathcal F^\dagger;\ \langle\cdot,\cdot\rangle_\mathrm H$ | In Section 5.2, the unnormalized real-to-half-complex discrete Fourier transform of a grid and its unnormalized inverse; the real inner product of half-complex arrays with weights 1 at the last-axis zero and Nyquist frequencies (the latter only for even $K_3$), and 2 at the other frequencies |
| $\hat U(t);\ \boldsymbol\nu,\ \mathbf J;\ \tilde{\mathbf m}_\mathbf J,\ \tilde m_{a,j};\ S_a,\ u_{a,j},\ F_a;\ G;\ A,\ B$ | In Section 5.2, the transform of the B-spline of order $n$ over a spacing of the grid, $(\sin\pi t/\pi t)^n$; the aliases $\boldsymbol\nu = \mathbf k + \mathbf J\odot\mathbf K$ of a point $\mathbf k$ of the grid, $\mathbf J\in\mathbb Z^3$, $\mathbf K = (K_1, K_2, K_3)$; their wave vectors, and along axis $a$ of an orthorhombic cell $\tilde m_{a,j} = (k_a + jK_a)/L_a$; the sum $\sum_j\hat U^2$ of an axis, the normalized weights $\hat U^2/S_a$ of its aliases, and its factor of the influence function; the influence function of `"P3M"`; the means of $\tilde m_a\tilde m_b$ that its virial takes |
| $g$ | In Sections 5.3 and 8.5, the points of an edge of a bin of the spreading into tiles on a device (8 by default) |
| $\psi_0^c,\ \lambda_0,\ C_0;\ c,\ c_1;\ \eta$ | In Section 5.6, the prolate spheroidal wave function of order zero and bandwidth $c$ on $[-1, 1]$, normalized to $\int_{-1}^{1}\psi^2 = 1$; its eigenvalue under $\varphi\mapsto\int_{-1}^1e^{icxt}\varphi(t)\,dt$; and $C_0 = \int_0^1\psi_0^c$. $c$ is the bandwidth of the splitting and $c_1$ that of the window (not the coefficients $c_i$ of the dispersion, nor the cell vector $\mathbf c$); $\eta = \psi_0^c(1)$ is the tolerance of the prolate kernel |
| $\chi,\ \Phi,\ \bar\chi,\ \hat\chi;\ \phi_1,\ P;\ \Delta$ | In Section 5.6, the splitting kernel $\chi = \psi_0^c/C_0$ on $[-1,1]$ (not a constraint $\chi_k$), $\Phi(y) = \int_0^y\chi$, the mean $\bar\chi(x) = (1 - \Phi(x))/(1 - x)$ of $\chi$ over $[x, 1]$, and its transform $\hat\chi(k) = \tfrac12\int_{-1}^1\chi(x)e^{-ikx}dx$, so that $\hat\chi(0) = 1$; the window of the spreading on $P$ points (not a pressure); the relative force error $\big(\sum_i\lVert\tilde{\mathbf F}_i - \mathbf F_i\rVert^2/\sum_i\lVert\mathbf F_i\rVert^2\big)^{1/2}$, with $\mathbf F_i$ the Coulomb forces of the topology (its exclusions and 1-4 scaling included) of a reference |
| $E_A$ | In Section 5.6, the estimate of the aliasing error of the spreading, eq. 10 of [[Liang2026]](references.md#liang2026) relative to the sum it perturbs |
| $v_k,\ i_k;\ \mathcal B_b;\ \oplus,\ \mathsf u$ | In Section 8.5, the value that item $k$ of a scatter-accumulate adds and the index it adds it at; the items of block $b$ of a device; floating-point addition, and the unit roundoff of a type, $2^{-24}$ in f32 and $2^{-53}$ in f64 |
| $\mathsf W$ | Virial tensor; $\mathsf W_{aa} = -\partial U/\partial\varepsilon_a$ along the scaling of positions and cell (Sections 6.4 and 6.5) |
| $\boldsymbol\varepsilon$ | In Section 3.6, a strain as a matrix, under which $\mathbf x_i \to (\mathsf 1 + \boldsymbol\varepsilon)\mathbf x_i$ and each cell vector, a row of $H$, alike, so that $\varepsilon_{aa}$ is a change of $\varepsilon_a$ and $\partial U/\partial\varepsilon_{ab} = -\mathsf W_{ba}$; $\mathbf K$ there is the force on the second end of a displacement $\mathbf d$ |
| $T,\ k_B$ | Temperature and Boltzmann's constant |
| $\gamma$; $D_i = k_BT/(m_i\gamma)$ | Friction of Langevin and Brownian dynamics, in 1/ps (in Section 6.4, $\gamma$ is the surface tension); the diffusion constant of particle $i$ in Brownian dynamics (Section 6.1) |
| $\mathcal P_{\mathbf x}$; $\mathbf v_\parallel$ | In Section 6.1, the projection that leaves no velocity along the constraints at $\mathbf x$ (the second half of RATTLE), and the part of a leapfrog kick along them, $\mathbf v_\parallel = (1 - \mathcal P_{\mathbf x_n})(\mathbf v_{n-\frac12} + h\mathbf a_n)$ |
| $\eta_j,\ p_{\eta_j},\ Q_j$; $H'$ | Position, momentum, and mass of thermostat $j$ of a Nosé–Hoover chain of $M$, and the energy that the chain conserves (Section 6.3) |
| $\tilde H,\ E_\text{bath},\ \Delta E_\text{sys}$ | The conserved energy of the log, $U + K + E_\text{bath}$; the energy the bath has taken, the sum of $-\Delta E_\text{sys}$ over the couplings, each of which gives the system $\Delta E_\text{sys}$ (Section 6.5) |
| $\boldsymbol\lambda,\ \lambda_m;\ \boldsymbol\lambda^{(k)}$ | In Section 6.8, the coupling parameters of `[free_energy]` and a component of them, $\lambda_\text{C}$ of the Coulomb and $\lambda_\text{V}$ of the Lennard-Jones; the state $k$. In Section 6.4, $\lambda = \sqrt V$ |
| $a_i,\ c_{ij}$ | 1 if particle $i$ is in the selection that `[free_energy]` decouples, 0 otherwise; 1 if exactly one of $i$, $j$ is, $c_{ij} = a_i + a_j - 2a_ia_j$ |
| $r_A,\ \alpha_\text{sc},\ p$ | The soft-core distance of a decoupled pair, $r_A^6 = \alpha_\text{sc}\sigma^6\lambda_\text{V}^p + r^6$, and its two parameters |
| $u_k,\ f_k$ | Reduced energy $U_k/k_BT$ of state $k$ and its reduced free energy $F_k/k_BT$ |
| $\boldsymbol\theta = (\theta_0, \dots, \theta_{M_\theta-1}),\ \iota;\ \hat{\boldsymbol\theta}$ | A tunable parameter of a Python model, and the map $\iota$ from its sites (particles, types, unordered pairs of types, tuples, or one constant) to its entries, $\iota(s) = -1$ for a site that keeps its value; the values at which stored frames were sampled (Section 3.6) |
| $S_n,\ w_n;\ N_S,\ g_n$ | A stored frame and its weight when frames sampled at $\hat{\boldsymbol\theta}$ are reweighted to $\boldsymbol\theta$ (Section 3.6); the number of frames, and the cotangent of the energy of frame $n$, $\partial L/\partial U(S_n)$ for a loss $L$ |
| $\tilde S_n,\ \boldsymbol\delta,\ q;\ \mathsf H$ | A frame as a trajectory file holds it, $\tilde S_n = S_n + \boldsymbol\delta$, the displacement that the rounding of the file adds, and the spacing of the grid of the rounding; the matrix of the second derivatives of $U$ in the positions (Section 3.6) |
| $b_s,\ c;\ d_i,\ J_{ab}$ | A value of a compiled program at the site $s$ that follows $\boldsymbol\theta$ (a charge, an entry of the table of $\sigma$ or $\epsilon$, a parameter of a tuple), and the constants that the host adds to the energy; a seed, a field of zeros over the particles through which the derivative in such values leaves the program, and its weight for a particle of type $a$ with a partner of type $b$ (Section 3.6) |
| $\mathbf z_n;\ \mathcal K_{U,\mathcal C};\ \omega$ | The state of a run after step $n$ that a checkpoint holds (positions, velocities, cell, the forces the next step begins with, the energy the coupling has taken); the transition kernel of one step, of the potential $U$ and the coupling $\mathcal C$ (thermostat, barostat, and their parameters); the seed of the random numbers (Section 3.6) |
| $\pi,\ \pi^{-1};\ j_{t,k},\ j'_{t,k}$ | In Section 3.6, an order of the particles, $\pi(p)$ the particle, in the order of the files, at place $p$, and its inverse; the member in slot $k$ of tuple $t$ of a tuple set in the order of the files, and at its place, $j'_{t,k} = \pi^{-1}(j_{t,k})$ (D257) |
| $O_d,\ \Gamma_d;\ s(g);\ G,\ G^\mathsf T$ | In Section 13, the particles that domain $d$ owns, and its ghosts, the images of particles near its region; the particle of which a ghost $g$ is an image (its source), displaced by $\mathbf n_gH$; the copy of owned values to ghosts, $(G\mathbf a)_g = \mathbf a_{s(g)}$, and its transpose, the sum of the ghosts' values at their owners, $(G^\mathsf T\mathbf b)_i = \sum_{g : s(g) = i}\mathbf b_g$ (D275) |
| $\rho_i = \sum_{j\ne i} f(r_{ij});\ F(\rho);\ \bar\rho_i$ | In Section 13, the density of the embedded-atom method at particle $i$, from the function $f$ of the distance; the embedding energy of a particle of density $\rho$; the derivative of the energy in $\rho_i$, $F'(\rho_i)$ (D279) |

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

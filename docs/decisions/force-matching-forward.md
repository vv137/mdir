# D[force-matching-forward]

Recorded 2026-11-08.

**Forward mode in the tunables of the gradient in the positions,
$\dot{\mathbf G} = (\partial^2U/\partial\mathbf x\,\partial\boldsymbol\theta)\,\dot{\boldsymbol\theta}$,
is the gradient in the positions of one potential, the derivative of $U$
along directions of its arguments,
$\Psi = \sum_s\dot b_s\,\partial U/\partial b_s$, which `md-differentiate`
builds from the sums of $U$ as D376 builds the tangent in the positions,
and whose forces the existing `forces` request gives; `md.evaluate` asks for
it with the requests `direction` and `direction_forces` and the attribute
`directions`. The host maps $\dot{\boldsymbol\theta}$ to the directions
$\dot b_s$ by the transpose of the chain rule of D230, a third program of
frames gives $\Psi$ and its forces, and `FrameEnergies.jvp(tangent)` returns
the tangents of the energies and of their gradients in the positions.**

Part F of the design on #726 (Ruling 21, Q6: forward mode in θ of the
forces is refused until this part), with the maintainer's ruling on PR
#778 (the interface as proposed, the one line of a frozen file in this PR,
one tangent a call, the adapters apart). It is the transpose of D381's product
and builds on D376 (the tangent in the IR), D379 (the reciprocal sum), D381
(the product program), and D383/D384 (the forces of a constructed potential,
the spread of the sites).

## The quantity

For a frame $n$, with the buffers $b_s(\boldsymbol\theta)$ of the program
(numbers, and fields of the particles: the seeds and the charges of D230),
and a tangent $\dot{\boldsymbol\theta}$ of the tunables,

$$
\dot b_s = \sum_m\frac{\partial b_s}{\partial\theta_m}\,\dot\theta_m,
\qquad
\Psi_n = \sum_s\dot b_s\,\frac{\partial U_n}{\partial b_s}
= \frac{d}{d\lambda}\,U_n(\mathbf b + \lambda\dot{\mathbf b})\Big|_{\lambda=0}.
$$

$U$ is twice continuously differentiable in $(\mathbf x, \mathbf b)$ except
where a pair crosses the cutoff, a set of measure zero (D376), so the
derivatives commute and

$$
\dot{\mathbf G}_{n,i} = \sum_m\frac{\partial^2U_n}{\partial\mathbf x_i\,\partial\theta_m}\,\dot\theta_m
= \frac{\partial\Psi_n}{\partial\mathbf x_i}.
$$

The tangent of the energy is
$\dot U_n = \Psi_n + \dot{\boldsymbol\theta}\cdot\partial U^{\mathrm{host}}_n/\partial\boldsymbol\theta$,
where $U^{\mathrm{host}}$ is what the host adds to the energy (the tails and
shift estimates of pair terms, the dispersion correction, the self term and
background of PME). It depends on the volume and $\boldsymbol\theta$ only, so
it adds nothing to $\dot{\mathbf G}$ at a fixed cell.

**The transpose identity.** With D381's product
$\mathbf p(g, \bar{\mathbf G}) = \sum_n[g_n\,\partial U_n/\partial\boldsymbol\theta + \sum_i\bar{\mathbf G}_{n,i}\cdot\partial^2U_n/\partial\mathbf x_i\,\partial\boldsymbol\theta]$,

$$
\dot{\boldsymbol\theta}\cdot\mathbf p(g, \bar{\mathbf G})
= \sum_n\Big[g_n\,\dot U_n + \sum_i\bar{\mathbf G}_{n,i}\cdot\dot{\mathbf G}_{n,i}\Big],
$$

for every $g$, $\bar{\mathbf G}$, $\dot{\boldsymbol\theta}$ (gate 3 of the
design, $10^{-12}$ relative in `double`). The term in $g$ holds only with the
host part of $\dot U_n$; the gate takes $g\ne0$.

## The IR

`md.evaluate` takes two more requests, `direction` ($\Psi$, `f64`) and
`direction_forces` ($-\partial\Psi/\partial\mathbf x_i$, of the type of the
positions), and the optional attribute `directions`, which names arguments of
the potential in increasing order, each a number (`f64`) or a field of the
particles with one component of `f64`. The op then takes one operand more
for each, after the arguments of the potential: its direction, of the type of
the argument. The requests of a direction and those of the tangent of D376 are
not mixed in one op. The values of the existing requests (0 to 6) are
unchanged, and so are the builders of the op without directions, so a program
without the new requests is expected to be printed and lowered as before (the
builder-equivalence corpus is in the validation plan).

`md-differentiate` builds $\Psi$ as `md.potential @<name>.direction_<n>...`,
with the potential's arguments then the directions, once for each set of
directed arguments. With the truncation expanded, it replaces each directed
number $b$ by $b + \lambda\dot b$ and each directed field $a$ by the map
$a + \lambda\dot a$, takes the derivative in $\lambda$ by `ScalarDerivative`,
and then sets $\lambda = 0$, which restores the arguments exactly:

- a sum over a relation, over tuples, or over particles that reads a directed argument becomes a sum of the same structure that gathers, for each gathered field that depends on the directions, its direction as well. The kernel yields its derivative through the numbers that it takes from outside, plus, for each such field, the derivative in its value at each end or member along the gathered direction ($\partial k/\partial a_i\,\dot a_i + \partial k/\partial a_j\,\dot a_j$ for a pair). A pair kernel stays invariant under exchange, so the sum keeps `exchange(symmetric)`, as derived;
- a field computed by a map from directed arguments takes, as its direction, the same map differentiated along the directions of the fields that it gathers;
- a reciprocal sum, whose energy $E = \tfrac12\mathbf c^{\mathsf T}\mathsf A(\mathbf x)\mathbf c$ is a quadratic form of its charges, gives $\dot{\mathbf c}^{\mathsf T}\mathsf A\mathbf c = \tfrac12[E(\mathbf c+\dot{\mathbf c}) - E(\mathbf c-\dot{\mathbf c})]$ exactly: two reciprocal sums, with $\dot{\mathbf c}$ the direction of the charges, **whatever the number of directed entries**. D379's derivative of the tangent takes two sums for each number, $2M$ for $M$ entries; forward mode takes two for all of them, because $\dot{\boldsymbol\theta}$ combines the entries into one direction before the sums.

The pass then asks $\Psi$ for its `energy` and `forces` by the existing path,
in one generated function (`<name>.direction_2_3.energy_forces`). The forces of
$\Psi$ need no new rule: its kernels read the distance and gathered values
only, as those of $U$ do, and the forces of the two reciprocal sums are the
op's own forces at the shifted charges. A sum that reads no directed argument
has no term, by the proof of activity, so a tunable of one term costs a pass
over that term alone.

Refused, by name: a value without a derivative rule, a learned stage, a
directed operand of a sum other than a gathered field of the particles (a
field of the tuples), a field computed other than by a map, the forces, the
virial, or the potential of a reciprocal sum along the directions, a
reciprocal sum with the list of D262 whose charges depend on the directions
(a direction may give a charge to a particle outside the list; the builder
gives no list where a charge is tunable, so the frame programs have none
there), and a potential on domains.

`md-check-derivatives` checks `direction` against central differences of
the energy along the directions, and `direction_forces` against central
differences of the forces along them, each component on its own: the finite
differences of $\mathbf G$ in $\boldsymbol\theta$.

## The frame evaluator

**The interface.** `FrameEnergies.jvp(tangent, *, gradients=("positions",))`.
`tangent` maps names of the tunables to arrays of the shapes of their
values; a name left out is zero, a name that is not a tunable, a shape, and
a value that is not finite are refused. It returns a `FrameTangent`:
`energy`, $\dot U_n$ `(K,)` float64 with what the host adds to the energy;
`gradients["positions"]`, $\dot{\mathbf G}_n$ `(K, N, 3)` float64 in the
order of the input, where `gradients` names it; and `version`. With
`gradients=()` there is no pass of the product program: $\dot U$ is the
Jacobian times the tangent, kept or by a second pass. One tangent a call:
`jacfwd` over $M$ entries is $M$ calls, and a gradient in many tunables is
the job of the reverse product (Q3 of the ruling). The frames are evaluated
again, so they must be a sequence, at the values of the tunables of
`evaluate`, as for `vjp`. The names follow `FrameEnergies` and `vjp`.

**The program.** A third program of frames, the product program of forward
mode, selected by `Control::frameProductDirection` (no control-file key),
which `Simulation::create` sets only with `frameProduct` and without
`frameProductPositions`. The evaluator compiles it at the first `jvp` that asks
for `"positions"`, in a simulation of its own, and gives it its values of the
tunables where their bits differ. Its boundary is that of D381; the
cotangent it hands over is zeros and not read. At each frame the entry takes
the directions from the host, the numbers of the slots and then one field of
$N$ values for each field of the gradient, in f64, through
`mdrtFrameDirection`. It evaluates `@tunable` once with `direction` and
`direction_forces`, `directions` naming every slot and field, spreads the
forces from the virtual sites to their atoms (D384), and hands $\Psi_n$ to
`mdrtWriteTunableGradient` and the forces to `mdrtWriteFrameProductForces`.
It evaluates no energy.

**The host** maps $\dot{\boldsymbol\theta}$ to the directions by the exact
transpose of `applyTunableChainRule` (`transposeTunableChainRule`): a slot
takes $\dot\theta$ of the entry its map names; a site of a field takes it at
its particle; a row of a table adds its scale times $\dot\theta$ of its entry
at every particle of its type. It adds
$\dot{\boldsymbol\theta}\cdot\partial U^{\mathrm{host}}/\partial\boldsymbol\theta$
at the volume of each frame to $\Psi_n$, the chain rule of the constants
alone, the same constants that the reverse product weights by $g$.

**Virtual sites.** The placement $\mathbf X = \mathbf s(\mathbf x)$ does not
depend on $\boldsymbol\theta$, so
$\dot{\mathbf G} = \mathsf J^{\mathsf T}\,\partial\Psi/\partial\mathbf X$
exactly, by the spread that the forces take. For the extra points of Amber
$\mathsf J$ depends on the positions, but no term in
$\partial\mathsf J/\partial\mathbf x$ arises, unlike D387: the derivative is
in $\boldsymbol\theta$. Both kinds are taken; the rows of the sites are zero.

**The one line of a frozen file**, by the maintainer's one-time exception on
PR #778: in `lib/Driver/BuilderSchedule.cpp`, the statement that fills
`EntryGradientOutputIR::product` passes `control.frameProductDirection` as the
new field `direction` of `EntryGradientProductIR` (`lib/Driver/EntryOutputIR.h`,
default false); the member is in `include/mdir/Driver/Control.h`. It is the
only change to a frozen file. No program without the flag changes.

## Coverage

| Term | Tunables | Forward mode |
|---|---|---|
| Pairs | per-type σ, ε (seeds), NBFIX pairs, constants of pair terms, the charges under a Coulomb cutoff | the pair sum along the directions |
| Tuples | parameters of tuple terms by their seeds, `dihedral_k` | the tuple sum along the seeds at their members |
| Terms of the positions | their constants and fields | the same |
| PME | the charges, per particle or tied | two reciprocal sums with their forces for all entries; the reverse product of D381 does not take them yet |
| Linear sites, extra points of Amber | as their atoms | the spread of $\partial\Psi/\partial\mathbf X$ |
| LJPME with tunables | — | refused, as D213 |
| Positional intermediate fields, learned stages, domains | — | refused by name |
| The cell (stress) | — | refused, until PR G |
| `terms="dependent"` | — | refused, as for `vjp` |

## The adapters

- **`mdir.torch`: forward AD is not formed, and on `main` it is silently zero.** `torch.library.custom_op` with `register_autograd` registers a backward only; PyTorch 2.14.1 has no registration of a JVP for a custom op. Measured on a frame evaluator of propane in water with `epsilon_pair` as a tunable: `torch.func.jvp` of `mdir.torch.evaluate(...).energy` in the tunable returns a tangent of 0, and `torch.autograd.forward_ad` returns `None`, both without an error, where $\sum_n\partial U_n/\partial\boldsymbol\theta\cdot\dot{\boldsymbol\theta} = 6.99\times10^{6}$; the same holds for `gradients["positions"]`. That breaks the refusal matrix of #726 ("never a zero"); it is an existing defect, reported as #777, and fixed apart by a refusal by name (the ruling, Q4). A forward mode of `mdir.torch` would need a `torch.autograd.Function` with a `jvp`, which `torch.compile` does not keep opaque as it does the custom op (D252); it is built only if asked for.
- **`mdir.jax`**: `mdir_frames_theta_tangent` with a nonzero $\dot{\mathbf G}$ wanted, through the JVP rule of Ruling 20, is part of the JAX rules of #750 (PR D of the design). It calls the path of this decision.

## Validation: the IR

`test/Integration/direction.mlir` (CPU, double): the model of D376's
`tangent.mlir`, six particles in a periodic cube with a pair across its
boundary, a Lennard-Jones and Coulomb pair term with a switch and charges as a
field, a bond, an angle that reads the charges, a dihedral, a term of the
absolute positions and the charges, and a term of the charges alone, with
directions on $\epsilon$ (a number of the pair term), the charges (a field),
the constant of the dihedral, and the constant of the term of the positions.

| Quantity | Reference | Value | Difference | Tolerance |
|---|---|---|---|---|
| $\bar{\mathbf G}\cdot\dot{\mathbf G}$, all four directions, CPU double | $\sum_n\dot b_n\,$`tangent_derivative(n)` (D376) | $-10.7583$ against $-10.7583$ | $1.7\times10^{-16}$ relative | $10^{-12}$ |
| the same, CPU mixed | the same | | $6.3\times10^{-8}$ relative | (record) |
| `direction`, CPU double | central differences of $U$ along the directions, step $10^{-5}$ | | within $10^{-9}+10^{-6}\lvert\cdot\rvert$ | $10^{-9}+10^{-6}\lvert\cdot\rvert$ |
| `direction_forces`, each of 18 components | central differences of $\mathbf F$ along the directions | | within the same | the same |

Each direction on its own: the dihedral and the term of the positions give
the identity to the bit; the pair term and the charges to the rounding.

`test/Integration/direction-pme.test` (CPU, double) appends to the program of
D379's `tangent-pme.test` (ten charges in a cell of 2.0 × 2.2 × 1.9 nm, a mesh
of 48 × 54 × 48, order 8, $\mathbf q = \mathbf q_0 + \lambda\mathbf w$):

| Quantity | Reference | Difference (CPU double) | CPU mixed |
|---|---|---|---|
| $\Psi$ along $\lambda$ | $\partial E/\partial\lambda$ (D161) | 0 | 0 |
| $-\mathbf t\cdot\mathbf F_\Psi$ along $\lambda$ | $\partial\Phi/\partial\lambda$ (D379) | 0 | 0 |
| $\Psi$ along the field $\mathbf q_0$ with the direction $\mathbf w$ | $\partial E/\partial\lambda$ | 0 | 0 |
| $-\mathbf t\cdot\mathbf F_\Psi$ along $\mathbf q_0$ | $\partial\Phi/\partial\lambda$ | 0 | 0 |
| $-\mathbf t\cdot\mathbf F_\Psi$ along $\lambda$ and the $\epsilon$ of a pair term | $\partial\Phi/\partial\lambda + \dot\epsilon\,\partial\Phi/\partial\epsilon$ | $1.1\times10^{-15}$ | $9.3\times10^{-11}$ |

The zeros are exact: both sides are the same half difference of the same two
reciprocal sums, so these rows check the construction, not the rule. The
rule is checked by central differences under `md-check-derivatives`, on the
same program with the three evaluations of $\Psi$ alone: $E$ and $\mathbf F$
are quadratic in the charges and linear in $\epsilon$, so the differences are
exact but for the rounding at any step. At the step 0.1, `direction` and
every component of `direction_forces` agree within $10^{-9}+10^{-8}\lvert\cdot\rvert$
(a RUN line of the test); at $10^{-2}$ the largest difference is
$3.6\times10^{-8}$ relative, and at $10^{-4}$ $2.0\times10^{-7}$, the rounding
of the mesh divided by the step.

`test/Integration/direction-gpu.test` runs both on the RTX 5090 in double:
the identity on the pairs, tuples, and the term of the positions to
$3.3\times10^{-16}$ relative ($-10.7583$ on both sides); with PME, $\Psi$
along $\lambda$ and along $\mathbf q_0$ equal to $\partial E/\partial\lambda$
to the bit, $-\mathbf t\cdot\mathbf F_\Psi$ against $\partial\Phi/\partial\lambda$
to $7.9\times10^{-16}$ for both, and with the pair $\epsilon$ to
$5.3\times10^{-16}$.

No existing test changes: the tests of `test/Dialect/MD` and
`test/Integration` pass on the CPU (95 passed, 32 unsupported for the device).

The construction by FileCheck (`test/Dialect/MD/Transforms/differentiate-direction.mlir`):
the pair sum gathering the direction of the charges with the derivative along
the number, the forces, the term that reads no directed argument absent, the
two reciprocal sums at the shifted charges, and no potential along
directions left. The refusals (`differentiate-direction-invalid.mlir`: a
listed reciprocal sum, the forces of a reciprocal sum along the directions)
and the verifier (`test/Dialect/MD/invalid.mlir`: `directions` missing,
without a request, out of order, of a wrong type, and mixed with the
tangent).

## Validation: the frame evaluator

`test/Driver/python-frame-jvp.test` (CPU), `python-frame-jvp-gpu.test`
(RTX 5090), `python-frame-jvp-amber{,-gpu}.test` (ParmEd), and
`python-frame-jvp-torchpme{,-gpu}.test` run `Inputs/python_frame_jvp.py`.
Each scenario takes a random tangent of all its tunables, scaled by their
values. The table gives the difference of the tangent of the gradient,
$\dot{\mathbf G}$, of the largest entry:

| Scenario, reference | CPU double | CPU mixed | GPU double | GPU mixed |
|---|---|---|---|---|
| `dihedral_k` (D372 fixtures, 3 frames): the closed form, $\sum_r\dot k_r\nabla[1+\cos(n_r\phi_r-\delta_r)]$ by dual numbers | 6.4e-16 | 1.2e-6 | 9.5e-16 | 1.2e-6 |
| per-type (geometric) and pair σ, ε with a pair set apart by NBFIX (propane and water, 224 atoms, 3 frames): a NumPy sum of the forces of the Lennard-Jones, differentiated in θ by a complex step | 3.8e-15 | 1.5e-6 | 3.2e-15 | 1.9e-6 |
| the charges tied in 25 entries under a Coulomb cutoff (dipeptide in water, 1,168 atoms, 2 frames): a NumPy sum of the forces of the shifted Coulomb and the 1-4 pairs | 1.1e-15 | 9.5e-7 | 1.1e-15 | 9.5e-7 |
| eight OPC waters with a linear site, σ, ε, and the charges: central differences in θ of the gradient of the NumPy model of D384 (Richardson, $h = 10^{-4}$) | 4.4e-12 | 3.8e-7 | 3.9e-12 | 3.3e-7 |
| the charges with PME, per particle (1,168 entries): the half difference of $\mathbf G$ at $\mathbf q\pm\dot{\mathbf q}$, exact for a quadratic | 6.9e-16 | 4.3e-7 | 2.5e-16 | 3.5e-7 |
| the same, tied in 25 entries | 6.0e-16 | 3.4e-7 | 4.0e-16 | 3.7e-7 |

Tolerances: 1e-10 against the NumPy and closed forms, 1e-8 against the
differences of the model, 1e-12 against the half differences, in double;
2e-5 in mixed.

- **The transpose identity** (gate 3), $\dot{\boldsymbol\theta}\cdot\mathbf p(g,\bar{\mathbf G}) = \sum_n[g_n\dot U_n + \bar{\mathbf G}_n\cdot\dot{\mathbf G}_n]$ with random $g\ne0$ and $\bar{\mathbf G}$, relative, double (CPU, GPU): `dihedral_k` 1.4e-14, 1.4e-14 (the sum cancels to $-0.2587$ from terms whose magnitudes sum to about $5\times10^{2}$: 7.6e-18 of that sum); σ, ε 3.8e-16, 1.9e-16 (the dispersion correction and its shift estimate in $\dot U$); the charges 1.2e-15, 2.6e-15; linear sites 1.3e-15, 1.5e-15; the extra points of Amber (ACE-ALA-NME and waters, 182 particles) 4.2e-16, 4.2e-16. Tolerance 1e-12. In mixed, of the sum of the magnitudes of the terms: at most 5.6e-8 (σ, ε, CPU), tolerance 1e-5. With PME the reverse product is not formed (D381), so the identity is not checked there.
- **Central differences of $\mathbf G$ in $\boldsymbol\theta$** along the tangent, by Ridders' method with the first step of D371's tests, in double: at most 1.5e-13 of the largest entry without PME (the extra points of Amber, GPU) and 3.7e-10 with it (CPU and GPU), tolerance 1e-6.
- **The tangent of the energy** against the rows of the Jacobian times the tangent: at most 1.1e-14 in double without PME; with PME 4.0e-11, the rows taking the potential of the mesh at the particles (D231) and the tangent two sums of the mesh, each with the rounding of the mesh (tolerance 1e-9); in mixed at most 3.5e-6 (tolerance 1e-5). With `gradients=()`, the same to the bit.
- **Mixed against double** on the same frames, $\dot{\mathbf G}$: at most 6.7e-6 (the extra points of Amber, CPU) of the largest entry, within 1e-5, the starting tolerance of Ruling 21 (Q9).
- **torch-pme** (Loche et al., J. Chem. Phys. 162, 142501 (2025)), the dipeptide in water with the charges of every particle, PME at $\beta = 3.5$ nm⁻¹: $\dot{\mathbf G}$ against the half difference of torch-pme's gradient (its sum of the mesh, the pairs within the cutoff, less the excluded pairs that the mesh holds, plus the 1-4 pairs, written in PyTorch), at the grids 64 and 128, 8.8e-5 and 7.9e-6 of the largest entry in double (8.8e-5 and 8.4e-6 in mixed, the same on the CPU and the RTX 5090). With twice the points MDIR changes by 6.8e-5 and torch-pme by 8.0e-5, and the tolerance is three times their sum, 4.5e-4, as `python-frame-gradients-torchpme.test` sets it.
- **The refusals:** a tangent that is not a mapping, a name that is not a tunable, a shape, a value that is not finite, a name of `gradients` other than `"positions"`, frames given once, values changed since the frames were evaluated, and `terms="dependent"`. A name left out is zero, and the evaluator gives the same tangent after them, to the bit.

Counts of the new tests: `differentiate-direction.mlir` failed once on a
pattern of FileCheck and `differentiate-direction-invalid.mlir` once on the
text of the refusal of the forces of a reciprocal sum (it is refused as a
value without a rule); `python_frame_jvp.py` failed once in mixed precision,
on the transpose identity of `dihedral_k` taken relative to its cancelling sum
(1.8e-5); it is now taken relative to the sum of the magnitudes of the terms
in mixed. Each was fixed and passed on its next run.

## Nothing else changes

Programs without the flag are unchanged, against `main` 6fc773a8 built in a
tree of its own, at the head d1fb4517 (the commits after it change only
tests and text; `main` gained only text since):

- The builder-equivalence corpus (`scripts/builder-equivalence/check.py compare`, the module and the lowered program, raw and printed, the CPU and GPU targets, both kinds of program, with the Amber suite): 1046 cases of 262 control files identical, 458 of them stopping with the same error in both.
- The order matrix (`scripts/builder-equivalence/order.py`): 560 cases identical, in its three sections: exact pair lists 480 (68 with the same error), padded 40 (20), searched 40 (20).
- Five checkpoints (JAC NVE, Factor IX NPT, and Cellulose NVE of the Amber suite, 200 steps each; the ala3 minimization; argon, 200 steps, on the GPU), in deterministic mode with the compile cache off, each tree's own executable in one job on the RTX 5090: every dataset of the final checkpoint identical to the bit, 5 of 5.
- The device code of existing programs does not change, so the sanitizer tier was not run. A memcheck of the new kernels alone (compute-sanitizer, `--tool memcheck --error-exitcode 1`, its own log file, a timeout of 30 minutes, RTX 5090): one forward pass of the frame evaluator on the torsion fixtures and on propane and water (the pair kernels along σ and ε), in double and mixed, 0 errors (exit status 0, 7 s each); the reciprocal sums along directions on the program of `direction-pme.test`, 0 errors (exit status 0). The same pass with the charges of the dipeptide under PME in the evaluator did not finish within the timeout under the sanitizer and is not counted.

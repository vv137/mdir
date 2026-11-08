# D[force-matching-forward]

Recorded 2026-11-08.

*Draft: the design and an IR prototype, for the maintainer's ruling on the
questions at the end.*

**Forward mode in the tunables of the gradient in the positions,
$\dot{\mathbf G} = (\partial^2U/\partial\mathbf x\,\partial\boldsymbol\theta)\,\dot{\boldsymbol\theta}$,
is the gradient in the positions of one potential, the derivative of $U$
along directions of its arguments,
$\Psi = \sum_s\dot b_s\,\partial U/\partial b_s$, which `md-differentiate`
builds from the sums of $U$ as D376 builds the tangent in the positions,
and whose forces the existing `forces` request gives; `md.evaluate` asks for
it with the requests `direction` and `direction_forces` and the attribute
`directions`. The host maps $\dot{\boldsymbol\theta}$ to the directions
$\dot b_s$ by the transpose of the chain rule of D230.**

Part F of the design on #726 (Ruling 21, Q6: forward mode in θ of the
forces is refused until this part). It is the transpose of D381's product
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

## The IR (prototyped in this PR)

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

## The frame evaluator (proposed; after the ruling)

- **A variant of the product program** (D381), selected by `Control::frameProductDirection` (no control-file key), compiled at the first forward product in a simulation of its own, as D383's variant is. Its boundary returns, as buffer 1, the directions in place of $\bar{\mathbf G}$: the numbers of the slots of D230, then the fields of the seeds and the charges, $N$ values each, in f64. Each frame evaluates `@tunable` with `direction` and `direction_forces`, `directions` naming every slot and field, and hands $\Psi_n$ to `mdrtWriteTunableGradient` and $-\dot{\mathbf G}_n$ to `mdrtWriteFrameProductForces`. It evaluates no energy.
- **The host** maps $\dot{\boldsymbol\theta}$ to the directions by the exact transpose of `applyTunableChainRule`: a slot takes $\dot\theta$ of the entry its map names, a site of a field takes it at its particle, a row of a table adds $\mathrm{scale}\cdot\dot\theta$ at every particle of its type. It adds $\dot{\boldsymbol\theta}\cdot\partial U^{\mathrm{host}}/\partial\boldsymbol\theta$ at the volume of each frame to $\Psi_n$, the same constants of D230 that the reverse product weights by $g$.
- **Virtual sites.** The placement $\mathbf x_{\mathrm{site}} = \mathbf s(\mathbf x_{\mathrm{atoms}})$ does not depend on $\boldsymbol\theta$, so $\dot{\mathbf G}_{\mathrm{atoms}} = \mathsf J^{\mathsf T}\dot{\mathbf G}_{\mathrm{sites}}$ exactly, by the spread of D298 that the forces take (D384). For the extra points of Amber $\mathsf J$ depends on the positions, but no term in $\partial\mathsf J/\partial\mathbf x$ arises, unlike D387: the derivative is in $\boldsymbol\theta$, not in $\mathbf x$. Both kinds are covered; the entry at a site is zero.
- **Selecting the variant needs one line of a frozen file** (the builder freeze; see Q2): in `lib/Driver/BuilderSchedule.cpp`, line 180,

  ```cpp
  gradient.product = EntryGradientProductIR{2, 1, control.frameProductPositions};
  ```

  becomes

  ```cpp
  gradient.product = EntryGradientProductIR{2, 1, control.frameProductPositions,
                                            control.frameProductDirection};
  ```

  with a field `direction` (default false) in `EntryGradientProductIR` (`lib/Driver/EntryOutputIR.h`, not frozen) and the member in `include/mdir/Driver/Control.h`. No program without the flag changes. This is the same kind of line as D383's exception.

## Coverage

| Term | Tunables | Forward mode |
|---|---|---|
| Pairs | per-type σ, ε (seeds), NBFIX pairs, constants of pair terms, charges under a Coulomb cutoff | the pair sum along the directions; prototyped |
| Tuples | parameters of tuple terms by their seeds, `dihedral_k` | the tuple sum along the seeds at their members; prototyped |
| Terms of the positions | their constants and fields | prototyped |
| PME | charges, tied or by a map | two reciprocal sums with their forces for all entries; prototyped (a number and a field) |
| Linear sites, extra points of Amber | as their atoms | $\mathsf J^{\mathsf T}\dot{\mathbf G}_{\mathrm{sites}}$ (proposed) |
| LJPME with tunables | — | refused, as D213 |
| Positional intermediate fields, learned stages, domains | — | refused by name |
| The cell (stress) | — | refused, until PR G |

## The adapters

- **`mdir.torch`: forward AD is not formed, and today it is silently zero.** `torch.library.custom_op` with `register_autograd` registers a backward only; PyTorch 2.14.1 has no registration of a JVP for a custom op. Measured on a frame evaluator of propane in water with `epsilon_pair` as a tunable: `torch.func.jvp` of `mdir.torch.evaluate(...).energy` in the tunable returns a tangent of 0, and `torch.autograd.forward_ad` returns `None`, both without an error, where $\sum_n\partial U_n/\partial\boldsymbol\theta\cdot\dot{\boldsymbol\theta} = 6.99\times10^{6}$; the same holds for `gradients["positions"]`. That breaks the refusal matrix of #726 ("never a zero"); it is an existing defect, reported as #777, not fixed here. A forward mode of `mdir.torch` would need a `torch.autograd.Function` with a `jvp`, which `torch.compile` does not keep opaque as it does the custom op (D252). Not in this PR.
- **`mdir.jax`**: `mdir_frames_theta_tangent` with a nonzero $\dot{\mathbf G}$ wanted, through the JVP rule of Ruling 20, is part of the JAX rules of #750 (PR D of the design). This PR gives the evaluator call that its implementation and lowering need (Q4).

## Validation of the prototype

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
pending; the numbers follow in a comment.

No existing test changes: the tests of `test/Dialect/MD` and
`test/Integration` pass on the CPU (95 passed, 32 unsupported for the device).

Not in the prototype: a FileCheck test of the construction
(`differentiate-direction.mlir`, as D376's `differentiate-tangent.mlir`), the
refusals (`differentiate-direction-invalid.mlir`), and the negative cases of
the verifier (`invalid.mlir`: requests of a direction and of the tangent in
one op, `directions` without a request, out of order, or of a wrong type).

## Validation plan (after the ruling)

- Gate 3 at the evaluator: $\dot{\boldsymbol\theta}\cdot\mathbf p(g,\bar{\mathbf G}) = \sum_n[g_n\dot U_n + \bar{\mathbf G}_n\cdot\dot{\mathbf G}_n]$ to $10^{-12}$ relative in double, with $g\ne0$, on the scenarios of D381 (LJ with per-type σ, ε and NBFIX, Coulomb cutoff with tied charges, `dihedral_k` on the D372 fixtures), PME charges, and the sites of D384.
- Finite differences of $\mathbf G$ in $\boldsymbol\theta$ (Ridders, the first step as in D371's tests), $10^{-6}$ of the largest entry.
- NumPy references for LJ, NBFIX, Coulomb, `dihedral_k` ($\dot{\mathbf G}$ in closed form), $10^{-10}$ relative in double; torch-pme for the PME charges, `torch.autograd.grad` of $\dot{\boldsymbol\theta}\cdot\partial E/\partial\mathbf q$ in the positions, with the tolerance set from the accuracy of both meshes.
- CPU and RTX 5090, mixed and double; mixed against double from measurement, starting at $10^{-5}$ of the largest entry (Q9 of Ruling 21).
- The refusals by name, and the evaluator unchanged after each.
- Nothing else changes: the builder-equivalence corpus N of N, the order matrix (all three sections), five checkpoints bitwise against a clean `main` tree, the full suite; the device code of existing programs does not change, so a memcheck of the new kernels alone (a lightweight scenario).
- E3 of the design (the restriction of a tunable to its sites) applies to forward mode as to the product: recorded with the plain path.

## Questions for the maintainer

- **Q1. The Python API.** Proposed: `FrameEnergies.jvp(tangent, *, gradients=("positions",))`, with `tangent` a mapping of the names of `evaluator.tunables` to arrays of their shapes (a name left out is zero), returning a `FrameTangent` with `energy`, $\dot U_n$ $(K,)$ float64, and `gradients["positions"]`, $\dot{\mathbf G}$ $(K, N, 3)$ float64 in the order of the input, with `version`: the tangent of the outputs of `evaluate`, in their structure. With `gradients=()` the energy alone, from the kept Jacobian ($\mathsf J\dot{\boldsymbol\theta}$) or the second pass, with no product pass. The frames and the values of the tunables are those of `evaluate`, as for `vjp`. Alternatives: a method of `FrameEvaluator` that takes the frames again (rejected: `vjp` sets the precedent of the frames held by `FrameEnergies`), or a keyword of `vjp` (rejected: a JVP is not a VJP).
- **Q2. The frozen line.** The one line of `lib/Driver/BuilderSchedule.cpp` above, in this PR as D383's was, or by the builder's owner?
- **Q3. Several tangents.** One $\dot{\boldsymbol\theta}$ a call in v1, so that `jacfwd` over $M$ entries is $M$ passes over the frames; or a leading axis of $P$ tangents, $P$ potentials $\Psi$ in one program (compiled for each $P$)?
- **Q4. The adapters.** `mdir.jax` forward mode of the forces with #750 (PR D), on this evaluator call; `mdir.torch` forward AD refused by name now, in a fix of its own (#777), and formed later only if asked?

# D[force-matching]

Recorded at its merge.

**The gradient of a loss on the forces in the tunables is the derivative in the tunables of the tangent of the potential in the positions, $\Phi = \sum_i \mathbf t_i\cdot\partial U/\partial\mathbf x_i$ at a fixed tangent $\mathbf t$, which `md-differentiate` builds as a potential of its own from the sums of $U$ and differentiates by the rules of D161 and D230; `md.evaluate` asks for it with the requests `tangent` and `tangent_derivative(n)`, the tangent its last operand.**

The design and the maintainer's Ruling 21 are on #726. This is its first
part (A): the construction in the IR. The product program of the frame
evaluator and `FrameEnergies.vjp(positions=)` (B) share this label; the
reciprocal sum, the JAX rules, the PyTorch backward, forward mode in the
tunables, and the strain each have a label of their own.

## The quantity

For a loss $L$ with the cotangent $\mathbf t_i = \partial L/\partial(\partial U/\partial\mathbf x_i)$
of the gradient in the positions, $\partial L/\partial\theta =
\sum_i(\partial^2U/\partial\mathbf x_i\,\partial\theta)^{\mathsf T}\mathbf t_i$.
$U$ is twice continuously differentiable in $(\mathbf x, \theta)$ except
where a pair crosses the cutoff, a set of measure zero, so the order of the
derivatives may be exchanged, and the product is $\partial\Phi/\partial\theta$
at fixed $\mathbf t$. The matrix of the mixed derivatives is never formed.

## The IR

`md.evaluate` takes two more requests: `tangent`, of type `f64`, which is
$\Phi$, and `tangent_derivative(n)`, of the type of `derivative(n)`, which is
$\partial\Phi/\partial\theta_n$ for a number and $\partial\Phi/\partial a_i$
for a field of one `f64` component. With either, the op takes the tangent
after the arguments of the potential, a field of the type of the positions;
`tangent_derivative(n)` names an argument of the potential. The values of
the existing requests (0 to 3) are unchanged, so a program without the new
requests is printed and lowered as before.

`md-differentiate` builds $\Phi$ as `md.potential @<name>.tangent`, the
potential's arguments then $\mathbf t$, once per potential. With the
truncation expanded, each sum that reads the positions becomes a sum of the
same structure that gathers $\mathbf t$ as well:

- over a relation, $u(r)$ becomes $u'(r)\,\mathbf d_{ij}\cdot(\mathbf t_i-\mathbf t_j)/r$, with `exchange(symmetric, derived)`;
- over tuples, $\sum_k(\partial u/\partial c_k)\,\dot c_k$, with $\dot{\mathbf d} = \mathbf t_a-\mathbf t_b$ for a displacement and $\dot c = \sum_m\partial c/\partial\mathbf x_m\cdot\mathbf t_m$ from the gradient of the coordinate in its displacements (`emitCoordinateGradient`, as the forces take it);
- over particles that gather the positions, the derivative of the kernel along $\mathbf t_i$.

The kernels are differentiated by `ScalarDerivative`, whose rules are closed,
so the derivative of $\Phi$ in a parameter, a second derivative of each
kernel, needs no new scalar rule. The values of the body above the sums take
their tangents by the same rules. A value that the proof of activity finds
independent of the positions has no tangent: such a term is absent from
$\Phi$, so that the product of a tunable that only one term reads is formed
from that term alone. The pass then asks $\Phi$ for its energy and its
`derivative(n)`, by the existing path, in one generated function
(`<name>.tangent.energy_derivative2`, for example). The requests of $U$ in
the same op go to a function of their own, and the results are returned in
the order of the requests. The tangent potentials are erased at the end of
the pass.

Refused, by name: a reciprocal sum (its tangent in the charges is a later
part of #726), a field computed from the positions, a pair kernel that reads
the displacement, a learned stage, a potential on domains, and any other op
that depends on the positions.

`md-check-derivatives` checks both requests by central differences along
$\mathbf t$: `tangent` against differences of the energy, and
`tangent_derivative(n)` against differences of `derivative(n)`, each value
of a field on its own.

## Alternatives

As the design on #726 compares them: forward mode over the functions that
`derivative(n)` generates needs tangent rules for every op of a generated
function (gathers, maps, stage gathers, transfers), and reverse mode over
the force kernels needs a second family of transposes. Building $\Phi$ from
the sums needs rules for the sums alone and reuses the seeds, the three
outcomes, and the chain rule of D230.

## Validation

- The construction, by FileCheck (`test/Dialect/MD/Transforms/differentiate-tangent.mlir`): the pair kernel, the dihedral with its displacements, a term of the absolute positions, a term that reads no position and has no tangent, a potential whose tangent is zero, the split of the requests of $U$ and of $\Phi$, and no tangent potential left. The refusals (`differentiate-tangent-invalid.mlir`) and the verifier (`test/Dialect/MD/invalid.mlir`, `ops.mlir`).
- Compiled and run on the CPU in double precision (`test/Integration/tangent.mlir`): six particles in a periodic cube with a pair across its boundary, a Lennard-Jones and Coulomb pair term with a switch and charges as a field, a bond, an angle that reads the charges, a dihedral, and a term of the absolute positions. Central differences along $\mathbf t$ at the step $10^{-5}$ agree within $10^{-9} + 10^{-6}$ of the value. Measured at that step (relative): the tangent $5.2\times10^{-9}$, the derivatives in $\epsilon$ $1.3\times10^{-7}$, in the dihedral's constant $6.9\times10^{-9}$, in the constant of the term of the positions $1.6\times10^{-11}$, and in the charges $6.7\times10^{-9}$ at particle 0, which are the truncation errors of the central differences. The tangent equals $-\sum_i\mathbf t_i\cdot\mathbf F_i$ of the forces of the same call within $10^{-12}$ relative.
- No program changes: the builder-equivalence check (`scripts/builder-equivalence/check.py compare`) between this branch and the same `main`, with the bench corpus, the module and the lowered stages, the CPU and GPU targets, and both kinds of program.

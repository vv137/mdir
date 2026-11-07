# D[force-matching-hvp]

Recorded at its merge.

**The product program of a frame evaluator also gives, where it is selected, the derivative of the product of force matching in the positions, $g\,\nabla U + \mathsf H\,\bar{\mathbf G}$: the forces of the tangent potential of D376 (the request `tangent_forces`), whose pair kernels read the displacement and take its derivative component by component; terms without a rule are refused by name.**

Part 1 of 4 of #771 (part of #726). It builds on D376 (the tangent), D381 (the product program), and D382 (`mdir.torch`).

## The quantity

For a frame with the cotangents $g$ on $U$ and $\bar{\mathbf G}$ on $\mathbf G = \partial U/\partial\mathbf x$, the product of D381 is $\partial\Phi/\partial\boldsymbol\theta$ with $\Phi = g\,U + \bar{\mathbf G}\cdot\mathbf G$. Its derivative in the positions at fixed cotangents is

$$
\frac{\partial\Phi}{\partial\mathbf x_i} = g\,\frac{\partial U}{\partial\mathbf x_i} + \sum_j\frac{\partial^2U}{\partial\mathbf x_i\,\partial\mathbf x_j}\,\bar{\mathbf G}_j,
$$

the Hessian-vector product. The second term is the gradient in the positions of the tangent $\bar{\mathbf G}\cdot\partial U/\partial\mathbf x$, which `md-differentiate` builds as a potential of its own (D376). The matrix $\mathsf H$ is never formed.

## The IR

`md.evaluate` takes the request `tangent_forces`, of the type of the positions: $-\partial\Phi_t/\partial\mathbf x_i$ for the tangent $\Phi_t$ along the tangent operand. The pass asks the tangent potential for its `forces`.

- **Pairs.** The kernel of the tangent of a pair term reads the displacement $\mathbf d_{ij} = \mathbf x_i-\mathbf x_j$ besides $r$. Its forces take $-\partial k/\partial\mathbf d$ at fixed $r$, a component at a time along the unit vectors, beside the radial factor, as the forces of a term over tuples take theirs. The kernel is invariant under the exchange of the particles, so the gather keeps the antisymmetric contract. The rule applies to tangent potentials alone. The forces of other kernels that read the displacement, and the virial of any, stay refused as before.
- **Tuples and particle sums.** Their tangents read displacements, coordinates, and positions, which the existing rules of the forces take as they are.
- **The tangent of a reciprocal sum** reads the forces of the op, whose derivative in the positions has no rule yet (part 3). It is refused by name.

`md-check-derivatives` checks `tangent_forces` against central differences of the tangent in each position.

## The product program

`Control::frameProductPositions` (no control-file key), beside `Control::frameProduct`, selects the variant. The builder copies it into `EntryGradientProductIR::positions`. That one line is in `lib/Driver/BuilderSchedule.cpp`, in the builder freeze, and its owner adds it.

The variant's `@tunable` evaluation also asks for `forces` and `tangent_forces`, and hands them to the host through `mdrtWriteFrameProductForces`. The host forms $g\,\partial U/\partial\mathbf x + \mathsf H\bar{\mathbf G}$ in f64, in the order of the input.

The evaluator compiles the variant at the first product that asks for the derivative in the positions, in a simulation of its own. A program whose terms have no rule fails to compile there, and the derivative in the positions is refused with the name of the term. The product in $\boldsymbol\theta$ of D381 stays available through the program without the variant. Programs without the flag are unchanged.

## The interface

The maintainer ruled on PR #772: option 1 of the three that the draft proposed.

`FrameEnergies.vjp(cotangent=None, *, positions=None, gradients=("tunables",))`. With `"positions"` in `gradients`, the result, a `FrameGradient`, also has `positions`, a `(K, N, 3)` float64 array in the order of the input: $g_n\,\partial U_n/\partial\mathbf x + \mathsf H_n\bar{\mathbf G}_n$, from the same pass as the derivative in $\boldsymbol\theta$.

- Without `"positions"`, `positions` is `None`, and nothing changes from D381.
- With `("positions",)` alone, the mapping is empty.
- Without a cotangent on the gradient, the result is $g_n\,\partial U_n/\partial\mathbf x$.

The names follow `gradients=` of `evaluate`. The other options were a method of its own, which costs two passes for both, and a flag, which names one thing where `gradients` names a set. `mdir.torch` keeps the refusal of D382 until part 4.

## Validation

(Filled in by the PR.)

# D[force-matching-product]

Recorded at its merge.

**`FrameEnergies.vjp(cotangent=None, *, positions=None)` gives, with `positions`, a cotangent on the gradients in the positions $\partial U_n/\partial\mathbf x_i$, the product $\sum_n[g_n\,\partial U_n/\partial\boldsymbol\theta + \sum_i\bar{\mathbf G}_{n,i}\cdot\partial^2U_n/\partial\mathbf x_i\,\partial\boldsymbol\theta]$ by a pass of a second program of the evaluator, the product program, compiled at the first product; it evaluates no energy and hands the derivatives of the energy and of its tangent (D376) to the chain rule of D230, where the host applies $g_n$.**

Part B of the design on #726 (Ruling 21: Q2, a second program; Q3, the
`vjp` extension only, no loss in MDIR; Q4, the cotangent is on
$\mathbf G = \partial U/\partial\mathbf x$, the sign of a force loss is the
caller's). It continues D376 (part A, the tangent in the IR) and uses the
configuration of D380 (PR 0).

## The interface

`vjp(cotangent=None, *, positions=None)`. `cotangent` is $(K,)$ on the
energies, as before; `positions` is $(K, N, 3)$ float64 on
`gradients["positions"]`, in the order of the input. With `positions`
alone the cotangent of the energy is zero. The result is a `FrameGradient`
with `zero` and `version`. The frames are evaluated again, so the frames
of `evaluate` must be a sequence (an iterator is refused), and the values
of the tunables must still be those of `evaluate`, as for the second pass.
The gradient in the positions need not have been asked of `evaluate`.

For a loss $L = \frac1K\sum_n\sum_i\lVert\mathbf F_{n,i}-\mathbf F^\mathrm{ref}_{n,i}\rVert^2$:

```python
out = evaluator.evaluate(frames, gradients=("tunables", "positions"))
forces = -out.gradients["positions"]
grad = out.vjp(positions=-2.0 / len(frames) * (forces - f_ref))   # dL/dG = -dL/dF
```

## The product program

`Control::frameProduct` (D380) selects it, in a simulation of its own that
the evaluator creates at the first product, with the values of the tunables
of the evaluator copied in where their bits differ. The ordinary frames
program does not change.

- **Its boundary** is `mdrtFrameProductBoundary`. It gives the positions and, as returned buffer 1, the cotangent of the gradient in the positions, in the type of the state. Request element 2 holds the cotangent of the energy (D380).
- **Each frame** evaluates `@tunable` once, with `derivative(n)` and `tangent_derivative(n)` of every slot and field of D230, and the cotangent as the tangent. It hands over the scalars (those of the energy, then those of the tangent) through `mdrtWriteTunableGradient`, and the fields (those of the energy, then those of the tangent) through `mdrtWriteTunableGradientField`. It evaluates no energy, no forces, and no virial: that is E2 of the design.
- **The host** applies the chain rule of D230 to each part, $p = g\,\mathcal C(\partial U/\partial\mathbf b; 1) + \mathcal C(\partial\Phi/\partial\mathbf b; 0)$. The second argument of $\mathcal C$ weights what the host adds to the energy (the tails, the shift estimates, the dispersion correction, the self term and background of PME). That addition depends on the volume and $\boldsymbol\theta$, not on the positions at a fixed cell. Since the host applies $g$ to it, it applies $g$ to the device part too, and the program does not read request 2.

## What is refused

The product of a program with virtual sites (#726, a later part) and the
derivative of the tangent of a reciprocal sum in the charge of each particle
(C(ii) of the design; the directional form of #757 takes a number that the charges depend on) are refused. The charges as a field with PME
therefore take no product yet: `vjp(positions=)` raises `UnsupportedError`,
naming the reason, when the product program cannot be compiled.
`terms="dependent"` is refused too.

## Validation

(Filled in by the PR.)

# D[force-matching-torch]

Recorded at its merge.

**In `mdir.torch`, the gradient of the energy in the positions is differentiable in the tunables. A cotangent on it reaches the product program of the evaluator (D381), whether the gradient is the output `gradients["positions"]` or the gradient of `energy` taken with `torch.autograd.grad(..., create_graph=True)`. Its derivatives in the positions (the Hessian) and in the cell stay refused, where a backward needs them.**

Part E of the design on #726 (Ruling 21; section 4 of the design, "`mdir.torch`"). It uses the product of D381 and adds nothing to the IR or to the evaluator.

## The interface

No new function and no new argument. Two forms of a loss on the forces train `theta` by conventional autograd. There is no loss in MDIR:

```python
# The gradient as an output, with positions that take no gradient.
out = mdir.torch.evaluate(evaluator, theta, frames, gradients=("positions",))
loss = ((-out.gradients["positions"] - f_ref) ** 2).sum(dim=(1, 2)).mean()
loss.backward()

# The gradient of the energy in positions that require one.
x = torch.tensor(frames, requires_grad=True)
out = mdir.torch.evaluate(evaluator, theta, x)
g, = torch.autograd.grad(out.energy.sum(), x, create_graph=True)
loss = ((-g - f_ref) ** 2).sum(dim=(1, 2)).mean()
grads = torch.autograd.grad(loss, list(theta.values()))
```

## How

- **`mdir::frame_vjp`.** Until now it refused a cotangent on the gradient in the positions that is not zero. Now it forms the product of `FrameEnergies.vjp(cotangent, positions=)`, with the cotangent of the energy in the same pass. That needs the frames again, so the record of the call keeps its source wherever the gradient in the positions is asked. A backward after the evaluator has evaluated other frames raises `SimulationError`, as the second pass does. The strain and the gradient in the cell still raise.
- **`mdir::frame_input_vjp`.** The product $g_n\mathbf G_n$ of the first backward now has a backward of its own: in $g$, $\sum\bar{\mathbf y}\cdot\mathbf G$; in $\mathbf G$, $g\,\bar{\mathbf y}$. Before, it raised. Its input $\mathbf G$ is the output of `mdir::frame_energies`, saved with `save_for_backward`, so that a backward with `create_graph=True` reaches that operation again with a cotangent on $\mathbf G$.
- **The refusal of the Hessian and of the derivative in the cell.** It applies where the backward takes a gradient through the positions or the cell, since the derivative of $\mathbf G$ there is not implemented. A tensor that takes a gradient enters `mdir::frame_energies` through a view of itself, a node of the graph of its own. The backward then asks the engine (`torch._C._will_engine_execute_node`) whether it will run that node. PyTorch does not answer this for a leaf under `torch.autograd.grad`, but it does for the view. So `torch.autograd.grad(loss, theta)` is given, while `torch.autograd.grad(loss, x)` and a `loss.backward()` that reaches positions requiring a gradient raise `UnsupportedError`.
- **A private function of PyTorch** (the maintainer's direction on #770). `torch._C._will_engine_execute_node` is not public, and this was tested with PyTorch 2.14.1. The question fails closed: where the function is missing, raises, or answers nothing, the gradient is taken as needed and the derivative is refused, also for `torch.autograd.grad(loss, theta)`; a test checks both cases. The detection is removed once the Hessian-vector product in the positions, $g\,\nabla U + \mathsf H\bar{\mathbf G}$, is formed for every term (a planned part of #726). The backward then gives the gradient in the positions as well.
- $\partial^2U/\partial\boldsymbol\theta^2$, the backward of `mdir::frame_vjp`, stays refused.

The rest of B's limits hold: virtual sites, and the charges as a field with PME, are refused when the product is compiled.

## Validation

`test/Driver/python-frame-product-torch.test` (CPU) and `-gpu.test` (RTX 5090) form the product of D381 by autograd, in both forms above. They run the scenarios of `python_frame_product.py` with the product of `mdir.torch` in place of `FrameEnergies.vjp`, so that the references of D381 check the backward of the adapter.

- **Both forms against `FrameEnergies.vjp`:** equal to the bit on every scenario, on the CPU and the RTX 5090, in double and mixed. The tolerances were $10^{-14}$ and $10^{-6}$.
- **The references of D381, through `mdir.torch`.** The relative error, of the largest entry, is that of `vjp`:

  | Reference | CPU double | CPU mixed | GPU double | GPU mixed |
  |---|---|---|---|---|
  | `dihedral_k`, the closed form along the cotangent | 4.6e-16 | 3.6e-6 | 1.0e-15 | 4.1e-6 |
  | $\sigma$, $\epsilon$ (propane and water), NumPy | 1.7e-11 | 4.2e-7 | 1.7e-11 | 1.3e-6 |
  | the charges under a Coulomb cutoff, NumPy | 4.7e-15 | 1.8e-6 | 2.5e-15 | 1.6e-6 |

  Central differences of the rows along the cotangent agree to 1.1e-12 to 9.1e-10 in double. Mixed against double agrees to 4.2e-7 (CPU) and 1.3e-6 (GPU).
- **`torch.autograd.gradcheck`, in double,** of $\mathbf G$ as a function of `theta`, in both forms. This is the second-order check of a force loss: the Jacobian of the forces in `theta` from the backward, against finite differences of the forces. It passed for `dihedral_k` on the torsion fixtures and for $\sigma$, $\epsilon$ of two pairs of types on propane and water, on the CPU and the GPU, with eps $10^{-5}$, atol $10^{-5}$, and rtol $10^{-5}$. With eps $10^{-6}$, atol $10^{-7}$, and rtol $10^{-6}$ it failed on propane. The entries of $\partial\mathbf G/\partial\boldsymbol\theta$ there are of $10^3$ to $10^4$, so a difference of $\mathbf G$ at a step of $10^{-6}$ carries a rounding of about $10^{-7}$.
- **`gradgradcheck`** differentiates the first backward in every input, so it needs $\partial^2U/\partial\boldsymbol\theta^2$, which stays refused. `gradcheck` of the gradient takes its place.
- **The refusals:**
  - the Hessian by `torch.autograd.grad(loss, x)`, by `loss.backward()`, and through a tensor that moves the positions;
  - the derivative of the gradient in the strain, and $\partial^2U/\partial\boldsymbol\theta^2$;
  - without the private function, or with no answer from it, the derivative even for `torch.autograd.grad(loss, theta)`;
  - a backward after another call of the adapter.

  After each, the evaluator gives the same gradient to the bit.
- **The existing tests pass unchanged.** `python-frame-gradients-torch.test` asserts the wording of two refusals that are still refusals: a backward through `gradients["positions"]` with positions that take a gradient, and the second derivative in the cell. The messages keep that wording.

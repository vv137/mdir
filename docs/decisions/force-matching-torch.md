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

(Filled in by the PR.)

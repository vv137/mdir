# D[torch-forward-refusal]

Recorded at its merge.

**Forward-mode AD through the operations of `mdir.torch` is refused by name: a tangent of `torch.func.jvp`, `jacfwd`, or `torch.autograd.forward_ad` that reaches an operation, on an argument of `evaluate` or on a cotangent of a backward, raises `UnsupportedError` ("forward mode ... is not implemented; reverse mode (torch.autograd.grad / backward) is"), naming the tensor that carries it; it never gives a zero.**

Issue #777. The maintainer ruled on PR #778 (2026-11-08): forward mode in `mdir.torch` is refused by name in this separate fix; forward mode itself is built only if asked.

## The defect

The operations of `mdir.torch` (D252, D382, D385) are `torch.library.custom_op`s with `register_autograd`, which gives a backward alone. PyTorch 2.14.1 wraps that backward in an `autograd.Function` without a `jvp`, and under forward mode it then carries no tangent through the operation and raises nothing. Measured on the torsion fixtures of `test/Driver/Inputs/python_frame_product.py` (`dihedral_k`, 2 frames, CPU, double) before this decision:

| Call | Result |
|---|---|
| `torch.func.jvp` of `energy` in `theta`, in the positions, in the cells | tangent 0 |
| `torch.func.jacfwd` of `energy` in `theta` | a Jacobian of zeros |
| `torch.func.jvp` of `gradients["positions"]` in `theta` | tangent 0 |
| `torch.autograd.forward_ad` (`make_dual`, `unpack_dual`) of `energy` in `theta`, in the positions; of `gradients["positions"]` | tangent `None` |
| a dual cotangent of `torch.autograd.grad(energy, theta, cotangent)` | tangent `None` |
| `torch.func.jvp` and `forward_ad` of a function under `torch.compile` (aot_eager) | 0 and `None` |
| `forward_ad` over `torch.autograd.grad(..., create_graph=True)` | `NotImplementedError` of PyTorch |

The expected tangent, $\sum_n\partial U_n/\partial\boldsymbol\theta\cdot\dot{\boldsymbol\theta}$, is not zero, so each of the silent results was wrong. This contradicts the refusal matrix of #726 and the rule of D252 that a derivative that is not implemented raises.

## The mechanism

The tangent is detected where a tensor enters an operation, by the public `torch.autograd.forward_ad.unpack_dual(t).tangent is not None`:

- in `evaluate` (its first function, `_begin`), on every tensor of `theta` and on the positions, the cells, and the tilts where they are tensors, whether or not they require a gradient (forward mode runs without one);
- in each backward of the operations (`mdir::frame_energies`, `frame_positions`, `frame_cell`, `frame_energy_in`, `frame_input_vjp`), on its cotangents: forward over reverse, a dual cotangent of a backward, would otherwise go through the backward without its tangent.

Probes on PyTorch 2.14.1 showed that the dual is visible at those two places and nowhere else: the body of a custom operation runs below the key of autograd, where the dispatcher has taken the tangent off. Inside `torch.func.jvp` the arguments are wrapped tensors of functorch, and `unpack_dual` at the current level gives their tangent all the same; under `vmap` of `jvp` too. Under `torch.compile` the functions of the adapter that check the arguments are outside the compiled graph (`torch.compiler.disable`, D385), and the check sees the dual there.

Outside a dual level, `unpack_dual` returns at once with no tangent, so reverse mode takes the same path as before, and gives the same numbers, to the bit.

Other mechanisms were set aside:

- `register_autograd` has no rule of forward mode, and `torch.library` has no public registration of one for a `custom_op`.
- A `torch.autograd.Function` with a `jvp` that raises would make the operation transparent to `torch.compile`, which keeps a custom operation opaque (D252).
- Refusing wherever a dual level is active (`torch.autograd.forward_ad._current_level`, or the interpreters of `torch._C._functorch`) reads private state, which D385 removed from the adapter, and refuses a `jvp` whose tangent does not reach the operations. Such a `jvp` (in a number that multiplies the energy) is given, and is tested.

## What stays

`torch.func.hessian`, `jacfwd(jacrev(...))`, and `jvp(grad(...))`, and `torch.func.grad` and `jacrev` alone, stop at their reverse level, before any tangent reaches the operations, in PyTorch's `RuntimeError` for an `autograd.Function` without the support of `torch.func` (it must override `setup_context`). That was so before this decision; it is no zero. The transforms of reverse mode of `torch.func` are not supported by `mdir.torch`, and refusing them by name is a separate question (open below).

`evaluate` keeps its signature. No key of a control file changes.

## Validation

`test/Driver/python-frames-torch-forward.test` (CPU) and `python-frames-torch-forward-gpu.test` (GPU), in double and mixed precision, on the torsion fixtures with `dihedral_k` tunable, 2 frames in a cubic cell, `test/Driver/Inputs/python_frames_torch_forward.py`:

| Check | Result |
|---|---|
| 19 forms: `jvp` in `theta`, the positions, the cells, the tilts, and of `gradients["positions"]`; `jacfwd` in `theta`, the positions, and of `gradients["positions"]`; `vmap` of `jvp`; `forward_ad` in `theta`, the positions, the cells, the tilts, and of `gradients["positions"]`; `forward_ad` over `autograd.grad` with `create_graph=True`; a dual cotangent through the energy and through the gradient in the positions; `jvp` and `forward_ad` under `torch.compile` (aot_eager) | each raises `UnsupportedError` with "forward mode", the text above, and the name of the tensor |
| `torch.func.hessian` in `theta` and in the positions | PyTorch's `RuntimeError`, never a zero |
| a `jvp` whose tangent does not reach the operations | the energy, to the bit |
| after the refusals, the gradients of reverse mode in `theta`, the positions, and the cell, and of a loss on the forces in `theta` | equal to those before, to the bit |

The script fails at its first form with the adapter of before this decision (a tangent of 0, no error). The existing tests of `mdir.torch` pass unchanged.

## Open

- Whether the transforms of reverse mode of `torch.func` (`grad`, `jacrev`, `hessian`) should be refused by name, `UnsupportedError`, in place of PyTorch's `RuntimeError`. Without a private function of PyTorch the adapter would have to recognize that error by its text.
- Forward mode itself (#726 part F) is built only if the maintainer asks.

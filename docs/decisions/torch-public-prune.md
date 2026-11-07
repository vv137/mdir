# D[torch-public-prune]

Recorded at its merge.

**In `mdir.torch`, the positions and the cell that take a gradient are nodes of the graph of their own, beside that of the tunables, so that autograd runs the backward of each only on a path to its input: the derivative of the gradient in the positions, in the positions, is formed where a backward takes a gradient in the positions (D383), and refused by the name of a term without a rule there alone; the private question to the engine of D382 goes.**

Part 4 of #771 (part of #726). The maintainer ruled on PR #775: the split is adopted now, before the reciprocal sum has a rule, and the shared pass of the draft is left out.

## What changes from D382

D382 refused the derivative of $\mathbf G = \partial U/\partial\mathbf x$ in the positions, the Hessian, wherever a backward needed it. To give `torch.autograd.grad(loss, theta)` while the positions require a gradient, it asked the engine of autograd, through the private `torch._C._will_engine_execute_node`, whether the backward would run a view of the positions. Where the function was missing or gave no answer, it refused even that gradient.

Now:

- `torch.autograd.grad(loss, x)`, and a `loss.backward()` that reaches positions requiring a gradient, are given where every term has a rule for the derivative in the positions (D383): pairs, tuples, terms of the positions, and the linear virtual sites of D[force-matching-sites]. The gradient in the positions of $\Phi = g\,U + \bar{\mathbf G}\cdot\mathbf G$ is $g\,\nabla U + \mathsf H\bar{\mathbf G}$.
- Where a term has no rule (the reciprocal sum of PME, the extra points of Amber that follow the directions of their bonds), such a backward raises `UnsupportedError`, naming the term. `torch.autograd.grad(loss, theta)` and `loss.backward(inputs=theta)` stay given there.
- The private function is not called, and the case without it goes.
- The derivative of $\mathbf G$ in the cell, the derivatives of the strain and of the gradient in the cell, a backward through the virial or an observable, and $\partial^2U/\partial\boldsymbol\theta^2$ stay refused as in D382, with the same wording. The derivative of $\mathsf H\bar{\mathbf G}$, a third derivative, is refused too.

`mdir.torch.evaluate` keeps its signature. No key of a control file changes.

## The mechanism

Autograd runs only the nodes that lie on a path to the inputs that a backward takes. The adapter gives the inputs nodes of their own:

- `mdir::frame_energies` takes the tunables alone. Its backward is the product in $\boldsymbol\theta$ of D381 (`mdir::frame_vjp`), as before.
- `mdir::frame_positions(x)` returns zeros, added to the outputs: the observed columns, $\mathbf G$, the gradient in the strain, and that in the cell. With a cotangent $\bar{\mathbf G}$ that is not zero, its backward forms $\mathsf H\bar{\mathbf G}$ by `mdir::frame_hvp`, a pass of the product program of D383 at a zero cotangent of the energy. A cotangent on any other of those outputs raises, as in D382. A zero $\bar{\mathbf G}$, which the first backward of a force gives, forms nothing.
- `mdir::frame_cell(cells, tilts)` does the same for the cell; every cotangent that is not zero raises there.
- `mdir::frame_energy_in(x, cells, tilts, ...)` returns zeros, added to the energy. Its backward is $g\,\mathbf G$ and $g\,\mathbf C$ (`mdir::frame_input_vjp`), with $\mathbf G$ and $\mathbf C$ the summed outputs. A backward of this backward then reaches both the node of the tunables (the product in $\boldsymbol\theta$) and that of the positions ($\mathsf H\bar{\mathbf G}$), and the engine runs each only where its input is taken.

These nodes are built in a function marked `torch.compiler.disable`. A compiled region becomes one node of the backward with an edge to every input it takes, so it cannot be pruned inside. The positions and the cell therefore stay out of the compiled graph, and `mdir::frame_energies` stays in it.

## The reliability of the pruning

The autograd notes of PyTorch (Backward Hooks execution) state that under `torch.autograd.grad`, the nodes of the tensors in `inputs` "will not be executed unless there is another input that depends on the gradient result of this Node". The docstring of `torch.autograd.graph.register_multi_grad_hook` shows `backward(inputs=(a,))`, which computes no gradient for another leaf. No document states in general terms that a node off every path to the inputs is not run. The tests observe it for `torch.autograd.grad(..., inputs)`, `backward(inputs=)`, a nested `create_graph=True`, and a forward under `torch.compile` (`aot_eager`).

The result does not depend on the pruning. A node of the positions that ran where it need not would form the right $\mathsf H\bar{\mathbf G}$, at the cost of a pass, or refuse by name. That is the direction in which D382 failed closed. One case is pruned neither here nor in D382: tunables and positions computed in one compiled region upstream. That region is one node, on the path to the tunables through the positions too, so with PME `torch.autograd.grad(loss, phi)` raises there. Upstream code that runs eagerly is pruned.

## The cost

Where the tunables and the positions are both taken, each node runs its own pass: the product in $\boldsymbol\theta$, and the pass that gives $\mathsf H\bar{\mathbf G}$. The draft shared one pass between them. It needed the engine to run the node of the positions first, which it does but no document states, and a check of the version of the values. The maintainer left it out. A later change may add it, with that check, if a measurement shows the need.

A force loss on a fluid of 1000 atoms (Lennard-Jones with NBFIX, and Coulomb under a cutoff), 16 frames, the best of 5. The reference is one pass of `FrameEnergies.vjp(..., gradients=("tunables", "positions"))`, since D382 refused this case:

| | CPU double | `main` | RTX 5090 mixed | `main` |
|---|---|---|---|---|
| `grad(L, theta)` | 15.7 ms | 15.8 ms | 2.4 ms | 2.5 ms |
| `grad(L, [theta, x])`: two passes | 36.0 ms | refused | 5.2 ms | refused |
| `grad(L, x)` | 20.6 ms | refused | 2.8 ms | refused |
| `vjp(positions=)`, $\boldsymbol\theta$ alone | 15.1 ms | 15.1 ms | 2.2 ms | 2.2 ms |
| `vjp(..., gradients=("tunables", "positions"))` | 20.1 ms | 20.0 ms | 2.7 ms | 2.7 ms |

`grad(L, theta)` costs what it cost. Taking both costs 1.8 (CPU) and 1.9 (GPU) times one pass of `vjp` with both.

## Alternatives

1. Remove the question and nothing else. Every backward with positions that require a gradient would then form $\mathsf H\bar{\mathbf G}$, and PME would be refused even for `grad(L, theta)`. Rejected: a regression.
2. Keep the question until every term has a rule. That keeps the private function for as long as PME and the extra points of Amber lack one.
3. One node that tries the derivative in the positions and falls back when it is refused. A backward that needs the positions would then be given a wrong value or would have to raise from the node of the tunables. Rejected.
4. An argument of `evaluate` that declares whether the derivative in the positions is wanted. It repeats what autograd knows, as a new argument.
5. The shared pass: left out (above).

## Validation

- `test/Driver/python-frame-hvp-torch.test` (CPU) and `-gpu.test` (RTX 5090), in double and mixed: on the torsion fixtures of D372 and the pair fluid of D383, every form of the backward against `FrameEnergies.vjp`, and the passes each form makes. The forms are `grad(L, theta)`, `grad(L, x)`, both together, `L.backward()`, `L.backward(inputs=theta)`, a loss with a term of the energy, and the forward compiled with `aot_eager`. With PME, `grad(L, theta)` and `backward(inputs=theta)` are given and equal to `vjp`, while the forms that take the positions are refused, naming the reciprocal sum.
- `test/Driver/python-frame-hvp-torch-amber.test` (CPU) and `-gpu.test` (RTX 5090), with ParmEd: OPC from a prmtop of Amber cut to 40 waters (the fixture of D[force-matching-sites]). `grad(L, theta)` is given and equal to `vjp`, and the forms that take the positions are refused, naming the extra points. The linear sites of OPC from GROMACS: `grad(L, x)` is given and equal to `vjp`.
- `python-frame-product-torch.test`: the refusals of the Hessian by `torch.autograd.grad`, by `backward()`, and through a tensor that moves the positions now run on PME, where they name the reciprocal sum. The block on the missing private function is replaced by `grad(L, theta)` on PME with the positions requiring a gradient, equal to `vjp` to the bit.
- The other tests of `mdir.torch` pass unchanged, with every refusal of the cell, the strain, the virial, and the observables in its wording.

The numbers of the comparisons are in the PR, and are recorded here at the merge.

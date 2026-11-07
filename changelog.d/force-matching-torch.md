### Added

- **Force matching through `mdir.torch`** (D[force-matching-torch], #726): the gradient in the positions, as the output `gradients["positions"]` or as the gradient of `energy` taken with `torch.autograd.grad(..., create_graph=True)`, is differentiable in `theta`, so a loss on the forces trains the tunables by the product of `FrameEnergies.vjp(positions=)`. The Hessian in the positions and the derivative in the cell stay refused where a backward needs them ([docs/python-frames.md](docs/python-frames.md)).

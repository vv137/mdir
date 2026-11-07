### Added

- **The derivative in the positions of force matching with the extra points of Amber** (D[force-matching-amber-curvature], #771): `FrameEnergies.vjp(..., gradients=("tunables", "positions"))`, and in `mdir.torch` `torch.autograd.grad(loss, x)` and a `loss.backward()` that reaches the positions, are given for OPC and TIP4P-Ew from a prmtop, with the curvature of their placement along the bisector, where they were refused ([docs/python-frames.md](docs/python-frames.md)).

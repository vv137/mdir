### Added

- **The product of force matching in the frame evaluator** (D[force-matching-product], #726): `FrameEnergies.vjp(cotangent=None, *, positions=None)` takes a cotangent on the gradients in the positions and returns the derivative in the tunables of a loss on the forces, $\sum_n\sum_i\bar{\mathbf G}_{n,i}\cdot\partial^2U_n/\partial\mathbf x_i\,\partial\boldsymbol\theta$, with the energy's term besides. It runs a second program of the evaluator, compiled at the first product, without forming the matrix of mixed derivatives ([docs/python-frames.md](docs/python-frames.md)).

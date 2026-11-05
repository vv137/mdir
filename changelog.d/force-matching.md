### Added

- **The tangent of a potential in the positions** (D376, #726): `md.evaluate` takes the requests `tangent`, $\sum_i \mathbf t_i\cdot\partial U/\partial\mathbf x_i$ along a tangent field given after the arguments, and `tangent_derivative(n)`, its derivative in an argument, the mixed derivative that force matching needs, without the matrix of mixed derivatives. `md-differentiate` builds the tangent from the sums of the potential; `md-check-derivatives` checks both by central differences ([docs/ops.md](docs/ops.md), Section 5.3.2).

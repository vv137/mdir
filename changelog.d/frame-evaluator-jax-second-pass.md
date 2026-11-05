### Added

- **`mdir.jax`: a second pass for the derivative in `params`** (D375, #138): `energy_fn(program, jacobian="recompute")` keeps no Jacobian in the tunables. It evaluates the frames again and contracts each frame's row with the tangent or the cotangent as it is formed, for `jax.grad`, `jax.jvp`, `jacfwd`, and `jacrev` in `params`, on the CPU and on a CUDA device. `jacobian="keep"`, the default, keeps the rows as before; above `jacobian_bytes` its error names the second pass.

### Added

- **`mdir.jax` on a CUDA device** (D373, #138): `mdir.jax.energy_fn` takes arrays on a CUDA device under `jax.jit`, `jax.grad`, `jax.jvp`, and `jax.vmap`, and returns its outputs there. A program compiled for the GPU or for the CPU may be called on either platform.

### Known limitations

- On a CUDA device, a call of `mdir.jax` copies its inputs and outputs through host memory on XLA's stream and waits for them, until the boundary of a batch is on the device (#138).

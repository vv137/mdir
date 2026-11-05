### Added

- **`mdir.jax`: the frame evaluator as a pure function of JAX on the CPU** (D[frame-evaluator-jax], #138). `mdir.jax.energy_fn(program)` returns `energy_fn(params, positions, cell)`, the potential energy of a frame. It works under `jax.jit`, `jax.grad` and `jax.jvp` (first order, in the tunables, the positions, and the cell), and `jax.vmap` over frames (one call for the batch); every second derivative raises. `mdir.jax.params(program)` gives the tunables as its pytree. The handler is a typed XLA FFI target built into the extension, with the FFI headers of jaxlib 0.11.2 vendored in `third_party/xla_ffi`; the extra `mdir[jax]` needs JAX 0.11.2 or later (`docs/python-frames.md`).

### Changed

- **`mdir.torch.evaluate` forms the gradient in the cell with the function of the extension that `mdir.jax` uses**, in place of NumPy, so that the two adapters give the same bits (D[frame-evaluator-jax]). On the frames of the tests it changes by at most $5.4\times10^{-14}$ of the largest entry of a frame, and by up to $3.3\times10^{-12}$ relative for the smallest entries.

### Known limitations

- `mdir.jax` runs on the CPU only, on the evaluator's existing loop of frames: the positions of a frame are copied within host memory, and each call waits for its frames. A derivative in the tunables above `jacobian_bytes` raises until the second pass is implemented (#138).

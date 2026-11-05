# D[frame-evaluator-jax-cuda]

Recorded at its merge.

**`mdir_frames` has a handler for the platform `"CUDA"`. Until the boundary of a batch is on the device (PR 1 of #138), it stages a call through the host on XLA's stream.** This is PR 3 of the design on #138 (Ruling 20), and it extends D371 with no change of the API. `mdir.jax.energy_fn` takes arrays on a CUDA device as it takes arrays on the CPU. See [python-frames.md, The JAX adapter](../python-frames.md#the-jax-adapter).

**Registration.** The handler is built where MDIR is built with CUDA. `_core._xla_ffi_platforms()` then names `"CUDA"` beside `"cpu"`. `mdir.jax` registers the type of the state and the target for `"CUDA"` too, when JAX has a CUDA backend in the process. The CUDA plugin of XLA keeps a registry of types of its own, so the state of an executable of CUDA is a type of its own. The extension links no CUDA library: it takes `libcuda.so.1` as the process has loaded it.

**The handler in v1.** On XLA's stream, it:

1. copies θ, the positions, and the cells to the host, and waits;
2. evaluates the frames on the loop of the CPU handler (the same function, so the two handlers give the same bits on one program);
3. copies the outputs that were asked back to XLA's buffers, and waits again.

The work of XLA before and after the call is therefore ordered with it, through two waits on the host. The design's call with no wait on the host comes with PR 1, inside the handler.

**The device.** A program of the GPU called on a device other than its runtime's is refused when the call runs (`MDRT_DEVICE`, else the first visible device). The check is in `execute`, because XLA 0.11.2 gives no device ordinal to `instantiate`.

The gate and its open question are on the PR.

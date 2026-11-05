# D[frame-evaluator-jax-second-pass]

Recorded at its merge.

**The derivative of `mdir.jax.energy_fn` in `params` has a second pass for a Jacobian above `jacobian_bytes`: two primitives, `mdir_frames_theta_tangent` and `mdir_frames_vjp`, whose handlers evaluate the frames and contract the row of each frame with the tangent or the cotangent as it is formed, with no Jacobian kept.** This is PR 4 of the design on #138 (Ruling 20). See [python-frames.md, The JAX adapter](../python-frames.md#the-jax-adapter).

**The choice is a static argument of the function, `energy_fn(program, *, jacobian="keep" | "recompute", jacobian_bytes=2**28)`** (the maintainer's ruling on the PR, option 2 of three). The JVP rule of `energy_fn` sees one frame, and every call of $K > 1$ frames is formed by the batching rule of `mdir_frames` from what the rule asked for, so the choice cannot be made per call. Its cost in reverse mode is one evaluation of the frames with the rows kept ($8KM$ bytes) against two with the second pass ($O(K + M)$), and that is what the user chooses. The other options were `jacobian_bytes` alone, taking the second pass below the Jacobian of one frame as `FrameEvaluator` does, and the second pass always.
- `"keep"`, the default, is the behavior of D371. One evaluation for a derivative in `params` is what the reweighting of #140 wants.
- `jacobian_bytes` stays the check of the total $8KM$ of the rows that `"keep"` asks for, for one frame and for the frames of a `vmap`. Above it, the error names `jacobian="recompute"`, a larger bound, and `jax.lax.map` over chunks under `jax.checkpoint`.
- An unknown value raises `InputError` when the function is made.

**The primitives.** `mdir_frames_theta_tangent` takes θ, the positions, the cells, and $P$ tangents $(P, M)$, and gives the products $(K, P)$. `mdir_frames_vjp` takes $P$ cotangents $(P, K)$ and gives the sums $(P, M)$.
- Each is linear in its last argument and is the transpose of the other.
- Their JVP rules refuse every second derivative, as D371's does.
- Their handlers, on the CPU and on CUDA (D373), run the loop of frames of `mdir_frames`. They add each frame's row as it is formed, in f64 in the order of the frames, and keep no Jacobian.
- In v1 the accumulation is on the host, where `evaluateFrame` gives the row. The boundary of a batch on the device (PR 1 of #138) moves it there, with no change of the primitives.

**The batching rules.**
- The tangents or the cotangents alone mapped (`jacfwd`, `jacrev`): one call.
- The frames alone mapped, for the tangents: one call.
- Otherwise (the sums of each example, as `vmap` of `grad` asks): the rows of the frames of the batch in one call of `mdir_frames`, contracted with each example's own tangents or cotangents. The result has the size of the rows.
- θ mapped: refused (Q4).

**The cost, measured.** On 4 frames, `grad` in `params` evaluates 4 frames with either choice: with `"recompute"` the energies of the first pass are not used, and XLA removes that call. `value_and_grad` evaluates 4 with `"keep"` and 8 with `"recompute"`.

**Validation.** The second pass is compared with the kept rows on the CPU and on the RTX 5090, double and mixed:
- every transformation in `params`, against `"keep"`: at most $6.2\times10^{-16}$ of the largest entry, tolerance $10^{-12}$;
- central differences in `params`: at most $2.5\times10^{-11}$, tolerance $10^{-6}$;
- `check_grads` passes.

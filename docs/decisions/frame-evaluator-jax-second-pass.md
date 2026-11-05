# D[frame-evaluator-jax-second-pass]

Recorded at its merge.

Proposed; the choice of the second pass is a question to the maintainer on its PR.

**The derivative of `mdir.jax.energy_fn` in `params` has a second pass for a Jacobian above `jacobian_bytes`: two primitives, `mdir_frames_theta_tangent` and `mdir_frames_vjp`, whose handlers evaluate the frames and contract the row of each frame with the tangent or the cotangent as it is formed, with no Jacobian kept.** This is PR 4 of the design on #138 (Ruling 20). See [python-frames.md, The JAX adapter](../python-frames.md#the-jax-adapter).

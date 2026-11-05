### Added

- **`mdir.jax`: a second pass for the derivative in `params`** (D[frame-evaluator-jax-second-pass], #138): above `jacobian_bytes`, the frames are evaluated again and each frame's row is contracted with the tangent or the cotangent as it is formed, so no Jacobian is kept.

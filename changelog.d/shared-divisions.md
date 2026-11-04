### Performance

- **Under `fast_math`, the divisions of an f64 kernel of a loop over pairs are products by one shared reciprocal of each divisor, and on a GPU a division by $\sqrt{x}$ is a product by $\mathrm{rsqrt}(x)$**, which shortens a step of JAC in double precision on a GPU. Double runs no longer give the same bits as before this change (deterministic runs still repeat their bits); the mixed and single modes are unchanged (D[shared-divisions], #731).

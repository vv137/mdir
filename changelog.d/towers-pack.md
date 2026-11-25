### Changed

- TOWERS selects between compact and wide GPU pack kernels once at setup. AUTO selects wide when the initial maximum exceeds 128 atoms per cell. The `[execution]` field `towers_shape.pack` can pin either kernel. The wide kernel sorts 129–256 atoms per cell with a warp bitonic network. Both kernels preserve quarter-cell and stable atom ID order (D[towers-pack]).

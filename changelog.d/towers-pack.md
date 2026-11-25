### Changed

- TOWERS selects between compact and wide GPU pack kernels using cell population. The wide kernel sorts 129–256 atoms per cell with a warp bitonic network. Both kernels preserve quarter-cell and stable atom ID order (D[towers-pack]).

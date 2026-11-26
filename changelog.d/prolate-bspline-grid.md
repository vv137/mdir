### Fixed

- With `kernel = "PROLATE"` and the `"BSPLINE"` window, the automatic grid is refined until the mesh error estimate of the window is at most `tolerance`, so that the force error follows the tolerance instead of staying at 2 to 3 × 10⁻³; the estimate, its bound, and the source of the grid are in the log and the manifest (D[prolate-bspline-grid], #985).

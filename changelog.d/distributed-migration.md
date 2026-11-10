### Added

- **A molecular dynamics loop on domains in the compiler: the layout, halo, and fields carried as one `md_dist.state`, with the halo reused while the skin covers it and migration otherwise** (D399, #38). The in-process executor runs it inside `scf.for` and `scf.if`; there is no control-file key yet.

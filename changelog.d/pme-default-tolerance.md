### Changed

- The default `[pme] tolerance` of the Gaussian kernel is 5 × 10⁻⁵ (was 10⁻⁵), at the same default grid: the relative force error falls to 0.62 to 0.68 of the old default on TIP3P 10k and 20k, JAC, Factor IX, Cellulose and STMV at the same cost (#927). The prolate kernel keeps its default of 10⁻⁵. Trajectories of inputs without `tolerance` or `beta` change. At a cutoff of 4.5 Å `mdir check` now warns of the force jump at the cutoff. D[pme-default-tolerance].

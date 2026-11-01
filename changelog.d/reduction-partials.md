### Performance

- **Less device memory for the global sums**: the partial sums of each reduction take 1024 rows instead of a row a particle, which saves 24 MiB on a box of 196,608 TIP3P atoms and 30 MiB on OPC at the same number of waters, with the same energies in double precision and the same rates (D[reduction-partials], #711).

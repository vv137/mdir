### Changed

- The per-atom input kernel of TOWERS recomputes the cell geometry only when the cell changes: thread 0 compares the lattice words with those of the last geometry and skips its serial chain of divisions and square roots when they are equal. On RTX 4090 the kernel's median went from 9.8 to 5.1 µs on Trp-cage ff19SB+OPC and from 10.5 to 4.9 µs on Trp-cage ff14SB+TIP3P (triclinic cells); rectangular cells are unchanged. The trajectory is unchanged (D[towers-input-geometry], #1050).

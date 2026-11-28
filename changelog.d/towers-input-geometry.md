### Changed

- The per-atom input kernel of TOWERS recomputes the cell geometry only when the cell changes: thread 0 compares the lattice words with those of the last geometry and skips its serial chain of divisions and square roots when they are equal. On RTX 4090, in NVT, the kernel's median went from 10.3 to 4.6 µs on Trp-cage ff19SB+OPC and from 10.5 to 4.3 µs on Trp-cage ff14SB+TIP3P (triclinic cells), and from 3.4 to 3.0 µs on TIP3P 10k. The trajectory is unchanged (D[towers-input-geometry], #1050).

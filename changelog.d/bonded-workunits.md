### Added

- Add opt-in bonded work units for mixed-precision GPU dynamics with
  checked Q18 force and Q32 energy sums and an ordered binary64 virial.
  Double precision and derivative and analysis entries use continuous forces.
  Each global add checks signed overflow using the preceding 64-bit word
  returned by the atomic instruction.
  [D436](docs/decisions/D436.md)

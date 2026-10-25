### Fixed

- A seed of $2^{53}$ or more is now fingerprinted exactly, so `--continue`
  and a Python continuation refuse a changed large seed. Integers below
  $2^{53}$ keep their text, and their checkpoints continue
  (D[fingerprint-integers], #620).

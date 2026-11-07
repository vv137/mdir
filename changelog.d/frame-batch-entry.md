### Changed

- Add the device-resident frame batch entry of #138 PR 1,
  D386, with a C++ caller and private DLPack exercise entry.
  It borrows the caller's stream and device positions/parameters, applies
  mapped and derived tunables without restarting, and writes energy,
  position/strain gradients, parameter rows, auxiliary columns and products
  on the device. LJ/PME and general pair-tail corrections, input validation,
  error recovery and measured numerical residency are included. Existing
  host frame evaluation and the public JAX handler remain unchanged.

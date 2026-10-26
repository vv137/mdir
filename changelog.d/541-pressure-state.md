### Added

- `dyn.load_pressure_state` and `dyn.store_pressure_state`, the state of
  the pressure of a barostat of Trotter type that scales every step (D92),
  in a memref of any memory space, and their expansion by
  `dyn-expand-couplings` into the text the driver writes. The driver does
  not write them yet; no program changes (#541, step 2c; D326).

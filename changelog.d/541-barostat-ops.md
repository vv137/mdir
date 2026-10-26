### Added

- The ops of the barostat in `dyn` (`dyn.group_kinetic`, `dyn.pressure`,
  `dyn.cell_rescaling`, `dyn.scale_cell`, `dyn.load_cell`,
  `dyn.store_cell`, `dyn.scale_positions`, `dyn.scaling_work`), with the
  cell in a memref of any memory space, and their expansion by
  `dyn-expand-couplings` into the text the driver writes. The driver does
  not write them yet; no program changes (#541, step 2b; D326,
  docs/dynamics.md, Section 2.2).

### Changed

- GPU kernels are named after their op and their ordinal in the
  lowering, such as `mdir_run_particle_for_integration3_188`, instead of
  after a line of the program's text. A profile or a sanitizer report
  names kernels the new way; no result changes (D354, #568).

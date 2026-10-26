### Changed

- `mdir emit --stage=module` and Python's `Program.ir` print the module as
  MLIR's printer prints it: values numbered by the printer, and full types
  instead of the builder's aliases. The terms of custom expressions still
  stand after their ops. Programs are unchanged, but the compile cache
  misses once (D352, #568).

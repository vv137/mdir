### Changed

- **`mdir` exports no symbols, and the Python extension only its module
  init**, so that the linker's section garbage collection removes what
  nothing reaches: on a Release build `bin/mdir` takes 148 MB instead of
  205 MB, and the extension `_core` 152 MB instead of 213 MB (#655).

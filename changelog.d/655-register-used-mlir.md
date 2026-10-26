### Changed

- **`mdir` and the Python extension register and link only the MLIR that
  their lowerings use** (`lib/Compiler/Registry.cpp`) instead of every
  upstream dialect, pass, extension, and translation: on a Release build
  `bin/mdir` takes 205 MB instead of 347 MB, and the extension `_core`
  213 MB instead of 361 MB. `MDIR_PIPELINE` may name MDIR's passes and the
  upstream passes of MDIR's pipelines; `mdir-opt` still knows all of MLIR
  (#655).

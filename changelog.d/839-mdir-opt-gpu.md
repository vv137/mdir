### Fixed

- **`mdir-opt` serializes GPU modules** (#839): it registers the translations to LLVM IR, so that the pipeline that `mdir emit --stage=pipeline` prints for a GPU run replays through `mdir-opt` to the end, as [docs/debugging.md](docs/debugging.md#failures-of-the-passes) describes; every GPU module failed in `mdir-gpu-module-to-binary` before.

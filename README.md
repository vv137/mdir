# MDIR

MDIR is an MLIR-based compiler stack for general-purpose molecular dynamics.
It is at an early stage: the design is written down and the first dialect
parses, verifies, and prints.

## Documents

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The design |
| [docs/ops-m0.md](docs/ops-m0.md) | Types and ops for the first milestone |
| [docs/decisions.md](docs/decisions.md) | Decisions and their status |
| [docs/prior-art.md](docs/prior-art.md) | Earlier work and what is taken from it |

## Requirements

- A C++17 compiler
- CMake 3.20 or later
- Ninja
- LLVM and MLIR 23.1.2, built with `scripts/build-llvm.sh`
- `lit`, for the tests (`pip install lit`)

## Building LLVM

```sh
LLVM_ROOT=$HOME/opt/llvm scripts/build-llvm.sh
```

The script clones the release, builds it with MLIR and the X86, NVPTX, and
AMDGPU targets, and installs it to `$LLVM_ROOT/23.1.2`.

## Building MDIR

```sh
LLVM_PREFIX=$HOME/opt/llvm/23.1.2

cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir \
    -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(command -v lit)

cmake --build build
```

## Testing

```sh
cmake --build build --target check-mdir
```

The tests need `mlir-opt`, `mlir-runner`, and `FileCheck` from the LLVM
installation. Some tests run generated code and compare its results with
reference values.

## Tools

`mdir-opt` parses, verifies, transforms, and prints MDIR modules.

```sh
build/bin/mdir-opt test/Dialect/MD/ops.mlir
```

| Pass | Effect |
|---|---|
| `--md-check-exchange` | Proves the exchange contracts of pair kernels. |
| `--md-expand-truncation` | Expands truncation attributes into kernels. |
| `--md-differentiate` | Replaces `md.evaluate` with calls to generated derivative functions. |

```sh
build/bin/mdir-opt input.mlir --md-check-exchange --md-differentiate
```

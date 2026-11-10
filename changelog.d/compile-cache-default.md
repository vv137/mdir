### Changed

- **The compile cache is on by default** for `mdir run` and Python, in `${XDG_CACHE_HOME:-~/.cache}/mdir/compile` (else `/var/tmp/mdir-<uid>/compile`) on a local file system, private to the user; its keys now name the build of MDIR and LLVM, the LLVM options of the command line, the scheduler, and the `ptxas` binary, so a rebuild or another toolchain starts cold; `MDIR_COMPILE_CACHE_MAX_MB=0` turns it off, and the log and manifest of `mdir run` say where it is ([docs/compile-cache.md](docs/compile-cache.md#where-the-cache-is), D[compile-cache-default], #826).

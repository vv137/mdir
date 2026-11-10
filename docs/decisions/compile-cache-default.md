# D[compile-cache-default]

Recorded at its merge.

**The compile cache is on by default, for `mdir run` and Python alike, in a private directory on a local file system, and every key of it begins with a salt that names the build of MDIR and LLVM (the ELF build ID of the image that holds the compiler), the LLVM options given on the command line, and the scheduler in force, so that an entry is used only when nothing that changes the compiled result differs.** Issue #826, from option 1 of #810; the maintainer's direction of 2026-11-10: whenever anything that changes the compiled result differs, the cache must not be used.

*The salt.* Computed once per process, and written in front of the key of every host object (D212), PTX and cubin (D214), and of the code a `Program` keeps (D236), through `describeMachine`:

- `image`: the ELF build ID (`NT_GNU_BUILD_ID`) of the image that holds `lib/Compiler`: `mdir`, or the Python extension. LLVM and MLIR are linked statically into both, so the ID covers every MDIR and LLVM effect that runs after a key is taken: `optimizeHostModule`, the target machine's options, how the PTX is serialized and ptxas called, a patched LLVM or one at the same version from another source, the dialect translations registered, the runtime templates. Every image is linked with `--build-id=sha1`. Without a note, the BLAKE3 of the image's file, once per process.
- `mdir`, `llvm`: the version, commit, and dirty flag of MDIR, the LLVM version and revision. Readable only; nothing relies on them.
- `cl-opts`: every LLVM option of the command line (`mdir` parses all registered options), by name with its value, sorted. A Python process has none.
- `test-salt`: `MDIR_COMPILE_CACHE_SALT_OVERRIDE`, for tests only: it stands for another build.

The scheduler before register allocation is not in the static part: the key of a host object and of a PTX holds `sched <name>`, the value of `-pre-RA-sched` when the key is taken, read from the option itself. The three format tags become 2, so entries of before are not read.

*The toolchain.* The key of a cubin holds the BLAKE3 of the ptxas binary beside the text of its `--version`. The hash of a file (libdevice, ptxas, a linked library) is kept for the process by path, device, inode, size, mtime, and ctime, so a file replaced with the same size and mtime is hashed again. An LLVM built with `MLIR_NVVM_EMBED_LIBDEVICE` links its embedded libdevice, which the key then hashes instead of the file (no LLVM that MDIR builds embeds it; this path is not exercised here).

*Where the cache is.* In order:

1. `MDIR_COMPILE_CACHE=off` turns it off.
2. `MDIR_COMPILE_CACHE_MAX_MB=0` turns it off; a value that is not a whole number is reported and the bound stays 2048 MiB.
3. `MDIR_COMPILE_CACHE_DIR` names the directory, as before. One that group or others may write is used, with a warning.
4. Otherwise `${XDG_CACHE_HOME:-$HOME/.cache}/mdir/compile`, if that file system is local.
5. Otherwise `/var/tmp/mdir-<uid>/compile`, if local.
6. Otherwise none, and the log says why.

A default directory and its parent are made with mode 0700. Before any entry is read or written, a default directory whose owner, or whose parent's owner, is not the effective user, or that group or others may write, is refused with a warning, and the cache is off; a parent of the user's that group or others may write (`~/.cache/mdir` under a umask of 002) loses that permission first, since whoever may write it could put another directory in the place of the default once it is checked: an entry's BLAKE3 is a checksum, not an authentication, and a host object is native code that runs. `mdir.clear_compile_cache()` without a directory clears the directory that a compile would use (`MDIR_COMPILE_CACHE_DIR`, else the default), and `MDIR_COMPILE_CACHE=off` still does not hide it.

*Writes.* The lock of the file `size` (D234) is taken with a timeout of 10 s instead of waiting without bound; a store that does not get it lists the directory as a store without locks does. No `fsync` before the rename: a torn entry fails its length and hash and is a miss, and a cold compile stores about 800 entries.

*The record.* `mdir run` writes `MDIR: compile cache:` in its log (on, where, from which rule; or off and why) with the counts of the run's GPU modules (PTX and cubins compiled, read, and stored), and the same as `compile_cache` in the manifest.

*What changes for users.* A second run of a program, a continuation, and a test of the suite take the PTX and cubins of the first; a rebuild of MDIR, an LLVM option, or another ptxas starts cold. `mdir` and the Python extension are two images with two build IDs, so they no longer share GPU entries. A warm cache also shortens the compilation of the candidates of the skin search (D281), whose time counts in its budget (`lib/Compiler/Skins.cpp`), so a warm search may afford more candidates.

*Left for #99.* `mdir run` still generates its host code through MLIR's `ExecutionEngine`, without the cache; moving it onto the owned engine, one owner of the scheduler option across threads (a Python thread's reset can land during another's code generation), and the D236 key for `mdir run` (built from the pipeline after `MDIR_PIPELINE`) belong there. Until then the default affects the GPU entries of `mdir run`, the host objects of the skin search, and everything of Python.

*On the machines of the project.* The helpers of the cluster set `MDIR_COMPILE_CACHE_DIR` on the shared store, since the local disks there are not for this work; a job run under another account gets its own per-user default.

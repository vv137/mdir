# The compile cache of host objects (D[compile-cache])

Issue #142. A Python simulation compiles its program in three stages: the
MLIR pipeline lowers it to an LLVM module (with the PTX of its kernels on a
GPU), LLVM generates the host object of that module, and the CUDA driver
compiles the PTX when the module is loaded. This item caches the second
stage, the relocatable host object, on disk, keyed by the content of the
module it was generated from. The other two stages are not cached here:
the pipeline runs on every compile, and the driver keeps its own cache of
compiled PTX. A per-module cache of the PTX belongs with the parallel
serialization of the GPU modules (#148), and `mdir run`, which still
compiles with MLIR's `ExecutionEngine`, takes the cache when it moves onto
the owned engine (#99).

## Use

The cache is off unless a directory is given. It is controlled by the
environment only; no control-file key and no Python argument changes.

| Variable | Effect |
|---|---|
| `MDIR_COMPILE_CACHE_DIR=<dir>` | Enables the cache. Entries go in `<dir>/host/`, which is made when the first entry is written. |
| `MDIR_COMPILE_CACHE=off` | Disables the cache even when a directory is set. |
| `MDIR_COMPILE_CACHE_MAX_MB=<n>` | Bounds the directory to $n$ MiB (2048 by default). After an entry is written, the entries used least recently are removed until the directory is within the bound; 0 keeps none. |

A directory may be shared by processes that run at the same time, and by
builds of MDIR (see the key below). Removing it, or any entry in it, is
always safe.

`Simulation.compile_stats` reports what compiling the simulation's program
cost and what the cache saved, as a dict:

| Key | Meaning |
|---|---|
| `programs` | Programs compiled (1 for a simulation) |
| `pipeline_seconds` | The MLIR pipeline of the simulation's program |
| `engine_seconds` | From the creation of the JIT engine to its entry: host code generation or the cache, linking, and on a GPU the loading of the kernels |
| `host_compiled`, `host_compile_seconds` | Host objects that LLVM generated, and the time that took |
| `cache_hits`, `cache_saved_seconds` | Host objects read from the cache, and the time their generation took when they were stored |
| `cache_rejected` | Entries found under the key's name but rejected (see Entries) |
| `cache_stored`, `cache_unstored` | Entries written, and entries that could not be written |
| `cache_lookup_seconds` | The time of keys and reads, hit or miss |

`examples/ala3/run.py` prints the compile times of each stage from them.

## The key

The host object is a function of the LLVM module and of the code generator
alone, so its key holds these and nothing of the MDIR build:

- a BLAKE3 hash of the module's bitcode, taken before code generation
  (which changes the module);
- the LLVM version, the target triple, the CPU name and its features, the
  level of code generation (D197), the code and relocation models, the
  data layout, the target options that change generated code, and the
  pre-RA scheduler that the simulation selects (D150);
- a format tag.

The PTX of the kernels is a global of the module, so a change of device
code changes the key too, and everything the program's constants depend on
(the cell at the start, for example) is in the module. A rebuild of MDIR
that leaves the generated module unchanged therefore hits, and one that
changes any part of the generated code misses. An object is not portable
between machines with different CPU features: a different CPU is a
different key.

This follows the design of #142 for the stages after MLIR. The MLIR
pipeline depends on MDIR's passes, so a cache of its output would need the
build identity in its key, and would miss after every rebuild; it is not
part of this item.

## Entries

An entry is a file named by a hash of its key, `<hash>.o`, holding a magic
tag, the full key, the length of the object, a BLAKE3 hash of the object,
the time its generation took, and the object. A read checks the tag, the
key, the length, and the hash, and asks LLVM to parse the object; any
mismatch rejects the entry as a miss, and the object is generated again
and replaces it. A hash collision between two keys is therefore a miss,
not a wrong object.

Entries are written to a unique temporary file in the same directory and
renamed into place, so a reader sees the old entry, the new one, or none,
never a part; processes that store the same key at once leave one entry.
A read copies the file rather than mapping it, since another process may
replace it. A hit touches the entry, which the eviction of the least
recently used reads. Temporary files older than an hour, left by a process
that died, are removed during eviction. A directory that cannot be written
leaves the run as it would be without a cache (`cache_unstored`).

The cache is the `llvm::ObjectCache` that ORC's `TMOwningSimpleCompiler`
takes, as in LLVM's `LLJITWithObjectCache` example. Only the program's
module is cached: the small modules that ORC makes for initialization name
their functions uniquely and would never hit.

## Ownership

A hit only replaces code generation. Each engine links its own copy of the
object into memory of its own and validates it before any frame is
registered, exactly as it does a generated object, so the boundary of D199
([JIT ownership](jit-invariants.md)) is unchanged. No linked code is shared
between simulations or processes.

## The test suite

`test/mdir_lit.py` gives every test of a suite the directory
`<build>/test/compile-cache`, so that the suites of different build trees
share nothing; `-Dcompile_cache=off` (or `MDIR_COMPILE_CACHE=off` in the
environment of lit) runs a suite without it. A test of the cache itself
sets its own directory or `MDIR_COMPILE_CACHE=off` in its RUN lines. A full
suite leaves 86 entries, 139 MB.

## Validation

The dipeptide in water with PME, SHAKE, and SETTLE, deterministic, 20
steps (`compile-cache-{mixed,double}{,-gpu}.test`,
`Inputs/compile_cache.py`); each scenario is a process of its own on one
directory:

- **Bit identity.** The positions and velocities after 20 steps with a
  cold and a warm cache equal those without a cache bit for bit, on the
  CPU and a GPU, in mixed and double precision; on the CPU also after a
  rejected entry is rebuilt. The Python ala3 example (GPU, mixed,
  deterministic, `--steps-scale 0.01`) writes the same energy files, DCD,
  and minimized positions, byte for byte, with a cold and a warm cache.
- **Misses.** Another module, another CPU and features, and another
  scheduler change the key (`compile-cache.test`); a run that is not
  deterministic and one in the other precision miss and store their own
  entries (`compile-cache-mixed.test`).
- **A rebuild hits.** A rebuild of MDIR after changing a message in
  `lib/Compiler/Compile.cpp` (the Python module's hash changed) hit the
  entries of the build before on the CPU and a GPU (mixed), and gave the
  state bit for bit; the engine took 0.09 s instead of 1.70 s on the CPU
  and 1.03 s instead of 3.09 s on a GPU.
- **Damaged entries.** An entry cut in half, an object with one byte
  flipped, an entry of another format, one under another key, and an
  intact entry that holds no object are rejected; the first two are
  generated again, stored, and hit by the next process
  (`compile-cache.test`, `compile-cache-{mixed,double}.test`).
- **The bound.** The entry used least recently goes first; a bound of
  0 MiB keeps no entry; a stale temporary file is removed and a recent one
  kept.
- **Concurrent processes.** Four processes started at once on an empty
  directory each generate or read the object, none is rejected, one entry
  remains, the states agree with no cache, and the next process hits.
  Every suite also runs up to sixteen GPU tests and the CPU tests on one
  directory at once.
- **Ownership.** `jit-memory.test` runs the D199 checks without a cache,
  with a cold one, and with a warm one, and the Python lifetime tests
  (`python-simulation-lifetime*`, CPU and GPU, three import orders) pass
  with a cold and a warm suite directory.

## Cost and gain

On one RTX 3090 (300 W) of a shared 128-core host, the dipeptide (one
program, mixed): reading the key and the entry takes 0.02 s on the CPU and
0.12 s on a GPU, whose module carries the PTX. An entry is 0.2 MB on the
CPU and 4.6 MB on a GPU for the dipeptide, and 2.6 to 10.5 MB for the
stages of the ala3 example on a GPU.

The Python ala3 example, GPU, mixed, `--steps-scale 0.01`, the driver's
cache of compiled PTX warm; per stage, the time from `mdir.compile` to a
simulation, and in parentheses the engine's part:

| Stage | Cache off | Cold | Warm | Warm, deterministic |
|---|---|---|---|---|
| 1-min | 13.8 s (2.1) | 13.0 s (1.8) | 12.2 s (0.8) | 12.2 s (0.9) |
| 2-nvt | 33.1 s (4.9) | 33.5 s (5.1) | 29.1 s (1.0) | 29.0 s (1.1) |
| 3-npt | 53.8 s (8.3) | 52.1 s (7.7) | 46.4 s (1.9) | 47.3 s (2.0) |
| 4-md | 51.0 s (7.0) | 50.2 s (7.8) | 50.9 s (7.2), a miss | 46.5 s (1.8) |
| Whole run | 153.8 s | 150.8 s | 140.6 s | 137.3 s |

A hit saves 1 to 6 s per stage of host code generation; the MLIR pipeline,
in `mdir.compile` (which lowers a program of its own, #151) and in the
simulation, is most of what remains. Without the deterministic mode the
production stage missed: it starts from the cell that the NPT stage
reached, which differs from run to run, and its program's constants depend
on it.

The full suite on that GPU at the default `gpu_workers` (16): 262 s with
the cache off, 250 s with a cold cache, and 243 s with a warm one. The
slowest test, `python-reporters-gpu.test`, took 207, 198, and 188 s, and
the GPU lifetime tests about 163, 128, and 124 s. Many programs of a
suite are compiled more than once, by one test or by several, so a cold
cache hits as well. The suite's wall time is set by its slowest tests, in
which the MLIR pipeline dominates.

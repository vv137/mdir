# The compile cache (D212, D214)

Issue #142. A Python simulation compiles its program in three stages: the
MLIR pipeline lowers it to an LLVM module (with the PTX of its kernels on a
GPU), LLVM generates the host object of that module, and the CUDA driver
compiles the PTX when the module is loaded. This item caches the second
stage, the relocatable host object, on disk, keyed by the content of the
module it was generated from. The other two stages are not cached here:
the pipeline runs on every compile, and the driver keeps its own cache of
compiled PTX. `mdir run`, which still compiles with MLIR's
`ExecutionEngine`, takes the cache of host objects when it moves onto the
owned engine (#99).

D214 (#148) serializes the GPU modules in parallel, keeps
the PTX and the cubin of each module in the same cache, and loads cubins
compiled for the device instead of PTX; see
[The GPU modules](#the-gpu-modules-dgpu-module-compile) below.

## Use

The cache is off unless a directory is given. It is controlled by the
environment only; no control-file key and no Python argument changes.

| Variable | Effect |
|---|---|
| `MDIR_COMPILE_CACHE_DIR=<dir>` | Enables the cache. Host objects go in `<dir>/host/` and the PTX and cubins of GPU modules in `<dir>/gpu/`, each made when its first entry is written. |
| `MDIR_COMPILE_CACHE=off` | Disables the cache even when a directory is set. |
| `MDIR_COMPILE_CACHE_MAX_MB=<n>` | Bounds the directory, host and GPU entries together, to $n$ MiB (2048 by default). After entries are written, the entries used least recently are removed until the directory is within the bound; 0 keeps none. |

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

A hit needs the same module from the same inputs, so the text that the
builder generates must be a function of the model alone, not of what the
process built before. The builder keeps no count or other state across
builds: its names are numbered per build (#152), and a model built after
another in one process has the text and the lowered text of a model built
first (`python-build-history.test`, on the CPU and a GPU).

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
suite left 86 entries, 139 MB, before the GPU entries; with them, it
leaves 114 MB of host objects and 684 MB of GPU entries
(D214).

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

| Stage | Cache off | Cold | Warm | Cold, deterministic | Warm, deterministic |
|---|---|---|---|---|---|
| 1-min | 13.8 s (2.1) | 13.0 s (1.8) | 12.2 s (0.8) | 13.4 s (1.9) | 12.2 s (0.9) |
| 2-nvt | 33.1 s (4.9) | 33.5 s (5.1) | 29.1 s (1.0) | 33.6 s (4.8) | 29.0 s (1.1) |
| 3-npt | 53.8 s (8.3) | 52.1 s (7.7) | 46.4 s (1.9) | 54.0 s (8.1) | 47.3 s (2.0) |
| 4-md | 51.0 s (7.0) | 50.2 s (7.8) | 50.9 s (7.2), a miss | 51.6 s (7.0) | 46.5 s (1.8) |
| Whole run | 153.8 s | 150.8 s | 140.6 s | 154.7 s | 137.3 s |

A hit saves 1 to 6 s per stage of host code generation; the MLIR pipeline,
in `mdir.compile` (which lowers a program of its own, #151) and in the
simulation, is most of what remains. Without the deterministic mode the
production stage missed: it starts from the cell that the NPT stage
reached, which differs from run to run, and its program's constants depend
on it.

The full suite on that GPU at the default `gpu_workers` (16): 262 s with
the cache off, 250 s with a cold cache, and 243 s with a warm one, single
runs on a shared host whose load varied between them (D208 saw 262 and
340 s from load alone), so the totals show no gain beyond noise. The
slowest test, `python-reporters-gpu.test`, took 207, 198, and 188 s, and
the GPU lifetime tests about 163, 128, and 124 s. Many programs of a
suite are compiled more than once, by one test or by several, so a cold
cache hits as well. The suite's wall time is set by its slowest tests, in
which the MLIR pipeline dominates.

## The GPU modules (D214)

Issue #148. A GPU program becomes many small GPU modules, one kernel each:
the production stage of the ala3 example has 400 under `mdir run` and 624
in the program of segments of a Python simulation. Upstream
`gpu-module-to-binary` serialized them one after another, each through
LLVM's NVPTX back end to PTX, and the driver compiled each PTX to machine
code when the program was loaded.

### Serialization in parallel

The pipeline of a GPU program ends with `mdir-gpu-lower-to-nvvm`. This is
upstream `gpu-lower-to-nvvm-pipeline`, pass for pass and with the same
options, except that `mdir-gpu-module-to-binary` serializes the modules.
That pass:

- serializes the modules on the threads of the context: the process's
  shared pool (D211), at most `MDIR_COMPILE_THREADS`;
- translates each module to LLVM IR in an LLVM context of its own, and
  reads the module without changing it;
- builds the symbol table before the threads start (upstream builds it
  lazily);
- replaces the modules with their binaries on one thread, in the order of
  the modules.

The result does not depend on the number of threads. The objects do not
carry upstream's `LLVMIRToISATimeInMs` and `ISAToBinaryTimeInMs`
properties. Those differed from run to run, so the lowered module is now
the same in every run. `mdir run` lowers on the same shared pool.

### Cubins for the device

The kernels are compiled for the architecture of the device that will run
them, and the driver loads them without compiling them. `ptxas` compiles
the PTX of each module, in parallel, with upstream's arguments,
`-arch sm_XY --opt-level <O>` (`O` is 2). At most 32 `ptxas` run at once:
on a 128-core host, more took the same wall time with four times the
system time. The `ptxas` is that of the toolkit whose libdevice the
kernels link (`CUDA_ROOT`, `CUDA_HOME`, `CUDA_PATH`, else the toolkit of
the build), else the one on `PATH`.

Two environment variables choose the binaries:

| Variable | Effect |
|---|---|
| `MDIR_GPU_BINARY=auto` | The default. Cubins for the device's architecture. Where no `ptxas` is found, or it fails for the architecture, the kernels are PTX for that architecture, and a message says so once. Where the architecture is not known, they are PTX for `sm_75`, as before this item. |
| `MDIR_GPU_BINARY=cubin` | Cubins. An unknown architecture, a missing `ptxas`, or a failure of `ptxas` is an error. |
| `MDIR_GPU_BINARY=ptx` | PTX, which the driver compiles at load: for `MDIR_GPU_ARCH` if it is set, else for `sm_75`. For debugging the driver's compilation, or for programs that run on other GPUs. |
| `MDIR_GPU_ARCH=sm_XY` | The architecture to compile for, instead of the device's. For compiling on one GPU for another. LLVM's NVPTX back end must know it. |

The architecture of the device comes from NVML
(`nvmlDeviceGetCudaComputeCapability`), not from the CUDA driver. Lowering
a program, `mdir.compile` included, therefore creates no CUDA state, and a
process may fork after it compiles; `fork-after-compile-gpu.test` runs the
program in a forked child. The device is chosen as the runtime chooses it:

- The device index is `Execution.device`, or `MDRT_DEVICE`, which the
  runtime takes over it. `mdir run` takes the first visible device.
- That index selects an entry of `CUDA_VISIBLE_DEVICES`, as CUDA does.
- An entry `GPU-<uuid>` names its device.
- An index entry is CUDA's index. CUDA orders the devices by their PCI
  bus under `CUDA_DEVICE_ORDER=PCI_BUS_ID`, as NVML does, and fastest
  first otherwise. Under the default order, the architecture is taken only
  when every device has the same one.

Where NVML is missing or cannot tell, the architecture is unknown, unless
`MDIR_GPU_ARCH` gives it. A program compiled for one architecture and
loaded on a device of another fails at load with a message that says so.

`mdir emit --stage=pipeline` shows what was chosen, for example
`mdir-gpu-lower-to-nvvm{cubin-chip=sm_86 cubin-format=bin binary=auto}`.

### The cache of the GPU modules

With `MDIR_COMPILE_CACHE_DIR` set, the pass keeps the PTX and the cubin of
each module in `<dir>/gpu/`, as entries of the format of the host objects
(below: full key, length, BLAKE3 hash, time of generation, data; atomic
writes; reads that copy). `MDIR_COMPILE_CACHE_MAX_MB` bounds the host and
GPU entries together, least recently used first.

The key of the PTX of a module (`<hash>.ptx`) holds:

- a BLAKE3 hash of the module's IR, printed in the generic form with
  nothing elided and without locations, which do not reach the PTX;
- its target attribute: the triple, `sm_XY`, the PTX features, and `O`;
- the value of `MDIR_GPU_BINARY`;
- a BLAKE3 hash of the libdevice it links, and of any other library its
  target links;
- the LLVM version and a format tag.

The MDIR build is not part of the key. At this point a module holds only
upstream ops of the LLVM and NVVM dialects, so a rebuild of MDIR that
generates the same kernels hits. A module that refers to a
`dense_resource` blob is not cached, since its printed IR does not hold
the blob's contents.

The key of a cubin (`<hash>.cubin`) holds:

- a BLAKE3 hash of its PTX;
- the text of `ptxas --version`;
- the arguments of `ptxas`;
- the value of `MDIR_GPU_BINARY`;
- a format tag.

A new toolkit therefore compiles the cached PTX again, and a new LLVM
generates new PTX.

A PTX entry must hold `.version` and `.target`, and a cubin entry must be
an ELF file; any other entry is rejected and generated again. Identical
modules within one program, or in the program that `mdir.compile` lowers
before the simulation's, share one entry. `mdir run` uses these entries too
when the directory is set, since the pass is part of its pipeline; its host
objects are not cached yet (#99).

`Simulation.compile_stats` gains these keys:

| Key | Meaning |
|---|---|
| `gpu_modules` | GPU modules serialized |
| `gpu_serialize_seconds` | Wall time of their serialization |
| `gpu_ptx_compiled`, `gpu_ptx_hits` | PTX generated by LLVM, and read from the cache |
| `gpu_cubin_compiled`, `gpu_cubin_hits` | Cubins generated by `ptxas`, and read from the cache |
| `gpu_compile_seconds` | Time of the PTX and cubins generated, summed over the modules |
| `gpu_cache_saved_seconds` | Time the hits took when they were stored, summed |
| `gpu_cache_lookup_seconds` | Time of keys and reads, summed over the modules |
| `gpu_cache_rejected`, `gpu_cache_stored`, `gpu_cache_unstored` | GPU entries rejected, written, and that could not be written |

### Validation

- **Byte identity.** The dipeptide in water with PME, SHAKE, and SETTLE,
  NPT, deterministic (253 GPU modules), was lowered through `mdir emit
  --stage=lowered`.
  - Before cubins, the result equals main's byte for byte at 1, 8, and 128
    threads, with the cache off, cold, and warm. Main's timing properties
    are stripped for the comparison.
  - With cubins, the result is the same at 1, 8, and 128 threads.
  - `gpu-module-compile-gpu.test` checks 1 and 8 threads, cold and warm.
- **Bit identity.** The same system through Python, deterministic, 20
  steps. The state equals main's bit for bit on the CPU and on a GPU, in
  mixed and double precision. On the GPU this holds with the cache off,
  cold, and warm, and with the driver's cache of compiled PTX disabled or
  warm. It also holds for kernels loaded as PTX for `sm_86` and as cubins
  for `sm_86` (`ptxas` 13.4 against a driver of CUDA 13.2).
- **The ala3 example.** The Python example, deterministic, at a hundredth
  of its steps, writes main's energy files, DCD, and minimized positions
  byte for byte in every configuration below.
- **Fallbacks and switches.** A run without `ptxas` gives the energies of
  the run with cubins (`gpu-module-compile-gpu.test`). The same test checks
  the errors of `MDIR_GPU_BINARY=cubin` and of values that are not known.
- **Damaged entries and fork.** A flipped PTX and cubin entry are rejected
  and rewritten (`compile-cache-mixed-gpu.test`). A child forked after
  `mdir.compile` runs the program on the GPU
  (`fork-after-compile-gpu.test`).

### Cost and gain

Measured on an RTX 3090 (300 W) of a shared 128-core host, with
`MDIR_COMPILE_THREADS` unset (128).

The dipeptide through Python, mixed, deterministic. Each cell is the wall
time of `Simulation(program)` (MLIR pipeline / JIT engine). "Driver cache
off" sets `CUDA_CACHE_DISABLE=1`.

| | Driver cache warm | Driver cache off |
|---|---|---|
| main | 15.7 s (12.2 / 3.5) | 21.0 s (12.6 / 8.3) |
| Parallel serialization, PTX | 8.0 s (4.6 / 3.4) | 13.5 s (4.6 / 8.8) |
| Cubins, compile cache off | 7.6 s (4.8 / 2.8) | 7.6 s (4.8 / 2.8) |
| Cubins, compile cache warm | | 4.7 s (4.2 / 0.5) |

The serialization of the 249 modules of the simulation's program takes:

- 0.7 to 0.9 s wall with cubins on 128 threads;
- 0.13 s from a warm cache.

Generating the PTX and cubins takes about 10 s of PTX and 26 s of
`ptxas`, summed over the modules. `ptxas` alone costs about 9 s of CPU
time serially for the 253 modules of `mdir run`'s program, where the
driver compiled the PTX in about 5.6 s. Cubins without a warm cache
therefore trade CPU time for wall time.

The ala3 example, the whole run (four stages, each `mdir.compile` and a
simulation):

| | Wall | CPU (user + system, with children) |
|---|---|---|
| main, driver cache warm | 147.1 s | 216 s |
| main, driver cache off | 188.6 s | 259 s |
| Parallel serialization, PTX | 82.5 s | 267 s |
| Parallel serialization, PTX cache warm | 63.7 s | 149 s |
| Cubins, compile cache off | 81.5 s | 372 s |
| Cubins, compile cache cold, driver cache off | 84.5 s | 350 s |
| Cubins, compile cache warm, driver cache off | 59.7 s | 136 s |

With a warm cache, the JIT part of a stage takes 0.3 to 0.5 s instead of
1.7 to 6.2 s, whatever the driver's cache. What remains is the MLIR passes
before the serialization: 2 to 12 s per simulation, and as much again in
`mdir.compile` (#151). The cache of the example holds 21 MB of host
objects and 78 MB of GPU entries.

The full suite on one RTX 3090 at the default `gpu_workers` (16):

- 225 s with an empty suite cache, against 250 s for D212's cold cache;
- 150 s with a warm one, against 243 s.

Both are single runs on a shared host. A full suite leaves 114 MB of host
objects and 684 MB of GPU entries.

### Why there are so many modules

The production stage of ala3 under `mdir run` has 400 GPU modules of one
kernel each. Of them, 223 are `tuple_for`, 93 `particle_for`, 33
`pair_for`, 13 `permute`, and the rest are kernels of the runtime's
templates. With the names of the kernels replaced, 292 of them are
distinct.

Each force evaluation that `md-inline` inlines has kernels of its own:

- at the start;
- in a step with and without energies;
- at the trial positions of the barostat.

The cache already makes a duplicate cost one read. Deduplicating or
merging the modules, to save loads and bytes of the host object, is #167.

### Ownership

A PTX or a cubin is data: a constant of the host module that the driver
loads into memory of its own. The cache changes only how those bytes are
made, before the host object is generated. The boundary of D199 is
unchanged ([jit-invariants.md](jit-invariants.md)).

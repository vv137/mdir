# The compile cache (D212, D214, D217, D227, D234, D[program-reuse])

Issue #142. A Python simulation compiles its program in three stages: the
MLIR pipeline lowers it to an LLVM module (with the PTX of its kernels on a
GPU), LLVM generates the host object of that module, and the CUDA driver
compiles the PTX when the module is loaded. This item caches the second
stage, the relocatable host object, on disk, keyed by the content of the
module it was generated from. The other two stages are not cached here:
the pipeline runs for every program that is compiled, and the driver
keeps its own cache of compiled PTX. Within a process, a `Program` keeps
the code of its simulations in memory, so that only the first of them
compiles at all
([Reuse within a process](#reuse-within-a-process)). `mdir run`, which still compiles with MLIR's
`ExecutionEngine`, takes the cache of host objects when it moves onto the
owned engine (#99).

D214 (#148) serializes the GPU modules in parallel, keeps
the PTX and the cubin of each module in the same cache, and loads cubins
compiled for the device instead of PTX; see
[The GPU modules](#the-gpu-modules-dgpu-module-compile) below.
D217 (#163) clears the cache and bypasses it for one
compile from Python; see [Controls from Python](#controls-from-python).
D227 (#162) takes the values that depend on the
starting cell out of the program's text, so that a stage that starts from
an equilibrated cell hits the entry of another; see
[Values of the start](#values-of-the-start-d227).
D234 (#196) keeps the total of the entries in a file, so
that a store does not list the directory; see
[The bound](#the-bound).

## Use

The cache is off unless a directory is given. The environment enables
it; no control-file key changes. From Python, `mdir.clear_compile_cache()`
empties it and `cache=False` bypasses it for one compile
([Controls from Python](#controls-from-python)).

| Variable | Effect |
|---|---|
| `MDIR_COMPILE_CACHE_DIR=<dir>` | Enables the cache. Host objects go in `<dir>/host/` and the PTX and cubins of GPU modules in `<dir>/gpu/`, each made when its first entry is written. |
| `MDIR_COMPILE_CACHE=off` | Disables the cache even when a directory is set. |
| `MDIR_COMPILE_CACHE_MAX_MB=<n>` | Bounds the directory, host and GPU entries together, to $n$ MiB (2048 by default). When entries written take the total over the bound, the entries used least recently are removed until the directory is within it; 0 keeps none ([The bound](#the-bound)). |

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
| `cache_bypassed` | Programs compiled with `cache=False`: 1 or 0 for a simulation. With it, every `cache_*` and `gpu_cache_*` count is 0. |
| `program_reused` | 1 if the simulation took the code that an earlier simulation of its `Program` left in memory, else 0 ([Reuse within a process](#reuse-within-a-process)). With it, `pipeline_seconds`, `host_compiled`, and every `cache_*` and `gpu_*` time and count are 0: nothing was lowered, generated, or looked up. |
| `reuse_saved_seconds` | With `program_reused`, the `pipeline_seconds` and the time of the host object (its generation, or its lookup on disk) of the simulation that left the code. |

`examples/ala3/run.py` prints the compile times of each stage from them.

## Controls from Python

D217, issue #163.

```python
cleared = mdir.clear_compile_cache()          # or clear_compile_cache(directory)
program = mdir.compile(system, state, integrator, ensemble, execution,
                       schedule, cache=False)
sim = mdir.Simulation(program)                # takes the program's choice
sim = mdir.Simulation(program, cache=False)   # or overrides it
sim.compile_stats["cache_bypassed"]           # 1
```

`cache=False` also bypasses the code that the `Program` keeps in memory
([Reuse within a process](#reuse-within-a-process)): the
simulation lowers and generates everything, and leaves nothing.

`mdir.clear_compile_cache(directory=None)` removes the entries of the
cache in `directory`, else in `MDIR_COMPILE_CACHE_DIR`.
`MDIR_COMPILE_CACHE=off` does not hide that directory from it.

- It removes the host objects (`host/*.o`) and the GPU entries
  (`gpu/*.ptx`, `gpu/*.cubin`) whose magic tag is this format's.
- It leaves the entries of another format, files of other names, and the
  directories themselves. It takes the bytes it removed off the total in
  the file `size` ([The bound](#the-bound)).
- It leaves the temporary files of writers under way, so that their
  renames succeed. Temporary files older than an hour are removed, as
  eviction removes them.

It returns what it removed:

| Key | Meaning |
|---|---|
| `directory` | The directory cleared, or `None` when none was given or set |
| `host_entries`, `gpu_entries` | Host objects and GPU entries removed |
| `bytes` | Their bytes |

MDIR keeps no entries in memory between compiles: each JIT engine and each
serialization of GPU modules reads the directory anew. A clear therefore
takes effect at the next compile, in this process and in others.

Other processes may write and read the directory during a clear. A writer
renames a complete entry into place and a reader copies the file, so a
clear can make a reader miss or remove an entry just written, but never
leaves a part of an entry. A later compile stores the entry again.

`cache=False` bypasses the cache for one compile, whatever the
environment says: no entry is read, written, or touched, no directory is
made, and nothing is evicted.

- `mdir.compile(..., cache=False)` is keyword only. It applies to the
  lowering of `Program.lowered_ir`, which `mdir.compile` no longer runs
  itself but which happens on the first read of `lowered_ir`
  (D224); on a GPU that lowering reads and writes the PTX and
  cubins of its modules. The program records the choice, the default of
  its simulations.
- `Simulation(program, cache=None)` is keyword only. `None` takes the
  program's choice, and `True` or `False` overrides it for the
  simulation's compile: its host object and the entries of its GPU
  modules.
- On a GPU the pipeline of a bypassed program shows `cache=false` among
  the options of `mdir-gpu-lower-to-nvvm` (`Program.pipeline`).

## On clusters

Where the directory goes:

- Point `MDIR_COMPILE_CACHE_DIR` at a scratch file system
  (`$SCRATCH/mdir-cache`) or at storage local to the node
  (`$TMPDIR/mdir-cache`). Do not put it in a home directory with a quota:
  a full suite leaves about 800 MB, and the bound
  (`MDIR_COMPILE_CACHE_MAX_MB`, 2048 MiB by default) applies to each
  directory.
- A purge of scratch, or a node-local directory that goes with its job,
  only costs a compile. Removing the directory, or any entry in it, is
  always safe.

Many ranks or jobs at once:

- Each lookup reads an entry, each hit touches it, and each store renames
  a file. A compile that stored entries then adds their bytes to the file
  `size` under a lock; it lists the directory only when the bound may be
  exceeded, and once a day (D234). Hundreds of
  ranks starting at once on one directory of a parallel file system
  therefore load its metadata server, mostly with misses.
- The file system has to carry locks of files (`fcntl`), as NFS and the
  parallel file systems do. Where it does not, every compile that stores
  lists the directory, which is slow once it holds thousands of entries.
- Prefer a directory local to each node. To share one directory across a
  large job, warm it first with one compile of the job's program, from a
  single process, so that the ranks only read.
- Concurrent jobs may share a directory. An entry is written to a
  temporary file in the same directory and renamed into place, and POSIX
  file systems, Lustre and GPFS among them, rename within a directory
  atomically; a reader copies the file it opened.

Nodes that differ:

- A GPU entry is keyed by the architecture it was compiled for: the key
  of the PTX holds the module's target attribute, `sm_XY` included, and
  the key of a cubin holds the arguments of `ptxas`, `-arch sm_XY`
  included. Nodes with different GPUs may therefore share a directory;
  each architecture keeps entries of its own.
- A host object is keyed by the CPU and its features, and by the module,
  which holds the binaries of its kernels. Nodes with different CPUs or
  GPUs miss each other's host objects rather than load them.

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
is in the module. The values that depend on the state the run starts from
are not constants of the module but arguments of its entry
([Values of the start](#values-of-the-start-d227)). A rebuild of MDIR
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

## Values of the start (D227)

Issue #162. A stage that continues from an equilibrated state, such as the
production stage of the ala3 example after its NPT stage, a resubmission,
or a replica with a cell of its own, starts from a cell that differs from
run to run. A program whose text held values computed from that cell
differed from the last one only in those numbers, and missed the cache.

The entry now takes them as its last arguments (`Program::startValues`,
`include/mdir/Driver/Builder.h`). The host computes each one exactly as the
constant was computed, so a run gives the same numbers as before to the
bit. This generalizes D213, under which a program with tunables already
took the barostat's two constants as arguments.

Every value that the program text or the pipeline takes from the starting
state:

| Value | Depends on | Now |
|---|---|---|
| `%baro_constant`, `%baro_energy_constant`: the virial and the energy of the correction for the dispersion and the PME background, times $V$ | the volume at the start, through rounding only ($V$ cancels) | arguments of the entry |
| `%rest_edge0` to `%rest_edge2`: the cell that the reference positions of the restraints belong to, with a barostat | a Python stage: the cell of its start; `mdir run`: the cell of the coordinate file | arguments of the entry |
| `%tilt_bx`, `%tilt_cx`, `%tilt_cy`: the tilts of a triclinic cell | the cell at the start | arguments of the entry |
| `%bstate0` to `%bstate8`: the pressure state of a barostat that scales every step (D92) | the checkpoint that `mdir run` continues | arguments of the entry |
| The edges `%lx`, `%ly`, `%lz` | the cell at the start | arguments of the entry already |
| The correction for the dispersion, the PME self term and background, the shift estimate (D210), the tails of pair terms (D209), the constants of `[free_energy]` that scale as $1/V$, and the part of the shift estimate that does not depend on the volume (#224) | the volume at the start | values of the host (`Output`), never in the text |
| The factors of the influence function of PME and LJPME | the grid | entry buffers already |
| The PME and LJPME grid, an attribute of `md.reciprocal` | a Python stage: the cell of its start; `mdir run`: the cell of the coordinate file | **structural**: stays in the text |
| The neighbor capacity, `width` of `convert-md-to-md-exec` | the positions and the cell at the start | **structural**: a pipeline option, in the lowered module; its estimate is rounded more coarsely, and `Execution.neighbor_capacity` fixes it |
| The edges of the coordinate file for an external term in the frame of the cell (D154) | the coordinate file, not the start | stays in the text; the Python model has no external terms |

The grid and the neighbor capacity shape the program's loops and buffers,
so they stay part of the key, and a change of either still compiles anew.
The grid of a Python stage follows the cell it starts from, but it changes
only when an edge crosses a size that the FFT takes, so a continued stage
usually keeps it. `System.pme_grid` fixes it for every stage.

The neighbor capacity is estimated as $\lceil 1.5\,n_\text{max}\rceil + 16$,
where $n_\text{max}$ is the most neighbors that a particle has at the
start. It does not change the results, since a build that finds more
neighbors makes room: `neighbor_capacity` 888, 896, and 1024 on ala3 at
constant pressure gave the same positions, velocities, and cell to the
bit, deterministic on the CPU and a GPU. It does move by a few neighbors
from one equilibrated state to another. Rounded up to a multiple of 8, as
it was, it gave 888, 896, and 888 for three equilibrated states of ala3,
and the second missed the cache on the width alone. By the maintainer's
decision on PR #192:

- **The estimate is rounded up to four significant bits**, that is to a
  multiple of an eighth of the power of two below it (64 between 512 and
  1024) and of 8. This costs at most 12.5% more room at first. All three
  states above take 896.
- **`Execution.neighbor_capacity`** in Python is the control file's
  `[execution] neighbor_capacity`. 0, the default, takes the estimate;
  `Program.plan["neighbor_capacity"]` is the capacity that a compile took.
  A workflow gives its stages one value where the estimates still round
  apart.

### Validation

- **Text.** On main, the program of segments of the ala3 production stage
  from three NPT-equilibrated states (seeds 1 to 3, edges 3.517 to
  3.551 nm) differed in `%baro_constant` and `%baro_energy_constant`, and
  with restraints also in `%rest_edge*`. Now the texts are the same.
- **Cache.** One process per state on one directory, CPU, mixed:

  | State | Capacity, multiple of 8 | With the values as arguments only | Capacity now | Now |
  |---|---|---|---|---|
  | seed 1 | 888 | stores | 896 | stores |
  | seed 3 | 888 | hits (main: a miss) | 896 | hits |
  | seed 2 | 896 | misses on the capacity | 896 | hits |

- **The ala3 example.** Three runs of `examples/ala3/run.py`
  (`--steps-scale 0.01`, GPU, mixed, the default mode, one seed) on one
  directory. The production stage starts from the cell that the NPT stage
  reached, different in every run; its grid is $32^3$ in all of them. On
  main the production stage of a second run missed whatever its capacity.
  With the values as arguments and the capacity a multiple of 8, the
  capacities were 864, 872, and 864: the third run hit the entry of the
  first and the second missed. Now the capacity is 896 in the three runs,
  and every stage of the second and third runs hits: the production stage
  compiles in 12.1 and 11.7 s with 0.5 s in the engine, against 19.3 s and
  6.4 s in the first run.
- **Tests.** `compile-cache-cell.test` and its GPU variant, on the
  dipeptide at constant pressure with restraints, deterministic, in mixed
  and double precision:
  - A start with its cell 1% larger has the text and the pipeline of the
    file's start (capacity 448; a multiple of 8 gave 432 and 424), hits
    its entry, and its 20 steps equal those of a compile without the cache
    bit for bit.
  - A start 2% larger estimates 416 and misses. With
    `Execution.neighbor_capacity` set to the 448 of the first it hits, and
    its 20 steps with either capacity are the same bit for bit.
  - A negative capacity is refused.
- **Bit identity with main.** Deterministic, the positions, velocities,
  cell, and energy file (or energies) after the run equal those of main to
  the bit, on the CPU and a GPU in mixed and double precision, for:
  - `mdir run` on ala3 at constant pressure with restraints, 500 steps, and
    its continuation from the checkpoint;
  - a barostat that scales every step (`work = "TROTTER"`, `interval = 1`),
    200 steps, and its continuation, which takes `%bstate*`;
  - a triclinic water box (the dodecahedron of `triclinic.test`) at constant
    volume and at constant pressure, which take `%tilt_*`;
  - the Python program of segments of the ala3 production stage from an
    equilibrated state, with and without restraints, 300 steps. On the CPU
    this holds in the default mode too.

  The capacities of these runs differ from main's (832 against 824 for
  ala3 from its file), so they also show that the capacity leaves the
  results as they are.
- **Speed.** The Amber suite on GPU 0 (RTX 3090, 300 W), mixed, in the
  settings of the suite (D114), one run per system:

  | System | main, ms/step | Now, ms/step | Rate | Capacity, main | Capacity now |
  |---|---|---|---|---|---|
  | JAC NVE | 0.204 | 0.204 | +0.1% | 960 | 960 |
  | JAC NVE 4 fs | 0.219 | 0.219 | +0.2% | 960 | 960 |
  | JAC NPT | 0.220 | 0.220 | −0.1% | 960 | 960 |
  | JAC NPT 4 fs | 0.229 | 0.229 | 0.0% | 960 | 960 |
  | Factor IX NVE | 0.590 | 0.591 | −0.1% | 976 | 1024 |
  | Factor IX NPT | 0.620 | 0.621 | −0.1% | 976 | 1024 |
  | Cellulose NVE | 2.704 | 2.696 | +0.3% | 1104 | 1152 |
  | Cellulose NPT | 2.792 | 2.787 | +0.2% | 1104 | 1152 |
  | STMV NPT 4 fs | 8.288 | 8.308 | −0.2% | 1016 | 1024 |

  The rates agree within 0.3%, with the larger capacities as without.

## Reuse within a process

D[program-reuse], issue #236. The cache on disk saves the host code generation and nothing
of the MLIR pipeline, whose output is its key; and without a directory
nothing was saved at all. Every `mdir.Simulation(program)` therefore
lowered its program again, also the second and later simulations of one
`Program` in a process: 5.4 s each on the dipeptide in water on the CPU
(3.0 s of pipeline and 2.4 s of code generation) and 9.4 s on a GPU,
and 3.1 s and 5.6 s on hits of the disk cache.

A `Program` now keeps the code of its simulations in memory. The first
simulation of a program lowers and generates code as before (or takes the
object of the disk cache), and leaves with the `Program`, under a key:

- the bitcode of the LLVM module that it gave its JIT engine, with the
  packed wrapper of the entry;
- the relocatable host object of that module.

A later simulation of the same `Program` whose key is the same makes its
engine from the two: it parses the bitcode, and when the engine asks for
the object of the module it gets a copy of the kept one. That is the path
of a hit on disk without the pipeline, the translation to LLVM IR, the key
of the bitcode, and the read. The engine links the copy into memory of its
own, as it does a generated object (D199), so simulations share no code
and may end in any order.

**What is kept is not `Program.lowered_ir`.** A simulation does not lower
`Program.ir`: it builds a program of segments of its own from the model
(D215, [python-segments.md](python-segments.md#compiles)), with another
entry, and lowers that one. The lowered text of the `Program` is for
reading; the kept code is that of the simulation's program.

**The key** is a BLAKE3 hash of everything that the lowering and the code
generation read, each part with its length:

- the text of the module that the builder gives for the simulation. The
  builder runs for every simulation (0.03 s on the dipeptide; it counts the
  neighbors for the width of the neighbor structures, so it grows with the
  system), and its text is a function of the model alone (see
  [The key](#the-key));
- the pipeline with its options: the width of the neighbor structures and
  the skin, the precision, the threads, the deterministic mode, and on a
  GPU the options of the device (its architecture, the binary format, and
  `cache=false`), which hold what the lowering takes from
  `CUDA_VISIBLE_DEVICES`, `MDRT_DEVICE`, `MDIR_GPU_ARCH`, and
  `MDIR_GPU_BINARY`;
- on a GPU, the CUDA toolkit that the environment names (`CUDA_ROOT`,
  `CUDA_HOME`, `CUDA_PATH`), whose libdevice and ptxas the lowering uses;
- the name of the entry;
- the description of the machine and of the code generator that the key on
  disk has (the LLVM version, the CPU and its features, the options of
  code generation, the scheduler).

A simulation whose key differs lowers anew and leaves code under its own
key; a `Program` keeps at most four. Since a `Program` is immutable and a
simulation builds from the model it captured, the keys of one `Program`
differ only when the environment changed between two simulations. The
values of the start (D227) and of the tunables (D213) are arguments of the
entry and data of the program, not code: `Simulation.tunables` already
takes the equality of the module's text as "the same program". Nothing
else that a simulation holds comes from the store: its program's
description, its control, its buffers, and its activation are made anew.

**On a GPU** nothing more is needed. The cubins (or the PTX) of the
kernels are constants of the host module, so they are in the kept object;
the constructors that load them are found by the engine in the IR of the
module, which is why the module is kept beside the object (an object
linked without its module would load no kernels). The kernels are loaded
onto the device again by each simulation.

**`cache=False`** (D217) bypasses the store as it bypasses the directory:
such a simulation takes no kept code and leaves none, and lowers and
generates everything. `MDIR_COMPILE_CACHE=off` and a process without
`MDIR_COMPILE_CACHE_DIR` do not: the store is independent of the cache on
disk. A simulation that takes kept code does not read the directory, so it
does not mark the entry as used either.

**Memory and threads.** The store belongs to the `Program` and is freed
with it; for the dipeptide on the CPU the object is 0.3 MiB, and the
bitcode is of that order. `Simulation.__init__` of several threads
run one after the other already (they share the mutex of runs); the store
has a mutex of its own, held for a lookup or an insertion.

`compile_stats` of a simulation that took kept code has `program_reused`
1, `reuse_saved_seconds` (the pipeline and the object of the simulation
that left the code), `engine_seconds` (the parse and the link, and on a
GPU the loading of the kernels), and 0 for every other time and count.

### Validation

`python-program-reuse.test` and `python-program-reuse-gpu.test`, on the
dipeptide in water with PME, SHAKE, and SETTLE in mixed precision and the
deterministic mode, each without a directory and with one: the positions
and velocities after 20 steps are the same bit for bit from the first
simulation, from two that took its code (one while the first lives, one
after it ended), from one with `cache=False`, from the simulations of a
second `Program` of the same inputs (whose first does not take the code of
the first program: it lowers, and hits the disk if there is a directory),
and from two simulations that two threads make at once; the simulations of
a `Program` compiled with `cache=False` never take kept code, unless they
are given `cache=True`.

Four simulations of one program in a process, the dipeptide in water
(2,269 particles, PME, mixed precision), the time of
`mdir.Simulation(program)` in seconds, on a shared machine:

| | First, before | Later, before | First, now | Later, now |
|---|---|---|---|---|
| CPU, no directory | 5.5 | 5.4 | 5.8 | 0.07 |
| CPU, directory, cold | 5.5 | 3.0 to 3.1 | 5.9 | 0.08 |
| CPU, directory, warm | 3.5 | 3.1 to 3.3 | 3.4 | 0.07 |
| GPU, no directory | 9.5 | 9.3 to 9.5 | 9.4 | 0.12 to 0.14 |
| GPU, directory, cold | 9.6 | 4.9 to 5.5 | 9.7 | 0.10 |
| GPU, directory, warm | 5.8 | 5.6 | 5.8 | 0.13 |

`python-memory.test`, whose 30 simulations of one program were most of its
time, takes 20 s instead of 181 s, and its memory stays flat: +0.00 MiB in
use from simulation 5 to simulation 30, and the address space mapped over
those simulations no longer grows (+576 MiB before, of engines whose
mappings the allocator kept).

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
that died, are removed when the directory is listed for eviction. A
directory that cannot be written
leaves the run as it would be without a cache (`cache_unstored`).

### The bound

D234, issue #196. To keep the directory within `MDIR_COMPILE_CACHE_MAX_MB`, a
process has to know what the directory holds. Listing it and reading the
status of every entry gives that, and it is how entries are evicted: the
entries are sorted by the time of their last use and removed, the least
recent first, until the total is within the bound. But a listing costs
file operations in proportion to the entries of the directory, whoever
stored them. Before D234, every run of the GPU pass that
stored an entry and every stored host object listed the directory. With
the 10,000 entries that a suite leaves in a shared directory, each compiled
program read 10,000 to 20,000 statuses, and on a network file system, with
sixteen tests at once, the tests stalled for minutes.

The cache therefore keeps its total in a file, `<dir>/size`: a tag, the
bytes of the entries, and the time of the last listing in seconds since
the epoch. A compile that stored entries (a host object; the PTX and
cubins of all the modules of one run of the GPU pass, together) opens the
file, locks it, adds the bytes it wrote, and writes it back. That is a
fixed number of file operations, whatever the directory holds. The lock is
a lock of the file between processes (`fcntl`), which a network file
system carries, and a mutex between the threads of a process.

The directory is listed, under the same lock, only when

- the total with the bytes just stored exceeds the bound;
- the file holds no total: a new directory, one written by an earlier
  version, or a file that was removed or damaged;
- the last listing is more than 24 hours old. The 24 hours are a fixed
  number of the implementation; no environment variable changes them.

A listing evicts as described above, removes stale temporary files, and
writes the exact total back. `mdir.clear_compile_cache` takes the bytes
it removed off the total.

What the total can get wrong, and what follows:

- **Too large.** An entry that replaces another of the same key is
  counted twice, and an entry removed by hand or by another version of
  MDIR is still counted. The bound is then reached early, and the listing
  that follows corrects the total and evicts nothing it need not.
- **Too small.** A process that dies between its stores and its update of
  the file leaves bytes that are not counted, at most the entries of one
  program. The directory may exceed the bound by that much until the next
  listing, at most 24 hours later. Entries copied into the directory by
  hand are found the same way.
- **Temporary files.** The temporary file of a process that died is
  removed at a listing once it is an hour old, so within a day, and no
  longer at the next store of any process.
- **No locks.** On a file system that refuses the lock, or when the file
  cannot be opened, every compile that stores lists the directory, as
  before.

A bound of 0 is exceeded by every store, so each lists and keeps nothing.
At the bound, each store that exceeds it lists and evicts down to the
bound and no further; a cache that is meant to stay full of entries in use
is better given a larger bound.

The file belongs to the directory, not to a format of entries: removing it
is safe, and the next store lists.

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
- **The total (D234).** `compile-cache.test` counts the
  listings of the directory: of 300 stores below the bound only the first,
  which finds no total, lists; stores up to the bound do not list, and the
  store that exceeds it lists once and removes the entry used least
  recently; a total that is too large, one listed more than a day ago, and
  a file without a total are each replaced by one listing; a file that
  cannot be used makes every store list, and the bound holds; four
  processes that store 100 entries each at once leave the exact total and
  list nothing; a bound of 0 lists at every store; a clear takes its bytes
  off the total. The file operations of a compile, counted with
  `strace -f` on a local disk (GPU, mixed precision; the four stages of
  `examples/ala3/run.py`, 3,264 entries stored by 4 runs of the GPU pass
  and 4 host objects):

  | Cache before the run | | Listings of `host/` or `gpu/` | Status calls on entries by path | Opens of `size` |
  |---|---|---|---|---|
  | Empty | before | 16 | 13,268 | |
  | | after | 2 | 304 | 8 |
  | 10,000 entries, no total | before | 16 | 93,268 | |
  | | after | 2 | 10,304 | 8 |
  | 10,000 entries and their total | after | 0 | 0 | 8 |

  A listing opens both subdirectories, so 16 are 8 evictions and 2 are
  one: the first store, which finds no total. The wall time of the run
  into an empty cache on a local disk does not change: 57.9 s and 58.0 s
  before, 59.9 s and 58.4 s after.
- **Concurrent processes.** Four processes started at once on an empty
  directory each generate or read the object, none is rejected, one entry
  remains, the states agree with no cache, and the next process hits.
  Every suite also runs up to sixteen GPU tests and the CPU tests on one
  directory at once.
- **Controls (D217).** `compile-cache-controls.test`
  (CPU, mixed) and `compile-cache-controls-gpu.test` (GPU, mixed), each
  scenario a process of its own:
  - `mdir.compile(..., cache=False)` on an empty directory makes none. On
    a warm one its simulation, which inherits the choice (`compile` itself
    lowers nothing, D224), generates the object and, on a GPU, every PTX and cubin,
    hits nothing, and leaves every entry with its size and time (a read
    would have touched it). The state equals that of no cache bit for
    bit. `Simulation(program, cache=False)` after a compile that used the
    cache does likewise for the simulation. `cache_bypassed` is 1.
  - A clear removes every entry of the warm cache (1 host object, and on a
    GPU every PTX and cubin) and leaves a file of another format and a
    writer's temporary file. The next process misses and stores, and the
    one after hits. A clear of another directory, or with no directory,
    removes nothing; under `MDIR_COMPILE_CACHE=off` it still clears the
    directory set.
  - One process clears over and over while another compiles the
    dipeptide four times. On the CPU, about 190,000 clears ran during the
    four compiles; each compile missed and stored, and the clears removed
    the 4 entries. Every entry left is well formed, and the next process
    reads the cache with no rejection and the state of no cache.
  - `compile-cache.test` checks what a clear removes and leaves, and races
    it against a forked writer of 3000 entries under 50 keys: 85 clears
    removed 2947 entries, every write succeeded, and the 14 entries left
    were intact, with no temporary file.
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
in `mdir.compile` (which lowered a program of its own until
D224, #151) and in the simulation, is most of what remains. Without the deterministic mode the
production stage missed: it starts from the cell that the NPT stage
reached, which differs from run to run, and its program's constants
depended on it. They no longer do
([Values of the start](#values-of-the-start-d227)).

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
modules within one program, or in the program that `Program.lowered_ir`
lowers and the simulation's, share one entry. `mdir run` uses these entries too
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
`mdir.compile`, which no longer lowers (D224, #151): the
example then takes 62 s instead of 85 s without the cache. The cache of the example holds 21 MB of host
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

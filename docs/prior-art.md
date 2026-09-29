# Prior Art

Last updated: 2026-09-29.

This document records earlier work that MDIR builds on, what is taken from
it, and what is not. Decisions that follow from it are listed in
[decisions.md](decisions.md).

## 1. PPMD (Saunders, Grant, Müller, 2018)

W. R. Saunders, J. Grant, E. H. Müller, "A Domain Specific Language for
Performance Portable Molecular Dynamics Algorithms," *Computer Physics
Communications*, 2018. doi:10.1016/j.cpc.2017.11.006. arXiv:1704.03329.

Statements in this section were checked against the arXiv version (v2).

### 1.1 What it is

A Python framework in which the user writes the algorithm in Python and the
per-particle or per-pair computation as a short C kernel. The framework
generates wrapper code for the kernel, compiles it at run time, and executes
it on CPUs with MPI or on GPUs. It follows OP2 and PyOP2, which do the same
for mesh-based PDE solvers.

### 1.2 Abstractions

| PPMD concept | Meaning |
|---|---|
| Particle Loop | For each particle, read and write properties of that particle. |
| Particle Pair Loop | For each pair `(i, j)`, read properties of both and modify properties of the first. |
| Local Particle Pair Loop | A pair loop restricted to pairs within a cutoff. |
| `ParticleDat`, `PositionDat` | Per-particle data. Positions are a distinguished kind because cell and neighbor lists depend on them. |
| `ScalarArray` | Global data, such as total energy. |
| Access descriptor | `READ`, `WRITE`, `RW`, `INC`, or `INC_ZERO`, declared for each datum passed to a loop. |
| `Constant` | A value substituted into the kernel source at compile time. |
| `IntegratorRange` | A time-step iterator that rebuilds cell and neighbor lists when needed. |

Both loop definitions require that the result is independent of the order of
execution.

The framework does not inspect kernel code. Access descriptors are its only
source of information about what a kernel does. From them it decides when a
halo exchange is needed and where write conflicts can occur.

### 1.3 Mapping to MDIR

| PPMD | MDIR | Change |
|---|---|---|
| Particle Loop, Pair Loop | `md_exec.particle_for`, `md_exec.pair_for` | Taken as is. |
| Opaque C kernel | IR region | The compiler can analyze and differentiate the body. Opaque kernels remain possible as external kernels. |
| Access descriptor | Op signature under value semantics | Derived and verified for generated kernels; declared for external kernels. |
| Dirty flag checked at run time | Dataflow analysis on state versions | Halo exchange placement becomes a compile-time decision. Run-time tracking is kept only at the boundary with host code. |
| `Constant` | Parameter with static binding | Same idea. |
| `IntegratorRange` | Neighbor structure validity plus the simulation driver | Generalized to cover cell changes and migration. |
| Pair loop writes to the first particle only | Baseline pair execution policy | One of several policies chosen by the planner. |
| Cell list on CPU, occupancy and neighbor matrices on GPU | Physical neighbor representation chosen in `md_exec` | Same semantics, different structure per target. |
| Kernel in, generated code out | `md` above `md_exec` | PPMD has no layer that knows the energy function. MDIR adds one. |

PPMD corresponds to `md_exec` and part of the runtime. It has no counterpart
to `md`, `mlff`, semantic differentiation, staged locality, or the planner.

### 1.4 What the paper does and does not show

**Newton's third law.** The paper loops over all directed pairs and writes
only to the first particle. It reports that this is faster overall than
computing each pair once, because the kernel then vectorizes.

The reason given is specific to PPMD. It relies on the C compiler's automatic
vectorization, which fails when the compiler must assume aliasing between
particles. The paper notes that clustered pair lists with explicit vector
operations solve this, and declines them because they need
architecture-specific intrinsics. MDIR generates vector code itself, so that
argument does not carry over. The reasons to adopt directed pairs as the MDIR
baseline are different: simpler back ends, a good fit for GPUs, and no
reverse communication.

**Performance.** The benchmark is a single-species Lennard-Jones liquid in
double precision.

| Result | Value |
|---|---|
| CPU, one 16-core node, 10^6 particles, 10^4 steps | PPMD 501 s, LAMMPS 569 s |
| CPU, 64 nodes | PPMD 23.8 s, LAMMPS 17.2 s |
| One GPU | PPMD 385 s, LAMMPS 275 s (LAMMPS also used the 16 host cores) |
| CPU weak scaling | Efficiency stays above 90% up to 1024 cores |
| GPU weak scaling | Efficiency falls to about 60% at 16 GPUs |
| Force kernel, fraction of peak floating-point rate | 16.5% on CPU, 11.9% on GPU |

The results show that the abstraction costs nothing relative to LAMMPS on
this system. They do not show that it reaches the performance of hand-tuned
SIMD cluster kernels, and they say nothing about inhomogeneous systems: the
paper states that the system is spatially homogeneous with little load
imbalance.

**GPU communication.** The paper attributes the weaker GPU scaling to
communication overhead and names overlapping communication with computation
as future work. This is the case the MDIR event token is designed for.

**Host overhead.** Each loop launch from Python costs 10 to 20 µs. That is
negligible at 10^6 particles and significant for small systems with step
times well under a millisecond.

**Memory ordering.** Performance degrades during a run as particle order in
memory decorrelates from spatial position. Periodic reordering is future
work in the paper.

**Exclusions.** These are possible today in PPMD: a `ParticleDat` holds the
global IDs of excluded particles and the kernel checks it. Native support is
not provided.

**Species.** Multiple species are handled by branching inside the kernel,
which the paper calls inefficient.

**Future work named by the authors.** Fast multipole and other long-range
methods, native multiple species, constraints and bonded interactions,
particle reordering, memory layout studies, and kernel generation from the
analytical form of a potential as in OpenMM, while keeping hand-written
kernels possible.

### 1.5 Beyond force calculation

The paper implements bond order analysis and common neighbor analysis with
the same loops. Common neighbor analysis takes three consecutive pair loops,
each reading per-particle results of the one before. This shows that the
relational primitives are useful outside Hamiltonians, and it is a multistage
locality case of the same shape as EAM.

## 2. Other systems referenced in the design discussion

These entries are summarized from general knowledge and have not been checked
against sources for this document.

| System | Relevance to MDIR |
|---|---|
| OpenMM | Custom forces from expression strings with symbolic differentiation; kernels generated and compiled when the context is created; `CustomIntegrator` as a model for `dyn`; Python object API. |
| GROMACS | Cluster pair lists; eighth-shell decomposition with dynamic load balancing; separate PME ranks; update groups; thread-MPI for running without an MPI library; fixed-interval pair list with a buffer estimate. |
| LAMMPS | Full-shell ghost atoms; pair styles declare forward and reverse communication; hook-based step loop. |
| HOOMD-blue | GPU-first design; counter-based random numbers keyed by particle and step; run-time compiled user potentials. |
| JAX-MD | Value semantics for state; forces by automatic differentiation of the energy. |
| AutoPas | Run-time selection among containers, traversals, data layouts, and Newton's-third-law usage. A model for the joint planner. |
| Cabana | Library of particle data structures, neighbor lists, and halo communication with AoSoA storage. A model for the runtime. |
| Allegro, SevenNet | Strictly local MLFF, and message-passing MLFF with per-layer feature communication. The two halo strategies for MLFF. |
| OP2, PyOP2 | Origin of loops with access descriptors. |
| MLIR `shard` dialect | Distributed dense tensors with halos. Reference for distributed grids. |

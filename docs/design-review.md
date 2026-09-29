# MDIR Design Review

Reviewed: revision 1 of [architecture.md](architecture.md), draft of
2026-09-29. The architecture document has since been revised; section and
roadmap references below refer to revision 1, which is no longer in the
repository. The outcome of each finding is
recorded in [decisions.md](decisions.md).

Upstream MLIR claims were checked against the documentation at
`mlir.llvm.org` on 2026-09-29. Items that were not checked are marked
"unverified."

## 1. Verdict

The layer boundaries are the right ones. Separating Hamiltonian, dynamics,
protocol, distribution, and execution matches how the concerns actually
decouple in production MD engines, and stopping at two custom execution
dialects is a sound restraint.

The design is weakest in three places:

1. It describes what each layer contains but not how layers exchange
   information. Several facts cross layer boundaries in ways the current
   boundary rule forbids.
2. It assumes more from upstream MLIR than upstream provides, especially for
   communication, asynchrony, and parallel force accumulation.
3. The roadmap cannot produce a running simulation, because the layer that
   generates executable kernels is not in it.

Findings are ordered by how much they would cost to fix later.

## 2. Critical findings

### 2.1 The roadmap has no path to executable code

v0 is `md`, `dyn`, `md_dist`. Nothing in that set lowers to loops, so v0
cannot run a simulation or be validated against a reference engine. `md_exec`
is required for even a single-core Lennard-Jones run, and `md_dist` is not.

**Recommendation.** Build a vertical slice first and treat `md_dist` as an
identity pass until the slice works:

```text
md → (differentiation) → md_exec → scf → llvm      single node, CPU
```

Revised order: `md`, `md_exec`, minimal `dyn`, then `md_dist`, then `mlff`
and `ensemble`. See Section 5.

### 2.2 Differentiation is unassigned

`md.evaluate` requests forces, virial, and `dH/dλ`, all of which are
derivatives of the energy. The design never says which layer differentiates.
This cannot be deferred, because it constrains the layers below:

- The backward pass of a message-passing model needs its own communication:
  adjoints accumulated on ghost particles must be sent back to their owners,
  layer by layer. `md_dist` can only plan that if the derivative program
  exists, with locality information intact, **before** distributed lowering.
- The same holds for many-body potentials such as EAM, where an intermediate
  per-particle quantity must be forward-communicated between two passes.
- Differentiating at the LLVM level (for example with Enzyme) happens after
  communication has been lowered, so the compiler would have to differentiate
  through halo exchange. Enzyme can do this, but the adjoint communication
  it produces mirrors the primal calls one for one. The planner never sees it
  and cannot merge, reorder, or overlap it with computation.

**Recommendation.** Differentiate at the semantic level. Make it an
`md`-to-`md` transformation whose output is again expressed in `md`/`mlff`
ops: symbolic derivatives for closed-form pair and bonded terms, reverse-mode
for `mlff`. Derive the virial from the same pass.

### 2.3 State is implicit, which defeats analysis

The examples mix three conventions: symbols (`@atoms`, `@H`), SSA values
(`%pairs`, `%h1`), and ops with no state operand at all
(`dyn.kick 0.5 * %dt`). In the `dyn` examples the state is an implicit global
that every op mutates.

With implicit state the compiler cannot answer the questions this design
depends on:

- Which ops change positions, and therefore invalidate neighbor structures?
- Which ops change the cell, and therefore invalidate the domain partition?
- Are two adjacent half-kicks fusible across a step boundary?

**Recommendation.** Give the semantic dialects value semantics:

```mlir
%s1 = dyn.kick  %s0, %f0, %half_dt
%s2 = dyn.drift %s1, %dt
%f1 = md.evaluate @H(%s2) request [#md.forces]
%s3 = dyn.kick  %s2, %f1, %half_dt
```

Integrator rewrites then become ordinary pattern rewrites on use-def chains,
and invalidation becomes a property of which state components an op
redefines. Decide in-place updates late, in the same spirit as MLIR
bufferization.

### 2.4 Locality is richer than a radius

"Support 5 Å implies halo ≥ 5 Å" holds only for a single-pass pair potential
with no list buffering. In general the halo depends on:

| Source | Effect on halo |
|---|---|
| Verlet buffer (skin) | Halo is cutoff plus skin; skin is an execution choice. |
| Message passing with L layers | Either a halo of L × cutoff with no per-layer exchange, or a halo of one cutoff with a feature exchange per layer. This is a cost-model decision. |
| Many-body terms (EAM, Tersoff) | Intermediate per-particle fields must be communicated between passes. |
| Bonded terms | Determined by the spatial extent of bonded groups, not by a cutoff. |
| Constraints, virtual sites, rigid bodies | Coupled atoms must be co-located or communicated. These live in `dyn`, so `md_dist` needs input from `dyn` as well as `md`. |
| Newton's third law | Computing each pair once requires reverse force accumulation; computing it twice does not. |

**Recommendation.** Define locality as an op interface, not an attribute. Each
interaction op reports a staged dependency description: for each stage, the
support radius, the per-particle fields read, and the per-particle fields
written. `md_dist` planning consumes only that interface. This also gives
`mlff` and `md.external_potential` a uniform way to participate.

## 3. Major findings

### 3.1 The top of the stack is a reference graph, not a lowering chain

The diagram draws `ensemble` above `md` and `dyn` as if it lowered into them.
In fact the three are peers in one module linked by references: `ensemble`
names states and `dyn` programs, `dyn` names a Hamiltonian, and lowering only
begins at `md_dist`. The document should say so, because it changes what
"layer" means for pass design.

Two bindings are missing:

- **State to dynamics.** Temperature is declared in `thermo.state`, but
  `dyn.thermostat` needs it. `dyn` programs must be parameterized by the
  thermodynamic state.
- **Consistency.** Nothing prevents pairing an NVT state with plain velocity
  Verlet. Let each `dyn` program declare the ensemble it samples and verify
  the pairing.

Replica exchange also needs cross-evaluation, the energy of replica `j`'s
configuration under state `i`'s Hamiltonian. That is an `md.evaluate` request
the protocol layer must be able to issue.

### 3.2 Nobody owns the step loop

The design covers one time step and the sampling protocol, but not the loop
between them. Unassigned responsibilities: the step loop itself, neighbor
rebuild triggers (dynamic, based on maximum displacement), migration and load
balancing cadence, trajectory and checkpoint output, and observables.

**Recommendation.** Add an explicit driver construct, owned by `dyn` or by a
thin layer above it, in which periodic and conditional events are visible to
the compiler.

### 3.3 Long-range interactions have no lowering path

`md` lists fields, but every lower layer assumes cutoff-based neighborhoods.
PME and PPPM need particle-to-grid spreading, a distributed FFT whose
decomposition differs from the particle decomposition, and grid-to-particle
gathering. None of `md_dist`'s concepts cover that.

**Recommendation.** Exclude mesh electrostatics from v0 explicitly and use
cutoff-based electrostatics (reaction field or damped shifted force). Reserve
the design space now, so that `md_dist` is not built on the assumption that
all communication is particle halo exchange.

### 3.4 The compile-time versus run-time boundary is undefined

The notes imply that everything is compiled, including
`md_exec.build_cells`, migration, and packing. These are dynamic,
data-structure-heavy operations where specialization gains little. The gains
are in the kernels: inlined potentials, fused terms, specialized parameters.

**Recommendation.** State the split: the compiler generates kernels and the
schedule; a runtime library owns spatial data structures, migration, and
communication buffers. `md_exec.build_*` ops lower to runtime calls.

A related decision is ahead-of-time versus just-in-time compilation. "Static
parameters" only make sense once it is settled whether particle count,
species, and force-field parameters are known at compile time.

### 3.5 Distribution and execution decisions are coupled

The ordering argument in the architecture document is sound for lowering. The
decisions themselves are not independent: domain boundaries should align with
the cell or cluster grid, whose spacing is an `md_exec` choice, and the skin
affects both the halo width and the rebuild interval.

**Recommendation.** Separate planning from lowering. A planner makes the
joint decision with a cost model and records it; the two lowerings then run
in the stated order and read the plan. Two further requirements:

- With one rank, `md_dist` must reduce to the identity.
- Replica parallelism and spatial decomposition are separate axes and must
  compose. "Replica communicator" alone does not describe that.

### 3.6 `md_exec` is overloaded and dependencies have three homes

`md_exec` owns neighbor structures, loop forms, reductions, layout, fusion
boundaries, and a task DAG. Ordering constraints are expressed in
`md_dist.await`, in `md_exec.task`/`md_exec.depend`, and in `async`/GPU
tokens.

**Recommendation.** Represent dependencies once, as SSA token values
introduced in `md_dist` and preserved through `md_exec`. Drop
`md_exec.task` and `md_exec.depend` unless a concrete need appears.

### 3.7 Parallel force accumulation needs an explicit conflict strategy

Force accumulation is a scatter-add. With each pair computed once, iterations
`i` and `j` both write `f[i]` and `f[j]`. Lowering that to a parallel loop is
a data race. The strategy must be chosen in `md_exec`:

| Strategy | Cost |
|---|---|
| Full neighbor list, each iteration writes only `f[i]` | Every pair computed twice; the usual GPU choice. |
| Half list with atomic adds | Atomic overhead; nondeterministic summation order. |
| Half list with coloring or per-thread buffers | Extra memory or preprocessing. |

This choice interacts with reverse accumulation in `md_dist` (Section 2.4)
and with reproducibility (Section 4).

## 4. Minor findings

- **Exclusions.** Bonded exclusions and scaled 1-4 pairs modify
  neighborhoods. Exclusion masks are a significant part of real pair-list
  kernels and are not mentioned.
- **Units.** Examples mix `nm` and `angstrom`. Choose an internal unit system
  and a normalization pass.
- **Precision and reproducibility.** Mixed precision and the summation order
  of parallel reductions are unaddressed. Decide whether bitwise
  reproducibility is a goal.
- **Cell changes.** A barostat changes the cell, which invalidates both the
  partition and the cell grid. This follows from Section 2.3 but deserves its
  own test case.
- **Time-dependent Hamiltonians.** Metadynamics and other history-dependent
  biases update Hamiltonian parameters from inside the dynamics. `H(x; θ)`
  allows this only if something is permitted to write `θ`.
- **Front end.** The notes do not say how users produce `md` IR. Importers
  for existing force-field and topology formats are also what makes
  validation against reference engines possible.
- **MLFF import.** Real models are trained in PyTorch. An `mlff` dialect
  needs an import path, and weights need a storage strategy. With an opaque
  `md.external_potential`, per-layer feature exchange is unavailable and the
  halo must be L × cutoff.
- **AoS layout.** A `memref` has a single element type, so an array of
  structures with mixed field types (for example `f32` position with `i32`
  species) cannot be one `memref`. SoA and AoSoA are unaffected.
- **Positioning.** The notes do not state what MDIR should do better than
  existing engines, nor the benchmark that would show it. Goals and non-goals
  should be written down before op design starts.

## 5. Upstream MLIR assumptions

| Claim in the notes | Status | Finding |
|---|---|---|
| Dialects `gpu`, `nvgpu`, `nvvm`, `amdgpu`, `rocdl`, `mpi`, `async`, `dlti` exist | Verified | All are listed upstream. |
| Halo exchange lowers to `mpi.isend`, `mpi.irecv`, `mpi.waitall` | Partly wrong | `mpi.waitall` does not exist. The dialect has 17 ops, including `mpi.isend`, `mpi.irecv`, and `mpi.wait`. It has no derived datatypes, Cartesian topologies, neighborhood collectives, or persistent requests. |
| `mpi` models MPI 4.0 and is unstable | Verified | The documentation says stability "is not guaranteed at this juncture" and recommends inquiring before use. |
| `async` expresses the interior/halo/boundary DAG | Misleading | `async` lowers to LLVM coroutines and requires a CPU thread-pool runtime. It does not model MPI requests or GPU streams. |
| GPU ops carry async tokens | Verified, limited | `!gpu.async.token` exists on a fixed set of ops such as `gpu.launch_func`, `gpu.memcpy`, and `gpu.alloc`. There is no upstream way to make a GPU launch depend on an MPI request. |
| `scf.forall` is target-independent and maps to GPU blocks and threads | Verified, with caveats | It is built around tensor `shared_outs`; the order of side effects across threads is unspecified. See Section 3.7. `scf.parallel` with `scf.reduce` covers scalar reductions such as energy. |
| `nvvm` has warp shuffle, vote, async copy, MMA/WGMMA, TMA | Verified | Most of these serve tensor-core workloads. Warp shuffle and vote are the ones relevant to MD kernels. |
| `nvvm` has atomics | Partly verified | The dialect's scope statement lists atomics, but no individual atomic op is documented. Atomic force accumulation would more likely use `memref.atomic_rmw`. |
| NVIDIA path `gpu` → `nvgpu` → `nvvm` → NVPTX | Verified | `gpu-lower-to-nvvm-pipeline` and `gpu-module-to-binary` with `#nvvm.target` / `#rocdl.target` exist. |
| Custom `memref` layouts cover SoA, AoSoA, cluster-packed | Plausible | See the AoS caveat in Section 4. |

Two consequences:

1. **Plan for a project-owned runtime ABI.** Communication and task
   scheduling will most likely lower to calls into a thin runtime library
   with its own token type. Treat the upstream `mpi` and `async` dialects as
   optional back ends, not as the default path.
2. **Look at the `shard` dialect.** Upstream has a dialect, formerly named
   `mesh`, that models tensors distributed over a process grid, with halo
   sizes and an `update_halo` op. It targets dense ranked tensors, so it does
   not fit particles that migrate. It is relevant prior art for `md_dist` and
   a possible fit for mesh fields (Section 3.3). The notes do not mention it.

Practical notes:

- MLIR has no stable C++ API. Pin one LLVM release and build out of tree.
- The development machine has no LLVM or MLIR tools on `PATH`
  (`mlir-opt`, `mlir-tblgen`, and `llvm-config` were not found). Building
  LLVM with MLIR is the first implementation task.

## 6. Recommended v0

One vertical slice, validated end to end:

| Item | Choice |
|---|---|
| System | Single-species Lennard-Jones fluid, periodic orthorhombic cell |
| Ensemble | NVE, velocity Verlet |
| Target | Single node, CPU first, then one GPU |
| Pipeline | `md` → differentiation → `md_exec` → `scf` → `llvm` |
| Runtime library | Cell list, neighbor list, I/O |
| `md_dist` | Identity pass |
| `dyn` | Kick, drift, force evaluation only |
| Validation | Energy conservation; forces against finite differences; energies and forces against a reference engine |

Build a reference interpreter for the semantic dialects alongside the
compiler. It serves as the correctness oracle for every lowering added later.

Second milestone: multi-rank execution of the same system, which exercises
`md_dist`, the locality interface, and the runtime ABI.

## 7. Decisions to make, in order

1. Goals, non-goals, and the target benchmark.
2. Ahead-of-time or just-in-time compilation.
3. State model: value semantics or implicit mutable state (Section 2.3).
4. Where differentiation happens (Section 2.2).
5. The locality interface (Section 2.4).
6. The compiler/runtime split and the runtime ABI (Sections 3.4 and 5).
7. v0 scope (Section 6).

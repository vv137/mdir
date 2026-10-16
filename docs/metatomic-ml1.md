# ML1: an Allegro model through an optional metatomic adapter (design)

Design of `D[metatomic-plugin]` (issue #137) for the first deliverable of
M3, ML1 ([roadmap](roadmap.md#7-learned-potentials-m3-and-distributed-execution-m4),
D166, D167, D169): one existing Allegro artifact evaluated through an
optional metatomic adapter, with energy, per-particle forces, and virial on
one GPU agreeing with the original backend. Nothing here is implemented in
MDIR yet. The feasibility study measured below ran outside the repository;
the wrapper it used is `scripts/mlip/nequip-to-metatomic.py`.

## 1. What has been ruled (the maintainer, 2026-10-16)

1. **Allegro stays the first model, through a wrapper of the upstream
   checkpoint.** No Allegro model has an official metatomic export (NequIP
   0.19.1 integrates with ASE, TorchSim, and LAMMPS ML-IAP; metatrain has
   no Allegro architecture). A wrapper of a few dozen lines turns the
   NequIP package into a metatomic artifact. It lives in the repository as
   an example of bringing a model without an official export
   (`scripts/mlip/nequip-to-metatomic.py`). Model files are never
   committed; tests build the artifact or are skipped.
2. **ML1's reference is NequIP's own calculator** (`NequIPCalculator`).
3. **Memory is a primary result of ML1**, not a later concern: MDIR began
   because learned potentials use too much device memory. The eager
   baseline in bytes per pair (Section 4) is the number every later step
   is measured against.
4. **The accelerated path is part of the design now** (Section 5): the
   contract must let ML1's opaque TorchScript call be replaced by an
   accelerated one without changing the control file.
5. **The plugin is a C++20 target behind a C ABI** (Section 6.1).

## 2. The model and the versions

| Item | Value |
|---|---|
| Model | `mir-group/Allegro-OAM-L:0.1`, Zenodo [doi:10.5281/zenodo.16980200](https://doi.org/10.5281/zenodo.16980200), file `Allegro-OAM-L-0.1.nequip.zip` (md5 `0db7f9b3c3a62e74d78b3fcf2973c462`, SHA-256 `3f0d3ca7bb136d4c2ee76278170fcbe756436f037e16c673a86e2c57d271c64e`, 80.7 MB) |
| License | CC-BY-4.0 (the model files); NequIP and Allegro MIT; metatomic and metatensor BSD-3-Clause |
| Architecture | Allegro, strictly local, $r_\mathrm{max}$ = 7 Å, 4 layers, 64 tensor channels, 9.7 million parameters, float32 weights; 89 elements, H to Pu |
| Training | OMat24, fine-tuned on sAlex and MPTrj (PBE); a model for inorganic solids. Its water is not validated physics: ML1 checks the interface, not the science (D166) |
| Packaged with | nequip 0.14.0, allegro 0.7.1, torch 2.7.0 (the package carries its code) |
| Evaluated with | torch 2.14.1 (CUDA 13.0), nequip 0.19.1, nequip-allegro 0.8.3, metatomic-torch 0.1.19, metatensor-torch 0.10.8, metatomic-ase 0.1.4, vesin 0.6.2, ase 3.29.0, NumPy 2.5.4; Python 3.12 |
| Devices | RTX 4090 (24 GB, driver 595.45) and RTX PRO 6000 Blackwell (96 GB) |

The environment is a uv lock of 80 packages, kept beside the measurements;
the implementation PR adds it under `scripts/mlip/` with the scripts.

## 3. Feasibility

### 3.1 The wrapper

`nequip-to-metatomic.py` loads the package, takes its energy network (the
module inside NequIP's `ForceStressOutput`), and wraps it as a metatomic
model that asks for one pair list, full and strict at 7 Å. The engine's
pair vectors become NequIP's `edge_vectors`, which NequIP uses as given;
the engine registers them with autograd (`register_autograd_neighbors`),
so forces and virial come from the engine's backward. The weights are
untouched. `torch.jit.script` still compiles the model under torch 2.14,
with a deprecation warning; `nequip-compile --mode torchscript` itself
refuses torch 2.10 and later, which the wrapper does not use. `--dtype`
sets the type of the inputs that the engine hands over: float64 (as
NequIP's calculator does) or float32.

With `selected_atoms` the wrapper keeps only the edges whose first atom is
selected and returns the energy of those centers. Allegro is strictly
local [[Musaelian2023]](references.md#musaelian2023), so this gives the
exact energy of the selection. An engine can therefore evaluate a system in
chunks of centers (Section 4.2).

### 3.2 Agreement with NequIP's calculator

Water boxes (water is chosen for MDIR's later work; this model was not
trained for it): 27 waters 5 Å apart in a cubic and in a triclinic 15 Å
cell (81 atoms, 2,830 and 2,968 directed pairs within 7 Å; $L \ge 2 r_c$),
and 8 waters at liquid density in a 6.2 Å box (24 atoms, 3,448 pairs; $L <
2 r_c$, several images of each atom). Metatomic's ASE calculator against
NequIP's, one RTX 4090:

| Case | Inputs | $\lvert\Delta E\rvert$ (eV) | max $\lvert\Delta F\rvert$ (eV/Å) | max $\lvert\Delta W\rvert$ (eV) |
|---|---|---|---|---|
| 81 atoms, cubic | float64 | $3.8 \times 10^{-6}$ of 366.7 | $2.2 \times 10^{-6}$ of 0.57 | $2.5 \times 10^{-6}$ of 24.3 |
| 81 atoms, triclinic | float64 | $2.8 \times 10^{-6}$ of 365.7 | $2.6 \times 10^{-6}$ of 0.61 | $4.8 \times 10^{-6}$ of 27.2 |
| 24 atoms, small box | float64 | $1.5 \times 10^{-6}$ of 113.9 | $7.8 \times 10^{-7}$ of 0.95 | $1.2 \times 10^{-6}$ of 5.1 |
| 81 atoms, cubic | float32 | $1.4 \times 10^{-5}$ | $3.3 \times 10^{-5}$ | $8.3 \times 10^{-5}$ |
| 81 atoms, triclinic | float32 | $1.5 \times 10^{-6}$ | $5.8 \times 10^{-5}$ | $1.1 \times 10^{-4}$ |
| 24 atoms, small box | float32 | $1.7 \times 10^{-7}$ | $2.0 \times 10^{-5}$ | $5.3 \times 10^{-5}$ |

The reference itself is not reproducible to the bit: five evaluations of
one configuration differ by up to $2.5 \times 10^{-6}$ eV/Å in the forces
(float64 inputs; $4.8 \times 10^{-6}$ with float32), and its energy moved by
$2.4 \times 10^{-6}$ eV between two jobs. The float64 differences are at
that level. Rounding the inputs to float32 adds up to $6 \times 10^{-5}$
eV/Å. The virial is symmetric to the bit and its net force is below
$10^{-15}$ eV/Å with float64 inputs ($10^{-6}$ with float32).

Central differences of the energy agree with the autograd forces within
$10^{-3}$ eV/Å (steps 0.01 and 0.005 Å) and with the virial within
$10^{-2}$ eV of 25 eV (strains $2 \times 10^{-3}$ and $10^{-3}$). The
float32 network makes the energy noisy at about $10^{-5}$ eV, which bounds
this check; the agreement with NequIP's own autograd above is the stronger
one.

A short NVE run through metatomic's ASE calculator (float64 inputs; 81
atoms; velocity Verlet, 0.5 fs, 2,000 steps; started at 300 K from the
lattice, which the model drives to about 1,200 K): the total energy has a
standard deviation of 2.2 meV against potential-energy fluctuations of 3.3
eV, and a drift of $-3.0$ meV/ps ($-3.5$ kJ/mol/ns per atom). The gate of
the implementation starts from a minimized configuration.

### 3.3 A C host through a C boundary

A prototype outside the repository checks the boundary of Section 6: a
plain C program stands in for MDIR's core (CUDA driver API, the primary
context, buffers from `cuMemAlloc`, a non-blocking stream of its own, as
`runtime/mdrt_cuda.c` does) and calls a C++20 plugin through `dlopen`
(`RTLD_LOCAL`) and five C functions. The plugin wraps the host's device
buffers with `torch::from_blob` (no copies), runs torch on the host's stream
(`c10::cuda::getStreamFromExternal` and a stream guard), builds the
metatomic `System` and the pair list (pairs, cell shifts, vectors), asks
metatomic for nm and kJ/mol (metatomic converts its units), differentiates
in the positions and a strain, and writes the forces into the host's
buffer. Against NequIP's calculator on the three cases:

| Inputs | $\lvert\Delta E\rvert$ (kJ/mol) | max $\lvert\Delta F\rvert$ (kJ/mol/nm) | max $\lvert\Delta W\rvert$ (kJ/mol) |
|---|---|---|---|
| float64 | $\le 3.9 \times 10^{-4}$ of 35,384 | $\le 3.4 \times 10^{-3}$ of 921 | $\le 7.9 \times 10^{-4}$ of 2,627 |
| float32 | $\le 6.2 \times 10^{-4}$ | $\le 3.6 \times 10^{-2}$ | $\le 9.0 \times 10^{-3}$ |

The float64 differences equal the run-to-run spread of the reference
($3.4 \times 10^{-3}$ kJ/mol/nm = $3.5 \times 10^{-6}$ eV/Å). The C path
and the Python path take the same time (59 ms against 61 ms at 2,830
pairs): the boundary costs less than the scatter of a measurement. Loading
the artifact takes 0.35 s.

Two facts of the build: the headers of libtorch 2.14 require C++20 (MDIR is
C++17), and torch must not be linked into MDIR's core, which links LLVM
without RTTI and exceptions (#98). Both lead to the plugin of Section 6.1.

## 4. Memory: the eager baseline and chunking

### 4.1 Bytes per pair

Liquid water (29.9 Å³ per molecule, random orientations on jittered sites),
from 4 to 1,024 waters, energy, forces, and virial in one call;
`torch.cuda.max_memory_allocated` above what the model holds, the driver's
view (`cudaMemGetInfo`, all of the process), and the shortest of three
evaluations after a warm-up:

| Path | Bytes per pair | Time per pair, RTX 4090 | Largest that ran on 24 GB | Largest on 96 GB |
|---|---|---|---|---|
| NequIP calculator, eager (reference) | 1.41 MB | 20.2 µs | 10,546 pairs (72 atoms): 13.9 GiB, driver 22.9 GiB | 42,652 pairs (288 atoms): 56.1 GiB |
| metatomic artifact, TorchScript, whole system | 1.22 MB | 20.3 µs | 10,546 pairs (72 atoms): 12.0 GiB | 42,652 pairs (288 atoms): 48.4 GiB |
| same, 64 centers per call | flat 10.3 to 11.1 GiB | 22.7 µs | 456,424 pairs (3,072 atoms, the largest tried): 10.4 s | the same: 7.4 s |
| same, 16 centers per call | flat 2.6 to 2.9 GiB | 31.2 µs | 456,424 pairs: 14.2 s | the same: 10.9 s |
| NequIP AOTInductor (`nequip-compile --mode aotinductor`) | 132 kB | 3.5 µs | 172,044 pairs (1,152 atoms): 21.2 GiB | 456,424 pairs: 56.1 GiB |
| NequIP AOTInductor with `enable_TritonContracter` | 85 kB | 2.4 µs | 220,724 pairs (1,536 atoms): 17.5 GiB | 456,424 pairs: 36.1 GiB, 0.96 s |

Each larger size failed for lack of memory (13,556 pairs for the eager
paths on 24 GB, 55,674 on 96 GB; 220,724 for AOTInductor and 342,762 with
Triton on 24 GB). The RTX PRO 6000 takes 0.67 (eager) to 0.91
(AOTInductor) of the RTX 4090's time. All paths agree with the reference, or with the 64-center chunks
where the reference ran out of memory: the forces within $5 \times 10^{-6}$
eV/Å at every size, the energy within $5 \times 10^{-4}$ eV of $1.4 \times
10^4$ eV at 3,072 atoms, which is float32 summation in another order.

**Where the bytes go** (RTX 4090, 48 atoms, 6,872 pairs, eager network with
`saved_tensors_hooks`): autograd keeps 97 kB per pair for the backward
(89% of it four kinds of per-pair float32 feature: 1,024 channels, 64 × 16,
and two of 64 × 16 × 1), while the peak, 1.20 MB per pair, is reached in
the forward. The peak is therefore a transient: one temporary of about 1
MB per pair in Allegro's strided contraction (`_contract`), the outer
product of the tensor features before it meets the Wigner 3j coefficients
(an attempt with 54,000 pairs asked for one allocation of 52.7 GiB). This
is what the fused Triton kernel removes. Recomputing the saved tensors
instead of keeping them would cut at most 8% of the eager peak. In the
Triton path, 85 kB per pair in all, the saved tensors are a larger share;
how large was not measured.

### 4.2 Chunking over centers

Allegro's energy is a sum over centers, and a center's energy depends only
on its own edges. An engine can then evaluate the energy of a subset of the
centers with their edges, run the backward, accumulate the forces and the
virial, free the graph, and take the next subset; the result is exact up to
the order of the sums. In metatomic this is `selected_atoms`: the engine
calls the model once per chunk with the chunk's atoms selected and the full
pair list in the `System`. Memory follows the chunk only if the model
drops the edges of the other centers, which the wrapper does; a model that
evaluates every center and selects afterward accepts `selected_atoms`
without saving memory. A chunk recomputes nothing: the edges of one center
belong to that center alone. What it repeats is the per-call work that does
not scale with the chunk (the pair vectors of the whole list, the
TorchScript call, the small kernels). Where the whole system fit (72
atoms, 10,546 pairs), 64 centers per call took 1.03 times its time and 16
centers 1.10 times; at 456,424 pairs, 16 centers took 1.37 times as long as
64 (31.2 against 22.7 µs per pair).

The driver's view grows above the allocated peak with chunks (21.4 GiB
against 10.8 GiB allocated at 64 centers): the caching allocator keeps
blocks of chunks of different sizes. The memory split of Section 6.5 has to
set `expandable_segments` or bound the allocator.

## 5. The accelerated path

### 5.1 What NequIP provides today

`nequip-compile --mode aotinductor` exports the model with `torch.export`
and compiles it with AOTInductor to a `.pt2` package of kernels for one
device kind, with dynamic numbers of atoms and edges (45 s to compile
here). Its targets fix the inputs and outputs: `ase` takes positions,
`edge_index`, `edge_cell_shift`, cell, and types, and returns energy, forces,
and stress, all derivatives inside the package. Modifiers swap kernels
before the export: `enable_TritonContracter` (Allegro's fused tensor product
in Triton, for AOTInductor only) and `enable_CuEquivarianceContracter`
(cuEquivariance, not installed here). On the water boxes these cut memory
by 9 (AOTInductor) and 14 times (with Triton) against the eager artifact,
and time by 5.9 and 8.6 times (Section 4.1). The package needs the neighbor
list as edges and cell shifts: the engine still owns it.

### 5.2 What metatomic can carry

metatomic-torch 0.1.19 serializes models with TorchScript only: an
artifact is a TorchScript archive, and C++ engines load it with
`metatomic_torch::load_atomistic_model`. A draft pull request upstream
(metatensor/metatomic#354, 2026-10-14) adds a base class for models in
eager PyTorch, which allows TorchScript or `torch.compile` inside a model
but not at its top level, and says nothing of `torch.export` or
AOTInductor or of how a C++ engine would load such a model. The
AOTInductor package cannot be put inside a TorchScript artifact, and a
TorchScript model that calls Triton kernels cannot be scripted. A fused
model in metatomic form is therefore not available now.

The adapter's C ABI can load an AOTInductor package directly
(`torch::inductor::AOTIModelPackageLoader`, whose `run` takes a CUDA
stream), with MDIR still owning the pair list, the buffers, and the stream.
That is a second backend behind the same C functions, not a change of
metatomic: the plugin converts MDIR's pair list to the package's inputs
(`edge_index` and `edge_cell_shift` instead of a metatomic `System`), and
takes its forces and virial as outputs instead of differentiating. Its
model description (types, cutoff, units) comes from the package's metadata
(NequIP writes `r_max`, `type_names`, and the model's dtype). Units are
the plugin's to convert, which metatomic does for its backend.

### 5.3 The contract, shaped now for both

The control file and the C ABI name what is evaluated, not how, so that a
later backend changes neither:

- **Backend.** A key selects the loader: `METATOMIC` (a TorchScript
  artifact through metatomic) in the first PR, `AOTI` (an AOTInductor
  package of NequIP's `ase` target) later. Both take the same pair list
  and give the same outputs. The model's file decides nothing that the run
  plan owns (D166, "model artifacts and execution configuration").
- **Precision.** The type of the inputs that MDIR hands over is the
  model's (metatomic's `dtype`, the package's `model_dtype` and its
  inputs), never chosen by MDIR's `precision`: MDIR computes the pair
  vectors in f64 and rounds them to that type, and adds the forces into its
  own force type.
- **Chunking.** A key bounds the centers per call (Section 4.2). The
  plugin honors it only for a model that declares that it is local (a
  finite interaction range and the selection of centers dropping the
  others' edges). The wrapper declares this in its metadata; any other
  model takes the whole system, and the log says so.
- **Buffers and completion.** MDIR owns every buffer it hands over and the
  stream; the plugin owns its allocator's memory and its model's
  intermediates. A call returns with its work enqueued on MDIR's stream.
  The energy and the virial are written to device buffers, so that a step
  does not wait for them on the host (Section 6.5).

### 5.4 Where MDIR's compilation adds to this

A fused kernel cuts the bytes of one tensor product. What remains is a
property of the whole step, which no model package sees: which per-pair
features are kept for the backward and which are recomputed, when the
pair list, the features, and the classical terms' buffers are alive, how
the forces of the model and of PME share memory, and how a step's chunks
are laid out. This is ML4's staged execution
([architecture plan](future-architecture-plan.md#external-ad-and-execution-layers)):
with the stages of a model visible to MDIR, it can fuse the gathers of
positions and pair vectors into the first stage, recompute instead of
store where a feature is cheap to recompute, and plan the memory of the
step across the model and the classical terms, as it plans the memory of
its own loops now. Saving device memory is the reason MDIR exists for
learned potentials: the same model on the same GPU for larger systems, or
on fewer GPUs.

| Route | Fits ML1's opaque call | Needs stages (ML4) |
|---|---|---|
| Chunking over centers | Yes: `selected_atoms`, for a declared local model | Finer chunks per stage |
| Recomputation instead of saved activations | Only inside the model (its own checkpointing) | Yes, chosen by MDIR across the step |
| Fused or accelerated kernels (NequIP's Triton or cuEquivariance, `torch.export`/AOTInductor) | Yes, as a second backend (Section 5.2) | Fusion with MDIR's gathers and classical terms |
| Memory planning across the step | No | Yes |

**Order.** (1) The TorchScript backend through metatomic, the smallest
correct start, with chunking. (2) The AOTInductor backend, which gives 9 to
14 times less memory now. (3) The pair list converted from MDIR's own
structures. (4) ML4's stages. The first PR's contract already holds the
backend key, the dtype rule, the chunk key, and device outputs, so that (2)
adds a loader and no key.

## 6. Design of the adapter

### 6.1 The component and its build

- `-DMDIR_ENABLE_METATOMIC=ON` (off by default) builds `libmdir_metatomic`,
  a C++20 shared library with RTTI and exceptions, linking libtorch,
  `metatomic_torch`, and `metatensor_torch`, found through
  `Torch_DIR` and the CMake packages of the Python wheels
  (`metatomic.torch.utils.cmake_prefix_path`). Its RUNPATH names the
  libtorch of the build.
- MDIR's core reaches it only through a C header,
  `include/mdir/MLIP/Plugin.h`: versioned (`mdir_mlip_abi_version`), opaque
  handles, explicit sizes, status codes and a message, and no exception
  crossing it. The core neither includes torch headers nor links torch; a
  build without the option is the classical build of today.
- The functions, as prototyped: `load` (path, backend, device, options),
  `describe` (types, cutoff, full or half, strict, dtype, outputs,
  interaction range, locality), `evaluate` (device buffers, the stream,
  the count of pairs; outputs added into MDIR's buffers), `free`, and
  `last_error`.

### 6.2 Loading libtorch

- `mdir run` loads `libmdir_metatomic` with `dlopen` (`RTLD_LOCAL`) only
  when the control file has a model term; it is found beside the `mdir`
  binary, or at `MDIR_MLIP_PLUGIN` (a new environment variable,
  `docs/environment.md`). libtorch comes with it through its RUNPATH.
- In Python the plugin must use the libtorch of the imported `torch`:
  `mdir` imports `torch` before it loads the plugin, and the plugin
  compares the version of the loaded libtorch with that of its build and
  refuses a mismatch, so that a process never holds two. The wheel of
  #133 ships neither libtorch nor the plugin; the plugin is an optional
  build against the user's torch.
- The plugin states the versions of torch and metatomic-torch that it was
  built and tested with, and the tests pin them.

### 6.3 The control file and Python

```toml
[[energy.model]]
name = "allegro"
backend = "METATOMIC"         # later "AOTI"
path = "allegro-oam-l-0.1.pt" # relative to the control file
centers_per_call = 0          # 0: the whole system in one call
```

- **Types.** The model takes atomic numbers: from the topology (Amber's
  `ATOMIC_NUMBER`, the elements of GROMACS and CHARMM types, the element
  column of a PDB file). A particle without an element, or one whose element
  the model lacks, is an error before compilation. Virtual sites (no mass)
  are never handed to the model (#137, comment).
- **Units.** None to give: MDIR asks metatomic for nm and kJ/mol, and
  metatomic converts from the model's units (Å and eV here); the log shows
  the model's units and the factors.
- **Particles.** ML1 covers every particle with mass and has no key for a
  selection; ML/MM (a `selection` key and `selected_atoms`) is #139.
- **Output.** The term appears in the log's list of terms at the start, as
  `MODEL <name>`, in the potential energy, and in the virial.
- **Python.** `mdir.System.models`, a list of `mdir.Model(path,
  backend=..., centers_per_call=...)`, the same contract; in the second PR.

### 6.4 The pair list

The model asks for a full, strict list at its cutoff. The conversion is a
kernel of MDIR's, writing pairs (two int32), cell shifts (three int32), and
pair vectors in f64 (rounded to the model's dtype) into buffers reused from
step to step, in the index space of the positions handed over (#137,
comment, items 2 and 3).

- **First PR: a list of the model's own, with images.** The eager baseline
  confines ML1 to small systems, whose cells are often narrower than
  $2 r_c$ = 1.4 nm (the largest liquid-water box that the eager model fits
  on 24 GB, 24 molecules, is 0.89 nm wide).
  MDIR's neighbor matrix keeps the minimum image and stores no shift. So
  the first PR builds the model's list on the device every step, over a
  grid of cells with all the images within $r_c$, without skin (the list
  must be strict at every call anyway), over the state's own order. Its
  cost is measured against the model's.
- **Later: from MDIR's structures**, when the cell is wider than $2(r_c +
  \text{skin})$: the matrix (D86) already holds each pair for both
  particles, a full list; the conversion maps places to rows through the
  build's permutation and computes the shift of each pair.
- The number of pairs is needed on the host to shape the tensors: one copy
  of 8 bytes per step, the one synchronization the first PR adds.

### 6.5 Sharing the device

- MDIR's runtime uses the CUDA driver API on the primary context, which
  torch's runtime also uses; the prototype shared it without any setting.
- The term is a `mdrt.host_call` with `in_place` (D215) in the step, after
  the pair list: the host function receives the device buffers of the
  positions, the forces, the pair list, and the cell, and MDIR's stream
  (`mgpuStreamCreate`'s shared stream), hands them to the plugin, and
  returns with the model's work enqueued there. The callee is bound in the
  symbol table of `lib/Compiler/Simulation.cpp`. #135 (D215) gave
  device-resident buffers to Python simulations; `mdir run` already keeps
  its state on the device within its entry, so nothing more is needed
  from it.
- The plugin's allocator and MDIR's share the device. The first PR records
  both peaks in the log, sets `expandable_segments` unless the user set
  `PYTORCH_CUDA_ALLOC_CONF`, and measures the default against a bound.

### 6.6 Precision and dtypes

| MDIR mode | Positions | Pair vectors to the model | Forces back |
|---|---|---|---|
| `mixed` | f64 state | computed in f64, rounded to the model's dtype | the model's dtype, added into MDIR's f32 forces |
| `double` | f64 | the same | added into f64 forces |

Energy and virial are accumulated in f64. The f64-input artifact agrees
with NequIP 10 to 20 times more closely than the f32-input one (Section
3.2) at the same cost: the wrapper's default is f64 inputs.

### 6.7 Outputs and derivative ownership

- Energy, forces, and the virial tensor $\mathsf W = -\partial E/\partial
  \boldsymbol\varepsilon$ (the derivative in a strain of the positions and
  the cell), which is MDIR's convention, positive for repulsion, and
  symmetric for this model. It enters the log's virial and pressure.
- Forces are the model's derivatives, computed by the external framework
  (D167): MDIR does not differentiate the term. `Simulation.tunables`
  gradients, `[free_energy]`, `observe`, and the frame evaluator's
  gradients in the tunables refuse a run with a model term.
- A model's `non_conservative_forces` and `non_conservative_stress`
  outputs are refused in ML1.

### 6.8 Determinism

Not guaranteed with a model term: torch's kernels add with atomics in an
order that the threads decide (five evaluations differ by $2.5 \times
10^{-6}$ eV/Å). The deterministic mode (D84) refuses a model term, and the
documents say so.

### 6.9 Scope of ML1

Finite-range local models, one GPU and the CPU, a fixed cell (NVE, NVT;
NPT is refused until its virial is validated in a run), no constraints, all
particles with mass in the model, no classical terms required beside it
(they may be present and add). One device per process.

## 7. Validation

- **Against NequIP's calculator**, on the three boxes of Section 3.2 and
  one at liquid density: energy, every force component, and the virial, on
  the CPU and a GPU, in `mixed` and `double`. Tolerances from the
  reference's own spread and the input dtype: float64 inputs
  $|\Delta F| \le 2 \times 10^{-5}$ eV/Å, $|\Delta W| \le 5 \times
  10^{-5}$ eV, $|\Delta E| \le 5 \times 10^{-5}$ eV; float32 inputs ten
  times that.
- **Finite differences** of the energy in the positions and in a strain,
  with the steps of Section 3.2 and tolerances set by the float32 noise.
- **Chunks** against the whole system, 1, 16, and 64 centers per call.
- **NVE** from a minimized configuration: drift and fluctuation of the
  total energy, with a step whose size is stated.
- **Order of the input**: the same system in two orders gives the same
  energy and the same forces put back in order (#137, comment).
- **Memory and time** (RTX 4090 and RTX PRO 6000, `MDIR_GPU` stated):
  peak device memory against pairs, and time per step against the model's
  own time in Python (Section 4.1), with the pair list, the copies, and the
  synchronization separated from the model.
- **Builds**: the classical build without the option, and the suite with
  it, the tests of the model `UNSUPPORTED` when the plugin, the Python
  environment, or the downloaded package is absent.

## 8. The first implementation PR and what follows

1. **First PR (`D[metatomic-plugin]`):** the CMake option, the C++20
   plugin with the C ABI (backend `METATOMIC` only), `[[energy.model]]` in
   `mdir run` with every particle, the model's own pair list with images,
   the host call, energy, forces, and virial into the log, chunking, the
   refusals of Sections 6.7 to 6.9, the wrapper's test (built from the
   package by script when present), and the validation of Section 7.
2. The Python surface (`mdir.System.models`), with the libtorch check.
3. The AOTInductor backend (`backend = "AOTI"`), against the same
   reference, with the memory of Section 4.1.
4. The pair list from MDIR's matrix where the cell allows it.
5. NPT with a model term; then ML/MM selection and the M3 single-GPU part
   of ML4 (PaiNN, MACE).

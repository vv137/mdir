# ML1: an Allegro model through an optional metatomic adapter

Design of `D270` (issue #137) for the first deliverable of
M3, ML1 ([roadmap](roadmap.md#7-learned-potentials-m3-and-distributed-execution-m4),
D166, D167, D169): one existing Allegro artifact evaluated through an
optional metatomic adapter, with energy, per-particle forces, and virial on
one GPU agreeing with the original backend. Sections 1 to 8 are the design
and the feasibility study, measured outside the repository with the wrapper
`scripts/mlip/nequip-to-metatomic.py`; Section 9 is what the first PR built
and its validation, Section 10 minimization and constant pressure (D272),
and Section 11 the AOTInductor backend (`D273`).

## 1. What has been ruled (the maintainer, 2026-10-16)

1. **Allegro stays the first model, through a wrapper of the upstream
   checkpoint.** No Allegro model has an official metatomic export (the
   `nequip` package 0.19.1 integrates with ASE, TorchSim, and LAMMPS ML-IAP; metatrain has
   no Allegro architecture). A wrapper of a few dozen lines turns the
   model package of the `nequip` package into a metatomic artifact. It lives in the repository as
   an example of bringing a model without an official export
   (`scripts/mlip/nequip-to-metatomic.py`). Model files are never
   committed; tests build the artifact or are skipped.
2. **ML1's reference is the `nequip` package's ASE calculator
   (`NequIPCalculator`), evaluating the Allegro model**
   ([Musaelian2023](references.md#musaelian2023); the `nequip` package also
   implements the NequIP architecture, [Batzner2022](references.md#batzner2022),
   which ML1 does not use). "The `nequip` package" below is the software;
   "NequIP" is the architecture only.
3. **Memory is a primary result of ML1**, not a later concern: MDIR began
   because learned potentials use too much device memory. The eager
   baseline in bytes per pair (Section 4) is the number every later step
   is measured against.
4. **The accelerated path is part of the design now** (Section 5): the
   contract must let ML1's opaque TorchScript call be replaced by an
   accelerated one without changing the control file.
5. **The plugin is a C++20 target behind a C ABI** (Section 6.1).

On the questions of this design (the maintainer on PR #364, 2026-10-16):

6. `[[energy.model]]` with `backend`, `path`, and `centers_per_call`, as
   proposed in Section 6.3.
7. The first PR has the plugin build its own pair list on the device, with
   images and no skin; the conversion from MDIR's structures comes later.
8. The inputs follow the model's dtype whatever `[execution] precision`
   says, and the wrapper defaults to float64 inputs.
9. The order: (1) TorchScript with chunks, from the control file **and from
   Python** in the first PR, both through the plugin and its C ABI, the
   plugin using the libtorch of the imported `torch` in Python; (2)
   AOTInductor; (3) the pair list from MDIR's structures; (4) ML4's stages.
10. The deterministic mode refuses a model term;
    `torch.use_deterministic_algorithms` is not forced.
11. The first PR is GPU only; the CPU target follows.

Peak device memory against pairs stays a headline result of every step,
against the eager baseline of 1.22 MB per pair. The split between torch's
allocator and MDIR's (`expandable_segments`) is left to the implementation,
with measurements.

## 2. The model and the versions

| Item | Value |
|---|---|
| Model | `mir-group/Allegro-OAM-L:0.1`, Zenodo [doi:10.5281/zenodo.16980200](https://doi.org/10.5281/zenodo.16980200), file `Allegro-OAM-L-0.1.nequip.zip` (md5 `0db7f9b3c3a62e74d78b3fcf2973c462`, SHA-256 `3f0d3ca7bb136d4c2ee76278170fcbe756436f037e16c673a86e2c57d271c64e`, 80.7 MB) |
| License | CC-BY-4.0 (the model files); the `nequip` and `allegro` packages MIT; metatomic and metatensor BSD-3-Clause |
| Architecture | Allegro, strictly local, $r_\mathrm{max}$ = 7 Å, 4 layers, 64 tensor channels, 9.7 million parameters, float32 weights; 89 elements, H to Pu |
| Training | OMat24, fine-tuned on sAlex and MPTrj (PBE); a model for inorganic solids. Its water is not validated physics: ML1 checks the interface, not the science (D166) |
| Packaged with | nequip 0.14.0, allegro 0.7.1, torch 2.7.0 (the package carries its code) |
| Evaluated with | torch 2.14.1 (CUDA 13.0), nequip 0.19.1, nequip-allegro 0.8.3, metatomic-torch 0.1.19, metatensor-torch 0.10.8, metatomic-ase 0.1.4, vesin 0.6.2, ase 3.29.0, NumPy 2.5.4; Python 3.12 |
| Devices | RTX 4090 (24 GB, driver 595.45) and RTX PRO 6000 Blackwell (96 GB) |

The environment is a uv lock of 80 packages, kept beside the measurements;
it is `scripts/mlip/requirements.lock` (from `requirements.in`, with the
index of PyTorch for CUDA 13.0: `uv pip sync requirements.lock
--extra-index-url https://download.pytorch.org/whl/cu130
--index-strategy unsafe-best-match`).

## 3. Feasibility

### 3.1 The wrapper

`nequip-to-metatomic.py` loads the package, takes its energy network (the
module inside the `nequip` package's `ForceStressOutput`), and wraps it as a metatomic
model that asks for one pair list, full and strict at 7 Å. The engine's
pair vectors become the `nequip` package's `edge_vectors`, which it uses
as given;
the engine registers them with autograd (`register_autograd_neighbors`),
so forces and virial come from the engine's backward. The weights are
untouched. `torch.jit.script` still compiles the model under torch 2.14,
with a deprecation warning; `nequip-compile --mode torchscript` itself
refuses torch 2.10 and later, which the wrapper does not use. `--dtype`
sets the type of the inputs that the engine hands over: float64 (as
the `nequip` package's calculator does) or float32.

With `selected_atoms` the wrapper keeps only the edges whose first atom is
selected and returns the energy of those centers. Allegro is strictly
local [[Musaelian2023]](references.md#musaelian2023), so this gives the
exact energy of the selection. An engine can therefore evaluate a system in
chunks of centers (Section 4.2).

### 3.2 Agreement with the `nequip` package's calculator

Water boxes (water is chosen for MDIR's later work; this model was not
trained for it): 27 waters 5 Å apart in a cubic and in a triclinic 15 Å
cell (81 atoms, 2,830 and 2,968 directed pairs within 7 Å; $L \ge 2 r_c$),
and 8 waters at liquid density in a 6.2 Å box (24 atoms, 3,448 pairs; $L <
2 r_c$, several images of each atom). Metatomic's ASE calculator against
the `nequip` package's, one RTX 4090:

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
this check; the agreement with the `nequip` package's own autograd above is the stronger
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
buffer. Against the `nequip` package's calculator on the three cases:

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
| `nequip` package's calculator, eager (reference) | 1.41 MB | 20.2 µs | 10,546 pairs (72 atoms): 13.9 GiB, driver 22.9 GiB | 42,652 pairs (288 atoms): 56.1 GiB |
| metatomic artifact, TorchScript, whole system | 1.22 MB | 20.3 µs | 10,546 pairs (72 atoms): 12.0 GiB | 42,652 pairs (288 atoms): 48.4 GiB |
| same, 64 centers per call | flat 10.3 to 11.1 GiB | 22.7 µs | 456,424 pairs (3,072 atoms, the largest tried): 10.4 s | the same: 7.4 s |
| same, 16 centers per call | flat 2.6 to 2.9 GiB | 31.2 µs | 456,424 pairs: 14.2 s | the same: 10.9 s |
| `nequip` package, AOTInductor (`nequip-compile --mode aotinductor`) | 132 kB | 3.5 µs | 172,044 pairs (1,152 atoms): 21.2 GiB | 456,424 pairs: 56.1 GiB |
| `nequip` package, AOTInductor with `enable_TritonContracter` | 85 kB | 2.4 µs | 220,724 pairs (1,536 atoms): 17.5 GiB | 456,424 pairs: 36.1 GiB, 0.96 s |

The next larger size failed for lack of memory where the table gives a
limit (13,556 pairs for the eager paths on 24 GB, 55,674 on 96 GB; 220,724
for AOTInductor and 342,762 with Triton on 24 GB); 456,424 pairs was the
largest size tried, not a limit, for the chunks and for AOTInductor on 96
GB. The RTX PRO 6000 takes 0.67 (eager) to 0.91
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

### 5.1 What the `nequip` package provides today

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
(the `nequip` package writes `r_max`, `type_names`, and the model's dtype). Units are
the plugin's to convert, which metatomic does for its backend.

### 5.3 The contract, shaped now for both

The control file and the C ABI name what is evaluated, not how, so that a
later backend changes neither:

- **Backend.** A key selects the loader: `METATOMIC` (a TorchScript
  artifact through metatomic) in the first PR, `AOTI` (an AOTInductor
  package of the `nequip` package's `ase` target) later. Both take the same pair list
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
  others' edges). The wrapper declares this in its metadata
  (`extra["mdir::local_centers"] = "true"`); any other
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
| Fused or accelerated kernels (the `nequip` package's Triton or cuEquivariance, `torch.export`/AOTInductor) | Yes, as a second backend (Section 5.2) | Fusion with MDIR's gathers and classical terms |
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
- **Python.** `mdir.System.models`, a list of `mdir.ModelTerm(name, path,
  backend=..., centers_per_call=...)`, the same contract, in the first PR
  ([python-model.md](python-model.md#learned-potentials)).

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
with the `nequip` package 10 to 20 times more closely than the f32-input one (Section
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

- **Against the `nequip` package's calculator**, on the three boxes of Section 3.2 and
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

As ruled (Section 1, item 9):

1. **First PR (`D270`), GPU only:** the CMake option, the
   C++20 plugin with the C ABI (the TorchScript backend through metatomic
   only), `[[energy.model]]` in `mdir run` and its counterpart in the
   Python model (`mdir.System.models`, in the manner of D245 and D246),
   with the libtorch check; every particle with mass; the plugin's own
   device pair list with images; the host call; energy, forces, and virial
   into MDIR's sums, the log, and the pressure; chunking; the refusals of
   Sections 6.7 to 6.9; `MDIR_MLIP_PLUGIN` in
   [environment.md](environment.md); the wrapper's test (built from the
   package by script when present); and the validation of Section 7.
2. The AOTInductor backend (`backend = "AOTI"`), against the same
   reference, with the memory of Section 4.1.
3. The pair list from MDIR's matrix where the cell allows it.
4. ML4's stages. NPT with a model term, the CPU target, and ML/MM
   selection fit between these steps.

## 9. What the first PR built (D270)

### 9.1 The pieces

- **The plugin**, `lib/MLIP/MetatomicPlugin.cpp`, built by
  `-DMDIR_ENABLE_METATOMIC=ON` as `libmdir_metatomic.so` beside the
  runtime libraries: C++20 with RTTI and exceptions, against the CMake
  packages of torch, metatomic-torch, and metatensor-torch of a Python
  environment (`CMAKE_PREFIX_PATH`; torch's package needs the CUDA compiler,
  `CMAKE_CUDA_COMPILER`). Its C boundary is `include/mdir/MLIP/Plugin.h`,
  version 1: `load`, `describe`, `evaluate`, `memory_stats`, `free`,
  `last_error`, and the versions of its build. `evaluate` adds the
  model's forces to the engine's own buffer of forces, of `force_bits`
  (64 or 32), and writes the energy and the virial to buffers of the
  device. The core loads it with
  `dlopen` (`RTLD_LOCAL`) from the directory of the runtime or from
  `MDIR_MLIP_PLUGIN`, and sets `PYTORCH_CUDA_ALLOC_CONF` to
  `expandable_segments:True` unless it is set.
- **The pair list** is the plugin's, on the device, every step: the
  particles wrapped into the cell, then every image within the cutoff
  (along each edge as many cells as the cutoff spans of the distance between
  the planes of the other two), searched by brute force in blocks of first
  particles of at most 256 MiB, full and strict, without skin. The number of
  pairs is the one wait of the host per evaluation (one per block).
- **Precision.** The plugin takes the positions in f64 (the state of
  `mixed` and `double`), computes the pair vectors and differentiates in
  f64, and rounds the positions, the cell, and the vectors to the model's
  dtype at the boundary of the model (Section 6.6); forces, energy, and
  virial come back in f64.
- **Chunks.** With `centers_per_call` > 0 and the metadata
  `mdir::local_centers = "true"` and a finite interaction range, the plugin
  calls the model once per chunk of centers with `selected_atoms`, the full
  pair list each time, and accumulates the gradients of each backward; a
  model without the declaration takes the whole system, and the log says so.
- **The step.** After every evaluation of the forces (the start, every
  kind of step) the program calls the host in place with the positions,
  the atomic numbers (a field of the program, `model_z`, which follows the
  order of the particles), and the cell of the step, its diagonal and tilts
  (`md_exec.cell_numbers`), and gives it the forces of the other terms as a
  destination:

  ```mlir
  %f1, %mu, %mw = mdrt.host_call @mdrtModelEvaluateEnergy(%i, %x1, %p_model_z,
                      %lx, %ly, %lz, %bx, %cx, %cy, %bits, %force_bits)
                      outs(%f1c : !vec) {in_place}
                      : (...) -> (!vec, f64, vector<9xf64>)
  ```

  The call declares what it writes as the loops of md_exec do: it takes
  `%f1c` in `outs` and yields `%f1`, the forces with the model's added, so
  the kick and every other reader of the forces read a value that only the
  call produces, and no pass can schedule or fuse them before it
  (`test/Dialect/MDExec/Transforms/host-call-outs.mlir`). Storage
  assignment gives the call the buffer of `%f1c` itself, on the device, in
  the type that the forces are stored in (f32 in `mixed`, f64 in `double`),
  and the plugin adds its forces there on the stream of the kernels
  (`mdrtDeviceExternalStream`): no buffer of the model's forces and no loop
  that adds them. The op also declares that it reads and writes the state of
  the host (MLIR's memory effects), so that it is never removed or
  reordered with other calls.
- **Energy and virial** are numbers that the call yields in a step that asks
  for them (`@mdrtModelEvaluateEnergy`, steps of energy and the start): the
  host waits for the device, reads them into buffers of the host, and the
  program adds them to the potential energy and the virial of the other
  terms (`%u1 = %u1c + %mu`, `%w1 = %w1c + %mw`). The log, the pressure, the
  molecular virial (D264), and a Python simulation's energies take them from
  there; a step without energies (`@mdrtModelEvaluate`) does not wait. The
  terms at the start show `MODEL <name>`, the energy of the last evaluation.
- **Python**: `mdir.System.models` and `mdir.ModelTerm`
  ([python-model.md](python-model.md#learned-potentials));
  `Simulation.model_energies()` gives the energy and the virial tensor.
- **Refused** with a model term: the CPU target, the deterministic mode,
  (in this first PR; Section 10 lifts them) a barostat and `[minimize]`,
  `[free_energy]`, `observe`, tunables, the frame
  evaluator, more than one model, a cell that is not periodic, `single`
  precision, constraints, virtual sites, and a particle without an element
  or mass or with an element that the model does not take. A run needs a
  topology, which gives the elements; without one it is refused.

### 9.2 Validation

On RTX 4090s (`MDIR_GPU=RTX4090`), torch 2.14.1, metatomic-torch 0.1.19,
nequip 0.19.1; the artifacts made by `nequip-to-metatomic.py` from the
package of Section 2 (`--dtype float64` and `--dtype float32`); the systems
and the reference of the `nequip` package by `scripts/mlip/water-box.py`, the evaluations by
`scripts/mlip/check-model-term.py` (Python model) and `mdir run`. Units:
kJ/mol and nm. The tolerances of Section 7 in these units: float64 inputs
$\lvert\Delta E\rvert \le 4.8 \times 10^{-3}$, $\lvert\Delta F\rvert \le 1.9
\times 10^{-2}$, $\lvert\Delta W\rvert \le 4.8 \times 10^{-3}$; float32 ten
times.

**Against the `nequip` package's calculator** (largest differences over the components):

| Case | Mode | Inputs | $\lvert\Delta E\rvert$ | max $\lvert\Delta F\rvert$ | max $\lvert\Delta W\rvert$ |
|---|---|---|---|---|---|
| 81 atoms, cubic (E = -35,384, max F 549, max W 2,340) | double | float64 | $2.9 \times 10^{-4}$ | $3.1 \times 10^{-3}$ | $9.0 \times 10^{-4}$ |
| | mixed | float64 | $3.6 \times 10^{-4}$ | $2.6 \times 10^{-3}$ | $1.4 \times 10^{-3}$ |
| | double | float32 | $1.9 \times 10^{-3}$ | $8.1 \times 10^{-3}$ | $5.5 \times 10^{-3}$ |
| | mixed | float32 | $1.9 \times 10^{-3}$ | $7.6 \times 10^{-3}$ | $7.2 \times 10^{-3}$ |
| 81 atoms, triclinic (E = -35,285) | double | float64 | $1.6 \times 10^{-4}$ | $2.8 \times 10^{-3}$ | $1.7 \times 10^{-3}$ |
| | mixed | float64 | $6.5 \times 10^{-5}$ | $3.2 \times 10^{-3}$ | $1.3 \times 10^{-3}$ |
| | double | float32 | $1.3 \times 10^{-3}$ | $6.7 \times 10^{-3}$ | $6.0 \times 10^{-3}$ |
| | mixed | float32 | $1.3 \times 10^{-3}$ | $6.8 \times 10^{-3}$ | $5.0 \times 10^{-3}$ |
| 24 atoms, 6.2 Å box (E = -10,989) | double | float64 | $2.6 \times 10^{-4}$ | $1.1 \times 10^{-3}$ | $2.7 \times 10^{-4}$ |
| | mixed | float64 | $4.0 \times 10^{-4}$ | $1.3 \times 10^{-3}$ | $4.7 \times 10^{-4}$ |
| | double | float32 | $6.6 \times 10^{-4}$ | $8.1 \times 10^{-3}$ | $2.5 \times 10^{-3}$ |
| | mixed | float32 | $1.6 \times 10^{-3}$ | $8.0 \times 10^{-3}$ | $4.0 \times 10^{-3}$ |
| 72 atoms, liquid (E = -32,825) | double | float64 | $2.3 \times 10^{-5}$ | $2.2 \times 10^{-3}$ | $5.8 \times 10^{-4}$ |
| | mixed | float64 | $6.4 \times 10^{-4}$ | $1.7 \times 10^{-3}$ | $9.2 \times 10^{-4}$ |
| | double | float32 | $1.1 \times 10^{-3}$ | $8.2 \times 10^{-3}$ | $4.5 \times 10^{-3}$ |
| | mixed | float32 | $2.8 \times 10^{-3}$ | $7.6 \times 10^{-3}$ | $4.8 \times 10^{-3}$ |

Every difference is within its tolerance, the float64 ones by a factor of 3
or more. The float64 force differences, $1.1$ to $3.2 \times 10^{-3}$
kJ/mol/nm ($1.1$ to $3.3 \times 10^{-6}$ eV/Å), are the run-to-run spread
of the reference (Section 3.2). These are the numbers of the call that
declares its write of the forces (Section 9.1), measured again on
2026-10-16; in `mixed` the model's forces are now added into the forces of
f32 where they were added by a loop before, and the differences stay where
they were.

**Chunks** (liquid, 72 atoms, double, float64 inputs), against the `nequip` package:
1 center per call $\lvert\Delta E\rvert = 1.5 \times 10^{-3}$, max
$\lvert\Delta F\rvert = 2.7 \times 10^{-3}$, max $\lvert\Delta W\rvert = 1.6
\times 10^{-3}$; 16 centers $1.8 \times 10^{-3}$, $2.3 \times 10^{-3}$, $2.8
\times 10^{-4}$; 64 centers $2.7 \times 10^{-4}$, $2.2 \times 10^{-3}$, $7.8
\times 10^{-4}$. The energy of the chunks differs from that of one call by
up to $2 \times 10^{-3}$ of $3.3 \times 10^4$ kJ/mol: the float32 network's
sums in another order.

**`mdir run` against the Python model**, 81 atoms minimized with ASE's FIRE
on the `nequip` package's calculator (`water-box.py --minimize`), velocities of 300 K, 20
steps of 0.5 fs, both from the same files: the total energy of every step
agrees within $2.1 \times 10^{-4}$ kcal/mol in `double` and $2.5 \times
10^{-4}$ in `mixed`, about the four decimals that the log prints, and the
energy at the start within $2 \times 10^{-4}$ kJ/mol. Not to the bit: the
model's own sums are not deterministic (Section 6.8).

**Constant energy**, the same system, 4,000 steps of 0.5 fs (2 ps), velocity
Verlet, mean temperature 211 K: in `double` the total energy drifts by
$-0.070$ kJ/mol/ps ($-0.87$ kJ/mol/ns per atom) and fluctuates by 0.13
kJ/mol (standard deviation about the fit) against 35 kJ/mol of the
potential energy; in `mixed` $-0.068$ kJ/mol/ps ($-0.84$ kJ/mol/ns per atom)
and 0.13 kJ/mol. (Before the call declared its write, with the loop that
added the model's forces: $-0.093$ and $-0.041$ kJ/mol/ps, 0.14 and 0.13
kJ/mol; the step took 86.3 ms then and now.) From the unminimized lattice (Section 3.2) the change of
the total energy after 0.02 ps follows $\Delta t^2$: 0.435, 0.107, and 0.026
kcal/mol with steps of 0.5, 0.25, and 0.125 fs.

**Constant temperature**, the same system in `mixed`, 400 steps of 0.5 fs
toward 300 K: with stochastic velocity rescaling (`V-RESCALE`, every 10
steps, 0.1 ps) the temperature rises from 172 to 216 to 227 K over 0.2 ps
and the conserved energy stays within 0.33 kcal/mol of its start; with
Langevin dynamics (5 /ps) it rises to 229 K. Both run at 72 ms per step. A
thermostat changes nothing of the term; these runs check only that it runs
under one.

**Memory against pairs** (liquid water, double, float64 inputs; the peak
of the plugin's allocator over the run, `torch`'s count):

| Waters (atoms) | Pairs | Centers per call | Peak allocated | Reserved | Per pair |
|---|---|---|---|---|---|
| 8 (24) | 3,458 | all | 4.03 GiB | 7.2 GiB | 1.25 MB |
| 16 (48) | 6,810 | all | 7.82 GiB | 14.1 GiB | 1.23 MB |
| 24 (72) | 10,576 | all | 12.06 GiB | 21.8 GiB | 1.22 MB |
| 32 (96) | 13,602 | all | 15.49 GiB | 16.7 GiB | 1.22 MB |
| 128 (384) | 55,112 | 64 | 10.65 GiB | 19.2 GiB | |
| 1,024 (3,072) | 456,886 | 64 | 11.59 GiB | 20.9 GiB | |
| 1,024 (3,072) | 456,632 | 16 | 3.05 GiB | 5.4 GiB | |

The whole system takes 1.22 to 1.25 MB per pair, the eager baseline of
Section 4.1. With `expandable_segments` the reserved memory stays near the
allocated at the largest sizes (16.7 GiB for 15.5 GiB at 13,602 pairs),
where the study's run without it failed at 13,556 pairs on 24 GB; the
reserved memory of chunks is about twice their allocated peak.

**Rate** (the steps after the first half of the run, `mdir run`, one
RTX 4090; atom-steps per second):

| Waters (atoms) | Pairs | Centers per call | ms per step | µs per pair | Atom-steps per s |
|---|---|---|---|---|---|
| 8 (24) | 3,458 | all | 73.4 | 21.2 | 327 |
| 24 (72) | 10,576 | all | 214.6 | 20.3 | 335 |
| 32 (96) | 13,602 | all | 276.2 | 20.3 | 348 |
| 128 (384) | 55,112 | 64 | 1,146 | 20.8 | 335 |
| 1,024 (3,072) | 456,886 | 64 | 9,520 | 20.8 | 323 |
| 1,024 (3,072) | 456,632 | 16 | 10,261 | 22.5 | 299 |

The time per pair is that of the model in Python in Section 4.1 (20.3 µs
whole, 22.7 µs with 64 centers): the pair list, the copies, and the
synchronization do not show at these sizes; they were not timed apart. The
first evaluations are slower while TorchScript's profiling executor
specializes the model (1.07 s for the first steps of 81 atoms against 61 ms
later).

**Not measured in this PR**: the RTX PRO 6000; the pair list timed apart
from the model; the order of the input (the program reorders the particles
by their positions at every rebuild, which the runs above cross); the CPU
target, which ML1 refuses.

## 10. Minimization and constant pressure (D272)

The maintainer's rulings of 2026-10-16 (#367): `[minimize]` and a barostat
take a model term. Since the model's call yields its energy and virial as
values of the program (Section 9.1), the minimizer and the barostat take
them as those of any other term; no key is added.

### 10.1 What changed

- **Minimization.** Steepest descent (D73, with the rejection of a trial
  whose constraints fail, D259, and the step of at most 0.1 Å, D260) takes
  the model's energy and forces at the start and in every trial
  (`@descend`): the call follows the evaluation and the restraints there as
  in a step.
- **Constant pressure.** Stochastic cell rescaling (D72, D77) takes the
  virial of the step, the model's included, in every coupling (isotropic,
  semi-isotropic, anisotropic) and every count of the work. With
  `work = "EXACT"` the energy and forces at the scaled positions are
  evaluated with the model in the scaled cell (`%bcell`), and a continued
  run that scales every step evaluates the model at its start. The pressure
  of the barostat is that of the groups of D264; a model term takes no
  constraints, so every particle is a group of one and the pressure is the
  atomic one, $P = (2K + \mathrm{tr}\,\mathsf W)/3V$ with the kinetic energy
  of the half steps.
- **The pair list follows the cell.** The call passes the cell of the step,
  its diagonal and tilts, at every evaluation, and the plugin builds its
  pair list anew from the positions and that cell each time (every image
  within the cutoff, no skin, Section 9.1). A cell that the barostat
  changes is therefore followed at the next evaluation, with nothing kept
  from the cell before; the plugin no longer holds the cell of the input.

### 10.2 Validation

RTX 4090, the float64 artifact, the `nequip` package's ASE calculator
(`NequIPCalculator`) evaluating the same Allegro model as the reference.

**The virial against the energy under strain**
(`scripts/mlip/virial-strain.py`): the 81-atom triclinic box of Section 3.2
(edges 15 Å, tilts 1.5, -2.25, 3 Å), strained by $x' = x(\mathsf I +
\epsilon)$, $\mathsf H' = \mathsf H(\mathsf I + \epsilon)$ in each of the
nine components, rotated back to MDIR's lower-triangular frame (which leaves
the energy), and evaluated by MDIR's Python model; $W_{ab} = -\partial
E/\partial\epsilon_{ab}$ by central differences at $h = 5 \times 10^{-3}$
and $2.5 \times 10^{-3}$ combined by Richardson's extrapolation. The largest
difference over the nine components from the virial that the run reports is
0.43 kJ/mol in `double` and 0.37 kJ/mol in `mixed`, against components up
to 2,627 kJ/mol ($1.6 \times 10^{-4}$ and $1.4 \times 10^{-4}$ of it); the
differences at the two $h$ alone are 1.6 and 0.39 kJ/mol, a factor of four,
the error of order $h^2$. The remainder is the energy's own resolution:
MDIR's evaluations of one configuration spread by $2 \times 10^{-4}$
kcal/mol (Section 6.8), which divided by $2h$ is 0.1 kJ/mol. The trace of
the program's virial, which the pressure takes, equals that of the model's
to the last digit printed.

**The pressure of the barostat** (D264): in the runs below, the pressure of
the log against $(2K + \mathrm{tr}\,\mathsf W)/3V$ from its own columns of the
kinetic energy at the step, the virial, and the volume: correlation
0.9999997, root-mean-square difference 25 atm against a spread of 5,583 atm,
the difference being the kinetic energy of the half steps that D264 takes.

**Constant pressure**: 24 waters (72 atoms) at liquid density (Section 4.1),
`mixed`, velocity Verlet, 0.5 fs, V-RESCALE (0.1 ps, every 10 steps),
C-RESCALE (isotropic, 1 ps, 1 atm, Trotter), four runs of 10 ps from the
same start with seeds 1 to 4. The volume falls from 718 Å³ and its mean
over the four runs is 804, 794, 768, 750, 757, and 772 Å³ in the windows
0-1, 2-3, 3-4, 4-6, 6-8, and 8-10 ps (each $\pm$ 3 to 15 Å³). Over 4-10 ps:
$V = 760 \pm 6$ Å³ (standard error over the four runs), a density of
$0.946 \pm 0.007$ g/cm³; the pressure $176 \pm 360$ atm against 1 atm,
with an instantaneous spread of 5,600 atm, which this box and length do not
resolve better; the temperature 302.7 K. 197 ms per step. A continued run
that begins to scale every step from a checkpoint of a run that did not,
and one with `work = "EXACT"` (`model-term-gpu.test`), run with a conserved
energy within 0.1 kcal/mol over 20 to 40 steps.

**Minimization from the unminimized lattice** (the 81-atom cubic box of
Section 3.2, -35,384 kJ/mol), with the tolerance of ASE's default, a
largest force of 0.01 eV/Å (0.2306 kcal/mol/Å), in both:

| | Steps | Energy (kJ/mol) | Largest force (kcal/mol/Å) |
|---|---|---|---|
| ASE's FIRE, the `nequip` package's calculator | 4,251 | -37,881.41 | 0.2306 |
| MDIR, steepest descent, `double` | 20,000 | -37,697.08 | 1.88 |
| MDIR, the same continued | 33,000 | -37,771.1 | 0.79 (the step fell to zero at 13,000) |
| MDIR, `mixed` | 20,000 | -37,696.8 | 2.09 |
| MDIR from FIRE's minimum | 0 | -37,881.41 | 0.2305 (stops at the first row) |

The surfaces are the same: at FIRE's minimum MDIR's energy is FIRE's to
0.01 kJ/mol and its largest force is below the tolerance, and at MDIR's
minimum after 20,000 steps the `nequip` package's calculator gives
-37,697.079 kJ/mol and the same largest force. The minima differ: FIRE's
trace has the force spikes of rearrangements (8.6 and 19.7 kcal/mol/Å at
steps 3,000 and 3,500) as its energy falls past MDIR's, which a descent does
not cross. Steepest descent also stalls short of the tolerance: its
direction is the force over the mass (D73), which the hydrogens dominate
along the stiff O-H stretch. Along it, at MDIR's minimum, the calculator's
energy falls by $1.2 \times 10^{-3}$ kcal/mol at 0.001 Å where the slope
predicts $2.3 \times 10^{-3}$, by $3.7 \times 10^{-4}$ at 0.003 Å, and rises
by 0.046 at 0.01 Å; the best decrease of a trial, about $10^{-3}$ kcal/mol,
approaches the resolution of the model's energy ($2 \times 10^{-4}$
kcal/mol between evaluations of one configuration, and as much between
configurations $10^{-5}$ Å apart), so trials are rejected at random and the
step shrinks to zero. A minimization with a model therefore stops where the
energy no longer resolves the step; FIRE, with inertia, does not depend on
single trials.

**Constant energy from MDIR's own minimum** (20,000 steps of steepest
descent), velocities of 300 K, 2 ps of 0.5 fs: in `double` the total energy
drifts by $-0.042$ kJ/mol/ps ($-0.52$ kJ/mol/ns per atom) with a fluctuation
of 0.14 kJ/mol about the fit, mean temperature 266 K; in `mixed` $-0.099$
kJ/mol/ps ($-1.2$ kJ/mol/ns per atom) and 0.15 kJ/mol, 262 K.

## 11. The AOTInductor backend (D273)

The second step of Section 8 (issue #373): `backend = "AOTI"` loads an
AOTInductor package of the `nequip` package (`.nequip.pt2`) in the same
plugin, `libmdir_metatomic`, behind the same C boundary, with the same
refusals and the same call of the host that declares its write of the
forces (Section 9.1). MDIR still owns the pair list, the buffers, and the
stream. The reason is memory (Section 1, item 3): the eager TorchScript
path takes 1.22 MB per pair (Section 9.2), the package of the same Allegro
model 132 kB per pair, in about a sixth of the time (Section 4.1, measured
again in 11.5).

### 11.1 The package and its loader

`torch::inductor::AOTIModelPackageLoader` (libtorch, C++) opens the package:
it extracts the archive to the temporary directory (`TMPDIR`), loads the
compiled wrapper (`*.wrapper.so`, which links libtorch and holds the
kernels, 42 MB here) with `dlopen`, and loads the weights to the device of
`device_index`. Its `run(inputs, stream)` takes a CUDA stream: the plugin
gives it MDIR's stream (`mdrtDeviceExternalStream`), and also makes that
stream torch's current one (`CUDAStreamGuard`), since the package's
intermediates come from torch's caching allocator on the current stream. So
`memory_stats`, `expandable_segments`, and the log of the plugin's memory
(Section 6.5) hold for this backend unchanged. The wrapper is linked against
the libtorch it was compiled with; a package of another torch fails when it
is loaded, and the plugin reports the loader's message.

`get_metadata()` gives, as strings, what the plugin needs to describe the
model: `r_max` (Å), `type_names`, `model_dtype`, `nequip_aoti_inputs`,
`nequip_aoti_outputs`, `AOTI_DEVICE_KEY`, and `AOTI_COMPUTE_CAPABILITY`,
which the `nequip` package writes, and the keys of MDIR's script (11.8).

### 11.2 Inputs and outputs

The plugin takes a package of the `nequip` package's `ase` target (what
`nequip-compile --target ase` writes and `NequIPCalculator.from_compiled_model`
reads) and refuses any other inputs or outputs, which the metadata names.
The compiled wrapper checks every input's type and shape:

| Input | Shape | Type | From MDIR |
|---|---|---|---|
| `pos` | (N, 3), N ≥ 2 | float64 | the positions, in Å |
| `edge_index` | (2, E), E ≥ 2 | int64 | the plugin's pair list (Section 9.1): row 0 the center $i$, row 1 the neighbor $j$ |
| `atom_types` | (N) | int64 | the index of each particle's element in `type_names` (not the atomic number) |
| `cell` | (1, 3, 3) | float64 | the cell, rows the edges, in Å |
| `edge_cell_shift` | (E, 3) | float64 | the shift $\mathbf s$ of each pair in cells |

The package forms the pair vector as $\mathbf r_{ij} = \mathbf x_j -
\mathbf x_i + \mathbf s\,\mathsf H$, the convention of the plugin's list,
so the list goes in as it is, its indices widened to int64 and its shifts
converted to float64. The outputs, all float64:

| Output | Shape | Used as |
|---|---|---|
| `atomic_energy` | (N, 1) | the energy of each center, for chunks (11.5) |
| `total_energy` | (1, 1) | the energy of a call of the whole system |
| `forces` | (N, 3) | added into MDIR's forces |
| `stress` | (1, 3, 3) | $\sigma = (\partial E/\partial\boldsymbol\varepsilon)/V$ |

The type names must be element symbols: the plugin maps MDIR's atomic
numbers to them (a table of the elements in the plugin) and refuses a
package whose names are not elements; a particle of an element that the
package lacks is refused before compilation, as for METATOMIC (Section
6.3).

### 11.3 Forces and the virial

An exported graph cannot be differentiated from outside: the package
returns tensors without an autograd graph, so there is no backward for the
plugin to run. The derivatives are those that the `nequip` package's
`ForceStressOutput` takes inside the compiled graph: the forces
$-\partial E/\partial\mathbf x$, and the derivative in a symmetric strain
$\boldsymbol\varepsilon$ of the positions and the cell, returned as the
stress $\sigma = (\partial E/\partial\boldsymbol\varepsilon)/V$ with $V =
\lvert\det\mathsf H\rvert$. MDIR's virial (Section 6.7) is therefore
$\mathsf W = -\partial E/\partial\boldsymbol\varepsilon = -V\sigma$, which
the plugin forms in f64 on the device; it is symmetric by construction.
The forces are added to MDIR's buffer of forces in its type (f32 in
`mixed`, f64 in `double`), and the energy and the virial are written to
buffers of the device, as for METATOMIC: nothing waits for them.

### 11.4 The type of the inputs

The `nequip` package exports its models with float64 positions, cell, and
shifts whatever the model's own type (`model_dtype`, float32 for
Allegro-OAM-L), and rounds the pair vectors to the model's type inside the
graph. There is no package with float32 inputs. The rule of Section 5.3
(the inputs follow the model, never `[execution] precision`) holds: the
plugin describes the inputs as 64 bits and hands over MDIR's f64 positions
as they are; `mixed` and `double` differ only in the type of the forces
that the model's are added to. The log shows `model_dtype` beside the
inputs.

### 11.5 Chunks of centers

The package takes any number of particles and of pairs from 2 up (dynamic
shapes of `torch.export`), and the pairs need not be all of the system's.
Allegro's energy is a sum over centers, and the energy of a center depends
only on the pairs whose first particle it is
[[Musaelian2023]](references.md#musaelian2023). A call with the positions of
all N particles and the pairs of a chunk of centers therefore gives:

- the forces and the stress of the energy of that chunk's centers alone: a
  particle without pairs has the constant energy of its type, which no
  derivative sees, so the forces and the virial of the chunks add up to
  those of the system;
- in `atomic_energy`, the energies of the chunk's centers. The
  `total_energy` of such a call also counts the constant energies of every
  other particle, so the plugin sums `atomic_energy` over the chunk's
  centers instead.

The pairs of a chunk are a contiguous range of the list, which is ordered by
center. A chunk with fewer than 2 pairs (the package's lower bound) is
merged into the next one rather than dropped, since its centers still have
an energy. As for METATOMIC (Section 5.3) the plugin honors
`centers_per_call` only for a package that declares itself local: the
metadata `mdir::local_centers = "true"`, which MDIR's script (11.8) writes
for an Allegro model and not for a NequIP model, whose message passing makes
a center's energy depend on the pairs of its neighbors. A package without
the declaration (`nequip-compile`'s own) takes the whole system in one
call, and the log says so.

Measured with the package in Python (the `nequip` package's pair list; an
RTX 4090 and an RTX PRO 6000): chunks of 1, 5, and 16 centers against one
call, on the triclinic 81-atom box and 24 waters at liquid density, give
the same forces within $5.6 \times 10^{-6}$ eV/Å, the stress within $3.5
\times 10^{-8}$ eV/Å³, and the energy within $3.3 \times 10^{-5}$ eV of
366 eV (float32 sums in another order). One call of the whole system took
132 to 134 kB per pair and 3.3 µs per pair on the RTX 4090 (96 and 384
atoms, 13,606 and 55,148 pairs), and 2.6 to 2.9 µs on the RTX PRO 6000; 512
waters (1,536 atoms, about 220,000 pairs) asked the RTX 4090 for one
allocation of 13.5 GiB and failed, which is where chunks take over.

### 11.6 Devices and the package for each

A package holds compiled kernels (SASS, `.cubin`) for the device that
compiled it, without PTX: on another architecture it loads and then fails
at the first call with "no kernel image is available for execution on the
device" (a package of the RTX 4090, `sm_89`, on the RTX PRO 6000,
`sm_120`). The plugin reads `AOTI_COMPUTE_CAPABILITY` when it loads the
package, compares it with the device's, and refuses a mismatch with both
named, before any step.

One package can serve several architectures: AOTInductor's
`emit_multi_arch_kernel` builds fat binaries with SASS for each compute
capability of `TORCH_CUDA_ARCH_LIST` and PTX for the compiling device.
`AOTI_COMPUTE_CAPABILITY` still names the compiling device alone, so MDIR's
script writes the list as `mdir::archs` ("89 120"), and the plugin takes a
package on a device whose capability is in that list. A package compiled on
an RTX 4090 with SASS for 8.9 and 12.0 runs on the RTX PRO 6000 and agrees
with one compiled there; its kernels were tuned on the RTX 4090, and it took
2.80 µs per pair against 2.64 for the package compiled there (13,606 pairs).

*Ruled* (the maintainer on PR #374, 2026-10-16): one `path`, refused where
the package has no kernels for the device; the user makes one package per
kind of device, or one for several with `--arch`.

### 11.7 Units

The `nequip` package records no units: a model is in the units of its
training data, and the package's ASE calculator assumes Å and eV unless it
is told otherwise. MDIR's script writes `mdir::length_unit` and
`mdir::energy_unit` (`--units`, Å and eV by default); the plugin converts
with the table that metatomic uses for METATOMIC
(`metatomic_torch::unit_conversion_factor`). *Ruled* (the maintainer on PR
#374, 2026-10-16): a package without the keys is taken as Å and eV, and the
log says so.

### 11.8 Making the package

`scripts/mlip/nequip-to-aoti.py` runs the export of `nequip-compile --mode
aotinductor --target ase` (the `nequip` package's `aot_export_model`,
including its check of the compiled model against the eager one) and adds
MDIR's metadata (`mdir::local_centers`, `mdir::length_unit`,
`mdir::energy_unit`, and with `--arch` `mdir::archs`):

```sh
python scripts/mlip/nequip-to-aoti.py Allegro-OAM-L-0.1.nequip.zip \
    allegro.nequip.pt2 [--modifiers enable_TritonContracter] [--arch 8.9 12.0]
```

It runs on a GPU of the kind that the package is for (25 to 50 s for this
model). `--modifiers` takes the `nequip` package's accelerations
(`enable_TritonContracter`, Allegro's fused tensor product in Triton,
Section 5.1); `--arch` needs `nvcc` on the path. A package of
`nequip-compile` itself also loads, without chunks and with the units
assumed:

```sh
nequip-compile Allegro-OAM-L-0.1.nequip.zip allegro.nequip.pt2 \
    --mode aotinductor --device cuda --target ase
```

### 11.9 What changes

- **The control file and Python.** `backend` takes `"AOTI"` beside
  `"METATOMIC"` (`[[energy.model]]`, `mdir.ModelTerm(backend="AOTI")`);
  `path` names the `.nequip.pt2`; `centers_per_call` as before. No key is
  added, as Section 5.4 foresaw.
- **The plugin.** `mdir_mlip_load` dispatches on the backend; the model
  holds either the TorchScript module or the loader. `describe` gives the
  cutoff (`r_max`), a full and strict list, 64-bit inputs, the elements of
  `type_names`, the units and their factors, the locality, and the
  framework. `evaluate` builds the same pair list, calls the package once
  or per chunk, adds the forces, and writes the energy and the virial
  (11.3). The C boundary is unchanged (version 1).
- **The core.** The log names the backend in the line of the model; the
  checks of the backend accept `AOTI`. The refusals of Section 9.1 apply
  unchanged.

- **The host's processor.** The package also holds host code, which
  AOTInductor compiles with `-march=native`: a package compiled on a host
  with AVX-512 stopped with an illegal instruction when it was loaded on a
  host with AVX2 alone (two kinds of node of the same cluster). The plugin
  compares the package's `AOTI_CPU_ISA` with the host's processor and
  refuses a mismatch, and `nequip-to-aoti.py` compiles the host code for
  x86-64-v3 (AVX2) by default (`--cpu native` for the compiling host's).

### 11.10 What the backend is and is not

An AOTInductor package has no standard calling convention: its inputs and
outputs are whatever its exporter chose. The backend `AOTI` takes the
`nequip` package's convention alone (`nequip-compile --target ase`, the
inputs and outputs that its metadata names, 11.2), whatever the
architecture, NequIP or Allegro, and refuses any other. Another framework's
export (MACE's, for example) would take a thin adapter of its own for the
names and shapes of its inputs and outputs, while the parts of the plugin
that do not depend on the model stay shared: the pair list, the buffers,
the stream, the units, and the declared effects of the call of the host. In
the long run, if metatomic gains `torch.export` and AOTInductor, the
backend `METATOMIC` should take such packages under one standard; ML4's
staged execution (Section 5.4) makes MDIR depend less on any framework's
convention.

### 11.11 Validation

RTX 4090s (`MDIR_GPU=RTX4090`, `sm_89`) and an RTX PRO 6000 Blackwell
(`MDIR_GPU=RTX6000`, `sm_120`); torch 2.14.1, nequip 0.19.1, nequip-allegro
0.8.3; packages made by `nequip-to-aoti.py` on each kind of device; the
systems and references of Section 9.2 (`water-box.py`, `check-model-term.py`,
`mdir run`), in kJ/mol and nm, with the tolerances of float64 inputs there
($\lvert\Delta E\rvert \le 4.8 \times 10^{-3}$, $\lvert\Delta F\rvert \le
1.9 \times 10^{-2}$, $\lvert\Delta W\rvert \le 4.8 \times 10^{-3}$).

**Against the `nequip` package's calculator** (the eager model in Python,
on an RTX 4090), largest differences over the components:

| Case | Device | Mode | $\lvert\Delta E\rvert$ | max $\lvert\Delta F\rvert$ | max $\lvert\Delta W\rvert$ |
|---|---|---|---|---|---|
| 81 atoms, cubic | RTX 4090 | double | $2.6 \times 10^{-4}$ | $2.6 \times 10^{-3}$ | $5.1 \times 10^{-4}$ |
| | | mixed | $2.4 \times 10^{-4}$ | $2.0 \times 10^{-3}$ | $4.2 \times 10^{-4}$ |
| | RTX PRO 6000 | double | $5.6 \times 10^{-5}$ | $2.8 \times 10^{-3}$ | $1.9 \times 10^{-3}$ |
| | | mixed | $6.0 \times 10^{-4}$ | $3.2 \times 10^{-3}$ | $1.4 \times 10^{-3}$ |
| 81 atoms, triclinic | RTX 4090 | double | $3.3 \times 10^{-4}$ | $2.1 \times 10^{-3}$ | $1.2 \times 10^{-3}$ |
| | | mixed | $8.4 \times 10^{-5}$ | $2.5 \times 10^{-3}$ | $1.5 \times 10^{-3}$ |
| | RTX PRO 6000 | double | $1.1 \times 10^{-3}$ | $3.5 \times 10^{-3}$ | $3.3 \times 10^{-3}$ |
| | | mixed | $3.9 \times 10^{-4}$ | $3.7 \times 10^{-3}$ | $2.5 \times 10^{-3}$ |
| 24 atoms, 6.2 Å box | RTX 4090 | double | $1.1 \times 10^{-3}$ | $7.8 \times 10^{-4}$ | $4.7 \times 10^{-4}$ |
| | | mixed | $3.1 \times 10^{-4}$ | $1.0 \times 10^{-3}$ | $1.2 \times 10^{-4}$ |
| | RTX PRO 6000 | double | $5.3 \times 10^{-4}$ | $1.6 \times 10^{-3}$ | $2.2 \times 10^{-4}$ |
| | | mixed | $7.5 \times 10^{-4}$ | $1.7 \times 10^{-3}$ | $4.0 \times 10^{-4}$ |
| 72 atoms, liquid | RTX 4090 | double | $3.2 \times 10^{-4}$ | $1.3 \times 10^{-3}$ | $6.5 \times 10^{-4}$ |
| | | mixed | $3.7 \times 10^{-4}$ | $1.5 \times 10^{-3}$ | $6.8 \times 10^{-4}$ |
| | RTX PRO 6000 | double | $3.7 \times 10^{-4}$ | $3.4 \times 10^{-3}$ | $1.4 \times 10^{-3}$ |
| | | mixed | $9.8 \times 10^{-4}$ | $3.7 \times 10^{-3}$ | $1.2 \times 10^{-3}$ |

Every difference is within its tolerance, by a factor of 4 or more, and at
the level of the METATOMIC backend on the same systems and devices (largest
$1.5 \times 10^{-3}$, $5.2 \times 10^{-3}$, and $3.4 \times 10^{-3}$). The
package of the RTX 4090 with kernels for both (`--arch 8.9 12.0`) gives on
the RTX PRO 6000 at most $1.3 \times 10^{-3}$, $3.7 \times 10^{-3}$, and
$3.2 \times 10^{-3}$. **AOTI against METATOMIC** (RTX 4090, double, the
same four systems): at most $5.4 \times 10^{-4}$ in the energy, $4.2 \times
10^{-3}$ in the forces, and $1.5 \times 10^{-3}$ in the virial, the spread
of either against the reference. **Chunks** (liquid, double) of 1, 16, and
64 centers: at most $1.3 \times 10^{-3}$, $3.2 \times 10^{-3}$, and $1.3
\times 10^{-3}$ against the reference on both devices.

**Constant energy**, the 81-atom system of Section 9.2 minimized with ASE's
FIRE, velocities of 300 K, 4,000 steps of 0.5 fs, velocity Verlet, RTX
4090: in `double` the total energy drifts by $-0.065$ kJ/mol/ps ($-0.80$
kJ/mol/ns per atom) with a fluctuation of 0.15 kJ/mol about the fit,
against 36 kJ/mol of the potential energy, mean temperature 213 K; in
`mixed` $-0.110$ kJ/mol/ps ($-1.35$ kJ/mol/ns per atom) and 0.15 kJ/mol.
METATOMIC gave $-0.070$ and $-0.068$ kJ/mol/ps and 0.13 kJ/mol (Section
9.2). The step took 15.4 ms against 86 ms with METATOMIC.

**Memory and rate against pairs** (liquid water, `mixed`, `mdir run`; the
peak of the plugin's allocator over the run; the time of an evaluation
with `MDIR_MLIP_TIMING=1`, its pair search apart, the mean over the
evaluations after the first; the rest of the step, without classical terms
here, is not counted). The model's time per pair is flat from 13,566 pairs
up, so a rate holds across sizes:

| Package | Device | Per pair (whole system) | µs per pair | Atom-steps per s | Largest whole system | Largest tried in chunks |
|---|---|---|---|---|---|---|
| METATOMIC (Section 9.2) | RTX 4090 | 1.22 MB | 20.7 | 300 to 350 | 96 atoms, 13,598 pairs | 3,072 atoms, 64 centers: 11.6 GiB |
| AOTI | RTX 4090 | 132 kB | 3.4 | 1,840 to 2,030 | 1,152 atoms, 171,636 pairs: 21.2 GiB | 12,288 atoms, 1,766,340 pairs, 512 centers: 9.4 GiB |
| AOTI, Triton | RTX 4090 | 85 kB | 2.2 | 2,580 to 3,120 | 1,920 atoms, 282,006 pairs: 22.3 GiB | 12,288 atoms, 1,024 centers: 11.9 GiB |
| AOTI | RTX PRO 6000 | 132 kB | 3.0 | 1,980 to 2,420 | 4,608 atoms, 677,284 pairs: 83.4 GiB | 24,576 atoms, 3,611,214 pairs, 2,048 centers: 38.3 GiB |
| AOTI, Triton | RTX PRO 6000 | 85 kB | 2.0 | 2,830 to 3,680 | 7,680 atoms, 1,125,480 pairs: 89.0 GiB | 24,576 atoms, 4,096 centers: 48.7 GiB |

The next size tried failed for lack of memory on the RTX 4090 (1,344 atoms
without Triton, 2,304 with it: one allocation of the model's of up to 13.5
GiB); on the RTX PRO 6000 the largest whole systems were the largest tried.
In chunks the memory follows the chunk (512 centers: 9.4 to 9.8 GiB from
3,072 to 12,288 atoms), and the time per pair stays that of the whole
system.

Time per step and ns/day at 0.5 fs (whole systems; the largest in chunks):

| Atoms | RTX 4090 | RTX 4090, Triton | RTX PRO 6000 | RTX PRO 6000, Triton |
|---|---|---|---|---|
| 96 | 48 ms, 0.89 ns/day | 32 ms, 1.36 | 40 ms, 1.09 | 26 ms, 1.66 |
| 384 | 189 ms, 0.23 | 123 ms, 0.35 | 165 ms, 0.26 | 110 ms, 0.39 |
| 1,152 | 593 ms, 0.073 | 387 ms, 0.11 | 518 ms, 0.083 | 347 ms, 0.12 |
| 3,072 | 1.60 s (512 centers), 0.027 | | 1.41 s, 0.031 | 0.97 s, 0.045 |
| 12,288 | 6.67 s (512 centers), 0.0065 | 4.52 s (1,024 centers), 0.0096 | 5.70 s (2,048 centers), 0.0076 | |
| 24,576 | | | 12.4 s (2,048 centers), 0.0035 | 8.69 s (4,096 centers), 0.0050 |

Against METATOMIC on the same device, AOTI takes 9 times less memory per
pair and 6 times less time (14 and 9 times with Triton), as the study of
Section 4.1 found in Python. What MD with this model can do on one GPU is
thus about 2,000 to 3,700 atom-steps per second: a nanosecond a day for
about 100 atoms at 0.5 fs, a tenth of one for about 1,000.

**The pair search** (the plugin's brute force over every image, with a
wait of the host at its count, Section 9.1), with `MDIR_MLIP_TIMING=1`:

| Atoms | Search, RTX 4090 | Share, AOTI | Share, Triton | Search, RTX PRO 6000 | Share, AOTI | Share, Triton |
|---|---|---|---|---|---|---|
| 96 | 0.5 ms | 1.1% | 1.7% | 0.5 ms | 1.4% | 2.1% |
| 1,152 | 5.6 ms | 0.9% | 1.4% | 3.7 ms | 0.7% | 1.1% |
| 3,072 | 37 ms | 2.3% | | 23 ms | 1.6% | 2.4% |
| 6,144 | 141 to 148 ms | 4.5% | 6.5% | 88 ms | | 4.4% |
| 12,288 | 559 to 590 ms | 8.9% | 12.4% | 355 ms | 6.2% | |
| 24,576 | | | | 1.40 to 1.44 s | 11.3% | 16.6% |

It grows as $N^2$ (four times per doubling from 3,072 atoms) while the
model grows as $N$: below 1% of the evaluation for METATOMIC at every size
(36 ms of 9.5 s at 3,072 atoms), it reaches 9 to 17% with AOTI at 12,288
to 24,576 atoms and would pass the model's time between about 100,000 and
200,000 atoms, depending on the package and the device. The
next step (the pair list from MDIR's structures, with a skin and without
the wait) starts from these numbers; this PR does not change the search.

**A package of `nequip-compile` itself** (no MDIR keys, RTX PRO 6000,
liquid, `centers_per_call = 16` asked): the log says that the units are
taken as Å and eV and that the package is not declared local, the whole
system goes in one call, and the Python model agrees with the reference
within $1.8 \times 10^{-4}$, $3.0 \times 10^{-3}$, and $1.3 \times 10^{-3}$.

**Not measured in this PR**: the CPU target, which ML1 refuses; NPT and
minimization with this backend beyond the test of `mdir run` (D272's
validation holds for any backend, since the plugin's outputs are the same).

## 12. The pair list from MDIR's neighbor structures (D[mlip-pairlist])

The third step of Section 8 (issue #378), the common prerequisite of the
staged models (#377) and of ML3. Until this step the plugin searched its
pair list itself at every evaluation, by brute force over every image
(Section 9.1): $O(N^2)$ times the images, 9 to 17% of an AOTI evaluation at
12,288 to 24,576 atoms (Section 11.11), and a wait of the host at its
`nonzero()`; the chunks of AOTI found their boundaries with
`searchsorted(...).cpu()`, a second wait.

### 12.1 What the IR states

The list is a value of the program, derived from a neighborhood at the
model's cutoff $r_c$ (the plugin's `cutoff`, read before the program is
built, 12.6):

```mlir
%mn = md.neighborhood %x1, %cell cutoff(0.7) : !vec -> !pairs
%me = md.edges %mn, %x1, %cell cutoff(0.7)
        : !pairs, !vec -> !md.edges<@atoms>
%f1 = mdrt.host_call @mdrtModelEvaluate(%mi, %x1, %p_model_z, %me, ...)
        outs(%f1c : !vec) {in_place} : (...) -> (!vec)
```

`!md.edges<@atoms>` is the set of directed pairs of images within the
cutoff, the list a local model takes,
$E = \{(i, j, \mathbf s) : (j, \mathbf s) \ne (i, \mathbf 0),\ \lvert \mathbf x_j - \mathbf x_i + \mathbf s\,\mathsf H \rvert < r_c\}$,
with $\mathbf s \in \mathbb Z^3$ the shift in cells, grouped by the
center $i$ in the order of the particles. Each pair is in it in both
directions, and every image: the cell may be narrower than $2 r_c$, or than
$r_c$. The neighborhood is lowered as any other (D86): a neighbor matrix
built with the skin of the run, refreshed when a particle has moved half of
it (D80 under a barostat), in the matrix kind, which `md-exec-choose-neighbors`
keeps for the edges as it keeps it for the triplets (D160): its rows hold
every candidate of a particle. `md.edges` becomes `md_exec.build_edges`,
whose storage form gives the buffers of 12.3, and the host call takes them in
place. The staged models of #377 lower over the same value.

### 12.2 Images and small cells: no ghosts

A row of the matrix holds a particle $j$ once if its nearest image is within
the reach $R = r_c + \text{skin}$ at the build: the minimum image in an
orthorhombic cell, whatever its edges (the search spans
$\min(n, 2 \cdot \text{range} + 1)$ cells of each axis, so no cell is
visited twice), and in a triclinic one the build that tests every image
where the reach exceeds half the least of $a_x$, $b_y$, $c_z$
([triclinic-m2.md](triclinic-m2.md), Section 2). A pair with any image
within $R$ is therefore in the row, and `build_edges` enumerates, for each
entry, every image $\mathbf s$ within $R$ (the ranges of $s_c$, then $s_b$,
then $s_a$, of the lower-triangular cell); the images of a particle itself,
$\lvert \mathbf s\,\mathsf H \rvert < R$ with $\mathbf s \ne \mathbf 0$,
are added to its row. No ghost particles are made: the 6.2 Å cell of the
tests, narrower than the cutoff of 7 Å, takes every image this way.

### 12.3 Slots fixed at each build, the strict list each step

- **At each build** of the matrix (the refresh that builds it), a kernel
  counts for each center $i$ the candidates $(j, \mathbf s)$ with
  $\lvert \mathbf r \rvert < R\,(1 + 10^{-6})$ at the positions of the
  build (f64; the margin covers the rounding of the test of validity):
  $C_i$. The build waits for the host anyway (for the longest row); the
  counts are copied to it, their prefix sums $O_i$ give each center its
  slots $[O_i, O_{i+1})$, $C = \sum_i C_i$ in all, and a second kernel
  writes the candidates, $(j, s_a, s_b, s_c)$ in int32, into the slots in
  the same order. The candidates are decided once, so the count and the
  candidates written are the same set.
- **Each step**, `build_edges` has one thread for each particle read its
  candidates, compute $\mathbf x_j - \mathbf x_i + \mathbf s\,\mathsf H$ in f64
  at the positions and the cell of the step, and write the rows (i, j,
  $s_a$, $s_b$, $s_c$) of those within $r_c$ to the front of its slots and
  the others to the back; the number of the first, $E_i$, stays on the
  device. A pair within $r_c$ now was within $R$ at the build while the
  structure is valid (each particle within half the skin of its scaled
  reference, D80), so it is a candidate; the white paper proves it
  (Section 4.6).
- **The order** is by center by construction, $i$ ascending, which the
  chunks of AOTI rely on (Section 11.5).
- The candidates and the slots are held by the runtime
  (`mdrtEdgesCreate`, `mdrtEdgesReserve`, `mdrtEdgesBuffer`), 36 bytes a
  slot, and grow by a quarter more than asked at a build that needs more,
  as the rows of the matrix do; the counts and the offsets are buffers of
  the structure. The plugin's own search took a block of up to 256 MiB.

### 12.4 How the plugin takes the list

The model's tensors are shaped on the host, so a list whose length is known
only on the device either is read (a wait) or is passed at a bound. Both are
built, behind `MDIR_MLIP_PAIRS`:

- **`padded`**: the model takes all $C$ slots as they are. The slots after
  the strict ones are real pairs of $r_c \le \lvert \mathbf r \rvert < R$,
  each once, on which a model whose cutoff function vanishes from $r_c$ on
  (Allegro's polynomial envelope, zero for $r \ge r_{\max}$) adds nothing.
  The boundaries of the chunks are $O_i$ at multiples of the chunk, read by
  the plugin once per build (12.5). Nothing waits; the model takes about
  $(R / r_c)^3$ times the pairs.
- **`exact`**: one copy of the strict counts of the chunks ($\lceil n / c
  \rceil$ numbers, or one for the whole system) per evaluation, then the
  strict slots gathered on the device, $E = \sum_i E_i$ pairs.
- **`search`**: the plugin's own search of D270, the comparison of
  principle 3; the program then builds no list.

Measured on the four test boxes (AOTI, RTX 4090): a call with the strict
list and the same call with inert edges added (each particle with its own
image beyond the cutoff), 20% and 100% more, differ by at most $4.1 \times
10^{-6}$ eV/Å in the forces and $1.6 \times 10^{-5}$ eV of 366 eV in the
energy, against a spread of $3.3 \times 10^{-6}$ eV/Å and $9.3 \times
10^{-6}$ eV between five calls of the strict list: the float32 sums of the
model in another order.

### 12.5 The C boundary, version 2

`mdir_mlip_inputs` gains the list: the slots (device, $C \times 5$ int32),
$C$, the strict counts $E_i$ and the offsets $O_i$ (device, $n$ and $n + 1$
int32), the number of the build of the slots (host, which no other build
in the process has, so that it tells the plugin when $O$ changed, across
the activations of a Python simulation too), and the mode. A null list makes the plugin search
itself. `mdir_mlip_cutoff` (new) gives the model's cutoff from its file
without a device. `MDIR_MLIP_ABI_VERSION` is 2; a plugin of version 1 is
refused.

### 12.6 The cutoff before the program

The cutoff is an attribute of the IR, so the program needs it when it is
built, before the model is loaded on the device: `mdir run`, `mdir emit`,
and a Python simulation ask the plugin (`mdir_mlip_cutoff`): AOTI reads
`r_max` and the units from the package's metadata, METATOMIC loads the
TorchScript archive on the host and keeps it for the load.

### 12.7 What was built

- **The IR.** `!md.edges<@atoms>` and `md.edges` (MD dialect), verified:
  the cutoff at most the neighborhood's, the positions and the cell of
  the neighborhood, no excluded pairs. `md_exec.build_edges`, whose
  storage form gives the slots, the counts, the offsets, and the number of
  their build (`memref<?x5xi32, 1>`, `memref<?xi32, 1>` twice, `i64`),
  f64 positions required. `convert-md-to-md-exec` makes one from the
  other as it does the triplets; `md-exec-choose-neighbors` keeps the
  structure of the edges a matrix while the classical terms may take
  groups; `md-exec-assign-storage` refuses it on the host. An
  `mdrt.host_call` takes an `!md.edges` in place only, as four arguments of
  its callee (`mdrt::getHostArgumentCount`).
- **The lowering** (`MDExecToGPU`): an `md_exec.empty_neighbors` whose
  structure reaches a `build_edges` keeps the slots (runtime,
  `mdrtEdgesCreate`), the counts and the offsets (device), and their copies
  on the host (freed where the function ends); every build of its matrix
  then counts the candidates, reads the counts, reserves the slots
  (`mdrtEdgesReserve`, which numbers the build in the process), copies the
  offsets back, and writes the candidates (12.3); `build_edges` is the
  kernel of each step. The CPU lowering refuses the op.
- **The driver.** `Builder::emitModelTerms` emits the neighborhood and the
  edges at the model's cutoff and calls `@mdrtModelEvaluateEdges` (or
  `...EdgesEnergy`) with them; with `MDIR_MLIP_PAIRS=search` it emits the
  calls of D270 and no list. The cutoff comes from the plugin
  (`driver::getModelCutoff`, `mdir_mlip_cutoff`) in the directory that
  `mdir run` and the Python module set (`driver::setPluginDirectory`). The
  log's line of the model says how the list is taken; its count of the
  builds of neighbor structures now counts the model's structure too.
- **The plugin** takes the slots (`takePairs`): every slot, with the
  chunks at the offsets, which it reads where the number of the build
  changed; or the pairs within the cutoff, from the counts of the chunks
  (one copy) and `nonzero_static` on the device, so the order of the slots
  is kept. `MDIR_MLIP_CHECK_PAIRS` compares the pairs within the cutoff
  with its own search (`comparePairs`) at every evaluation.
- **The C boundary**, version 2 (12.5); the core refuses a plugin of
  version 1 before it looks for the functions of version 2.

### 12.8 Validation

RTX 4090 (`MDIR_GPU=RTX4090`) unless said; torch 2.14.1, nequip 0.19.1;
the AOTI package made by `nequip-to-aoti.py --arch 8.9 12.0` on an RTX
4090, the METATOMIC artifact with float64 inputs; the boxes and references
of Section 9.2; tolerances of float64 inputs there.

**The same pairs.** With `MDIR_MLIP_CHECK_PAIRS=1` the pairs within the
cutoff of MDIR's list are the plugin's search as a set at every
evaluation, in every run below: the four boxes of Section 9.2 (the 6.2 Å
cell, narrower than the cutoff, included) in `exact` and `padded`, AOTI
and METATOMIC, `mixed` and `double`, whole and in chunks of 5, 21
evaluations each (20 steps of NVE from the lattice); two tilted cells
narrower than $2r_c$ (8 waters in 6.2 Å, tilts 0.15, -0.25, 0.2 of the
edge; 27 waters in 10.5 Å, tilts 0.1, -0.15, 0.2), where a pair has
several images along tilted lattice vectors; liquid water of 3,072 and
12,288 atoms (wide cells, the matrix's minimum image) in chunks of 512,
under NVE and under NPT with stochastic cell rescaling, through the
builds that the moving particles and the scaled cell cause; and 3,072
atoms with the classical terms in groups beside the model's matrix
(`neighbor_structure = "GROUPS"`). `model-term-aoti-gpu.test` and
`model-term-gpu.test` run the check on the 6.2 Å cell, the former under
the barostat too.

**Against the `nequip` package's calculator** (largest differences over
the components and the cases; 84 cases on the four boxes, 24 on the two
tilted ones):

| Pair list | Backend | $\lvert\Delta E\rvert$ | max $\lvert\Delta F\rvert$ | max $\lvert\Delta W\rvert$ |
|---|---|---|---|---|
| exact | AOTI | $3.2 \times 10^{-3}$ | $3.3 \times 10^{-3}$ | $2.9 \times 10^{-3}$ |
| exact | METATOMIC | $8.3 \times 10^{-4}$ | $3.5 \times 10^{-3}$ | $1.3 \times 10^{-3}$ |
| padded | AOTI | $2.9 \times 10^{-3}$ | $4.0 \times 10^{-3}$ | $3.2 \times 10^{-3}$ |
| padded | METATOMIC | $2.4 \times 10^{-3}$ | $3.7 \times 10^{-3}$ | $3.0 \times 10^{-3}$ |
| search (D270) | AOTI | $3.2 \times 10^{-3}$ | $3.5 \times 10^{-3}$ | $2.6 \times 10^{-3}$ |
| search (D270) | METATOMIC | $5.4 \times 10^{-4}$ | $3.3 \times 10^{-3}$ | $1.5 \times 10^{-3}$ |
| all, tilted cells | AOTI | $2.7 \times 10^{-3}$ | $3.1 \times 10^{-3}$ | $7.3 \times 10^{-4}$ |

Every difference is within its tolerance ($4.8 \times 10^{-3}$, $1.9
\times 10^{-2}$, $4.8 \times 10^{-3}$); the largest energies are those of
chunks of 5 centers, the float32 sums of the package in another order
(Section 11.11), in every way of taking the list. The padded list gives
the model more pairs, whose zero contributions change the order of its
float32 sums: its differences are those of the others.

**Time and memory** (liquid water, AOTI, `mixed`, `mdir run`, the skin of
the tests, 0.5 Å; whole systems up to 1,152 atoms, chunks of 512 centers
from 3,072): the plugin's part of the list with `MDIR_MLIP_TIMING=1`;
MDIR's kernels of the list with `MDRT_PROFILE=1 MDRT_WAIT=1`, the kernel of
each step and the two kernels of each build per step (a build every 2 to 5
steps here); the step the mean of the second half of a run without
timing; the peak of the plugin's allocator.

| Atoms | Pairs | Search (ms) | Exact, plugin (ms) | MDIR's kernels per step (ms) | Step, search (ms) | Step, exact | Step, padded | Peak, search and exact (GiB) | Peak, padded |
|---|---|---|---|---|---|---|---|---|---|
| 96 | 13,660 | 0.51 | 0.19 | 0.13 + 0.26 | 47.7 | 47.8 | 59.4 | 1.71 | 2.09 |
| 384 | 55,628 | 0.91 | 0.21 | 0.16 + 0.26 | 192 | 192 | 230 | 6.88 | 8.27 |
| 1,152 | 171,782 | 5.6 | 0.26 | 0.16 + 0.29 | 595 | 590 | out of memory | 21.2 | |
| 3,072 | 456,554 | 37 | 0.31 | 0.17 + 0.37 | 1,610 | 1,572 | 1,889 | 9.6 | 11.6 |
| 6,144 | 900,448 | 147 | 0.36 | 0.20 + 0.36 | 3,253 | 3,109 | 3,878 | 9.5 | 11.9 |
| 12,288 | 1,766,834 | 587 | 0.47 | 0.35 + 0.35 | 6,695 | 6,095 | 7,680 | 9.4 | 11.9 |

MDIR's list costs about 1 ms a step whatever the size here, against a
search that grows as $N^2$: at 12,288 atoms the step is 9% shorter. Its
kernels have a thread for each particle, so below a few thousand
particles they take the time of one thread's candidates (0.6 ms a build),
not of the system's. The peak of the plugin's allocator is the same with
the exact list; MDIR's slots and candidates add 36 bytes a slot (80 MB at
12,288 atoms, 1.26 times the pairs), and the plugin no longer takes the
search's block of up to 256 MiB. The padded list gives the model 1.23 to 1.26
times the pairs with this skin: 20 to 26% more time and peak memory, and
the whole system of 1,152 atoms, which fits on 24 GB with the exact list,
asks for 12.6 GiB more and fails.

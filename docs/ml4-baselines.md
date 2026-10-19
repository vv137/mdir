# ML4 baselines: pretrained PaiNN and MACE through the opaque model term

Steps 1 and 2 of issue #377 (`D286`), the prerequisites of
ML4's staged models ([roadmap](roadmap.md#7-learned-potentials-m3-and-distributed-execution-m4),
D167, D169, and the milestone ruling of 2026-10-16 on #377): one pretrained
PaiNN and one pretrained MACE, each pinned, run as one whole-system call
through the existing model term (D270, D277), validated against its own
package's calculator, and measured in peak device memory and time per pair.
These are the numbers that the staged models of steps 4 and 5 are compared
with. Section 1 is the survey of artifacts and the ruling; Section 2 the
baselines: their validation, and their memory and time against size.

## 1. Artifacts

### 1.1 What the model term takes

The model term has two backends (D270, D273): `METATOMIC`, a TorchScript
archive of metatomic's interface, which the plugin differentiates in the
positions and a strain; and `AOTI`, an AOTInductor package of the `nequip`
package's `ase` target, whose inputs and outputs are that package's alone
([metatomic-ml1.md](metatomic-ml1.md#1110-what-the-backend-is-and-is-not),
Section 11.10). Neither PaiNN nor MACE has a package of that target, so the
opaque baselines use `METATOMIC`: an eager TorchScript call of the model,
with its intermediates saved by autograd for the backward. An AOTInductor
export of MACE or PaiNN would need an adapter of its own in the plugin and
is outside these baselines.

Neither package exports metatomic artifacts of these models itself, so two
wrappers in the manner of `nequip-to-metatomic.py` turn a checkpoint into
an artifact without changing its weights:

- `scripts/mlip/painn-to-metatomic.py`: a PaiNN of SchNetPack 2
  ([Schütt2021](references.md#schutt2021)) from a Lightning checkpoint; the
  engine's pair vectors become SchNetPack's `_Rij`.
- `scripts/mlip/mace-to-metatomic.py`: a model file of mace-torch
  ([Batatia2022](references.md#batatia2022)); the wrapper hands MACE the
  edges and the shifts $\mathbf s\,\mathsf H$, and MACE forms its edge
  vectors from the positions and the cell, which the plugin differentiates.
  It takes the model in float64 only, the type of its stored weights.
  metatrain 2026.4.1 has an experimental MACE architecture that also
  exports such files (by a training of zero epochs that moves the reference
  energies and the scale into its heads); the wrapper calls the file's own
  network instead.

Both declare no `mdir::local_centers`: a message-passing model's atom
energy depends on atoms up to $L\,r_c$ away ($L$ layers of cutoff $r_c$), so
the plugin evaluates the whole system in one call. Their environment is
`scripts/mlip/ml4-requirements.lock` (from `ml4-requirements.in`, with the
index of PyTorch for CUDA 13.0, as for `requirements.lock`): torch 2.14.1,
mace-torch 0.3.16 (which pins e3nn 0.4.4, so it cannot share the
environment of the `nequip` package), schnetpack 2.2.0, metatomic-torch
0.1.19, metatensor-torch 0.10.8, ase 3.29.0. The artifacts are saved with the
torch of the plugin's build.

### 1.2 The survey (2026-10-18)

**PaiNN.** No public PaiNN trained on water was found.

| Candidate | Source and pin | License | Species | $r_c$ | $L$ | Features | Outputs and units | Notes |
|---|---|---|---|---|---|---|---|---|
| **nablaDFT PaiNN, split `train_large`** (ruled; [Khrabrov2024](references.md#khrabrov2024)) | nablaDFT 2.0's `models_checkpoints.json`, key `PaiNN_train_large`: `painn_100k.ckpt` on the project's object store, 5,398,754 bytes, SHA-256 `6c81c40a99b2618ae4c0e5f96d441ece5c12bf47e1d8137cbd62883bcf77ef06` (ETag MD5 `8aff6cdaa4d4698441584ba86f6ed27e`) | MIT (the repository; the checkpoints carry no license of their own) | H, C, N, O, F, S, Cl, Br | 5 Å, cosine cutoff | 6 | 128 scalars and $3 \times 128$ vector components per atom; 100 Gaussians; filters not shared; 1.34 million parameters; float32 | energy in Hartree (formation energy from atomic references, ωB97X-D/def2-SVP), forces by autograd | SchNetPack 2 network (nablaDFT's `config/model/painn.yaml`); drug-like molecules in vacuum, no water. Lightning 2.1.4 checkpoint with `epoch` and `global_step` 0, but trained: on 60 conformations of nablaDFT's `test_2k_conformers` the energy MAE is $5.2 \times 10^{-4}$ Hartree and the force MAE $4.2 \times 10^{-4}$ Hartree/Å. Smaller splits (2k, 5k, 10k) exist with the same network |
| Open Catalyst PaiNN (OC20 S2EF) | fairchem 1's model zoo | | | 12 Å | 6 | 512 | forces predicted directly | Not conservative (direct forces), and its code left fairchem-core 2; rejected |
| SchNetPack | | MIT | | | | | | Ships no pretrained PaiNN |
| JAX PaiNN | chemtrain (Apache-2.0) implements PaiNN, trained on water (H₂O-PBE0TS) and SPICE in the chemtrain-deploy paper ([Fuchs2025](references.md#fuchs2025)) | | | | | | | No weights published that were found; InstaDeep's `mlip` (JAX) has no PaiNN |

**MACE.**

| Candidate | Source and pin | License | Species | $r_c$ | $L$ | Features | Outputs and units | Notes |
|---|---|---|---|---|---|---|---|---|
| **MACE-OFF23 medium** (ruled) | `ACEsuit/mace-off` at commit `91a78c5`, `mace_off23/MACE-OFF23_medium.model`, 18,350,596 bytes, SHA-256 `4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7` | Academic Software License (academic, non-commercial; copyleft for derived works) | H, C, N, O, F, P, S, Cl, Br, I | 5 Å | 2 | hidden 128x0e+128x1o (last layer 128x0e), spherical harmonics to $\ell = 3$, correlation order 3, 8 Bessel functions; 1.43 million parameters; stored in float64 (all 29 parameter tensors), trained in float64 by the published training script (`mace_off23/train_scripts/medium.sh` at the same commit: `--default_dtype="float64"`), and evaluated in float64 by mace-torch's `mace_off()` (its default `default_dtype`) | energy in eV, forces by autograd | SPICE (ωB97M-D3(BJ)/def2-TZVPPD), water clusters included; its paper simulates liquid water ([Kovacs2025](references.md#kovacs2025)) |
| MACE-OFF23 small | the same commit, `MACE-OFF23_small.model`, SHA-256 `165cce4c…` | ASL | the same | 4.5 Å | 2 | 96x0e (invariant messages); 0.69 million | eV | Has no $\ell = 1$ features, so step 5 would not exercise the irreducible representations |
| MACE-MP-0 medium (the MIT alternative, in the fetch list) | `ACEsuit/mace-mp` release `mace_mp_0`, `2023-12-03-mace-128-L1_epoch-199.model`, 44,422,970 bytes, SHA-256 `01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845` | MIT | 89 elements, H to Pu | 6 Å | 2 | 128x0e+128x1o, $\ell \le 3$, correlation 3, 10 Bessel; 4.69 million; stored in float64 (mace-torch's `mace_mp()` casts it to float32 by default; the wrapper takes it in float64, as every MACE here) | eV | MPtrj (PBE and PBE+U), materials; its water is not validated physics ([Batatia2025](references.md#batatia2025)) |
| MACE-MP-0 small | the same release, `2023-12-10-mace-128-L0_energy_epoch-249.model`, SHA-256 `2ddb079c…` | MIT | 89 | 6 Å | 2 | 128x0e; 3.85 million | eV | No $\ell = 1$ features |
| MACE-MPA-0, OMAT-0, MATPES, MH-1, OMOL-0 | `ACEsuit/mace-foundations` and `mace-mp` releases | MIT (MPA-0), ASL or other (the rest) | | 6 Å | 2 | medium or larger | eV | Materials or multi-head models; none adds to the two above for water |
| JAX MACE | ACEsuit's `mace-jax` imports mace-torch checkpoints (`mace-jax-from-torch`); chemtrain exports MACE-MH-1 through it with `jax.export` to StableHLO, with symbolic numbers of atoms and edges, and an optional OpenEquivariance kernel that the export carries as a custom call; InstaDeep's `mlip` 0.2.5 ([Brunken2025](references.md#brunken2025)) publishes JAX MACE, NequIP, ViSNet, and eSEN trained on SPICE2 | `mlip`'s weights: non-commercial only; `mace-jax`: no license file found | | | | | | Converted weights count as the same model only when the conversion is shown equivalent (D167); not tested here, it belongs to step 3's comparison of StableHLO with AOTInductor |

**The license of MACE-OFF23.** MACE-OFF23's model files are distributed
under the Academic Software License (ASL) of their repository: use is
allowed for academic, non-commercial purposes only, and any derived work
that is redistributed must be licensed under the ASL with its source. A
metatomic artifact made from the file by `mace-to-metatomic.py` is such a
derived work. MDIR therefore never commits or redistributes the file or an
artifact of it: a user fetches the file with `fetch-ml4-artifacts.py`,
which states the license, and makes the artifact locally, accepting the
ASL's terms (as mace-torch's own `mace_off()` asks its users to). A user who
cannot accept them takes MACE-MP-0 medium (MIT), which the same scripts
fetch and wrap.

### 1.3 The wrappers against their packages

The 81-atom water lattice of ML1 (27 waters 5 Å apart in a 15 Å cubic cell,
`water-box.py`'s configuration), metatomic's ASE calculator with the
artifact against the package's own ASE calculator, on the CPU:

| Artifact | Reference | $E$ | $\lvert\Delta E\rvert$ | max $\lvert F \rvert$, max $\lvert\Delta F\rvert$ (eV/Å) | max $\lvert W \rvert$, max $\lvert\Delta W\rvert$ (eV) |
|---|---|---|---|---|---|
| MACE-OFF23 medium, float64 | `MACECalculator`, float64 | −56,190.26 eV | $2.9 \times 10^{-11}$ | 0.642, $1.9 \times 10^{-15}$ | 9.50, $1.6 \times 10^{-14}$ |
| MACE-MP-0 medium, float64 | `MACECalculator`, float64 | −382.81 eV | $2.3 \times 10^{-13}$ | 0.270, $1.9 \times 10^{-15}$ | 3.43, $7.6 \times 10^{-15}$ |
| nablaDFT PaiNN, float32 | SchNetPack's `SpkCalculator`, float32, with a `Strain` input for the stress | −252.73 eV (−9.29 Hartree) | $8.9 \times 10^{-6}$ | 0.878, $4.2 \times 10^{-6}$ | 5.92, $1.9 \times 10^{-5}$ |

The virial is $\mathsf W = -V\sigma$ from ASE's stress. In float64 the
MACE wrapper is the package's evaluation to rounding; the PaiNN
differences are those of float32 sums in another order. SchNetPack's
network of the checkpoint has no stress output; its reference adds
SchNetPack's `Strain` module and `Forces(calc_stress=True)` around the same
modules, without changing a weight.

### 1.4 The ruling and the pinning

*Ruled* (the maintainer on PR #436, 2026-10-18): the recommended pair.

- **PaiNN: the nablaDFT PaiNN of `train_large`** (MIT). The only public
  PaiNN with conservative forces and published weights that was found. It
  was trained on drug-like molecules in vacuum: on water it checks the
  interface and the memory, not the physics, as Allegro-OAM-L did for ML1
  (D166). A PaiNN for water would need training, which needs a ruling of
  its own (D167).
- **MACE: MACE-OFF23 medium** (ASL, above). Water is in its training set and
  its paper simulates liquid water; its hidden features carry $\ell = 1$,
  which step 5 needs; float64, the precision of its published training
  script and of its package's calculator. The weights are never committed
  or redistributed; users fetch them with the script.
- **MACE-MP-0 medium** (MIT, the same feature sizes, 6 Å) stays in the fetch
  script's list as the alternative for users who cannot accept the ASL; the
  same wrapper takes it.
- **Pinning.** `scripts/mlip/fetch-ml4-artifacts.py DIR [NAME ...]` downloads
  each checkpoint from an immutable URL (a commit of a repository, a release
  asset, or an object of a store) and checks its SHA-256; a file that
  differs is removed and the script fails. Weights and artifacts are never
  committed. The tests of the model term take the downloaded directory and
  are unsupported without it, as those of ML1 are without the Allegro
  package.

## 2. The opaque baselines

Each artifact runs as one whole-system call of the model term
(`backend = "METATOMIC"`, MDIR's exact pair list of D277, the plugin's
autograd for the forces and the virial). torch 2.14.1, mace-torch 0.3.16,
schnetpack 2.2.0, metatomic-torch 0.1.19 (`ml4-requirements.lock`); the
plugin built against the same torch. RTX 4090 (`MDIR_GPU=RTX4090`, 24 GB)
and RTX PRO 6000 Blackwell (`MDIR_GPU=RTX6000`, 96 GB). Units: kJ/mol and
nm.

### 2.1 Validation against the packages' calculators

The systems, written by `water-box.py`: 8 waters in a 6.2 Å box (24 atoms,
narrower than $2 r_c$ = 10 Å, so that the list takes several images of each
atom), 27 waters 5 Å apart in a cubic and in a triclinic 15 Å cell (81
atoms), and liquid water of 64 and 256 waters (192 and 768 atoms). The
reference of each is the package's ASE calculator on the same device
(`water-box.py --kind mace`, `MACECalculator` in float64; `--kind painn`,
SchNetPack's `SpkCalculator` in float32 with its `Strain` input); MDIR's
evaluation is the Python model (`check-model-term.py`), in `double` and
`mixed`. Five evaluations of one configuration by either calculator are
identical, so the differences below are MDIR's.

Largest differences over the five systems and both devices:

| Model | Mode | $\lvert\Delta E\rvert$ | max $\lvert\Delta F\rvert$ | max $\lvert\Delta W\rvert$ | Tolerance ($E$, $F$, $W$) |
|---|---|---|---|---|---|
| MACE-OFF23 medium, float64 | double | $1.5 \times 10^{-7}$ of $5.1 \times 10^{7}$ | $5.4 \times 10^{-12}$ of 3,561 | $7.6 \times 10^{-11}$ of 12,326 | $4.8 \times 10^{-3}$, $1.9 \times 10^{-2}$, $4.8 \times 10^{-3}$ |
| | mixed | $1.5 \times 10^{-7}$ | $1.2 \times 10^{-4}$ | $7.3 \times 10^{-11}$ | the same |
| nablaDFT PaiNN, float32 | double | $4.1 \times 10^{-2}$ of $2.4 \times 10^{5}$ | $1.3 \times 10^{-1}$ of 9,424 | $7.1 \times 10^{-2}$ of 12,540 | $10^{-6}\lvert E\rvert$, $3 \times 10^{-5}\max\lvert F\rvert$, $10^{-5}\max\lvert W\rvert$ |
| | mixed | $4.1 \times 10^{-2}$ | $1.2 \times 10^{-1}$ | $7.7 \times 10^{-2}$ | the same |

The tolerances of MACE are those of ML1 for float64 inputs
([metatomic-ml1.md](metatomic-ml1.md#7-validation), Section 7). In
`double` MACE is the package's evaluation to the rounding of float64 (the
sums of its scatters in MDIR's order of the pairs); in `mixed` the
forces are added into MDIR's forces of f32, whose rounding is the
difference ($1.2 \times 10^{-4}$ of 3,561 is $3 \times 10^{-8}$ of it). PaiNN's
network is float32, and its differences are those of float32 sums in
another order; they grow with the system, so its tolerances are relative,
and every case is within them by a factor of 1.6 or more (the largest relative
differences: energy $1.7 \times 10^{-7}$, forces $1.3 \times 10^{-5}$, virial
$6.2 \times 10^{-6}$, all at 768 atoms). On the smallest box, which the test
`model-term-ml4-gpu.test` runs, the PaiNN differences are within ten times
ML1's absolute tolerances ($1.1 \times 10^{-3}$, $1.2 \times 10^{-2}$, $2.4
\times 10^{-3}$).

The reference must take the coordinates as MDIR reads them: `water-box.py`
rounded the positions to the $10^{-7}$ nm of the `.gro` file but not the
cell, and the liquid boxes, whose edges are not multiples of $10^{-7}$ nm,
then differed by up to $7.6 \times 10^{-4}$ in the forces of MACE in
`double`, where the evaluations agree to $10^{-11}$ in the same cell. The
script now rounds the cell too; the numbers above are with it.

### 2.2 Peak device memory against pairs

Liquid water (29.9 Å³ per molecule, `model-sweep.py`), from 64 waters
doubling until a size fails. MDIR: `mdir run`, `mixed`, 40 steps of NVE at
0.5 fs, the peak of the plugin's allocator over the run (torch's count),
which is the model's: the model term's own buffers (the positions, the
forces, the pair list's slots of 36 bytes, D277) are MDIR's and small
beside it. The package: its calculator's energy, forces, and stress in
Python, `torch.cuda.max_memory_allocated` above the model's weights.
"Per pair" is the peak over the directed pairs within $r_c$.

| Model | Device | Per pair, MDIR | Per pair, package | Largest that ran (MDIR) | Next size |
|---|---|---|---|---|---|
| MACE-OFF23 medium, float64 | RTX 4090 | 163 to 168 kB | 160 to 163 kB | 1,536 atoms, 83,422 pairs: 12.7 GiB | 3,072 atoms fail in both (one allocation of 6.22 GiB) |
| | RTX PRO 6000 | 163 to 172 kB | 160 to 167 kB | 6,144 atoms, 330,250 pairs: 50.2 GiB | 12,288 atoms fail in both (25.4 GiB at once) |
| nablaDFT PaiNN, float32 | RTX 4090 | 61 to 63 kB | 62 to 65 kB | 6,144 atoms, 330,176 pairs: 18.8 GiB | 12,288 fail in both |
| | RTX PRO 6000 | 61 to 68 kB | 61.5 to 70 kB | 24,576 atoms, 1,313,206 pairs: 74.8 GiB | 49,152 fail in both |

The peak grows in proportion to the pairs on both devices (the highest
bytes per pair are those of the smallest box, where the weights and the
fixed workspace weigh most), and the opaque call takes what the package
takes, within 3%: MDIR adds nothing to the model's memory, and saves
nothing either. The limit is the same in both, a size apart on each
device. MACE's failing allocation is one tensor of $4.1 \times 10^4$ bytes
per pair in float64 (5,120 values; 6.22 GiB at 163,128 pairs, 25.4 GiB at
665,312): a per-pair tensor of the interaction,
before its sum over the neighbors, which the opaque call materializes for
every pair at once. Where its bytes go in detail was not traced.

Against ML1's Allegro-OAM-L (Section 4.1 of metatomic-ml1.md), the eager
baselines per pair are: Allegro 1.22 MB (TorchScript) and 132 kB
(AOTInductor); MACE-OFF23 medium 164 kB; PaiNN 61 kB. A message-passing model cannot be cut into chunks of centers
as Allegro can (its atom energies depend on atoms up to $L\,r_c$ away), so
these per-pair peaks bound the system on one GPU: about 1,500 atoms of
water for MACE on 24 GB, 6,000 on 96 GB; 6,000 and 24,000 for PaiNN.

### 2.3 Time

The time per step of the second half of each run of `mdir run` (without
the first evaluations, which TorchScript's profiling executor specializes);
the package's shortest of three evaluations after a warm-up. The neighbor
structures of these runs, started hot from an unrelaxed liquid with the
0.5 Å skin of the tests, were rebuilt every step; MDIR's list costs about
1 ms a step (D277), inside the step times below.

| Model | Device | µs per pair, MDIR (step) | µs per pair, package | ms per step, MDIR: 192 atoms; 1,536; the largest | Atom-steps per s at 1,536 atoms |
|---|---|---|---|---|---|
| MACE-OFF23 medium, float64 | RTX 4090 | 11.5 to 12.3 | 11.5 to 12.2 | 124; 962 | 1,600 |
| | RTX PRO 6000 | 9.0 to 9.8 | 9.4 to 10.9 | 99; 747; 3,012 at 6,144 | 2,060 |
| nablaDFT PaiNN, float32 | RTX 4090 | 0.50 to 0.79 | 0.71 to 0.88 (2.7 at 192 atoms) | 8.2; 44.4; 167 at 6,144 | 34,600 |
| | RTX PRO 6000 | 0.32 to 0.79 | 0.50 to 0.62 (2.1 at 192 atoms) | 8.3; 28.5; 419 at 24,576 | 53,900 |

At 0.5 fs, a step of 1,536 atoms is 0.045 ns/day for MACE on the RTX 4090
(0.058 on the RTX PRO 6000) and 0.97 (1.5) for PaiNN. The opaque call runs
at the package's speed: the time per pair of MDIR's step and of the
package's evaluation agree within the scatter of the measurements for
MACE, and MDIR's step is shorter than the package's evaluation for PaiNN
(the package builds its neighbor list on the host and copies it each
time). MACE runs in float64, of which both devices have little throughput.
The time per pair is flat from about 40,000 pairs
up; below, the fixed cost of a call (a few milliseconds) dominates.

### 2.4 What the baselines fix for steps 3 to 5

- **Memory is the bound.** On one GPU, the largest whole system of each
  model is set by one per-pair intermediate that the eager call holds for
  every pair at once; MDIR's opaque call reproduces the package's peak to
  3%. Steps 4 and 5 are measured against the per-pair peaks of Section 2.2
  at the same sizes and devices, with the same artifacts.
- **Chunks of centers do not apply.** The artifacts declare no
  `mdir::local_centers`: with $L$ layers, the energy of an atom depends on
  its neighbors' features of the layer before, which depend on theirs, so
  by induction on the layers on every atom within $L\,r_c$ (30 Å for this
  PaiNN, 10 Å for MACE-OFF23). The opaque route to larger systems is
  therefore a halo of $L\,r_c$ around each chunk, which recomputes most of
  the system; ML4's stages make the edge stage of each layer a loop over
  pairs that MDIR can tile, store, or recompute.
- **PaiNN in stages** (step 4a, D[ml4-painn-stages]) is measured against
  these rows in Section 8.6 of [ml4-ir.md](ml4-ir.md#86-peak-device-memory-and-time):
  652 MiB at 6,144 atoms against the opaque call's 18.7 GiB, and 196,608
  atoms on the same 24 GB.
- **The references stand.** Both wrappers reproduce their packages (MACE to
  $10^{-11}$ in float64, PaiNN at float32's rounding), so a staged model is
  compared with the opaque call, and through it with the package.

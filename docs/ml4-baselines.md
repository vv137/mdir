# ML4 baselines: pretrained PaiNN and MACE through the opaque model term

Steps 1 and 2 of issue #377 (`D[ml4-baselines]`), the prerequisites of
ML4's staged models ([roadmap](roadmap.md#7-learned-potentials-m3-and-distributed-execution-m4),
D167, D169, and the milestone ruling of 2026-10-16 on #377): one pretrained
PaiNN and one pretrained MACE, each pinned, run as one whole-system call
through the existing model term (D270, D277), validated against its own
package's calculator, and measured in peak device memory and time per pair.
These are the numbers that the staged models of steps 4 and 5 are compared
with. Section 1 is the survey of artifacts and the recommendation; the
baselines follow when the artifacts are ruled.

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
| **MACE-OFF23 medium** (ruled) | `ACEsuit/mace-off` at commit `91a78c5`, `mace_off23/MACE-OFF23_medium.model`, 18,350,596 bytes, SHA-256 `4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7` | Academic Software License (academic, non-commercial; copyleft for derived works) | H, C, N, O, F, P, S, Cl, Br, I | 5 Å | 2 | hidden 128x0e+128x1o (last layer 128x0e), spherical harmonics to $\ell = 3$, correlation order 3, 8 Bessel functions; 1.43 million parameters; trained and stored in float64 | energy in eV, forces by autograd | SPICE (ωB97M-D3(BJ)/def2-TZVPPD), water clusters included; its paper simulates liquid water ([Kovacs2025](references.md#kovacs2025)) |
| MACE-OFF23 small | the same commit, `MACE-OFF23_small.model`, SHA-256 `165cce4c…` | ASL | the same | 4.5 Å | 2 | 96x0e (invariant messages); 0.69 million | eV | Has no $\ell = 1$ features, so step 5 would not exercise the irreducible representations |
| MACE-MP-0 medium (the MIT alternative, in the fetch list) | `ACEsuit/mace-mp` release `mace_mp_0`, `2023-12-03-mace-128-L1_epoch-199.model`, 44,422,970 bytes, SHA-256 `01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845` | MIT | 89 elements, H to Pu | 6 Å | 2 | 128x0e+128x1o, $\ell \le 3$, correlation 3, 10 Bessel; 4.69 million; stored in float64 (mace-torch's calculator casts it to float32 by default) | eV | MPtrj (PBE and PBE+U), materials; its water is not validated physics ([Batatia2025](references.md#batatia2025)) |
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
| MACE-MP-0 medium, float32 | `MACECalculator`, float64 | | $2.8 \times 10^{-4}$ | $9.5 \times 10^{-7}$ | $4.3 \times 10^{-6}$ |
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
  which step 5 needs; float64 as trained. The weights are never committed
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

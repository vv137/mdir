# MDIR

MDIR is a compiler for molecular dynamics, built on MLIR. It compiles each
run before it runs, specialized to the system, its parameters, and the
target: the potential, the integrator, the constraints, and the couplings
are written in dialects of molecular dynamics, differentiated, fused,
given a precision by role, and lowered to the CPU with OpenMP or to NVIDIA
GPUs. It runs Amber (`prmtop`), GROMACS (`top`, `itp`, `gro`), and CHARMM
(`psf`, `crd`, `rtf`, `prm`, `str`) inputs, or a potential written in a
control file, from the `mdir` command or from Python. Forces come from
differentiating the energy in the IR, so the same machinery gives
derivatives in parameters; bitwise determinism is a mode a run can choose;
and the passes rely only on facts that the IR states and checks.
[docs/overview.md](docs/overview.md) describes the design in brief, with
diagrams, and the white paper of the first milestone is in
[docs/paper/](docs/paper/README.md).

## Status

| Milestone | Scope | State |
|---|---|---|
| M1 | An all-atom protein in water with an Amber force field, on one node, CPU and GPU (the full definition, with its rate target, is in [docs/roadmap.md](docs/roadmap.md)) | Works today: bonded terms with CMAP and Urey–Bradley angles, Lennard-Jones with the dispersion correction or CHARMM's force switch, particle mesh Ewald, SHAKE and SETTLE, virtual sites, restraints, thermostats and barostats, minimization, in single, mixed, or double precision |
| M2a | The Python API | Works today |
| M2b | Differentiable simulation | In progress: tunable parameters, their gradient, and a frame evaluator with a PyTorch operation work today; a JAX adapter is planned |
| M3 | Learned potentials on one GPU | In progress: metatomic (TorchScript) and AOTInductor models through an optional plugin, and PaiNN in neural stages of the IR |
| M4 | Distributed execution | In progress: the `md_dist` dialect, its verifier, and an in-process executor; no run takes it yet |

[docs/roadmap.md](docs/roadmap.md) has the details and what comes next.

## Quickstart

[Install MDIR](#installing), then, from the root of the source tree, check
the CPU installation and run the supplied liquid argon example:

```sh
export PATH="$HOME/opt/mdir/bin:$PATH"   # where MDIR is installed
mdir doctor --target=cpu
mkdir argon-run
cp examples/argon/argon.toml examples/argon/argon.pdb argon-run/
mdir check argon-run/argon.toml
mdir run argon-run/argon.toml > argon-run/argon.log
```

The example runs 864 atoms for 2000 steps and writes its energy log and
`argon.dcd` trajectory in `argon-run/`. `mdir check` reports the planned
run and warnings before execution. Run `mdir doctor` to check every built
target, including the CUDA driver, visible devices, and a short GPU run.
Use `mdir doctor --target=gpu` to check just the GPU.

The same can be done from Python with the package `mdir`. A short run of
tri-alanine in water from its Amber files, in nm, ps, kJ/mol, and K:

```python
import mdir

loaded = mdir.load_amber("examples/ala3/ala3.prmtop", "examples/ala3/ala3.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
integrator.timestep = 0.002
ensemble.kind, ensemble.temperature = mdir.EnsembleKind.NVT, 300.0
execution, schedule = mdir.Execution(), mdir.Schedule()
execution.target = mdir.Target.CPU               # or mdir.Target.GPU
schedule.steps = 1000

sim = mdir.Simulation(mdir.compile(system, state, integrator, ensemble,
                                   execution, schedule))
sim.reporters.append(mdir.EnergyReporter("ala3.dat", 100))
sim.run(1000)
positions = sim.state().positions                # (N, 3) in nm, in input order
```

A whole protocol, minimization, heating at constant volume, equilibration
at constant pressure with restraints, then production, is
`examples/ala3/run.py` in Python and `examples/ala3/*.toml` as control
files.

## Performance

With the inputs, steps, and output of the Amber GPU benchmark suite, on
one RTX 4090 in mixed precision, MDIR runs the nine runs of the suite at
83.4 ns/day on STMV (1,067,095 atoms, 4 fs) to 2406.1 ns/day on JAC at
4 fs (23,558 atoms; 1276.8 ns/day at 2 fs) with the searched reaches, and
at 79.0 to 2311.5 ns/day with the default reaches (white paper,
Table 10.2); outside its steps it spends 7.9 to 27.9 s of a run (D327).
On the CPU, at constant energy with a step of 2 fs on
one 16-core node, MDIR in mixed precision at 16 threads takes 9.3 ms a
step on JAC, 129.5 ms on a TIP3P water box of about 103k atoms, and 180.3 ms on
Cellulose (408,609 atoms; white paper, Table 10.8). Both are in
[Section 10 of the white paper](docs/paper/10-performance.md).

## Installing

The current release is [v0.2.0](docs/release-notes/v0.2.0.md), a source
release: it has no binary tarball and no wheel. Its changes are in
[CHANGELOG.md](CHANGELOG.md).

- **From a release.** Download the source archive from
  [GitHub Releases](https://github.com/vv137/mdir/releases) and build it
  with LLVM 23.1.2 and HDF5 as [docs/building.md](docs/building.md)
  describes; the installed tree, `bin/mdir` with its runtime, needs
  neither build tree.
- **In a container.** [packaging/](packaging/README.md) builds a Docker
  or an Apptainer image with LLVM, HDF5, CUDA, and MDIR:

  ```sh
  docker build -f packaging/Dockerfile -t mdir:0.2.0 .
  docker run --rm --gpus all -v "$PWD":/work mdir:0.2.0 doctor
  apptainer build mdir-0.2.0.sif docker-daemon://mdir:0.2.0
  apptainer run --nv mdir-0.2.0.sif doctor
  ```

- **The Python package.** No wheel is published with 0.2.0. A wheel for
  CPython 3.10–3.13 on Linux x86-64 is built from the source tree, with
  `pip wheel .` or the target `wheels` of `packaging/Dockerfile.manylinux`
  for manylinux_2_28, as [docs/python-package.md](docs/python-package.md),
  Section 6, describes; its extra `cuda` brings cuFFT and ptxas. The
  `mdir` command is not in the wheel. A wheel built that way installs
  with:

  ```sh
  python -m venv mdir-env
  mdir-env/bin/pip install "dist/mdir-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl[cuda]"
  mdir-env/bin/python -c "import mdir; print(mdir.__version__)"
  ```

Building from source, the CMake options, the tests, and `mdir-opt` are in
[docs/building.md](docs/building.md).

## Using MDIR

**Running.** `mdir run` reads a TOML control file, in Å, kcal/mol, and ps,
compiles the run, and executes it; `[execution]` selects the target, `cpu`
or `gpu`, the threads, and the precision. `mdir template md` prints a
control file with every keyword, and `mdir template amber` one for an
Amber topology. The driver and its keys are in
[docs/driver.md](docs/driver.md) and
[Appendix A of the white paper](docs/paper/A-control-file.md).

| Command | Does |
|---|---|
| `mdir run <control file> [--continue [--no-append]] [--max-walltime=<time>]` | Compiles the run and executes it; `--continue` goes on from its checkpoint until it is complete, and a run stops at a checkpoint on SIGTERM or at the wall time; a run that is not continued keeps the outputs of an earlier one as `#<name>.<n>#` with the exit status 75 ([docs/driver.md](docs/driver.md), Section 2.7) |
| `mdir emit <control file> [--stage=module\|lowered\|pipeline]` | Prints the program of the run, as it is built or as it is executed, or the passes between |
| `mdir check <control file> [--json]` | Reads the input and reports the system, planned run, output paths and intervals, and preflight warnings, without compiling or writing ([the preflight reference](docs/driver.md#14-preflight)) |
| `mdir template md\|amber` | Prints a control file with every keyword, for terms in the control file or for an Amber topology |
| `mdir template minimize\|nvt\|npt\|production` | Prints one stage of the standard pipeline, with checkpoints connecting the stages ([docs/driver.md](docs/driver.md#13-standard-pipeline-templates)) |
| `mdir checkpoint <file> [<file>]` | Describes a checkpoint, or compares the states of two |
| `mdir doctor [--target=cpu\|gpu]` | Checks the installation by compiling and running a small system |
| `mdir version` | Prints the version, the commit, and what the build supports |
| `mdir bug-report <control file> [--run]` | Collects what a report of a defect needs ([docs/debugging.md](docs/debugging.md)) |

**Examples.** The molecular examples run minimization, heating at
constant volume, equilibration at constant pressure with restraints, then
production; each example's README has its stages and settings
([Appendix C of the white paper](docs/paper/C-usage.md) walks through
inputs, equilibration, checkpoints, and diagnosis):

| Example | System | Shows |
|---|---|---|
| [examples/ala3](examples/ala3) | Tri-alanine with ff19SB in OPC water, 4,905 particles | A whole protocol from a topology of tleap, with PME, SETTLE, SHAKE, and the virtual sites of OPC, as control files and in Python |
| [examples/ubiquitin](examples/ubiquitin) | Ubiquitin (1UBQ) with ff19SB in OPC water, 26,031 particles | A tutorial from the PDB file, with 10 ns of production on the groups and the dual list |
| [examples/popc-bilayer](examples/popc-bilayer) | A bilayer of 126 POPC lipids of Lipid21 in TIP3P water, 31,680 atoms | A tutorial from nothing, with 30 ns at semi-isotropic constant pressure and the area per lipid |
| [examples/argon](examples/argon) | Liquid argon, 864 atoms | A run from a PDB file with the terms in the control file, and the same system as a module written by hand ([docs/building.md](docs/building.md#71-the-argon-example-by-hand)) |

**Python.** The package `mdir` loads the same inputs, compiles them with
the same model and IR builder as the control file, and runs a
`Simulation` that persists across calls, with reporters, checkpoints that
`mdir run` reads and writes alike, DLPack views of its buffers, tunable
parameters, and a frame evaluator
([docs/python-package.md](docs/python-package.md); the index of the Python
documents is [docs/python.md](docs/python.md)).

**Tools.** `mdir-opt` runs the passes of MDIR on IR, as `mlir-opt` does,
for development ([docs/building.md](docs/building.md#7-tools)).

## Documents

| Reader | Documents |
|---|---|
| Users | [docs/driver.md](docs/driver.md) (the driver and the control file); [docs/python-package.md](docs/python-package.md) and [docs/python.md](docs/python.md) (the Python package and its documents); [examples/](examples/); [Appendix C](docs/paper/C-usage.md) (usage); [docs/environment.md](docs/environment.md) (environment variables); [docs/debugging.md](docs/debugging.md) (when something fails) |
| Design | [docs/overview.md](docs/overview.md) (start here); [docs/execution-contracts.md](docs/execution-contracts.md); [docs/primitives-map.md](docs/primitives-map.md) and [docs/device-primitives.md](docs/device-primitives.md); [docs/ops.md](docs/ops.md); topics: [neighbors](docs/neighbors.md), [tuples](docs/tuples.md), [nonbonded](docs/nonbonded.md), [PME](docs/pme.md), [constraints](docs/constraints.md), [dynamics](docs/dynamics.md), [CHARMM](docs/charmm.md), [learned potentials](docs/metatomic-ml1.md) and [their IR](docs/ml4-ir.md), [distributed execution](docs/distributed-ml2.md); [docs/validation.md](docs/validation.md); [docs/prior-art.md](docs/prior-art.md); [docs/architecture.md](docs/architecture.md) (the first design, and where it went) |
| Development | [CONTRIBUTING.md](CONTRIBUTING.md); [docs/building.md](docs/building.md); [docs/workflow.md](docs/workflow.md); [docs/principles.md](docs/principles.md); [docs/decisions.md](docs/decisions.md); [docs/roadmap.md](docs/roadmap.md); [CHANGELOG.md](CHANGELOG.md) and [docs/release-notes/](docs/release-notes/v0.1.0.md) |
| The white paper | [docs/paper/](docs/paper/README.md): the design, the methods with their derivations, correctness, and performance (`scripts/paper/build-pdf.sh` builds the PDF) |

## License and citing

MIT. See [LICENSE](LICENSE). To cite MDIR, cite its white paper: Minsoo
Kim, *MDIR: A Compiler for Molecular Dynamics*, white paper of the first
milestone, October 2026 ([docs/paper/](docs/paper/README.md)).

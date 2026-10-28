# The Amber GPU benchmark suite

`bench.py` runs the explicit-solvent systems of the benchmark suite of
Amber (ambermd.org/GPUPerformance.php) with MDIR and with GROMACS, with the
settings of their Amber inputs, and tabulates the rates in ns/day beside
those that ambermd.org reports for pmemd.cuda.

The inputs are not part of MDIR. `bench.py prepare` downloads the suite
(`Amber24_Benchmark_Suite.tar.gz`) into a work directory and converts each
system for GROMACS with ParmEd. MDIR and pmemd read the files of the suite
as they are, `prmtop` and `inpcrd` (D133): the JAC and Factor IX
topologies are in the format before Amber 7, and Factor IX gives 2283
dihedral terms by the convention of a negative periodicity, which a
topology that ParmEd writes again loses. The conversion for GROMACS makes
those terms explicit first.

The topologies of the suite carry no atomic numbers, and ParmEd would take
the elements from the masses. In the systems with repartitioned masses
that gives helium for a hydrogen of 3.024 amu and boron for a carbon of
9.994 amu (#469). The conversion takes each atom's element from its type
instead (`TYPE_ELEMENTS` in `bench.py`), and refuses an atom whose mass,
with the repartitioned mass of its hydrogens given back, is not that of the
element. The elements reach `[ atomtypes ]` of the GROMACS topology and
its SETTLE of rigid water. After converting, `prepare` checks
`[ atomtypes ]` of every system, also of those converted before, and names
a topology to convert again.

`scripts/validation/suite/run.py` compares the energy terms at the start
and the forces of every system with pmemd on the CPU (or sander).

## Systems

| Name | Directory of the suite | Atoms | Ensemble | Step |
|---|---|---|---|---|
| `jac_nve`, `jac_npt` | `JAC_production_*` (dihydrofolate reductase) | 23,558 | NVE, NPT | 2 fs |
| `jac_nve_4fs`, `jac_npt_4fs` | `JAC_production_*_4fs`, masses of hydrogen repartitioned | 23,558 | NVE, NPT | 4 fs |
| `factorix_nve`, `factorix_npt` | `FactorIX_production_*` | 90,906 | NVE, NPT | 2 fs |
| `cellulose_nve`, `cellulose_npt` | `Cellulose_production_*` | 408,609 | NVE, NPT | 2 fs |
| `stmv_npt_4fs` | `STMV_production_NPT_4fs` (satellite tobacco mosaic virus) | 1,067,095 | NPT | 4 fs |

The systems in implicit solvent (`GB/`) are left out; MDIR has no
generalized Born model.

## Settings

As in `mdin.GPU` of each system:

| Item | Amber input | MDIR | GROMACS |
|---|---|---|---|
| Cutoff | `cut=8.` | `cutoff = 8` | `rcoulomb = rvdw = 0.8` |
| Neighbor lists | skin of 2 Å (pmemd) | `pairlist_distance = 10` by default; `--skin`, `--prune-skin`, or `"AUTO"` | Verlet buffer from its tolerance |
| Particle mesh Ewald | `dsum_tol` 1e-6 (NVE), 1e-5 (NPT): $\beta$ from $\mathrm{erfc}(\beta r_c)/r_c$ = `dsum_tol`, 0.39467 and 0.34864 Å⁻¹; pmemd.cuda's grid, order 4 | the same `beta` and `grid`, order 4, `influence = "OPTIMAL"` (D134) | `ewald-rtol` = $\mathrm{erfc}(\beta r_c)$ of the same $\beta$, `fourier-nx`, `-ny`, `-nz` of the same grid, order 4; `-notunepme` |
| Dispersion | correction of energy and pressure | the same | `DispCorr = EnerPres` |
| Constraints | SHAKE of the bonds of hydrogen, rigid water | SHAKE, RATTLE, SETTLE | LINCS, SETTLE |
| NVE | velocities of the restart file | the same | `continuation = yes` |
| NPT | Berendsen thermostat, `tautp=10`; Monte Carlo barostat | Bussi thermostat, 1 ps; stochastic cell rescaling, 2 ps | `v-rescale`, 1 ps; `c-rescale`, 2 ps |
| Coupling interval | Monte Carlo barostat every 100 steps (`mcbarint`) | Both every 25 steps | `nsttcouple = 100`, `nstpcouple = 25`, its own choices for these inputs |

The grids are those that pmemd.cuda takes (`grid` of `SYSTEMS` in
`bench.py`): 64³ for JAC, 144 × 84 × 80 for Factor IX, 256 × 128 × 128
for Cellulose (where MDIR's own rule would take 270 × 126 × 126), and
224 × 224 × 240 for STMV. Until D134 MDIR and GROMACS took $\beta$ from
$\mathrm{erfc}(\beta r_c)$ = `dsum_tol`, larger than Amber's, and
grids of their own.

MDIR runs on a GPU in mixed precision. GROMACS runs with the nonbonded
forces, PME, the bonded forces, and the update on the GPU, 8 OpenMP
threads, and its counters reset halfway (`-resethway`).

## Use

```sh
bench.py --work DIR prepare --parmed-python PYTHON_WITH_PARMED
bench.py --work DIR run mdir [SYSTEM ...] --mdir PATH_TO_MDIR
bench.py --work DIR run mdir --mdir PATH_TO_MDIR --neighbor-structure GROUPS \
    --skin 3 --prune-skin 0.6        # a dual list, D114: the settings of the suite since
bench.py --work DIR run mdir --mdir PATH_TO_MDIR --skin AUTO
                                     # the reaches chosen by the run (D281)
bench.py --work DIR run gromacs [SYSTEM ...] --gmx PATH_TO_GMX
bench.py --work DIR run mdir --mdir PATH_TO_MDIR --skin AUTO --suite-protocol \
    [--energies-only] [--record FILE --variant NAME --repeat N]
bench.py --work DIR run pmemd --pmemd PATH_TO_PMEMD_CUDA --suite-protocol \
    [--energies-only] [--record FILE --variant NAME --repeat N]
                                     # the protocol of mdin.GPU for both
bench.py --work DIR report
```

The work directory defaults to `$MDIR_BENCH_DIR`, or `~/mdir-benchmarks`.
Each system has a directory there with the converted inputs, the control
file and the `.mdp` that a run writes, and the logs; `results.json` keeps
the rates.

`--skin AUTO` writes `pairlist_distance = "AUTO"` and, unless
`--prune-skin` gives the inner skin, `pruned_distance = "AUTO"`: what a
control file without the two keys means (D281,
[neighbors.md](../../../docs/neighbors.md#6-the-choice-of-the-skins)),
except that the log then states the reaches and their source. A run
searches for its skins only when it is long (at least $5 \times 10^{12}$
particles times steps), far beyond the suite's runs, and otherwise takes
the choice of an earlier run of the same key from the cache of the skins
(`MDIR_SKIN_CACHE_DIR`), else the defaults of D261. To time the reaches a
long run would choose, run each system once with `MDIR_SKIN_SEARCH=always`,
which measures the candidates and caches its choice, and time the runs
after it, which take the cache; `scripts/paper/run-suite-repeats.sh` does
this for the white paper (Section 10). `--prune-skin AUTO` with a number
for `--skin` leaves the inner list alone to the run.

## The protocol of the suite's inputs

Without `--suite-protocol`, MDIR runs `steps` of `SYSTEMS` (JAC 10,000,
Factor IX 4000, Cellulose 1000, STMV 500), writes the energies twice and
nothing else, and its rate is over the second half of the run, while
pmemd runs `mdin.GPU` and its rate is over all steps.
`--suite-protocol` (D327) runs both programs under the
protocol of each system's `mdin.GPU`:

| | `mdin.GPU` | MDIR |
|---|---|---|
| Steps | `nstlim`: JAC 100,000, Factor IX 20,000, Cellulose 30,000, STMV 2000 | `steps` = `nstlim` |
| Energies | every `ntpr` steps (2500, 1000, 1000, 1000) in `mdout` and `mdinfo` | `energy_interval` = `ntpr`, the log |
| Trajectory | every `ntwx` steps, NetCDF (`ioutfm=1`), f32 positions | `trajectory_interval` = `ntwx`, DCD: f32 positions and the cell, uncompressed, the format of MDIR closest in cost |
| Restarts | every `ntwr` steps (50,000, 10,000, 8000, 1000) and at the end | `checkpoint_interval`: `ntwr`, or for Cellulose 6000, the largest multiple of `ntwx` below `ntwr` that divides `nstlim` |
| Rate | "Average timings for last N steps", N = `nstlim` − `ntpr`, from pmemd's output | the same steps, from the row of step `ntpr` to that of step `nstlim` in MDIR's log, by the clock of bench.py as each row arrives; MDIR prints a row once the device has finished its step and flushes the log |

`--energies-only` keeps the energies and drops the trajectory and the
restarts: MDIR without `trajectory` and `checkpoint`, pmemd with a copy of
`mdin.GPU` with `ntwx=0` and `ntwr=nstlim`, in `SYSTEM/pmemd-energies`
(pmemd still writes its restart at the end). Every run records the wall
time of its process, from its start to its exit; pmemd's own wall times
of setup and of the rest (whole seconds) and MDIR's time of compilation
are recorded as well. `--record FILE` appends all the numbers of a run as
one line of JSON, with `--variant` and `--repeat`, which
`scripts/paper/run-suite-repeats.sh` and `suite-table.py` use for the
white paper (Section 10).

`bench.py --work DIR smoke --mdir PATH_TO_MDIR` runs 50 steps (two
coupling periods), with energy output only at the start and end. Add
`--analytic-bonds` to exercise the optional one-bond projection. The scale
test runs both variants; a custom `--steps` must respect coupling periods.

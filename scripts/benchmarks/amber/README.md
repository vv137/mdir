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
| Neighbor lists | skin of 2 Å (pmemd) | `pairlist_distance = 10` | Verlet buffer from its tolerance |
| Particle mesh Ewald | `dsum_tol` 1e-6 (NVE), 1e-5 (NPT) | `tolerance`, spacing 1 Å, order 4 | `ewald-rtol`, `fourierspacing = 0.1`, order 4; `-notunepme` |
| Dispersion | correction of energy and pressure | the same | `DispCorr = EnerPres` |
| Constraints | SHAKE of the bonds of hydrogen, rigid water | SHAKE, RATTLE, SETTLE | LINCS, SETTLE |
| NVE | velocities of the restart file | the same | `continuation = yes` |
| NPT | Berendsen thermostat, `tautp=10`; Monte Carlo barostat | Bussi thermostat, 1 ps; stochastic cell rescaling, 2 ps | `v-rescale`, 1 ps; `c-rescale`, 2 ps |
| Coupling interval | Monte Carlo barostat every 100 steps (`mcbarint`) | Both every 25 steps | `nsttcouple = 100`, `nstpcouple = 25`, its own choices for these inputs |

`erfc(β rc)` is the tolerance of MDIR and of GROMACS; Amber's `dsum_tol`
is `erfc(β rc) / rc`, which gives a slightly smaller β.

MDIR runs on a GPU in mixed precision. GROMACS runs with the nonbonded
forces, PME, the bonded forces, and the update on the GPU, 8 OpenMP
threads, and its counters reset halfway (`-resethway`).

## Use

```sh
bench.py --work DIR prepare --parmed-python PYTHON_WITH_PARMED
bench.py --work DIR run mdir [SYSTEM ...] --mdir PATH_TO_MDIR
bench.py --work DIR run mdir --mdir PATH_TO_MDIR --neighbor-structure GROUPS \
    --skin 3 --prune-skin 0.6        # a dual list, D114: the settings of the suite since
bench.py --work DIR run gromacs [SYSTEM ...] --gmx PATH_TO_GMX
bench.py --work DIR report
```

The work directory defaults to `$MDIR_BENCH_DIR`, or `~/mdir-benchmarks`.
Each system has a directory there with the converted inputs, the control
file and the `.mdp` that a run writes, and the logs; `results.json` keeps
the rates.

`bench.py --work DIR smoke --mdir PATH_TO_MDIR` runs 50 steps (two
coupling periods), with energy output only at the start and end. Add
`--analytic-bonds` to exercise the optional one-bond projection. The scale
test runs both variants; a custom `--steps` must respect coupling periods.

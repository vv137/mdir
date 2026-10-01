# 10. Performance

## 10.1 Conditions

*Table 10.1. What was measured, and how.*

| Item | MDIR | pmemd.cuda |
|---|---|---|
| Program | `mdir` at D114, built with LLVM and MLIR 23.1.2, PTX compiled by the driver at load | pmemd.cuda of Amber 26, SPFP [[LeGrand2013]](references.md#legrand2013), built with CUDA 13 |
| Device | One NVIDIA RTX 3090 (GPU 0), power capped at 300 W, driver 595.84; the node's other three GPUs ran other users' jobs | The same |
| Inputs | The systems of the Amber 24 GPU benchmark suite (PME), each topology and restart file written again by ParmEd in the current Amber format (`scripts/benchmarks/amber/bench.py`) | The inputs of the suite as they are (`mdin.GPU`, `prmtop`, `inpcrd`) |
| Model | Cutoff 8 Å; PME with $\operatorname{erfc}(\beta r_c) = 10^{-6}$ at constant energy and $10^{-5}$ at constant pressure, grid spacing at most 1 Å, order 4; SHAKE on the bonds of hydrogen and rigid water; the correction for the dispersion | The same, by the inputs |
| Couplings (NPT) | Stochastic velocity rescaling [[Bussi2007]](references.md#bussi2007) and stochastic cell rescaling [[Bernetti2020]](references.md#bernetti2020), every 25 steps, $\tau_T$ = 1 ps, $\tau_P$ = 2 ps | Berendsen's thermostat [[Berendsen1984]](references.md#berendsen1984), $\tau$ = 10 ps; Monte Carlo barostat every 100 steps |
| Neighbors | Groups of 16 (Section 4.4); dual list, outer reach 11 Å, inner 8.6 Å at 2 fs and 9.2 Å at 4 fs (Section 4.5) | Its own |
| Precision | Mixed (Table 7.1), `fast_math` | SPFP |
| Steps | JAC 10,000; FactorIX 4000; Cellulose 1000; STMV 500 | Those of `mdin.GPU`: JAC NVE 100,000, FactorIX NVE 20,000, Cellulose NVE 30,000, STMV 2000, … |
| Rate | Over the second half of the run, past the compilation, the set-up, and the first build | Over all steps, as pmemd.cuda reports it; its rate over the steps after the first report differs by at most 0.5% |
| Output | Energies twice; no trajectory | Energies and trajectory every 1000 or 2500 steps, as the inputs ask |

The two programs ran one after the other on the same device, MDIR over
the whole suite and then pmemd.cuda, three times
(`scripts/paper/run-suite-repeats.sh`). The benchmark script watches the
device every two seconds during each run and marks the rate as no
measurement if another process used it; no run of the repeats was
marked. The ratio of a
system is taken within each repeat and then averaged. The table is made
by `scripts/paper/suite-table.py` and the figure by
`scripts/paper/plot-suite.py`.

## 10.2 Rates

*Table 10.2. Rates over the Amber suite, mean ± sample standard deviation
over three repeats. "Energy changed by" is the change of the total
energy (NVE) or of the conserved energy (NPT) between the first and the
last row of MDIR's log, relative to its value. The counts are the mean
number of steps between builds of the outer list and between prunings of
the inner list. The rates are from three repeats; the change of the
energy, the counts, and the time of compilation from the last two, whose
outputs were kept.*

| System | Atoms | MDIR, ns/day | pmemd.cuda, ns/day | MDIR / pmemd.cuda | Energy changed by (MDIR) | Builds, prunings: every | Compiled in |
|---|---|---|---|---|---|---|---|
| `jac_nve` | 23,558 | 770.9 ± 3.8 | 615.9 ± 1.7 | 125.2 ± 0.4% | 1.0e-04 to 1.3e-04 | 16.3, 2.8 | 10–11 s |
| `jac_nve_4fs` | 23,558 | 1452.6 ± 6.3 | 1147.0 ± 3.7 | 126.6 ± 0.1% | 3.1e-03 to 3.4e-03 | 10.7, 3.5 | 10–11 s |
| `jac_npt` | 23,558 | 703.2 ± 3.4 | 583.4 ± 1.5 | 120.5 ± 0.6% | 3.5e-04 to 4.1e-04 | 16.4, 2.8 | 22 s |
| `jac_npt_4fs` | 23,558 | 1365.2 ± 5.7 | 1127.5 ± 2.8 | 121.1 ± 0.8% | 8.8e-04 to 9.1e-04 | 12.7, 4.1 | 22 s |
| `factorix_nve` | 90,906 | 284.3 ± 1.2 | 262.5 ± 0.1 | 108.3 ± 0.4% | 5.6e-06 to 9.5e-06 | 14.9, 2.7 | 10 s |
| `factorix_npt` | 90,906 | 267.1 ± 1.3 | 250.2 ± 0.4 | 106.8 ± 0.6% | 2.7e-04 to 2.8e-04 | 14.8, 2.6 | 22 s |
| `cellulose_nve` | 408,609 | 62.4 ± 0.3 | 61.1 ± 0.0 | 102.1 ± 0.5% | 6.4e-05 to 6.5e-05 | 13.7, 2.2 | 9 s |
| `cellulose_npt` | 408,609 | 60.0 ± 0.2 | 58.1 ± 0.1 | 103.3 ± 0.3% | 8.3e-05 to 1.1e-04 | 13.8, 2.2 | 18–19 s |
| `stmv_npt_4fs` | 1,067,095 | 41.4 ± 0.1 | 37.4 ± 0.1 | 110.6 ± 0.3% | 1.1e-04 to 1.2e-04 | 6.7, 2.1 | 22–23 s |

![Rates of MDIR and pmemd.cuda over the Amber suite](figures/suite.png)

*Figure 10.1. The rates of Table 10.2 on a logarithmic scale, with the
ratio of the means above each pair.*

MDIR is faster than pmemd.cuda on every system of the suite. The margin
is largest on the smallest system and smallest on Cellulose:

- **JAC** (23,558 atoms; a step of about 225 µs at 2 fs). A step of this
  size has few warps of work per kernel and many kernels, so what the
  host and the launches cost matters as much as the arithmetic. The
  integration kernel (Section 8.2) does in one launch what took three,
  the flags of the tests reach the host without waiting for the stream
  (Section 8.3, from 695 to 739 ns/day), and the dual list keeps an
  inner list of reach 8.6 Å, pruned every 2.8 steps, under an outer list
  of 11 Å built every 16, so the loop over pairs reads the pairs within
  8.6 Å rather than within the 10 Å of one list (Section 4.5).
- **Cellulose** (408,609 atoms). The device is busy 97% of a step (Table
  8.2); the loop over pairs and PME take two thirds of it, and both are
  close to the limits of their resources (the loop over groups at 92% of
  the peak of the pipe of loads and shuffles). Here the ratio rests on the
  arithmetic of the kernels: each pair once, tables of the radial
  functions (Section 7.2), the bricks of the spreading (Section 5.3).
- **STMV** at 4 fs (1,067,095 atoms) builds its outer list every 6.7
  steps and prunes every 2.1. Its ratio lies between those of FactorIX
  and Cellulose; the measurements of this paper do not separate its
  causes.

## 10.3 Energy in the measured runs

A rate is reported with the conservation of the run that measured it
(Section 12). The changes in Table 10.2 are over runs of 2 to 40 ps and
include the transient of the start of each restart file (Section 13);
they are not drifts. Over 2 ns, the rate of drift of JAC is 1.5
kcal/mol/ns against 5.9 for pmemd.cuda (Section 9.4).

## 10.4 Compilation

Each run compiles its program first (Section 3.6). The time is that of
the column "Compiled in" of Table 10.2: about 10 s for a system at
constant energy and about 20 s at constant pressure, whose module has
the programs of the barostat in addition, independent of the size of the
system. It is spent once per run: 4% of a run of 2 ns of JAC, which
takes 4 minutes, and 0.1% of a run of 100 ns.

## 10.5 Over long runs

Over 2 ns of JAC at constant energy before the dual list (D112), the rate
fell from 726 to 722 ns/day, most of it in the first 0.5 ns, and then
stayed flat; an earlier build fell by 1.8% in the same way. The loops over
pairs read the particles in the order of the last build (D86) and do not
slow; what slows is the loops that read the state in its own order, which
is sorted only where a run begins (Section 4.2).

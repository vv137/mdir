# 10. Performance

## 10.1 Conditions

The two programs are measured under one protocol (D327):
each runs the steps of the suite's own input, writes the output that
input asks for, and is timed over the same steps. Earlier versions of
this section gave MDIR shorter runs, timed over their second half, with
the energies written twice and no trajectory, while pmemd.cuda ran the
inputs as they are; that protocol favored MDIR, and its ratios are kept
only as history, in the row of D327 in
[decisions.md](../decisions.md).

*Table 10.1. What was measured, and how.*

| Item | MDIR | pmemd.cuda |
|---|---|---|
| Program | `mdir` of `main` at afc96bf5 (after D320), built with LLVM and MLIR 23.1.2, kernels loaded as cubins (D214); the build is of the branch of D327, whose commit changes only scripts | pmemd.cuda of Amber 26, SPFP [[LeGrand2013]](references.md#legrand2013), built with CUDA 13 for the RTX 4090 |
| Device | One NVIDIA RTX 4090 (450 W) of one compute node (ada-003), on 2026-10-22, in one job of the cluster's scheduler that held the device alone (`gpu-run --idle`); other jobs ran on the node's other GPUs | The same |
| Inputs | The systems of the Amber 24 GPU benchmark suite (PME), the topologies and restart files of the suite as they are (`prmtop`, `inpcrd`; two topologies are in the format before Amber 7, D133), with the settings of `scripts/benchmarks/amber/bench.py --suite-protocol` | The inputs of the suite as they are (`mdin.GPU`, `prmtop`, `inpcrd`) |
| Model | Cutoff 8 Å; PME with the $\beta$ of pmemd, $\mathrm{erfc}(\beta r_c)/r_c = 10^{-6}$ at constant energy and $10^{-5}$ at constant pressure (0.39467 and 0.34864 Å⁻¹), the grids of pmemd.cuda (64³; 144 × 84 × 80; 256 × 128 × 128; 224 × 224 × 240), order 4, and Amber's influence function (D134); SHAKE on the bonds of hydrogen and rigid water; the correction for the dispersion | The same, by the inputs |
| Couplings (NPT) | Stochastic velocity rescaling [[Bussi2007]](references.md#bussi2007) and stochastic cell rescaling [[Bernetti2020]](references.md#bernetti2020), every 25 steps, $\tau_T$ = 1 ps, $\tau_P$ = 2 ps | Berendsen's thermostat [[Berendsen1984]](references.md#berendsen1984), $\tau$ = 10 ps; Monte Carlo barostat every 100 steps. The couplings differ; neither program's dynamics was changed for the measurement |
| Neighbors | Groups of 16 (Section 4.4) with a dual list (Section 4.5), as `AUTO` takes them from 23,000 particles (D261), with two choices of the reaches: *searched*, those that a search chose and cached for each system (`"AUTO"`, D281; Table 10.2), and *default*, the defaults of D261 with no search and no cache (`MDIR_SKIN_CACHE=off`): 11 Å and an inner list at 8.6 Å, 9.2 Å at 4 fs | Its own: a skin of 2 Å. pmemd.cuda is not tuned beyond its inputs |
| Precision | Mixed (Table 7.1), `fast_math` | SPFP |
| Steps | Those of `mdin.GPU` (`nstlim`): JAC 100,000; Factor IX 20,000; Cellulose 30,000; STMV 2000 | The same |
| Rate | Over the steps from the first report (step `ntpr`) to the last: 97,500 steps of JAC, 19,000 of Factor IX, 29,000 of Cellulose, 1000 of STMV. Timed by the benchmark script's clock at the rows of those two steps in the log, which MDIR prints once the device has finished the step; over MDIR's own window, the second half, that clock agrees with MDIR's to within 0.05% in all 80 runs | The same steps: its "Average timings for last N steps" |
| Output | The energies every `ntpr` steps (2500 for JAC, 1000 for the others) in the log; positions every `ntwx` steps (the same) in DCD, f32 without compression, the format of MDIR closest in cost to pmemd.cuda's; a checkpoint (H5MD) every `ntwr` steps (50,000; 10,000; 6000 for Cellulose, whose 8000 does not divide its steps; 1000) | The energies every `ntpr` steps in `mdout` and `mdinfo`; positions every `ntwx` steps in NetCDF (`ioutfm=1`); a restart every `ntwr` steps and at the end |
| Output, second variant | The energies alone (Table 10.3) | `ntwx=0` and `ntwr` = `nstlim` in a copy of `mdin.GPU`: the energies and the restart at the end |
| Wall time | From the start of the process to its exit, compilation included, by the benchmark script's clock (Table 10.4) | The same; and its own setup and non-setup wall times, in whole seconds |
| Files | On the shared file system of the cluster | The same |

The programs ran one after the other on the same device, three times, in
one job of 85 minutes (`scripts/paper/run-suite-repeats.sh`): before the
repeats, the search of the skins for each system; then in each repeat
MDIR with the searched reaches, pmemd.cuda, MDIR with the default
reaches, MDIR with the searched reaches and the energies alone, and
pmemd.cuda with the energies alone, each over the whole suite. The
benchmark script watches the device every two seconds during each run
and marks the rate as no measurement if another process used it. One of
the 135 runs was marked, MDIR with the energies alone on `jac_nve_4fs` in
the second repeat (a process of `gdscheck`, a tool of NVIDIA GPUDirect
Storage, appeared on the device); it is left out, and that cell of Table
10.3 is from two repeats. A ratio is taken within each repeat, between
runs on the same device in the same job, and then averaged; ratios from
different nodes of the cluster are not mixed, since the same program's
rate differs between them by up to 6% (pmemd.cuda on Cellulose, 117.3
and 124.5 ns/day on two nodes, #405). The tables are made by
`scripts/paper/suite-table.py` and the figure by
`scripts/paper/plot-suite.py`.

**The skins.** A run chooses its reaches by measuring candidates (D281,
Section 4.5) only where it is long, from $5 \times 10^{12}$ particles
times steps, about five hours on this device; a shorter run takes the
choice that an earlier run of the same key cached, else the defaults of
D261. The suite's runs are far shorter, so before the repeats each system
ran once, untimed, with `MDIR_SKIN_SEARCH=always`, which measured six or
seven candidates and cached the choice in 79 to 201 s a system; every
timed run of the searched variant took its reaches from that cache, and
every run of the default variant took the defaults, and their logs say
so. A run at the threshold affords fewer candidates, those whose
compilation and steps fit within 2% of its time, so the cached choice is
that of a run several times longer. The cost of the search, once per key
(Section 4.5), is in neither the rates nor the wall times.

## 10.2 Rates

*Table 10.2. Rates over the Amber suite in ns/day, mean ± sample standard
deviation over three repeats, all with the output of `mdin.GPU`. The
ratio is taken within each repeat and then averaged. "Steps, timed" is
the window of both programs, from the first report to the last step.
"Searched reaches" are the outer and inner reaches that the search chose
and the searched runs took from the cache. "Energy changed by" is the
change of the total energy (NVE) or of the conserved energy $\tilde H$
(NPT, Section 6.5) between the first and the last row of the log of
MDIR's searched runs, relative to its value; the counts are the mean
number of steps between builds of the outer list and between prunings of
the inner list in those runs. pmemd.cuda's rates at constant pressure
spread more between the repeats than in earlier jobs (up to ±13.1
ns/day on `jac_npt_4fs`), which widens the spread of those ratios.*

| System | Atoms | Steps, timed | pmemd.cuda | MDIR, searched reaches | MDIR / pmemd.cuda | MDIR, default reaches | MDIR / pmemd.cuda | Searched reaches, Å | Energy changed by (MDIR) | Builds, prunings: every |
|---|---|---|---|---|---|---|---|---|---|---|
| `jac_nve` | 23,558 | 97,500 of 100,000 | 869.1 ± 1.7 | 1276.8 ± 4.5 | 146.9 ± 0.8% | 1232.4 ± 2.3 | 141.8 ± 0.5% | 13.5 / 8.8 | 8.5e-05 to 1.4e-04 | 57.5, 3.6 |
| `jac_nve_4fs` | 23,558 | 97,500 of 100,000 | 1609.3 ± 2.5 | 2406.1 ± 8.8 | 149.5 ± 0.5% | 2311.5 ± 3.5 | 143.6 ± 0.2% | 13.5 / 8.8 | 3.1e-03 to 3.6e-03 | 26.7, 2.8 |
| `jac_npt` | 23,558 | 97,500 of 100,000 | 818.9 ± 8.8 | 1148.5 ± 6.0 | 140.3 ± 0.8% | 1125.4 ± 2.4 | 137.4 ± 1.3% | 13 / 8.85 | 3.5e-04 to 5.3e-04 | 46.7, 3.8 |
| `jac_npt_4fs` | 23,558 | 97,500 of 100,000 | 1576.2 ± 13.1 | 2233.3 ± 2.8 | 141.7 ± 1.0% | 2184.5 ± 4.0 | 138.6 ± 1.1% | 13 / 8.7 | 6.1e-04 to 1.1e-03 | 29.1, 2.9 |
| `factorix_nve` | 90,906 | 19,000 of 20,000 | 444.3 ± 5.7 | 558.1 ± 0.2 | 125.6 ± 1.6% | 556.0 ± 1.6 | 125.1 ± 1.2% | 11 / 8.6 | 6.2e-06 to 1.8e-05 | 14.8, 2.7 |
| `factorix_npt` | 90,906 | 19,000 of 20,000 | 412.8 ± 8.4 | 519.3 ± 0.7 | 125.8 ± 2.5% | 519.9 ± 1.3 | 126.0 ± 2.3% | 11 / 8.6 | 2.7e-04 to 2.9e-04 | 14.8, 2.7 |
| `cellulose_nve` | 408,609 | 29,000 of 30,000 | 124.0 ± 0.3 | 145.3 ± 0.2 | 117.2 ± 0.4% | 142.4 ± 0.2 | 114.8 ± 0.2% | 11.5 / 8.45 | 1.2e-04 to 1.6e-04 | 17.4, 1.9 |
| `cellulose_npt` | 408,609 | 29,000 of 30,000 | 115.1 ± 0.4 | 138.3 ± 0.2 | 120.2 ± 0.3% | 135.4 ± 0.2 | 117.7 ± 0.3% | 11.5 / 8.4 | 5.0e-04 to 5.8e-04 | 17.6, 1.9 |
| `stmv_npt_4fs` | 1,067,095 | 1,000 of 2,000 | 74.7 ± 1.0 | 83.4 ± 0.5 | 111.7 ± 1.1% | 79.0 ± 0.4 | 105.8 ± 1.6% | 12.5 / 8.8 | 8.8e-04 to 9.4e-04 | 13.8, 1.9 |

![Rates of MDIR and pmemd.cuda over the Amber suite](figures/suite.png)

*Figure 10.1. The rates of Table 10.2 on a logarithmic scale, with the
mean ratio of MDIR with the searched reaches to pmemd.cuda above each
group.*

On this device, with the steps and the output of the suite's inputs,
MDIR's rate with the searched reaches is 140% to 150% of pmemd.cuda's on
JAC, 126% on Factor IX at constant energy and at constant pressure, 117%
and 120% on Cellulose, and 112% on STMV; with the default reaches it is
137% to 144% on JAC, 125% and 126% on Factor IX, 115% and 118% on
Cellulose, and 106% on STMV. The sample standard deviation of each of
MDIR's rates is at most 0.6% of its mean.

**The reaches.** The search is worth 2.1% to 4.1% on JAC (searched
against default in the same repeats: 3.6% and 4.1% at constant energy at
2 and 4 fs, 2.1% and 2.2% at constant pressure), 2.0% and 2.1% on
Cellulose, and 5.6% on STMV. On Factor IX it kept the defaults, 11 /
8.6 Å, and the two variants differ by 0.4% and −0.1%, within the noise.
The search chose the same reaches for STMV as in the measurement before
(12.5 / 8.8 Å), and for Cellulose 11.5 / 8.45 Å and 11.5 / 8.4 Å, 1.7%
and 2.2% faster than the defaults in its own measurement, where the
search of the measurement before kept the defaults: the fastest candidate
on Cellulose gains about 2%, close to the noise of a search (0.9% to
2.3%), so the choice there can differ from one search to the next. The
reaches change the speed and not the pairs within the cutoff (Section
4.5).

**What changed from the protocol before.** With the searched reaches,
the ratio moved from the previous table (D285, 2026-10-18, same node) by
−1.4 and −1.5 points on JAC at constant energy, +1.2 and +0.2 at constant
pressure, −4.1 and −1.7 on Factor IX, −7.3 and −5.4 on Cellulose, and
−13.8 on STMV. Two causes are clear:

- **STMV: the checkpoint in the window.** STMV's window is 1000 steps,
  about 4 s, and holds MDIR's checkpoint at step 1000: the run writes 1
  million particles in H5MD to the shared file system and begins a new
  segment, whose structures are built anew (Section 10.4). With the
  energies alone, MDIR's rate is 93.1 ns/day and 124.2% of pmemd.cuda's
  with its energies alone (Table 10.3); the checkpoint and the frame cost
  MDIR about 0.43 s, 10.4% of the window. pmemd.cuda also writes a
  restart at step 1000; whether its "last 1000 steps" include it, its
  output does not say. The cost of a checkpoint of MDIR on a large system
  is issue #548.
- **The run is longer and both programs' rates moved between jobs.** The
  previous table timed MDIR over steps 500 to 1000 of Cellulose, just
  after the sort of the state where a run begins (Section 4.2), and 2000
  to 4000 of Factor IX; this one over 1000 to 30,000 and 1000 to 20,000.
  Within these runs, MDIR's rate over the second half is 0.2% to 0.4%
  below its rate over the whole window on Factor IX and Cellulose, so the decay of the order of
  the state (Section 10.5) is a small part. The program also changed
  (D285 to D320), and pmemd.cuda's own rate moved between the two jobs on
  the same node: 902.0 to 869.1 ns/day on JAC NVE (−3.6%), 121.1 to
  124.0 on Cellulose NVE (+2.4%). MDIR's moved by −4.5% on JAC NVE,
  −3.9% on Factor IX NVE, and −3.6% on Cellulose NVE. Which share of the
  remaining change belongs to which cause was not measured.

**JAC** (23,558 atoms; a step of about 135 µs at 2 fs). A step of this
size has few warps of work per kernel and many kernels, so what the host
and the launches cost matters as much as the arithmetic. The integration
kernel (Section 8.2) does in one launch what took three, the flags of
the tests reach the host without waiting for the stream (Section 8.3),
and the dual list keeps an inner list of reach 8.8 Å, pruned every 3.6
steps, under an outer list of 13.5 Å built every 57.5, so the loop over
pairs reads the pairs within 8.8 Å and the builds, which cost a fixed
time of launches each, are rare (Section 4.5). **Cellulose** (408,609
atoms) and **STMV** (1,067,095 atoms) keep the device busy nearly the
whole step (Table 8.2, measured on an RTX 3090), and the loop over pairs
and PME take most of it; the rate rests on the arithmetic and the memory
traffic of those kernels: each pair once, tables of the radial functions
(Section 7.2), the tiles of the spreading (Section 5.3), and, from 4096
groups, the pruning inside the loop over pairs (D284, Section 4.5).

**The output.** Table 10.3 compares the rates with the energies alone.

*Table 10.3. Rates with the energies alone, in ns/day,
over the same windows, and the share of the output: 1 − (rate with the
output of `mdin.GPU`) / (rate with the energies alone), within each
repeat. A share below its standard deviation, as pmemd.cuda's on most
systems and both programs' on JAC, is not distinguished from zero. The
cell of MDIR on `jac_nve_4fs` is from two repeats (above).*

| System | pmemd.cuda, energies only | MDIR, energies only | MDIR / pmemd.cuda | Output's share: pmemd.cuda | MDIR |
|---|---|---|---|---|---|
| `jac_nve` | 870.3 ± 1.3 | 1273.6 ± 0.6 | 146.3 ± 0.2% | 0.1 ± 0.3% | −0.2 ± 0.4% |
| `jac_nve_4fs` | 1604.7 ± 14.6 | 2409.4 ± 2.2 | 150.5 ± 1.6% | −0.3 ± 1.1% | −0.1 ± 0.0% |
| `jac_npt` | 818.5 ± 11.7 | 1152.8 ± 1.6 | 140.9 ± 1.9% | −0.1 ± 2.2% | 0.4 ± 0.7% |
| `jac_npt_4fs` | 1580.6 ± 15.4 | 2238.2 ± 2.6 | 141.6 ± 1.2% | 0.3 ± 1.6% | 0.2 ± 0.2% |
| `factorix_nve` | 452.2 ± 1.6 | 559.7 ± 0.1 | 123.8 ± 0.4% | 1.7 ± 1.3% | 0.3 ± 0.0% |
| `factorix_npt` | 420.9 ± 4.5 | 521.9 ± 0.0 | 124.0 ± 1.3% | 1.9 ± 2.5% | 0.5 ± 0.1% |
| `cellulose_nve` | 126.0 ± 0.7 | 146.7 ± 0.2 | 116.5 ± 0.5% | 1.6 ± 0.7% | 1.0 ± 0.2% |
| `cellulose_npt` | 115.9 ± 1.6 | 139.7 ± 0.1 | 120.5 ± 1.6% | 0.7 ± 1.6% | 1.0 ± 0.2% |
| `stmv_npt_4fs` | 74.9 ± 1.4 | 93.1 ± 0.0 | 124.2 ± 2.4% | 0.3 ± 3.0% | 10.4 ± 0.6% |

The output costs MDIR 1.0% on Cellulose and at most 0.5% on JAC and
Factor IX, whose windows hold a checkpoint at most once in 10,000 steps;
on STMV it costs 10.4%, the checkpoint inside its short window (above).

**Time to solution.** Table 10.4 gives the whole wall time of each run.

*Table 10.4. The wall time of each run with the
output of `mdin.GPU`, from the start of the process to its exit, in
seconds; the part of it outside the steps, wall time − `nstlim` Δt /
rate (derived); pmemd.cuda's own setup wall time (whole seconds) and
MDIR's time of compilation, both measured; and the wall time of a run of
1 ns, the part outside the steps plus 1 ns at the rate of Table 10.2, in
minutes (derived). MDIR with the searched reaches; the search, once per
key, is not included.*

| System | Wall time of the run, s: pmemd.cuda | MDIR | Of it outside the steps (derived), s: pmemd.cuda | MDIR | pmemd.cuda's setup, s | MDIR compiled in, s | Wall time of 1 ns (derived), min: pmemd.cuda | MDIR |
|---|---|---|---|---|---|---|---|---|
| `jac_nve` | 20.5 ± 0.1 | 21.4 ± 0.0 | 0.6 ± 0.1 | 7.9 ± 0.0 | 0 ± 1 | 7.4 ± 0.0 | 1.67 ± 0.00 | 1.26 ± 0.00 |
| `jac_nve_4fs` | 22.0 ± 0.1 | 22.3 ± 0.1 | 0.6 ± 0.0 | 7.9 ± 0.0 | 0 ± 0 | 7.4 ± 0.0 | 0.90 ± 0.00 | 0.73 ± 0.00 |
| `jac_npt` | 21.7 ± 0.3 | 29.5 ± 0.3 | 0.6 ± 0.1 | 14.5 ± 0.3 | 0 ± 1 | 13.9 ± 0.1 | 1.77 ± 0.02 | 1.50 ± 0.01 |
| `jac_npt_4fs` | 22.6 ± 0.4 | 29.9 ± 0.1 | 0.7 ± 0.2 | 14.5 ± 0.1 | 0 ± 1 | 13.9 ± 0.1 | 0.93 ± 0.01 | 0.89 ± 0.00 |
| `factorix_nve` | 8.7 ± 0.3 | 14.9 ± 0.1 | 0.9 ± 0.2 | 8.7 ± 0.1 | 1 ± 0 | 7.5 ± 0.1 | 3.26 ± 0.04 | 2.73 ± 0.00 |
| `factorix_npt` | 9.2 ± 0.2 | 21.9 ± 0.2 | 0.8 ± 0.1 | 15.3 ± 0.2 | 1 ± 1 | 14.1 ± 0.2 | 3.50 ± 0.07 | 3.03 ± 0.00 |
| `cellulose_nve` | 45.0 ± 0.6 | 48.2 ± 0.2 | 3.2 ± 0.5 | 12.6 ± 0.1 | 3 ± 1 | 7.0 ± 0.0 | 11.67 ± 0.04 | 10.12 ± 0.01 |
| `cellulose_npt` | 48.0 ± 0.2 | 55.6 ± 0.3 | 2.9 ± 0.1 | 18.1 ± 0.2 | 2 ± 1 | 12.5 ± 0.1 | 12.57 ± 0.04 | 10.71 ± 0.02 |
| `stmv_npt_4fs` | 16.1 ± 0.4 | 36.1 ± 0.7 | 6.9 ± 0.3 | 27.9 ± 0.7 | 6 ± 0 | 13.8 ± 0.3 | 19.40 ± 0.27 | 17.72 ± 0.12 |

MDIR spends 7.9 to 27.9 s of a run outside its steps, pmemd.cuda 0.6 to
6.9 s; most of MDIR's is the compilation (Section 10.4), and on STMV
also the reading of the 200 MB topology and the checkpoint at the end.
The suite's runs are short, 8.7 to 48 s for pmemd.cuda, so the whole run
takes longer with MDIR on every system, by 0.3 s on JAC at 4 fs and
constant energy up to 20 s on STMV. For a run of 1 ns the fixed part
weighs less: the derived wall time of MDIR is 75% to 96% of pmemd.cuda's
(JAC NPT at 4 fs 0.89 against 0.93 minutes; STMV 17.7 against 19.4).

**Earlier measurements.** The ratios of the protocol before, at D134 on
an RTX 3090 and at D280 and D285 on an RTX 4090, are kept as history in
the row of D327 in [decisions.md](../decisions.md).
Under this protocol MDIR's rate is at least pmemd.cuda's on every system
of the suite with either choice of the reaches, the first milestone's
target (D114); the closest is STMV with the default reaches, 105.8 ±
1.6%.

## 10.3 Energy in the measured runs

A rate is reported with the conservation of the run that measured it
(Section 12). The changes in Table 10.2 are over runs of 8 to 400 ps and
include the transient of the start of each restart file (Section 13);
they are not drifts. Over 2 ns, the total energy of JAC rose by 1.5
kcal/mol/ns in MDIR and 5.9 in pmemd.cuda, one run of each under the
conditions of Section 9.4.

## 10.4 Compilation

Each run compiles its program first (Section 3.6). The time is that of
the column "MDIR compiled in" of Table 10.4: 7.0 to 7.5 s for a system
at constant energy and 12.5 to 14.1 s at constant pressure, whose module
has the programs of the barostat in addition, independent of the size of
the system. It is spent once per run: 5% of a run of 2 ns of JAC, which
takes 2.4 minutes, and 0.1% of a run of 100 ns.

A Python simulation can take its host object from the compile cache
(Section 3.6). For the stages of the ala3 example on a GPU in mixed
precision it removes 1 to 6 s of host code generation per stage, of 12 to
54 s of compilation, most of which is then the MLIR pipeline; the whole
example at a hundredth of its steps, in the deterministic mode, took 137 s
with a warm cache against 155 s with a cold one
([compile-cache.md](../compile-cache.md)). Within a process a `Program`
keeps the LLVM module and the host object of its first simulation, and
its later simulations link them without the pipeline (D236):
for the dipeptide in water, 0.07 s on the CPU and 0.13 s on a GPU for
each simulation after the first, instead of 5.4 s and 9.4 s.

The GPU modules of a program, 400 for the production stage of the ala3
example under `mdir run`, are serialized in parallel and loaded as
cubins (D214). The serialization was 62% of the MLIR
pipeline on a GPU, and the driver compiled every PTX serially at load.
The same example, also deterministic at a hundredth of its steps, took
147 s on main with the driver's cache of compiled PTX warm, and 189 s with
it disabled. It took 82 s once the modules were serialized in parallel,
about the same with cubins and the compile cache off, and 60 s with the
compile cache warm, even with the driver's cache disabled. The JIT part of
a stage then falls to 0.3 to 0.5 s; the MLIR passes before the
serialization are most of what remains.

A Python simulation runs its parts in one activation of its entry
(Section 3.6, D215), so a part costs its steps and a
boundary: on JAC in mixed precision a part of 10 steps takes 2.75 ms, its
steps 2.73 ms, where it took 10.3 ms when every part copied its buffers to
the device and built its structures anew; a part of one step takes 0.309
ms instead of 8.77 (`docs/python-segments.md`).

What remains outside the parts is the start of an activation, which every
update of the parameters in a fit pays. It renumbered the tuples of the
topology and built their incidence structures on the host, 1.7 to 1.8 kB
an atom, and its cost for each atom grew with the system: 0.09 µs on JAC,
0.15 µs on Factor IX, and 0.77 µs on Cellulose for the host's part. The
algorithm is linear; the memory was not free. The C library recycles
freed blocks up to 32 MiB, and a larger structure was mapped from the
system, faulted in page by page, and unmapped at every start: none of the
five structures of JAC, one of Factor IX, four of Cellulose. A simulation
now keeps the structures across its activations (Section 3.6,
D253): a start at which the particles have
not moved compares the order and the members with the kept copies (0.08,
0.24, and 1.7 ms on the three systems) and builds nothing, and a start in
a new order builds in the kept memory (2.4, 7.5, and 71 ms on the host,
where it took 2, 14, and 315 ms). On one RTX PRO 6000 Blackwell in mixed
precision, where the upload goes with the build, a start with its
evaluation at which the particles have not moved takes 2.0, 6.1, and 30
ms on the three systems where it took 5.0, 61, and 413 ms, and one in a
new order 5.1, 17, and 98 ms where it took 5.0, 58, and 387 ms. An update
of the parameters with its evaluation took 8.3, 30, and 203 ms where it
took 15, 135, and 867 ms, its two starts the smaller part of it: the rest
was a sort of the members of every tuple set, copies of the model, and
quadratures of the tails repeated for every pair of classes of particles.
Since D254 the simulation writes the values into a model that it
keeps, the sort is one pass, and a build computes each quadrature once; on
one RTX 4090 an update with its evaluation takes 5.4, 16, and 87 ms where
it took 8.4, 31, and 204 ms, and 8.1, 22, and 112 ms where it took 121,
189, and 259 ms with the tail of a pair term in the correction. The
arrays of the program that no value of a tunable enters, which a build
from the model with every tunable site set to not-a-number makes the same
to the bit, are kept since D256: an update with its
evaluation then takes 4.6, 13, and 65 ms, about its two starts. Each start
copied the state and the whole model into new memory for the arguments of
the entry, one number at a time: 0.44, 1.76, and 11.5 ms of a start of
2.0, 6.0, and 30 ms. Since D258 the compile
proves that the entry writes no buffer of the model (Section 3.6), a start
hands it the arrays of the program as they are, and the buffers of the
state are kept and written anew: the arguments take 0.06, 0.18, and 1.9
ms, a start at which the particles have not moved 1.63, 4.25, and 20.8 ms,
and an update with its evaluation 3.62, 8.92, and 42.8 ms on one RTX 4090,
with the steps unchanged.

The segments of `mdir run` begin in a new order each (Section 3.6), so
they cannot take what is kept, but they build in the kept memory
(D255). On one RTX 4090 in mixed precision, with the
settings of the Amber suite and a checkpoint every $n$ steps, the
structures cost a segment 4.6 ms on JAC, 27 ms on Factor IX, 152 ms on
Cellulose, and 465 ms on STMV, of which the build on the host in new
memory was 55% to 67% on the three larger systems and the allocation and
upload most of the rest: 3% to 8% of a segment at $n = 1000$ and 0.3% to
1% at $n = 10{,}000$. Written in the kept memory and uploaded 16 MiB at a
time they cost 4.5, 15, 78, and 238 ms; at $n = 10{,}000$ a run is 0.45%,
0.88%, and 0.81% faster on Factor IX, Cellulose, and STMV, and at
$n = 1000$ 3.5%, 5.0%, and 5.5%. The CPU target gains nothing that a run
can see, since a step there costs 40 to 720 ms on these systems; what
remained on a GPU was the build on the host and the upload.

On a device that build is now three kernels, a scatter and two gathers
from the structure in the order of the files (Section 3.6,
D257), and the order is compared there, so that a start in a
new order copies 4 bytes between the host and the device. On the same
device a start in a new order of a Python simulation with its evaluation
takes 1.98 ms on JAC, 5.67 ms on Factor IX, and 26.2 ms on Cellulose
where it took 5.29, 18.1, and 96.9 ms, and the functions of the store,
the comparison and the launches included, 0.07, 0.17, and 1.8 ms of it;
`mdir run` with a checkpoint every 1,000 steps is 2.3% to 4.5% faster on
the four systems, and a run of the suite as it is, one segment, keeps its
rate. The device held the members and an index of each tuple set
besides, 6% to 10% of the incidence structures. Since the structures are
ragged rows built from the members on the device (D314),
the index is gone and the structures are 9 times smaller on STMV; a
start in a new order takes 0.1 ms more on JAC, for the launches of the
build, and the same on Factor IX and Cellulose.

**Reading a large GROMACS coordinate file (D323).** After the
neighbor count at the start became exact and bounded by cells (D316),
reading the numeric fields of a `.gro` file was the largest measured part
of preparation. The reader now converts a complete finite decimal field
directly to binary64 and retains its general conversion for other forms.
The file's width and the values of the positions and velocities do not
change; the compiled program and the integration loop do not read text.
In one sequential main-versus-change run on an otherwise idle RTX 4090,
STMV from the GROMACS topology took 19.375 to 14.223 s from `mdir run`
to its first energy row at 1,067,095 particles (73,629,636 input bytes),
and its ×11 replica took 103.277 to 45.680 s at 11,738,045 particles
(809,925,187 bytes). The compiler's own time was 10.62 to 10.41 s and
10.32 to 10.60 s respectively. The main baseline was 1817dc5d. Each is one run, with the driver's kernel
cache initially cold and then shared by the sequential runs; the figures
describe startup on that node. In a separate single-process CPU read of the same inputs, the complete positions and velocities of each size have SHA-256 hashes equal to the old reader, and topology plus coordinate reading took 8.64 to 3.34 s and 86.39 to 30.17 s. The rates of the 1,000 integration steps of the GPU timing job were 4.23 to 4.15 ms/step at ×1 and 57.49 to 57.62 ms/step at ×11; the change does not alter a step kernel.

## 10.5 Over long runs

Over 2 ns of JAC at constant energy before the dual list (D112), the rate
fell from 726 to 722 ns/day, most of it in the first 0.5 ns, and then
stayed flat; an earlier build fell by 1.8% in the same way. The loops over
pairs read the particles in the order of the last build (D86) and do not
slow; what slows is the loops that read the state in its own order, which
is sorted only where a run begins (Section 4.2).

## 10.6 Against GROMACS on a protein in OPC

The target of the milestone was the rate of pmemd.cuda; GROMACS 2026.3
[[Pall2020]](references.md#pall2020), built with CUDA and run on the same
device, is a second reference, measured on a small system. The system is
ubiquitin (1UBQ) in 5700 OPC waters with amber19sb.ff, 24,031 particles
in a triclinic cell, prepared and equilibrated by GROMACS
(`scripts/validation/protein/run.py`); its energy terms agree with
GROMACS's (Section 9.1). Both run 60,000 steps of 2 fs from the
equilibrated state in mixed precision, timed over the second half, with a
cutoff of 9 Å, PME with $\mathrm{erfc}(\beta r_c) = 10^{-5}$, SHAKE
on the bonds of hydrogen and rigid water, and at constant pressure the
same couplings every 25 steps. MDIR uses groups and the dual list with
reaches of 12 and 9.6 Å, within 0.4% of the best of 11 to 12 Å and 9.4 to
9.8 Å; GROMACS its own list, its nonbonded terms and PME on the GPU, and
its update on the CPU, which virtual sites need.

*Table 10.5. Ubiquitin in OPC, RTX 3090 at 300 W, at D118, mean ± sample
standard deviation over three repeats, each program after the other on
the device, with nothing else on the host. GROMACS at constant energy over
two of them: the third gave 789.6 ns/day, having spent 0.45 s writing its
output against 0.04 s in the other two ("Write traj." of its accounting of
cycles, a stall of the shared file system). "Energy changed by" as in
Table 10.2, over 120 ps, the range of the three. At constant pressure MDIR
counts the work of the barostat from the virial of the groups (D116); with
the count before it, $-228$ kcal/mol/ns (Section 13). The other settings
of GROMACS are from one run each, before D118.*

| Ensemble | MDIR, ns/day | GROMACS, ns/day | MDIR / GROMACS | Energy changed by: MDIR | GROMACS |
|---|---|---|---|---|---|
| NVE | 605.2 ± 0.4 | 842.9 ± 2.9 (nstlist 80; 689.7 with the nstlist of 10 that it keeps at constant energy) | 71.8 ± 0.2% | $2.4\times10^{-5}$ to $4.4\times10^{-5}$ | $1.7\times10^{-4}$ to $2.0\times10^{-4}$ |
| NPT | 571.5 ± 0.7 | 865.5 ± 4.4 (771.8 with `verlet-buffer-tolerance` $5\times10^{-5}$) | 66.0 ± 0.4% | $1.6\times10^{-6}$ to $1.2\times10^{-4}$ | $6.7\times10^{-3}$ to $7.1\times10^{-3}$ |

*Table 10.6. The device time of a step, NVE, from nsys: MDIR over 20,000
steps before D118, GROMACS over the 3528 steps that the profile captured.
MDIR's kernels are sorted by their kind. D118 changed none of them; it
took the time the device stood idle in a step from 49 to 29 µs.*

| Part | MDIR, µs | GROMACS, µs |
|---|---|---|
| Loops over pairs | 91.6 (the inner list of 9.6 Å) | 87.7 (one kernel; a list of 10.7 Å built every 80 steps) |
| PME | 47.3 | 76.7, on a second stream beside the nonbonded kernel |
| Loops over particles: the drift, the kicks, and the other loops over particles | 48.9 | The update on the host |
| Loops over tuples: bonded terms and excluded pairs 22.5, SETTLE 11.8, placement of the sites 4.9 | 39.3 | Bonded terms, SETTLE, and sites on the host; excluded pairs in the nonbonded kernel |
| Builds and prunings of the lists | 34.6 | In the nonbonded kernel, every 80 steps |
| Layout and reductions | | 18.3 |
| On the device | 261.8, one stream | 182.8, two streams |
| Wall time of a step | 310 | 214 |

The loop over pairs costs the same in both programs, and MDIR's PME takes
47.3 µs of the device against GROMACS's 76.7. GROMACS runs beside that loop what MDIR runs after it: PME on a
second stream of the device, and the bonded terms, the update, and the
constraints on the 64 threads of the host. MDIR runs every part on one
stream; its loops over particles and over tuples and its builds and
prunings take 123 µs of the device in a step, and the host's time between
its launches 48 µs of the 310. Its own second stream for the reciprocal
sum (D81, D87), tried on this system, gives 597 ns/day against 609
without (D118). On JAC, with three sites per water and a cutoff of 8 Å, the same
structure's rate is 147% of pmemd.cuda's at constant energy (Table 10.2); on
this system its rate is 66% to 72% of GROMACS's, and the measurements place the
difference outside the loop over pairs. Most of it is the policy of the
lists rather than the kernels, as the matched comparison below shows. GROMACS's conserved energy at
constant pressure changed by $5\times10^{-3}$ to $7\times10^{-3}$ at every
tolerance of its buffer that was tried ($5\times10^{-3}$, $5\times10^{-4}$,
and $5\times10^{-5}$ kJ/mol/ps per atom). Its thermostat alone, without
the barostat, moves it by $+370$ to $+610$ kcal/mol/ns (`nsttcouple` 1 or
25, $\tau_T$ 1 or 10 ps, `nstcalcenergy` 1 or 100), against $-70$ for its
total energy at constant energy; that was not investigated further.

**The policy of the lists, matched.** GROMACS's rate in Table 10.5 is that
of its default policy. Its buffer is set from a tolerance on the energy
error of missed pairs (`verlet-buffer-tolerance`). With a thermostat and
the update on the device, GROMACS also searches every 50 steps instead of
10 and prunes a rolling inner list every 4 steps. MDIR's default misses
no pair: its lists are rebuilt when a displacement could bring a pair
within reach (Section 4). Table 10.7 compares the two programs on JAC with
the ensemble and the policy matched. RTX 3090 at 300 W (GPU 0), 50,000
steps of 2 fs, cutoff 8 Å, PME on a grid of $64^3$, the bonds of hydrogen
constrained, mixed precision, MDIR at 264b6d7 with groups and the dual
list (11 and 8.6 Å), GROMACS 2026.3 with its nonbonded terms, PME, bonded
terms, and update on the device.

*Table 10.7. JAC, ms per step (one or two runs each; the second, where
there is one, agreed within 0.001), and the drift of the conserved energy
in kJ/mol/ps per atom (MDIR: end minus start over 100 ps; GROMACS: its
"Conserved energy drift").*

| Ensemble and list policy | MDIR | GROMACS | MDIR / GROMACS, time | Drift: MDIR | GROMACS |
|---|---|---|---|---|---|
| NVE (GROMACS keeps a search every 10 steps) | 0.213 | 0.216 | 0.99 | $-1.0\times10^{-5}$ | $-1.7\times10^{-4}$ |
| NVE, GROMACS with an exact buffer (rlist 1.1 nm) | 0.213 | 0.261 | 0.82 | $-1.0\times10^{-5}$ | $-7.8\times10^{-5}$ |
| V-rescale, GROMACS searching every 10 steps | 0.215 | 0.225 | 0.96 | $-8.9\times10^{-6}$ | $-1.2\times10^{-4}$ |
| V-rescale, GROMACS default (search every 50, rolling prune every 4) | 0.215 | 0.147 | 1.46 | $-8.9\times10^{-6}$ | $-1.6\times10^{-4}$ |

With the same ensemble and the same list policy, MDIR's time of a step is
0.96 and 0.99 of GROMACS's (0.82 against GROMACS with an exact buffer).
MDIR's conserved energy drifts by $1.0\times10^{-5}$ and
$8.9\times10^{-6}$ kJ/mol/ps per atom and GROMACS's by $1.2\times10^{-4}$
to $1.7\times10^{-4}$ with its buffer set from its default tolerance, which
admits a bounded error from missed pairs by design. GROMACS's 0.147 ms is its search every
50 steps: its accounting of cycles shows 801 searches instead of 4001
over the run, 67 µs a step, which is the whole difference. It allows that
search only with a thermostat and the update on the device. MDIR rebuilds
its lists every 16.4 steps on average and prunes every 2.8. A smaller
skin makes it slower (pairlist 10 Å: 0.220 ms; 9.5 Å: 0.226 ms), so the
3 Å skin of D114 stays. MDIR's rate below 0.21 ms on this system would
need a list that lives longer, through a buffer set from a tolerance as
GROMACS's is. That is not the default, since a missed pair is not allowed
by default, and it would be an option.


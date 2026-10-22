# CPU step baseline

`bench.py` times the CPU target and CPU reference engines on the same
molecular systems. Issue #552 uses its NVE variant before selecting host
primitive strategies under #549. The existing `coupled` mode retains D267's
NVT and NPT size sweep.

Prepare each directory under `CPU_BENCH_DIR` with `sys.prmtop` and
`sys.inpcrd` in Amber format, and `gmx.top` and `eq.gro` for the same atoms
in the same order. `eq.cpt` and `npt.h5` are needed only for the coupled NPT
mode. JAC and the TIP3P boxes prepared for D267 have this layout. An Amber
suite system may use links to its prepared `system.parm7`, `system.rst7`,
`system.top`, and `system.gro`, respectively. Keep benchmark outputs in the
system directory; they are not repository inputs.

For each system, run one Slurm allocation with 16 cores and no other CPU
benchmark in that allocation. Give the engine binaries through `MDIR`,
`GMX`, and `PMEMD`. For example, the body of a submitted command for JAC is:

```sh
CPU_BENCH_DIR=... MDIR=... GMX=... PMEMD=... \
  python scripts/benchmarks/cpu-breakdown/bench.py jac 4 \
  m1,m16,d1,d16,g1,g16,x1,x16,p1 --ensemble nve --tag baseline-jac
```

Use `--cutoff 8` for JAC and Cellulose, `--cutoff 9` for TIP3P. The script
alternates the variant order by round. Discard round 0; report the mean and
sample standard deviation of the remaining rates. It writes one JSON line
per run and the exact input files under `runs/TAG/`. Do not mix executable
revisions in one tag. The MDIR rate is the second half of its steps;
GROMACS uses `-resethway`; pmemd reports all integration steps. Report this
timing-window difference with the comparison.

Use `--grid NX,NY,NZ` when a grid chosen from GROMACS's 1 Å spacing has
a prime factor other than 2, 3, or 5, which pmemd refuses. Specify one
nearby FFT-friendly grid for all three engines. Cellulose's automatic
grid is `280,128,128`; use `288,128,128` for the three-engine comparison.
Use `--min-steps` only for a pilot on a large system, and state the final
number of steps for every variant.

The NVE mode disables thermostats and barostats. All programs use a 2 fs
step, a common cutoff, PME with order 4 and tolerance $10^{-5}$, and
constraints on bonds to hydrogen. GROMACS chooses the PME grid at a 1 Å
spacing and the script gives that grid to MDIR and pmemd. GROMACS's default
Verlet policy is `gT`; `xT` uses an exact buffer of 3 Å and a rebuild every
10 steps. MDIR's default neighbor matrix uses a 1.5 Å reach; `mTrD` or
`dTrD` requests another reach. pmemd's neighbor policy is its own.
These are separate policy comparisons, not identical list algorithms.

The `p1` run uses the same Amber topology and coordinate file as MDIR.
Where the file includes velocities it restarts from those velocities; if
not, pmemd generates velocities from 300 K with a fixed seed. In the latter
case, label its rate contextual because the starting states differ.
GROMACS reads `eq.gro`; compare its coordinates and velocities with the
Amber input before calling a pair of rates fully matched. These checks
and the programs' versions belong in the report beside the rates.

Serial pmemd is available; `d16` is shown for the CPU target's thread
scaling but has no 16-thread pmemd counterpart. The script accepts
`--steps-scale X` for a short pilot. Final rates need enough steps for a
stable rate after initialization and pair-list builds.

# D[best-defaults]

Recorded at its merge.

`neighbor_structure = "AUTO"` takes TOWERS (D428) in `mdir run` and
`mdir check` where the device's entry of the traversal table admits the
input, and the rule of D261 otherwise.

## The rule

The traversal table is a table of the device policies of D420, keyed by
architecture like the others, with its value, its measurement basis and a
fallback flag. Its one entry is sm_89: TOWERS from 4,608 particles. The
default entry has no threshold, so every other architecture keeps D261,
and the reason says that TOWERS was not measured as the choice of AUTO on
that architecture.

Before the device is asked, the rule keeps D261 for an input that TOWERS
does not take or that belongs to the list structures:

- the CPU, a precision other than MIXED, the deterministic mode;
- a cell that is not periodic, a triclinic cell (not measured);
- a minimization, a program of frames or segments, a model or external
  term, implicit solvent, the derivative in the tunables;
- `rebuild_interval`, a given `pairlist_distance` or `pruned_distance`
  (TOWERS rebuilds at every evaluation and uses neither),
  `rigid_units = "ON"`.

Without a visible CUDA device the rule stops there and keeps D261. In
`mdir run` two more cases keep D261. The continuation of a checkpoint that
a list structure wrote keeps that structure, so the run keeps its stored
forces and barostat state; a change of traversal would recompute both.
A refusal of the TOWERS preflight, for example a cell fewer than three
cutoffs across, sends the run back to D261 with the refusal as the reason;
under an explicit `TOWERS` the same refusal is an error, as before.
`mdir check` runs the preflight without the setup trial and falls back the
same way. Python (`mdir.compile`) keeps D261: the preflight needs the
setup of `mdir run`.

## The launch

TOWERS taken by AUTO uses the derived launch of the preflight, untimed,
or the winner of an earlier trial cached for the same grid. The timed
trial of D432 compiled 48 candidates and took 62 to 72 s of setup
(`trial_cost_ms`) on each input below. In NVT and NVE its launch was 0 to
4.7% faster than the derived one (3.5% on JAC NVT, 4.0% on JAC NVE, 4.7%
on the box of 3,072 atoms); with the virial it was 1.5% faster on JAC NPT
and 3 to 5% slower on Factor IX and Cellulose NPT (tables). An explicit `neighbor_structure = "TOWERS"` keeps the trial.
At a fixed cell height every launch gives the same force and energy words
(D428), so the choice changes speed only.

## Record

The log has a line `MDIR: AUTO traversal: TOWERS taken: ...`,
`not taken: ...` or `refused at preflight: ...`, and `mdir check` the
same line after the skins and `towers_auto` in its JSON. The manifest
records `resolved_choices.neighbor_structure` (#952): category `rule`,
the value, the reason, the TOWERS reason, a refusal of the preflight, the
lookup of the table (architecture, threshold, fallback flag, basis), and
the key that overrides it, `[execution] neighbor_structure`. The reason of
the D261 structure is unchanged where AUTO keeps it.

## Measurements

RTX 4090, mixed precision, 2 fs, NVT with V-rescale every 10 steps, the
default PME of the time (tolerance $10^{-5}$, spacing 1.2 Å), the
controls of the benchmark without `[pme]`, `pairlist_distance` or
`neighbor_structure`. Two jobs in opposite orders, each arm twice per job
(ABAB, BABA), the 10%-trimmed mean of the row intervals after the first,
SM clock medians 2,565 to 2,760 MHz, at most 70 °C. Columns: ns/day of
D261 (main at 03506508), of AUTO TOWERS with the derived launch, and of
explicit TOWERS with the trial, the mean of the two jobs; the ratios are
per job.

| Input | Atoms | D261 | AUTO TOWERS | / D261 | Trial TOWERS | AUTO / trial |
|---|---:|---:|---:|---|---:|---:|
| TIP3P | 3,072 | 2,452 | 2,383 | 0.971, 0.972 | 2,500 | 0.953 |
| TIP3P | 4,608 | 2,311 | 2,396 | 1.038, 1.036 | 2,439 | 0.982 |
| TIP3P | 6,912 | 2,052 | 2,146 | 1.050, 1.042 | 2,204 | 0.974 |
| TIP3P 10k | 9,989 | 1,654 | 2,048 | 1.240, 1.237 | 2,061 | 0.994 |
| TIP3P 20k | 19,880 | 1,255 | 1,494 | 1.190, 1.191 | 1,497 | 0.998 |
| JAC | 23,558 | 1,260 | 1,401 | 1.110, 1.114 | 1,452 | 0.965 |
| Factor IX | 90,906 | 584.6 | 645.4 | 1.104, 1.104 | 646.2 | 0.999 |
| Cellulose | 408,609 | 153.0 | 169.3 | 1.106, 1.108 | 169.5 | 0.999 |
| STMV | 1,067,095 | 57.3 | 63.1 | 1.104, 1.100 | 63.1 | 1.000 |

The suite's NPT (C-rescale) and NVE systems at 2 fs, same protocol:

| Input | D261 | AUTO TOWERS | / D261 | Trial TOWERS |
|---|---:|---:|---|---:|
| JAC NPT | 1,207 | 1,284 | 1.065, 1.062 | 1,304 |
| JAC NVE | 1,331 | 1,493 | 1.123, 1.120 | 1,556 |
| Factor IX NPT | 570.9 | 607.4 | 1.063, 1.065 | 575.5 |
| Factor IX NVE | 613.9 | 677.5 | 1.103, 1.104 | 678.6 |
| Cellulose NPT | 150.9 | 160.5 | 1.055, 1.072 | 155.8 |

The smallest box, 3,072 atoms, ran 3% slower under TOWERS than on the
matrix, so the entry begins at the smallest box measured faster, 4,608
atoms. The rows of 3,072 to 6,912 atoms come from a separate pair of jobs
of these three arms. With the virial, the trial's launch was slower than
the derived one on Factor IX and Cellulose.

Against explicit MATRIX on TIP3P 10k with one feature each (Langevin,
Nosé–Hoover, leapfrog, isotropic, semi-isotropic and anisotropic
C-rescale, LJ-PME, restraints, POTENTIAL_SHIFT) and on a box of 13,824
OPC sites, AUTO took TOWERS, every run completed, and the relative RMS
force difference after 2 NVE steps of $10^{-9}$ ps was $4.4\times10^{-6}$
($6.4\times10^{-6}$ for OPC), the size of the mixed-precision difference
between the structures.

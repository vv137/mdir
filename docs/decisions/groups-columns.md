# D[groups-columns]

Recorded at its merge.

**The particles of a column of the compact order of the groups' build (Section 2 of `docs/neighbor-groups.md`) are an entry of the tuning policies (D420), `GroupsPolicy::columnParticles`: 96 by default, measured on `sm_89`, where the build had 64 since D89. The major version 12 (the RTX 5090, the RTX PRO 6000 Blackwell) keeps 64 by an entry of its own, measured on none of its devices, until a comparison there shows that 96 gains. The option `groups-column-particles` of `convert-md-exec-to-gpu` overrides the entry; the build over units keeps its own entry (`SegmentsPolicy::columnUnits`, 128, D421).** Issue #912 (the maintainer's ruling of 2026-11-18 on its results).

## Why

#912 found by removal that the loop over groups is bound by the throughput of its atomic reductions: an entry adds what it received to its particle with one f32 reduction a component at the end of each round. With them predicated off the loop takes 0.67 to 0.72 of its time; the shuffles and the table loads, under that bound, are worth 2 to 6%. The cost of the loop therefore follows the entries of the lists. A column of 96 particles instead of 64 makes a group a box of about 1:1:1.33 instead of a 1:1:2 prism. Its volume within the reach is smaller, so the lists hold 2 to 3% fewer entries (a model of Section 8.4 that reproduces MDIR's pools to 0.1%), and the build sorts fewer chunks.

## Measured

RTX 4090, mixed precision, NPT, the timing protocol (two jobs, the variants interleaved, about 100 intervals a run, the stationary trimmed mean). Ratio of the step to the columns of 64, job 1 / job 2:

| Particles a column | JAC | Factor IX | Cellulose | STMV | TIP3P (309,876 sites) |
|---|---|---|---|---|---|
| 64, µs a step | 169.3 / 162.3 | 349.6 / 336.4 | 1312.7 / 1279.4 | 4133 / 4044 | 1047.9 / 1017.9 |
| 80 | 0.978 / 0.979 | 0.982 / 0.983 | 0.984 / 0.982 | 0.992 / 0.993 | 0.996 / 0.997 |
| 96 | 0.979 / 0.979 | 0.978 / 0.978 | 0.971 / 0.967 | 0.971 / 0.972 | 0.992 / 0.992 |
| 112 | 0.988 / 0.987 | 0.987 / 0.986 | 0.970 / 0.968 | 0.989 / 0.987 | 0.993 / 0.991 |
| 128 (cubes) | 0.987 / 0.988 | | 0.979 / 0.978 | | 0.992 / 0.993 |
| 192 | 0.998 / 0.996 | | 0.984 / 0.986 | | 1.008 / 1.010 |

JAC and Cellulose ran controls with a relative RMS force error of about 4e-4; Factor IX and STMV the suite's controls. The details and the per-kernel split are in `docs/neighbor-groups.md`, Section 11.9.

**Forces.** The forces with 96 differ from those with 64 by 3.2e-7 (JAC) and 5.3e-7 (Cellulose) of the RMS force, against 1.2e-7 and 3.8e-7 for 64 against itself. The order of the sums changed. The error against a double-precision reference is unchanged: 3.978e-4 and 4.203e-4.

**Not measured on `sm_12x`.** The maintainer's ruling asked for the same comparison on an RTX 5090 or an RTX PRO 6000, against `main` on the same kind, with 64 kept there if 96 does not gain. Neither device could take a job while this PR was prepared: the RTX 5090's machine did not answer, and every core of the RTX PRO 6000's node was held by another job. The entry of the major version 12 therefore keeps 64, so the programs for `sm_12x` are those of before. Its measurement is open.

GATES

## Alternatives

- **The cube (128).** It shortens the lists about as much and builds them more slowly: 0.978 to 0.993 of the step.
- **Groups of 8.** By the model, 21 to 24% fewer slots but 53 to 57% more entries, so more reductions: not built (#912).
- **Fewer reductions an entry.** These are a change of the arithmetic of the sums, #917.

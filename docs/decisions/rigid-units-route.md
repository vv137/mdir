# D[rigid-units-route]

Recorded 2026-11-18.

**Slice S4b of D417. The driver takes the pair term over the segments of the rigid kinds of D403 by the key `[performance] rigid_units = "AUTO" | "ON" | "OFF"` (D417, Section 9, ruling 7), with no alias. `ON` takes the most numerous eligible kinds, at most two (D405), and refuses, naming the reason, a program where the route cannot apply. `OFF` is `main`'s general route. `AUTO` takes the general route until its rule is measured. One resolution before the build, next to that of `neighbor_structure` (D261), decides the route and records it: `mdir check`, the log, the manifest, and the fingerprint's execution part state, for each kind, whether it is taken and why. The covering radius of a taken kind is measured again at the positions that the run begins with and widened to them where the input lies beyond it (#795, ruling 5), so that the first evaluation, before any constraint, misses no pair. The test route `MDIR_RIGID_UNITS_TEST` is removed.** Issue #914, after D422.

Status: design accepted (Section 7); the implementation is in progress.

## 1. The key

```toml
[performance]
rigid_units = "AUTO"   # AUTO (the default), ON, OFF
```

- An execution choice of the "same accuracy" class (#790): the route changes the order of the sums, the f32 operations, and how the force of a site reaches its frame (rule A, D417, Section 1), not the potential. A continuation may change it, as it may change the keys of `[execution]`.
- `[performance]` is a new table. It holds this key alone; the other choices of #790 join it later.
- In Python, `Execution.rigid_units` (`mdir.RigidUnits.Auto`, the default, `On`, `Off`), and the resolution in `mdir.compile`'s description of the program beside `neighbor_structure_reason`.

| Value | The route |
|---|---|
| `AUTO` | The general route, with the reason "AUTO takes the general route until its rule is measured". `mdir check` still reports what `ON` would take. |
| `ON` | The units, for the kinds that Section 2 takes. A program where the route cannot apply is refused before anything is built, with the reason. |
| `OFF` | The general route: `main`'s program. |

## 2. What `ON` takes

The candidates are the rigid kinds of D403, numbered as `mdir check` numbers them. A kind is eligible unless one of these holds, each a reason that `mdir check` reports for it:
- some of its molecules have a pair of their own that is not excluded (D403's fact; the segment would leave the pair out);
- some of its molecules have an excluded pair with a particle outside the molecule.

A kind of one particle (an ion) is not taken, with the reason "one particle": its units would give the segmented loop nothing that the rest does not. `ON` takes the eligible kinds of two or more particles in a fixed order, by their count of molecules and then by their first particle (D403's order, which is `mdir check`'s numbering), at most two (D405's `MaxPairKinds`). An eligible kind after the two is not taken, with the reason "beyond the two most populous". The particles of no taken kind are the rest. The rest's anchors (#799) and its radius are those of D421, Section 3.

A kind that is not taken is a report, not a refusal. `ON` refuses where no kind is taken.

## 3. The refusals of `ON`

Each names the key and the reason, in this form:

```text
'rigid_units = "ON"' cannot apply: <reason>
```

| Reason | Where it is decided |
|---|---|
| the CPU target: lists of units run on a GPU | the control file |
| precision `SINGLE` or `DOUBLE`: the units take `MIXED` | the control file |
| the deterministic mode | the control file |
| a minimization | the control file |
| a cell that is not periodic | the control file |
| `[free_energy]` | the control file |
| `lennard_jones = "PME"` | the control file |
| `[pme] kernel` other than `GAUSSIAN` | the control file |
| frame, tunable, or derivative programs (the internal entries of D251 and #726, tunables, `observe`) | the control file, or the Python model |
| a system without a topology | the system |
| no rigid kind (the system has no rigid group of D403) | the system |
| no kind is eligible, followed by each kind's reason | the system |
| the covering reach, X Å (the pair list distance plus twice Y Å, the largest covering radius of the units), exceeds half the least face width of the cell, Z Å | the system and its cell at the input |
| a run that writes the forces of its frames (Python's `reports(forces=True)`) where a taken kind has a site: its written forces would not take the general transpose of the site (#916) | the reports (Section 7, ruling 3) |

The reach is also checked at every build, as since S2; a run whose cell shrinks below it stops there with the message of `mdrtSegmentsStopReach`.

Every refusal but that of the written forces is made by the resolution, which `mdir run`, `mdir emit`, `mdir check`, and `mdir.compile` all call before anything is built; the model of Python sets some of its conditions (tunables, observed constants) after the reader's checks, so the reader does not make them. The written forces are refused where a simulation's reports ask for them. Under `AUTO` and `OFF` none applies.

## 4. The radius at the input

D422 left the first evaluation of a run open: it packs the frames at the input positions, before any constraint, and the refresh that builds checks no extents. A place beyond its covering radius there may lose a pair within the cutoff at step 0.

The resolution measures, at the positions that the run begins with (the input's, or the checkpoint's), the distance of each place of each copy of a taken kind from its anchor, and of each anchored member of the rest from its anchor. A bond is taken in its image nearest the anchor; a site is placed by the kind's weights. Where the largest distance of a kind, or of the rest, exceeds its nominal radius (the constrained distances of the frame plus the margin of $10^{-4}$ nm, D421, Section 3), the radius of that kind becomes the measured distance plus the margin, for the whole run. An input within the nominal radius keeps it, and with it the bytes of its program: the radius is a constant of the program's type, so a radius that followed every rounding of the input would compile a new program for each state, as the positions after a constraint step differ from the constrained distances by their rounding. `mdir check` (text and JSON) and the log's header give, for each kind and the rest, both radii and the measured excess, so that a file of rounded coordinates is visible as such. The reach against the cell is checked with the widened radii, so an input far from its constraints is refused (`ON`) by the reach.

The measurement is on the host, in f64, once. It changes no device code.

**Why widening and not a stop.** The fixture of the rhombic dodecahedron (`test/Driver/Inputs/triclinic/dodecahedron.gro`, 403 waters, coordinates in nm to three decimals) has all 806 bonds O–H between 0.09600 and 0.10091 nm at the input, against the radius 0.09582 nm of the constrained frame. A stop at the input would refuse it, and every file of GROMACS's format at three decimals: its rounding alone moves a hydrogen by up to 0.00087 nm, more than the margin. #795's ruling 5 (2026-11-08) took this way: "at the first evaluation, a failed check widens the reach; after a constraint step, it stops the run with a message". After the first constraint step the writer of D422 checks the places against the widened radius, and stops the run where one lies beyond it.

## 5. The records

**`mdir check`**, after the rigid groups. Alanine dipeptide in OPC (`test/Driver/rigid-units-route.test`), whose inpcrd lies within the nominal radii:

```text
rigid units:        ON: the units (ON was given)
  kind 1:   taken, 387 units of 4 places (3 in the frame, 1 site), radius 0.8734 Ang
  the rest: 22 particles, 12 anchored, radius 1.0910 Ang
  reach:    12.18 Ang, half the least face width 12.58 Ang
```

The rhombic dodecahedron of the tests, whose .gro lies beyond them:

```text
  kind 1:   taken, 403 units of 3 places (3 in the frame, 0 sites), radius 1.0101 Ang (0.9582 constrained, widened by 0.052 Ang to the input's 1.0091)
```

D405's mixture of four kinds under `AUTO` (`test/Driver/check-rigid-groups.test`):

```text
rigid units:        AUTO: general (AUTO takes the general route until its rule is measured)
  kind 1:   ON would take: 3 units of 3 places (3 in the frame, 0 sites), radius 0.9582 Ang
  kind 2:   ON would take: 2 units of 4 places (3 in the frame, 1 site), radius 0.9582 Ang
  kind 3:   not taken: beyond the two most populous
  kind 4:   not taken: one particle
  the rest: 10 particles, 2 anchored, radius 1.2010 Ang
  reach:    12.4 Ang, half the least face width 15 Ang
```

Under `OFF` the line is `OFF: general (OFF was given)`, with no line of the kinds. `ON` that is refused is an error of `mdir check`, with its reason. A kind of three atoms that no kind takes, as kind 3 above, is in the rest: its atoms anchored to its first, as D421's rest takes a rigid water.

The JSON report has an object `run.rigid_units`: `requested`, `route` (`units` or `general`), `reason`, `kinds` (for each kind: `kind`, `taken` (by this run), `taken_by_on`, `reason`, `units`, `places`, `frame`, `sites`, and for a kind that `ON` takes `radius_angstrom`, `constrained_radius_angstrom`, `input_distance_angstrom`, `widened_angstrom`), `rest` (`particles`, `anchored`, and the same four lengths), `reach_angstrom`, `half_least_face_width_angstrom`.

**The log** prints the first line of `mdir check`'s report in its header, beside the line of the neighbor structure, where the key is given or the units are taken, and under `ON` the lines of the kinds, with their widening. **The manifest** has `rigid_units_requested`, `rigid_units`, `rigid_units_reason`, and `rigid_units_route` (the JSON report's object), beside `neighbor_structure_*` (`docs/run-manifest.schema.json`). **Python**: `Program.plan["rigid_units"]`, `["rigid_units_requested"]`, `["rigid_units_reason"]`, and `["rigid_units_kinds"]`, the numbers of the kinds taken.

**The fingerprint** (D172): `[performance] rigid_units` in its execution part, only where the key is given. A checkpoint of `main`, or of a run without the key, continues without a note; a change of the value between two runs is a change of execution, which a continuation may make.

**The skin search** (D281) compiles its candidates from the resolved control, so they take the same route and the same radii.

## 6. The test route

`MDIR_RIGID_UNITS_TEST` and its row in `docs/environment.md` are removed. Its tests move to the key:
- `test/Driver/rigid-units-test-invalid.test` becomes `rigid-units-route.test`, the test of the key's refusals, one case for each row of Section 3 but the written forces (`rigid-units-written-forces-gpu.test`), and the cases of the numbers of kinds, which no longer exist, become the cases of Section 2 (no rigid kind, a kind of one particle alone, three eligible kinds and an ion, the facts of D403 broken);
- `segments_contract.py`, `segments_trajectory.py`, `segments-schedule-ptx.test`, and the sanitizer's `segments-traverse-gpu.test` take `rigid_units = "ON"` (Python: `Execution.rigid_units`).

## 7. The maintainer's ruling (2026-11-18, on #915)

The recommendations were accepted:
1. **The radius at the input is widened, not a stop** (Section 4), as #795's ruling 5 did. The widening and the measured excess are reported for each kind by `mdir check`, in text and JSON, and in the log's header. The run stops only where the widened reach does not fit the cell, and at the writer after the first constrained step (D422).
2. **More than two eligible kinds: two are taken and the rest reported**, with no refusal (Section 2). Kinds of two or more particles come first; a kind of one particle is not taken by `ON`. Then by count, then by first particle. The reasons are "one particle" and "beyond the two most populous". A negative test has three eligible kinds of several particles and one kind of ions.
3. **Written frame forces with a taken kind that has a site: `ON` refuses**, with its reason (Section 3). The general transpose of #795's ruling 6 is #916.

## 8. Validation

**#795's validation table, moved to the key.**

| #795's row | Here | Result |
|---|---|---|
| The physical fixtures against the CPU f64 oracle and general double | `rigid-units-frames-gpu.test` (`Inputs/rigid_units_frames.py`), D405's mixture (TIP3P, a four-site water with a linear site, rigid molecules of three Lennard-Jones atoms in the rest, an ion), plain cutoffs, the input and two later frames | within the bounds at every frame, the largest ratio to its bound 0.46; the oracle and the general double route agree within 5e-13 kJ/mol and 4e-12 kJ/mol/nm (on the dodecahedron within 1.1e-10 and 8.5e-10) |
| The anchors (alanine dipeptide in OPC under NPT, in TIP3P under NVT, PME) | the same harness, four frames | largest ratio 0.67 in OPC (from the first step on) and 0.67 in TIP3P (from the input on) |
| The triclinic cell (the rhombic dodecahedron of 403 TIP3P waters, PME) | the same harness, three frames, the input's radius widened: with plain cutoffs from the input on, against the oracle as well; with PME from the first step on | largest ratio 0.92 (plain cutoffs) and 0.78 (PME) |
| The reach's stop | `rigid-units-reach-gpu.test`: at the resolution (`rigid-units-route.test`, `FAR`) and at a build under NPT at 2000 atm (12.38 Ang against half the shrinking face width) | the run stops with both lengths; under `OFF` it runs to its end |
| The control's refusals | `rigid-units-route.test`, `rigid-units-written-forces-gpu.test` | every refusal of Section 3 with its message |
| The short trajectories and the coverage (S4a) | `segments-trajectories-gpu.test`, `segments-writer-coverage-gpu.test`, `md-alg-segments-contract-gpu.test`, now on the key | pass |

The criterion of the same-frame fixtures is #795's: the units against the general double route within 1.5 times the general mixed route's difference, or 1e-5 of the largest force and 1e-6 of the potential, the forces projected onto the constraints of the waters. The general route is that of the groups (D415), whose sums in mixed precision are made in f32 and widened once as the units' are; its difference is the largest of 5 runs at each frame and over the frames (D422's lesson: a baseline of one sample flickers). The matrix makes its sums otherwise: against it, the units' potential at the dodecahedron's input differed from the double route's by 2.9e-2 kJ/mol where the matrix's did by 9.5e-4.

**The potential at the dodecahedron's input.** The input of the dodecahedron holds close contacts: its Lennard-Jones energy is 12,380 kJ/mol there and 3,145 kJ/mol 25 steps later. At the input the units' potential differs from the double route's by 2.93e-2 kJ/mol with PME and 2.76e-2 with plain cutoffs, the groups' by 1.84e-2 and 1.99e-2 (the largest of 5 runs), 1.59 and 1.39 times; with the Lennard-Jones alone, 2.97e-2 against 2.03e-2. At the later frames both are smaller (with plain cutoffs 1.8e-3 and 4.3e-3 for the units, 9.8e-4 and 1.5e-3 for the groups, within the bound of 3.0e-2 that the input sets). The forces agree as closely as the groups' at every frame. This is the rounding of the sums in f32 of a frame whose terms are large, about 2e-6 of the Lennard-Jones energy, not a pair left out: a missed pair within the cutoff would change the potential by tens of kJ/mol with plain cutoffs. The fixture compares the dodecahedron's input with plain cutoffs, against the oracle as well, and with PME from the first step on.

**The input of a site that is not linear in its frame.** Amber's extra point of OPC lies at $\boldsymbol\phi(\mathbf x)$, which equals the weights' $\sum_k w_{sk}\mathbf x_k$ on the constrained frame only. At an input off the constraints the two routes therefore evaluate the site at different places: on alanine dipeptide in OPC the units' potential at the input differed from the general double route's by 0.32 kJ/mol (the general mixed route's by 8.8e-3), and the forces by 0.60 kJ/mol/nm (4.9e-2). From the first step on, where SETTLE has constrained the frame, they agree within the bounds; the fixture compares OPC from there (`--after-input`). A run on the units thus reports a potential at step 0 that depends on how far its input lies from its constraints.

**Mutations, each tried once and reverted** (`rigid-units-route.test`, rebuilt in a tree of its own):
- every refusal of Section 3 removed in turn (the CPU target, the precision, the deterministic mode, a minimization, an open cell, `[free_energy]`, LJPME, the kernel of PME, the derivative programs, no topology, no rigid kind, the reach): each makes its case fail;
- the choice of the kinds: without the rule of one particle, of the facts of D403 (incomplete, outside), or of the limit of two, the report of the mixture fails;
- the widening at the input removed: the dodecahedron's report fails;
- the reduction of a bond to its image nearest the anchor removed along c and along a: the split waters' reports fail;
- the radii of the two taken kinds exchanged (a realistic bug of the plan): `rigid-units-frames-gpu.test` fails, the writer stopping the mixture's run after its first step with the message of the extents.

**`OFF` and the programs without a taken kind** (D406's corpus, `check.py compare --stages lowered --targets cpu,gpu --kinds run,segments --precisions mixed,double --execution neighbor_structure=GROUPS --instances`, `MDIR_GPU_BINARY=ptx`, against a fresh tree of `main` at 61eafe0d): `sm_89` and `sm_120` each 4,420 cases, 4,372 identical (2,466 the same error in both). The 48 others are 16 cases each of three control files: the list of valid tables in the message of `keyword-hints.test`, which now names `[performance]`, and the two new tests' files that give `rigid_units = "ON"`, which `main` refuses as an unknown table.

**The key against the test route.** With `rigid_units = "ON"`, each of #795's six systems, from its checkpoint, lowers (`sm_89`) to the same bytes as `main`'s test route (`MDIR_RIGID_UNITS_TEST=1`), and so does alanine dipeptide in OPC from its inpcrd: S4a's measurements (D422) hold for the key.

**The checkpoints.** A checkpoint that `main` writes and one that this branch writes without the key (CPU, double, 10 steps of alanine dipeptide in OPC) hold the same positions, velocities, forces, and fingerprint. This branch continues `main`'s checkpoint without a note; with `rigid_units = "OFF"` given, the note of a change of execution names `[performance] rigid_units: (none) -> "OFF"`.

**The sanitizer.** No device code changes. The two tests of the tier that the key now drives (`segments-traverse-gpu.test`, `segments-writer-gpu.test`) pass under memcheck, racecheck, and initcheck.

## 9. Performance

RTX 4090s through the scheduler, by the timing protocol: two jobs, two rounds each, the second in reverse; `main` at 61eafe0d (a fresh tree), this branch without the key (`AUTO`, general), and, on #795's six systems, this branch with `rigid_units = "ON"`, interleaved; wall runs of about 100 intervals (200,000 steps on #795's systems; the suite's from 10,000 to 200,000 by its rate), the trimmed-mean stationary rate; every run became stationary. The suite's first pair of jobs had a disturbed second round (both builds slower by up to 1.85 times in the same runs); it was run again in two jobs of its own, which the table gives.

| System | `main`, ms/step | `AUTO`, ms/step | `AUTO` / `main` | `ON` / `main` |
|---|---:|---:|---:|---:|
| rd-opc | 0.1572 | 0.1568 | 0.9975 (0.9950–0.9997) | 0.793 (0.790–0.796) |
| rd-tip3p | 0.1283 | 0.1286 | 1.0024 (1.0005–1.0053) | 0.897 (0.895–0.898) |
| ubq-opc | 0.1734 | 0.1736 | 1.0011 (0.9997–1.0034) | 0.773 (0.770–0.777) |
| ubq-tip3p | 0.1378 | 0.1382 | 1.0028 (0.9994–1.0057) | 0.885 (0.884–0.886) |
| opc-large | 0.2870 | 0.2870 | 0.9999 (0.9968–1.0022) | 0.727 (0.715–0.740) |
| charmm-tip3p | 0.1876 | 0.1874 | 0.9989 (0.9983–0.9993) | 0.790 (0.784–0.796) |
| jac_nve | 0.1423 | 0.1424 | 1.0007 (0.9987–1.0030) | — |
| jac_npt | 0.1560 | 0.1558 | 0.9990 (0.9977–1.0004) | — |
| jac_nve_4fs | 0.1528 | 0.1530 | 1.0014 (0.9989–1.0036) | — |
| jac_npt_4fs | 0.1611 | 0.1614 | 1.0022 (0.9997–1.0050) | — |
| factorix_nve | 0.3209 | 0.3215 | 1.0021 (0.9986–1.0108) | — |
| factorix_npt | 0.3436 | 0.3437 | 1.0002 (0.9966–1.0055) | — |
| cellulose_nve | 1.2539 | 1.2548 | 1.0007 (0.9990–1.0025) | — |
| cellulose_npt | 1.3104 | 1.3092 | 0.9991 (0.9965–1.0011) | — |
| stmv_npt_4fs | 4.1204 | 4.1139 | 0.9984 (0.9943–1.0019) | — |

`AUTO` keeps `main`'s program, and its rate is `main`'s within the spread of the rounds on every system. `ON` repeats D422's ratios on #795's six systems.

## 10. Gates

- #795's validation table, moved to the key: the physical fixtures against the CPU f64 oracle and general double, the anchors and the triclinic cell, the reach's stop, and the control's refusals.
- `OFF`, and every program without a taken kind, identical to `main`: the corpus (D406, `sm_89` and `sm_120`) and the checkpoints.
- End-to-end timings on #795's six systems and the Amber suite against `main`, by the timing protocol, with `AUTO` general.
- A negative test for every refusal; the mutation checks recorded.
- The full suite at the final head.

## 11. Open

- **`AUTO`'s rule**, to be measured; Cellulose under NPT (1.115 to 1.120 of `main`'s wall, D422) is its counterexample.
- **Written forces** by the general transpose of a site (#916).
- **The potential at the input of a site not linear in its frame** (Section 8): the units evaluate it at the weights' place.

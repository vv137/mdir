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

**`mdir check`**, after the rigid groups (the numbers are illustrative):

```text
rigid units:        ON: the units (rigid_units = "ON")
  kind 1:   taken, 2,148 units of 4 places (3 in the frame, 1 site), radius 0.8895 Ang
  kind 2:   not taken: 12 molecules with a pair of their own that is not excluded
  the rest: 22 particles, 12 anchored, radius 1.0901 Ang
  reach:    12.18 Ang, half the least face width 12.58 Ang
```

With a radius widened at the input, its line ends `radius 1.0091 Ang (0.9582 constrained, widened at the input)`. Under `AUTO`, the first line is `AUTO: general (AUTO takes the general route until its rule is measured)`, and the lines of the kinds say what `ON` would take. Under `OFF`, `OFF: general`, and no line of the kinds. `ON` that is refused is an error of `mdir check`, with its reason.

The JSON report has an object `rigid_units`: `requested`, `route` (`units` or `general`), `reason`, `kinds` (for each kind: `kind`, `taken`, `reason`, `units`, `places`, `sites`, `radius_angstrom`, `constrained_radius_angstrom`), `rest` (`particles`, `anchored`, `radius_angstrom`), `reach_angstrom`, `half_least_face_width_angstrom`.

**The log** prints the first line of `mdir check`'s report in its header, beside the line of the neighbor structure, and the lines of the taken kinds. **The manifest** has `rigid_units_requested`, `rigid_units`, and `rigid_units_reason`, beside `neighbor_structure_*`, and the kinds as in the JSON report.

**The fingerprint** (D172): `[performance] rigid_units` in its execution part, only where the key is given. A checkpoint of `main`, or of a run without the key, continues without a note; a change of the value between two runs is a change of execution, which a continuation may make.

**The skin search** (D281) compiles its candidates from the resolved control, so they take the same route and the same radii.

## 6. The test route

`MDIR_RIGID_UNITS_TEST` and its row in `docs/environment.md` are removed. Its tests move to the key:
- `test/Driver/rigid-units-test-invalid.test` becomes the test of the key's refusals, one case for each row of Section 3, and the cases of the numbers of kinds, which no longer exist, become the cases of Section 2 (no kind, no eligible kind, a third eligible kind);
- `segments_contract.py`, `segments_trajectory.py`, `segments-schedule-ptx.test`, and the sanitizer's `segments-traverse-gpu.test` take `rigid_units = "ON"` (Python: `Execution.rigid_units`).

## 7. The maintainer's ruling (2026-11-18, on #915)

The recommendations were accepted:
1. **The radius at the input is widened, not a stop** (Section 4), as #795's ruling 5 did. The widening and the measured excess are reported for each kind by `mdir check`, in text and JSON, and in the log's header. The run stops only where the widened reach does not fit the cell, and at the writer after the first constrained step (D422).
2. **More than two eligible kinds: two are taken and the rest reported**, with no refusal (Section 2). Kinds of two or more particles come first; a kind of one particle is not taken by `ON`. Then by count, then by first particle. The reasons are "one particle" and "beyond the two most populous". A negative test has three eligible kinds of several particles and one kind of ions.
3. **Written frame forces with a taken kind that has a site: `ON` refuses**, with its reason (Section 3). The general transpose of #795's ruling 6 is #916.

## 8. Gates

- #795's validation table, moved to the key: the physical fixtures against the CPU f64 oracle and general double, the anchors and the triclinic cell, the reach's stop, and the control's refusals.
- `OFF`, and every program without a taken kind, identical to `main`: the corpus (D406, `sm_89` and `sm_120`) and the checkpoints.
- End-to-end timings on #795's six systems and the Amber suite against `main`, by the timing protocol, with `AUTO` general.
- A negative test for every refusal; the mutation checks recorded.
- The full suite at the final head.

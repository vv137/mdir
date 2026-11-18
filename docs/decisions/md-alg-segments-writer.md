# D[md-alg-segments-writer]

Recorded 2026-11-18.

**Slice S4a of D417. The writer of a list of units: one kernel a step, after the last op that writes the positions and before the first loop that reads the list. It packs the f32 frames of the units (D95; D417, Section 4.3). It tests the units' reference points against the least half-margin of the segments, in f32 with a counted margin of rounding (one mobility group, D419, Section 3). It checks that every place of every unit lies within its covering radius, by the convex rule of D417, Section 3. Its results are flags of D113, which the host reads where the refresh needs them, after the launches of the reciprocal sum (#795's U3). The refresh no longer tests or reads back. The loops over the list read the writer's frames.** Issue #905, after D421 (the maintainer: "do S4a after the train", 2026-11-18). The leader's rulings on #897 and #906: the test fused into the writer is a requirement of S4a; the check of the extents is in the writer; the test is in f32.

## 1. The op and its place

```mlir
%moved, %stale = md_exec.write_unit_frames %l at(%x, %cell)
                   cutoff(0.9) skin(0.1) prune_skin(0.05)
                   : !sp, !md.field<@atoms, 3 x f64>
%l1 = md_exec.refresh_segmented_pairs %l at(%x, %cell) moved(%moved) stale(%stale)
        cutoff(0.9) skin(0.1) prune_skin(0.05) : !sp, !md.field<@atoms, 3 x f64>
```

- `md_exec.write_unit_frames` takes a list of units, the positions and cell of a step, and the refresh's reaches. It writes the list's frames at those positions and returns two flags:
  - `moved`: some reference point has moved, since the build, by more than the least half-margin $\chi$ of the segments (paper, Section 4.7), or the list has not been built;
  - `stale`: the same against the last pruning of a dual list, with the inner reach.

  It stops the run where a place of a unit lies beyond its covering radius.
- `md_exec.refresh_segmented_pairs` takes `moved` and `stale` as `md_exec.refresh_neighbors` does (D80), and then makes no test of its own. Its verifier takes them only from one writer of its own list, positions, cell, and reaches (`test/Dialect/MDExec/segments-writer-invalid.mlir`, nine cases).
- **The rule of the writer** (the leader's ruling on #906): *after the last writer of the positions and before the first reader of the frames.*
  - `md-exec-expose-validity` places the writer, in the value form, right after the op that gives the refresh's positions. That is the last op of the step that writes them, after the constraints and a barostat's scaling (`test/Dialect/MDExec/Transforms/expose-validity-segments.mlir`).
  - Before any op is lowered, `convert-md-exec-to-gpu` refuses an op that writes the positions between the writer and its refresh, or between the refresh and a loop that reads the frames. A loop at positions other than its refresh's packs the frames itself, as in S3.
  - A refresh of an empty list builds whatever has moved, and takes no writer. It packs the frames itself.

**Why the check of the extents is in the writer.** D417's table in Section 3 put it in the constraint kernel. What "after each position constraint" protects is that no loop reads a frame whose extents were not checked. The writer runs after the last writer of the positions, the barostat's scaling under NPT included, and before every loop over the list. It holds every member of the unit, so it reads no member twice, and the constraint kernels of programs without a taken kind keep their bytes. The leader accepted the deviation on that condition (#906).

## 2. The kernel

`convert-md-exec-to-gpu` lowers the writer to one launch, a thread for each member of each unit. One thread a unit took 9.3 µs a call on the large OPC box: it was latency-bound, with 128 threads to an SM. Three threads to a water hide the dependent loads, and the same box takes 6.8 µs. The launch is behind a flag of the host that says the list has been built, so the first refresh, which builds, has none.
- **The anchor's thread** writes the anchor, $\mathbf x_0$ moved by the shift of its place in f64 and narrowed to f32. It also writes the fields that the loops read (found from the traversals that take the refresh's result) and tests the reference point.
- **The thread of member $k \ge 1$** writes its offset $\mathbf h_k$, formed in f64 as the image of its bond nearest the anchor and narrowed to f32. The image is found in f32, and in f64 only where the f32 image moves the bond (a molecule split by a face). The thread checks its distance from the anchor.
- **The rest.** A particle of the rest writes its own frame and tests its reference point if it is its own. An anchored member checks its distance from its anchor against $\rho$, and its anchor's unit tests the anchor.
- **The buffers** belong to the list: frames 0 and 1 (the anchors and the offsets) and one frame a field, `mdrtGroupsFrame` of the runtime, sized by the places and dropped with the other buffers at a growth (`GroupsABI.h`, ABI 3). After a build, the refresh packs every place, empty places as zeros.
- **The flags.** `moved`, `stale`, and the failure of the extents are three flags of D113, set by plain stores. `replaceWithFlags` waits for them before their first use. That is the stop of the extents, placed just before the refresh, and in a step it follows the launches of the reciprocal sum.

**The kernel** on `sm_89`: 40 registers, 440 SASS instructions, 66 of them of FP64. Most of those are on the rare paths, the f64 image of a split bond and of an anchored member. The executed f64 work of a water is the anchor's shift, the two bonds, and their narrowing. One attempt to shorten it, the threads numbered in 32 bits, changed nothing measurable (6.84 to 6.81 µs a call on the large OPC box). The writer without its test took 6.16 µs there and 3.47 µs on ubiquitin in OPC. So the test adds 0.65 and 0.70 µs a call, and the packing is the rest. #795's writer: 40 registers, 304 instructions, 48 of FP64, 2.7 to 3.3 µs a call.

## 3. The test in f32

The test is in f32, with a counted margin of rounding (the leader's ruling on #906). Its derivation is in the paper, Section 4.7:
- With $a = \mathrm{fl}(\mathbf a)$, $t = \mathrm{fl}(\mathrm{fl}(\mathbf a^\text{ref})\,\mathrm{fl}(\mathbf m))$ and $\mathbf d = \mathrm{fl}(a - t)$, the true displacement is at most $\lVert\mathbf d\rVert + \delta_\chi$, where $\delta_\chi = 2^{-20}(\lVert a\rVert_1 + \lVert t\rVert_1)$.
- The test passes where $\mathrm{fl}(\mathbf d\cdot\mathbf d) \le \mathrm{fl}(\mathrm{fl}(z^2)(1 - 2^{-18}))$, with $z = \mathrm{fl}(\chi - \delta_\chi) > 0$ and $\chi$ narrowed below itself. No unit that moved by more than $\chi$ passes.
- A displacement or a margin that is not a number fails.

The check of a radius $e$ compares the f32 square of an offset narrowed from f64 with the largest f32 at most $e^2(1 - 2^{-20})$. A site outside the hull of its frame is formed in f64 from the f64 offsets.

## 4. Validation

**The check of the extents** (`test/Integration/segments-writer-extents-gpu.test`, `Inputs/segments_writer.py`). Two frames; the writer of the second checks the list that the first built. The run must end where every place is within its radius, and stop with the message of the extents in three probes:
- *convex*: a hydrogen 0.11 nm from its oxygen, against the kind's 0.09582 nm;
- *hull*: a site of weights (3, −1, −1) 0.117 nm from its anchor, against 0.1 nm, with its frame atoms within it;
- *anchor*: an anchored member of the rest 0.12 nm from its anchor, against $\rho$ = 0.10901 nm.

Each check was removed once and restored, and each time its own probe ended while the others still stopped:
- without the check of a frame atom, the convex probe ended;
- with every site taken as convex, the hull probe ended;
- without the check of an anchored member, the anchor probe ended.

**The margin of the test.** The same harness moves eight waters whole by $\chi(1 + 2^{-23})$, $\chi$ = 0.05 nm, in directions for which the f32 test without its margin passes every one of them: the generator simulates the kernel's arithmetic. The list must be built again (2 builds). Moved by $\chi(1 - 2^{-10})$, it must not be (1 build). With the margin removed from the kernel ($z = \chi$, factor 1), the list was not built again and the test failed. Tried once and reverted.

**The coverage of S2 with the writer** (`test/Integration/segments-writer-coverage-gpu.test`). The refresh takes the writer's test at the positions of every particle, and the exact check of D419 runs after every frame. The trajectories:
- the rhombic dodecahedron of TIP3P under NPT, 40 frames;
- alanine dipeptide in OPC under NVT and in TIP3P under NPT, 60 frames each;
- the mixture of two kinds ($K = 2$), 30 frames;
- growth, each molecule moved whole while the cell grows by half.

No pair is without an entry, none is listed twice, none is in a wrong segment, and none is excluded. Mutations, each tried once and reverted:
- the half-margin taken at the largest radii alone: growth leaves a pair without an entry;
- the outer test never set: the dodecahedron leaves 20,714 pairs without an entry, and the margin probe makes 1 build.

**#795's short trajectories** (`test/Driver/segments-trajectories-gpu.test`, `Inputs/segments_trajectory.py`). Three waters of OPC and of TIP3P, 1 fs, 100 steps, both integrators, NVE and NPT. The units are compared with the general path in double precision, deterministic, against #795's declared bounds:
- positions, the volume, and the velocities within 3 times the general mixed route's differences, or 1e-5 of the scale;
- the velocities compared at the step and projected onto the constraints, as in #795's fullVelocity;
- the baseline is the largest difference of 5 general mixed runs at each sample, recorded once in `Inputs/segments_trajectory_bounds.json`. A baseline of one run changes from run to run, and the gate flickered (the leader's ruling on #906).

| Case | Largest ratio to its bound | Repeats passed |
|---|---:|---:|
| OPC, velocity Verlet, NVE | (filled at the head) | 10 of 10 |
| OPC, velocity Verlet, NPT | | 10 of 10 |
| OPC, leapfrog, NVE | | 10 of 10 |
| OPC, leapfrog, NPT | 0.87 | 10 of 10 |
| TIP3P, velocity Verlet, NVE | | 10 of 10 |
| TIP3P, velocity Verlet, NPT | | 10 of 10 |
| TIP3P, leapfrog, NVE | | 10 of 10 |
| TIP3P, leapfrog, NPT | | 10 of 10 |

A mutation of the writer, each offset written to the row of the next member, takes the OPC case beyond its bounds at the first sample. Tried once and reverted.

**The velocities of leapfrog under the barostat.** With the bound of one general mixed run, OPC, leapfrog, NPT exceeded the bound on its velocities in 5 of 6 runs, and the same case on `main`'s tree (S3, without the writer) in 1 of 2. The units' trajectory is the same to the bit on both trees and from run to run.
- **The unprojected velocities.** The unprojected full-step velocities of the units and of the double route differ by 0.068 to 0.10 nm/ps. That difference lies wholly along the constraints, and it equals the component along them of $h\,\Delta\mathbf F/m$, the difference of the carried forces. The forces of a site reach the atoms by the weights on the units and by the Jacobian on the general path (D421, Section 5).
- **The projected velocities.** Projected onto the constraints, the difference is $2.6\times10^{-6}$ nm/ps at step 10 and grows about twofold every 20 steps, to $8.1\times10^{-5}$ at step 100. It is a divergence that grows from the tolerance of the projection, as the general mixed route's own difference grows. It is not a defect: the same case on the S3 path behaves alike, and the coupling of leapfrog takes the projected $\mathbf v_{n+1}$ (below).
- **The coupling.** It takes the projected $\mathbf v_{n+1}$, so its kinetic energy does not depend on the route. In `DynamicsIR.cpp` the kick to $\mathbf v_{n+1}$ is projected before the thermostat, the barostat, the pressure, and the log read it, and `docs/dynamics.md` now says so. On alanine dipeptide in OPC the kinetic energy of the unprojected velocity is 8 to 13.5% above the projected one, and the log's agrees with the projected one within 0.3%.

**The corpus** (D406, `check.py compare ... --stages lowered --targets cpu,gpu --kinds run,segments --precisions mixed,double --execution neighbor_structure=GROUPS --instances`, `MDIR_GPU_BINARY=ptx`, against a fresh tree of `main` at 79582a24):
- `sm_89`: 4,388 cases, 4,388 identical (2,482 of them the same error in both);
- `sm_120`: (filled at the head).

**The sanitizer tier** (`test/Sanitizer/segments-writer-gpu.test`, and `segments-traverse-gpu.test`, whose runs take the writer of each step). memcheck, initcheck, and racecheck pass. It held an RTX 4090 for 81 s.

## 5. Performance

RTX 4090s through the scheduler, by the timing protocol:
- two jobs on two devices, two rounds each, the second in reverse;
- `main` at bab3f1fe, #795 at 745c9f12, and the units interleaved;
- wall runs of 200,000 steps (100 intervals; the trimmed-mean stationary rate) and captures of 60,000 steps;
- every run became stationary;
- the steps of a capture counted by the gather of PME, once a step on every route.

**S4a's gate** (the leader's ruling on #897). The list and the whole writer of the units a step against #795's list a step, paired within a job and round, in µs:

| System | Units: list + writer | #795: list | Difference | Spread of #795's rounds |
|---|---:|---:|---:|---:|
| rd-opc | 6.63 (6.50–6.74) | 11.06 (10.83–11.43) | −4.44 | 0.60 |
| rd-tip3p | 6.81 (6.70–6.93) | 11.64 (11.53–11.72) | −4.83 | 0.19 |
| ubq-opc | 6.99 (6.87–7.09) | 11.88 (11.75–12.01) | −4.89 | 0.26 |
| ubq-tip3p | 7.19 (7.04–7.36) | 14.39 (12.89–18.71) | −7.20 | 5.81 |
| opc-large | 9.37 (9.21–9.51) | 16.12 (13.20–24.74) | −6.75 | 11.54 |
| charmm-tip3p | 7.88 (7.75–8.08) | 18.48 (16.00–24.61) | −10.60 | 8.60 |

The gate is met on every system, with the whole writer counted. Only its test, 0.7 µs a call, is the test of the gate, and #795's writer, 2.7 to 3.3 µs a step, is not in its list.

**The wall a step** (ms; units against #795 and `main`, paired within a job and round):

| System | #795 | `main` | Units | Units / #795 | Units / `main` |
|---|---:|---:|---:|---:|---:|
| rd-opc | 0.1354 | 0.1541 | 0.1220 | 0.902 (0.901–0.902) | 0.792 (0.791–0.792) |
| rd-tip3p | 0.1192 | 0.1261 | 0.1128 | 0.947 (0.944–0.949) | 0.895 (0.894–0.895) |
| ubq-opc | 0.1452 | 0.1705 | 0.1317 | 0.907 (0.903–0.913) | 0.773 (0.770–0.774) |
| ubq-tip3p | 0.1307 | 0.1356 | 0.1201 | 0.918 (0.916–0.920) | 0.886 (0.884–0.887) |
| opc-large | 0.2130 | 0.2826 | 0.2074 | 0.974 (0.933–0.989) | 0.734 (0.731–0.735) |
| charmm-tip3p | 0.1622 | 0.1840 | 0.1468 | 0.905 (0.899–0.911) | 0.798 (0.793–0.802) |

The units' wall falls below #795's and `main`'s on every system. In S3 it was above #795's on all six and above `main`'s on the two TIP3P systems with a solute (1.019 to 1.027).

**#795's target**, a four-site water with a protein at the rate of `main`'s TIP3P of the same system, measured as the units' wall against `main`'s in the same jobs and rounds:
- units of rd-opc against `main`'s rd-tip3p: 0.968 (0.965–0.969);
- units of ubq-opc against `main`'s ubq-tip3p: 0.972 (0.969–0.975).

It is met on the RTX 4090. The RTX 5090, where it was first defined, was down.

**The evaluation a call**, units against #795: 0.885, 1.042, 0.887, 0.983, 1.069 (1.060–1.075), and 0.961 in the order of the table. In S3's round 2, rd-tip3p was 1.027 and opc-large 0.996. The kernel and the layout of its frames are those of S3. A likely cause, not checked: the frames are now written before the reciprocal sum, so they have left the L2 cache by the time the loop reads them, where S3 packed them just before the loop.

**The step**, ubiquitin in OPC. The ranges are of the four captures under NVE, and of one capture of 20,000 steps under NPT:

| | #795 | `main` | Units |
|---|---:|---:|---:|
| NVE: a step, µs | 151.2–155.1 | 174.1–179.2 | 136.2–140.5 |
| device idle a step, µs | 19.5–20.1 | 18.4–19.2 | 16.8–17.3 |
| kernels a step (launches) | 26.9 | 27.4 | 25.9 |
| graph launches; streams | 0; 1 | 0; 1 | 0; 1 |
| blocking copies a step (µs) | 0.06 (10.0–10.3) | 0.10 (9.3–9.7) | 0.04 (1.4–1.6) |
| waits for events a step (µs of the host) | 1.00 (41.0–45.6) | 3.00 (61.4–67.5) | 2.00 (37.9–42.6) |
| integration and constraints a step, µs | 22.5–23.2 | 22.6–23.3 | 22.6–23.3 |
| the writer a step, µs | 2.7 | — | 4.1–4.2 |
| NPT (C-rescale): a step, µs | 350.0 | 219.2 | 188.7 |
| blocking copies a step under NPT (µs) | 0.66 (173.8) | 0.70 (32.1) | 0.63 (22.5) |

S3's per-step copy of the refresh's test (1.04 a step, 57 µs, and 37.4 µs of idle device) is gone.

**The waits that remain**, with their causes. Every step the host waits twice for an event:
- for the writer's flags, which decide whether the host launches a build;
- for the flag of the constraints (D278), as on `main`.

The host waits 38 to 43 µs a step in all, while the device works. They are the waits that a later step to graphs of CUDA would have to move onto the device.

## 6. Open

- The evaluation a call above S3's on the large OPC box and rd-tip3p (Section 5). Its cause is a hypothesis, to be measured with the candidates of #904.
- The first evaluation of a run, at the positions of the input before any constraint, packs its frames at the refresh, which checks no extents. #795 measured the extents at the build there. The extents at the input are left to S4b, whose route reads inputs from files.

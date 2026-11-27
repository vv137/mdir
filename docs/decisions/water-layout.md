# D[water-layout]

Recorded at its merge.

**A list of units reads each copy of a taken rigid kind as consecutive particles. The order of the places names, for a copy, the particle of its place 0, and its place $p$ is the particle $p$ after it. The order of the state already keeps the particles of each copy together (D44, D110), so this is the layout by molecule of #794, and the list holds no table of members. The writer (D422), the SIMT schedule (D421), and the serial definition (D417, Section 6) form every member's index from that one number. The traversal adds the forces of a unit's places to its frame atoms there. The writer takes the places in their order, a thread a member of the unit at each place, so it writes the frames in the order that the loops read them. No input is refused for this. A kind whose molecules are not of consecutive particles in the input (D403's count `scattered`) is not taken: its particles run as particles of the rest, and `mdir check` and the run warn (`rigid_units_scattered`) with the count. A refusal under `ON` depends only on the reasons of before: where no kind is taken and a kind is passed over for being scattered alone, `ON` takes the general route and says why, and a kind that such a reason refuses refuses as before, scattered or not. The setup checks that every copy it takes is consecutive, as an internal check of consistency. Programs off the route are unchanged.** Issue #794, after D424.

## 1. What #794 asked, and what the code held

#794 was written from the prototype of #795. That prototype packed each water's positions out of the particle arrays every step and unpacked the forces back. On `main` (D417 to D424), the two halves stand as follows:

- **The pack** is the writer of D422. Once a step it reads the positions of every unit's members, writes the f32 frames at the unit's place, and tests the reference points.
- **The unpack** is no longer a kernel. The traversal adds a unit's forces to its members' particles with atomics. It found each member through a table, `members`, indexed by the slot that the place's value of `order` held.

Every member access therefore took two dependent loads, `order[place]` and then `members[slot + k]`, before the position or the atomic. The writer also took a unit's place (`placeOf[unit]`) and its value (`unitValue[unit]`) before it could load anything.

## 2. The measurement before the design

RTX 4090, mixed precision, `rigid_units = "ON"` with the groups, `main` at 03506508. Each figure comes from one Nsight Systems capture of the kernels: 12,000 steps, 6,000 for Factor IX, and 3,000 for Cellulose, with an energy row every 500 steps. The step is the log's rate during the capture.

| System | Taken copies | Rest | Step, µs | Writer, µs a step | Writer share |
|---|---:|---:|---:|---:|---:|
| TIP3P, 9,989 atoms, NVT | 3,222 | 323 | 113 | 3.51 | 3.1% |
| TIP3P, 19,880 atoms, NVT | 6,513 | 341 | 134 | 3.80 | 2.8% |
| ubiquitin, OPC, NVE | 6,166 | 1,265 | 142 | 4.19 | 2.9% |
| large OPC box, NVE | 16,329 | 0 | 205 | 6.81 | 3.3% |
| JAC, NVE | 7,023 | 2,489 | 144 | 4.06 | 2.8% |
| JAC, NPT | 7,023 | 2,489 | 161 | 4.11 | 2.6% |
| Factor IX, NVE | 28,358 | 5,832 | 349 | 11.25 | 3.2% |
| Factor IX, NPT | 28,358 | 5,832 | 372 | 11.24 | 3.0% |
| Cellulose, NVE | 105,855 | 91,044 | 1,527 | 45.17 | 3.0% |
| Cellulose, NPT | 105,855 | 91,044 | 1,609 | 45.40 | 2.8% |

The writer takes about 3% of the step on every system, small and large. Its test of the reference points and its check of the extents are part of that time: D422 measured them at 0.65 to 0.70 µs a call. Any layout keeps the test and the check, because the theorem of validity needs them at every step.

The unpack shows only in the loop over pairs. There, the partner's members are found once for each entry, and their loads stand before the atomics of its forces, so the table's dependent load lies on that path. Section 5 measures the loop before and after this decision. Its only change to the loop's code is that the table is gone.

## 3. The layout

**What is permuted.** Nothing beyond what the order of the state already does. Where a run or segment begins, the particles are sorted by the cell of their anchor and, within a cell, by their number in the input (D44). Since D110 the anchor of every particle of a copy is the copy's oxygen, so all of a copy's particles share one key. A copy whose particles are consecutive in the input therefore stays consecutive, in the order of its places. The layout by molecule of #794 is this order. It needs no new sort, no permutation in the loop over steps, and no index map.

**What the list holds.** For a copy, the value of `order` at its place is the particle of the copy's place 0. Member $k$, the frame atom at place $\mathrm{frame}_k$ of its kind, is that particle plus $\mathrm{frame}_k$. For the rest, `order` holds the particle, as before. The table `members` and its buffer are removed. The buffer form of `md_alg.traverse_pairs` over segments takes 8 buffers, or 13 with `inner`.

**The check.** The setup loops on the host over the copies of each taken kind, as it did to fill the table. It now checks that the places of each copy are consecutive particles. Where one is not, it stops the run with `mdrtSegmentsStopScattered` and the count. By the argument above this cannot happen for a kind that `ON` takes, so the check guards the argument; it is not a refusal.

**Molecules that are not consecutive: an automatic choice, not a refusal.** The rule of the project is that a defect of an input is reported, and that no input stops running. Where a kind has a molecule that D403 counts as `scattered` (not of consecutive particles in the input), the run therefore keeps running under `ON`:
- The kind is not taken. Its reason is "N molecules not of consecutive particles; its particles run in the rest".
- Its particles join the rest, which the same list runs as single particles with their exclusions, and with the hydrogens anchored to their oxygen (D421, Section 3). That is the general pair body inside the list of units, and its results agree with the general route within the bounds of `rigid-units-frames-gpu.test`.
- The other kinds are taken as before, so the kind after it in the order may take its slot.
- `mdir check`, the run's log, and the JSON report warn with the code `rigid_units_scattered` and the count, and the JSON's kind has `scattered`.
- Where no kind is taken and a kind is passed over for being scattered alone, `ON` takes the general route: the reasons of before would have taken that kind, so `main` did not refuse the input either. A kind that a reason of before refuses (one particle, an excluded pair outside the molecule, a pair of its own not excluded) refuses as on `main`, whether it is scattered or not, so a scattered kind neither causes a refusal nor hides one. The reason is "ON was given, but no kind of consecutive particles is eligible: the general route", and the manifest and the log record it as for every resolution of the route (#952).

Such inputs exist only where a topology interleaves another molecule's particles into a molecule's, which a GROMACS `[ moleculetype ]` can describe (`kind_pairs_mixture.py scattered`). `test/Driver/rigid-units-route.test` checks the report and the route (`SCATTERED`, `SCATTERED-ONLY`).

**The writer.** One kernel a step, as in D422, now with a thread for each member of the unit at each place: $M p + k$ for place $p$ and member $k$, over the places of the last build (`sizes[0]` on the host, which the build already reads back). The thread loads the place's value of `order` and takes the rank of the place from where the ranks begin (`rankPlaces`), as the pack of a build does. It never reads `placeOf` or `unitValue`. A copy's reference point is its anchor, the particle at place $\mathrm{frame}_0$. The rest's comes, as before, from `unitRef`, found through `unitOfParticle`. An empty place is not written: the refresh that built the list wrote it as zeros, and nothing writes it again until the next build. Every frame holds the same numbers as before, formed by the same operations (D95; D422, Section 2). Only the thread that writes a value and the order of the stores change.

**Outputs, the checkpoint, and continuation.** The order of the state, and the map from it to the order of the input, are unchanged. Every file is written in the order of the input, as before. A checkpoint does not depend on the route, and a continuation may switch `rigid_units`.

**The neighbor structures.** Only the list of units changes. The groups (D415), the matrix, and TOWERS have no template kernel. Their programs keep their bytes, and so does every program under `AUTO` or `OFF`. `ON` still refuses TOWERS (D424).

**The constraints.** SETTLE and the integration kernels are not touched. They read the state in its order, as before.

**Determinism.** The units refuse the deterministic mode (D424), so its programs are unchanged. On the route, the frames are the same numbers. The forces of a step are the same sums in another order of the f32 atomics, which already vary from run to run on `main`.

## 4. Alternatives not taken

- **Permuting the state at every build, so that the loops read the state.** The tuples of every term would have to be renumbered at each build. Today that is done on the host, at the start of a run. The loops would read f64 positions where they now read f32 frames, and they would read the order of the state, which decays between sorts. That decay is what the order of the places exists to avoid (paper, Section 4.2: 117 to 208 µs a step over 40 ps of JAC).
- **Writing the frames in the integration kernel, where SETTLE has a water's new positions.** That would save the writer's loads of the positions and its launch. The test, the check, and the stores of the frames would stay. Under a barostat, the scaling after the constraints moves the positions again, so the writer would still be needed on those steps. It is left for a decision of its own, with its measurement. The kernel of an integration run is shared by every program, so its bytes are a gate that this decision does not need to open.
- **Keeping the table for kinds that are not consecutive.** Two code paths for one list, for inputs that interleave molecules. Running their particles in the rest keeps one path and still runs every input.
- **Refusing `ON` for such a kind.** Refusing would change how an existing input runs; the project reports defects of inputs instead.

## 5. The kernels after

Captures of `main` and of this decision, in one job on one RTX 4090, interleaved and then in reverse order (mixed, `ON`, the controls of Section 2). Each figure is µs, averaged over the two rounds. "A call" is the mean over every call of the kernel, its calls that prune and that evaluate the energy included. "A step" counts every kernel of the second half of the capture.

| System | Writer a call, `main` | Writer, this | Loop a call, `main` | Loop, this | Kernels a step, `main` | Kernels, this |
|---|---:|---:|---:|---:|---:|---:|
| TIP3P, 9,989 atoms | 3.53 | 2.60 | 32.99 | 31.15 | 90.9 | 87.9 |
| ubiquitin, OPC | 4.28 | 3.29 | 40.01 | 39.03 | 125.1 | 123.2 |
| JAC, NVE | 4.10 | 3.44 | 56.37 | 52.75 | 130.2 | 125.7 |
| Factor IX, NVE | 11.42 | 6.78 | 192.11 | 178.19 | 338.3 | 319.2 |
| Cellulose, NVE | 45.84 | 29.44 | 999.91 | 963.11 | 1527.2 | 1474.3 |

- The writer is 16 to 41% faster a call.
- The loop over pairs is 2.4 to 7.3% faster a call. Its partner's members no longer wait for a load of the table before their atomics.
- The kernels of a step fall by 1.9 to 53 µs.
- The device's idle time a step is the same on both trees within 0.3 µs, and so is the number of the build's launches; the branch never launches fewer.
- So the shorter step comes from the kernels. No refresh is skipped.

**The rate.** The timing protocol, on RTX 4090s at 2 fs, in two jobs on two devices. Each job runs two rounds, the second in reverse order. In each round `main` with `ON`, this decision with `ON`, and `main`'s default program run interleaved. Each run is a wall run with a row every hundredth of its steps, and the rate is the trimmed stationary mean. Under load the SM clocks were 2,685 to 2,790 MHz. Each ratio is the mean over the four job and round pairs, with the range in parentheses.

| System | `main`, `ON`, ns/day | This, `ON`, ns/day | This / `main`, a step | `main`, default, ns/day |
|---|---:|---:|---:|---:|
| TIP3P, 9,989 atoms, NVT | 1,708.1 | 1,755.6 | 0.973 (0.965–0.977) | 1,675.3 |
| TIP3P, 19,880 atoms, NVT | 1,399.5 | 1,423.2 | 0.983 (0.981–0.986) | 1,280.0 |
| JAC, NVE | 1,299.7 | 1,336.6 | 0.972 (0.967–0.979) | 1,263.0 |
| JAC, NPT | 1,142.6 | 1,172.7 | 0.974 (0.973–0.977) | 1,154.2 |
| Factor IX, NVE | 515.3 | 532.1 | 0.969 (0.963–0.974) | 566.7 |
| Factor IX, NPT | 478.8 | 493.7 | 0.970 (0.966–0.975) | 529.5 |
| Cellulose, NVE | 113.8 | 118.2 | 0.963 (0.962–0.965) | 147.9 |
| Cellulose, NPT | 107.7 | 111.0 | 0.970 (0.968–0.972) | 141.7 |
| ubiquitin, OPC, NVE | 1,274.9 | 1,279.4 | 0.997 (0.994–0.999) | 1,035.7 |
| large OPC box, NVE | 838.5 | 855.7 | 0.980 (0.975–0.984) | 638.6 |

The two systems of OPC ran in two jobs of their own, by the same protocol (SM clocks 2,655 to 2,745 MHz under load). The step with `ON` is shorter on every system, in every round; on ubiquitin in OPC by 0.1 to 0.6%. The kernels of Section 5 account for it: on Cellulose they fall by 53 µs a step, and its wall falls by 56. The default program is the same program on both trees (Section 6), so its rate is given once. `ON` is still slower than the default on JAC NPT, Factor IX, and Cellulose, and faster on the TIP3P boxes, JAC NVE, and the systems of OPC, as D422 found for Cellulose. That is the open question of `AUTO`'s rule, not this decision's.

## 6. Validation

RTX 4090, at this decision's head and `main` at 03506508.

**Off the route.** The corpus of D406 was compared against `main`: `check.py compare --stages lowered --targets cpu,gpu --kinds run,segments --precisions mixed,double --execution neighbor_structure=GROUPS`, with `MDIR_GPU_BINARY=ptx`, on `sm_89`. Of 4,452 cases, 4,444 are identical; 2,526 of those stop with the same error in both. The 8 that differ are the cases of `rigid_units = "ON"`: `rigid-units-reach-gpu.test`'s `shrink.toml` and `rigid-units-route.test`'s `dodecahedron.toml`, each on the GPU in mixed precision, for runs and segments, with and without the GROUPS variant. Every program on the CPU, in double precision, or in the deterministic mode is therefore unchanged, and so is every program under `AUTO` or `OFF`.

**The same frames** (`test/Driver/rigid-units-frames-gpu.test`, with the bounds of D424). The units against the general route in double precision, within 1.5 times the general mixed route's difference, or 1e-5 of the largest force and 1e-6 of the potential. The largest ratio to the bound:

| Case | Ratio |
|---|---:|
| D405's mixture, plain cutoffs | 0.458 |
| the mixture with the site listed first | 0.247 |
| alanine dipeptide, OPC, PME, NPT | 0.665 |
| alanine dipeptide, TIP3P, PME | 0.667 |
| rhombic dodecahedron, plain cutoffs | 0.860 |
| rhombic dodecahedron, PME | 0.186 |
| the mixture with an ion inside each water of four sites (those waters in the rest) | 0.867 |

A mutation that ignores the places of the frame atoms (member $k$ at the first particle plus $k$) takes the case with the site listed first to 6.6e5 times its bound, while the others still pass. It was tried once and reverted.

**#795's short trajectories** (`segments-trajectories-gpu.test`, eight cases of OPC and TIP3P under both integrators, NVE and NPT) pass. On these three waters the units' route is reproducible to the bit. Four of the eight cases were compared to the bit, chosen so that each pair of integrator and ensemble appears once and each water twice: OPC under velocity Verlet NVE and leapfrog NPT, TIP3P under velocity Verlet NPT and leapfrog NVE. In each of the four, the units' positions and velocities at every tenth step of 100 are the same to the bit from `main`'s tree and from this decision's, and from run to run. The other four were not compared to the bit; they pass within the declared bounds. Only the place of the data changed, and #794 asked for this gate.

**The contract and the coverage.** `md-alg-segments-contract-gpu.test`, `segments-writer-coverage-gpu.test`, `segments-writer-extents-gpu.test`, `segments-build-forces-gpu.test`, `segments-build-coverage-gpu.test`, and `md-alg-segments-serial.test` pass. The serial definition reads a copy from its first particle, and its oracle checks that every copy it builds is consecutive.

**The internal check and the automatic choice.** `segments-build-reach-stop-gpu.test` hands the lowering a copy with its last two places exchanged, which the driver never does, and the setup stops with the message of `mdrtSegmentsStopScattered`. `rigid-units-route.test` checks the reason, the warning, and the general route for the mixture with an ion inside each water of four sites; for that water beside a kind refused for its exclusions (the general route, as `main` took the water); and, with the refusal of `main`, for that water whose exclusions are also incomplete. `rigid-units-frames-gpu.test` also checks the warning in the log of `mdir run`. `rigid-units-frames-gpu.test` runs that mixture under `ON` (the waters of four sites in the rest) against the general route.

**Trajectories of the suite's systems.** 50 steps with `ON` from the same input, the final positions compared. On `main` the route is not reproducible to the bit, since the f32 atomics of its loop add in an order that varies. The largest difference of a coordinate, in nm:

| System | `main` against `main` | this against `main` |
|---|---:|---:|
| TIP3P, 9,989 atoms, NVT | 9.2e-7 | 9.4e-7 |
| ubiquitin, OPC, NVE | 1.48e-6 | 1.46e-6 |
| JAC, NVE | 2.47e-6 | 1.90e-6 |
| JAC, NPT | 2.21e-6 | 1.92e-6 |

**The sanitizer.** memcheck, initcheck, and racecheck pass `test/Sanitizer/segments-writer-gpu.test`, `segments-traverse-gpu.test`, and `segments-build-gpu.test`. The job held an RTX 4090 for 74 s.

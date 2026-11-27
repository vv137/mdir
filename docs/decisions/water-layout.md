# D[water-layout]

Recorded 2026-11-27.

**A list of units reads each copy of a taken rigid kind as consecutive particles. The order of the places names, for a copy, the particle of its place 0, and its place $p$ is the particle $p$ after it. The order of the state already keeps the particles of each copy together (D44, D110), so this is the layout by molecule of #794, and the list holds no table of members. The writer (D422), the SIMT schedule (D421), and the serial definition (D417, Section 6) form every member's index from that one number. The traversal adds the forces of a unit's places to its frame atoms there. The writer takes the places in their order, a thread a member of the unit at each place, so it writes the frames in the order that the loops read them. The setup stops a run in which a copy is not consecutive. `ON` does not take a kind whose molecules are not of consecutive particles in the input (D403's count `scattered`), and gives that reason. Programs off the route are unchanged.** Issue #794, after D424.

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

The unpack shows only in the loop over pairs. There, the partner's members are found once for each entry, and their loads stand before the atomics of its forces, so the table's dependent load lies on that path. Section 5 measures the loop with and without the table.

## 3. The layout

**What is permuted.** Nothing beyond what the order of the state already does. Where a run or segment begins, the particles are sorted by the cell of their anchor and, within a cell, by their number in the input (D44). Since D110 the anchor of every particle of a copy is the copy's oxygen, so all of a copy's particles share one key. A copy whose particles are consecutive in the input therefore stays consecutive, in the order of its places. The layout by molecule of #794 is this order. It needs no new sort, no permutation in the loop over steps, and no index map.

**What the list holds.** For a copy, the value of `order` at its place is the particle of the copy's place 0. Member $k$, the frame atom at place $\mathrm{frame}_k$ of its kind, is that particle plus $\mathrm{frame}_k$. For the rest, `order` holds the particle, as before. The table `members` and its buffer are removed. The buffer form of `md_alg.traverse_pairs` over segments takes 8 buffers, or 13 with `inner`.

**The check.** The setup loops on the host over the copies of each taken kind, as it did to fill the table. It now checks that the places of each copy are consecutive particles. Where one is not, it stops the run with `mdrtSegmentsStopScattered` and the count. By the argument above this cannot happen for a kind that `ON` takes, so the check guards the argument; it is not a refusal.

**Eligibility.** A kind that D403 counts as `scattered` (some molecule not of consecutive particles in the input) is not taken, with the reason "N molecules not of consecutive particles". Such inputs exist only where a topology interleaves another molecule's particles into a molecule's, which a GROMACS `[ moleculetype ]` can describe (`kind_pairs_mixture.py scattered`). The other kinds are taken as before, so the kind after it in the order may take its slot (`test/Driver/rigid-units-route.test`, `SCATTERED`).

**The writer.** One kernel a step, as in D422, now with a thread for each member of the unit at each place: $M p + k$ for place $p$ and member $k$, over the places of the last build (`sizes[0]` on the host, which the build already reads back). The thread loads the place's value of `order` and takes the rank of the place from where the ranks begin (`rankPlaces`), as the pack of a build does. It never reads `placeOf` or `unitValue`. A copy's reference point is its anchor, the particle at place $\mathrm{frame}_0$. The rest's comes, as before, from `unitRef`, found through `unitOfParticle`. An empty place is not written: the refresh that built the list wrote it as zeros, and nothing writes it again until the next build. Every frame holds the same numbers as before, formed by the same operations (D95; D422, Section 2). Only the thread that writes a value and the order of the stores change.

**Outputs, the checkpoint, and continuation.** The order of the state, and the map from it to the order of the input, are unchanged. Every file is written in the order of the input, as before. A checkpoint does not depend on the route, and a continuation may switch `rigid_units`.

**The neighbor structures.** Only the list of units changes. The groups (D415), the matrix, and TOWERS have no template kernel. Their programs keep their bytes, and so does every program under `AUTO` or `OFF`. `ON` still refuses TOWERS (D424).

**The constraints.** SETTLE and the integration kernels are not touched. They read the state in its order, as before.

**Determinism.** The units refuse the deterministic mode (D424), so its programs are unchanged. On the route, the frames are the same numbers. The forces of a step are the same sums in another order of the f32 atomics, which already vary from run to run on `main`.

## 4. Alternatives not taken

- **Permuting the state at every build, so that the loops read the state.** The tuples of every term would have to be renumbered at each build. Today that is done on the host, at the start of a run. The loops would read f64 positions where they now read f32 frames, and they would read the order of the state, which decays between sorts. That decay is what the order of the places exists to avoid (paper, Section 4.2: 117 to 208 µs a step over 40 ps of JAC).
- **Writing the frames in the integration kernel, where SETTLE has a water's new positions.** That would save the writer's loads of the positions and its launch. The test, the check, and the stores of the frames would stay. Under a barostat, the scaling after the constraints moves the positions again, so the writer would still be needed on those steps. It is left for a decision of its own, with its measurement. The kernel of an integration run is shared by every program, so its bytes are a gate that this decision does not need to open.
- **Keeping the table for kinds that are not consecutive.** Two code paths for one list, for inputs that interleave molecules. The reason in `mdir check` names them instead.

## 5. The kernels after

The same captures for `main` and this decision, with the same controls (RTX 4090, mixed, `ON`). Each figure is µs a call, averaged over every call of the kernel in the capture. The loop over pairs includes its calls that prune and that evaluate the energy.

| System | Writer, `main` | Writer, this | Loop, `main` | Loop, this | Loop ratio |
|---|---:|---:|---:|---:|---:|
| TIP3P, 9,989 atoms | 3.51 | 2.66 | 32.96 | 31.86 | 0.967 |
| TIP3P, 19,880 atoms | 3.80 | 3.06 | 49.24 | 47.74 | 0.970 |
| ubiquitin, OPC | 4.19 | 3.35 | 39.64 | 39.95 | 1.008 |
| large OPC box | 6.81 | 4.55 | 72.71 | 71.45 | 0.983 |
| JAC, NVE | 4.06 | 3.50 | 55.55 | 53.77 | 0.968 |
| JAC, NPT | 4.11 | 3.51 | 49.50 | 47.98 | 0.969 |
| Factor IX, NVE | 11.25 | 6.89 | 188.93 | 181.91 | 0.963 |
| Factor IX, NPT | 11.24 | 6.88 | 167.16 | 161.43 | 0.966 |
| Cellulose, NVE | 45.18 | 29.90 | 985.68 | 984.24 | 0.999 |
| Cellulose, NPT | 45.41 | 30.05 | 881.12 | 878.45 | 0.997 |

- The writer is 14 to 39% faster.
- The loop is about 3% faster on the TIP3P boxes, JAC, and Factor IX.
- The loop is unchanged within 0.3% on Cellulose, and 0.8% slower on ubiquitin in OPC.

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

The step with `ON` is shorter on every system, in every round. That is more than the kernels alone account for, since the host waits for the writer's flags every step (D422, Section 5) and the writer now ends sooner. The default program is the same program on both trees (Section 6), so its rate is given once. `ON` is still slower than the default on JAC NPT, Factor IX, and Cellulose, as D422 found for Cellulose. That is the open question of `AUTO`'s rule, not this decision's.

## 6. Validation

See the pull request for the numbers. The gates:
- Programs off the route are unchanged: the corpus of D406 (`check.py compare`) on `sm_89`, and the checkpoints.
- On the route, the same-frame forces and energies against the general route in double precision (`test/Driver/rigid-units-frames-gpu.test`). A case with the site listed before its frame makes the frame atoms places 1 to 3 of a copy. A mutation that ignores the places of the frame atoms fails that case.
- #795's short trajectories (`test/Driver/segments-trajectories-gpu.test`), the contract of the schedule (`md-alg-segments-contract-gpu.test`), the coverage of the writer (`segments-writer-coverage-gpu.test`) and its checks of the extents (`segments-writer-extents-gpu.test`), the forces of the build (`segments-build-forces-gpu.test`), and the serial definition (`md-alg-segments-serial.test`).
- 50-step trajectories against `main` on the route, beside two runs of `main`.

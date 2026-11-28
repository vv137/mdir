# D[device-coupling]

Recorded 2026-11-28.

**On the GPU the arithmetic of the coupling step on the sums of the step runs on the device. `[execution] coupling_step = "DEVICE"`, the default on the GPU; `"HOST"` keeps the path of before.** Issue #1026, after D448 (recorded spans), D449 (term streams) and D450 (graph replay of TOWERS steps under a barostat).

## 1. What changes

The step that ends a period of coupling (D72, `docs/dynamics.md` Section 2.2.5) computed on the host, from global sums that it copied from the device and waited for:

- the factor $\alpha$ of stochastic velocity rescaling [[Bussi2007]](../references.md#bussi2007) (`mdrtBussiFactor`);
- the heat that goes to the bath (`mdrtAddBath`);
- under stochastic cell rescaling [[Bernetti2020]](../references.md#bernetti2020), the pressure, the strain and the new cell.

`convert-md-exec-to-gpu{device-coupling=true}` (`lib/Conversion/MDExecToGPU/DeviceCoupling.cpp`) now does the following, before the pass moves the chains of D156 into the kernels:

1. **The factor.** A call of `mdrtBussiFactor` becomes two draws of the host, `mdrtBussiNormal` and `mdrtBussiRest` (`runtime/mdrt.c`), and the ops of the rest of the C function in its order of operations (Section 2). One thread computes $\alpha$ into a slot of the function's buffer of results on the device. The kernels that scale the velocities read the slot instead of taking $\alpha$ by value.
2. **The bath.** The bath is a slot of the same buffer. It starts from the host's value at the entry of the function (`mdrtBath`). One thread adds each heat to it in the order of the program, in the kernel that also stores $\alpha$. The host takes it back (`mdrtSetBath`) before every row of the log, every checkpoint and every return.
3. **The pressures of a scaling.** One thread computes the pressures that the strain takes (one, two or three, for isotropic, semi-isotropic and anisotropic coupling) into a buffer of eight numbers. The host copies that buffer into page-locked memory and computes the strain, the new cell and its checks as before. The virial that a TOWERS dispatch observes for a scaling is read like a sum: each of its copies takes a snapshot of its own on the device, since the step after a scaling writes the buffer again, and the copy stays where the dispatch put it. The helpers of PME that read back the reciprocal virial are inlined, with a snapshot of their results on the device, as for device frames. Only the coupling kernels read such sums of the second stream (`term_streams`), after a join by event. The one-thread kernels therefore read those virials on the device.
4. **Page-locked buffers.** With the key, the host's copy of the f64 results of a function, the buffer of the pressures, and the small buffers that the copies of the observed and reciprocal virials still reach at outputs are page-locked memory of the runtime (`mdrtHostResults`). The copies that remain at outputs are therefore transfers of the device's copy engine, as the maintainer's addition to #1026 asks. The integer results of a function (the counts that the builds of the neighbor structures read) stay on the stack: they are not copies of the coupling step.

`keepReadbacksOnDevice` (D156) then clones what each one-thread kernel takes from the sums into it. A copy whose loads the host no longer reads is removed. Every host use that remains keeps its copy, so an expression that the pass does not move stays correct on the host.

The one-thread kernels carry `mdir.frame_host_arithmetic`: the passes that contract products or make explicit fused multiply-adds (#97, #102) leave them out.

## 2. Why the trajectory is the same

1. [derived] The draws of V-rescale (a normal number and, by the rejection of [[Marsaglia2000]](../references.md#marsaglia2000), a gamma number) and those of C-rescale depend on the seed, the step and $N_f$, not on the state (`docs/dynamics.md` Section 2.2.6). The host makes the same calls of Philox in the same order. `mdrtBussiRest` draws the first normal number again and discards it, so its gamma number is that of `mdrtBussiFactor`.
2. [derived] The rest of `mdrtBussiFactor` uses only $+$, $-$, $\times$, $/$ and $\sqrt{\ }$, which IEEE 754 rounds correctly on the host (SSE2, without FMA: `libmdrt` is built without `-march`) and on the device, where the f64 division and square root of the kernels round to nearest. The bitwise tests of Section 5 confirm it. The ops follow the C expression's evaluation order: `kinetic + (1 - decay) * (target * (r1 * r1 + rest) / freedom - kinetic) + 2 * r1 * sqrt(kinetic * target / freedom * (1 - decay) * decay)`. The one-thread kernels have no contractions. The factor is therefore the host's to the bit, and so is everything after it.
3. [derived] The bath is one value that receives the same additions in the same order. Starting from the host's value and returning it before every read gives the host the value it had.
4. [derived] The pressures are the same pure ops of the same sums (D156). The strain, the exponential of the scaling, the f32 rounding of the cell and its checks stay on the host's libm, so the cell is the host's to the bit.

So in the deterministic mode the checkpoint and every row of the log are bitwise those of `HOST` (Section 5).

## 3. What the host still reads, and why

| Step | Before (`HOST`) | After (`DEVICE`) |
|---|---|---|
| V-rescale, no output | the results (8 kB) and a wait | nothing |
| C-rescale, no output | the results (8 kB) four to seven times, the observed virial (24 B, TOWERS) twice, the reciprocal virial (56 B) twice, with a wait each | the pressures (64 B) and a wait |
| A row or a checkpoint | the results | the results, the bath |

1. [derived] Under a barostat the next step's launches take the new cell by value, and the host decides on it.
   - In the lowered program of the dipeptide's NPT run, the build of the neighbor matrix (`mdrt_gpu_build_neighbors_matrix`) takes the cell as an argument. It counts the cells of each axis on the host with `mdrt_gpu_cell_count` (the edge over the width) and sizes its launches (`mdrt_gpu_grid`) and its buffers from that count.
   - The PME wrappers and the kernels that the host launches take the edges and their reciprocals by value.
   - With TOWERS the grid is fixed (D450, Section 3.2), but the host still fills the mirror of the chunks from its own copy of the cell, and the steps outside the chunks launch with the cell by value.
   - The host also checks every new cell and passes it to the thread of the frames (`mdrtSetBox`).

   The host therefore needs the cell before it issues the next step. Removing that read needs grids fixed under a barostat for MATRIX and GROUPS, launches and wrappers that read the cell on the device, and the checks of the cell on the device.
2. [derived] The strain takes $\ln(\lambda'/\lambda)$, $e^{\varepsilon/3}$ and, with a cell in f32, the logarithm of the rounded scale. To keep the trajectory, the device would have to compute these as this host's libm does, bit for bit. CUDA's double-precision `log` and `exp` are accurate to within 1 ulp, not correctly rounded (CUDA C++ Programming Guide, the mathematical functions), so they can differ from glibc's in the last bit. A port of glibc's algorithms to the device, checked against the host's libm over many arguments, was not tried: running the downloaded source of the algorithms was not permitted in this work. This decision keeps the strain on the host and copies the pressures, which the device computes exactly as the host did.
3. [measured] Under `step_graph = "GRAPH"` the coupling step still reads the accepted step of the chunks (`mdrtStepRead`, 8 B, D439), which the draws take. After this change both draws share one read (Section 4).
4. Not covered: a simulation of the Python interface (Section 6); the Nose–Hoover chain, which integrates its own state on the host (D163a), and Langevin dynamics, whose thermostat acts in the kernels of every step and has no factor. For these the key resolves to `HOST`, and the record says why (Section 6). The CPU target has no device.

## 4. Copies and waits

From the runtime's counters (`MDRT_HOST_READS=1`, `runtime/mdrt_cuda.c`). Each line is the difference between runs of 4,000 and 2,000 steps of JAC with one row at the end, divided by the number of additional coupling periods (V-rescale every 10 steps in NVT; V-rescale and C-rescale every 25 steps in NPT; RTX 4090, mixed precision). These are per period, all steps included: the 20 B copies are the counts that the builds of the groups read, and the flag waits are the checks of the lists (D113, D118). Neither depends on the coupling.

| Input | Path | `HOST` (main's path): copies (bytes: count) / waits | `DEVICE`: copies / waits |
|---|---|---|---|
| JAC NVT | defaults (GROUPS, SPANS) | 20: 0.63; 8,192: 1.00 / 1.63 | 20: 0.61 / 0.61 |
| JAC NVT | TOWERS + TERMS, SPANS | 8,192: 1.00 / 1.00 | none / 0 |
| JAC NVT | TOWERS + TERMS, GRAPH | 8: 1.00; 48: 1.00; 8,192: 1.00 / 3.00 | 8: 1.00; 48: 1.00 / 2.00 |
| JAC NPT | defaults (GROUPS, SPANS) | 20: 1.54; 56: 2.00; 8,192: 4.00 / 7.54 | 20: 1.54; 64: 1.00 / 2.54 |
| JAC NPT | TOWERS + TERMS, SPANS | 24: 2.00; 56: 2.00; 8,192: 7.00 / 11.00 | 64: 1.00 / 1.00 |

Flag waits per period, the same in both paths: 30 (NVT, defaults), 75 (NPT, defaults), 11 and 25 (TOWERS + TERMS, SPANS), 2 (GRAPH).

1. [measured] With V-rescale alone, the period no longer copies anything for the coupling, in every path. Under `GRAPH`, what remains are the accepted step (8 B) and the progress of the chunk (48 B), which D439 reads after each chunk. A first version took one read of the step for each of the two draws (8 B: 2.00); the draws now share it.
2. [measured] Under C-rescale, the copies of a period fall from 6 (defaults) and 11 (TOWERS + TERMS) to 1, the pressures, with one wait.
3. [measured] JAC NPT with TOWERS + TERMS under `GRAPH` does not compile, on main as on this branch ("operand #0 does not dominate this use"). That is a defect of its own, outside this decision.

## 5. Validation

`test/Driver/device-coupling-gpu.test` and `test/Driver/device-coupling-towers-gpu.test` (`Inputs/check_device_coupling.py`) run the dipeptide in water: 1,168 particles, PME, SHAKE and SETTLE, V-rescale every 10 steps, 2 fs, with C-rescale every 10 steps in NPT. The oracle is the same program with `coupling_step = "HOST"`, which is the lowering of before.

- **Deterministic mode, MATRIX: bitwise.** The checkpoint (positions, velocities, forces, cell), the bath that it keeps, and every row of the log are compared byte for byte, over 300 steps. Cases: mixed and double precision; spans and launches one by one (NVT); isotropic, semi-isotropic and anisotropic coupling in mixed precision; and isotropic in double (NPT).
- **Against main's binary: bitwise.** The same comparisons with the binary of main (864cad127, without the key) against the branch's default (`DEVICE`), 300 steps: NVT and isotropic NPT in mixed and double precision, semi-isotropic and anisotropic NPT in mixed precision. The checkpoint, the bath and the seven rows are the same bytes. (One-off; the tests take `HOST`, the lowering of before, as the oracle.)
- **A continuation.** From a checkpoint at step 150, 160 steps more with a row every 40 steps: bitwise. The bath goes to the device from the checkpoint.
- **GROUPS** (not deterministic), NVT and NPT, 40 steps: the relative L2 difference of the positions and velocities from `HOST` is at most ten times that of two `HOST` runs, with a floor of $10^{-5}$.
- **TOWERS** (not deterministic), 200 steps at 1 fs, NVT, and NPT at 500 bar with a time constant of 0.1 ps, under `LAUNCH`, `SPANS` and `GRAPH`, against two `HOST` runs under `LAUNCH`, with the same bound; NPT also with `term_streams = "RECIPROCAL"` and `"TERMS"` under `LAUNCH` and `SPANS`. Every row falls on the same step, and its values agree within $10^{-4}\lvert x\rvert + 0.05$, the tolerance of `check_term_streams_graph.py`. A first version, whose cloned reads of the observed virial took the buffer after the step that follows a scaling had written it again, gave rows whose energy with the bath was off by 2.4 and 9.7 (in the units of the log) at the second and third rows; the comparison of the values caught it.

| Case | Positions: `DEVICE`, `HOST` spread | Velocities: `DEVICE`, `HOST` spread |
|---|---|---|
| NVT, LAUNCH | 1.5e-7, 8.8e-8 | 1.2e-5, 7.9e-6 |
| NVT, SPANS | 8.3e-8, 8.8e-8 | 7.3e-6, 7.9e-6 |
| NVT, GRAPH | 1.0e-7, 8.8e-8 | 8.4e-6, 7.9e-6 |
| NPT, LAUNCH | 1.6e-7, 1.4e-7 | 1.2e-5, 1.1e-5 |
| NPT, SPANS | 1.2e-7, 1.4e-7 | 9.3e-6, 1.1e-5 |
| NPT, GRAPH | 1.2e-7, 1.4e-7 | 9.4e-6, 1.1e-5 |

Relative L2 differences from the first `HOST` run after 200 steps. The bound is the larger of ten times the spread and $10^{-5}$.

- **No copy per coupling.** Runs of 200 and 400 steps with one row: the copies that grow with the 20 additional couplings. MATRIX NVT: `HOST` 20 copies of 8 kB, `DEVICE` none. MATRIX NPT: `HOST` 80 of 8 kB and 40 of 56 B, `DEVICE` 20 of 64 B. TOWERS NVT: `HOST` 20 of 8 kB, `DEVICE` none. TOWERS NPT: `HOST` 80 of 8 kB, 40 of 24 B and 40 of 56 B, `DEVICE` 20 of 64 B; with the reciprocal sum on the second stream, `HOST` 140 of 8 kB, 40 of 24 B and 40 of 56 B, `DEVICE` 20 of 64 B. Every other copy, the builds' counts, grows alike in both paths, which run the same trajectory to the bit.
- **The record and the cases left on the host.** `test/Driver/coupling-step.test` checks with `mdir check --json`: `DEVICE` by default on the GPU, NVT and NPT; `HOST` on request, on the CPU target, under a Nose–Hoover chain, under Langevin dynamics and without coupling, each with its reason; a value that is not a choice is refused. `test/Driver/python-compile-gpu.test` checks that a program compiled through the Python interface for the GPU has no coupling step on the device, and that its pipeline is that of `mdir run` with `coupling_step = "HOST"`.
- **GROUPS** refuses the deterministic mode (`neighbor_structure = "GROUPS"` needs not `deterministic`), so its comparison is by the spread, not bitwise.

## 6. The key

| Value | Meaning |
|---|---|
| `"DEVICE"` (default) | The coupling step on the device, where the run allows it. Resolves to `HOST`, a rule recorded in `resolved_choices.coupling_step` with its reason: on the CPU target, in a simulation of the Python interface, under a Nose–Hoover chain, under Langevin dynamics, in a run without coupling. |
| `"HOST"` | The sums are copied and the coupling step computed on the host, as before. |

The setup log line `MDIR: coupling_step` and the manifest's `resolved_choices.coupling_step` give the request, the value, the category (`rule`), the reason, the inputs and `changes_rounding` (false). `mdir pin` writes the key. A simulation of the Python interface keeps the host path, a rule with its reason. With the device path its tests failed: the bath of a simulation continued from a checkpoint of `mdir run`, and the output of a barostat run in parts against the whole run, differed. The parts of a simulation keep and restore the bath and the cell between calls of the program (`lib/Compiler/Simulation.cpp`, D196, D238), and the device path has not been made to follow them. That is left open (Section 8). The pass also keeps the host path for a function whose buffer of results is full or which has a coupling inside a kernel, with a warning of the compiler that gives the reason. Neither happens in the programs that the driver writes.

## 7. Throughput

RTX 4090, 2 fs, mixed precision, the controls of the gap table (V-rescale every 10 steps in NVT, 25 in NPT, where C-rescale acts too). JAC, TIP3P 10k and Trp-cage ran 100,000 steps with a row every 1,000; Factor IX ran 50,000 with a row every 500. Arms: main (864cad127) and the branch, both with their defaults. The defaults are SPANS and AUTO neighbor structure: GROUPS for JAC and Factor IX, MATRIX for TIP3P 10k and Trp-cage. For JAC both were also run with `neighbor_structure = "TOWERS"` and `term_streams = "TERMS"`.

Two jobs on two cards ran in opposite orders (`gpu-run --idle`). Each job made one untimed warm run per arm, then pass 1 in its order and pass 2 in reverse. The rate is the 10%-trimmed mean of the row intervals after the stationary cut of D417 (`MDIR_CHECKPOINT_TIMES=1`). The median SM clocks under load were 2,730–2,745 MHz, at most 66 °C, and no run throttled.

The force error is the relative RMS force at the input configuration against a double-precision reference, by the protocol of #1007, with main's defaults. Trp-cage's reference is MDIR on the CPU in double precision with PME at $10^{-11}$. The coupling step does not change the forces.

| Input | Force error | main | branch | Change |
|---|---:|---:|---:|---:|
| JAC NVT (23,558) | 1.31e-3 | 1392.6 / 1389.5 | 1410.4 / 1407.8 | +1.3% / +1.3% |
| JAC NVT, TOWERS + TERMS | 1.31e-3 | 1810.3 / 1805.0 | 1839.9 / 1835.5 | +1.6% / +1.7% |
| JAC NPT | 1.31e-3 | 1274.3 / 1274.4 | 1314.9 / 1313.3 | +3.2% / +3.0% |
| JAC NPT, TOWERS + TERMS | 1.31e-3 | 1461.7 / 1457.5 | 1597.3 / 1593.5 | +9.3% / +9.3% |
| Factor IX NPT (90,906) | 1.43e-3 | 583.4 / 584.7 | 590.4 / 592.4 | +1.2% / +1.3% |
| TIP3P 10k NPT (9,989) | 7.32e-4 | 1683.4 / 1694.7 | 1739.9 / 1750.9 | +3.4% / +3.3% |
| Trp-cage ff14SB + TIP3P NVT (6,545) | 2.08e-4 | 1484.0 / 1486.9 | 1500.3 / 1504.5 | +1.1% / +1.2% |
| Trp-cage ff14SB + TIP3P NPT | 2.08e-4 | 1196.4 / 1205.1 | 1260.6 / 1261.9 | +5.4% / +4.7% |

ns/day, first job / second job, each the mean of its two passes.

1. [measured] Every row is faster, by 1.1% to 9.3%. No suite system measured here is slower. Cellulose and STMV were not measured.
2. [computed] On JAC with TOWERS + TERMS, NPT cost 22.8 µs per step more than NVT on main (118.4 against 95.6 µs, the means of the two jobs) and 14.3 µs on the branch (108.3 against 94.0). Removing the reads of the sums and the virials takes 8.5 µs of the 22.8. What remains is the virial kernels (about 11 µs, #1026) and the one read of the pressures.
3. [computed] In NVT the gain is 1.1% to 1.7%: 1.6 µs per step on JAC with either structure (124.2 against 122.6 µs, 95.6 against 94.0), for one copy of 8 kB and one wait per period of 10 steps.

## 8. What remains

- The new cell of a scaling on the device, so that a C-rescale step reads nothing either. This needs grids fixed under a barostat for MATRIX and GROUPS, and launches that read the cell on the device (Section 3).
- The coupling step on the device in a simulation of the Python interface (Section 6).
- Under `GRAPH`, the read of the accepted step that the draws take (D439).
- The strain on the device needs `log` and `exp` that agree with the host's libm to the bit, or a decision that accepts trajectories that differ from those of `HOST` in the last bits of the cell (Section 3).

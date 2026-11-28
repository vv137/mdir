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
3. **The pressures of a scaling.** One thread computes the pressures that the strain takes (one, two or three, for isotropic, semi-isotropic and anisotropic coupling) into a buffer of eight numbers. The host copies that buffer into page-locked memory and computes the strain, the new cell and its checks as before. The virial that a TOWERS dispatch observes for a scaling is read like a sum (its copy is marked as a readback), and the helpers of PME that read back the reciprocal virial are inlined, with a snapshot of their results on the device (as for device frames). The kernels and the one-thread kernels therefore read those virials on the device.
4. **Page-locked results.** With the key, the host's copy of the f64 results of a function is page-locked memory of the runtime (`mdrtHostResults`), so the copies that remain, at outputs, are asynchronous transfers (the maintainer's addition to #1026).

`keepReadbacksOnDevice` (D156) then clones what each one-thread kernel takes from the sums into it. A copy whose loads the host no longer reads is removed. Every host use that remains keeps its copy, so an expression that the pass does not move stays correct on the host.

The one-thread kernels carry `mdir.frame_host_arithmetic`: the passes that contract products or make explicit fused multiply-adds (#97, #102) leave them out.

## 2. Why the trajectory is the same

1. [derived] The draws of V-rescale (a normal number and, by the rejection of [[Marsaglia2000]](../references.md#marsaglia2000), a gamma number) and those of C-rescale depend on the seed, the step and $N_f$, not on the state (`docs/dynamics.md` Section 2.2.6). The host makes the same calls of Philox in the same order. `mdrtBussiRest` draws the first normal number again and discards it, so its gamma number is that of `mdrtBussiFactor`.
2. [derived] The rest of `mdrtBussiFactor` uses only $+$, $-$, $\times$, $/$ and $\sqrt{\ }$, which IEEE 754 rounds correctly on the host (SSE2, without FMA: `libmdrt` is built without `-march`) and on the device (`div.rn.f64`, `sqrt.rn.f64`). The ops follow the C expression's evaluation order: `kinetic + (1 - decay) * (target * (r1 * r1 + rest) / freedom - kinetic) + 2 * r1 * sqrt(kinetic * target / freedom * (1 - decay) * decay)`. The one-thread kernels have no contractions. The factor is therefore the host's to the bit, and so is everything after it.
3. [derived] The bath is one value that receives the same additions in the same order. Starting from the host's value and returning it before every read gives the host the value it had.
4. [derived] The pressures are the same pure ops of the same sums (D156). The strain, the exponential of the scaling, the f32 rounding of the cell and its checks stay on the host's libm, so the cell is the host's to the bit.

So in the deterministic mode the checkpoint and every row of the log are bitwise those of `HOST` (Section 5).

## 3. What the host still reads, and why

| Step | Before (`HOST`) | After (`DEVICE`) |
|---|---|---|
| V-rescale, no output | the results (8 kB) and a wait | nothing |
| C-rescale, no output | the results twice (8 kB each), the observed virial (24 B, TOWERS), the reciprocal virial (56 B) twice, with a wait each | the pressures (64 B) and a wait |
| A row or a checkpoint | the results | the results, the bath |

1. [derived] Under a barostat the next step's launches take the new cell by value. They are the PME wrappers, the builds of MATRIX and GROUPS, whose grids `mdrt_gpu_cell_count` sizes on the host from the cell, and the arguments that the TOWERS mirror of D450 fills. The host therefore needs the cell before it issues the next step. Removing that read needs grids fixed under a barostat (as TOWERS has, D450 Section 3.2) and launches that read the cell on the device.
2. [derived] The strain takes $\ln(\lambda'/\lambda)$, $e^{\varepsilon/3}$ and, with a cell in f32, the logarithm of the rounded scale. To keep the trajectory, the device would have to compute these as this host's libm does, bit for bit. CUDA's `log` and `exp` are not correctly rounded and differ from glibc in the last bit for a share of arguments. This decision keeps the strain on the host and copies the pressures, which the device computes exactly as the host did.
3. [measured] Under `step_graph = "GRAPH"` the coupling step still reads the accepted step of the chunks (`mdrtStepRead`, 8 B, D439), which the draws take. After this change both draws share one read (Section 4).
4. Not covered: the Nose–Hoover chain, which integrates its own state on the host (D163a), and Langevin dynamics, whose thermostat acts in the kernels of every step and has no factor. For these the key resolves to `HOST`, and the record says why (Section 6). The CPU target has no device.

## 4. Copies and waits

From the runtime's counters (`MDRT_HOST_READS=1`, `runtime/mdrt_cuda.c`). Each line is the difference between runs of 4,000 and 2,000 steps of JAC with one row at the end, divided by the number of additional coupling periods (V-rescale every 10 steps in NVT; V-rescale and C-rescale every 25 steps in NPT; RTX 4090, mixed precision). These are per period, all steps included: the 20 B copies are the counts that the builds of the groups read, and the flag waits are the checks of the lists (D113, D118). Neither depends on the coupling.

COUNTS

## 5. Validation

`test/Driver/device-coupling-gpu.test` and `test/Driver/device-coupling-towers-gpu.test` (`Inputs/check_device_coupling.py`) run the dipeptide in water: 1,168 particles, PME, SHAKE and SETTLE, V-rescale every 10 steps, 2 fs, with C-rescale every 10 steps in NPT. The oracle is the same program with `coupling_step = "HOST"`, which is the lowering of before.

- **Deterministic mode, MATRIX: bitwise.** The checkpoint (positions, velocities, forces, cell), the bath that it keeps, and every row of the log are compared byte for byte, over 300 steps. Cases: mixed and double precision; spans and launches one by one (NVT); isotropic, semi-isotropic and anisotropic coupling in mixed precision; and isotropic in double (NPT).
- **A continuation.** From a checkpoint at step 150, 160 steps more with a row every 40 steps: bitwise. The bath goes to the device from the checkpoint.
- **GROUPS** (not deterministic), NVT and NPT, 40 steps: the relative L2 difference of the positions and velocities from `HOST` is at most ten times that of two `HOST` runs, with a floor of $10^{-5}$.
- **TOWERS** (not deterministic), 200 steps at 1 fs, NVT, and NPT at 500 bar with a time constant of 0.1 ps, under `LAUNCH`, `SPANS` and `GRAPH`, against two `HOST` runs under `LAUNCH`, with the same bound. Every row falls on the same step.

TOWERS_TABLE

- **No copy per coupling.** Runs of 200 and 400 steps with one row: the copies that grow with the 20 additional couplings. MATRIX NVT: `HOST` 20 copies of 8 kB, `DEVICE` none. MATRIX NPT: `HOST` 80 of 8 kB and 40 of 56 B, `DEVICE` 20 of 64 B. TOWERS NVT: `HOST` 20 of 8 kB, `DEVICE` none. TOWERS NPT: `HOST` 80 of 8 kB, 40 of 24 B and 40 of 56 B, `DEVICE` 20 of 64 B. Every other copy, the builds' counts, grows alike in both paths, which run the same trajectory to the bit.
- **The record and the cases left on the host.** `test/Driver/coupling-step.test` checks with `mdir check --json`: `DEVICE` by default on the GPU, NVT and NPT; `HOST` on request, on the CPU target, under a Nose–Hoover chain, under Langevin dynamics and without coupling, each with its reason; a value that is not a choice is refused.

## 6. The key

| Value | Meaning |
|---|---|
| `"DEVICE"` (default) | The coupling step on the device, where the run allows it. Resolves to `HOST`, a rule recorded in `resolved_choices.coupling_step` with its reason: on the CPU target, under a Nose–Hoover chain, under Langevin dynamics, in a run without coupling. |
| `"HOST"` | The sums are copied and the coupling step computed on the host, as before. |

The setup log line `MDIR: coupling_step` and the manifest's `resolved_choices.coupling_step` give the request, the value, the category (`rule`), the reason, the inputs and `changes_rounding` (false). `mdir pin` writes the key. Python's `Execution` does not take it; a simulation follows the default. The pass also keeps the host path for a function whose buffer of results is full or which has a coupling inside a kernel. Neither happens in the programs that the driver writes.

## 7. Throughput

TIMING

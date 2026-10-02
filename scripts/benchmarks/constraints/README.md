# Constraint solver experiment

MDIR's current constraint topology contains disjoint XH, XH2 and XH3
clusters, plus three-distance rigid waters. Its default position solver
already uses the nonsymmetric cross-time Newton matrix, six M-SHAKE
iterations, displacement accumulation, the position-induced velocity
impulse, and the final RATTLE projection. There is no large connected
constraint network to partition into useful Tensor Core fronts.

This experiment adds an opt-in specialization for XH groups:

```toml
[constraints]
hydrogen_bonds = true
analytic_bonds = true
```

The default is `false`. For predicted bond `r`, old bond `s`, and target
length `d`, solve `|r + t*s|² = d²` using the rationalized near root:

```
e = d*d - dot(r, r)
b = dot(r, s)
t = e / (b + sqrt(b*b + dot(s, s)*e))
lambda = t / (1/ma + 1/mb)
da = -lambda*s/ma
db = +lambda*s/mb
```

Accept only if `b > 0` and the original corrected squared-distance
residual is at most `1e-12*d²` in double precision or `1e-6*d²` in
mixed/single precision. Ordered comparisons reject NaNs. Rejected
candidates take the existing six Newton iterations; this fallback has
that solver's existing fixed-iteration limitations. These tolerances
apply to the bond displacement calculation, not to coordinates after
rounding to storage precision. The algorithm preserves the old bond
direction and image, the mass-weighted impulse and the existing virial.

## Reproduce the comparison

Build MDIR as in the top-level README. No new dependencies are needed.
The driver switch compares both algorithms in the same binary.

```sh
python3 scripts/benchmarks/constraints/bench.py \
  --mdir /absolute/path/to/build/bin/mdir --work /tmp/bonds-cpu \
  --steps 5000 --repeat 3

CUDA_VISIBLE_DEVICES=1 python3 scripts/benchmarks/constraints/bench.py \
  --mdir /absolute/path/to/build/bin/mdir --work /tmp/bonds-gpu \
  --target GPU --precision MIXED --steps 10000 --repeat 5

CUDA_VISIBLE_DEVICES=1 python3 scripts/benchmarks/constraints/bench.py \
  --mdir /absolute/path/to/build/bin/mdir --work /tmp/bonds-protein \
  --control /path/to/jac_nve/mdir.toml --steps 10000 --repeat 5
```

The synthetic input has `side³` noninteracting diatomics (`--side 12`
by default), at 100 K and 2 fs. It intentionally favors this
specialization. `--control` uses a real system's settings and resolves
its input paths; it disables trajectory/checkpoint output in the copied
control. Runs alternate order, save every log, and report the median
ratio of complete MD-stage times after startup, excluding compilation.
The reported ms/step has 0.001 ms resolution. Use an otherwise idle GPU;
the script requires an explicit single visible device and never selects
additional GPUs.

Initial measurements on 2026-10-02 (Release, GCC 9.5, LLVM 23.1.2):

| System | Target | Newton ms/step | Analytic ms/step | Median ratio |
|---|---|---:|---:|---:|
| 1,728 diatomics, 5,000 steps, 3 repeats | Threadripper PRO 3995WX, one thread, double | 0.413 | 0.252 | 1.64 |
| Vacuum peptide, 10,000 steps, 5 repeats | Threadripper PRO 3995WX, one thread, double | 0.071 | 0.070 | 1.014 |
| 1,728 diatomics, 10,000 steps, 5 repeats | RTX 3090, mixed | 0.043 | 0.043 | 1.00 |
| JAC protein/water, 23,558 atoms, 10,000 steps, 3 repeats | RTX 3090, mixed | 0.211 | 0.211 | 1.00 |

The initial synthetic GPU measurements were made on a shared device
and do not establish a small performance difference. The JAC comparison
used an otherwise idle GPU 1 and likewise showed no resolved gain. CPU NVE relative energy
changes were `1.687e-13` (Newton) and `1.845e-13` (analytic).
Machine-readable timing samples are in `results/`. The vacuum-peptide change is near the timing resolution. No protein/water
speedup is inferred from the favorable synthetic CPU case. A larger
synthetic attempt (`--side 32`, GPU mixed) failed with nonfinite positions
in the baseline before its first timed sample and was excluded; it does
not provide a comparison of algorithms.

## Schur/Tensor Core feasibility

`schur_update.cu` is the supplied reference package's standalone cuBLAS
experiment, compiled with CUDA 13.4 and exercised on the RTX 3090. It
computes `C -= F*T`; it does not implement a constraint solver. For example:

```sh
nvcc -O3 -std=c++17 -arch=sm_86 schur_update.cu -lcublas -o schur_update
CUDA_VISIBLE_DEVICES=1 ./schur_update 3 3 4096
CUDA_VISIBLE_DEVICES=1 ./schur_update 32 64 1024
```

Preliminary shared-device results (cuBLAS 130800), milliseconds per batch:

| separator × interior, batch | Strict FP32 | TF32 permitted | TF32 sampled relative max error |
|---|---:|---:|---:|
| 3 × 3, 4,096 | 0.01329 | 0.01325 | 5.18e-8 |
| 32 × 64, 1,024 | 0.12622 | 0.03940 | 5.21e-5 |

These timings omit assembly, packing, factorization, substitution,
refinement, residuals and integration. A compute-mode request does not
prove Tensor Core instruction selection. The current 1–3-constraint
groups do not provide the larger shape's work; enabling this library call
in MDIR's integrator would add an unsupported performance assumption.
The supplied NumPy/SciPy reference was also rerun successfully, including
cross-time Jacobians, coupled Schur solves, FP16 product refinement,
velocity tangency, redundant-triangle rejection, and toy reversibility.
That reference is separate from MDIR's numerical implementation.

A complete TC-Schur-RATTLE implementation still requires a general
connected-constraint representation, symbolic separators and fill,
GPU-resident factor/solve scheduling, precision refinement and failure
reporting, then force/virial and integration validation. This patch does
not claim that implementation. The profitable path for the current
one-bond topology is a specialization of the existing discrete solve.

## Validation

`test/Driver/analytic-bonds.test` executes the generated MLIR kernel on
67 cases and checks independent bond, momentum, direction, and root
invariants. Cases include unequal/equal/repartitioned masses, tiny and
large corrections, periodic images, near tangency and the backward
predictor's Newton fallback. It also compares the complete vacuum-peptide
trajectory's energy/virial log with the original solver and checks bond
lengths. `analytic-bonds-gpu.test` compares a solvated peptide in double
and mixed precision. `test/Sanitizer/analytic-bonds-gpu.test` exercises
the fused integration under memcheck, initcheck and racecheck.

Validation on 2026-10-02 after merging main: all 163 lit tests passed
with `CUDA_VISIBLE_DEVICES=1`, `-Dsanitize=1`, and the Amber suite enabled.
This includes all nine Amber systems with each variant (18 runs,
50 steps), the 4 fs repartitioned-mass cases, 1,067,095-atom STMV, and
triclinic cells. The smoke harness uses a 50-step horizon and start/end
output to respect its coupling periods and two-row check.

After merging main through `a74977f` (triclinic support and virtual-site
ordering), the synthetic CPU comparison was repeated: median 0.413 versus
0.252 ms/step, 1.64×, as recorded above. The other timing samples predate
that merge; each result names its base commit. The added
`analytic-bonds-triclinic.test` shifts an isolated-bond hydrogen by a
tilted lattice vector and compares CPU, GPU matrix/double, and GPU
groups/mixed runs with the Newton solver. Those comparisons pass.

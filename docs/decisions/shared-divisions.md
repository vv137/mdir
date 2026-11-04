# D[shared-divisions]

Recorded at its merge.

**Under `fast_math`, the f64 divisions of a kernel of a loop over pairs by one value share one reciprocal, and on a device a division by `math.sqrt(x)` is a product by `math.rsqrt(x)` (`md-exec-share-divisions`).** Issue #731, from the measurement of #681 and #723; the maintainer ruled on 2026-11-04 that the shared divisions go in.

## The rewrite

`md-exec-share-divisions` (`lib/Dialect/MDExec/Transforms/ShareDivisions.cpp`)
runs where the experiment of #723 acted: after `md-exec-assign-precision`
(and `md-exec-approximate`), when the types are fixed and `cse` has made
each value one SSA value, and right before `md-exec-assign-algebra`. The
pipeline (`lib/Compiler/Compile.cpp`) includes it when `fast_math` is set,
the default, with `rsqrt=true` for a GPU. In each block of the kernel of an
`md_exec.pair_for`, for the scalar `arith.divf` in f64:

- with `rsqrt`, a division by `math.sqrt(x)` becomes a product by
  `math.rsqrt(x)`, one per root, placed after it; `1 / sqrt(x)` becomes the
  `rsqrt` itself, and a root that nothing else uses is removed;
- two or more divisions by any other SSA value `y` share one reciprocal
  `1 / y`, placed after the definition of `y` (at the start of the block
  when `y` is an argument of the block, as `r2` is); `a / y` becomes
  `a · (1/y)`, and a division `1 / y` becomes the reciprocal. Without
  `rsqrt` a root is such a value.

A lone division by a value that is not a root stays. The kernels of
`md_exec.tuple_for` and `md_exec.particle_for` stay as they are (not
measured; the integrators and constraints keep their arithmetic, as for
D112), as do vector divisions and f32 kernels: under `fast_math` an f32
kernel takes approximate divisions already (D90), and the f32 pair kernel
of JAC has one division, `1/r2`, with nothing to share. The mixed and
single modes therefore do not change.

In the fused pair kernel of JAC in double precision (the direct sum of
Ewald and Lennard-Jones), the divisions `1/r2` and `q/sqrt(r2)` become
`1/r2` and `q · rsqrt(r2)`.

## Why `rsqrt` on a device only

`math.rsqrt` lowers by target:

- On a device (MLIR's MathToNVVM patterns, through `convert-gpu-to-nvvm`)
  to libdevice's `__nv_rsqrt` and `__nv_rsqrtf`. In the libdevice of
  CUDA 13.2, `__nv_rsqrt` is one call of `llvm.nvvm.rsqrt.approx.d`, PTX's
  `rsqrt.approx.f64`; ptxas expands it into `MUFU.RSQ64H` and a refinement
  in f64 (seen in the SASS for sm_120), so it is close to correctly rounded
  although PTX calls it approximate. `__nv_rsqrtf` is `rsqrt.approx.f32`
  (`.ftz` under `__CUDA_FTZ`).
- On the host (MathToLLVM, which the CPU pipeline runs before MathToLibm)
  to `1 / sqrt(x)`, two correctly rounded operations: nothing is saved, and
  a lone `a / sqrt(x)` would round once more. The host therefore shares a
  root like any other divisor.

The error against a reference in long double, in ulp of the result, over
$2^{24}$ arguments of each set: "wide", mantissas uniform and exponents
uniform in $[-30, 30]$; "pair", $r^2$ log-uniform in $[10^{-3}, 1]$ nm²;
$a$ uniform in $[-50, 50]$. Device: RTX 5090 (sm_120), CUDA 13.2,
`-prec-div=true -prec-sqrt=true -ftz=false`; host: x86-64, IEEE `sqrt` and
division. Maximum (mean):

| Quantity | f64 wide | f64 pair | f32 wide | f32 pair |
|---|---|---|---|---|
| device `rsqrt(y)` (libdevice) | 0.507 (0.250) | 0.508 (0.250) | 1.511 (0.320) | 1.514 (0.325) |
| device `1/sqrt(y)` | 1.491 (0.351) | 1.495 (0.357) | 1.490 (0.352) | 1.490 (0.358) |
| device `a · rsqrt(y)` | 1.473 (0.349) | 1.471 (0.346) | 2.398 (0.410) | 2.452 (0.410) |
| device `a / sqrt(y)` (as written) | 1.470 (0.342) | 1.478 (0.345) | 1.473 (0.342) | 1.474 (0.345) |
| host `1/sqrt(y)` (= host `rsqrt`) | 1.491 (0.351) | 1.495 (0.357) | 1.490 (0.352) | 1.490 (0.358) |
| host `a · (1/sqrt(y))` | 1.924 (0.426) | 1.943 (0.426) | 1.945 (0.426) | 1.927 (0.426) |
| host `a / sqrt(y)` (as written) | 1.470 (0.342) | 1.478 (0.345) | 1.473 (0.342) | 1.474 (0.345) |
| `sqrt(y) · (1/y)` | 1.962 (0.427) | 1.962 (0.429) | 1.942 (0.427) | 1.950 (0.429) |
| `a · (sqrt(y) · (1/y))` | 2.758 (0.495) | 2.680 (0.490) | 2.785 (0.495) | 2.744 (0.490) |

In f64 on the device, `a · rsqrt(y)` is as accurate as `a / sqrt(y)`
(1.47 ulp at most, both); on the host `a · (1/sqrt(y))` would be 1.9 ulp
for no gain. In f32 the device's `rsqrt` adds an ulp to the quotient, and
f32 kernels are left out. A shared reciprocal, `a · (1/y)`, rounds twice:
within $2\mathsf u + \mathsf u^2$ of $a/y$ against $\mathsf u$ for the
quotient as written ($\mathsf u = 2^{-53}$; white paper, Section 7.2).

`sqrt(y) · (1/y)`, which needs no `rsqrt` when the kernel divides by `y`
too, was tried as well: it leaves the same single division, but it rounds
four times and reaches 2.8 ulp.

## Accuracy against the current program

To follow in #734.

## Determinism

To follow in #734.

## Timing

To follow in #734.

## Bits

The bits of every double run with `fast_math` change against the program
before this decision, on the CPU and on a GPU, in the default and the
deterministic modes; the mixed and single modes are unchanged. The rewrite
is fixed at compilation, so a deterministic run gives the same bits from
run to run.

## Ablation

No control key of its own: the maintainer ruled on #734 (2026-11-04) that
the existing handles are enough. The pass is one entry of the pipeline, and
an ablation takes one of two routes:

- `fast_math = false` in `[execution]` leaves it out, together with the
  other rewrites of `fast_math` (`md-exec-simplify-distance`,
  `md-exec-approximate`, `md-exec-expand-radial`, contraction);
- to leave out this pass alone, print the pipeline of the run, remove the
  pass, and give the result through `MDIR_PIPELINE`
  ([environment.md](../environment.md)), which reproduces the program
  before this decision:

  ```sh
  mdir emit run.toml --stage=pipeline \
    | sed -E 's/md-exec-share-divisions(\{[^}]*\})?,//' > pipeline
  MDIR_PIPELINE="$(cat pipeline)" mdir run run.toml
  ```

## Left open

- Tuple kernels: the kernel of the excluded pairs of JAC divides four times
  by one root; not measured (#731 asks for pair kernels first).
- A device with full-rate f64: the RTX 5090 executes f64 at a small
  fraction of its f32 rate, so its double step is bound by f64 arithmetic
  and the gain there overstates the gain on such a device.

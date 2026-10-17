# Measuring Ewald summation with prolates against MDIR's PME

The scripts of this directory measure, for issue #334 (D266),
whether the prolate spheroidal wave function of order zero (PSWF) as the
splitting and spreading kernel of particle mesh Ewald, "Ewald summation
with prolates" (ESP) [[Liang2026]](../../../docs/references.md#liang2026),
is faster on one GPU than MDIR's Gaussian splitting with B-splines at the
same accuracy. None of it is part of MDIR; the program `esp_bench.cu` is a
prototype of the kernels, written to be measured.

The accuracy is the relative force error of the paper's eq. 16,

$$
\Delta = \frac{\big(\tfrac1N\sum_i\lvert\mathbf F_i - \mathbf F_i^\text{ref}\rvert^2\big)^{1/2}}
{\big(\tfrac1N\sum_i\lvert\mathbf F_i^\text{ref}\rvert^2\big)^{1/2}},
$$

for the Coulomb forces of all pairs (no exclusions: they add the same
analytic correction to every method), against a reference that is the
Gaussian Ewald sum at two settings ($r_c$ = 1.4 and 1.6 nm, erfc $10^{-12}$
and $10^{-14}$, B-splines of order 10 and 12 on grids of 0.40 and 0.34 Å),
which agree to $2\times10^{-12}$.

## Files

| File | What it does |
|---|---|
| `pswf.py` | $\psi_0^c$ as a Legendre series whose coefficients are the eigenvector of the tridiagonal matrix of the prolate operator [[Xiao2001]](../../../docs/references.md#xiao2001); the bandwidth $c$ from $\psi_0^c(1) = \varepsilon$ with $\int_{-1}^1\psi^2 = 1$ (the paper's Table 2: 9.5392, 12.024, 14.471 at $10^{-3}$, $10^{-4}$, $10^{-5}$); $\chi$, $\Phi$, $\hat\chi$ normalized to $\hat\chi(0) = 1$, and the window $\phi_1$ |
| `accuracy.py` | The systems of the Amber suite, the direct sum over pairs, a reciprocal sum with B-splines or the PSWF window (energy and forces by the analytic derivative), $\Delta$ |
| `run_ref.py NAME [NSAMPLE]` | The reference forces, of all particles or of a sample |
| `sweep.py NAME gauss ORDER`, `sweep.py NAME esp P`, `sweep.py NAME points` | $\Delta$ against the grid, for B-splines of order 4, 6, 8 over $\mathrm{erfc}(\beta r_c)$ from $3\times10^{-4}$ to $10^{-7}$, and for ESP with $P$ = 4 to 8 over $\varepsilon$ from $10^{-3}$ to $5\times10^{-6}$ and $c_1 = c(\varepsilon)$ or $c(\varepsilon/2)$; `points`: MDIR's default and the suite's settings |
| `best.py NAME` | The smallest grid that reaches each target $\Delta$, by method |
| `check_cases.py NAME NSAMPLE` | $\Delta$ of the chosen cases on a sample of another system |
| `gen_case.py`, `make_cases.py NAME...` | The input of the prototype: positions, charges, the polynomials of the weights (fitted in f64 to $3\times10^{-8}$ of $\psi_0^{c_1}(0)$), the factors along each edge, and $\hat\chi$ as a Chebyshev series in $2x^2 - 1$ |
| `esp_bench.cu` | The prototype: places, spreading (`brick` as MDIR's order 4, D108 and D109; `direct`; `tiled4`, `tiled8`: the particles binned by their first point each step and spread into blocks of the grid in shared memory, as cuFINUFFT spreads [[Shih2021]](../../../docs/references.md#shih2021)), cuFFT, the product, gathering (G lanes a particle; 32 lanes take the weights by shuffles); f32 or f64; prints the median time of each step and of the sequence |
| `proto_job.py`, `proto_all.py`, `table.py`, `compare.py` | A GPU job over the cases and variants; the best variant of each case; the table of the PR by system and target |
| `pswf_reference.py C...` | ψ₀ᶜ by the Nyström method on the integral operator in mpmath, the reference of `test/Driver/prolate-function.test` |
| `mk_mdir.py NAME STEPS`, `mdir_job.py RUNS OUT ROUNDS MDIR` | Control files of `mdir run` for the Gaussian cases and for order 4 at the ESP grids; a GPU job that alternates them in rounds, the first dropped, then one profiled run of each |
| `grid_sweep.py make/run/report` | MDIR's own $\Delta$ on the CPU in double at explicit grids, over the Coulomb forces of the topology (the forces less those of the system without charges), against the Gaussian at erfc 10⁻¹³, 0.33 Å, order 8, checked by the prolate kernel at 10⁻⁹; the eq. 10 estimate of each run; the calibration of the grid rule |
| `perf_cases.py NAME STEPS OUT [MAIN]`, `ws_table.py LOG...` | The whole step at matched $\Delta$ (the grids by the spacing of JAC), with runs of a build of `main` for the order-4 path; the table from the round lines of a job's log |
| `dynamics.py make/run/report` | JAC in NVE (the drift, with the slopes of the two halves) and NPT (the mean volume), both kernels, mixed and double precision |
| `cpu_cases.py accuracy/mixed/pick` | The CPU target (issue #371): the sweep of $\Delta$ at a cutoff of 9 Å for `grid_sweep.py run` and `report` (the Gaussian of orders 4, 6, 8 at erfc $10^{-4}$ to $10^{-7}$ over spacings, the prolate kernel at its automatic grid for $P$ = 5, 6, 7), the same cases in mixed precision, and the matched cases of each tier $\varepsilon$ = $10^{-3}$, $10^{-4}$, $10^{-5}$: for each Gaussian order the grid of fewest points whose $\Delta$ is at most 1.1 times that of the prolate default |
| `cpu_job.py NAME CASES TIER ROUNDS 16,4,1` | One job a system and tier: `mdir run` on the CPU target in mixed precision, every case at 16 threads and the prolate default and the three Gaussian orders at 4 and 1, alternated in rounds (the first dropped), then one run of each under the sampler of `scripts/benchmarks/cpu-breakdown/` |
| `cpu_table.py NAME...` | The step at each number of threads, the parts of the sampled runs (`cpu-breakdown/prof.py --json`), and an estimate of the step with a shorter, Coulomb-only cutoff from the parts |

## Use

`ESP_WORK` names a work directory and `MDIR_BENCH_DIR` the directory that
`scripts/benchmarks/amber/bench.py prepare` filled. The Python scripts need
NumPy and SciPy, and ParmEd to read the systems once.

```sh
python run_ref.py jac                        # reference of all 23,558 particles
python sweep.py jac points
python sweep.py jac gauss 4                  # and 6, 8; esp 4 ... 8 (one process each)
python best.py jac
python make_cases.py jac cel stmv
nvcc -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_120,code=sm_120 -o $ESP_WORK/esp_bench esp_bench.cu -lcufft
python proto_all.py 4090 brick,tiled4,tiled8 f jac cel stmv   # on the GPU
python table.py $ESP_WORK/proto2-jac-4090.jsonl check
python run_ref.py cel 3000 && python check_cases.py cel 3000
python mk_mdir.py jac 10000 && python mdir_job.py $ESP_WORK/runs/runs-jac.json out.jsonl 6 PATH_TO_MDIR
```

The CPU target: `ESP_CPU_DIR` holds `sys/NAME/` (`sys.prmtop`, `sys.inpcrd`,
and `npt.h5` for a state equilibrated at constant pressure, which is then
the input), `MDIR` names a frozen copy of the program, and `SAMPLER` the
library built from `cpu-breakdown/sampler.c`. Each job has 16 cores.

```sh
python cpu_cases.py accuracy jac $ESP_CPU_DIR/acc/jac
python grid_sweep.py run $MDIR $ESP_CPU_DIR/acc/jac 4      # 4 runs of 4 threads at once
python grid_sweep.py report $MDIR $ESP_CPU_DIR/acc/jac $ESP_CPU_DIR/acc/jac/report.jsonl
python cpu_cases.py pick $ESP_CPU_DIR/acc/jac $ESP_CPU_DIR/acc/jac/cases.json
python cpu_cases.py mixed $ESP_CPU_DIR/acc/jac CASE...     # then run, report, and pick again
python cpu_cases.py accuracy cel $ESP_CPU_DIR/acc/cel $ESP_CPU_DIR/acc/jac/cases.json
python cpu_job.py jac $ESP_CPU_DIR/acc/jac/cases.json 1e-5 6 16,4,1 --tag t1e-5
python ../cpu-breakdown/prof.py RUN/prof --program RUN/mdir.toml --json RUN/parts.json
python cpu_table.py jac tip3p-32k tip3p-103k cel
```

The direct sum as a table of $r^2$ (issue #380, D[prolate-pair-table]):
`PAIR_DIR` holds `sys/NAME/` as above (`jac`, `tip3p-34k`, `tip3p-103k`,
`cellulose`); `MAIN` and `BRANCH` name frozen copies of the two programs.
The force error with the table and with the cells (`MDIR_PROLATE_CELLS=1`)
on the CPU and a GPU, the tables rebuilt as the pass builds them, and the
step alternated against `main` (16 cores; on a GPU, an idle device):

```sh
python pair_delta.py make $PAIR_DIR/delta/jac $PAIR_DIR/sys/jac/sys.prmtop $PAIR_DIR/sys/jac/sys.inpcrd
python pair_delta.py run $BRANCH $PAIR_DIR/delta/jac 4
python pair_delta.py report $BRANCH $PAIR_DIR/delta/jac
$BRANCH emit run.toml > module.mlir && python pair_table.py module.mlir 0.9
python pair_job.py $PAIR_DIR/cpu/jac-1e-5 jac 1e-5 5 16,4,1 main=$MAIN,br=$BRANCH --sample
python pair_job.py $PAIR_DIR/gpu/jac jac 1e-3,1e-4,1e-5 5 1 main=$MAIN,br=$BRANCH --gpu --profile
MDIR_MAIN=$MAIN MDIR_BRANCH=$BRANCH python pair_summary.py $PAIR_DIR/cpu $PAIR_DIR/gpu
```

# Measuring Ewald summation with prolates against MDIR's PME

The scripts of this directory measure, for issue #334 (D[prolate-ewald]),
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
| `esp_bench.cu` | The prototype: places, spreading (`brick` as MDIR's order 4, D108 and D109; `direct`; `tiled4`, `tiled8`), cuFFT, the product, gathering (G lanes a particle; 32 lanes take the weights by shuffles); f32 or f64; prints the median time of each step and of the sequence |
| `proto_job.py`, `proto_all.py`, `table.py` | A GPU job over the cases and variants; the best variant of each case |
| `mk_mdir.py NAME STEPS`, `mdir_job.py RUNS OUT ROUNDS MDIR` | Control files of `mdir run` for the Gaussian cases and for order 4 at the ESP grids; a GPU job that alternates them in rounds, the first dropped, then one profiled run of each |

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

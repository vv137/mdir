# A K–Cl NBFIX fitted to the osmotic pressure

The σ of the K⁺–Cl⁻ pair of the Joung–Cheatham ions in TIP3P water
[[JoungCheatham2008]](../../docs/references.md#joungcheatham2008),
[[Jorgensen1983]](../../docs/references.md#jorgensen1983), fitted to the
measured osmotic pressure of KCl(aq) by reweighting the rows of the runs
in the parameter [[ThalerZavadlav2021]](../../docs/references.md#thalerzavadlav2021).
This is the study of #140. The osmotic pressure is the force on two walls
that act on the ions alone [[LuoRoux2010]](../../docs/references.md#luoroux2010).
The fit trains at about 2 m and holds about 1 m out as a check of
transfer.

## The method

**The parameter.** The correction of the K–Cl pair is a pair term that is
linear in two numbers:

$$u(r) = x_{12}\,\frac{c_{12}}{r^{12}} - x_6\,\frac{c_6}{r^6},\qquad c_{12} = 4\varepsilon_0\sigma_0^{12},\quad c_6 = 4\varepsilon_0\sigma_0^6,$$

where σ₀ = 3.75781 Å and ε₀ = 0.0830263 kcal/mol are the
Lorentz–Berthelot values of the pair. The pair then has $C_{12}(1 + x_{12})$
and $C_6(1 + x_6)$. The fitted parameter is σ' at fixed ε₀, with
$x_{12} = (\sigma'/\sigma_0)^{12} - 1$ and $x_6 = (\sigma'/\sigma_0)^6 - 1$.

**Reweighting without frames.** The term's `observe = ["x12", "x6"]` writes
$A_{12} = \partial U/\partial x_{12}$ and $A_6 = \partial U/\partial x_6$ at
every row. They include the tail and the estimate of the shift at the row's
volume (D209, D210). Since U is linear in $x_{12}$ and $x_6$,

$$U_{\sigma'}(S_n) - U_{\hat\sigma}(S_n) = \Delta x_{12}\,A_{12}(S_n) + \Delta x_6\,A_6(S_n)$$

exactly. Every reweighted average at any σ', its effective sample size, and
the gradient $-\beta\,\mathrm{Cov}(O, \partial U/\partial\sigma')$ therefore
come from two numbers a row, with no coordinates stored.
`test/Driver/kcl-osmotic-reweight.test` checks this identity against the
energies of other states and against single points.

**The observables.** Both come from the columns of `observe`:

- The osmotic pressure is $\Pi = (\langle$`lower.d_z0`$\rangle - \langle$`upper.d_z0`$\rangle)/(2A)$, the force on the two flat-bottom walls ($k$ = 10 kcal/mol/Å², at $L_z/4$ and $3L_z/4$) over the area $A$ of the cell.
- The concentration $c$ is the KCl molarity in the slab $\lvert z - L_z/2\rvert < 10$ Å, counted by a term of zero strength (`slab.d_s`, which counts K⁺ and Cl⁻).

**The target.** The experimental value is the osmotic pressure against
pure water across a membrane that passes only water:

$$\Pi_\text{exp} = -\frac{RT\ln a_w}{\bar V_w} = \frac{\varphi\,\nu\,m\,M_w RT}{\bar V_w},$$

with these ingredients:

- **φ(m):** the osmotic coefficient of Hamer and Wu, Table 28 (KCl, 25 °C) [[HamerWu1972]](../../docs/references.md#hamerwu1972).
- **m ↔ c:** the molality m of the slab's molarity c, from the densities of the CRC Handbook's table of concentrative properties (20 °C) [[CRCHandbook]](../../docs/references.md#crchandbook), which takes them from [[SohnelNovotny1985]](../../docs/references.md#sohnelnovotny1985).
- **$\bar V_w$:** the molar volume of pure water at 25 °C.

The fit drives $f(\sigma') = \langle\Pi\rangle_{\sigma'} - \Pi_\text{exp}(\langle c\rangle_{\sigma'})$
to zero. The concentration stays in $f$: changing σ' moves ions between
the slab and the walls.

**The update.** From the rows of the last runs at $\hat\sigma$, `osmotic.py`:

1. reweights $f$ to a grid of σ';
2. finds the root within the region where the pooled $N_\text{eff}/N$ (Kish) stays at or above 0.5, or, when $f$ has no root there (short runs leave it noisy), takes the Newton step $-f/(\partial f/\partial\sigma')$ of the covariance gradient;
3. limits the step to that region.

The next runs sample there, each replica from its last state. The loop
stops when $\lvert f\rvert$ of a fresh set of runs is below its standard
error. Errors are taken over the independent replicas, and the error of the
root by the jackknife over them.

## The files

| File | What |
|---|---|
| `build.py` | KCl and TIP3P on a lattice in 48 × 48 × 96 Å, the ions in the middle half; `tleap` (AmberTools) writes `system.prmtop` and `system.inpcrd` |
| `1-min.toml` | Steepest descent, 2000 steps, the ions restrained at 10 kcal/mol/Å² |
| `2-nvt.toml` | Velocities at 300 K, 100 ps of 2 fs at constant volume, Langevin at 1/ps, the same restraints |
| `3-npt.toml` | 1 ns at 1 atm with stochastic cell rescaling, the ions restrained at 1 kcal/mol/Å² with their references scaled with the cell |
| `4-md.toml` | Production at constant volume in the cell of stage 3, at σ', without restraints; a row every 0.2 ps |
| `run.py` | Fills the templates for each replica, runs them, and runs the fit |
| `osmotic.py` | The experimental target, the analysis, and the reweighting |

All stages use the following settings:

- PME with a 9 Å cutoff and `coulomb_modifier = "POTENTIAL_SHIFT"`.
- The Lennard-Jones truncated (`NONE`), with the correction for the dispersion. The correction includes the K–Cl term (D209), and the columns of `observe` are those of the shifted potential that the forces sample (D210).
- SETTLE on the waters.
- A GPU in mixed precision.

The walls and the slab scale with the cell in stages 1–3. In stage 4 they
stand at fixed z in the cell of each replica's stage 3.

## Running it

```sh
python3 examples/kcl-osmotic/build.py kcl-2m --pairs 126 --waters 6858
python3 examples/kcl-osmotic/run.py equilibrate kcl-2m          # 8 replicas
python3 examples/kcl-osmotic/run.py fit kcl-2m                  # sigma0, then updates

python3 examples/kcl-osmotic/build.py kcl-1m --pairs 65 --waters 7108
python3 examples/kcl-osmotic/run.py equilibrate kcl-1m
python3 examples/kcl-osmotic/run.py sample kcl-1m --name s0 --ns 5.5
python3 examples/kcl-osmotic/run.py sample kcl-1m --name h1 --sigma <fitted> --start s0 --ns 5.5
python3 examples/kcl-osmotic/run.py analyze kcl-1m --name h1 --sigma <fitted>
```

Details:

- `mdir` is taken from `PATH`, or from `--mdir` or `MDIR`.
- The scripts need NumPy, and `build.py` needs `tleap`.
- `run.py` runs the replicas at once (`--parallel` sets how many), each in `<system>/rep<i>`, and keeps the history of a fit in `<system>/fit.json`. A fit run again skips the sets of runs that are complete.
- `osmotic.py SIGMA_HAT <system>/rep*/NAME` analyzes any set of runs directly.

**Run lengths.** By default each production run is 2.5 ns, with the first
0.5 ns dropped, so 8 replicas give 16 ns per point. On an RTX 5090 a set of
8 runs at once advances at about 1,160 ns/day in all, so a point takes
about 25 minutes and the equilibration about 30. At 16 ns per point the
standard error of $f$ is about 2 bar at 2 m. A run of the example at its defaults at about 2 m
(`build.py`, `equilibrate`, `fit`) on an RTX 5090 took 70 minutes:

| Point | $f$ |
|---|---|
| σ₀ | −4.1 ± 2.4 bar |
| σ₀ + 0.007 Å (one update, at the edge of the trust region) | +1.4 ± 2.7 bar |

It stopped at σ₀ + 0.007 Å, the point the study reached with five times
the sampling.

The published numbers (#140, Step 3) came from longer runs:

- 8 replicas × 10.5 ns at about 2 m (80 ns a point);
- 8 × 5.5 ns at 1 m (40 ns);
- the first 0.5 ns of each run dropped;
- `--ns 10.5` and `--ns 5.5` reproduce these lengths.

The systems of the study were built with OpenMM's Modeller rather than the
lattice of `build.py`, with the same numbers of ions and waters and no
restraints in stages 1–3.

## Results of #140 (RTX 5090, mixed precision)

| | σ₀ = 3.75781 Å | σ₁ = 3.76481 Å |
|---|---|---|
| About 2 m (1.95 M), $f$ (training) | −2.14 ± 0.98 bar | +0.89 ± 1.09 bar |
| About 1 m (0.98 M), $f$ (held out) | −0.09 ± 1.23 bar | +1.05 ± 1.16 bar |

- **The fitted σ'** is σ₀ + 0.006 = 3.7638 ± 0.0033 (stat.) ± 0.004 (syst.) Å. This is $C_{12}$ +1.9% and $C_6$ +1.0% over Lorentz–Berthelot. The systematic is mostly the PME grid, bounded by reweighting 2,400 frames to a 160 × 160 × 320 grid of order 8.
- **One update** from σ₀ met the stopping rule.
- **The held-out point** agrees with experiment within 1 SE, and with the reweighted prediction from σ₀ (+0.29 ± 1.35 bar).
- **Correlation times**, as $g\,\Delta t$ by blocking: about 10 ps for Π, 200–300 ps for c, and 25–55 ps for $\partial U/\partial\sigma'$.
- **The NPT stage** reports −5 ± 5 atm against its 1 atm setting.

The comments of #140 give the full tables. Under the cruder form
$\Pi = \varphi\,\nu\,c\,RT$, which is lower by $c/(m\rho_w^0)$ (2.7% at 1 m,
5.7% at 2 m), the same runs would put the force field 3 bar above
experiment at 2 m rather than 2 bar below.

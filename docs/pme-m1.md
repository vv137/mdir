# Particle Mesh Ewald in M1

Status: decided (2026-09-30), D69 to D71. Stage M1h of
[design-m1.md](design-m1.md), whose Section 8.2 lists the parts. The keys
in brackets are those of [references.md](references.md).

## 1. The sum

The Coulomb energy of a periodic system of point charges is split by the
Ewald sum [[Ewald1921]](references.md#ewald1921) with a splitting
parameter β, in the units of MDIR with `f` the Coulomb constant
([conventions.md](conventions.md)):

```text
E = E_dir + E_rec + E_self + E_excl + E_Q

E_dir  = f Σ_{i<j, not excluded, r < rc} q_i q_j (erfc(β r) / r − s)
E_rec  = f / (2π V) Σ_{m ≠ 0} exp(−π² m² / β²) / m² · |S(m)|²
E_self = −f β / √π · Σ_i q_i²
E_excl = −f Σ_{(i,j) excluded} q_i q_j erf(β r) / r
E_Q    = −f π Q² / (2 V β²)
```

| Symbol | Meaning |
|---|---|
| `m` | A vector of the reciprocal lattice, `(k_x / L_x, k_y / L_y, k_z / L_z)` with whole `k`, in nm⁻¹ |
| `S(m)` | The structure factor `Σ_i q_i exp(2π i m · x_i)` |
| `s` | The shift of the direct sum: `erfc(β rc) / rc`, or 0 |
| excluded | The pairs of the relation `excluded` of the topology: one, two, and three bonds apart, with the sites of Section 19 of design-m1.md. The pairs three bonds apart are among them; their scaled Coulomb is the term `pairs14`, as before, and E_excl takes their share of E_rec out again. |
| `Q` | The net charge `Σ_i q_i`; E_Q is the energy of the uniform background that neutralizes it [[Hub2014]](references.md#hub2014) |

E_rec is computed by smooth particle mesh Ewald
[[Essmann1995]](references.md#essmann1995), after
[[Darden1993]](references.md#darden1993): the charges are spread to a
grid of `K_1 × K_2 × K_3` points with cardinal B-splines of order `p`
(4 by default), the grid is transformed by a real-to-complex FFT, each
point is multiplied by the influence function, and the inverse transform
gives the potential on the grid, from which the forces follow by the
derivatives of the same B-splines:

```text
Q(k)    = Σ_i q_i M_p(K_1 u_1i − k_1) M_p(K_2 u_2i − k_2) M_p(K_3 u_3i − k_3)
E_rec   = ½ Σ_k Q(k) (θ ⋆ Q)(k),   θ = F⁻¹[B C]
B(m)    = |b_1(m_1)|² |b_2(m_2)|² |b_3(m_3)|²       the Euler exponential splines
C(m)    = f / (π V) · exp(−π² m² / β²) / m²,  C(0) = 0
```

with `u_ai` the fractional coordinates of particle `i` along edge `a`.

The virial of the reciprocal sum, in the convention of
[conventions.md](conventions.md) (`W = Σ d ⊗ F`, the pressure
`(2K + tr W) / (3V)`), is the derivative under a strain of the cell:

```text
W_ab = Σ_{m ≠ 0} E_m (δ_ab − 2 (1 / m² + π² / β²) m_a m_b)
```

where `E_m` is the share of `m` in E_rec. E_Q adds `E_Q δ_ab`; E_self adds
nothing; E_dir and E_excl add the virial of their pairs, as every term over
pairs does.

## 2. What is new, and where (D69)

| Part | Where |
|---|---|
| E_dir | The pair term of the topology, `f q_i q_j (erfc(β r) / r − s)`: no new op. `math.erfc` has a derivative. |
| E_excl | An `md.sum_tuples` over the relation `excluded` with `distance(0, 1)` and the charges gathered: no new op |
| E_self, E_Q | Numbers from the driver |
| E_rec | One new op, `md.reciprocal %x, %q, %cell, %influence`, with the grid, the order, and β as attributes, that yields the energy. Differentiation asks the same op for the forces and the virial, which it computes from the grid; it is not differentiated through. |
| The lowering of `md.reciprocal` | Templates in IR, as the build of neighbor structures is (`lib/Runtime/Templates`): spreading, the product with the influence function with the sums of the energy and the virial, and gathering the forces, for the CPU and for a GPU. The FFT is a call of the runtime. |
| FFT | On the host, pocketfft [[Reinecke2019]](references.md#reinecke2019) under the BSD license, in `libmdrt`; on a device, cuFFT of the CUDA toolkit, in `libmdrt_cuda`, with a plan kept for each size of grid (D64). |
| The influence function | `B C` for each point of the half-complex grid, computed by the driver from β, the cell, and the grid, and passed as a table. A cell that changes (the barostat, M1j) makes it a value to compute again. |

## 3. Spreading that does not depend on the order (D70)

Many particles add to one point of the grid. Additions of floating-point
numbers in the order in which threads arrive give results that differ from
run to run, which breaks the deterministic level (P6). The grid is
therefore accumulated in fixed point [[LeGrand2013]](references.md#legrand2013):
each contribution `q_i M M M` is rounded to a 64-bit integer at the scale
2⁴⁰ and added with an integer atomic, whose sum does not depend on the
order. The grid is converted to floating point before the FFT.

| Item | Value |
|---|---|
| Scale | 2⁴⁰: a contribution is resolved to 10⁻¹² e, and a point holds up to 2⁶³ / 2⁴⁰ ≈ 8 × 10⁶ e, far beyond any point of a real system |
| Host | The same fixed point, with atomics of the threads of OpenMP |
| Mixed precision | The B-splines in f32, the accumulation in fixed point, the FFT in f32 on a device and in f64 on the host; the energy and the virial summed in f64 |

## 4. Parameters (D71)

| Keyword of `[energy]` | Meaning | Default |
|---|---|---|
| `electrostatic = "PME"` | Particle mesh Ewald, with the cutoff `cutoffdist` for the direct sum | |
| `pme_alpha` | β, in Å⁻¹ | From `pme_alpha_tol` |
| `pme_alpha_tol` | β such that `erfc(β rc) = pme_alpha_tol`, by bisection, as both engines find it | 10⁻⁵ |
| `pme_ngrid_x`, `_y`, `_z` | The numbers of points of the grid | From `pme_max_spacing` |
| `pme_max_spacing` | The largest spacing of the grid, in Å; each number of points is the smallest product of 2, 3, 5, and 7 that gives no wider spacing | 1.2 |
| `pme_nspline` | The order of the B-splines, 4 to 8 | 4 |
| `pme_shift` | Shift the direct sum to 0 at the cutoff, as GROMACS does by default | false, as sander |

β, the grid, and the order can each be given, so that a run can take those
of sander (`ew_coeff`, `nfft1` to `nfft3`, `order`) or of GROMACS
(`ewald-rtol`, `fourier-nx` to `-nz`, `pme-order`) for a comparison.

## 5. How the engines differ

| Item | sander (`eedmeth = 1`) | GROMACS (Verlet) | In MDIR |
|---|---|---|---|
| The direct sum at the cutoff | Not shifted | Shifted (`coulomb-modifier = Potential-shift`) | `pme_shift` |
| The terms of the log | `EEL`: all but the pairs three bonds apart | `Coulomb (SR)` and `Coul. recip.`; which of E_self, E_excl, and the shift of excluded pairs goes into which is found by comparison before the terms are compared one by one | The five terms apart, and their sum |
| A net charge | The background, with a warning | The background, with a warning | E_Q |

## 6. Validation

| Test | Compared with | Tolerance |
|---|---|---|
| The three waters of Section 19 of design-m1.md, energy and forces | An Ewald sum in a script: the direct sum over periodic images, many wave vectors, E_self, E_excl | 10⁻⁶ with a fine grid |
| The virial | `−dU/dλ` under a uniform scaling of the positions and the cell at fixed numbers of points (`test/Driver/Inputs/check_virial.py`) | 10⁻⁵ |
| Dipeptide in OPC | sander with `ew_coeff`, `nfft`, and `order` of MDIR | The ratio of the Coulomb constants |
| A peptide in water | GROMACS with the same β, grid, and order, with the shift | 10⁻⁶ (mixed precision) |
| Rigid OPC with SETTLE at 2 fs, at constant energy | The change of the total energy against the square of the time step | |
| CPU, OpenMP, GPU, mixed precision | Each other; a GPU in double equals the CPU where spreading is deterministic | |

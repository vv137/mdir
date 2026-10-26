# D[nep-design]

Recorded at its merge.

**A NEP4 model of GPUMD is evaluated as the shape of EAM: one gather over the pairs into a field of radial descriptors and angular moments, then a neural stage of each atom for the invariants and the network, with forces and virial from `md-differentiate`; no triplet loop and no new op.**

By the addition theorem, NEP's three-body descriptor
$\frac{2l+1}{4\pi}\sum_j\sum_k g_n g_n P_l(\cos\theta_{jik})$, $j = k$
included, is $\sum_m \lvert\sum_j g_n Y_{lm}\rvert^2$; the four-body and
five-body terms are $\kappa\,\mathrm{tr}\,\mathsf T^3$ of the $l = 2$ tensor
and $c_5\lvert\mathbf v\rvert^4$ of the $l = 1$ vector of the same sums
([nep.md](../nep.md), Section 2). The field has
$(n^R_\max+1) + (n^A_\max+1)[(l_\max+1)^2-1]$ components (202 for the water
model), the field type of #441; the pair and atom stages are `md.stage`
(D288); the derivative is the map of the atom stage's VJP and a second
gather over the pairs that reads the adjoint at both ends, as for EAM
(D279). GPU triplets (#108) are not needed.

An independent evaluator (`scripts/validation/nep/reference.py`) agrees
with calorine's f64 evaluator to $1.8\times10^{-15}$ eV per atom and
$2.0\times10^{-13}$ eV/Å on a water and a carbon model, and the direct
triplet sum agrees with the moments to $2.5\times10^{-16}$ in the
descriptors; f32 per pair with f64 sums differs from f64 by at most
$4.1\times10^{-7}$ eV per atom and $2.1\times10^{-5}$ eV/Å, GPUMD's f32
from calorine by $7.1\times10^{-7}$ and $6.3\times10^{-5}$.

The keys are proposed, not ruled: `[[energy.model]]` with `backend =
"NEP"`. The reader writes the forward and VJP bodies of both stages, if the
maintainer rules so against D167. The implementation is #584, qNEP's
charges #585. Closes #583.

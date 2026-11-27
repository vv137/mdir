# D[pme-default-tolerance]

Recorded at its merge.

The default `[pme] tolerance` of the Gaussian kernel is $5\times10^{-5}$,
was $10^{-5}$; the default largest spacing stays 1.2 Å (#927). β follows
from $\mathrm{erfc}(\beta r_c)$ = `tolerance` as before: 0.3186 Å⁻¹ at
9 Å and 0.3585 Å⁻¹ at 8 Å, against 0.3470 and 0.3904. The grid of a
system does not change, so neither does the cost of a step. The prolate
kernel keeps its default $\psi_0^c(1) = 10^{-5}$ (D266), in the control
file and in Python (`System.pme_tolerance`, which takes the Gaussian
default where it is not set). A given `tolerance`, `beta` or `grid` is
used as before.

## Why

At the default grid the reciprocal part dominates the error: a smaller β
lowers it more than the larger tail $\mathrm{erfc}(\beta r_c)$ of the
direct sum raises it. Relative RMS force error over all atoms at the
input configuration, MDIR mixed precision on an RTX 4090 against the
double-precision references of the benchmark (OpenMM CUDA double at
$10^{-6}$, Cellulose and STMV at the validated references), 2 NVE steps of
$10^{-9}$ ps:

| Input | Grid at 1.2 Å | $10^{-5}$ (old) | $5\times10^{-5}$ (new) | new / old | $10^{-4}$ | $5\times10^{-5}$ at 1.0 Å |
|---|---|---:|---:|---:|---:|---:|
| TIP3P 10k, 9 Å | 40³ | 7.32e-4 | 4.99e-4 | 0.68 | 5.12e-4 | 2.81e-4 |
| TIP3P 20k, 9 Å | 50³ | 7.62e-4 | 5.17e-4 | 0.68 | 5.24e-4 | 2.91e-4 |
| JAC, 8 Å | 54³ | 1.311e-3 | 8.24e-4 | 0.63 | 7.35e-4 | 4.37e-4 |
| Factor IX, 8 Å | 120×70×70 | 1.430e-3 | 8.91e-4 | 0.62 | 7.84e-4 | 4.76e-4 |
| Cellulose, 8 Å | 224×108×108 | 1.228e-3 | 7.74e-4 | 0.63 | 6.90e-4 | 4.15e-4 |
| STMV, 8 Å | 192³ | 1.267e-3 | 7.92e-4 | 0.63 | — | 4.14e-4 |

In double precision, on the GPU and on the CPU, the errors of TIP3P 10k
and JAC are those of mixed precision to three digits: $7.32\times10^{-4}$
to $4.99\times10^{-4}$ and $1.311\times10^{-3}$ to $8.24\times10^{-4}$;
at these errors the mesh, not the arithmetic, sets them.

A spacing of 1.0 Å halves the error again but has 1.6 to 1.7 times the
points. Under TOWERS on the RTX 4090 it ran 7% slower on TIP3P 10k, 2%
slower on TIP3P 20k, 4% faster on JAC and 5% slower on Cellulose, and on
the list structures 0.2% faster to 5.6% slower. The FFT path decides much
of it: the axis FFT of D435 takes TIP3P 10k's 40³ and JAC's 64³, cuFFT
takes 48³, 54³ and 60³. The spacing is therefore not a default at the
same cost; whether it becomes one depends on the meshes that the axis FFT
covers (#991). $10^{-4}$ lowers the error of the
proteins a little further at the same grid, with twice the tail at the
cutoff; $5\times10^{-5}$ is the value of #927.

## The tail at the cutoff

The force of a pair of charges jumps at $r_c$ by
$f\,\lvert q_iq_j\rvert[\mathrm{erfc}(\beta r_c)/r_c^2 +
2\beta e^{-\beta^2r_c^2}/(\sqrt\pi r_c)]$ (D395). For the largest product
of the OPC dipeptide (1.845 e²) the new default gives 0.290 kJ/mol/nm at
9 Å (0.068 before) and 1.16 at 4.5 Å (0.271 before), where `mdir check`
now warns (the limit is 1 kJ/mol/nm); its advice names $10^{-5}$.

## Energy at constant volume

The larger tail adds to the work of the pairs that cross the cutoff. It
was not resolved in the drift of the energy at constant energy, unshifted,
1 ns at 2 fs from the input velocities, four runs each (the runs differ
through the order of the floating-point atomics), slope of the total
energy from 20 ps on, in kcal/mol/ns, mean ± standard error:

| Input | $10^{-5}$ | $5\times10^{-5}$ |
|---|---:|---:|
| TIP3P 10k | −0.31 ± 0.15 | +0.14 ± 0.31 |
| JAC | +1.69 ± 0.17 | +1.17 ± 1.39 |
| Factor IX | +1.83 ± 0.90 | +2.88 ± 1.99 |

The spread between runs is larger at $5\times10^{-5}$; a run at constant
energy that needs less takes `coulomb_modifier = "POTENTIAL_SHIFT"` or a
smaller tolerance, as before.

## Continuations

The keys of a control file, and so the fingerprint of D172, are the same
before and after a change of a default; the β of a run is not. A
checkpoint therefore records what the run resolved, in a group `resolved`
of its fingerprint: the Gaussian tolerance (where no β is given), β, the
grid, and the traversal, the same entries from `mdir run` and from Python
(`getResolvedFingerprint`). A continuation, with `--continue` or as
`checkpoint` of `[input]`, compares them with its own; a difference is a
note in the log and `effective.resolved_changes` in the manifest (name,
the checkpoint's value, the run's, and whether the checkpoint's was
inferred), not a refusal, as #980 rules for `[distribute]`. The run uses
its own values. A checkpoint written before the group took the defaults of
its build: where its fingerprint has neither `[pme] tolerance` nor
`[pme] beta`, its tolerance is $10^{-5}$ and β follows from the cutoff,
and the note says that they are inferred. The group is written, hashed
and read only where present, so that older checkpoints read and verify as
before (`test/Driver/checkpoint-resolved.test`).

# CPU baseline records (#552)

The `cpu-baseline-2026-10-22/` JSONL files retain the per-run rate and
conditions from `bench.py`. Round 0 is the warm-up; rounds 1–3 are reported.
The MDIR run logs identify the measured executable as commit
`89535e1f130b`. They have no `with uncommitted changes` suffix, which is
the build's cleanliness report. Relative to main `a8a5f7ff`, that commit
changes only this benchmark's `README.md` and `bench.py`; compiled MDIR
source is unchanged.
GROMACS 2026.3 was built in mixed precision, and pmemd 26 was the serial
double-precision program. The three systems were run sequentially on one
CPU node in separate 16-core Slurm allocations. Exact allocation details
remain in the original job records and are omitted here along with local
executable and scratch paths. `export_results.py` removes these fields;
`baseline_summary.py baseline-final jac tip3p-103k cellulose` regenerates
the paper's rate and ratio tables from the checked-in files. The pmemd
records use the all-steps `Per Step(ms)` value from `mdout`; the older
benchmark runner converted a two-decimal `ns/day` field, which was too
coarse for these systems. The precise timing is retained in the raw
`mdout` files, so no run needed to be repeated for the correction.
GROMACS records likewise use the `ms/step` column of each `md.log`
`Performance` line rather than a conversion from rounded `ns/day`.

All commands used the variants
`m1,m16,d1,d16,g1,g16,x1,x16,p1`, `--ensemble nve`, four rounds, and
`--tag baseline-final`. JAC used the default cutoff of 8 Å and grid
$64^3$. TIP3P used `--min-steps 100 --steps-scale 0.2`, the default cutoff
of 9 Å, and grid $160^3$. Cellulose used the same two step-count flags,
`--grid 288,128,128`, and the default cutoff of 8 Å. The JSONL rows state
the actual step count of each variant. See the parent README for the
engine-specific controls, PME tolerance conversion, actual NVE list policies, and
different timing windows.

The SHA-256 hashes below identify the prepared inputs. The JAC and
Cellulose Amber and GROMACS coordinates agree within `.gro` rounding;
TIP3P uses different starting states and its cross-engine ratios are
contextual.

| System | Input | SHA-256 |
|---|---|---|
| JAC | `sys.prmtop` | `bc6eb0f7773e5d93c0cdcce366539c0efd97570e21f5acf1248b1ed53dac25ba` |
| JAC | `sys.inpcrd` | `7f084ccce438dc61e49bbf6c3ee2d2a56cf1d057dcd64cace42ae7b2a0105cfe` |
| JAC | `gmx.top` | `6e2036cc14d1a1949a2adc48c60b229feb92639b7ef1d553c2e35059ae9dc7b6` |
| JAC | `eq.gro` | `dcc77cb391d22ca1f9eca764bc446b5a7c76293d292d2f5ac49487d8b77f0d5e` |
| TIP3P | `sys.prmtop` | `7ee4597bf4af42c2db0c824057215b588252619241002681aaf57f5507156265` |
| TIP3P | `sys.inpcrd` | `09ac59ac8c0146f833d1f37442553655860097a5d5770c2c03849eaff7c8e06f` |
| TIP3P | `gmx.top` | `3a997e373a105123d075c3e0b2fc071a1d2f83e63ae0770da1d12bfc03fb3801` |
| TIP3P | `eq.gro` | `dd94e3cc85b30445e1c836f6e6f87f095cfbfc7d8d711f39125e36e59f3aa54f` |
| Cellulose | `sys.prmtop` | `4dd27de54f64bdf0fc7b1ba18f42b986c8546d9491ccf1d7c3389a0ea1416b66` |
| Cellulose | `sys.inpcrd` | `dc022bac5b6c146adb53f1734987964fdd9a58a505f5153eeb9861b3fc3b4cd8` |
| Cellulose | `gmx.top` | `45790a11d2984f5a3109c1735fd1cb59a4ab7ce5fc3a8dfe02d4484b7d63d3d4` |
| Cellulose | `eq.gro` | `9f7609c519f4716498b86c848629bd71f535dc76f7c8bcb6994dbc2497df89e7` |

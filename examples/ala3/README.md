# Tri-alanine in OPC water

The zwitterionic peptide NALA-ALA-CALA with ff19SB in OPC water, 4,905
particles in a cubic box of 37.5 Å, made by `tleap -f ala3.leap`. The
four stages of a run, each beginning from the checkpoint of the one
before:

| Stage | Control file | What |
|---|---|---|
| 1 | `1-min.toml` | Steepest descent, 2000 steps, heavy atoms of the peptide restrained at 10 kcal/mol/Å² |
| 2 | `2-nvt.toml` | Velocities at 300 K, 50 ps at constant volume with stochastic velocity rescaling, the same restraints |
| 3 | `3-npt.toml` | 100 ps at 1 atm with stochastic cell rescaling, restraints at 1 kcal/mol/Å² |
| 4 | `4-md.toml` | 1 ns at 1 atm without restraints, a frame every ps |

All stages use PME with the Coulomb potential shifted to zero at the
cutoff, SETTLE on the waters, SHAKE and RATTLE on the bonds of hydrogen, a
step of 2 fs, and a GPU in mixed precision. The restraints hold the heavy
atoms of the peptide (`!:WAT & !@H*`) at their positions in `ala3.inpcrd`
in stages 1 to 3.

The same stages run in two ways, side by side:

| | Control files | Python |
|---|---|---|
| Files | `1-min.toml` … `4-md.toml` | `run.py` |
| Run | `examples/ala3/run.sh ala3-run build/bin/mdir` | `PYTHONPATH=build/python uv run examples/ala3/run.py` |
| Between stages | each stage reads the checkpoint of the one before (`[input] checkpoint`) | each stage starts from `simulation.state()` of the one before |
| Velocities | drawn at 300 K by stage 2 from `[dynamics] seed` (314159 by default) | `InitialState.draw_velocities(system, 300.0, seed)` with the same seed, `--seed` |
| Restraints | `[[restraints]]`, 10, 10, and 1 kcal/mol/Å² | `System.restraints`, the same constants in kJ/mol/nm², `System.restraint_reference` the inpcrd positions |
| Production output | the log, `md.dcd`, `md.h5` | `md.dat` (`EnergyReporter`), `md.dcd` (`TrajectoryReporter`), the density printed by a `CallbackReporter` |
| Host instead of GPU | `target = "cpu"` in `[execution]` | `--cpu` (and `--threads N`) |

The Python route needs the extension, built with `-DMDIR_ENABLE_PYTHON=ON`
into `build/python`, and the Python 3.12 that it was built for. The script
declares Python 3.12 and NumPy in its PEP 723 header, so
[uv](https://docs.astral.sh/uv/) runs it without a prepared environment:

```sh
PYTHONPATH=build/python uv run examples/ala3/run.py --out ala3-python
```

The outputs go to `ala3-python` (`--out`). `--steps-scale 0.01` multiplies
the steps and intervals of every stage for a quick check, and `--precision
Double` and `--deterministic` select the precision and the deterministic
mode. Without uv, any Python 3.12 with NumPy runs the script in the same way:
`PYTHONPATH=build/python python3.12 examples/ala3/run.py`.

`mdir template minimize`, `nvt`, `npt`, and `production` print these
stages with `system.prmtop` and `system.inpcrd` as placeholder paths.
Edit the paths, selections, run lengths, and execution for your system,
then run the saved files in order with `mdir run --continue FILE`.

`render.py` renders the trajectory as a movie with matplotlib and ffmpeg:

```sh
python examples/ala3/render.py ala3-run/ala3.prmtop ala3-run/md.dcd ala3.mp4
```

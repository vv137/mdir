# Tri-alanine in OPC water

The zwitterionic peptide NALA-ALA-CALA with ff19SB in OPC water, 4,905
particles in a cubic box of 37.5 Å, made by `tleap -f ala3.leap`. The
four stages of a run, each beginning from the checkpoint of the one
before:

| Stage | Control file | What |
|---|---|---|
| 1 | `1-min.toml` | Steepest descent, 2000 steps, heavy atoms of the peptide restrained at 10 kcal/mol/Å² |
| 2 | `2-nvt.toml` | Velocities at 300 K, 50 ps at constant volume with the Bussi thermostat, the same restraints |
| 3 | `3-npt.toml` | 100 ps at 1 bar with stochastic cell rescaling, restraints at 1 kcal/mol/Å² |
| 4 | `4-md.toml` | 1 ns at 1 bar without restraints, a frame every ps |

All stages use PME, SETTLE on the waters, SHAKE and RATTLE on the bonds of
hydrogen, a step of 2 fs, and a GPU in mixed precision; set `target =
"cpu"` in `[execution]` to run on the host.

```sh
examples/ala3/run.sh ala3-run build/bin/mdir
```

`render.py` renders the trajectory as a movie with matplotlib and ffmpeg:

```sh
python examples/ala3/render.py ala3-run/ala3.prmtop ala3-run/md.dcd ala3.mp4
```

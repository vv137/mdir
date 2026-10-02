# A POPC bilayer in TIP3P water

A bilayer of 126 POPC lipids, 63 in each leaflet, with Lipid21
[[Dickson2022]](../../docs/references.md#dickson2022) in 4,932 TIP3P
waters [[Jorgensen1983]](../../docs/references.md#jorgensen1983): 31,680
atoms in a cell of about 64 × 64 × 72 Å once equilibrated. It is built by
PACKMOL-Memgen [[SchottVerdugo2019]](../../docs/references.md#schottverdugo2019),
PACKMOL [[Martinez2009]](../../docs/references.md#martinez2009), and tleap
of AmberTools (`build.sh`), and runs at 303 K and 1 bar with the area of
the bilayer (x and y) and its height (z) coupled apart: semi-isotropic
stochastic cell rescaling (D119).

| Stage | File | What |
|---|---|---|
| 0 | `build.sh`, `popc.leap` | The input of PACKMOL from PACKMOL-Memgen, the packing (about ten minutes), and `popc.prmtop` and `popc.inpcrd` |
| 1 | `1-min.toml` | Steepest descent, 5000 steps from the close contacts that PACKMOL leaves, the phosphorus atoms restrained at 10 kcal/mol/Å² |
| 2 | `2-nvt.toml` | Velocities at 303 K, 100 ps of 1 fs at constant volume with stochastic velocity rescaling, the same restraints |
| 3 | `3-npt.toml` | 500 ps of 1 fs at 1 bar, semi-isotropic, restraints at 1 kcal/mol/Å²: the gaps of the packing close, and the cell loses about a sixth of its volume |
| 4 | `4-md.toml` | 30 ns at 1 bar without restraints, a frame every 20 ps; the area per lipid settles over the first 10 ns |

Each stage begins from the checkpoint of the one before. All use PME with
$\operatorname{erfc}(\beta r_c) = 10^{-5}$ and a cutoff of 10 Å with the
correction for the dispersion, SETTLE on the waters, SHAKE and RATTLE on
the bonds of hydrogen, and a GPU in mixed precision; stages 2 to 4 take
the groups of 16 with a dual list (outer 13 Å, inner 10.6 Å), and the
restraints are positional, on the phosphorus of each head group, with
their reference scaled with the cell axis by axis. The thermostat and the
barostat act every 25 steps, with time constants of 1 and 5 ps and a
compressibility of $4.5\times10^{-5}$ /bar on both.

```sh
examples/popc-bilayer/run.sh popc-run build/bin/mdir
```

On an RTX 3090, stages 2 and 3 run at about 230 ns/day at their step of
1 fs, and stage 4 at about 440 ns/day: 35 minutes for 10 ns. The log
has the volume in its last column, but not the area; the frames have the
cell, and `area.py` gives the area per lipid, the height, and the modulus
of the area from them:

```sh
python examples/popc-bilayer/area.py popc-run/md.dcd --skip 10000
```

The area of a bilayer this size is correlated over some nanoseconds, so 20
ns give only a handful of independent samples; the error that `area.py`
prints counts them. At the end of a run at semi-isotropic coupling the log
also gives the mean pressures that the barostat took, of x and y and of z,
and their difference, which is zero on average at zero tension.

The stages render as a movie with the cell, seen from the side, with the
area and the height of each frame (numpy, matplotlib, and ffmpeg):

```sh
python scripts/render/movie.py popc-run/popc.prmtop popc-run/npt.dcd \
    popc-run/md.dcd bilayer.mp4 --mode membrane --ps 5 20
```

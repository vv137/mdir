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
| 1 | `1-min.toml` | Steepest descent, 5000 steps from the close contacts that PACKMOL leaves |
| 2 | `2-relax.toml` | Velocities at 303 K, 100 ps of 1 fs at 1 bar: the gaps of the packing close, and the cell loses about a sixth of its volume |
| 3 | `3-equil.toml` | 10 ns at 1 bar, in which the area per lipid settles |
| 4 | `4-md.toml` | 20 ns at 1 bar, a frame every 20 ps |

Each stage begins from the checkpoint of the one before. All use PME with
$\operatorname{erfc}(\beta r_c) = 10^{-5}$ and a cutoff of 10 Å with the
correction for the dispersion, SETTLE on the waters, SHAKE and RATTLE on
the bonds of hydrogen, and a GPU in mixed precision; stages 2 to 4 take
the groups of 16 with a dual list (outer 13 Å, inner 10.6 Å). The
thermostat and the barostat act every 25 steps, with time constants of 1
and 5 ps and a compressibility of $4.5\times10^{-5}$ /bar on both.

```sh
examples/popc-bilayer/run.sh popc-run build/bin/mdir
```

On an RTX 3090, stage 2 runs at about 230 ns/day at its step of 1 fs,
and stages 3 and 4 at about 440 ns/day: 35 minutes for 10 ns. The log
has the volume in its last column, but not the area; the frames have the
cell, and `area.py` gives the area per lipid, the height, and the modulus
of the area from them:

```sh
python examples/popc-bilayer/area.py popc-run/md.dcd
```

The area of a bilayer this size is correlated over some nanoseconds, so 20
ns give only a handful of independent samples; the error that `area.py`
prints counts them. At the end of a run at semi-isotropic coupling the log
also gives the mean pressures that the barostat took, of x and y and of z,
and their difference, which is zero on average at zero tension.

The stages render as a movie with the cell, seen from the side, with the
area and the height of each frame (numpy, matplotlib, and ffmpeg):

```sh
python scripts/render/movie.py popc-run/popc.prmtop popc-run/relax.dcd \
    popc-run/equil.dcd bilayer.mp4 --mode membrane --ps 2 20
```

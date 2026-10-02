# Ubiquitin in OPC water

Human ubiquitin (PDB 1UBQ [[VijayKumar1987]](../../docs/references.md#vijaykumar1987),
76 residues, its crystal waters left out) with ff19SB
[[Tian2020]](../../docs/references.md#tian2020) in OPC water
[[Izadi2014]](../../docs/references.md#izadi2014), in a rectangular box
12 Å beyond the protein: 26,031 particles, 6200 waters of four sites
each. The protein is neutral, so no ions are added. `ubq.leap` builds
it with tleap of AmberTools; the topology and coordinates are made by the
tutorial rather than kept here.

| Stage | Control file | What |
|---|---|---|
| 0 | `ubq.leap` | `tleap -f ubq.leap`: `ubq.prmtop` and `ubq.inpcrd` |
| 1 | `1-min.toml` | Steepest descent, 2000 steps, the heavy atoms of the protein restrained at 10 kcal/mol/Å² |
| 2 | `2-nvt.toml` | Velocities at 300 K, 100 ps at constant volume with stochastic velocity rescaling, the same restraints |
| 3 | `3-npt.toml` | 200 ps at 1 bar with stochastic cell rescaling, restraints at 1 kcal/mol/Å² |
| 4 | `4-md.toml` | 10 ns at 1 bar without restraints, a frame every 10 ps |

Each stage begins from the checkpoint of the one before. All use PME with
$\operatorname{erfc}(\beta r_c) = 10^{-5}$ and a cutoff of 9 Å, SETTLE on
the waters, SHAKE and RATTLE on the bonds of hydrogen, a step of 2 fs, and
a GPU in mixed precision; stages 2 to 4 take the groups of 16 with a dual
list (outer 12 Å, inner 9.6 Å), which only a GPU has. To run on the host,
set `target = "CPU"` with `threads = <n>` in `[execution]`, remove
`neighbor_structure` and `pruned_distance`, and set `pairlist_distance =
10.0`.

```sh
examples/ubiquitin/run.sh ubiquitin-run build/bin/mdir
```

On a cluster, where a job ends before a long stage does, run a stage as
`mdir run --continue --max-walltime <time> 4-md.toml` in each job: it
continues the stage from its checkpoint, appends to its trajectory, stops
at a checkpoint before the wall time with the exit status 75, and exits
with 0 once the stage is complete (docs/driver-m0.md, Section 2.7).

On an RTX 3090, stage 2 runs at about 570 ns/day and stages 3 and 4,
with the barostat, at about 520: 28 minutes for the 10 ns of stage 4. The
log reports the rate over the second half of each stage. At constant
pressure the volume is the last column of the log and the conserved
energy the column before it; over stage 3 the volume falls from the
253,300 Å³ of the box of tleap to about 195,700 Å³ as the water fills its
corners.

The trajectory renders as a movie with the cell, the backbone as a tube,
and the side chains as balls and sticks (numpy, matplotlib, and ffmpeg):

```sh
python scripts/render/movie.py ubiquitin-run/ubq.prmtop ubiquitin-run/md.dcd \
    ubiquitin.mp4 --mode protein --ps 10
```

"""Writes a Lennard-Jones fluid whose build of groups has empty groups
(#413), in the directory given: short-chunk.pdb and run.toml.

129 argon atoms, a simple tetragonal lattice of 4 x 4 x 8 sites 3.8 Å apart
in x and y and 4.2 Å in z, and one atom more at the center of the first
cell. Particles take a column of about 64 particles' width: here the
cell is narrower than one (4 x 3.8 Å against (64 a^2 c)^(1/3) = 15.7 Å), so
the build has one column of 129 particles, cut into chunks of 64, 64, and
1. The chunk of one particle makes four groups (docs/groups-m1.md), the
last three of them empty: their first places hold no particle.

    short_chunk.py DIRECTORY
"""

import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
a, c, nx, nz = 3.8, 4.2, 4, 8
sites = [((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * c)
         for k in range(nz) for j in range(nx) for i in range(nx)]
sites.append((a, a, c))
lx, lz = nx * a, nz * c
lines = [f"CRYST1{lx:9.3f}{lx:9.3f}{lz:9.3f}  90.00  90.00  90.00 P 1           1"]
for n, (x, y, z) in enumerate(sites, 1):
    lines.append(f"HETATM{n:5d}  AR  AR  A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          AR")
lines.append("END")
(directory / "short-chunk.pdb").write_text("\n".join(lines) + "\n")
(directory / "run.toml").write_text(f"""\
[input]
coordinates = "short-chunk.pdb"

[output]
energy_interval = 10

[energy]
cutoff            = 6.0
pairlist_distance = 7.0
pruned_distance   = 6.5

[[energy.pair]]
name       = "lj"
expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
mixing     = "lorentz-berthelot"

[[energy.type]]
name    = "AR"
mass    = 39.95
epsilon = 0.238465
sigma   = 3.4

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = 20
seed       = 413

[ensemble]
ensemble    = "NVE"
temperature = 100.0

[boundary]
type = "PERIODIC"
box  = [{lx}, {lx}, {lz}]

[execution]
target    = "GPU"
precision = "MIXED"
neighbor_structure = "GROUPS"
""")

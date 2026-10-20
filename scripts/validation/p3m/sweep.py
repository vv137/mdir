#!/usr/bin/env python3
"""The relative force error Delta of the three influence functions of PME,
"SPME", "OPTIMAL", and "P3M", at fixed (K, P, beta), against the reference
of scripts/benchmarks/prolate-ewald/grid_sweep.py (D[p3m-influence], #468;
docs/pme-m1.md, Section 1.1).

    sweep.py make DIR SYSTEM TARGET SUITE REFERENCES
        SYSTEM      jac or cel
        TARGET      cpu-double, cpu-mixed, gpu-double, or gpu-mixed
        SUITE       the directory PME of the Amber benchmark suite
        REFERENCES  a directory of grid_sweep.py's `make` and `run` for the
                    system, whose `reference` and `uncharged` runs DIR links

then `grid_sweep.py run MDIR DIR [JOBS]` and `grid_sweep.py report MDIR DIR`.
beta is pmemd's at dsum_tol 1e-6 and r_c = 8 A (D134); the grids are the
suite's and coarser ones of the spacings below.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "benchmarks", "prolate-ewald"))
import grid_sweep  # noqa: E402

SYSTEMS = {
    "jac": ("JAC_production_NVE", [64, 64, 64]),
    "cel": ("Cellulose_production_NVE", [256, 128, 128]),
}
BETA = 0.39467022824439113
# (largest spacing in A, or None for the suite's grid; order)
GRIDS = {
    "jac": [(None, 4), (1.2, 4), (1.5, 4), (2.0, 4), (2.0, 6), (2.5, 6)],
    "cel": [(None, 4), (1.5, 4), (2.0, 4), (2.0, 6)],
}


def main():
    if len(sys.argv) != 7 or sys.argv[1] != "make":
        sys.exit(__doc__)
    d, system, target, suite, references = sys.argv[2:7]
    name, suite_grid = SYSTEMS[system]
    top, crd = f"{suite}/{name}/prmtop", f"{suite}/{name}/inpcrd"
    edges = grid_sweep.box(crd)
    os.makedirs(d, exist_ok=True)
    for ref in ("reference", "uncharged"):
        if not os.path.exists(f"{d}/{ref}"):
            os.symlink(os.path.abspath(f"{references}/{ref}"), f"{d}/{ref}")
    device, precision = target.split("-")
    for spacing, order in GRIDS[system]:
        grid = suite_grid if spacing is None else [
            grid_sweep.smooth(math.ceil(e / spacing - 1e-9)) for e in edges]
        for influence in ("SPME", "OPTIMAL", "P3M"):
            case = f"{influence}-o{order}-{grid[0]}x{grid[1]}x{grid[2]}"
            pme = [f"beta = {BETA}", f"order = {order}", f'influence = "{influence}"',
                   f"grid = [{grid[0]}, {grid[1]}, {grid[2]}]"]
            text = grid_sweep.control(top, crd, 8.0, pme)
            text = text.replace('target    = "CPU"', f'target    = "{device.upper()}"')
            text = text.replace('precision = "DOUBLE"', f'precision = "{precision.upper()}"')
            os.makedirs(f"{d}/{case}", exist_ok=True)
            open(f"{d}/{case}/run.toml", "w").write(text)


if __name__ == "__main__":
    main()

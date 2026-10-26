"""GPUMD on the frames of a reference, and its comparison with calorine, for
docs/nep.md (D356, #583).

Usage:
  gpumd.py setup REF.npz MODEL DIR     one directory DIR/frameN per frame,
                                       with model.xyz and run.in
  gpumd.py compare REF.npz DIR         GPUMD's dump.xyz of each frame
                                       against REF.npz (calorine.py)

Each run.in evaluates the model once (time step 0) and writes the energy,
the forces, and the virial in double precision. Run GPUMD in each
DIR/frameN on a GPU (docs/nep.md, Section 7.1).
"""
import os
import re
import sys

import numpy as np


def setup(ref, model, outdir):
    model = os.path.abspath(model)
    for f in range(int(ref["nframes"])):
        d = os.path.join(outdir, f"frame{f}")
        os.makedirs(d, exist_ok=True)
        pos, cell, sym = ref[f"pos{f}"], ref[f"cell{f}"], ref[f"sym{f}"]
        with open(os.path.join(d, "model.xyz"), "w") as fh:
            lattice = " ".join(f"{x:.10f}" for x in cell.reshape(-1))
            fh.write(f'{len(sym)}\npbc="T T T" Lattice="{lattice}" Properties=species:S:1:pos:R:3\n')
            for s, p in zip(sym, pos):
                fh.write(f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")
        with open(os.path.join(d, "run.in"), "w") as fh:
            fh.write(f"potential {model}\nvelocity 1\nensemble nve\ntime_step 0\n"
                     "dump_xyz 1 dump.xyz precision double potential force virial\nrun 1\n")


def compare(ref, outdir):
    rows = []
    for f in range(int(ref["nframes"])):
        lines = open(os.path.join(outdir, f"frame{f}", "dump.xyz")).read().splitlines()
        n, head = int(lines[0]), lines[1]
        props = re.search(r"Properties=(\S+)", head).group(1).split(":")
        col, cols = 0, {}
        for name, _, width in zip(props[3::3], props[4::3], props[5::3]):
            cols[name] = (col, col + int(width)); col += int(width)
        data = np.array([l.split()[1:] for l in lines[2:2 + n]], float)
        pos, F = data[:, slice(*cols["pos"])], data[:, slice(*cols["forces"])]
        # GPUMD wraps the positions into the cell.
        frac = (pos - ref[f"pos{f}"]) @ np.linalg.inv(ref[f"cell{f}"])
        if np.abs(frac - np.round(frac)).max() > 1e-9:
            raise SystemExit(f"frame {f}: the atoms are not in the order of the reference")
        E = float(re.search(r"energy=(\S+)", head).group(1))
        W = np.array(re.search(r'virial="([^"]+)"', head).group(1).split(), float).reshape(3, 3)
        Ec, Fc, Wc = ref[f"energy{f}"], ref[f"forces{f}"], ref[f"virial{f}"]
        rows.append({"dE/atom": abs(E - Ec) / n, "dF": np.abs(F - Fc).max(),
                     "rms_dF": np.sqrt(((F - Fc) ** 2).mean()), "dW/atom": np.abs(W - Wc).max() / n})
        print(f"frame {f} ({n} atoms): " + " ".join(f"{k}={v:.2g}" for k, v in rows[-1].items()))
    print("max over frames: " + " ".join(f"{k}={max(r[k] for r in rows):.2g}" for k in rows[0]))


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup(np.load(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "compare":
        compare(np.load(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit(__doc__)

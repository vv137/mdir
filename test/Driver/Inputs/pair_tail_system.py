"""A small mixture of two Lennard-Jones species A and B, single-atom
molecules of GROMACS, for the tail of pair terms beyond the cutoff
(D[pair-dispersion-correction]). Writes

  plain.top   A and B with the rule of Lorentz and Berthelot
  fixed.top   the same with the A-B pair set apart in [ nonbond_params ]
              (sigma' and epsilon')
  system.gro  the coordinates: a jittered cubic lattice, fixed by a seed

    pair_tail_system.py directory count-A count-B edge-Å sigma'-Å epsilon'-kcal/mol

A and B have sigma = 3.4 Å and epsilon = 0.24 kcal/mol, mass 40."""

import random
import sys

KCAL = 4.184


def main():
    out, na, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    edge, sig, eps = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    head = ["[ defaults ]", "  1 2 no 1.0 1.0", "", "[ atomtypes ]"]
    for name in ("A", "B"):
        head.append(f"  {name}T 0 40.0 0.0 A 0.340000 {0.24 * KCAL:.6f}")
    body = []
    for name in ("A", "B"):
        body += ["", "[ moleculetype ]", f"  {name} 1", "", "[ atoms ]",
                 f"  1 {name}T 1 {name} {name} 1 0.0 40.0"]
    body += ["", "[ system ]", "mixture", "", "[ molecules ]", f"A {na}",
             f"B {nb}", ""]
    nbfix = ["", "[ nonbond_params ]",
             f"  AT BT 1 {sig / 10:.6f} {eps * KCAL:.6f}"]
    open(f"{out}/plain.top", "w").write("\n".join(head + body))
    open(f"{out}/fixed.top", "w").write("\n".join(head + nbfix + body))

    rng = random.Random(7)
    per = 1
    while per ** 3 < na + nb:
        per += 1
    spacing = edge / per
    sites = [(i, j, k) for i in range(per) for j in range(per)
             for k in range(per)]
    rng.shuffle(sites)
    lines = []
    for index in range(na + nb):
        name = "A" if index < na else "B"
        x, y, z = ((s + 0.5 + rng.uniform(-0.1, 0.1)) * spacing / 10
                   for s in sites[index])
        lines.append(f"{(index + 1) % 100000:5d}{name:<5s}{name:>5s}"
                     f"{(index + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
    gro = ["mixture", str(na + nb)] + lines + [
        f"{edge / 10:10.5f}{edge / 10:10.5f}{edge / 10:10.5f}"]
    open(f"{out}/system.gro", "w").write("\n".join(gro) + "\n")


if __name__ == "__main__":
    main()

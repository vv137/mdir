"""dH/dlambda of the Coulomb at the first row of a [free_energy] file of
shifted-derivatives.test, against a sum over the pairs of the coordinates
(D210). The first particle is decoupled; with a Coulomb
cutoff its pairs within r_c contribute (1 - lambda) f q_i q_j (1/r - 1/r_c),
the cutoff shifted to 0, so

    dH/dlambda_C = -sum f q_i q_j (1/r - 1/r_c),

where the cut potential would give -sum f q_i q_j / r. Prints both, the
difference from the file, and whether it is within the tolerance.

    check_shifted.py system.gro file.dhdl charge-e tolerance-kcal/mol"""

import math
import sys

COULOMB = 332.06371329919205  # kcal Å/(mol e²), CODATA 2018
CUTOFF = 12.0


def main():
    gro, dhdl, charge, tolerance = sys.argv[1], sys.argv[2], float(
        sys.argv[3]), float(sys.argv[4])
    lines = open(gro).read().splitlines()
    count = int(lines[1])
    atoms = lines[2:2 + count]
    x = [[float(line[20 + 8 * a:28 + 8 * a]) * 10 for a in range(3)]
         for line in atoms]
    q = [charge if line[5:10].strip() == "A" else -charge for line in atoms]
    box = [float(v) * 10 for v in lines[2 + count].split()[:3]]
    shifted = cut = 0.0
    for j in range(1, count):
        d = [x[0][a] - x[j][a] for a in range(3)]
        d = [d[a] - box[a] * round(d[a] / box[a]) for a in range(3)]
        r = math.sqrt(sum(c * c for c in d))
        if r < CUTOFF:
            shifted -= COULOMB * q[0] * q[j] * (1 / r - 1 / CUTOFF)
            cut -= COULOMB * q[0] * q[j] / r
    rows = [line.split() for line in open(dhdl) if not line.startswith("#")]
    names = open(dhdl).readline().lstrip("#").split()
    row = dict(zip(names, map(float, rows[0])))
    ok = True
    # The states are lambda_C = 0, 0.5, 1 at the run's 0.5, with the same
    # lambda_V; the Coulomb is linear in lambda_C, so its share of
    # U_k - U_run is (lambda_k - 0.5) dH/dlambda_C, and that of the
    # Lennard-Jones is checked against "POTENTIAL_SHIFT" by the test.
    difference = row["dHdl.coulomb"] - shifted
    good = abs(difference) <= tolerance
    ok &= good
    print(f"dHdl.coulomb: {row['dHdl.coulomb']:.6f} shifted sum {shifted:.6f} "
          f"(cut {cut:.6f}) difference {difference:.1e} "
          + ("ok" if good else "FAILED"))
    print("shifted Coulomb: " + ("ok" if ok else "FAILED"))


if __name__ == "__main__":
    main()

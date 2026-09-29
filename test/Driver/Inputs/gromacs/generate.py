"""Writes system.gro: four molecules of propane and sixty of water, placed
on a grid in a cubic box, each turned by a rotation from a linear
congruential generator, with six decimals.

Usage: python3 generate.py
"""
import math

EDGE = 2.0
state = 12345


def draw():
    global state
    state = (state * 1103515245 + 12345) % 2147483648
    return state / 2147483648.0


def rotation():
    """A rotation from three angles."""
    a, b, c = (2.0 * math.pi * draw() for _ in range(3))
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b),
                              math.sin(b), math.cos(c), math.sin(c))
    return [[ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
            [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
            [-sb, cb * sc, cb * cc]]


def turn(matrix, point):
    return [sum(matrix[r][k] * point[k] for k in range(3)) for r in range(3)]


def propane():
    """Positions in nm of C1, H11, H12, H13, C2, H21, H22, C3, H31..H33."""
    cc, ch = 0.1526, 0.109
    half = math.radians(109.5) / 2.0
    c1 = [-cc * math.sin(half), -cc * math.cos(half), 0.0]
    c2 = [0.0, 0.0, 0.0]
    c3 = [cc * math.sin(half), -cc * math.cos(half), 0.0]
    atoms = [("C1", c1)]
    for k, name in enumerate(["H11", "H12", "H13"]):
        phi = 2.0 * math.pi * k / 3.0
        atoms.append((name, [c1[0] - ch * 0.33, c1[1] - ch * 0.94 * math.cos(phi) * 0.5,
                             c1[2] + ch * 0.94 * math.sin(phi)]))
    atoms.append(("C2", c2))
    atoms.append(("H21", [0.0, ch * 0.58, ch * 0.82]))
    atoms.append(("H22", [0.0, ch * 0.58, -ch * 0.82]))
    atoms.append(("C3", c3))
    for k, name in enumerate(["H31", "H32", "H33"]):
        phi = 2.0 * math.pi * k / 3.0
        atoms.append((name, [c3[0] + ch * 0.33, c3[1] - ch * 0.94 * math.cos(phi) * 0.5,
                             c3[2] + ch * 0.94 * math.sin(phi)]))
    return "PRO", atoms


def water():
    theta = math.radians(104.52) / 2.0
    return "SOL", [("OW", [0.0, 0.0, 0.0]),
                   ("HW1", [0.09572 * math.sin(theta), 0.09572 * math.cos(theta), 0.0]),
                   ("HW2", [-0.09572 * math.sin(theta), 0.09572 * math.cos(theta), 0.0])]


def main():
    molecules = [propane() for _ in range(4)] + [water() for _ in range(60)]
    per_side = 5
    spacing = EDGE / per_side
    lines = []
    atom = 0
    for index, (residue, atoms) in enumerate(molecules):
        cell = [index % per_side, (index // per_side) % per_side,
                index // (per_side * per_side)]
        centre = [(c + 0.5) * spacing for c in cell]
        matrix = rotation()
        for name, point in atoms:
            atom += 1
            x = [centre[k] + v for k, v in enumerate(turn(matrix, point))]
            lines.append("%5d%-5s%5s%5d%12.6f%12.6f%12.6f" % (
                index + 1, residue, name, atom, x[0], x[1], x[2]))
    with open("system.gro", "w") as out:
        out.write("Propane and water\n%5d\n" % atom)
        out.write("\n".join(lines) + "\n")
        out.write("%10.5f%10.5f%10.5f\n" % (EDGE, EDGE, EDGE))


if __name__ == "__main__":
    main()

"""constrain_opc.py IN.inpcrd OUT.inpcrd: the waters of four sites of
Inputs/opc/three.inpcrd (O, H1, H2, EP each) put on their constraints, the
oxygen, the bisector of the hydrogens, and the plane of the molecule kept:
O-H 0.872433 Ang, H-H 1.371205 Ang (OPC, opc.prmtop), and the extra point
at the place of its definition, x_O + a (u + v) / |u + v|, which on the
constrained frame is that of the weights of D403, w = 0.14772060947071208
of each hydrogen (rigid-units-route.test)."""
import math
import sys

src, out = sys.argv[1:3]
rows = open(src).read().splitlines()
count = int(rows[1].split()[0])
flat = [float(rows[2 + k // 6][12 * (k % 6):12 * (k % 6) + 12]) for k in range(3 * count)]
x = [flat[3 * i:3 * i + 3] for i in range(count)]
OH, HH, W = 0.872433, 1.371205, 0.14772060947071208
half = math.asin(HH / (2 * OH))


def sub(a, b):
    return [p - q for p, q in zip(a, b)]


def unit(a):
    n = math.sqrt(sum(p * p for p in a))
    return [p / n for p in a]


for o in range(0, count, 4):
    u, v = sub(x[o + 1], x[o]), sub(x[o + 2], x[o])
    b = unit([p + q for p, q in zip(unit(u), unit(v))])
    ub = sum(p * q for p, q in zip(u, b))
    p_ = unit([p - ub * q for p, q in zip(u, b)])
    c, s_ = math.cos(half), math.sin(half)
    x[o + 1] = [xo + OH * (c * bb + s_ * pp) for xo, bb, pp in zip(x[o], b, p_)]
    x[o + 2] = [xo + OH * (c * bb - s_ * pp) for xo, bb, pp in zip(x[o], b, p_)]
    x[o + 3] = [xo + W * ((h1 - xo) + (h2 - xo)) for xo, h1, h2 in zip(x[o], x[o + 1], x[o + 2])]
flat = [v for p in x for v in p]
body = [''.join(f'{v:12.7f}' for v in flat[k:k + 6]) for k in range(0, len(flat), 6)]
open(out, 'w').write('\n'.join([rows[0], rows[1], *body, rows[-1]]) + '\n')

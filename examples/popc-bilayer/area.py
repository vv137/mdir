#!/usr/bin/env python3
"""The area per lipid, the height of the cell, and the modulus of the area
of a bilayer from the cells of the frames of a DCD trajectory.

    area.py md.dcd [--lipids 126] [--ps 20] [--skip 1000] [--temperature 303]

The area per lipid is L_x L_y over the lipids of a leaflet; its error is
that of the mean of samples spaced by their statistical inefficiency, the
correlation of the area lasting nanoseconds. The modulus of the area is
K_A = k_B T <A> / var(A). Needs numpy.
"""
import argparse
import math
import struct

import numpy as np


def cells(path):
    """The edges of the cell of each frame, in Å."""
    data = open(path, "rb").read()
    blocks, at = [], 0
    while at + 4 <= len(data):
        (length,) = struct.unpack_from("<i", data, at)
        if at + 8 + length > len(data):
            break  # a frame being written
        blocks.append(at + 4)
        at += 8 + length
    frames = blocks[3:]
    out = []
    for start in range(0, len(frames) - 3, 4):
        v = struct.unpack_from("<6d", data, frames[start])
        out.append([v[0], v[2], v[5]])
    return np.array(out)


def inefficiency(x):
    """1 + 2 the sum of the autocorrelation until it first falls below 0."""
    x = np.asarray(x) - np.mean(x)
    n, var = len(x), np.dot(x, x) / len(x)
    g = 1.0
    for t in range(1, n // 2):
        c = np.dot(x[:-t], x[t:]) / (n - t) / var
        if c <= 0:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(g, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory")
    parser.add_argument("--lipids", type=int, default=126)
    parser.add_argument("--ps", type=float, default=20.0,
                        help="ps between frames")
    parser.add_argument("--skip", type=float, default=1000.0,
                        help="ps left out at the start")
    parser.add_argument("--temperature", type=float, default=303.0)
    args = parser.parse_args()
    c = cells(args.trajectory)
    t = np.arange(1, len(c) + 1) * args.ps
    c = c[t > args.skip]
    area = c[:, 0] * c[:, 1]
    apl = area / (args.lipids / 2)
    g = inefficiency(apl)
    error = apl.std(ddof=1) * math.sqrt(g / len(apl))
    kT = 1.380649e-23 * args.temperature
    modulus = kT * area.mean() / area.var(ddof=1) * 1e23  # mN/m
    print(f"{len(c)} frames after {args.skip:g} ps, "
          f"{len(c) / g:.0f} independent")
    print(f"area per lipid {apl.mean():.2f} ± {error:.2f} Å² "
          f"(sd {apl.std(ddof=1):.2f})")
    print(f"height {c[:, 2].mean():.2f} Å, volume "
          f"{np.prod(c, axis=1).mean() / 1000:.2f} nm³")
    print(f"modulus of the area {modulus:.0f} mN/m")


if __name__ == "__main__":
    main()

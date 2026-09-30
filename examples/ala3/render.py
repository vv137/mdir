"""Renders the trajectory of the example as a movie: the peptide as balls
and sticks, and the waters within 5 Å of it, seen by a camera that turns
slowly about the peptide.

    render.py ala3.prmtop md.dcd ala3.mp4 [--fps 25] [--size 720]

Needs numpy, matplotlib, and ffmpeg on PATH (for instance a conda
environment with the packages ffmpeg, numpy, and matplotlib)."""

import argparse
import os
import shutil
import struct
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {1: "#f2f2f2", 6: "#5a5a5a", 7: "#3050f8", 8: "#e02020"}
RADII = {1: 0.22, 6: 0.38, 7: 0.38, 8: 0.38}


def read_prmtop(path):
    sections, flag, width = {}, None, None
    for line in open(path):
        if line.startswith("%FLAG"):
            flag = line.split()[1]
            sections[flag] = []
        elif line.startswith("%FORMAT"):
            spec = line[line.index("(") + 1 : line.index(")")]
            letter = next(c for c in spec if c.isalpha())
            width = int(spec.split(letter)[1].split(".")[0])
        elif line.startswith("%") or flag is None:
            continue
        else:
            text = line.rstrip("\n")
            sections[flag] += [text[i : i + width].strip()
                               for i in range(0, len(text), width)
                               if text[i : i + width].strip()]
    return sections


def read_dcd(path):
    """Yields the cell edges and the positions (Å) of each frame."""
    data = open(path, "rb").read()
    blocks, at = [], 0
    while at < len(data):
        (length,) = struct.unpack_from("<i", data, at)
        blocks.append(data[at + 4 : at + 4 + length])
        at += 8 + length
    has_cell = struct.unpack_from("<i", blocks[0], 4 + 10 * 4)[0] != 0
    count = struct.unpack_from("<i", blocks[2])[0]
    step = 4 if has_cell else 3
    frames = blocks[3:]
    for start in range(0, len(frames) - step + 1, step):
        cell = None
        if has_cell:
            values = struct.unpack("<6d", frames[start])
            cell = np.array([values[0], values[2], values[5]])
        xyz = [np.frombuffer(b, dtype="<f4", count=count)
               for b in frames[start + (1 if has_cell else 0) : start + step]]
        yield cell, np.stack(xyz, axis=1).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prmtop")
    parser.add_argument("dcd")
    parser.add_argument("movie")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--timestep", type=float, default=1.0,
                        help="ps between frames, for the label")
    args = parser.parse_args()

    top = read_prmtop(args.prmtop)
    count = int(top["POINTERS"][0])
    numbers = np.array([int(v) for v in top["ATOMIC_NUMBER"]])
    labels = top["RESIDUE_LABEL"]
    pointers = [int(v) - 1 for v in top["RESIDUE_POINTER"]] + [count]
    residue = np.empty(count, dtype=int)
    for r in range(len(labels)):
        residue[pointers[r] : pointers[r + 1]] = r
    water = np.array([labels[r] == "WAT" for r in residue])
    peptide = np.flatnonzero(~water)
    oxygens = np.flatnonzero(water & (numbers == 8))
    bonds = []
    for key in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        v = [int(x) for x in top[key]]
        bonds += [(v[n] // 3, v[n + 1] // 3) for n in range(0, len(v), 3)]
    bonds = [(i, j) for i, j in bonds if not water[i] and not water[j]]

    work = tempfile.mkdtemp(prefix="ala3-frames-")
    try:
        size = args.size / 100
        figure = plt.figure(figsize=(size, size), dpi=100)
        axes = figure.add_axes([0, 0, 1, 1])
        for index, (cell, x) in enumerate(read_dcd(args.dcd)):
            # The peptide whole: each atom in the image nearest the first.
            p = x[peptide]
            if cell is not None:
                p -= cell * np.round((p - p[0]) / cell)
            center = p.mean(axis=0)
            x = x.copy()
            x[peptide] = p
            # The waters in the image nearest the peptide, within 5 Å.
            w = x[oxygens]
            if cell is not None:
                w -= cell * np.round((w - center) / cell)
            distance = np.min(np.linalg.norm(
                w[:, None, :] - p[None, :, :], axis=2), axis=1)
            w = w[distance < 5.0]
            # A camera that turns about the vertical, once in 40 s.
            angle = 2 * np.pi * index / (40 * args.fps)
            c, s = np.cos(angle), np.sin(angle)
            turn = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            P = (p - center) @ turn.T
            W = (w - center) @ turn.T
            position = {a: k for k, a in enumerate(peptide)}

            axes.clear()
            axes.set_facecolor("black")
            axes.set_xlim(-9, 9)
            axes.set_ylim(-9, 9)
            axes.set_aspect("equal")
            axes.axis("off")
            axes.scatter(W[:, 0], W[:, 1], s=90, c="#ff6060", alpha=0.35,
                         linewidths=0, zorder=0)
            depth = P[:, 2]
            order = np.argsort(depth)
            for i, j in bonds:
                a, b = P[position[i]], P[position[j]]
                z = 1 + np.searchsorted(np.sort(depth), (a[2] + b[2]) / 2)
                axes.plot([a[0], b[0]], [a[1], b[1]], color="#b0b0b0",
                          linewidth=3, zorder=z - 0.5, solid_capstyle="round")
            for rank, k in enumerate(order):
                z = numbers[peptide[k]]
                axes.add_patch(plt.Circle(
                    (P[k, 0], P[k, 1]), RADII.get(z, 0.4),
                    color=COLORS.get(z, "#ff80ff"), zorder=rank + 1,
                    ec="black", lw=0.5))
            axes.text(-8.6, 8.4, f"{index * args.timestep / 1000:.3f} ns",
                      color="white", fontsize=14, va="top",
                      zorder=len(order) + 2)
            figure.savefig(os.path.join(work, f"{index:05d}.png"),
                           facecolor="black")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate",
             str(args.fps), "-i", os.path.join(work, "%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             args.movie], check=True)
        print(f"wrote {index + 1} frames to {args.movie}")
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()

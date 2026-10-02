#!/usr/bin/env python3
"""Renders a trajectory of MDIR as a movie, with the edges of the cell:
a lipid bilayer seen from the side, or a protein in water, with a camera
that turns slowly about the normal of the bilayer (z) or the vertical.

    scripts/render/movie.py STRUCTURE.gro TRAJECTORY.dcd... MOVIE.mp4
        [--mode membrane|protein] [--ps 10...] [--fps 25] [--stride 1]
        [--width 1280] [--height 720] [--jobs 8]

STRUCTURE gives the names of the atoms and residues (a .gro in the order
of the topology, as `gmx editconf` or ParmEd writes it); each TRAJECTORY
is a DCD with the cell in each frame, as `[output] trajectory` writes it,
and they are played one after the other; PS is the time between the
frames of each (one value for all, or one for each). Bonds are found once from the distances of
the first frame; each molecule is made whole along them in every frame and
put into the cell by its center, so that what crosses a face of the cell
stays whole and sticks out of it. Hydrogens and virtual sites are not
drawn; waters are their oxygens.

Needs numpy, matplotlib, and ffmpeg on PATH (the environment ~/opt/render
has them).
"""
import argparse
import multiprocessing
import os
import shutil
import struct
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

WATER = {"SOL", "WAT", "HOH", "TIP3", "OPC"}
ELEMENT_COLORS = {"C": "#909090", "N": "#3050f8", "O": "#e02020",
                  "P": "#ff9a00", "S": "#e0c020"}


def read_gro(path):
    lines = open(path).read().splitlines()
    count = int(lines[1])
    residues, residue_names, names = [], [], []
    for line in lines[2:2 + count]:
        residues.append(int(line[0:5]))
        residue_names.append(line[5:10].strip())
        names.append(line[10:15].strip())
    # Residue numbers wrap at 100000 and repeat across molecules; a new
    # residue begins wherever the number or the name changes.
    index, previous = [], None
    for number, name in zip(residues, residue_names):
        if (number, name) != previous:
            index.append(len(index) and index[-1] + 1)
            previous = (number, name)
        else:
            index.append(index[-1])
    return np.array(index), np.array(residue_names), np.array(names)


def element(name):
    letter = name.lstrip("0123456789")[:1].upper()
    return letter if letter in "CHNOPS" else "X"


def read_dcd(path):
    """The cells (Å, edges a, b, c) and the positions (Å) of the frames."""
    data = open(path, "rb").read()
    blocks, at = [], 0
    while at < len(data):
        (length,) = struct.unpack_from("<i", data, at)
        if at + 8 + length > len(data):
            break  # a frame being written
        blocks.append((at + 4, length))
        at += 8 + length
    header = data[blocks[0][0]:blocks[0][0] + blocks[0][1]]
    has_cell = struct.unpack_from("<i", header, 4 + 10 * 4)[0] != 0
    if not has_cell:
        raise SystemExit("the trajectory has no cell")
    count = struct.unpack_from("<i", data, blocks[2][0])[0]
    frames = blocks[3:]
    cells, positions = [], []
    for start in range(0, len(frames) - 3, 4):
        values = struct.unpack_from("<6d", data, frames[start][0])
        cells.append([values[0], values[2], values[5]])
        xyz = [np.frombuffer(data, dtype="<f4", count=count, offset=o)
               for o, _ in frames[start + 1:start + 4]]
        positions.append(np.stack(xyz, axis=1))
    return np.array(cells), np.array(positions, dtype=np.float64)


def find_bonds(x, cell, residue, elements):
    """Pairs within a residue or of consecutive residues closer than a
    bond: 1.25 Å with a hydrogen, 1.95 Å between heavy atoms."""
    bonds = []
    starts = np.flatnonzero(np.diff(residue, prepend=-1))
    ends = np.append(starts[1:], len(residue))
    for r in range(len(starts)):
        a = np.arange(starts[r], ends[min(r + 1, len(starts) - 1)])
        d = x[a][:, None, :] - x[a][None, :, :]
        d -= cell * np.round(d / cell)
        d = np.linalg.norm(d, axis=2)
        hydrogen = elements[a] == "H"
        limit = np.where(hydrogen[:, None] | hydrogen[None, :], 1.25, 1.95)
        site = elements[a] == "X"
        ok = (d < limit) & ~site[:, None] & ~site[None, :]
        i, j = np.nonzero(np.triu(ok, 1))
        bonds += [(a[p], a[q]) for p, q in zip(i, j)
                  if residue[a[p]] == r or residue[a[q]] == r]
    return np.unique(np.array(bonds, dtype=np.int64).reshape(-1, 2), axis=0)


def molecules_of(count, bonds, residue):
    """Molecules from the bonds (virtual sites join their residue), and the
    levels of a breadth-first tree of each, to make them whole."""
    parent = np.arange(count)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in bonds:
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    # A particle without bonds (a virtual site) joins the first atom of its
    # residue.
    bonded = np.zeros(count, bool)
    bonded[bonds.ravel()] = True
    first = {}
    for i in range(count):
        first.setdefault(residue[i], i)
    extra = [(first[residue[i]], i) for i in range(count)
             if not bonded[i] and first[residue[i]] != i]
    edges = np.concatenate([bonds, np.array(extra, np.int64).reshape(-1, 2)])
    for i, j in extra:
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    molecule = np.array([root(i) for i in range(count)])
    _, molecule = np.unique(molecule, return_inverse=True)

    neighbors = [[] for _ in range(count)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    seen = np.zeros(count, bool)
    levels = []
    for start in range(count):
        if seen[start]:
            continue
        seen[start] = True
        frontier, depth = [start], 0
        while frontier:
            following = []
            for i in frontier:
                for j in neighbors[i]:
                    if not seen[j]:
                        seen[j] = True
                        following.append((i, j))
            if following:
                while len(levels) <= depth:
                    levels.append([])
                levels[depth] += following
            frontier = [j for _, j in following]
            depth += 1
    levels = [np.array(level, np.int64) for level in levels]
    return molecule, levels


def whole(x, cell, levels):
    x = x.copy()
    for level in levels:
        d = x[level[:, 1]] - x[level[:, 0]]
        x[level[:, 1]] = x[level[:, 0]] + d - cell * np.round(d / cell)
    return x


def catmull_rom(points, samples):
    """A smooth curve through the points, `samples` a segment."""
    p = np.concatenate([points[:1], points, points[-1:]])
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    out = []
    for k in range(1, len(p) - 2):
        a, b, c, d = p[k - 1], p[k], p[k + 1], p[k + 2]
        out.append(0.5 * (2 * b + (c - a) * t
                          + (2 * a - 5 * b + 4 * c - d) * t ** 2
                          + (3 * b - a - 3 * c + d) * t ** 3))
    out.append(points[-1:])
    return np.concatenate(out)


def circular_mean(values, length):
    angle = 2 * np.pi * values / length
    return np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) \
        * length / (2 * np.pi)


class Scene:
    def __init__(self, args, residue, residue_names, names, cells, positions):
        self.args = args
        self.cells, self.positions = cells, positions
        self.elements = np.array([element(n) for n in names])
        self.water = np.isin(residue_names, list(WATER))
        bonds = find_bonds(positions[0], cells[0], residue, self.elements)
        self.molecule, self.levels = molecules_of(len(names), bonds, residue)
        self.molecules = self.molecule.max() + 1
        sizes = np.bincount(self.molecule)
        heavy = (self.elements != "H") & (self.elements != "X")
        self.oxygens = np.flatnonzero(self.water & (self.elements == "O"))
        solute = heavy & ~self.water
        if args.mode == "protein":
            self.focus = np.flatnonzero(self.molecule == np.argmax(
                np.where(np.bincount(self.molecule, weights=~self.water),
                         sizes, 0)))
            self.focus = self.focus[heavy[self.focus]]
        else:
            self.focus = np.flatnonzero(solute)
        self.atoms = np.flatnonzero(solute)
        self.colors = np.array([ELEMENT_COLORS.get(e, "#ff80ff")
                                for e in self.elements[self.atoms]])
        self.sizes = np.full(len(self.atoms), 28.0)
        if args.mode == "membrane":
            # The chains of the lipids by residue (Lipid21 splits a lipid
            # into its head and its two chains), the head by element.
            chain = {"PA": "#d8d8d8", "OL": "#e8d070", "MY": "#c0d8ff",
                     "ST": "#d0d0d0", "AR": "#f0b0b0", "DHA": "#b0f0b0"}
            for k, a in enumerate(self.atoms):
                if residue_names[a] in chain and self.elements[a] == "C":
                    self.colors[k] = chain[residue_names[a]]
                if self.elements[a] in "PN":
                    self.sizes[k] = 90.0
        else:
            # The backbone as a tube through the alpha carbons, the side
            # chains as balls and sticks from the alpha carbon out.
            ca = (names == "CA") & ~self.water
            self.trace = np.flatnonzero(ca)
            backbone = np.isin(names, ["N", "C", "O", "OXT", "CA"])
            side = solute & ~backbone
            self.atoms = np.flatnonzero(side)
            self.colors = np.array([ELEMENT_COLORS.get(e, "#ff80ff")
                                    for e in self.elements[self.atoms]])
            self.sizes = np.full(len(self.atoms), 26.0)
            keep = heavy[bonds[:, 0]] & heavy[bonds[:, 1]] & \
                ~self.water[bonds[:, 0]] & \
                (side[bonds[:, 0]] | side[bonds[:, 1]])
            self.bonds = bonds[keep]
        # The extent of the view: the cell of the first frame from every
        # angle of the camera, with a margin.
        L = cells[0]
        radius = 0.5 * np.hypot(L[0], L[1])
        tilt = np.radians(args.tilt)
        half = 0.5 * L[2] * np.cos(tilt) + radius * np.sin(tilt)
        self.half_height = 1.06 * max(half, radius * args.height / args.width)
        self.half_width = self.half_height * args.width / args.height

    def frame(self, index):
        cell = self.cells[index]
        x = whole(self.positions[index], cell, self.levels)
        # The center of the view: the protein, or the bilayer along z (the
        # circular mean, since it may straddle a face).
        if self.args.mode == "protein":
            center = x[self.focus].mean(axis=0)
        else:
            center = 0.5 * cell
            center[2] = circular_mean(x[self.focus, 2], cell[2])
        centers = np.zeros((self.molecules, 3))
        np.add.at(centers, self.molecule, x)
        centers /= np.bincount(self.molecule)[:, None]
        shift = cell * np.round((centers - center) / cell)
        x -= shift[self.molecule]
        return x - center, cell

    def draw_protein(self, axes, x, project):
        """The tube, the sticks, and the balls in slabs of depth, far to
        near, so that what is nearer covers what is behind it."""
        tube = catmull_rom(x[self.trace], 6)
        tu, tv, td = project(tube)
        tc = plt.cm.rainbow(np.linspace(0, 1, len(tube) - 1))
        tmid = 0.5 * (td[:-1] + td[1:])
        u, v, d = project(x[self.atoms])
        p = {a: k for k, a in enumerate(self.atoms)}
        pu, pv, pd = project(x)
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        bmid = 0.5 * (pd[i] + pd[j])
        edges = np.linspace(min(td.min(), d.min()) - 1e-6,
                            max(td.max(), d.max()) + 1e-6, 25)
        for slab in range(24):
            low, high = edges[24 - slab - 1], edges[24 - slab]
            z = 2 + slab / 24
            s = (tmid >= low) & (tmid < high)
            if s.any():
                k = np.flatnonzero(s)
                segments = np.stack([np.stack([tu[k], tv[k]], 1),
                                     np.stack([tu[k + 1], tv[k + 1]], 1)], 1)
                axes.add_collection(LineCollection(
                    segments, colors=tc[k], linewidths=9, zorder=z,
                    capstyle="round"))
            s = (bmid >= low) & (bmid < high)
            if s.any():
                segments = np.stack([np.stack([pu[i[s]], pv[i[s]]], 1),
                                     np.stack([pu[j[s]], pv[j[s]]], 1)], 1)
                axes.add_collection(LineCollection(
                    segments, colors="#b8b8b8", linewidths=1.8,
                    zorder=z + 0.01))
            s = (d >= low) & (d < high)
            if s.any():
                axes.scatter(u[s], v[s], s=self.sizes[s], c=self.colors[s],
                             linewidths=0.3, edgecolors="black",
                             zorder=z + 0.02)

    def render(self, index, path):
        args = self.args
        x, cell = self.frame(index)
        angle = 2 * np.pi * index / (args.turn * args.fps)
        c, s = np.cos(angle), np.sin(angle)
        tilt = np.radians(args.tilt)
        ct, st = np.cos(tilt), np.sin(tilt)

        def project(p):
            u = c * p[..., 0] - s * p[..., 1]
            w = s * p[..., 0] + c * p[..., 1]
            v = -w * st + p[..., 2] * ct
            depth = w * ct + p[..., 2] * st
            return u, v, depth

        figure = plt.figure(figsize=(args.width / 100, args.height / 100),
                            dpi=100)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.set_facecolor("black")
        axes.set_xlim(-self.half_width, self.half_width)
        axes.set_ylim(-self.half_height, self.half_height)
        axes.set_aspect("equal")
        axes.axis("off")

        # The edges of the cell, those behind the center under the atoms.
        h = 0.5 * cell
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1)
                            for k in (-1, 1)]) * h
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if np.sum(corners[a] != corners[b]) == 1]
        cu, cv, cd = project(corners)
        for a, b in edges:
            front = cd[a] + cd[b] < 0
            axes.plot([cu[a], cu[b]], [cv[a], cv[b]], color="#f0f0f0",
                      lw=1.6 if front else 1.0, alpha=0.9 if front else 0.45,
                      zorder=5 if front else 0)

        ou, ov, od = project(x[self.oxygens])
        axes.scatter(ou, ov, s=6 if args.mode == "membrane" else 4,
                     c="#3a8ee6", alpha=0.3, linewidths=0, zorder=1)
        if args.mode == "protein":
            self.draw_protein(axes, x, project)
        else:
            u, v, d = project(x[self.atoms])
            order = np.argsort(-d)
            axes.scatter(u[order], v[order], s=self.sizes[order],
                         c=self.colors[order], linewidths=0.3,
                         edgecolors="black", zorder=2)

        time = self.times[index] / 1000
        if args.mode == "membrane":
            area = cell[0] * cell[1]
            label = (f"{time:6.2f} ns   area {area / 100:6.2f} nm²   "
                     f"height {cell[2] / 10:5.2f} nm")
        else:
            label = (f"{time:6.2f} ns   volume "
                     f"{np.prod(cell) / 1000:6.2f} nm³")
        axes.text(0.02, 0.97, label, transform=axes.transAxes,
                  color="white", fontsize=15, va="top", family="monospace",
                  zorder=6)
        if args.title:
            axes.text(0.98, 0.97, args.title, transform=axes.transAxes,
                      color="#c0c0c0", fontsize=13, va="top", ha="right",
                      zorder=6)
        figure.savefig(path, facecolor="black")
        plt.close(figure)


SCENE = None


def render_one(item):
    index, path = item
    SCENE.render(index, path)


def main():
    global SCENE
    parser = argparse.ArgumentParser()
    parser.add_argument("structure")
    parser.add_argument("trajectory", nargs="+")
    parser.add_argument("movie")
    parser.add_argument("--mode", choices=("membrane", "protein"),
                        default="membrane")
    parser.add_argument("--ps", type=float, nargs="+", default=[10.0],
                        help="ps between the frames of each trajectory")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--tilt", type=float, default=15.0,
                        help="degrees of the camera above the plane")
    parser.add_argument("--turn", type=float, default=40.0,
                        help="seconds of the movie per turn of the camera")
    parser.add_argument("--title", default="")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    residue, residue_names, names = read_gro(args.structure)
    steps = args.ps * len(args.trajectory) if len(args.ps) == 1 else args.ps
    if len(steps) != len(args.trajectory):
        raise SystemExit("one --ps for all trajectories or one for each")
    cells, positions, times, start = [], [], [], 0.0
    for path, step in zip(args.trajectory, steps):
        c, x = read_dcd(path)
        if x.shape[1] != len(names):
            raise SystemExit(f"{x.shape[1]} particles in {path}, "
                             f"{len(names)} in the structure")
        t = start + step * np.arange(1, len(c) + 1)
        start = t[-1]
        cells.append(c[::args.stride])
        positions.append(x[::args.stride])
        times.append(t[::args.stride])
    cells, positions = np.concatenate(cells), np.concatenate(positions)
    SCENE = Scene(args, residue, residue_names, names, cells, positions)
    SCENE.times = np.concatenate(times)
    work = tempfile.mkdtemp(prefix="mdir-movie-")
    try:
        items = [(k, os.path.join(work, f"{k:05d}.png"))
                 for k in range(len(cells))]
        with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
            pool.map(render_one, items, chunksize=4)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate",
             str(args.fps), "-i", os.path.join(work, "%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             args.movie], check=True)
        print(f"wrote {len(items)} frames to {args.movie}")
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()

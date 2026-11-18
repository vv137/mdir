"""Probes of the writer of a list of units (D422):
its check of the extents after the positions of a step (R5, the convex rule
of D417, Section 3).

  segments_writer.py MODE WORK
  segments_writer.py builds DUMPS
      WORK/harness.mlir: an empty list of units carried through a loop over
      two frames, the writer and the refresh that takes its test at each
      (the form that md-exec-expose-validity gives a loop over steps). The
      first frame builds the list; the writer of the second checks it.

The system, in a cubic cell of 3 nm: two waters of a kind of three atoms
(TIP3P's lengths, covering radius 0.09582 nm), one copy of a kind of three
atoms and a site of weights (w0, w1, w2) on them (covering radius 0.1 nm),
and a molecule of the rest of two atoms, its second anchored to its first
(rho = 0.10901 nm). MODE:

  ok       every place within its radius: the site of the second kind at
           -0.4 (h1 + h2), 0.047 nm from its anchor; the run ends;
  convex   a hydrogen of the first water 0.11 nm from its oxygen, beyond
           the kind's radius: a frame atom, which the convex rule checks;
  hull     the second kind's site at -(h1 + h2), weights (3, -1, -1),
           0.117 nm from its anchor, with its frame atoms 0.09572 nm from
           it: a site outside the hull of its frame, which only the
           explicit check of such a site sees;
  anchor   the anchored member of the rest 0.12 nm from its anchor,
           beyond rho.

  beyond   eight waters in all, each moved whole at the second frame by
           zeta (1 + 2^-23), zeta = skin / 2 = 0.05 nm the least half-margin
           of the segments in a cell that does not change, each in a
           direction whose rounding alone would let it pass the f32 test
           without its counted margin: the test with its margin must find
           them moved, and the list is built again (2 builds);
  inside   the same waters moved by zeta (1 - 2^-10), beyond the counted
           margin delta = 2^-20 (|a|_1 + |t|_1) < 2^-15 nm of positions
           within 3 nm: the test must not find them moved (1 build).

Each probe of the extents must stop the run with their message. `builds
DUMPS` prints the builds of the last dump of the list (dump-segments).
Lengths in nm.
"""
import math
import sys
from pathlib import Path

import numpy as np

mode, work = sys.argv[1], Path(sys.argv[2])
if mode == 'builds':
    last = sorted(work.glob('dump_*.bin'))[-1]
    head = np.frombuffer(last.read_bytes()[:48], np.int64)
    print(f'builds {int(head[4])}')
    sys.exit(0)
assert mode in ('ok', 'convex', 'hull', 'anchor', 'beyond', 'inside')
OH, HH = 0.09572, 0.15136
HALF = math.asin(HH / (2 * OH))
CUTOFF, PAIRLIST, PRUNED, RHO = 0.8, 0.9, 0.85, 0.10901
weights = [3.0, -1.0, -1.0] if mode == 'hull' else [1.8, -0.4, -0.4]


def water(o, h1=OH):
    o = np.array(o)
    a = o + h1 * np.array([0.0, math.cos(HALF), math.sin(HALF)])
    b = o + OH * np.array([0.0, math.cos(HALF), -math.sin(HALF)])
    return [o, a, b]


w0 = water([1.0, 1.0, 1.0], 0.11 if mode == 'convex' else OH)
w1 = water([2.0, 1.0, 1.0])
p = water([1.0, 2.0, 1.0])
site = weights[0] * p[0] + weights[1] * p[1] + weights[2] * p[2]
x = np.array([2.0, 2.0, 2.0])
h = x + np.array([0.0, 0.0, 0.12 if mode == 'anchor' else 0.1])
more = [water(o) for o in ([0.3, 0.4, 2.6], [2.7, 0.3, 2.5], [0.5, 2.6, 0.4], [2.6, 2.7, 2.7],
                           [1.5, 2.6, 2.0], [2.5, 1.6, 0.5])] if mode in ('beyond', 'inside') else []
first = np.array(w0 + w1 + p + [site, x, h] + [a for w in more for a in w])
second = first + np.array([0.01, -0.005, 0.002])
if mode in ('beyond', 'inside'):
    zeta = (PAIRLIST - CUTOFF) / 2
    step = zeta * (1 + 2.0 ** -23 if mode == 'beyond' else 1 - 2.0 ** -10)
    second = first.copy()
    rng = np.random.default_rng(7)
    f = np.float32
    zeta32 = f(zeta * (1 - 2.0 ** -22))

    def bare(x, ref):
        # The f32 test without its counted margin: whether it passes.
        a, t = x.astype(f), ref.astype(f) * f(1.0)
        d = a - t
        return f(f(d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) <= f(zeta32 * zeta32)
    for o in [0, 3] + [12 + 3 * k for k in range(len(more))]:
        # For `beyond`, a direction whose rounding alone would let the unit
        # pass the f32 test without its counted margin, so that only the
        # margin finds it moved.
        for _ in range(1000):
            u = rng.normal(size=3)
            moved = first[o] + step * u / np.linalg.norm(u)
            if mode == 'inside' or bare(moved, first[o]):
                break
        second[o:o + 3] += moved - first[o]
frames = np.array([first, second])
N = len(first)
waters = [[0, 1, 2], [3, 4, 5]] + [[12 + 3 * k, 13 + 3 * k, 14 + 3 * k] for k in range(len(more))]
kinds = [(np.array(waters), '#md.rigid_kind<frame = [0, 1, 2], sites = [], weights = [], extent = 9.58200000000000e-02>'),
         (np.array([[6, 7, 8, 9]]), f'#md.rigid_kind<frame = [0, 1, 2], sites = [3], weights = [{", ".join(repr(float(w)) for w in weights)}], extent = 1.00000000000000e-01>')]
anchors = np.array([[11, 10]])
excluded = np.array([[10, 11]])
seg = 'unique, minimum, removed, grouped_by(group)'
K = len(kinds)
sp = (f'!mdrt.segmented_pairs<@atoms, kinds [{", ".join(d for _, d in kinds)}], '
      f'segments [{", ".join(f"({f}, {g}) {seg}" for f in range(K + 1) for g in range(f + 1))}]>')
skin, prune = PAIRLIST - CUTOFF, PRUNED - CUTOFF


def hexed(array):
    return '"0x' + np.ascontiguousarray(array).tobytes().hex().upper() + '"'


lines = ['module {',
         f'  memref.global "private" constant @frames : memref<2x{N}x3xf64> = dense<{hexed(frames.astype("<f8"))}>',
         f'  memref.global "private" constant @anchors : memref<1x2xi32> = dense<{hexed(anchors.astype("<i4"))}>',
         f'  memref.global "private" constant @excluded : memref<1x2xi32> = dense<{hexed(excluded.astype("<i4"))}>']
for f, (t, _) in enumerate(kinds):
    lines.append(f'  memref.global "private" constant @tuples{f} : memref<{t.shape[0]}x{t.shape[1]}xi32> = dense<{hexed(t.astype("<i4"))}>')


def device(name, global_, rows, width):
    return f'''    %g_{name} = memref.get_global @{global_} : memref<{rows}x{width}xi32>
    %h_{name} = memref.cast %g_{name} : memref<{rows}x{width}xi32> to memref<?x{width}xi32>
    %n_{name} = memref.dim %h_{name}, %c0 : memref<?x{width}xi32>
    %{name} = gpu.alloc (%n_{name}) : memref<?x{width}xi32, 1>
    %w0_{name} = gpu.wait async
    %w1_{name} = gpu.memcpy async [%w0_{name}] %{name}, %h_{name} : memref<?x{width}xi32, 1>, memref<?x{width}xi32>
    gpu.wait [%w1_{name}]'''


body = [f'''  func.func @main() {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %N = arith.constant {N} : index
    %edge = arith.constant 3.0 : f64
    %fr = memref.get_global @frames : memref<2x{N}x3xf64>''',
        device('da', 'anchors', 1, 2), device('dx', 'excluded', 1, 2)]
for f, (t, _) in enumerate(kinds):
    body.append(device(f'dk{f}', f'tuples{f}', t.shape[0], t.shape[1]))
body.append(f'''    %du = gpu.alloc (%N) : memref<?xi32, 1>
    %hp = memref.alloc(%N) : memref<?x3xf64>
    %dp = gpu.alloc (%N) : memref<?x3xf64, 1>
    %l0 = md_exec.empty_segmented_pairs %du kinds(%dk0, %dk1 : memref<?x3xi32, 1>, memref<?x4xi32, 1>) exclude(%dx : memref<?x2xi32, 1>) anchor(%da : memref<?x2xi32, 1>) radius({RHO}) cutoff({CUTOFF}) skin({skin:.9f}) : memref<?xi32, 1> -> {sp}
    %lend = scf.for %t = %c0 to %c2 step %c1 iter_args(%l = %l0) -> ({sp}) {{
      scf.for %i = %c0 to %N step %c1 {{
        %x0 = memref.load %fr[%t, %i, %c0] : memref<2x{N}x3xf64>
        %x1 = memref.load %fr[%t, %i, %c1] : memref<2x{N}x3xf64>
        %x2 = memref.load %fr[%t, %i, %c2] : memref<2x{N}x3xf64>
        memref.store %x0, %hp[%i, %c0] : memref<?x3xf64>
        memref.store %x1, %hp[%i, %c1] : memref<?x3xf64>
        memref.store %x2, %hp[%i, %c2] : memref<?x3xf64>
      }}
      %w8 = gpu.wait async
      %w9 = gpu.memcpy async [%w8] %dp, %hp : memref<?x3xf64, 1>, memref<?x3xf64>
      gpu.wait [%w9]
      %cell = md.orthorhombic_cell %edge, %edge, %edge
      %moved, %stale = md_exec.write_unit_frames %l at(%dp, %cell) cutoff({CUTOFF}) skin({skin:.9f}) prune_skin({prune:.9f}) : {sp}, memref<?x3xf64, 1>
      %l1 = md_exec.refresh_segmented_pairs %l at(%dp, %cell) moved(%moved) stale(%stale) cutoff({CUTOFF}) skin({skin:.9f}) prune_skin({prune:.9f}) : {sp}, memref<?x3xf64, 1>
      scf.yield %l1 : {sp}
    }}
    return
  }}
}}''')
(work / 'harness.mlir').write_text('\n'.join(lines + body) + '\n')
if mode in ('beyond', 'inside'):
    # The f32 test of the kernel, simulated in the same order of operations,
    # with its counted margin and without it (z = zeta, c = 1): the margin
    # is what keeps a unit moved by more than zeta from passing.
    f = np.float32
    zeta32 = f(zeta * (1 - 2.0 ** -22))
    passes = {True: 0, False: 0}
    tested = [0, 3] + [12 + 3 * k for k in range(len(more))]
    for o in tested:
        a = second[o].astype(f)
        t = first[o].astype(f) * f(1.0)
        d = a - t
        d2 = f(f(d[0] * d[0] + d[1] * d[1]) + d[2] * d[2])
        sizes = f(f(f(abs(a[0]) + abs(t[0])) + f(abs(a[1]) + abs(t[1]))) + f(abs(a[2]) + abs(t[2])))
        z = f(zeta32 - f(sizes * f(2.0 ** -20)))
        passes[True] += bool(z > 0 and d2 <= f(f(z * z) * f(1 - 2.0 ** -18)))
        passes[False] += bool(d2 <= f(zeta32 * zeta32))
    print(f'{mode}: of {len(tested)} units, the f32 test passes {passes[True]} with its counted margin '
          f'and {passes[False]} without it')
d_site = float(np.linalg.norm(site - p[0]))
print(f'{mode}: {N} particles; the site of the second kind {d_site:.4f} nm from its anchor')

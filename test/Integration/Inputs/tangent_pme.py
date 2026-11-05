"""The tangent of a reciprocal sum in the positions and its derivative in a
number that the charges depend on (D379, #726).

Writes an MLIR file whose `main` evaluates, by md.reciprocal (smooth
particle mesh Ewald), the energy E, the forces, the tangent
Phi = sum_i t_i . dE/dx_i along a field t, its derivative in lambda, where
the charges are q_i = q0_i + lambda w_i, and dE/dlambda, and compares each
with the Ewald sum of the reciprocal space evaluated here with NumPy:

  E   = f / (2 pi V) sum_{m != 0} g(m) |S(m)|^2,
  g   = exp(-pi^2 m^2 / beta^2) / m^2,   S(m) = sum_j q_j exp(2 pi i m.x_j),
  Phi = f / (pi V) sum_m g Re(conj(S) D),  D(m) = 2 pi i sum_j q_j (m.t_j) exp(2 pi i m.x_j),
  dPhi/dlambda = f / (pi V) sum_m g Re(conj(S_w) D + conj(S) D_w),

with S_w and D_w the sums with w in place of q (both are linear in the
charges). The closed forms are checked against central differences here
before they are written. A check prints the relative error and then 1 if
it is within the tolerance (the first argument), else 0.

  tangent_pme.py TOLERANCE OUT.mlir [fd]

With `fd`, the forces are not requested, for a run under
md-check-derivatives, which then checks the requests of the tangent alone.
"""
import sys

import numpy as np

tolerance = float(sys.argv[1])
out = sys.argv[2]
differences = len(sys.argv) > 3 and sys.argv[3] == "fd"

f = 138.935458
beta = 3.0
edges = np.array([2.0, 2.2, 1.9])
grid, order = (48, 54, 48), 8
rng = np.random.default_rng(726)
n = 10
x = rng.uniform(0.0, 1.0, (n, 3)) * edges
t = rng.normal(size=(n, 3))
q0 = rng.uniform(-1.0, 1.0, n)
w = rng.uniform(-0.5, 0.5, n)
lam = 0.3
kmax = 16

ks = np.array([(a, b, c) for a in range(-kmax, kmax + 1) for b in range(-kmax, kmax + 1)
               for c in range(-kmax, kmax + 1) if (a, b, c) != (0, 0, 0)], dtype=float)
m = ks / edges
m2 = np.sum(m * m, axis=1)
g = np.exp(-np.pi ** 2 * m2 / beta ** 2) / m2
volume = np.prod(edges)


def structure(x, charges):
    return np.exp(2j * np.pi * (x @ m.T)) .T @ charges


def directional(x, charges, t):
    phase = np.exp(2j * np.pi * (x @ m.T))  # (n, M)
    rate = t @ m.T  # (n, M): m . t_j
    return 2j * np.pi * np.sum(charges[:, None] * rate * phase, axis=0)


def energy(x, q):
    return f / (2 * np.pi * volume) * np.sum(g * np.abs(structure(x, q)) ** 2)


def gradient(x, q):
    phase = np.exp(2j * np.pi * (x @ m.T))
    s = structure(x, q)
    # dE/dx_i = f/(pi V) sum g Re(conj(S) q_i 2 pi i m e^{i theta_i})
    terms = np.real(np.conj(s)[None, :] * q[:, None] * 2j * np.pi * phase)  # (n, M)
    return f / (np.pi * volume) * (terms * g[None, :]) @ m


def tangent(x, q, t):
    return f / (np.pi * volume) * np.sum(g * np.real(np.conj(structure(x, q)) * directional(x, q, t)))


def tangent_slope(x, q, w, t):
    return f / (np.pi * volume) * np.sum(g * np.real(
        np.conj(structure(x, w)) * directional(x, q, t) + np.conj(structure(x, q)) * directional(x, w, t)))


def energy_slope(x, q, w):
    return f / (np.pi * volume) * np.sum(g * np.real(np.conj(structure(x, q)) * structure(x, w)))


q = q0 + lam * w
e, grad = energy(x, q), gradient(x, q)
phi, slope, de = tangent(x, q, t), tangent_slope(x, q, w, t), energy_slope(x, q, w)

# The closed forms against central differences of the energy.
h = 1e-5
assert abs((energy(x + h * t, q) - energy(x - h * t, q)) / (2 * h) - phi) < 1e-7 * abs(phi)
assert abs(np.sum(grad * t) - phi) < 1e-12 * abs(phi)
assert abs((tangent(x, q + h * w, t) - tangent(x, q - h * w, t)) / (2 * h) - slope) < 1e-7 * abs(slope)
assert abs((energy(x, q + h * w) - energy(x, q - h * w)) / (2 * h) - de) < 1e-7 * abs(de)


def moduli(count):
    # |b(k)|^2 of the B-splines of the order, as the builder computes them.
    mn = np.zeros(order + 1)
    mn[1] = 1.0
    for k in range(3, order + 1):
        nxt = np.zeros(order + 1)
        for j in range(1, k):
            nxt[j] = (j * mn[j] + (k - j) * mn[j - 1]) / (k - 1)
        mn = nxt
    values = []
    for k in range(count):
        z = sum(mn[j + 1] * np.exp(2j * np.pi * k * j / count) for j in range(order - 1))
        values.append(1.0 / abs(z) ** 2)
    return values


longest = max(grid)
table = np.zeros((3, longest))
for axis in range(3):
    table[axis, :grid[axis]] = moduli(grid[axis])


def dense(array, shape, kind="f64"):
    rows = np.atleast_2d(array) if len(shape) == 2 else array
    if len(shape) == 1:
        return "[" + ", ".join(f"{v:.17e}" for v in rows) + "]"
    return "[" + ", ".join("[" + ", ".join(f"{v:.17e}" for v in row) + "]" for row in rows) + "]"


def load(name, symbol, shape, kind):
    # A field of the values of a global, in a buffer of its own.
    return f"""  %{name}_s = memref.get_global @{symbol} : memref<{n}{shape}xf64>
  %{name}_d = memref.cast %{name}_s : memref<{n}{shape}xf64> to memref<?{shape}xf64>
  %{name}_c = arith.constant 0 : index
  %{name}_n = memref.dim %{name}_d, %{name}_c : memref<?{shape}xf64>
  %{name}_b = memref.alloc(%{name}_n) : memref<?{shape}xf64>
  memref.copy %{name}_d, %{name}_b : memref<?{shape}xf64> to memref<?{shape}xf64>
  %{name} = mdrt.from_buffer %{name}_b : memref<?{shape}xf64> to {kind}"""


if differences:
    evaluate = """  %u, %p, %dp, %du = md.evaluate @mesh(%x, %cell, %q0, %w, %lam, %moduli, %t)
      request [energy, tangent, tangent_derivative(4), derivative(4)]
      : (!vec, !md.cell, !real, !real, f64, !grid, !vec) -> (f64, f64, f64, f64)"""
    dot = ""
else:
    evaluate = """  %u, %f, %p, %dp, %du = md.evaluate @mesh(%x, %cell, %q0, %w, %lam, %moduli, %t)
      request [energy, forces, tangent, tangent_derivative(4), derivative(4)]
      : (!vec, !md.cell, !real, !real, f64, !grid, !vec)
        -> (f64, !vec, f64, f64, f64)"""
    dot = """  // The tangent is -sum t . F of the forces of the same call.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cn = arith.constant NPART : index
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>
  %tb = mdrt.to_buffer %t : !vec to memref<?x3xf64>
  %zero = arith.constant 0.0 : f64
  %dot = scf.for %i = %c0 to %cn step %c1 iter_args(%acc = %zero) -> (f64) {
    %inner = scf.for %k = %c0 to %c3 step %c1 iter_args(%a = %acc) -> (f64) {
      %fk = memref.load %forces[%i, %k] : memref<?x3xf64>
      %tk = memref.load %tb[%i, %k] : memref<?x3xf64>
      %mk = arith.mulf %fk, %tk : f64
      %s = arith.subf %a, %mk : f64
      scf.yield %s : f64
    }
    scf.yield %inner : f64
  }
  call @check(%p, %dot) : (f64, f64) -> ()
""".replace("NPART", str(n))

lines = f'''!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>
!grid = !md.table<2, f64>
!pairs = !md.relation<@atoms, 2, unordered>
md.particle_set @atoms

md.potential @mesh(%x: !vec, %cell: !md.cell, %q0: !real, %w: !real,
                   %lam: f64, %moduli: !grid) -> f64 {{
  %q = md.map_particles gather(%q0, %w : !real, !real) {{
  ^bb0(%a: f64, %b: f64):
    %s = arith.mulf %lam, %b : f64
    %c = arith.addf %a, %s : f64
    md.yield %c : f64
  }} : !real
  %u, %f, %v = md.reciprocal %x, %q, %cell, %moduli
      grid([{grid[0]}, {grid[1]}, {grid[2]}]) order({order}) beta({beta}) coulomb({f})
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  md.return %u : f64
}}

// With a pair term that a second number scales: the derivative of the
// tangent in it reads the pairs alone, and the reciprocal sum, which does
// not depend on it, takes no part, by the proof of activity.
md.potential @both(%x: !vec, %cell: !md.cell, %q0: !real, %w: !real,
                   %lam: f64, %moduli: !grid, %eps: f64) -> f64 {{
  %n = md.neighborhood %x, %cell cutoff(0.9) : !vec -> !pairs
  %up = md.sum_relation %n, %x, %cell exchange(symmetric) truncation(switch, from = 0.7) {{
  ^bb0(%r: f64, %d: vector<3xf64>):
    %sigma = arith.constant 0.3 : f64
    %i6 = arith.constant 6 : i32
    %sr = arith.divf %sigma, %r : f64
    %s6 = math.fpowi %sr, %i6 : f64, i32
    %k = arith.mulf %eps, %s6 : f64
    md.yield %k : f64
  }} : !pairs, !vec -> f64
  %q = md.map_particles gather(%q0, %w : !real, !real) {{
  ^bb0(%a: f64, %b: f64):
    %s = arith.mulf %lam, %b : f64
    %c = arith.addf %a, %s : f64
    md.yield %c : f64
  }} : !real
  %ur, %f, %v = md.reciprocal %x, %q, %cell, %moduli
      grid([{grid[0]}, {grid[1]}, {grid[2]}]) order({order}) beta({beta}) coulomb({f})
      : !vec, !real, !grid -> f64, !vec, vector<9xf64>
  %u = arith.addf %up, %ur : f64
  md.return %u : f64
}}

memref.global "private" constant @positions : memref<{n}x3xf64> = dense<{dense(x, (n, 3))}>
memref.global "private" constant @tangents : memref<{n}x3xf64> = dense<{dense(t, (n, 3))}>
memref.global "private" constant @q0 : memref<{n}xf64> = dense<{dense(q0, (n,))}>
memref.global "private" constant @w : memref<{n}xf64> = dense<{dense(w, (n,))}>
memref.global "private" constant @moduli : memref<3x{longest}xf64> = dense<{dense(table, (3, longest))}>

// The lowering of the reciprocal sum parses its template, which holds ops
// of llvm, while the pass manager runs; a use of the dialect here loads it
// before, as the driver does.
llvm.func @tangent_pme_unused()

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64) {{
  %tolerance = arith.constant {tolerance:.3e} : f64
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %magnitude = math.absf %reference : f64
  %relative = arith.divf %error, %magnitude : f64
  %agrees = arith.cmpf ole, %relative, %tolerance : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%relative) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}}

func.func @main() {{
{load("x", "positions", "x3", "!vec")}
{load("t", "tangents", "x3", "!vec")}
{load("q0", "q0", "", "!real")}
{load("w", "w", "", "!real")}
  %ms = memref.get_global @moduli : memref<3x{longest}xf64>
  %md = memref.cast %ms : memref<3x{longest}xf64> to memref<?x?xf64>
  %moduli = mdrt.from_buffer %md : memref<?x?xf64> to !grid
  %ax = arith.constant {edges[0]:.17e} : f64
  %ay = arith.constant {edges[1]:.17e} : f64
  %az = arith.constant {edges[2]:.17e} : f64
  %cell = md.orthorhombic_cell %ax, %ay, %az
  %lam = arith.constant {lam:.17e} : f64
  %eps = arith.constant 0.05 : f64

{evaluate}
  %pb, %dpb, %dpe = md.evaluate @both(%x, %cell, %q0, %w, %lam, %moduli, %eps, %t)
      request [tangent, tangent_derivative(4), tangent_derivative(6)]
      : (!vec, !md.cell, !real, !real, f64, !grid, f64, !vec) -> (f64, f64, f64)

  %e_ref = arith.constant {e:.17e} : f64
  %phi_ref = arith.constant {phi:.17e} : f64
  %slope_ref = arith.constant {slope:.17e} : f64
  %de_ref = arith.constant {de:.17e} : f64
  call @check(%u, %e_ref) : (f64, f64) -> ()
  call @check(%p, %phi_ref) : (f64, f64) -> ()
  call @check(%dp, %slope_ref) : (f64, f64) -> ()
  call @check(%du, %de_ref) : (f64, f64) -> ()
  // The pair term does not depend on lambda: the derivative of the tangent
  // of @both in lambda is that of @mesh.
  call @check(%dpb, %dp) : (f64, f64) -> ()

{dot}  call @printF64(%dpe) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}}
'''
with open(out, "w") as stream:
    stream.write(lines)

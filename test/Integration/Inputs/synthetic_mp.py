"""A synthetic message-passing model for the tests of the neural stages (#377, step 3).

Two layers of the form of PaiNN [Schutt2021]: per-atom features of 4 scalars
and 4 vectors (irreps 4x0e + 4x1o, in the layout mul_ir: the 4 scalars, then
the 3 components of each vector), an embedding of the species, an edge stage
with a cosine cutoff and a radial basis, a node stage that mixes the vector
channels, and a readout. Its weights are drawn from a fixed seed. It stands
in for a trained model in the tests of docs/ml4-ir.md, Section 6:

    python3 synthetic_mp.py stages     the bodies of the stages, as func.func
                                       of arith, math, and vector: each
                                       forward body, and its vector-Jacobian
                                       product from a reverse sweep of this
                                       script (the stand-in for external AD)
    python3 synthetic_mp.py reference  the energy, the forces, and the virial
                                       of a small periodic system, evaluated
                                       from the definitions below over the
                                       whole system, independently of the
                                       bodies and of md-differentiate: the
                                       forces and the virial by one reverse
                                       sweep of the whole energy, checked
                                       against central differences of it,
                                       extrapolated in the step
    python3 synthetic_mp.py check      the bodies' arithmetic, evaluated in
                                       Python, against the NumPy model, and
                                       the VJPs against differences

Lengths are in nm and the energy in kJ/mol; the numbers have no physical
meaning. The model is invariant under rotations, so its virial is
symmetric. `stages` runs in any Python 3 (lit runs it); `reference` and
`check` need NumPy.
"""

import math
import sys

try:
    import numpy as np
except ImportError:  # `stages` needs no NumPy, so that lit runs it.
    np = None

F = 4                      # channels: 4 scalars and 4 vectors
N = F + 3 * F              # 16 components a particle
HIDDEN = 6                 # width of the hidden layers of the stages
RADIAL = 3                 # functions of the radial basis
CUTOFF = 0.5               # nm
EPSILON = 1.0e-2           # inside the norm of the vector channels
SEED = 377

# The system: 10 particles of two species in a cubic cell of 1.2 nm, more
# than twice the cutoff, so that a pair has one image within it.
COUNT = 10
EDGE = 1.2


def weights():
    """The parameters of the model, from the seed: normal numbers by the
    method of Box and Muller from a linear congruential generator, so that
    they are the same in every Python, with or without NumPy."""
    state = [SEED]

    def uniform():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) \
            % 2 ** 64
        return ((state[0] >> 11) + 0.5) / 2.0 ** 53

    def normal():
        return math.sqrt(-2.0 * math.log(uniform())) * \
            math.cos(2.0 * math.pi * uniform())

    def draw(*shape, scale=0.5):
        if len(shape) == 1:
            return [scale * normal() for _ in range(shape[0])]
        return [draw(*shape[1:], scale=scale) for _ in range(shape[0])]

    w = {"embed": draw(2, F)}
    for layer in (1, 2):
        w[f"edge{layer}"] = {
            "radial": draw(3 * F, RADIAL),
            "a1": draw(HIDDEN, 2 * F), "b1": draw(HIDDEN, scale=0.1),
            "a2": draw(3 * F, HIDDEN), "b2": draw(3 * F, scale=0.1),
        }
        w[f"node{layer}"] = {
            "u": draw(F, F), "v": draw(F, F),
            "a1": draw(HIDDEN, 2 * F), "b1": draw(HIDDEN, scale=0.1),
            "a2": draw(3 * F, HIDDEN), "b2": draw(3 * F, scale=0.1),
        }
    w["readout"] = {"a1": draw(5, F), "b1": draw(5, scale=0.1),
                    "a2": draw(5), "b2": 0.3}
    return w


def system():
    """Positions and species: no pair within 0.2 nm, none within 0.02 nm
    of the cutoff, at which a difference would straddle the edge of the
    relation."""
    rng = np.random.default_rng(SEED + 1)
    while True:
        x = rng.uniform(0.0, EDGE, size=(COUNT, 3))
        r = []
        for i in range(COUNT):
            for j in range(i + 1, COUNT):
                d = x[i] - x[j]
                d -= EDGE * np.round(d / EDGE)
                r.append(np.linalg.norm(d))
        r = np.array(r)
        if r.min() > 0.2 and np.abs(r - CUTOFF).min() > 0.02 and \
                (r < CUTOFF).sum() >= 2 * COUNT:
            break
    z = np.array([i % 2 for i in range(COUNT)], dtype=np.int32)
    return np.round(x, 6), z


# ---------------------------------------------------------------------------
# The stages, written once over an arithmetic `m` (NumPy floats, or the
# tracer below), one element at a time.
# ---------------------------------------------------------------------------

def embed(m, w, z):
    s = [m.add(m.const(w["embed"][0][f]),
               m.mul(z, m.const(w["embed"][1][f] - w["embed"][0][f])))
         for f in range(F)]
    return s + [m.const(0.0)] * (3 * F)


def mlp(m, a1, b1, a2, b2, inputs):
    hidden = []
    for k in range(len(a1)):
        t = m.const(b1[k])
        for q, value in enumerate(inputs):
            t = m.add(t, m.mul(m.const(a1[k][q]), value))
        hidden.append(m.tanh(t))
    out = []
    for k in range(len(a2)):
        t = m.const(b2[k])
        for q, value in enumerate(hidden):
            t = m.add(t, m.mul(m.const(a2[k][q]), value))
        out.append(t)
    return out


def edge(m, w, r, d, hi, hj, zi, zj):
    """The message from j to the center i over the pair (i, j)."""
    x = m.mul(r, m.const(1.0 / CUTOFF))
    cutoff = m.mul(m.const(0.5),
                   m.add(m.cos(m.mul(m.const(math.pi), x)), m.const(1.0)))
    basis = [m.div(m.sin(m.mul(m.const(k * math.pi), x)), r)
             for k in range(1, RADIAL + 1)]
    filters = []
    for q in range(3 * F):
        t = m.const(0.0)
        for k in range(RADIAL):
            t = m.add(t, m.mul(m.const(w["radial"][q][k]), basis[k]))
        filters.append(m.mul(cutoff, t))
    phi = mlp(m, w["a1"], w["b1"], w["a2"], w["b2"], hj[:F] + hi[:F])
    species = m.add(m.const(1.0), m.add(m.mul(m.const(0.1), zi),
                                        m.mul(m.const(0.2), zj)))
    x = [m.mul(species, m.mul(phi[q], filters[q])) for q in range(3 * F)]
    direction = [m.div(d[c], r) for c in range(3)]
    s = x[:F]
    v = []
    for f in range(F):
        for c in range(3):
            v.append(m.add(m.mul(x[F + f], direction[c]),
                           m.mul(x[2 * F + f], hj[F + 3 * f + c])))
    return s + v


def node(m, w, h, a):
    s = [m.add(h[f], a[f]) for f in range(F)]
    v = [m.add(h[q], a[q]) for q in range(F, N)]
    uv, vv = [], []
    for f in range(F):
        for c in range(3):
            tu, tv = m.const(0.0), m.const(0.0)
            for g in range(F):
                tu = m.add(tu, m.mul(m.const(w["u"][f][g]), v[3 * g + c]))
                tv = m.add(tv, m.mul(m.const(w["v"][f][g]), v[3 * g + c]))
            uv.append(tu)
            vv.append(tv)
    norms, dots = [], []
    for f in range(F):
        t, p = m.const(EPSILON), m.const(0.0)
        for c in range(3):
            t = m.add(t, m.mul(vv[3 * f + c], vv[3 * f + c]))
            p = m.add(p, m.mul(uv[3 * f + c], vv[3 * f + c]))
        norms.append(m.sqrt(t))
        dots.append(p)
    o = mlp(m, w["a1"], w["b1"], w["a2"], w["b2"], s + norms)
    out = [m.add(s[f], m.add(o[2 * F + f], m.mul(o[F + f], dots[f])))
           for f in range(F)]
    for f in range(F):
        for c in range(3):
            out.append(m.add(v[3 * f + c], m.mul(o[f], uv[3 * f + c])))
    return out


def readout(m, w, h):
    return mlp(m, w["a1"], w["b1"], [w["a2"]], [w["b2"]], h[:F])[0]


class Numbers:
    """The arithmetic of NumPy floats."""

    def const(self, value):
        return float(value)

    add = staticmethod(lambda a, b: a + b)
    mul = staticmethod(lambda a, b: a * b)
    div = staticmethod(lambda a, b: a / b)
    tanh = staticmethod(math.tanh)
    sqrt = staticmethod(math.sqrt)
    sin = staticmethod(math.sin)
    cos = staticmethod(math.cos)


# ---------------------------------------------------------------------------
# The model in NumPy over a whole system, from the definitions above.
# ---------------------------------------------------------------------------

def pairs(x, cell):
    """The directed pairs (i, j) within the cutoff, d_ij = x_i - x_j + n H,
    over the 27 nearest images; with the cell larger than twice the cutoff
    at most one image of a pair is within it."""
    out = []
    shifts = [np.array([a, b, c]) for a in (-1, 0, 1) for b in (-1, 0, 1)
              for c in (-1, 0, 1)]
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            for n in shifts:
                d = x[i] - x[j] + n @ cell
                r = np.linalg.norm(d)
                if r < CUTOFF:
                    out.append((i, j, r, d))
    return out


def energy(w, x, cell, z):
    m = Numbers()
    h = [embed(m, w, float(z[i])) for i in range(len(x))]
    edges = pairs(x, cell)
    for layer in (1, 2):
        a = [[0.0] * N for _ in range(len(x))]
        for i, j, r, d in edges:
            msg = edge(m, w[f"edge{layer}"], r, list(d), h[i], h[j],
                       float(z[i]), float(z[j]))
            a[i] = [p + q for p, q in zip(a[i], msg)]
        h = [node(m, w[f"node{layer}"], h[i], a[i]) for i in range(len(x))]
    return sum(readout(m, w["readout"], h[i]) for i in range(len(x)))


def derivatives(w, x, cell, z):
    """Forces F = -dU/dx and the virial W, W_ba = -dU/de_ab under the strain
    x -> (1 + e) x, H -> H (1 + e)^T, both by central differences with the
    steps h, h/2, and h/4 combined to cancel the terms in h^2 and h^4
    (Richardson)."""
    def central(f, h):
        return (f(h) - f(-h)) / (2.0 * h)

    def richardson(f, h=1.0e-3):
        first = (4.0 * central(f, h / 2.0) - central(f, h)) / 3.0
        second = (4.0 * central(f, h / 4.0) - central(f, h / 2.0)) / 3.0
        return (16.0 * second - first) / 15.0

    forces = np.zeros_like(x)
    for i in range(len(x)):
        for c in range(3):
            def shifted(h):
                y = x.copy()
                y[i, c] += h
                return energy(w, y, cell, z)
            forces[i, c] = -richardson(shifted)
    virial = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            def strained(h):
                e = np.eye(3)
                e[a, b] += h
                return energy(w, x @ e.T, cell @ e.T, z)
            virial[b, a] = -richardson(strained)
    return forces, virial


# ---------------------------------------------------------------------------
# The bodies as MLIR, by tracing the stages; the VJP by a reverse sweep.
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, op, args=(), value=None):
        self.op, self.args, self.value = op, args, value


class Tracer:
    """Records the scalar operations of a stage."""

    def __init__(self):
        self.nodes = []
        self.constants = {}

    def _node(self, op, *args, value=None):
        n = Node(op, args, value)
        self.nodes.append(n)
        return n

    def const(self, value):
        value = float(value)
        if value not in self.constants:
            self.constants[value] = self._node("const", value=value)
        return self.constants[value]

    def input(self, name):
        return self._node("input", value=name)

    def add(self, a, b): return self._node("add", a, b)
    def sub(self, a, b): return self._node("sub", a, b)
    def mul(self, a, b): return self._node("mul", a, b)
    def div(self, a, b): return self._node("div", a, b)
    def neg(self, a): return self._node("neg", a)
    def tanh(self, a): return self._node("tanh", a)
    def sqrt(self, a): return self._node("sqrt", a)
    def sin(self, a): return self._node("sin", a)
    def cos(self, a): return self._node("cos", a)


def adjoint(t, outputs, cotangents, wanted):
    """Appends to the tracer `t` the reverse sweep of the nodes that `outputs`
    depend on, seeded with `cotangents`, and returns the cotangents of the
    nodes `wanted` (a constant zero where there is none)."""
    bar = {}
    for out, seed in zip(outputs, cotangents):
        bar[out] = t.add(bar[out], seed) if out in bar else seed
    for n in reversed(list(t.nodes)):
        if n not in bar or n.op in ("const", "input"):
            continue
        g = bar[n]
        if n.op == "add":
            parts = [(n.args[0], g), (n.args[1], g)]
        elif n.op == "sub":
            parts = [(n.args[0], g), (n.args[1], t.neg(g))]
        elif n.op == "mul":
            parts = [(n.args[0], t.mul(g, n.args[1])),
                     (n.args[1], t.mul(g, n.args[0]))]
        elif n.op == "div":
            q = t.div(g, n.args[1])
            parts = [(n.args[0], q), (n.args[1], t.neg(t.mul(q, n)))]
        elif n.op == "neg":
            parts = [(n.args[0], t.neg(g))]
        elif n.op == "tanh":
            parts = [(n.args[0],
                      t.mul(g, t.sub(t.const(1.0), t.mul(n, n))))]
        elif n.op == "sqrt":
            parts = [(n.args[0], t.div(t.mul(t.const(0.5), g), n))]
        elif n.op == "sin":
            parts = [(n.args[0], t.mul(g, t.cos(n.args[0])))]
        elif n.op == "cos":
            parts = [(n.args[0], t.neg(t.mul(g, t.sin(n.args[0]))))]
        else:
            raise ValueError(n.op)
        for arg, part in parts:
            if arg.op == "const":
                continue
            bar[arg] = t.add(bar[arg], part) if arg in bar else part
    return [bar.get(n, t.const(0.0)) for n in wanted]


MLIR_OPS = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf",
            "div": "arith.divf", "neg": "arith.negf", "tanh": "math.tanh",
            "sqrt": "math.sqrt", "sin": "math.sin", "cos": "math.cos"}


def vector_type(size):
    return "f64" if size == 1 else f"vector<{size}xf64>"


def emit(name, arguments, results, t):
    """A func.func named `name` of the traced nodes of `t`. `arguments` is a
    list of (type, nodes): ("f64" | "i32", [node]) or (size, [nodes]);
    `results` a list of lists of nodes."""
    lines = []
    names = {}
    signature = []
    for k, (kind, nodes) in enumerate(arguments):
        arg = f"%arg{k}"
        if kind == "i32":
            signature.append(f"{arg}: i32")
            lines.append(f"  %z{k} = arith.sitofp {arg} : i32 to f64")
            names[nodes[0]] = f"%z{k}"
        elif len(nodes) == 1:
            signature.append(f"{arg}: f64")
            names[nodes[0]] = arg
        else:
            signature.append(f"{arg}: {vector_type(len(nodes))}")
            for c, node in enumerate(nodes):
                v = f"%a{k}_{c}"
                lines.append(f"  {v} = vector.extract {arg}[{c}] : f64 from "
                             f"{vector_type(len(nodes))}")
                names[node] = v
    used = set()

    def mark(n):
        stack = [n]
        while stack:
            n = stack.pop()
            if n in used:
                continue
            used.add(n)
            stack.extend(n.args)
    for nodes in results:
        for n in nodes:
            mark(n)
    count = 0
    for n in t.nodes:
        if n not in used or n in names:
            continue
        if n.op == "const":
            names[n] = f"%c{count}"
            lines.append(f"  %c{count} = arith.constant {n.value!r} : f64")
        elif n.op == "input":
            raise ValueError("an input without an argument")
        else:
            names[n] = f"%v{count}"
            operands = ", ".join(names[a] for a in n.args)
            lines.append(f"  %v{count} = {MLIR_OPS[n.op]} {operands} : f64")
        count += 1
    returned, types = [], []
    for k, nodes in enumerate(results):
        if len(nodes) == 1:
            returned.append(names[nodes[0]])
            types.append("f64")
        else:
            v = f"%r{k}"
            lines.append(f"  {v} = vector.from_elements "
                         + ", ".join(names[n] for n in nodes)
                         + f" : {vector_type(len(nodes))}")
            returned.append(v)
            types.append(vector_type(len(nodes)))
    lines.append(f"  return {', '.join(returned)} : {', '.join(types)}")
    head = (f"func.func @{name}({', '.join(signature)}) -> "
            + (types[0] if len(types) == 1 else f"({', '.join(types)})")
            + " {")
    return "\n".join([head] + lines + ["}"])


def trace(stage, w):
    """The tracer, the arguments, and the outputs of one stage."""
    t = Tracer()
    if stage == "embed":
        z = t.input("z")
        return t, [("i32", [z])], embed(t, w, z)
    if stage.startswith("edge"):
        r = t.input("r")
        d = [t.input(f"d{c}") for c in range(3)]
        hi = [t.input(f"hi{q}") for q in range(N)]
        hj = [t.input(f"hj{q}") for q in range(N)]
        zi, zj = t.input("zi"), t.input("zj")
        out = edge(t, w[stage], r, d, hi, hj, zi, zj)
        return t, [(1, [r]), (3, d), (N, hi), (N, hj), ("i32", [zi]),
                   ("i32", [zj])], out
    if stage.startswith("node"):
        h = [t.input(f"h{q}") for q in range(N)]
        a = [t.input(f"a{q}") for q in range(N)]
        return t, [(N, h), (N, a)], node(t, w[stage], h, a)
    h = [t.input(f"h{q}") for q in range(N)]
    return t, [(N, h)], [readout(t, w["readout"], h)]


STAGES = ["embed", "edge1", "node1", "edge2", "node2", "readout"]


def stages():
    w = weights()
    print("// The bodies of the stages of the synthetic model of")
    print("// Inputs/synthetic_mp.py (#377, step 3), written by")
    print("// `python3 synthetic_mp.py stages`: each forward body `<stage>.f`")
    print("// and its vector-Jacobian product `<stage>.b`, from the script's")
    print("// reverse sweep. Do not edit.")
    print()
    for stage in STAGES:
        t, arguments, out = trace(stage, w)
        print(emit(f"{stage}.f", arguments, [out], t))
        print()
        if stage == "embed":
            continue
        cotangents = [t.input(f"y{q}") for q in range(len(out))]
        differentiable = [(kind, nodes) for kind, nodes in arguments
                          if kind != "i32"]
        wanted = [n for _, nodes in differentiable for n in nodes]
        bars = adjoint(t, out, cotangents, wanted)
        results, k = [], 0
        for kind, nodes in differentiable:
            results.append(bars[k:k + len(nodes)])
            k += len(nodes)
        print(emit(f"{stage}.b", arguments + [(len(out), cotangents)],
                   results, t))
        print()


def evaluate(t, values, outputs):
    """The traced nodes evaluated with the inputs `values` (name to float)."""
    m = Numbers()
    result = {}
    for n in t.nodes:
        if n.op == "const":
            result[n] = n.value
        elif n.op == "input":
            result[n] = values.get(n.value, 0.0)
        elif n.op == "sub":
            result[n] = result[n.args[0]] - result[n.args[1]]
        elif n.op == "neg":
            result[n] = -result[n.args[0]]
        else:
            result[n] = getattr(m, n.op)(*[result[a] for a in n.args])
    return [result[n] for n in outputs]


def check():
    """The traced VJP of each stage against central differences of its
    traced forward body, along random directions."""
    w = weights()
    rng = np.random.default_rng(1)
    worst = 0.0
    for stage in STAGES[1:]:
        t, arguments, out = trace(stage, w)
        cot = [t.input(f"y{q}") for q in range(len(out))]
        inputs = [n for kind, nodes in arguments if kind != "i32"
                  for n in nodes]
        bars = adjoint(t, out, cot, inputs)
        values = {n.value: rng.uniform(-0.5, 0.5) for n in inputs}
        if stage.startswith("edge"):
            values["r"] = 0.31
            values["d0"], values["d1"], values["d2"] = 0.1, -0.2, 0.2186321
        values["zi"], values["zj"] = 1.0, 0.0
        for q in range(len(out)):
            values[f"y{q}"] = rng.uniform(-1.0, 1.0)
        v = {n.value: rng.uniform(-1.0, 1.0) for n in inputs}
        lhs = 0.0
        h = 1.0e-5
        plus = dict(values)
        minus = dict(values)
        for n in inputs:
            plus[n.value] += h * v[n.value]
            minus[n.value] -= h * v[n.value]
        up = evaluate(t, plus, out)
        down = evaluate(t, minus, out)
        for q in range(len(out)):
            lhs += values[f"y{q}"] * (up[q] - down[q]) / (2.0 * h)
        rhs = sum(b * v[n.value] for b, n in
                  zip(evaluate(t, values, bars), inputs))
        worst = max(worst, abs(lhs - rhs) / max(abs(rhs), 1.0))
        print(f"{stage}: <y, J v> = {lhs:.15g} by differences, "
              f"<J^T y, v> = {rhs:.15g}")
    x, z = system()
    edges = pairs(x, EDGE * np.eye(3))
    print(f"largest relative difference: {worst:.3g}")
    print(f"pairs within the cutoff: {len(edges)} directed")


def gradient(w, x, cell, z):
    """The energy, the forces, and the virial by one reverse sweep over the
    whole system: the positions and a strain e are the inputs of a trace of
    the energy, with d_ij = (1 + e)(x_i - x_j + n H) at e = 0 (the strain of
    `derivatives`), so that F = -dU/dx and W_ba = -dU/de_ab to rounding,
    independently of the rules of md-differentiate."""
    t = Tracer()
    xs = [[t.input(f"x{i}_{c}") for c in range(3)] for i in range(len(x))]
    strain = [[t.input(f"e{a}{b}") for b in range(3)] for a in range(3)]
    h = [embed(t, w, t.const(float(z[i]))) for i in range(len(x))]
    edges = []
    for i, j, _, d in pairs(x, cell):
        shift = d - (x[i] - x[j])
        base = [t.add(t.sub(xs[i][c], xs[j][c]), t.const(shift[c]))
                for c in range(3)]
        dd = []
        for a in range(3):
            v = base[a]
            for b in range(3):
                v = t.add(v, t.mul(strain[a][b], base[b]))
            dd.append(v)
        r = t.sqrt(t.add(t.mul(dd[0], dd[0]),
                         t.add(t.mul(dd[1], dd[1]), t.mul(dd[2], dd[2]))))
        edges.append((i, j, r, dd))
    for layer in (1, 2):
        a = [[t.const(0.0)] * N for _ in range(len(x))]
        for i, j, r, dd in edges:
            msg = edge(t, w[f"edge{layer}"], r, dd, h[i], h[j],
                       t.const(float(z[i])), t.const(float(z[j])))
            a[i] = [t.add(p, q) for p, q in zip(a[i], msg)]
        h = [node(t, w[f"node{layer}"], h[i], a[i]) for i in range(len(x))]
    u = t.const(0.0)
    for i in range(len(x)):
        u = t.add(u, readout(t, w["readout"], h[i]))
    inputs = [n for row in xs for n in row] + \
        [n for row in strain for n in row]
    bars = adjoint(t, [u], [t.const(1.0)], inputs)
    values = {f"x{i}_{c}": x[i, c] for i in range(len(x)) for c in range(3)}
    results = evaluate(t, values, [u] + bars)
    forces = -np.array(results[1:1 + 3 * len(x)]).reshape(len(x), 3)
    virial = -np.array(results[1 + 3 * len(x):]).reshape(3, 3).T
    return results[0], forces, virial


def reference():
    w = weights()
    x, z = system()
    cell = EDGE * np.eye(3)
    u = energy(w, x, cell, z)
    traced, forces, virial = gradient(w, x, cell, z)
    fd_forces, fd_virial = derivatives(w, x, cell, z)
    print(f"energy of the trace against NumPy: {abs(traced - u):.3g}")
    print("forces of the sweep against differences: "
          f"{np.abs(forces - fd_forces).max():.3g} of "
          f"{np.abs(forces).max():.3g}")
    print("virial of the sweep against differences: "
          f"{np.abs(virial - fd_virial).max():.3g} of "
          f"{np.abs(virial).max():.3g}")
    print("positions:", ", ".join(repr(float(v)) for v in x.flatten()))
    print("species:", ", ".join(str(int(v)) for v in z))
    print(f"energy: {u!r}")
    print("forces:", ", ".join(repr(float(v)) for v in forces.flatten()))
    print("virial:", ", ".join(repr(float(v)) for v in virial.flatten()))
    print(f"asymmetry of the virial: {np.abs(virial - virial.T).max():.3g}")
    print(f"sum of the forces: {np.abs(forces.sum(axis=0)).max():.3g}")


if __name__ == "__main__":
    {"stages": stages, "reference": reference, "check": check}[sys.argv[1]]()

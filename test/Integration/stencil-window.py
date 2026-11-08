"""Independent cardinal-window, derivative and periodic-assignment checks.

The compiler uses Cox-de Boor recursion; this reference uses the distinct
truncated-power definition [deBoor2001], docs/references.md.
All new arithmetic here is reference f64, not a claim of Mixed execution.
"""
import ctypes
import itertools
import math
from pathlib import Path
import random
import sys

# Fixed before the first run: scalar values/analytic derivatives/duality.
ATOL = 2e-13
DTOL = 2e-12
FDTOL = 2e-9


def generate(path):
    functions = []
    for kind in ("value", "jvp", "v0", "v1", "v2",
                 "order2", "order3", "order5", "order6", "order7", "order8"):
        order = int(kind[5:]) if kind.startswith("order") else 4
        lines = [f"func.func @{kind}(%x: f64, %y: f64, %z: f64,"
                 " %a: f64, %b: f64, %c: f64) -> f64 {",
                 "  %u = vector.from_elements %x, %y, %z : vector<3xf64>",
                 "  %d = vector.from_elements %a, %b, %c : vector<3xf64>",
                 f"  %w = md.bspline_window %u order({order}) : vector<3xf64>"]
        if kind == "value" or kind.startswith("order"):
            result = "%w"
        elif kind == "jvp":
            lines.append("  %j = md.stencil_jvp %w wrt(0) %d :"
                         " f64, vector<3xf64> -> f64")
            result = "%j"
        else:
            lines += ["  %one = arith.constant 1.0 : f64",
                      "  %v = md.stencil_vjp %w wrt(0) %one :"
                      " f64, f64 -> vector<3xf64>",
                      f"  %p = vector.extract %v[{kind[1:]}] :"
                      " f64 from vector<3xf64>"]
            result = "%p"
        lines += [f"  return {result} : f64", "}"]
        functions.append("\n".join(lines))
    Path(path).write_text("\n".join(functions) + "\n")


def cardinal(x, derivative=0, order=4):
    power = order - 1 - derivative
    return sum((-1)**k * math.comb(order, k) * max(x-k, 0.0)**power
               for k in range(order+1)) / math.factorial(power)


def reference(u):
    weights = [cardinal(x) for x in u]
    value = math.prod(weights)
    gradient = [cardinal(u[a], 1) *
                math.prod(weights[b] for b in range(3) if a != b)
                for a in range(3)]
    return value, gradient


def check(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = [getattr(lib, name) for name in ("value", "jvp", "v0", "v1", "v2")]
    for function in funcs:
        function.argtypes = [ctypes.c_double] * 6
        function.restype = ctypes.c_double
    # Higher orders amplify cancellation in the independent power formula.
    # This primal-only tolerance is fixed before these new comparisons.
    primal_order_bound = 2e-10
    order_error = 0.0
    for order in (2, 3, 5, 6, 7, 8):
        function = getattr(lib, f"order{order}")
        function.argtypes = [ctypes.c_double] * 6
        function.restype = ctypes.c_double
        for n in range(17):
            u = (order*n/16, order*.37, order*.61)
            expected = math.prod(cardinal(x, order=order) for x in u)
            error = abs(function(*u, 0., 0., 0.)-expected)
            assert error <= primal_order_bound, (order, u, error)
            order_error = max(order_error, error)
    rng = random.Random(452)
    cases = [tuple(rng.uniform(0, 4) for _ in range(3)) for _ in range(40)]
    cases += [(x, 1.3, 2.2) for x in
              (-1e-7, 0, 1e-7, 1-1e-7, 1, 1+1e-7,
               2-1e-7, 2, 2+1e-7, 3, 4-1e-7, 4, 4+1e-7)]
    maxima = dict(value=0.0, jvp=0.0, vjp=0.0, duality=0.0, fd=0.0)
    for u in cases:
        direction = tuple(rng.uniform(-1, 1) for _ in range(3))
        value, grad = reference(u)
        actual = [function(*u, *direction) for function in funcs]
        errors = dict(value=abs(actual[0]-value),
                      jvp=abs(actual[1]-sum(a*b for a, b in zip(grad, direction))),
                      vjp=max(abs(a-b) for a, b in zip(actual[2:], grad)),
                      duality=abs(actual[1]-sum(a*b for a, b in zip(actual[2:], direction))))
        h = 1e-5
        plus = funcs[0](*(a+h*b for a, b in zip(u, direction)), *direction)
        minus = funcs[0](*(a-h*b for a, b in zip(u, direction)), *direction)
        errors["fd"] = abs(actual[1]-(plus-minus)/(2*h))
        for name, error in errors.items():
            maxima[name] = max(maxima[name], error)
            bound = FDTOL if name == "fd" else ATOL if name == "value" else DTOL
            assert math.isfinite(error) and error <= bound, (name, u, error, bound)

    # Construct the periodic relation independently of amplitudes. Each
    # particle has order^3 distinct wrapped indices, including zero charges.
    shape = (4, 5, 6)
    positions = [(-1e-8, .31, .7), (0., .31, .7), (1e-8, .31, .7),
                 (1-1e-8, .31, .7), (1., .31, .7), (1+1e-8, .31, .7)]
    amplitudes = [1., -2., 0., 1., -2., 0.]
    grid = [rng.uniform(-1, 1) for _ in range(math.prod(shape))]
    columns = []
    for pos in positions:
        fractional = [p-math.floor(p) for p in pos]
        scaled = [p*n for p, n in zip(fractional, shape)]
        starts = [math.floor(u)-3 for u in scaled]
        column = {}
        for offsets in itertools.product(range(4), repeat=3):
            index = tuple((s+o) % n for s, o, n in zip(starts, offsets, shape))
            flat = (index[0]*shape[1]+index[1])*shape[2]+index[2]
            u = tuple(v-(s+o) for v, s, o in zip(scaled, starts, offsets))
            actual = funcs[0](*u, 0., 0., 0.)
            expected, _ = reference(u)
            assert abs(actual-expected) <= ATOL
            assert flat not in column
            column[flat] = actual
        assert len(column) == 64
        assert abs(sum(column.values())-1.) <= ATOL
        columns.append(column)
    spread = [sum(q*column.get(k, 0.) for q, column in zip(amplitudes, columns))
              for k in range(math.prod(shape))]
    gather = [sum(grid[k]*w for k, w in column.items()) for column in columns]
    lhs = sum(a*b for a, b in zip(spread, grid))
    rhs = sum(a*b for a, b in zip(amplitudes, gather))
    assert abs(lhs-rhs) <= DTOL
    assert gather[2] != 0.0  # A zero charge retains its nonzero cotangent.
    assert max(abs(columns[0].get(k, 0)-columns[3].get(k, 0))
               for k in range(math.prod(shape))) <= ATOL
    print(f"{len(cases)} compiled window/derivative cases; "
          f"102 additional order-2..8 primal cases, max error {order_error:.3g}; "
          f"{len(columns)*64} periodic entries; max errors {maxima}; "
          f"assignment duality error {abs(lhs-rhs):.3g}")


if __name__ == "__main__":
    {"generate": generate, "check": check}[sys.argv[1]](sys.argv[2])

"""Seeded safe-domain scalar expressions, checked by energy differences.

The oracle is the primal expression evaluated at perturbed coordinates and
lambda, independently of the generated derivative rules. No derivative
values are embedded in this generator.
"""
import random
import sys
from pathlib import Path

out = Path(sys.argv[1])
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
count = int(sys.argv[3]) if len(sys.argv) > 3 else 12
lines = ["!vec = !md.field<@atoms, 3 x f64>",
         "!pairs = !md.relation<@atoms, 2, unordered>",
         "md.particle_set @atoms"]

def potential(index):
    ops = []
    serial = 0
    def op(name, *args):
        nonlocal serial
        result = f"%v{serial}"
        serial += 1
        ops.append(f"    {result} = {name} {', '.join(args)} : f64")
        return result
    def const(x):
        return op("arith.constant", f"{x:.17e}")
    one, tenth = const(1), const(.1)
    def expression(depth):
        if not depth:
            return rng.choice(["%r", "%lambda", const(rng.uniform(.2, .8))])
        x = expression(depth - 1)
        choice = rng.randrange(12)
        if choice < 4:
            return op(["math.sin", "math.cos", "math.tanh", "math.atan"][choice], x)
        if choice == 4:
            return op("math.exp", op("arith.mulf", tenth, x))
        positive = op("arith.addf", one, op("arith.mulf", x, x))
        if choice < 8:
            return op(["math.sqrt", "math.log", "math.absf"][choice - 5], positive)
        if choice == 8:
            return op("math.powf", positive, "%lambda")
        y = expression(depth - 1)
        if choice == 9:
            return op("arith.divf", y, positive)
        return op(["arith.addf", "arith.mulf"][choice - 10], x, y)
    result = expression(3)
    lines.extend([f"md.potential @random{index}(%x: !vec, %cell: !md.cell, %lambda: f64) -> f64 {{",
        "  %n = md.neighborhood %x, %cell cutoff(4.0) : !vec -> !pairs",
        "  %u = md.sum_relation %n, %x, %cell exchange(symmetric) {",
        "  ^bb0(%r: f64, %d: vector<3xf64>):", *ops,
        f"    md.yield {result} : f64", "  } : !pairs, !vec -> f64",
        "  md.return %u : f64", "}"])

for i in range(count):
    potential(i)
lines.extend(["func.func @main() {",
  "  %buf = memref.alloc() : memref<2x3xf64>",
  "  %zero = arith.constant 0 : index", "  %one = arith.constant 1 : index",
  "  %two = arith.constant 2 : index"])
for i, xyz in enumerate([[.2, .3, .4], [9., 1.4, 1.7]]):
    for j, value in enumerate(xyz):
        lines.extend([f"  %p{i}{j} = arith.constant {value:.17e} : f64",
                      f"  memref.store %p{i}{j}, %buf[%{'zero' if i == 0 else 'one'}, %{['zero', 'one', 'two'][j]}] : memref<2x3xf64>"])
lines.extend([
  "  %dyn = memref.cast %buf : memref<2x3xf64> to memref<?x3xf64>",
  "  %x = mdrt.from_buffer %dyn : memref<?x3xf64> to !vec",
  "  %edge = arith.constant 1.0e1 : f64",
  "  %cell = md.orthorhombic_cell %edge, %edge, %edge",
  "  %lambda = arith.constant 3.7e-1 : f64"])
for i in range(count):
    lines.extend([f"  %u{i}, %f{i}, %w{i}, %d{i} = md.evaluate @random{i}(%x, %cell, %lambda)",
        "    request [energy, forces, virial, derivative(2)]",
        "    : (!vec, !md.cell, f64) -> (f64, !vec, vector<9xf64>, f64)"])
lines.extend(["  return", "}"])
out.write_text("\n".join(lines) + "\n")
if len(sys.argv) > 4 and sys.argv[4] == "bad-rule":
    with out.open("a") as stream:
        stream.write('''
// Deliberately incorrect analytic results: the checker must reject them.
md.function @random0.energy_forces_virial_derivative2(%x: !vec, %cell: !md.cell, %lambda: f64)
    -> (f64, !vec, vector<9xf64>, f64) {
  %zero = arith.constant 0.0 : f64
  %w = arith.constant dense<0.0> : vector<9xf64>
  %f = md.map_particles gather(%x : !vec) {
  ^bb0(%p: vector<3xf64>):
    %z = arith.constant dense<0.0> : vector<3xf64>
    md.yield %z : vector<3xf64>
  } : !vec
  md.return %zero, %f, %w, %zero : f64, !vec, vector<9xf64>, f64
}
''')

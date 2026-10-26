#!/usr/bin/env python3
"""lift_couplings.py: the ops of the couplings of dyn in a module that
`mdir emit` printed.

Replaces, in the text of a module as the driver writes it today, each sum,
loop, and call of the host that an op of a coupling of `dyn` expands to
(docs/dynamics.md, Section 2.2.3) with that op:

  md.sum_particles of m v                     dyn.momentum
  md.sum_particles of (m / 2) v ⊙ v, or of
    (m (v · v)) / 2                           dyn.kinetic
  md.sum_particles of F ⊙ F / m, times Δt²/8  dyn.half_step_excess
  md.map_particles of the velocities alone
    that subtracts and multiplies             dyn.shift, dyn.scale
  md.map_particles of Philox                  dyn.noise
  func.call @mdrtBussiFactor                  dyn.velocity_rescaling
  func.call @mdrtNoseHooverFactor             dyn.nose_hoover
  func.call @mdrtAddBath                      dyn.to_bath

`dyn-expand-couplings` then expands the ops; the module it gives must be
that of the input, which shows that each op expands to the text the driver
writes (#541, step 2). The lifting reads the shapes of the text and does
not check the arithmetic: a shape that it lifts and that the expansion does
not give back shows as a difference.

The attributes of `dyn.nose_hoover` are not in the text (the chain is
configured on the host) and take placeholders, which the expansion does
not read.

  lift_couplings.py MODULE [-o OUTPUT] [--counts]
"""

import argparse
import re
import sys

SSA = r"%[A-Za-z0-9_$.#-]+"
HEADER = re.compile(
    r"^(\s*)(" + SSA + r") = md\.(sum_particles|map_particles) gather\((.*) : (.*)\) \{$")
BLOCK = re.compile(r"^\s*\^bb0\((.*)\):$")
END = re.compile(r"^(\s*)\} : (.*)$")
DEFINE = re.compile(r"^\s*(" + SSA + r"(?:, " + SSA + r")*) = (.*)$")
NAME = re.compile(SSA)


def split_top(text):
    """The comma-separated items of `text`, outside of <> and ()."""
    items, depth, current = [], 0, ""
    for char in text:
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth -= 1
        if char == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        items.append(current.strip())
    return items


def normalize(body, arguments):
    """The lines of a kernel with the arguments as $i, the values it
    defines as #i, and the values from outside as @name; and the names of
    the values from outside, in order."""
    names = {name: "$%d" % i for i, name in enumerate(arguments)}
    outer = []
    lines = []
    count = 0
    for line in body:
        text = line.strip()
        match = DEFINE.match(text)
        defined = []
        rest = text
        if match:
            defined = [d.strip() for d in match.group(1).split(",")]
            rest = match.group(2)

        def rename(m):
            name = m.group(0)
            if name in names:
                return names[name]
            outer.append(name)
            return "@"
        rest = NAME.sub(rename, rest)
        for d in defined:
            names[d] = "#%d" % count
            count += 1
        lines.append((", ".join(names[d] for d in defined) + " = " + rest)
                     if defined else rest)
    return lines, outer


MOMENTUM = [
    "#0 = vector.broadcast $1 : f64 to vector<3xf64>",
    "#1 = arith.mulf #0, $0 : vector<3xf64>",
    "md.yield #1 : vector<3xf64>",
]
KINETIC_AXES = [
    "#0 = arith.constant 5.000000e-01 : f64",
    "#1 = arith.mulf #0, $1 : f64",
    "#2 = vector.broadcast #1 : f64 to vector<3xf64>",
    "#3 = arith.mulf $0, $0 : vector<3xf64>",
    "#4 = arith.mulf #2, #3 : vector<3xf64>",
    "md.yield #4 : vector<3xf64>",
]
KINETIC = [
    "#0 = arith.constant 5.000000e-01 : f64",
    "#1 = arith.mulf $0, $0 : vector<3xf64>",
    "#2 = vector.reduction <add>, #1 : vector<3xf64> into f64",
    "#3 = arith.mulf $1, #2 : f64",
    "#4 = arith.mulf #0, #3 : f64",
    "md.yield #4 : f64",
]
EXCESS = [
    "#0 = arith.mulf $0, $0 : vector<3xf64>",
    "#1 = arith.constant 0.000000e+00 : f64",
    "#2 = arith.constant 1.000000e+00 : f64",
    "#3 = arith.cmpf oeq, $1, #1 : f64",
    "#4 = arith.select #3, #2, $1 : f64",
    "#5 = arith.divf #2, #4 : f64",
    "#6 = arith.select #3, #1, #5 : f64",
    "#7 = vector.broadcast #6 : f64 to vector<3xf64>",
    "#8 = arith.mulf #0, #7 : vector<3xf64>",
    "md.yield #8 : vector<3xf64>",
]


class Lifter:
    def __init__(self, lines):
        self.lines = lines
        self.out = []
        self.counts = {}

    def emit(self, op, line):
        self.counts[op] = self.counts.get(op, 0) + 1
        self.out.append(line)

    def constant(self, name, before):
        """The value and type of the constant `name` that the lines before
        `before` define last."""
        pattern = re.compile(r"^\s*" + re.escape(name) + r" = arith\.constant (\S+) : (\S+)$")
        for line in reversed(self.lines[:before]):
            match = pattern.match(line)
            if match:
                return match.group(1)
        raise ValueError("no constant " + name)

    def lift_loop(self, i):
        """Lifts the loop at line i; returns the next line, or None."""
        header = HEADER.match(self.lines[i])
        if not header:
            return None
        indent, result, kind, gathered, types = header.groups()
        gathered = [g.strip() for g in gathered.split(",")]
        types = split_top(types)
        block = BLOCK.match(self.lines[i + 1])
        arguments = [a.split(":")[0].strip() for a in split_top(block.group(1))]
        j = i + 2
        while not (END.match(self.lines[j]) and
                   len(END.match(self.lines[j]).group(1)) == len(indent)):
            j += 1
        body = self.lines[i + 2:j]
        result_type = END.match(self.lines[j]).group(2)
        lines, outer = normalize(body, arguments)
        field = types[0] if types else ""
        if kind == "sum_particles" and len(gathered) == 2:
            v, m = gathered
            if lines == MOMENTUM:
                self.emit("dyn.momentum", "%s%s = dyn.momentum %s, %s : %s" %
                          (indent, result, v, m, field))
                return j + 1
            if lines == KINETIC_AXES or lines == KINETIC:
                self.emit("dyn.kinetic", "%s%s = dyn.kinetic %s, %s : %s -> %s" %
                          (indent, result, v, m, field, result_type))
                return j + 1
            if lines == EXCESS:
                return self.lift_excess(i, j, indent, result, v, m, field)
        if kind == "map_particles" and len(gathered) == 1:
            lifted = self.lift_velocities(indent, result, gathered[0], field, lines, outer)
            if lifted:
                for line in lifted:
                    self.emit(line.split("=")[1].split()[0], line)
                return j + 1
        if (kind == "map_particles" and len(gathered) == 3 and
                any("arith.mului_extended" in l for l in body)):
            return self.lift_noise(i, j, indent, result, gathered, types, body)
        return None

    def lift_excess(self, i, j, indent, result, f, m, field):
        """E = Δt²/8 Σ F ⊙ F / m: the sum and the five lines after it."""
        after = [l.strip() for l in self.lines[j + 1:j + 6]]
        if len(after) < 5:
            return None
        patterns = [
            r"(%\S+) = arith\.mulf (%\S+), (%\S+) : f64",
            r"(%\S+) = arith\.constant 1\.250000e-01 : f64",
            r"(%\S+) = arith\.mulf (%\S+), (%\S+) : f64",
            r"(%\S+) = vector\.broadcast (%\S+) : f64 to vector<3xf64>",
            r"(%\S+) = arith\.mulf (%\S+), (%\S+) : vector<3xf64>",
        ]
        matches = [re.fullmatch(p, l) for p, l in zip(patterns, after)]
        if not all(matches):
            return None
        d2, eighth, scale, splat, product = matches
        dt = d2.group(2)
        if (d2.group(3) != dt or scale.group(2) != d2.group(1) or
                scale.group(3) != eighth.group(1) or
                splat.group(2) != scale.group(1) or
                product.group(2) != result or
                product.group(3) != splat.group(1)):
            return None
        # A value between with another use would be undefined in the
        # lifted module, which then does not parse.
        self.emit("dyn.half_step_excess",
                  "%s%s = dyn.half_step_excess %s, %s, %s : %s" %
                  (indent, product.group(1), f, m, dt, field))
        return j + 6

    def lift_velocities(self, indent, result, v, field, lines, outer):
        """A chain of dyn.shift and dyn.scale, or None."""
        ops = []
        current = "$0"
        k = 0
        index = 0
        defined = 0
        while k < len(lines):
            line = lines[k]
            sub = re.fullmatch(r"#(\d+) = arith\.subf (\$0|#\d+), @ : vector<3xf64>", line)
            splat = re.fullmatch(r"#(\d+) = vector\.broadcast @ : f64 to vector<3xf64>", line)
            mul = re.fullmatch(r"#(\d+) = arith\.mulf @, (\$0|#\d+) : vector<3xf64>", line)
            if sub and sub.group(2) == current:
                ops.append(("dyn.shift", outer[index], "vector<3xf64>"))
                index += 1
                current = "#" + sub.group(1)
                k += 1
                continue
            if splat and k + 1 < len(lines):
                product = re.fullmatch(r"#(\d+) = arith\.mulf #" + splat.group(1) +
                                       r", (\$0|#\d+) : vector<3xf64>", lines[k + 1])
                if product and product.group(2) == current:
                    ops.append(("dyn.scale", outer[index], "f64"))
                    index += 1
                    current = "#" + product.group(1)
                    k += 2
                    continue
            if mul and mul.group(2) == current:
                ops.append(("dyn.scale", outer[index], "vector<3xf64>"))
                index += 1
                current = "#" + mul.group(1)
                k += 1
                continue
            if line == "md.yield %s : vector<3xf64>" % current and k == len(lines) - 1 and ops:
                break
            return None
        out = []
        value = v
        for n, (op, operand, kind) in enumerate(ops):
            name = result if n == len(ops) - 1 else "%%lift_%s_%d" % (result[1:].replace("#", "_"), n)
            if op == "dyn.shift":
                out.append("%s%s = dyn.shift %s, %s : %s" % (indent, name, value, operand, field))
            else:
                out.append("%s%s = dyn.scale %s, %s : %s, %s" %
                           (indent, name, value, operand, field, kind))
            value = name
        return out

    def lift_noise(self, i, j, indent, result, gathered, types, body):
        text = "\n".join(body)
        step = re.search(r"= arith\.shrui (%\S+), ", text).group(1)
        words = [int(w) for w in re.findall(r"= arith\.constant (-?\d+) : i32", text)]
        # c3, the two multipliers, then the key of round 0.
        k0, k1 = words[3] & 0xFFFFFFFF, words[4] & 0xFFFFFFFF
        seed = (k1 << 32) | k0
        if seed >= 1 << 63:
            seed -= 1 << 64
        decay = re.search(r"arith\.constant dense<(\S+)> : vector<3xf64>", text).group(1)
        spread = None
        for a, b in zip(body, body[1:]):
            if "= math.sqrt" in b:
                match = re.search(r"= arith\.constant (\S+) : f64$", a.strip())
                if match:
                    spread = match.group(1)
        if spread is None:
            return None
        v, m, ids = gathered
        self.emit("dyn.noise",
                  "%s%s = dyn.noise %s, %s, %s, %s {decay = %s : f64, seed = %d : i64, "
                  "spread = %s : f64} : %s, %s" %
                  (indent, result, v, m, step, ids, decay, seed, spread, types[0], types[2]))
        return j + 1

    def lift_call(self, i):
        line = self.lines[i]
        match = re.match(r"^(\s*)(%\S+) = func\.call @mdrtBussiFactor\((.*)\) : ", line)
        if match:
            indent, result, operands = match.groups()
            seed, step, kinetic, target, freedom, decay = [o.strip() for o in operands.split(",")]
            values = [self.constant(n, i) for n in (seed, target, freedom, decay)]
            self.emit("dyn.velocity_rescaling",
                      "%s%s = dyn.velocity_rescaling %s, %s {decay = %s : f64, freedom = %s : "
                      "f64, seed = %s : i64, target_kinetic = %s : f64} : f64" %
                      (indent, result, kinetic, step, values[3], values[2], values[0],
                       values[1]))
            return True
        match = re.match(r"^(\s*)(%\S+) = func\.call @mdrtNoseHooverFactor\((%\S+), (%\S+)\) : ",
                         line)
        if match:
            indent, result, step, kinetic = match.groups()
            self.emit("dyn.nose_hoover",
                      "%s%s = dyn.nose_hoover %s, %s {chain_length = 1 : i64, freedom = 1.0 : "
                      "f64, period = 1.0 : f64, temperature = 1.0 : f64, time_constant = 1.0 : "
                      "f64} : f64" % (indent, result, kinetic, step))
            return True
        match = re.match(r"^(\s*)func\.call @mdrtAddBath\((%\S+)\) : \(f64\) -> \(\)$", line)
        if match:
            indent, energy = match.groups()
            self.emit("dyn.to_bath", "%sdyn.to_bath %s : f64" % (indent, energy))
            return True
        return False

    def run(self):
        i = 0
        while i < len(self.lines):
            following = self.lift_loop(i) if "md." in self.lines[i] else None
            if following is not None:
                i = following
                continue
            if "func.call @mdrt" in self.lines[i] and self.lift_call(i):
                i += 1
                continue
            self.out.append(self.lines[i])
            i += 1
        return claim_noise(self.out)


CLAIMS = re.compile(r"attributes \{(.*)\} \{$")


def claim_noise(lines):
    """The claims of a program whose body has a dyn.noise, as the verifier
    of dyn.program requires them (docs/dynamics.md, Section 2.2.4): it
    requires the temperature and provides thermostatting, and neither
    symplectic nor time_reversible."""
    out = list(lines)
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("dyn.program @"):
            continue
        indent = line[:len(line) - len(line.lstrip())]
        j = i + 1
        while lines[j] != indent + "}":
            j += 1
        if not any("= dyn.noise " in l for l in lines[i:j]):
            continue
        match = CLAIMS.search(line)
        items = split_top(match.group(1)) if match else []
        items = [item for item in items
                 if not item.startswith("provides =") and
                 not item.startswith("requires =")]
        items.append('provides = ["thermostatting"]')
        items.append('requires = ["temperature"]')
        claims = "attributes {" + ", ".join(sorted(items)) + "} {"
        out[i] = (line[:match.start()] + claims if match
                  else line[:-1] + claims)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("module")
    parser.add_argument("-o", "--output")
    parser.add_argument("--counts", action="store_true",
                        help="print the number of each op lifted on stderr")
    args = parser.parse_args()
    with open(args.module) as f:
        lines = f.read().split("\n")
    lifter = Lifter(lines)
    out = "\n".join(lifter.run())
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    if args.counts:
        for op in sorted(lifter.counts):
            sys.stderr.write("%s: %d\n" % (op, lifter.counts[op]))


if __name__ == "__main__":
    main()

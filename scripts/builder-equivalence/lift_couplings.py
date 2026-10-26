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
  memref.load of the cell                     dyn.load_cell
  the arithmetic of the pressure              dyn.pressure
  func.call @mdrtBarostatStrain...            dyn.cell_rescaling
  the scaled cell, its stores, mdrtSetBox     dyn.scale_cell, dyn.store_cell
  the sums of the centers of the groups       dyn.group_kinetic
  the scaled positions and groups             dyn.scale_positions
  the work of a scaling, `trotter`, `trotter_before`, and `first_order`
                                              dyn.scaling_work

`dyn-expand-couplings` then expands the ops; the module it gives must be
that of the input, which shows that each op expands to the text the driver
writes (#541, step 2). The lifting reads the shapes of the text and does
not check the arithmetic: a shape that it lifts and that the expansion does
not give back shows as a difference.

An op of the barostat takes the place of the first line that it replaces;
the lines of pure arithmetic between that do not depend on a call that it
replaces stay after it. The expansion computes them again, and the cse of
the pipeline keeps the first of the two, so the lowered program is that of
the text where the expansion is.

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

    # The barostat (step 2b of #541).

    def value(self, name, before):
        """The value of the constant `name`, or None."""
        try:
            return self.constant(name, before)
        except ValueError:
            return None

    def definition(self, name, before):
        """The text after `name = ` of the last line before `before` that
        defines `name`, or None."""
        pattern = re.compile(r"^\s*" + re.escape(name) + r" = (.*)$")
        for line in reversed(self.lines[:before]):
            match = pattern.match(line)
            if match:
                return match.group(1)
            if line.startswith("  func.func") or line.startswith("  dyn.program"):
                break
        return None

    def edges_of(self, volume, before):
        """The edges L0, L1, L2 of the volume `volume` = (L0 L1) L2."""
        text = self.definition(volume, before)
        match = text and re.fullmatch(r"arith\.mulf (%\S+), (%\S+) : f64", text)
        if not match:
            return None
        xy = self.definition(match.group(1), before)
        first = xy and re.fullmatch(r"arith\.mulf (%\S+), (%\S+) : f64", xy)
        if not first:
            return None
        return first.group(1), first.group(2), match.group(2)

    def replace(self, i, j, ops, results):
        """Lines i to j (exclusive) by `ops`; the lines of pure arithmetic
        between that depend on no removed value stay after them."""
        removed = set(results)
        kept = []
        for line in self.lines[i:j]:
            match = DEFINE.match(line.strip())
            defined = [d.strip() for d in match.group(1).split(",")] if match else []
            uses = set(NAME.findall(match.group(2) if match else line))
            if (not defined or "func.call" in line or
                    any(d in removed for d in defined) or uses & removed):
                removed.update(defined)
                continue
            kept.append(line)
        for op, line in ops:
            self.emit(op, line)
        self.out.extend(kept)
        return j

    def match_lines(self, i, patterns):
        """The matches of `patterns` on the lines from i, or None."""
        found = []
        for k, pattern in enumerate(patterns):
            if i + k >= len(self.lines):
                return None
            match = re.fullmatch(pattern, self.lines[i + k].strip())
            if not match:
                return None
            found.append(match)
        return found

    def lift_load_cell(self, i):
        first = re.fullmatch(r"(%\S+) = memref\.load (%\S+)\[(%\S+)\] : memref<([36])xf64>",
                             self.lines[i].strip())
        if not first:
            return None
        memory, count = first.group(2), int(first.group(4))
        indent = self.lines[i][:len(self.lines[i]) - len(self.lines[i].lstrip())]
        names = []
        for k in range(count):
            match = re.fullmatch(r"(%\S+) = memref\.load " + re.escape(memory) +
                                 r"\[(%\S+)\] : memref<" + str(count) + "xf64>",
                                 self.lines[i + k].strip())
            if not match or self.value(match.group(2), i) != str(k):
                return None
            names.append(match.group(1))
        self.emit("dyn.load_cell", "%s%s = dyn.load_cell %s : memref<%dxf64> -> %s" %
                  (indent, ", ".join(names), memory, count, ", ".join(["f64"] * count)))
        return i + count

    def is_value(self, name, i, value):
        text = self.value(name, i)
        try:
            return text is not None and float(text) == value
        except ValueError:
            return False

    def lift_pressure(self, i):
        f = r"(%\S+)"
        found = self.match_lines(i, [
            f + r" = arith\.addf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.addf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.divf " + f + r", " + f + r" : f64",
            f + r" = arith\.mulf " + f + r", " + f + r" : f64",
            f + r" = vector\.broadcast " + f + r" : f64 to vector<3xf64>",
            f + r" = arith\.addf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.divf " + f + r", " + f + r" : f64",
            f + r" = vector\.broadcast " + f + r" : f64 to vector<3xf64>",
            f + r" = arith\.mulf " + f + r", " + f + r" : vector<3xf64>",
        ])
        if not found:
            return None
        k2, w0, wc, wc3, wcb, w, pc, pcb, p3 = found
        kinetic, volume = k2.group(2), wc.group(3)
        if not (k2.group(3) == kinetic and w0.group(2) == k2.group(1) and
                wc3.group(2) == wc.group(1) and self.is_value(wc3.group(3), i, 1.0 / 3.0) and
                wcb.group(2) == wc3.group(1) and w.group(2) == w0.group(1) and
                w.group(3) == wcb.group(1) and pc.group(3) == volume and
                self.is_value(pc.group(2), i, 16.6053906717) and
                pcb.group(2) == pc.group(1) and p3.group(2) == w.group(1) and
                p3.group(3) == pcb.group(1)):
            return None
        edges = self.edges_of(volume, i)
        if not edges:
            return None
        indent = self.lines[i][:len(self.lines[i]) - len(self.lines[i].lstrip())]
        line = "%s%s = dyn.pressure %s, %s, %s, %s, %s, %s : vector<3xf64>" % (
            indent, p3.group(1), kinetic, w0.group(3), *edges, wc.group(2))
        return self.replace(i, i + 9, [("dyn.pressure", line)], [p3.group(1)])

    def lift_strain(self, i):
        """The calls of the strain and the arithmetic about them."""
        f = r"(%\S+)"
        call = r" = func\.call @(mdrtBarostatStrain\w*)\((.*)\) : .*"
        indent = self.lines[i][:len(self.lines[i]) - len(self.lines[i].lstrip())]
        third = 1.0 / 3.0
        # Isotropic.
        found = self.match_lines(i, [
            f + r" = vector\.reduction <add>, " + f + r" : vector<3xf64> into f64",
            f + r" = arith\.mulf " + f + r", " + f + r" : f64",
            f + call,
            f + r" = arith\.mulf " + f + r", " + f + r" : f64",
            f + r" = vector\.broadcast " + f + r" : f64 to vector<3xf64>",
        ])
        if found and found[2].group(2) == "mdrtBarostatStrain":
            reduce, mean, strain, scaled, splat = found
            args = [a.strip() for a in strain.group(3).split(",")]
            seed, step, pressure, target, volume, kt, beta, rate = args
            if not (mean.group(2) == reduce.group(1) and self.is_value(mean.group(3), i, third)
                    and pressure == mean.group(1) and scaled.group(2) == strain.group(1)
                    and self.is_value(scaled.group(3), i, third)
                    and splat.group(2) == scaled.group(1)):
                return None
            edges = self.edges_of(volume, i)
            if not edges:
                return None
            attrs = self.strain_attrs(i, "isotropic", seed, target, [beta], rate, kt, None)
            line = "%s%s = dyn.cell_rescaling %s, %s, %s, %s, %s %s : vector<3xf64>" % (
                indent, splat.group(1), reduce.group(2), *edges, step, attrs)
            return self.replace(i, i + 5, [("dyn.cell_rescaling", line)], [splat.group(1)])
        # Semi-isotropic.
        found = self.match_lines(i, [
            f + r" = vector\.extract " + f + r"\[0\] : f64 from vector<3xf64>",
            f + r" = vector\.extract " + f + r"\[1\] : f64 from vector<3xf64>",
            f + r" = vector\.extract " + f + r"\[2\] : f64 from vector<3xf64>",
            f + r" = arith\.addf " + f + r", " + f + r" : f64",
            f + r" = arith\.mulf " + f + r", " + f + r" : f64",
            f + call,
            f + call,
            f + r" = arith\.mulf " + f + r", " + f + r" : f64",
            f + r" = vector\.from_elements " + f + r", " + f + r", " + f + r" : vector<3xf64>",
        ])
        if found and found[5].group(2) == "mdrtBarostatStrainArea":
            px, py, pz, s, area, xy, z, half, out = found
            a = [x.strip() for x in xy.group(3).split(",")]
            h = [x.strip() for x in z.group(3).split(",")]
            seed, step, pxy, target, volume, l2, kt, beta, rate, tension = a
            if not (px.group(2) == py.group(2) == pz.group(2) and pxy == area.group(1)
                    and self.is_value(area.group(3), i, 0.5) and h[2] == pz.group(1)
                    and h[0] == seed and h[1] == step and out.group(2) == half.group(1)
                    and out.group(3) == half.group(1) and out.group(4) == z.group(1)
                    and half.group(2) == xy.group(1)):
                return None
            edges = self.edges_of(volume, i)
            if not edges or edges[2] != l2:
                return None
            attrs = self.strain_attrs(i, "semi_isotropic", seed, target, [beta, h[6]], rate,
                                      kt, tension)
            line = "%s%s = dyn.cell_rescaling %s, %s, %s, %s, %s %s : vector<3xf64>" % (
                indent, out.group(1), px.group(2), *edges, step, attrs)
            return self.replace(i, i + 9, [("dyn.cell_rescaling", line)], [out.group(1)])
        # Anisotropic.
        patterns = []
        for k in range(3):
            patterns += [f + r" = vector\.extract " + f + r"\[%d\] : f64 from vector<3xf64>" % k,
                         f + r" = arith\.constant %d : i64" % k, f + call]
        patterns.append(f + r" = vector\.from_elements " + f + r", " + f + r", " + f +
                        r" : vector<3xf64>")
        found = self.match_lines(i, patterns)
        if found and found[2].group(2) == "mdrtBarostatStrainAxis":
            betas = []
            calls = [found[3 * k + 2] for k in range(3)]
            args = [[x.strip() for x in c.group(3).split(",")] for c in calls]
            seed, step, _, _, target, volume, kt, _, rate = args[0]
            for k in range(3):
                a = args[k]
                if not (a[0] == seed and a[1] == step and a[2] == found[3 * k + 1].group(1)
                        and a[3] == found[3 * k].group(1) and a[4] == target and
                        a[5] == volume and a[6] == kt and a[8] == rate and
                        found[3 * k].group(2) == found[0].group(2)):
                    return None
                betas.append(a[7])
            out = found[9]
            if [out.group(2), out.group(3), out.group(4)] != [c.group(1) for c in calls]:
                return None
            edges = self.edges_of(volume, i)
            if not edges:
                return None
            attrs = self.strain_attrs(i, "anisotropic", seed, target, betas, rate, kt, None)
            line = "%s%s = dyn.cell_rescaling %s, %s, %s, %s, %s %s : vector<3xf64>" % (
                indent, out.group(1), found[0].group(2), *edges, step, attrs)
            return self.replace(i, i + 10, [("dyn.cell_rescaling", line)], [out.group(1)])
        return None

    def strain_attrs(self, i, coupling, seed, target, betas, rate, kt, tension):
        values = [self.value(n, i) for n in [seed, target, rate, kt] + betas]
        if any(v is None for v in values):
            raise ValueError("a constant of the strain is not in the text")
        seed, target, rate, kt = values[:4]
        text = ("{compressibility = [%s], coupling = \"%s\", kT = %s : f64, rate = %s : f64, "
                "seed = %s : i64, target = %s : f64" %
                (", ".join(values[4:]), coupling, kt, rate, seed, target))
        if tension is not None:
            text += ", tension = %s : f64" % self.value(tension, i)
        return text + "}"

    def loop_at(self, i):
        """The loop whose header is line i: (result, op, the text after the
        op's name up to its kernel, the index of its last line, the text
        after `} : `), or None."""
        match = re.match(r"^(\s*)(%\S+) = md\.(\w+) (.*) \{$", self.lines[i])
        if not match:
            return None
        indent = match.group(1)
        j = i + 1
        while j < len(self.lines):
            end = re.match(r"^(\s*)\} : (.*)$", self.lines[j])
            if end and len(end.group(1)) == len(indent):
                return match.group(2), match.group(3), match.group(4), j, end.group(2)
            j += 1
        return None

    def kernel_of(self, i, j):
        """The normalized lines of the kernel of the loop from line i to j."""
        block = BLOCK.match(self.lines[i + 1])
        arguments = [a.split(":")[0].strip() for a in split_top(block.group(1))]
        return normalize(self.lines[i + 2:j], arguments)[0]

    def lift_group_kinetic(self, i):
        """The kinetic energy of the centers of the groups."""
        first = self.loop_at(i)
        if not first or first[1] != "sum_particles":
            return None
        if self.kernel_of(i, first[3]) != KINETIC_AXES:
            return None
        kinetic = first[0]
        v, m = [g.strip() for g in re.match(r"gather\((.*) :", first[2]).group(1).split(",")]
        k = first[3] + 1
        second = self.loop_at(k)
        if not second or second[1] != "sum_particles" or \
                self.kernel_of(k, second[3]) != EXCESS:
            return None
        force = second[0]
        f = re.match(r"gather\((%\S+), ", second[2]).group(1)
        k = second[3] + 1
        groups, types, x, cell = [], [], None, None
        f_ = r"(%\S+)"
        while True:
            loop = self.loop_at(k)
            if not loop or loop[1] != "sum_tuples":
                break
            head = re.match(f_ + r", " + f_ + r", " + f_ +
                            r" coordinates\(displacement\(1, 0\)\) gather\(" + f_ + r", " +
                            f_ + r" :", loop[2])
            if not head or head.group(4) != v or head.group(5) != m:
                return None
            relation, x0, cell0 = head.group(1), head.group(2), head.group(3)
            after = self.match_lines(loop[3] + 1, [
                f_ + r" = arith\.constant dense<5\.000000e-01> : vector<3xf64>",
                f_ + r" = arith\.mulf " + f_ + r", " + f_ + r" : vector<3xf64>",
                f_ + r" = arith\.subf " + f_ + r", " + f_ + r" : vector<3xf64>"])
            if not after or after[1].group(2) != loop[0] or after[2].group(2) != kinetic:
                return None
            kinetic = after[2].group(1)
            k = loop[3] + 4
            forces = self.loop_at(k)
            if not forces or forces[1] != "sum_tuples" or not forces[2].startswith(
                    "%s, %s, %s coordinates(displacement(1, 0)) gather(%s, %s :" %
                    (relation, x0, cell0, f, m)):
                return None
            sub = self.match_lines(forces[3] + 1, [
                f_ + r" = arith\.subf " + f_ + r", " + f_ + r" : vector<3xf64>"])
            if not sub or sub[0].group(2) != force or sub[0].group(3) != forces[0]:
                return None
            force = sub[0].group(1)
            k = forces[3] + 2
            if x not in (None, x0) or cell not in (None, cell0):
                return None
            x, cell = x0, cell0
            groups.append(relation)
            types.append(split_top(loop[4].split(" -> ")[0])[0])
        if not groups:
            return None
        center = None
        momentum = self.loop_at(k)
        if momentum and momentum[1] == "sum_particles" and \
                self.kernel_of(k, momentum[3]) == MOMENTUM:
            com = self.match_lines(momentum[3] + 1, [
                f_ + r" = arith\.divf " + f_ + r", " + f_ + r" : vector<3xf64>",
                f_ + r" = arith\.mulf " + f_ + r", " + f_ + r" : vector<3xf64>",
                f_ + r" = arith\.constant dense<5\.000000e-01> : vector<3xf64>",
                f_ + r" = arith\.mulf " + f_ + r", " + f_ + r" : vector<3xf64>",
                f_ + r" = arith\.subf " + f_ + r", " + f_ + r" : vector<3xf64>"])
            if not com or com[4].group(2) != kinetic:
                return None
            center = com[0].group(3)
            kinetic = com[4].group(1)
            k = momentum[3] + 6
        tail = self.match_lines(k, [
            f_ + r" = arith\.mulf " + f_ + r", " + f_ + r" : f64",
            f_ + r" = arith\.constant 1\.250000e-01 : f64",
            f_ + r" = arith\.mulf " + f_ + r", " + f_ + r" : f64",
            f_ + r" = vector\.broadcast " + f_ + r" : f64 to vector<3xf64>",
            f_ + r" = arith\.mulf " + f_ + r", " + f_ + r" : vector<3xf64>",
            f_ + r" = arith\.addf " + f_ + r", " + f_ + r" : vector<3xf64>"])
        if not tail or tail[4].group(2) != force or tail[5].group(2) != kinetic or \
                tail[5].group(3) != tail[4].group(1):
            return None
        dt = tail[0].group(2)
        indent = self.lines[i][:len(self.lines[i]) - len(self.lines[i].lstrip())]
        line = "%s%s, %s, %s = dyn.group_kinetic %s, %s, %s, %s, %s, %s groups(%s : %s)%s : %s" % (
            indent, kinetic, tail[4].group(1), tail[5].group(1), x, v, f, m, cell, dt,
            ", ".join(groups), ", ".join(types),
            " center(%s)" % center if center else "", "!md.field<@atoms, 3 x f64>")
        self.emit("dyn.group_kinetic", line)
        return k + 6

    def lift_scale_positions(self, i):
        """The positions scaled by μ, and each rigid group by its center."""
        first = self.loop_at(i)
        if not first or first[1] != "map_particles":
            return None
        head = re.fullmatch(r"gather\((%\S+) : (.*)\)", first[2])
        body = self.lines[i + 2:first[3]]
        if not head or len(body) != 2:
            return None
        mul = re.fullmatch(r"(%\S+) = arith\.mulf (%\S+), (%\S+) : vector<3xf64>",
                           body[0].strip())
        block = BLOCK.match(self.lines[i + 1])
        argument = block.group(1).split(":")[0].strip()
        if not mul or mul.group(3) != argument:
            return None
        x, mu, field = head.group(1), mul.group(2), head.group(2)
        scaled, k, groups, types, masses, cell = first[0], first[3] + 1, [], [], None, None
        while True:
            change = self.loop_at(k)
            if not change or change[1] != "gather_tuples":
                break
            m = re.match(r"(%\S+), " + re.escape(x) + r", (%\S+) coordinates\(.*\) gather\((%\S+) :",
                         change[2])
            if not m:
                break
            moved = self.loop_at(change[3] + 1)
            if not moved or moved[1] != "map_particles" or not moved[2].startswith(
                    "gather(%s, %s :" % (scaled, change[0])):
                break
            groups.append(m.group(1))
            types.append(split_top(change[4].split(" -> ")[0])[0])
            cell, masses = m.group(2), m.group(3)
            scaled = moved[0]
            k = moved[3] + 1
        indent = self.lines[i][:len(self.lines[i]) - len(self.lines[i].lstrip())]
        if not groups:
            # No rigid groups: the mass and the cell are not in the text, and
            # the scaling alone is not lifted.
            return None
        line = "%s%s = dyn.scale_positions %s, %s, %s, %s groups(%s : %s) : %s" % (
            indent, scaled, x, mu, masses, cell, ", ".join(groups), ", ".join(types), field)
        self.emit("dyn.scale_positions", line)
        return k

    def splat_of(self, name, i):
        """`x` of `name` = vector.broadcast x, or None."""
        text = self.definition(name, i)
        m = text and re.fullmatch(r"vector\.broadcast (%\S+) : f64 to vector<3xf64>", text)
        return m.group(1) if m else None

    def share_of(self, name, i):
        """(C, V) of `name` = broadcast((C / V) * ⅓), or None."""
        x = self.splat_of(name, i)
        text = x and self.definition(x, i)
        m = text and re.fullmatch(r"arith\.mulf (%\S+), (%\S+) : f64", text)
        if not m or not self.is_value(m.group(2), i, 1.0 / 3.0):
            return None
        text = self.definition(m.group(1), i)
        d = text and re.fullmatch(r"arith\.divf (%\S+), (%\S+) : f64", text)
        return (d.group(1), d.group(2)) if d else None

    def log_of_inverse(self, inverse, i):
        """ln μ of μ⁻¹ = 1 / exp(ln μ), or None."""
        text = self.definition(inverse, i)
        m = text and re.fullmatch(r"arith\.divf (%\S+), (%\S+) : vector<3xf64>", text)
        if not m or not self.is_dense_one(m.group(1), i):
            return None
        mu = m.group(2)
        text = self.definition(mu, i)
        e = text and re.fullmatch(r"math\.exp (%\S+) : vector<3xf64>", text)
        return (mu, e.group(1) if e else None)

    def is_dense_one(self, name, i):
        text = self.definition(name, i)
        return bool(text and re.fullmatch(
            r"arith\.constant dense<1\.000000e\+00> : vector<3xf64>", text))

    def is_dense_half(self, name, i):
        text = self.definition(name, i)
        return bool(text and re.fullmatch(
            r"arith\.constant dense<5\.000000e-01> : vector<3xf64>", text))

    def velocities_tail(self, i, inverse_line):
        """The four lines of Σ (μ⁻² − 1) K from line i: (inverse, K, sum),
        or None."""
        f = r"(%\S+)"
        found = self.match_lines(i, [
            f + r" = arith\.mulf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.subf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.mulf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = vector\.reduction <add>, " + f + r" : vector<3xf64> into f64"])
        if not found:
            return None
        sq, less, product, total = found
        if not (sq.group(2) == sq.group(3) and less.group(2) == sq.group(1) and
                self.is_dense_one(less.group(3), i) and product.group(2) == less.group(1) and
                total.group(2) == product.group(1)):
            return None
        return sq.group(2), product.group(3), total.group(1)

    def lift_work(self, i):
        f = r"(%\S+)"
        indent = self.lines[i][:len(self.lines[i]) - len(self.lines[i].lstrip())]
        # trotter: the share of C at V', the mean virial, the log strain.
        found = self.match_lines(i, [
            f + r" = arith\.divf " + f + r", " + f + r" : f64",
            f + r" = arith\.mulf " + f + r", " + f + r" : f64",
            f + r" = vector\.broadcast " + f + r" : f64 to vector<3xf64>",
            f + r" = arith\.addf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.addf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.mulf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = arith\.mulf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = vector\.reduction <add>, " + f + r" : vector<3xf64> into f64"])
        if found:
            ca, ca3, cab, wb, ws, wm, wl, work = found
            tail = self.velocities_tail(i + 8, None)
            last = self.match_lines(i + 12, [f + r" = arith\.subf " + f + r", " + f + r" : f64"])
            if tail and last and ca3.group(2) == ca.group(1) and \
                    self.is_value(ca3.group(3), i, 1.0 / 3.0) and cab.group(2) == ca3.group(1) and \
                    wb.group(3) == cab.group(1) and ws.group(3) == wb.group(1) and \
                    wm.group(2) == ws.group(1) and self.is_dense_half(wm.group(3), i) and \
                    wl.group(2) == wm.group(1) and work.group(2) == wl.group(1) and \
                    last[0].group(2) == tail[2] and last[0].group(3) == work.group(1):
                tw = self.definition(ws.group(2), i)
                t = tw and re.fullmatch(r"arith\.addf (%\S+), (%\S+) : vector<3xf64>", tw)
                share = t and self.share_of(t.group(2), i)
                logs = self.log_of_inverse(tail[0], i)
                if share and share[0] == ca.group(2) and logs and logs[1] == wl.group(3):
                    line = ("%s%s = dyn.scaling_work kinetic(%s) log_strain(%s) virial_before(%s) "
                            "virial_after(%s) constant(%s) volumes(%s, %s) {count = \"trotter\"} : f64"
                            % (indent, last[0].group(1), tail[1], wl.group(3), t.group(1),
                               wb.group(2), ca.group(2), share[1], ca.group(3)))
                    return self.replace(i, i + 13, [("dyn.scaling_work", line)],
                                        [last[0].group(1)])
        # trotter_before.
        found = self.match_lines(i, [
            f + r" = arith\.mulf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = vector\.reduction <add>, " + f + r" : vector<3xf64> into f64"])
        if found:
            wl, work = found
            tail = self.velocities_tail(i + 2, None)
            last = self.match_lines(i + 6, [f + r" = arith\.subf " + f + r", " + f + r" : f64"])
            tw = self.definition(wl.group(2), i)
            t = tw and re.fullmatch(r"arith\.addf (%\S+), (%\S+) : vector<3xf64>", tw)
            share = t and self.share_of(t.group(2), i)
            if tail and last and share and work.group(2) == wl.group(1) and \
                    last[0].group(2) == tail[2] and last[0].group(3) == work.group(1):
                logs = self.log_of_inverse(tail[0], i)
                if logs and logs[1] == wl.group(3):
                    line = ("%s%s = dyn.scaling_work kinetic(%s) log_strain(%s) virial_before(%s) "
                            "constant(%s) volumes(%s) {count = \"trotter_before\"} : f64"
                            % (indent, last[0].group(1), tail[1], wl.group(3), t.group(1),
                               share[0], share[1]))
                    return self.replace(i, i + 7, [("dyn.scaling_work", line)],
                                        [last[0].group(1)])
        # first_order.
        found = self.match_lines(i, [
            f + r" = arith\.subf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = vector\.reduction <add>, " + f + r" : vector<3xf64> into f64",
            f + r" = arith\.addf " + f + r", " + f + r" : f64",
            f + r" = arith\.mulf " + f + r", " + f + r" : f64",
            f + r" = vector\.broadcast " + f + r" : f64 to vector<3xf64>",
            f + r" = arith\.mulf " + f + r", " + f + r" : vector<3xf64>",
            f + r" = vector\.reduction <add>, " + f + r" : vector<3xf64> into f64"])
        if found:
            m1, tr, wt, wt3, wtb, wl, work = found
            tail = self.velocities_tail(i + 7, None)
            last = self.match_lines(i + 11, [f + r" = arith\.addf " + f + r", " + f + r" : f64"])
            bwc = self.definition(wt.group(3), i)
            d = bwc and re.fullmatch(r"arith\.divf (%\S+), (%\S+) : f64", bwc)
            if tail and last and d and self.is_dense_one(m1.group(2), i) and \
                    wt.group(2) == tr.group(1) and wt3.group(2) == wt.group(1) and \
                    self.is_value(wt3.group(3), i, 1.0 / 3.0) and wtb.group(2) == wt3.group(1) and \
                    wl.group(2) == m1.group(1) and wl.group(3) == wtb.group(1) and \
                    work.group(2) == wl.group(1) and last[0].group(2) == work.group(1) and \
                    last[0].group(3) == tail[2]:
                logs = self.log_of_inverse(tail[0], i)
                if logs and logs[0] == m1.group(3):
                    line = ("%s%s = dyn.scaling_work kinetic(%s) strain(%s) virial_before(%s) "
                            "constant(%s) volumes(%s) {count = \"first_order\"} : f64"
                            % (indent, last[0].group(1), tail[1], m1.group(3), tr.group(2),
                               d.group(1), d.group(2)))
                    return self.replace(i, i + 12, [("dyn.scaling_work", line)],
                                        [last[0].group(1)])
        return None

    def lift_scale_cell(self, i):
        """The scaled cell and its stores, then mdrtSetBox and mdrtSetTilt."""
        f = r"(%\S+)"
        first = re.fullmatch(f + r" = vector\.extract " + f + r"\[0\] : f64 from vector<3xf64>",
                             self.lines[i].strip())
        if not first:
            return None
        mu = first.group(2)
        column = [0, 1, 2, 0, 0, 1]
        k, line, factors, edges, scaled, memory, count = 0, i, {}, [], [], None, None
        while k < 6:
            if k < 3:
                m = re.fullmatch(f + r" = vector\.extract " + re.escape(mu) +
                                 r"\[%d\] : f64 from vector<3xf64>" % k, self.lines[line].strip())
                if not m:
                    return None
                factors[k] = m.group(1)
                line += 1
            m = re.fullmatch(f + r" = arith\.mulf " + f + r", " + f + r" : f64",
                             self.lines[line].strip())
            st = re.fullmatch(r"memref\.store " + f + r", " + f + r"\[" + f +
                              r"\] : memref<([36])xf64>", self.lines[line + 1].strip())
            if not (m and st and m.group(3) == factors[column[k]] and st.group(1) == m.group(1)
                    and self.value(st.group(3), i) == str(k)):
                return None
            if memory is None:
                memory, count = st.group(2), int(st.group(4))
            if st.group(2) != memory:
                return None
            edges.append(m.group(2))
            scaled.append(m.group(1))
            line += 2
            k += 1
            if k == count:
                break
        box = re.fullmatch(r"func\.call @mdrtSetBox\((.*)\) : \(f64, f64, f64\) -> \(\)",
                           self.lines[line].strip())
        if not box or [x.strip() for x in box.group(1).split(",")] != scaled[:3]:
            return None
        line += 1
        if count == 6:
            tilt = re.fullmatch(r"func\.call @mdrtSetTilt\((.*)\) : \(f64, f64, f64\) -> \(\)",
                                self.lines[line].strip())
            if not tilt or [x.strip() for x in tilt.group(1).split(",")] != scaled[3:]:
                return None
            line += 1
        indent = self.lines[i][:len(self.lines[i]) - len(self.lines[i].lstrip())]
        types = ", ".join(["f64"] * count)
        ops = [("dyn.scale_cell", "%s%s = dyn.scale_cell %s by %s : %s -> %s" %
                (indent, ", ".join(scaled), ", ".join(edges), mu, types, types)),
               ("dyn.store_cell", "%sdyn.store_cell %s into %s : memref<%dxf64>" %
                (indent, ", ".join(scaled), memory, count))]
        return self.replace(i, line, ops, scaled)

    def run(self):
        i = 0
        while i < len(self.lines):
            following = None
            if "md.sum_particles" in self.lines[i]:
                following = self.lift_group_kinetic(i)
            if following is None and "md.map_particles" in self.lines[i]:
                following = self.lift_scale_positions(i)
            if following is None and "md." in self.lines[i]:
                following = self.lift_loop(i)
            if following is None and "memref.load" in self.lines[i]:
                following = self.lift_load_cell(i)
            if following is None and "arith.addf" in self.lines[i]:
                following = self.lift_pressure(i)
            if following is None and ("vector.reduction" in self.lines[i] or
                                      "vector.extract" in self.lines[i]):
                following = self.lift_strain(i)
            if following is None and "vector.extract" in self.lines[i]:
                following = self.lift_scale_cell(i)
            if following is None and ("arith.divf" in self.lines[i] or
                                      "arith.mulf" in self.lines[i] or
                                      "arith.subf" in self.lines[i]):
                following = self.lift_work(i)
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

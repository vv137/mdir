"""Compare the loops over a list of units with those of #795 (D[md-alg-segments]).

    compare_segment_loops.py REFERENCE.mlir CONVERTED.mlir [FUNCTION ...]

REFERENCE holds the md_exec.template_for loops of #795 at 004a0735, after
md-differentiate and convert-md-to-md-exec; CONVERTED the
md_exec.segment_for loops of convert-md-to-md-exec over the segment windows
of D405 (one region and one table each). The loops of each FUNCTION (an
md.potential or md.function; all that both have, by default) are compared
in their order. For each loop: the segment (WW, SW, SS of #795 are
segment(0, 0), (1, 0), (1, 1)), the kind of value (an energy, a strain, or
a force), and, site pair by site pair, the expression of its contribution.

The expression is named as compare_segment_windows.py names it: each value
by its operation, attributes, types, and the names of its operands,
recursively, with the order of the operations, the names of the values, and
repeated constants not counting. The site pair's displacement is a name of
its own, D<m>: in #795 the value that the body yields as the m-th
displacement, the arithmetic of its placement included; in the converted
loop the argument of the region for site pair m. A gathered field's value
at member k is G<field>.<k> in both. The force of a site pair is #795's
force at the slot of its first place (the sites' layout, O, H1, H2, E, or
the frame's), which is the converted loop's one value; the converted
table's places say which slot. The weight of the site, which #795 has in
the placement, so not in a contribution, does not enter the comparison.
"""
import re
import sys

SEGMENTS = {'water': (0, 0), 'mixed': (1, 0), 'atoms': (1, 1)}
VALUE = re.compile(r'%[A-Za-z_0-9]+(?:#[0-9]+)?')


def regions_after(lines, i):
    """The regions of the op whose header is line i: for each, its label
    (the word before '{' or 'segment(f, g) sites [...]'), its block
    arguments, its body, and its yield; and the line after the op."""
    found = []
    label = lines[i]
    i += 1
    while True:
        arguments = re.findall(r'(%[A-Za-z_0-9]+): ([^,)]+)', lines[i])
        body = []
        i += 1
        while not lines[i].strip().startswith('md_exec.yield'):
            body.append(lines[i].strip())
            i += 1
        yielded = lines[i].strip()
        found.append((label, arguments, body, yielded))
        i += 1
        closing = lines[i].strip()
        if closing.startswith('} :') or closing == '}' or not closing.endswith('{'):
            return found, i + 1
        label = closing
        i += 1


def loops(text, op):
    """The loops `op` of each function, in order."""
    result = {}
    lines = text.splitlines()
    function = None
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith('//'):
            i += 1
            continue
        match = re.match(r'\s*md\.(?:potential|function) @([\w.]+)\(', lines[i])
        if match:
            function = match.group(1)
        if f'md_exec.{op} ' in lines[i]:
            header = lines[i]
            regions, i = regions_after(lines, i)
            result.setdefault(function, []).append((header, regions))
            continue
        i += 1
    return result


class Canon:
    def __init__(self):
        self.table = {}

    def intern(self, key):
        return self.table.setdefault(key, len(self.table))

    def name(self, names, body, values):
        def rename(text):
            return VALUE.sub(lambda m: names.get(m.group(0), m.group(0)), text)
        for line in body:
            match = re.match(r'(%[A-Za-z_0-9]+)(?::(\d+))? = (.*)$', line)
            assert match, line
            result, count, rest = match.groups()
            if result in names:  # a displacement, named D<m>
                continue
            key = rename(rest)
            if count:
                for k in range(int(count)):
                    names[f'{result}#{k}'] = f'E{self.intern(key + f"#{k}")}'
            else:
                names[result] = f'E{self.intern(key)}'
        return [rename(v) for v in values]


def yielded_values(line):
    return re.match(r'md_exec\.yield (.*?) :', line).group(1).split(', ')


def kind(header):
    if 'outs(' in header:
        return 'force'
    return 'strain' if 'vector<9xf64>' in header.split('->')[-1] else 'energy'


def reference_loop(canon, header, regions):
    """#795: the segment, the kind, and the contributions by site pair."""
    pairs = [int(x) for x in re.search(r'pairs \[([^\]]*)\]', header).group(1).split(', ')]
    tag = next(k for k, count in enumerate(pairs) if count)
    label, arguments, body, yielded = regions[tag]
    count = pairs[tag]
    values = yielded_values(yielded)
    slots = len(values) // count - 1
    displacements = values[count * slots:]
    names = {}
    ncoord = {0: 5, 1: 3, 2: 1}[tag]
    fields = re.search(r'ins\(([^:]*):', header)
    nfields = len(fields.group(1).split(',')) if fields else 0
    arity = {0: 6, 1: 4, 2: 2}[tag]
    for index, (name, _) in enumerate(arguments):
        if index < ncoord:
            names[name] = f'C{index}'
        else:
            field, member = divmod(index - ncoord, arity)
            names[name] = f'G{field}.{member}'
    for m, value in enumerate(displacements):
        names[value] = f'D{m}'
    named = canon.name(names, body, values)
    return list(SEGMENTS.values())[tag], kind(header), named, count, slots


def converted_loop(canon, header, regions):
    (label, arguments, body, yielded), = regions
    segment = tuple(int(x) for x in re.search(r'segment\((\d+), (\d+)\)', label).groups())
    table = [(int(p), int(q)) for p, q in re.findall(r'\((\d+), (\d+)\) \[', label)]
    names = {}
    M = len(table)
    fields = re.search(r'ins\(([^:]*):', header)
    members = (len(arguments) - 2 * M) // (len(fields.group(1).split(',')) if fields else 1) if fields else 0
    for index, (name, _) in enumerate(arguments):
        if index < 2 * M:
            names[name] = f'D{index // 2}' if index % 2 else f'R{index // 2}'
        else:
            field, member = divmod(index - 2 * M, members)
            names[name] = f'G{field}.{member}'
    values = yielded_values(yielded)
    return segment, kind(header), canon.name(names, body, values), table


def main():
    reference, converted = (open(p).read() for p in sys.argv[1:3])
    wanted = sys.argv[3:]
    first, second = loops(reference, 'template_for'), loops(converted, 'segment_for')
    functions = wanted or sorted(set(first) & set(second))
    canon = Canon()
    compared = 0
    for function in functions:
        a, b = first.get(function, []), second.get(function, [])
        if len(a) != len(b):
            raise SystemExit(f'@{function}: {len(a)} and {len(b)} loops')
        for x, y in zip(a, b):
            segment, what, values, count, slots = reference_loop(canon, *x)
            segment2, what2, values2, table = converted_loop(canon, *y)
            where = f'@{function} {what} segment{segment2}'
            if (segment, what) != (segment2, what2) or count != len(table):
                raise SystemExit(f'{where}: the loops differ: {segment} {what} {count} '
                                 f'and {segment2} {what2} {len(table)}')
            for m, (p, q) in enumerate(table):
                expected = values[m * slots + p] if what == 'force' else values[m]
                if expected != values2[m]:
                    raise SystemExit(f'{where}: site pair ({p}, {q}) differs')
            print(f'{where}: {len(table)} site pairs, the same expressions')
            compared += 1
    print(f'identical: {compared} loops')


main()

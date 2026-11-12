"""Compare the site windows of two programs as expressions (D[kind-pairs]).

    compare_segment_windows.py REFERENCE.mlir GENERATED.mlir [OLD=NEW ...]

REFERENCE holds the windows of D[rigid-water-template]
(md.sum_template_window and md.gather_template_window over WW, SW and SS),
GENERATED those of md.kind_pairs (md.sum_segment_window and
md.gather_segment_window over segment(0, 0), (1, 0) and (1, 1)). The
functions of GENERATED that hold windows are compared with those of the
same names in REFERENCE, window by window in their order. For each pair, the
values that the bodies yield are compared as expressions: every value is
named by its operation, its attributes and types, and the names of its
operands, recursively, with the block's arguments by their position and the
function's arguments by their number. The order of the operations, the
names of the values, and repeated constants do not count; any other
difference in a yielded expression does. OLD=NEW replaces a literal in
REFERENCE before the comparison (a constant that is known to differ); each
replacement is reported with its count.
"""
import re
import sys

SEGMENTS = {'water_water': (0, 0), 'solute_water': (1, 0), 'solute_solute': (1, 1)}
VALUE = re.compile(r'%[A-Za-z_0-9]+(?:#[0-9]+)?')


def windows(text):
    """The windows of each function: (header, arguments, types, lines,
    yield, result type)."""
    found = {}
    potential = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r'\s*md\.(?:potential|function) @([\w.]+)\(', line)
        if match:
            potential = match.group(1)
        if re.search(r'md\.(sum|gather)_(template|segment)_window ', line):
            header = line.strip()
            arguments = re.findall(r'(%[A-Za-z_0-9]+): ', lines[i + 1])
            types = re.findall(r'%[A-Za-z_0-9]+: ([^,)]+)', lines[i + 1])
            body = []
            i += 2
            while not lines[i].strip().startswith('md.yield'):
                body.append(lines[i].strip())
                i += 1
            yielded = lines[i].strip()
            result = lines[i + 1].split('->')[-1].strip()
            found.setdefault(potential, []).append(
                (header, arguments, types, body, yielded, result))
        i += 1
    return found


class Canon:
    """Interned names of expressions, shared by both programs."""

    def __init__(self):
        self.table = {}

    def intern(self, key):
        return self.table.setdefault(key, len(self.table))

    def body(self, arguments, lines, yielded):
        names = {name: f'A{index}' for index, name in enumerate(arguments)}

        def rename(text):
            return VALUE.sub(lambda m: names.get(m.group(0), m.group(0)), text)

        for line in lines:
            match = re.match(r'(%[A-Za-z_0-9]+)(?::(\d+))? = (.*)$', line)
            assert match, line
            result, count, rest = match.groups()
            key = rename(rest)
            if count:
                for k in range(int(count)):
                    names[f'{result}#{k}'] = f'E{self.intern(key + f"#{k}")}'
            else:
                names[result] = f'E{self.intern(key)}'
        values = re.match(r'md\.yield (.*?) :', yielded).group(1).split(', ')
        return [rename(value) for value in values]


def segment(header):
    match = re.search(r'segment\((\d+), (\d+)\)', header)
    if match:
        return int(match.group(1)), int(match.group(2))
    relation = re.search(r'_template_window %(\w+),', header).group(1)
    return SEGMENTS[relation]


def gathered(header):
    match = re.search(r'gather\(([^:]*):', header)
    return match.group(1).strip() if match else ''


def operation(header):
    return re.search(r'md\.(sum|gather)_', header).group(1)


def cutoff(header):
    return re.search(r'cutoff\(([^)]*)\)', header).group(1)


def main():
    reference, generated = (open(path).read() for path in sys.argv[1:3])
    for replacement in sys.argv[3:]:
        old, new = replacement.split('=')
        count = reference.count(old)
        reference = reference.replace(old, new)
        print(f'replaced {old} by {new} in the reference: {count} times')
    canon = Canon()
    first, second = windows(reference), windows(generated)
    missing = sorted(set(second) - set(first))
    if missing:
        raise SystemExit(f'the reference has no {missing}')
    pairs = 0
    for potential in sorted(second):
        if len(first[potential]) != len(second[potential]):
            raise SystemExit(f'@{potential}: {len(first[potential])} and '
                             f'{len(second[potential])} windows')
        for a, b in zip(first[potential], second[potential]):
            where = f'@{potential} {operation(b[0])} segment{segment(b[0])}'
            for what, x, y in [('operation', operation(a[0]), operation(b[0])),
                               ('segment', segment(a[0]), segment(b[0])),
                               ('result', a[5], b[5]),
                               ('cutoff', cutoff(a[0]), cutoff(b[0])),
                               ('gathered fields', gathered(a[0]), gathered(b[0])),
                               ('block arguments', a[2], b[2])]:
                if x != y:
                    raise SystemExit(f'{where}: {what} differ: {x} and {y}')
            x, y = (canon.body(w[1], w[3], w[4]) for w in (a, b))
            if x != y:
                raise SystemExit(f'{where}: the yielded expressions differ')
            print(f'{where}: {len(x)} yielded values, the same expressions')
            pairs += 1
    print(f'identical: {pairs} windows')


main()

"""The widths of the shared-memory loads and stores in the SASS of the tiled
gathering kernels of the prototype (P0 of #337): whether the compiler
merged adjacent points of a column into LDS.64 or LDS.128, in which case an
analysis of banks by single words does not hold.
lds_width.py SASS_DUMP  (cuobjdump -sass -arch sm_89 BINARY > SASS_DUMP)"""
import collections, re, sys

current, counts = None, collections.defaultdict(collections.Counter)
for line in open(sys.argv[1]):
    m = re.search(r'Function : (_Z\d+(gatherTiled|gatherPreTiled)If(\w+?)EEvi)', line)
    if m:
        ints = re.findall(r'Li(\d+)E', m.group(3))
        pad = m.group(3).endswith('Lb1')
        current = (m.group(2), tuple(int(v) for v in ints), pad)
        continue
    if line.strip().startswith('Function :'):
        current = None
        continue
    if current:
        for op in re.findall(r'\b(LDS(?:\.\w+)*|STS(?:\.\w+)*)\b', line):
            counts[current][op] += 1
for key in sorted(counts):
    kind, ints, pad = key
    if kind == 'gatherTiled':
        P, D, B, G = ints
    else:
        P, B, G = ints
        D = 0
    if G not in (8, 32) or P not in (4, 5, 6, 8):
        continue
    print(f'{kind:15s} P {P} D {D:2d} lanes {G:2d} padded {int(pad)}: ' +
          ', '.join(f'{op} {n}' for op, n in sorted(counts[key].items())))

"""Larger SW-window fixture and exact entry-multiset comparison.

The independent coverage oracle is reused without changing its checker,
classification, or tolerances. Only the generated population is enlarged;
the existing dyadic geometry, reference frames, mobility phases and exact
all-image enumeration remain its own. This wrapper does not run MDIR.
"""
import argparse
import pathlib
import sys

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
generate = sub.add_parser('generate')
generate.add_argument('program')
generate.add_argument('reference')
generate.add_argument('--triclinic', action='store_true')
generate.add_argument('--reframed', action='store_true')
generate.add_argument('--growth', action='store_true')
compare = sub.add_parser('compare')
compare.add_argument('split', type=pathlib.Path)
compare.add_argument('unsplit', type=pathlib.Path)
args = parser.parse_args()
if args.command == 'compare':
    def canonical(path):
        words = iter(int(v) for v in path.read_text().split())
        contexts, entries = [], []
        for marker in words:
            assert marker == 917031, ('phase marker', marker)
            phase = next(words)
            n = next(words)
            order = tuple(next(words) for _ in range(n))
            row_count = next(words)
            rows, class_counts = [], [0, 0, 0]
            offset = 0
            for row in range(row_count):
                tag, first, begin, count = (next(words) for _ in range(4))
                assert 0 <= tag < 3 and begin == offset and count >= 0
                rows.append((tag, first, count))
                class_counts[tag] += count
                offset += count
                for _ in range(count):
                    partner, image, mask, reserved = (next(words) for _ in range(4))
                    # Keep group/ownership, class, image, mask and every public
                    # payload field exact. Sorting preserves multiplicity.
                    entries.append((phase, row, tag, first, partner, image, mask, reserved))
            chunks, flags, total = (next(words) for _ in range(3))
            assert total == offset and next(words) == 917032
            contexts.append((phase, order, tuple(rows), tuple(class_counts), chunks, flags, total))
        return contexts, sorted(entries)

    left_context, left = canonical(args.split)
    right_context, right = canonical(args.unsplit)
    assert left_context == right_context, 'group/class/count/chunk/flag behavior differs'
    if left != right:
        for i, (a, b) in enumerate(zip(left, right)):
            if a != b:
                raise AssertionError(('exact entry mismatch', i, a, b))
        raise AssertionError(('entry multiplicity/count mismatch', len(left), len(right)))
    print('SW split/unsplit: exact entry multiset, all fields and multiplicity', len(left))

else:
    oracle = pathlib.Path(__file__).with_name('template_search_oracle.py')
    source = oracle.read_text()
    old = 'waters, atoms = 19, 17'
    assert source.count(old) == 1, 'oracle population definition changed'
    source = source.replace(old, 'waters, atoms = 257, 17')
    sys.argv = [str(oracle), 'generate', args.program, args.reference]
    for option in ['triclinic', 'reframed', 'growth']:
        if getattr(args, option):
            sys.argv.append('--' + option)
    if args.growth:
        sys.argv.append('--refresh')
    exec(compile(source, str(oracle), 'exec'), {'__name__': '__main__', '__file__': str(oracle)})

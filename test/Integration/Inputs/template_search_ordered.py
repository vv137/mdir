"""Compare compacted logical records without sorting or merging masks.

The independent coverage checker must also pass for both inputs. Physical
allocator tickets are absent from this dump; row-local output order is not.
"""
import argparse
import json
import pathlib
import subprocess
import sys


def records(path, reference):
    words = iter(map(int, path.read_text().split()))
    phases, dispatch = [], []
    for phase, ref in enumerate(reference['phases']):
        if reference.get('dispatch'):
            assert next(words) == 917034 and next(words) == phase
            mode, active = next(words), next(words)
            assert active == int(ref['active'][0]) + 2 * int(ref['active'][1])
            dispatch.append(mode)
        assert next(words) == 917031 and next(words) == phase
        count = next(words)
        order = tuple(next(words) for _ in range(count))
        row_count = next(words)
        rows = []
        for _ in range(row_count):
            header = tuple(next(words) for _ in range(4))
            payload = tuple(tuple(next(words) for _ in range(4))
                            for _ in range(header[3]))
            rows.append((header, payload))
        counters = tuple(next(words) for _ in range(3))
        assert next(words) == 917032
        phases.append((order, tuple(rows), counters))
    assert next(words, None) is None, 'unexpected trailing records'
    return phases, dispatch


parser = argparse.ArgumentParser()
parser.add_argument('general', type=pathlib.Path)
parser.add_argument('automatic', type=pathlib.Path)
parser.add_argument('reference', type=pathlib.Path)
parser.add_argument('--auto-paths', required=True,
                    help='Expected actual automatic owner path per phase, comma separated')
args = parser.parse_args()
reference = json.loads(args.reference.read_text())
assert reference.get('dispatch'), 'ordered image tests require actual selector observations'
oracle = pathlib.Path(__file__).with_name('template_search_oracle.py')
for path in [args.general, args.automatic]:
    subprocess.run([sys.executable, str(oracle), 'check', str(path),
                    str(args.reference)], check=True)
general, general_paths = records(args.general, reference)
automatic, automatic_paths = records(args.automatic, reference)
expected = list(map(int, args.auto_paths.split(',')))
assert general_paths == [2] * len(general), ('forced general was not selected', general_paths)
assert automatic_paths == expected, ('actual automatic dispatch', automatic_paths, expected)
assert len(general) == len(automatic)
for phase, (wanted, actual) in enumerate(zip(general, automatic)):
    assert wanted == actual, ('ordered general/specialized records differ', phase)
print('exact ordered general/specialized records:', len(general), 'phases passed')
print('actual automatic image paths:', ','.join(map(str, automatic_paths)))

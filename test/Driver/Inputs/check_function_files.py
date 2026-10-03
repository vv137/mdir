"""Exercise file-backed grids against inline grids and independent spline oracles.

check_function_files.py TEST_SOURCE_DIRECTORY SCRATCH_DIRECTORY CPU|GPU MDIR
Only the generated inputs and outputs in SCRATCH_DIRECTORY are modified.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

source, scratch, target = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
mdir = sys.argv[4]
scratch.mkdir(parents=True, exist_ok=True)
scratch = scratch.resolve()
for name in ('dipeptide.prmtop', 'dipeptide.inpcrd'):
    shutil.copyfile(source / 'Inputs' / 'dipeptide' / name, scratch / name)


def invoke(*args, expected=0, working=None):
    result = subprocess.run(args, cwd=working or scratch, text=True, capture_output=True)
    if result.returncode != expected:
        raise AssertionError(f'{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result.stdout + result.stderr


def convert(text, prefix):
    """Write nested lists in their existing order, the last argument fastest."""
    current = ''
    grids = []
    out = []
    for line in text.splitlines():
        if line.startswith('[[energy.function]]'):
            current = 'function'
        elif line.startswith('['):
            current = ''
        if current and line.strip().startswith('name '):
            name = ast.literal_eval(line.split('=', 1)[1].strip())
        if current and line.strip().startswith('values '):
            values = ast.literal_eval(line.split('=', 1)[1].strip())
            sizes = []
            inner = values
            while isinstance(inner, list):
                sizes.append(len(inner))
                inner = inner[0]
            def flatten(rows):
                if isinstance(rows, list):
                    for row in rows:
                        yield from flatten(row)
                else:
                    yield rows
            flat = list(flatten(values))
            path = f'{prefix}-{name}.dat'
            data = '# The last argument varies fastest.\n\n' + ''.join(
                f'  {v:+.17e}\t# value {i}\n' for i, v in enumerate(flat))
            (scratch / path).write_text(data)
            grids.append((path, sizes, flat))
            out.extend((f'values_file = "{path}"', f'shape = {sizes}'))
        else:
            out.append(line)
    return '\n'.join(out) + '\n', grids


for dimension, fixture, oracle in (
    ('1d', 'tabulated-functions.test', 'check_tabulated.py'),
    ('nd', 'tabulated-functions-nd.test', 'check_functions.py'),
):
    fixture_text = (source / fixture).read_text()
    text = fixture_text.split('#--- run.toml\n')[1].split('#--- end')[0]
    text = re.sub(r'^target\s*=.*$', f'target = "{target}"', text, flags=re.M)
    for precision in ('DOUBLE', 'MIXED'):
        label = f'{dimension}-{target.lower()}-{precision.lower()}'
        inline = re.sub(r'^precision\s*=.*$', f'precision = "{precision}"', text, flags=re.M)
        inline = inline.replace('terms.h5', f'{label}.h5')
        file_text, grids = convert(inline, label)
        if dimension == 'nd' and precision == 'DOUBLE':
            file_text = file_text.replace('[output]', '[output]\nmanifest = "inputs.jsonl"')
        (scratch / 'inline.toml').write_text(inline)
        (scratch / 'file.toml').write_text(file_text)
        # Running from another directory also exercises control-relative paths.
        emitted_inline = invoke(mdir, 'emit', str(scratch / 'inline.toml'), working=scratch.parent)
        emitted_file = invoke(mdir, 'emit', str(scratch / 'file.toml'), working=scratch.parent)
        assert emitted_file == emitted_inline, 'file and inline grids generated different IR'
        plain = inline.split('#--- terms')[0].replace(f'{label}.h5', 'plain.h5')
        (scratch / 'plain.toml').write_text(plain)
        invoke(mdir, 'run', 'plain.toml')
        inline_log = invoke(mdir, 'run', 'inline.toml')
        inline_forces = invoke(mdir, 'checkpoint', '--print=forces', f'{label}.h5')
        file_log = invoke(mdir, 'run', 'file.toml')
        file_forces = invoke(mdir, 'checkpoint', '--print=forces', f'{label}.h5')
        inline_numbers = [float(v) for line in inline_forces.splitlines()
                          for v in line.split()[2:5]]
        file_numbers = [float(v) for line in file_forces.splitlines()
                        for v in line.split()[2:5]]
        assert len(inline_numbers) == len(file_numbers)
        repeat_difference = max(abs(a-b) for a, b in zip(inline_numbers, file_numbers))
        scale = max(1.0, max(map(abs, inline_numbers)))
        repeat_tolerance = (1e-12 if precision == 'DOUBLE' else 2e-6) * scale
        if target == 'CPU':
            assert inline_forces == file_forces, 'loaded grid changed CPU forces'
        else:
            # GPU reductions need not produce the same final bit twice.
            assert repeat_difference <= repeat_tolerance, (repeat_difference, repeat_tolerance)
        inline_energies = re.findall(r'^MDIR:\s+(\w+)\s+(-?[\d.]+)$', inline_log, re.M)
        file_energies = re.findall(r'^MDIR:\s+(\w+)\s+(-?[\d.]+)$', file_log, re.M)
        assert dict(file_energies).keys() == dict(inline_energies).keys()
        for name, actual in file_energies:
            reference = float(dict(inline_energies)[name])
            bound = (1e-6 if precision == 'DOUBLE' else 2e-5) * max(1.0, abs(reference))
            assert abs(float(actual) - reference) <= bound, (name, actual, reference)
        (scratch / 'terms.forces').write_text(file_forces)
        (scratch / 'plain.forces').write_text(invoke(mdir, 'checkpoint', '--print=forces', 'plain.h5'))
        tolerance = '1e-6' if precision == 'DOUBLE' else '2e-5'
        if dimension == 'nd':
            oracle_args = ['inline.toml', 'dipeptide.prmtop', 'dipeptide.inpcrd',
                           'plain.forces', 'terms.forces', tolerance]
        else:
            oracle_args = ['dipeptide.prmtop', 'dipeptide.inpcrd',
                           'plain.forces', 'terms.forces', tolerance]
        report = invoke(sys.executable, str(source / 'Inputs' / oracle), *oracle_args)
        assert 'forces against differences of the energy: ok' in report, report
        for name, reference in re.findall(r'^(\w+) (-?[\d.]+)$', report, re.M):
            reference = float(reference)
            actual = float(dict(file_energies)[name])
            bound = float(tolerance) * max(1.0, abs(reference))
            assert abs(actual - reference) <= bound, (name, actual, reference, bound)
            print(f'{label} {name}: reference {reference:.6f}, difference {actual-reference:.9g}, tolerance {bound:.9g} kcal/mol')
        print(label + ': identical IR; equivalent energies and forces; independent oracle passed')
        print(f'inline/file force difference {repeat_difference:.9g}; tolerance {repeat_tolerance:.9g} kJ/mol/nm')
        print(report.strip())
        if dimension == 'nd' and precision == 'DOUBLE':
            records = [json.loads(line) for line in (scratch / 'inputs.jsonl').read_text().splitlines()]
            inputs = next(record['inputs'] for record in records if 'inputs' in record)
            for path, _, _ in grids:
                entry = next(v for v in inputs if v['path'] == str(scratch / path))
                assert entry['role'] == 'tabulated_function'
                assert entry['sha256'] == hashlib.sha256((scratch / path).read_bytes()).hexdigest()
            # Resolved part outputs and the previous-checkpoint path are also
            # protected; these names differ from the literal output keys.
            for reserved, extra_output in (
                ('part-input.part0002.log', 'log = "part-input.log"\n'),
                (f'{label}.h5.prev', ''),
            ):
                saved_grid = (scratch / grids[0][0]).read_bytes()
                (scratch / reserved).write_bytes(saved_grid)
                collision = file_text.replace(grids[0][0], reserved)
                collision = collision.replace('[output]', '[output]\n' + extra_output)
                collision = collision.replace('steps      = 1', 'steps      = 2')
                (scratch / 'part-collision.toml').write_text(collision)
                failure = invoke(mdir, 'run', '--continue', '--no-append',
                                 'part-collision.toml', expected=1)
                assert 'a tabulated input of the run' in failure, failure
                assert (scratch / reserved).read_bytes() == saved_grid
                (scratch / reserved).unlink()
            print('resolved part and previous-checkpoint collisions: passed')
            # Switching between a file and equivalent inline values preserves physics.
            (scratch / 'inline.toml').write_text(inline.replace('steps      = 1', 'steps      = 2'))
            invoke(mdir, 'run', '--continue', 'inline.toml')
            path = grids[0][0]
            saved = (scratch / path).read_text()
            (scratch / path).write_text(saved.replace(f'{grids[0][2][0]:+.17e}', '0.125', 1))
            changed = file_text.replace('steps      = 1', 'steps      = 3')
            (scratch / 'file.toml').write_text(changed)
            failure = invoke(mdir, 'run', '--continue', 'file.toml', expected=1)
            assert '[[energy.function]]' in failure and 'differ' in failure, failure
            (scratch / path).write_text(saved)
            moved = path + '.moved'
            shutil.copyfile(scratch / path, scratch / moved)
            (scratch / 'file.toml').write_text(changed.replace(path, moved))
            invoke(mdir, 'run', '--continue', 'file.toml')
            print('manifest hashes and continuation grid changes: passed')

# An asymmetric 2-by-3 grid makes a silent transpose observable in the IR.
example = text.split('#--- terms')[0] + '\n'.join((
    '\n[[energy.function]]', 'name = "order"', 'discrete = true',
    'values = [[11, 12, 13], [21, 22, 23]]', '',
))
file_example, grids_example = convert(example, 'order')
assert grids_example[0][1:] == ([2, 3], [11, 12, 13, 21, 22, 23])
(scratch / 'order-inline.toml').write_text(example)
(scratch / 'order-file.toml').write_text(file_example)
inline_ir = invoke(mdir, 'emit', 'order-inline.toml')
file_ir = invoke(mdir, 'emit', 'order-file.toml')
assert inline_ir == file_ir
assert 'fn_order' in file_ir
print('2-by-3 file order matches inline nested lists: passed')

# Input errors are checked once on the CPU; they precede compilation.
if target == 'CPU':
    # The current nd fixture contains continuous, discrete, and periodic grids.
    base, grids = convert(text, 'errors')
    path, sizes, flat = grids[0]
    first = f'values_file = "{path}"\nshape = {sizes}'

    def rejected(label, changed, expected, data=None):
        if data is not None:
            (scratch / 'bad.dat').write_text(data)
            changed = changed.replace(f'"{path}"', '"bad.dat"', 1)
        (scratch / 'bad.toml').write_text(changed)
        report = invoke(mdir, 'check', str(scratch / 'bad.toml'), expected=1)
        assert expected in report, (label, report)
        if data is not None and 'finite decimal' in expected:
            assert 'bad.dat:2:' in report, report

    valid = ' '.join(str(x) for x in flat)
    rejected('missing', base.replace(path, 'missing.dat'), 'cannot read')
    rejected('empty path', base.replace(path, ''), 'nonempty')
    rejected('missing shape', base.replace(first, f'values_file = "{path}"'), "expected 'shape'")
    for shape in ('[]', '[1, 1, 1, 1]', '[0]', '[-1]', '[2.5]', '[4294967296]',
                  '[4294967295, 4294967295, 4294967295]'):
        rejected('bad shape', base.replace(f'shape = {sizes}', f'shape = {shape}', 1),
                 'grid size' if shape not in ('[]', '[1, 1, 1, 1]') else "expected 'shape'")
    rejected('both', base.replace(first, first + '\nvalues = [1, 2]'), 'mutually exclusive')
    rejected('inline shape', base.replace(first, 'values = [1, 2]\nshape = [2]'), "requires 'values_file'")
    rejected('short', base, f'expected {len(flat)} values', valid.rsplit(' ', 1)[0])
    rejected('long', base, f'more than {len(flat)} values', valid + ' 0')
    rejected('empty', base, 'found 0', '# no values\n')
    for token in ('nan', 'inf', '1e999', '1,2', '0x1p0', '1D0', 'oops'):
        rejected(token, base, 'expected a finite decimal number', '# comment\n' + token)
    for key in ('checkpoint', 'manifest'):
        collision = base.replace('checkpoint = "terms.h5"', f'{key} = "{path}"' +
                                 ('\ncheckpoint = "terms.h5"' if key == 'manifest' else ''))
        rejected('collision', collision, 'a tabulated input of the run')
    alias = scratch / 'grid-alias.dat'
    alias.symlink_to(scratch / path)
    rejected('symlink collision', base.replace('checkpoint = "terms.h5"',
              'checkpoint = "grid-alias.dat"'), 'a tabulated input of the run')
    # Existing interpolation validation also applies to grids loaded from files.
    rejected('short axis', base.replace(f'shape = {sizes}', f'shape = [1, {len(flat)}]', 1),
             'at least two values along each argument')
    periodic_path, _, periodic_flat = grids[-1]
    changed = (scratch / periodic_path).read_text().replace(f'{periodic_flat[0]:+.17e}', '0.125', 1)
    (scratch / periodic_path).write_text(changed)
    rejected('periodic endpoints', base, 'last values along each argument equal to the first')
    print('malformed files, dimensions, periodic endpoints, and output collisions: passed')

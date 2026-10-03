"""Cell-area output against a cross product of checkpoint cell vectors.

Uses only the standard library. Checkpoint summaries print six significant
figures; 0.01 Å² covers their propagated rounding, while the log and column
file must agree to their respective four and six decimal places.
"""
import math
from pathlib import Path
import re
import subprocess
import sys

source, scratch, target = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
mdir = sys.argv[4]
scratch.mkdir(parents=True, exist_ok=True)


def run(control, *options, status=0):
    result = subprocess.run([mdir, 'run', *options, str(control)],
                            text=True, capture_output=True)
    assert result.returncode == status, result.stdout + result.stderr
    return result.stdout


def columns(path):
    lines = path.read_text().splitlines()
    names, units = lines[0].split()[1:], lines[1].split()[1:]
    assert names[-2:] == ['volume', 'area'], names
    assert units[-2:] == ['Å^3', 'Å^2'], units
    rows = [[float(v) for v in line.split()] for line in lines[2:]]
    assert rows and all(len(row) == len(names) for row in rows)
    return rows


def check(name, text, fixed_height=False):
    rows = columns(scratch / (name + '.energy'))
    log = [[float(v) for v in line.split()[1:]]
           for line in text.splitlines()
           if re.match(r'INFO: +[0-9]+ ', line)]
    assert len(log) == len(rows)
    assert 'VOLUME           AREA' in text
    assert max(abs(a-b) for r, s in zip(rows, log)
               for a, b in zip(r, s)) <= 5.0001e-5
    summary = subprocess.check_output(
        [mdir, 'checkpoint', str(scratch / (name + '.h5'))], text=True)
    box = [float(v) for v in re.search(r'box: +(\S+) (\S+) (\S+)', summary).groups()]
    tilt = re.search(r'tilts: +(\S+) (\S+) (\S+)', summary)
    bx = float(tilt.group(1)) if tilt else 0.0
    a, b = [box[0], 0.0, 0.0], [bx, box[1], 0.0]
    cross = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
             a[0]*b[1]-a[1]*b[0]]
    reference = math.sqrt(sum(v*v for v in cross)) * 100.0  # nm² to Å²
    difference = abs(rows[-1][-1] - reference)
    assert difference < 0.01, (name, rows[-1][-1], reference, difference)
    if fixed_height:
        assert box[2] == 2.32, box
        assert max(abs(r[-2] / 23.2 - r[-1]) for r in rows) < 1e-6
    print(f'{name}: reference {reference:.6f} Å², difference {difference:.6f} Å² (tolerance 0.01)')


def extract(name):
    text = (source / name).read_text()
    return text.split('#--- run.toml\n')[1].split('#--- end')[0]


base = extract('barostat-semi.test').replace('2000', '20').replace('1000', '10')
base = base.replace('coordinates = "mixture.pdb"',
                    f'coordinates = "{source / "Inputs/mixture.pdb"}"')
base = base.replace('[output]', '[output]\nenergy = "NAME.energy"')
for precision in ['double', 'mixed']:
    for coupling in ['ISOTROPIC', 'SEMI_ISOTROPIC', 'ANISOTROPIC']:
        name = precision + '-' + coupling.lower()
        text = base.replace('SEMI_ISOTROPIC', coupling).replace('semi.h5', 'NAME.h5')
        if coupling == 'SEMI_ISOTROPIC':
            text = text.replace('[barostat]', '[barostat]\ncompressibility_z = 0.0')
        text += f'\n[execution]\ntarget = "{target}"\nprecision = "{precision}"\n'
        text = text.replace('NAME', name)
        control = scratch / (name + '.toml')
        control.write_text(text)
        log = run(control)
        check(name, log, coupling == 'SEMI_ISOTROPIC')
        assert columns(scratch / (name + '.energy'))[0][-1] == 538.24
        if precision == 'mixed' and coupling == 'SEMI_ISOTROPIC':
            # Stop after step 10 and continue to 20; the area columns must
            # be identical to the uninterrupted run, including truncation.
            continued = scratch / 'continued.toml'
            continued.write_text(text.replace(name, 'continued').replace('checkpoint_interval = 20', 'checkpoint_interval = 10'))
            run(continued, '--continue', '--max-walltime', '0.000001', status=75)
            run(continued, '--continue')
            assert (scratch / 'continued.energy').read_bytes() == (scratch / (name + '.energy')).read_bytes()

# A tilted b vector catches using |a| |b| instead of the face area.
gro = scratch / 'tilted.gro'
lines = (source / 'Inputs/triclinic/dodecahedron.gro').read_text().splitlines()
cell = lines[-1].split()
cell[5] = '0.05'  # b_x in nm
lines[-1] = ' '.join(cell)
gro.write_text('\n'.join(lines) + '\n')
base = extract('triclinic-npt.test').replace('steps      = 200', 'steps      = 20')
base = base.replace('time_step  = 0.002', 'time_step  = 0.0002')
base = base.replace('energy_interval     = 50', 'energy_interval     = 10')
base = base.replace('checkpoint_interval = 100', 'checkpoint_interval = 20')
base = base.replace('"water.top"', f'"{source / "Inputs/triclinic/water.top"}"')
base = base.replace('"dodecahedron.gro"', '"tilted.gro"')
base = base.replace('[output]', '[output]\nenergy = "NAME.energy"')
for precision in ['double', 'mixed']:
    name = precision + '-triclinic'
    text = base.replace('whole.h5', name + '.h5').replace('NAME', name)
    text = text.replace('"CPU"', f'"{target}"').replace('"DOUBLE"', f'"{precision}"')
    control = scratch / (name + '.toml')
    control.write_text(text)
    check(name, run(control))
    assert columns(scratch / (name + '.energy'))[0][-1] == 676.0
print('cell area, units, log agreement, and exact continuation: ok')

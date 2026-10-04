"""Analytic oracle for U = lambda_wall*k*|x2-x1|^2 (#67).

Two uncharged, noninteracting argon atoms: r^2 = 27 Angstrom^2,
k = 2 kcal/mol/Angstrom^2, lambda = 0.25. Thus U = 13.5 kcal/mol,
dU/dlambda = 54 kcal/mol, and state differences are -13.5, 0, 40.5.
F1 = 2*lambda*k*(x2-x1) = (5,1,-1) kcal/mol/Angstrom; F2 = -F1.
Also check dU/dlambda against a central difference with h = 0.1.
"""
from pathlib import Path
import re
import subprocess
import sys

scratch, target, precision, mdir, source = sys.argv[1:]
scratch = Path(scratch)
scratch.mkdir(parents=True, exist_ok=True)
text = Path(source).read_text()
for name, end in [('two.top', 'two.gro'), ('two.gro', 'run.toml')]:
    data = text.split('#--- ' + name + '\n')[1].split('#--- ' + end)[0]
    (scratch / name).write_text(data)

def run(name, lam):
    control = f'''[input]
topology = "two.top"
coordinates = "two.gro"
[output]
energy_interval = 1
free_energy = "{name}.dhdl"
checkpoint = "{name}.h5"
checkpoint_interval = 1
[free_energy]
state = 1
[free_energy.lambdas]
wall = [0.0, {lam}, 1.0]
[energy]
cutoff = 9.0
[[energy.compound]]
name = "spring"
expression = "lambda_wall*k*distance(p1,p2)^2"
particles = [[1,2]]
k = 2.0
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 1e-9
steps = 1
[ensemble]
ensemble = "NVE"
temperature = 0.0
[boundary]
type = "NONE"
[execution]
target = "{target}"
precision = "{precision}"
threads = 1
'''
    (scratch / (name + '.toml')).write_text(control)
    result = subprocess.run([mdir, 'run', name + '.toml'], cwd=scratch,
                            text=True, capture_output=True, check=True)
    (scratch / (name + '.log')).write_text(result.stdout)
    energies = [float(v) for v in re.findall(r'MDIR:\s+spring\s+([-+\d.eE]+)', result.stdout)]
    assert energies, result.stdout
    rows = [list(map(float, line.split())) for line in
            (scratch / (name + '.dhdl')).read_text().splitlines()
            if line and not line.startswith('#')]
    return energies[0], rows[0]

energy, row = run('spring', 0.25)
# All numbers in kcal/mol; checkpoint force output is kJ/mol/nm.
tol = 2e-5 if precision == 'MIXED' else 1e-6
checks = [('energy', energy, 13.5), ('dHdl', row[2], 54.0)]
checks += [(f'dU.{i}', value, ref) for i, (value, ref) in
           enumerate(zip(row[3:], [-13.5, 0.0, 40.5]))]
assert len(row) == 6, row
for quantity, value, ref in checks:
    error = abs(value - ref)
    assert error <= tol, (quantity, value, ref, tol)
    print(f'{quantity}: reference {ref:g}, difference {error:.3g}, tolerance {tol:g}')

force_text = subprocess.check_output([mdir, 'checkpoint', '--print=forces',
                                      str(scratch / 'spring.h5')], text=True)
forces = [list(map(float, line.split()[2:])) for line in force_text.splitlines()
          if line and not line.startswith('#')]
assert len(forces) == 2, force_text
reference = [[v * 41.84 for v in force] for force in [[5, 1, -1], [-5, -1, 1]]]
force_error = max(abs(a-b) for force, ref in zip(forces, reference)
                  for a, b in zip(force, ref))
force_tol = 2e-3 if precision == 'MIXED' else 1e-6
assert force_error <= force_tol, (forces, reference, force_error)
print(f'forces: reference max 209.2 kJ/mol/nm, difference {force_error:.3g}, tolerance {force_tol:g}')
plus, _ = run('plus', 0.35)
minus, _ = run('minus', 0.15)
derivative = (plus-minus)/0.2
assert abs(row[2]-derivative) <= tol, (row[2], derivative)
print(f'central difference: reference {derivative:g}, difference {abs(row[2]-derivative):.3g}')
print('compound lambda: ok')

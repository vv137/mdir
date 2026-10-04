"""Declaration regressions for #67, including names unused by expressions."""
from pathlib import Path
import subprocess
import sys

scratch, mdir, inputs = sys.argv[1:]
scratch = Path(scratch)
scratch.mkdir(parents=True, exist_ok=True)
base = f'''[output]
energy_interval = 1
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 1
[ensemble]
ensemble = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[input]
topology = "{inputs}/dipeptide/dipeptide.prmtop"
coordinates = "{inputs}/dipeptide/dipeptide.inpcrd"
[free_energy.lambdas]
wall = [0.0, 1.0]
[energy]
cutoff = 9.0
[[energy.parameter]]
name = "w"
value = 1.0
[[energy.function]]
name = "tablef"
values = [0.0, 1.0]
min = 0.0
max = 10.0
'''
terms = {
    "pair": ('r', '', ['r', 't', 'q1', 'q2', 'sigma', 'epsilon',
                         'sigma1', 'sigma2', 'epsilon1', 'epsilon2', 'coulomb',
                         'w1', 'w2']),
    "bond": ('r', 'particles = [[2,19]]', ['r', 't', 'w1', 'w2']),
    "angle": ('theta', 'particles = [[2,5,7]]', ['theta', 't', 'w1', 'w3']),
    "dihedral": ('theta', 'particles = [[2,5,7,9]]', ['theta', 't', 'w1', 'w4']),
    "compound": ('distance(p1,p2)', 'particles = [[2,5,7,9,11]]',
                 ['t', 'w1', 'w5', 'distance__1_2']),
    "external": ('x', 'particles = [2]', ['x', 'y', 'z', 'q', 't']),
    "centroid": ('r', 'groups = ["@2", "@19"]', ['r', 'dx', 'dy', 'dz', 't']),
}
count = 0
for kind, (expression, particles, reserved) in terms.items():
    table = 'bond' if kind == 'centroid' else kind
    for name in reserved + ['w', 'tablef', 'sin', 'lambda_wall']:
        # A list and a scalar, both used and unused where syntactically valid.
        for value in (['1.0', '[1.0]'] if table != 'pair' else ['1.0']):
            for used in (False, True):
                expr = expression + ('+' + name if used else '')
                path = scratch / 'collision.toml'
                path.write_text(base + f'''[[energy.{table}]]
name = "term"
expression = "{expr}"
{particles}
{name} = {value}
''')
                result = subprocess.run([mdir, 'check', str(path)],
                                        text=True, capture_output=True)
                assert result.returncode != 0, (kind, name, value, used)
                assert (f"'{name}'" in result.stderr and 'not' in result.stderr
                        or '__' in name and 'reserved' in result.stderr), result.stderr
                count += 1

# Triplet variables and type parameters are also protected before use.
triplet_base = '''[input]
coordinates = "unused.pdb"
[energy]
cutoff = 9.0
[[energy.type]]
name = "A"
mass = 1.0
w = 1.0
[[energy.function]]
name = "tablef"
values = [0.0, 1.0]
min = 0.0
max = 10.0
[[energy.triplet]]
expression = "r12+r13"
cutoff = 9.0
'''
for name in ['r12', 'r13', 'r23', 'theta', 't', 'w1', 'w2', 'w3', 'tablef', 'sin']:
    path.write_text(triplet_base + f'{name} = 1.0\n')
    result = subprocess.run([mdir, 'check', str(path)], text=True, capture_output=True)
    assert result.returncode != 0 and f"'{name}'" in result.stderr, result.stderr
    count += 1

# An undeclared lambda must still fail; unrelated parameter names stay valid.
valid = base + '''[[energy.compound]]
name = "term"
expression = "lambda_wall*k*distance(p1,p2)^2"
particles = [[2,19]]
k = 1.0
'''
path.write_text(valid)
result = subprocess.run([mdir, 'check', str(path)], text=True, capture_output=True)
assert result.returncode == 0, result.stderr
path.write_text(valid.replace('lambda_wall*k', 'lambda_unknown*k'))
result = subprocess.run([mdir, 'check', str(path)], text=True, capture_output=True)
assert result.returncode != 0 and "'lambda_unknown'" in result.stderr, result.stderr
print(f'namespace collisions: {count} rejected; compound lambda accepted')

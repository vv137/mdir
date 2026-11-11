"""Device-free native emission contracts; no kernel is compiled or run."""
import pathlib
import subprocess
import sys

tool, opt, fixtures, work = sys.argv[1:]
fixtures, work = pathlib.Path(fixtures), pathlib.Path(work)
base = f'''
[input]
topology = "{fixtures / 'three.prmtop'}"
coordinates = "{fixtures / 'three.inpcrd'}"
[output]
energy_interval = 1
[energy]
cutoff = 9.0
pairlist_distance = 10.0
electrostatics = "CUTOFF"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 1
[ensemble]
ensemble = "NVE"
temperature = 300.0
[constraints]
rigid_water = true
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
'''


def emit(name, text):
    control = work / f'{name}.toml'
    control.write_text(text)
    result = subprocess.run([tool, 'emit', '--stage=module', str(control)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    if result.returncode:
        raise SystemExit(f'{name}: {result.stderr}')
    (work / f'{name}.mlir').write_text(result.stdout)
    return result.stdout


on = emit('on', base + '\n[performance]\n"water.template" = "ON"\n')
for operation in ('md.template_pairs', 'md.sum_template_window', 'template_water'):
    if operation not in on:
        raise SystemExit(f'ON did not construct {operation}')
if 'md.template_constraint_epoch' not in on:
    raise SystemExit('ON did not record native rigid-water constraint provenance')
derived = subprocess.run([opt, str(work / 'on.mlir'), '--md-differentiate', '--verify-each'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if derived.returncode:
    raise SystemExit(f'ON first derivatives: {derived.stderr}')
if 'md.gather_template_window' not in derived.stdout:
    raise SystemExit('ON did not derive the native template force window')
if 'md.template_constraint_epoch' not in derived.stdout:
    raise SystemExit('differentiation lost native rigid-water constraint provenance')
(work / 'on-derived.mlir').write_text(derived.stdout)
off = emit('off', base + '\n[performance]\n"water.template" = "OFF"\n')
auto = emit('auto', base)
if off != auto:
    raise SystemExit('ordinary OFF/AUTO programs differ')
if 'md.sum_template_window' in off or 'template_water' in off:
    raise SystemExit('ordinary program acquired a template window')
if 'md.template_constraint_epoch' in off:
    raise SystemExit('ordinary program acquired template constraint provenance')
# A double-precision program takes the general path: AUTO constructs the
# OFF program, and ON is refused, naming the precision.
double = base.replace('precision = "MIXED"', 'precision = "DOUBLE"')
assert double != base
double_auto = emit('double-auto', double)
if double_auto != emit('double-off', double + '\n[performance]\n"water.template" = "OFF"\n'):
    raise SystemExit('double-precision AUTO/OFF programs differ')
if any(op in double_auto for op in ('md.template_pairs', 'md.sum_template_window', 'template_water')):
    raise SystemExit('double-precision AUTO constructed the template')
control = work / 'double-on.toml'
control.write_text(double + '\n[performance]\n"water.template" = "ON"\n')
refused = subprocess.run([tool, 'emit', '--stage=module', str(control)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if refused.returncode == 0 or 'MIXED precision is required' not in refused.stderr:
    raise SystemExit(f'double-precision ON was not refused: {refused.stderr}')
print('native water-template ON constructed and differentiated; ordinary OFF/AUTO programs identical')
print('double precision: AUTO takes the general path, ON is refused')

"""Native R1 lowering gate; stops before GPU compilation or execution."""
import pathlib
import re
import subprocess
import sys

tip3p = '--tip3p' in sys.argv
arguments = [x for x in sys.argv[1:] if x != '--tip3p']
mdir, opt, fixtures, work = arguments
fixtures, work = pathlib.Path(fixtures), pathlib.Path(work)
work.mkdir(parents=True, exist_ok=True)

def invoke(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise SystemExit(f'{command}: {result.stderr}')
    return result.stdout

# The existing fixture defines the unchanged physical model. Only the
# integrator is varied, so this checks the real native builder provenance.
for integrator in ('VELOCITY_VERLET', 'LEAPFROG'):
    control = work / (integrator + '.toml')
    control.write_text(f'''[input]
topology = "{fixtures / 'three.prmtop'}"
coordinates = "{fixtures / 'three.inpcrd'}"
[output]
energy_interval = 1
[energy]
cutoff = 9.0
pairlist_distance = 10.0
electrostatics = "CUTOFF"
[dynamics]
integrator = "{integrator}"
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
[performance]
"water.template" = "ON"
''')
    if tip3p:
        invoke([sys.executable, str(pathlib.Path(__file__).with_name('water_template_tip3p.py')),
                str(fixtures), str(work)])
        text = (work / 'control.toml').read_text()
        text = text.replace('"three.top"', '"' + str(work / 'three.top') + '"')
        text = text.replace('"three.gro"', '"' + str(work / 'three.gro') + '"')
        text = text.replace('steps = 2', 'steps = 1')
        text = text.replace('integrator = "VELOCITY_VERLET"', 'integrator = "' + integrator + '"')
        control.write_text(text)
    module = invoke([mdir, 'emit', '--stage=module', str(control)])
    pipeline = invoke([mdir, 'emit', '--stage=pipeline', str(control)]).strip()
    # Keep the actual configured pipeline through execution-to-GPU lowering;
    # no PTX, cubin, or device probe is needed for this structural contract.
    match = re.search(r'convert-md-exec-to-gpu\{[^}]*\}', pipeline)
    assert match, 'native GPU lowering missing from configured pipeline'
    prefix = pipeline[:match.end()]
    source = work / (integrator + '.mlir')
    source.write_text(module)
    lowered = invoke([opt, str(source), '--pass-pipeline=builtin.module(' + prefix + ')'])
    (work / (integrator + '-gpu.mlir')).write_text(lowered)
    assert 'md_exec.' not in lowered, 'native execution operation was left unlowered'
    names = re.findall(r'gpu.launch[^\n]*function\(@([^)]*)\)', lowered)
    fused = [name for name in names if '_integration' in name and '_template_publication' in name]
    model = 'TIP3P' if tip3p else 'OPC'
    if tip3p and integrator == 'VELOCITY_VERLET':
        # One entry-time acrossCount clear plus the two original work
        # kernels. The clear carries the same integration purpose name.
        assert len(fused) == 3, ('TIP3P VV fusion closure missing', names)
        assert all('_particle_for_integration' in name for name in fused), fused
        bodies = []
        for match in re.finditer(r'gpu.launch[^\n]*function\(@([^)]*)\)', lowered):
            if match[1] not in fused:
                continue
            begin = lowered.index('{', match.end())
            depth, end = 1, begin + 1
            while depth:
                depth += (lowered[end] == '{') - (lowered[end] == '}')
                end += 1
            bodies.append(lowered[begin + 1:end - 1])
        clears = [body for body in bodies if body.count('memref.store') == 1
                  and 'gpu.shuffle' not in body and 'math.' not in body]
        assert len(clears) == 1, 'fused closure must have exactly one count clear'
        assert 'mdrtFlagStart' in lowered and 'mdrtFlagFinish' in lowered
        print(model + ' ' + integrator + ': native complete-output integration publication selected')
    else:
        # OPC's actual final template writer follows a separate extra-point
        # tuple/update, so the marked SETTLE run does not own publication.
        # TIP3P LF's predictor already closes the preceding velocity run;
        # the original inRows precondition prevents using it in two runs.
        reason = 'extra-point final writer' if not tip3p else 'predictor already owned by preceding run'
        assert not fused, (model, integrator, 'expected proven standalone fallback', reason, names)
        assert any('_integration' in name for name in names), 'ordinary integration fusion disappeared'
        print(model + ' ' + integrator + ': standalone publication retained (' + reason + ')')

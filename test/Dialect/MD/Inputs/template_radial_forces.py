"""Check radial construction, all class layouts, and the anisotropic fallback."""
from pathlib import Path
import re
import subprocess
import sys

binary, source = sys.argv[1:]
original = Path(source).read_text()
for kind, width in [('ww', 8), ('sw', 5), ('ss', 2), ('anisotropic', 6)]:
    text = original
    if kind == 'anisotropic':
        text = text.replace('%e = arith.mulf %half, %r2 : f64',
                            '%axis = vector.extract %d[0] : f64 from vector<3xf64>\n'
                            '    %e = arith.mulf %axis, %axis : f64')
    elif kind in ('sw', 'ss'):
        begin = text.index('  ^bb0(')
        end = text.index('    %square =', begin)
        if kind == 'sw':
            body = '''  ^bb0(%base: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):
    %alpha = arith.constant 0.75 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %bs = arith.addf %b1, %b2 : vector<3xf64>
    %pb = arith.mulf %weight, %bs : vector<3xf64>
    %d = arith.subf %base, %pb : vector<3xf64>
'''
        else:
            body = '''  ^bb0(%d: vector<3xf64>):
'''
        text = text[:begin] + body + text[end:]
        text = text.replace('md.sum_template_window %ww,', f'md.sum_template_window %{kind},')
        text = text.replace('} : !ww, !x -> f64', f'}} : !{kind}, !x -> f64')
    args = [binary, '--md-differentiate', '--md-inline', '--convert-md-to-md-exec',
            '--md-exec-fuse-loops', '--md-exec-assign-precision=mode=mixed',
            '--canonicalize', '--cse', '--verify-each']
    run = subprocess.run(args, input=text, text=True, capture_output=True)
    assert run.returncode == 0, (kind, run.stderr)
    output = run.stdout
    assert ('md.template_site_forces' in output) == (kind != 'anisotropic'), kind
    # Fusion retains one cutoff displacement for each of force, energy and virial.
    yields = re.findall(r'md_exec.yield[^\n]* : ([^\n]+)', output)
    assert any(types.count('vector<3xf32>') == width + 3 for types in yields), kind
    if kind != 'anisotropic':
        assert 'dense<[1.000000e+00, 0.000000e+00, 0.000000e+00]>' not in output, kind
# The same force destination can combine radial and nonradial windows.
# Its mixed layout must recover real-member forces before concatenation.
start = original.index('  %u = md.sum_template_window')
end = original.index('  md.return %u', start)
second = original[start:end].replace('%u =', '%v =')
second = second.replace('%e = arith.mulf %half, %r2 : f64',
                        '%axis = vector.extract %d[0] : f64 from vector<3xf64>\n'
                        '    %e = arith.mulf %axis, %axis : f64')
text = original.replace('  md.return %u : f64',
                        second + '  %total = arith.addf %u, %v : f64\n'
                        '  md.return %total : f64')
run = subprocess.run([binary, '--md-differentiate', '--md-inline',
                      '--convert-md-to-md-exec', '--md-exec-fuse-loops',
                      '--md-exec-accumulate-destinations', '--verify-each'],
                     input=text, text=True, capture_output=True)
assert run.returncode == 0, run.stderr
assert 'md_exec.template_for' in run.stdout
# Invalid flags are rejected by name rather than indexing an unchecked row.
radial = subprocess.run([binary, '--md-differentiate', '--md-inline',
                         '--convert-md-to-md-exec'], input=original,
                        text=True, capture_output=True)
assert radial.returncode == 0, radial.stderr
for flags in ['1, 1', '2, 1, 1']:
    mutated = radial.stdout.replace('array<i32: 1, 1, 1>', f'array<i32: {flags}>')
    refused = subprocess.run([binary], input=mutated, text=True, capture_output=True)
    assert refused.returncode != 0 and 'binary site-force flags' in refused.stderr, refused.stderr
print('radial construction: WW/SW/SS and anisotropic fallback passed')

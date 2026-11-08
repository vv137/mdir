"""Mutate only the new reduction; ordinary refusal fixtures run separately."""
import pathlib
import subprocess
import sys

tool, source = sys.argv[1:]
text = pathlib.Path(source).read_text()
cases = [
    (text.replace('%fromA = arith.addf %base, %a1',
                  '%fromA = arith.mulf %base, %a1'),
     'cannot prove template-window site geometry'),
    (text.replace('extent(0.1)', 'extent(-0.1)'),
     'expected positive finite cutoff and extent'),
    (text.replace('alpha(0.125)', 'alpha(0.75)'),
     'expected a finite template alpha between zero and one half'),
    (text.replace('md.sum_template_window %ww,',
                  'md.sum_template_window %unproved,').replace(
                      '%water: !water) -> f64',
                      '%water: !water, %unproved: !ww) -> f64').split(
                          'md.function @entry')[0],
     "expected a proved class domain from 'md.template_pairs'"),
]
for index, (mutation, diagnostic) in enumerate(cases):
    result = subprocess.run([tool, '-'], input=mutation, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0 or diagnostic not in result.stderr:
        raise SystemExit(f'negative {index}: expected {diagnostic}\n{result.stderr}')
parameter = text.split('md.function @entry')[0].replace(
    '%water: !water) -> f64', '%water: !water, %q: f64) -> f64')
parameter += '''
md.function @parameter(%x: !x, %h: !md.cell, %types: !i,
                       %water: !water, %q: f64) -> f64 {
  %g = md.evaluate @window(%x, %h, %types, %water, %q) request [derivative(4)]
       : (!x, !md.cell, !i, !water, f64) -> f64
  md.return %g : f64
}
'''
result = subprocess.run([tool, '-', '--md-differentiate'], input=parameter,
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
diagnostic = 'supports energy, forces and virial only; higher derivatives and tunables require the general path'
if result.returncode == 0 or diagnostic not in result.stderr:
    raise SystemExit(f'parameter refusal: expected {diagnostic}\n{result.stderr}')
active_gather = text.replace('alpha(0.125) {',
    'alpha(0.125) gather(%x : !x) {').replace(
        '%b2: vector<3xf64>):', '%b2: vector<3xf64>, ' +
        ', '.join(f'%p{i}: vector<3xf64>' for i in range(6)) + '):')
cell_capture = text.replace('  %u = md.sum_template_window',
    '  %edges = md_exec.cell_edges %h : vector<3xf64>\n  %u = md.sum_template_window')
cell_capture = cell_capture.replace('    md.yield %e, %d',
    '    %edge = vector.extract %edges[0] : f64 from vector<3xf64>\n'
    '    %withCell = arith.addf %e, %edge : f64\n    md.yield %withCell, %d')
for mutation, diagnostic in [
    (active_gather, 'cannot differentiate with respect to the positions through this use'),
    (cell_capture, 'template-window captures must not hide position or cell dependence'),
]:
    result = subprocess.run([tool, '-', '--md-differentiate'], input=mutation,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0 or diagnostic not in result.stderr:
        raise SystemExit(f'activity refusal: expected {diagnostic}\n{result.stderr}')
print(f'template-window refusal contracts: {len(cases) + 3} passed')

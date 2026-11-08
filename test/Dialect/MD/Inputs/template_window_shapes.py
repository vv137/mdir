"""Verify all three class signatures and constant-alpha site placement."""
import pathlib
import subprocess
import sys

tool, source = sys.argv[1:]
text = pathlib.Path(source).read_text()
start = text.index('    %fromA =')
end = text.index('    %square =')
signature = '^bb0(%base: vector<3xf64>, %a1: vector<3xf64>, %a2: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):'
ep = '''    %alpha = arith.constant 0.125 : f64
    %weight = vector.broadcast %alpha : f64 to vector<3xf64>
    %asum = arith.addf %a1, %a2 : vector<3xf64>
    %bsum = arith.addf %b1, %b2 : vector<3xf64>
    %aoff = arith.mulf %weight, %asum : vector<3xf64>
    %boff = arith.mulf %weight, %bsum : vector<3xf64>
    %fromA = arith.addf %base, %aoff : vector<3xf64>
    %d = arith.subf %fromA, %boff : vector<3xf64>
'''
sw = text[:start] + '    %d = arith.subf %base, %b1 : vector<3xf64>\n' + text[end:]
sw = sw.replace(signature, '^bb0(%base: vector<3xf64>, %b1: vector<3xf64>, %b2: vector<3xf64>):')
sw = sw.replace('md.sum_template_window %ww,', 'md.sum_template_window %sw,').replace('} : !ww, !x', '} : !sw, !x')
ss = text[:start] + '''    %zero = arith.constant dense<0.0> : vector<3xf64>
    %d = arith.addf %base, %zero : vector<3xf64>
''' + text[end:]
ss = ss.replace(signature, '^bb0(%base: vector<3xf64>):')
ss = ss.replace('md.sum_template_window %ww,', 'md.sum_template_window %ss,').replace('} : !ww, !x', '} : !ss, !x')
wide = (text[:start] + ep + text[end:]).replace('0.125', '0.75')
for name, mutation in [('WW extra point', text[:start] + ep + text[end:]),
                       ('former alpha bound', text.replace('alpha(0.125)', 'alpha(0.75)')),
                       ('WW alpha 0.75', wide),
                       ('SW', sw), ('SS', ss)]:
    result = subprocess.run([tool, '-', '--md-differentiate', '--verify-each'],
                            input=mutation, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise SystemExit(f'{name}: {result.stderr}')
print('template-window shape contracts: 5 passed')

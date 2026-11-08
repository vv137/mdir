"""Both sides of the proved-root evaluator admission, without numerical runs."""
import argparse
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('optimizer')
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    original = args.source.read_text()

    def run(text, *options):
        return subprocess.run([args.optimizer, *options], input=text, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def expand(text):
        result = run(text, '--md-expand-reciprocal=functions=energy')
        assert result.returncode == 0, result.stderr
        return result.stdout

    expanded = expand(original)
    positive = run(expanded, '--md-differentiate', '--verify-each')
    assert positive.returncode == 0, positive.stderr
    assert 'md.evaluate' not in positive.stdout
    passed = 1

    def reject(name, text, *diagnostics):
        nonlocal passed
        result = run(text, '--md-differentiate', '--verify-each')
        assert result.returncode != 0, (name, 'incorrectly admitted')
        for diagnostic in diagnostics:
            assert diagnostic in result.stderr, (name, diagnostic, result.stderr)
        assert 'error: ' in result.stderr, (name, result.stderr)
        passed += 1

    def replace(text, pattern, replacement):
        changed, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
        assert count == 1, pattern
        return changed

    reject('kernel arithmetic', expanded.replace('math.exp', 'math.sin', 1),
           'Gaussian product kernel does not match its declared arithmetic')
    reject('missing parameter', replace(expanded, r'md.reciprocal_beta = [^,}]+, ', ''),
           'expected explicit Gaussian product parameters and operands')
    reject('forged parameter', replace(expanded, r'md.reciprocal_beta = [^,}]+',
                                      'md.reciprocal_beta = 1.8 : f64'),
           'Gaussian product kernel does not match its declared arithmetic')
    no_parameters = replace(expanded, r'\{md.reciprocal_beta = [^}]+\}', '{}')
    reject('no proof parameters', no_parameters,
           'unsupported stencil VJP for active cell operand in ordinary virial composition')
    reject('spectral energy', expanded.replace('arith.constant 5.000000e-01 : f64',
                                              'arith.constant 2.500000e-01 : f64', 1),
           'expected the Hermitian spectral half-inner-product energy kernel')
    reject('grid identity', replace(expanded,
           r'(md.map_grid[^\n]+grid\()\[4, 5, 6\]', r'\g<1>[4, 5, 7]'),
           'expected a shared forward transform and its explicit Gaussian product')

    transform = re.search(r'^\s*(%\w+) = md.linear_transform [^\n]+', expanded, re.M)
    assert transform
    copied = transform[0].replace(transform[1]+' =', '%case_fft =', 1)
    other_fft = expanded[:transform.end()]+'\n'+copied+expanded[transform.end():]
    other_fft = other_fft.replace('md.map_grid('+transform[1]+',', 'md.map_grid(%case_fft,', 1)
    reject('shared transform', other_fft,
           'expected a shared forward transform and its explicit Gaussian product')

    checked = re.search(r'^\s*(%\w+) = md.checked_rectangular_cell (%\w+)\n', expanded, re.M)
    assert checked
    unchecked = expanded[:checked.start()]+'\n'+expanded[checked.end():]
    unchecked = re.sub(re.escape(checked[1])+r'(?![\w#])', checked[2], unchecked)
    reject('rectangular proof', unchecked,
           'reciprocal grid composition requires the checked rectangular-cell precondition')
    reject('matrix provenance', expanded.replace('arith.constant 0.000000e+00 : f64',
                                                'arith.constant 1.000000e+00 : f64', 1),
           'expected one assignment and its exact explicit row-cell matrix')
    reject('window order', expanded.replace('order(4)', 'order(3)'),
           'reciprocal grid derivatives currently require order-four B-splines')
    changed_window = replace(expanded,
        r'^(\s*)(%\w+ = md.bspline_window (%\w+) order\(4\)[^\n]+)\n\s*md.yield (%\w+) : f64',
        lambda m: (m[1]+m[2]+'\n'+m[1]+'%case_two = arith.constant 2.0 : f64\n'+
                   m[1]+'%case_weight = arith.mulf '+m[4]+', %case_two : f64\n'+
                   m[1]+'md.yield %case_weight : f64'))
    reject('common window body', changed_window,
           'reciprocal assignment requires the common order-four cardinal window')

    def before_return(text, operations, value=None):
        def inject(m):
            returned = value(m[2]) if value else m[2]
            return operations+'\n'+m[1]+'md.return '+returned+' : f64'
        return replace(text, r'^(\s*)md.return (%\w+) : f64', inject)

    rho = re.search(r'(%\w+) = md.scatter_stencil', expanded)[1]
    tensor_escape = before_return(expanded,
        '    %case_index = arith.constant 0 : index\n'
        f'    %case_read = tensor.extract {rho}[%case_index, %case_index, %case_index] : tensor<4x5x6xf64>',
        lambda value: value)
    reject('tensor escape', tensor_escape,
           'unsupported reciprocal grid evaluator composition: unproved use escapes the spectral graph')
    matrix = re.search(r'(%\w+) = vector.from_elements[^\n]+vector<9xf64>', expanded)[1]
    matrix_escape = before_return(expanded,
        f'    %case_read = vector.extract {matrix}[0] : f64 from vector<9xf64>')
    reject('matrix escape', matrix_escape,
           'unsupported reciprocal grid evaluator composition: unproved use escapes the spectral graph')

    for name, ty, source_operand in [('other positions', '!x', '%y'),
                                      ('other cell', '!md.cell', '%other_cell')]:
        other = original.replace('%m: !m)', f'%m: !m, {source_operand}: {ty})')
        other = other.replace('@energy(%x, %h, %q, %m)',
                              f'@energy(%x, %h, %q, %m, {source_operand})')
        other = other.replace('(!x, !md.cell, !q, !m)', f'(!x, !md.cell, !q, !m, {ty})')
        if ty == '!x':
            other = other.replace('md.reciprocal %x,', 'md.reciprocal %y,')
        else:
            other = other.replace('md.reciprocal %x, %q, %h,',
                                  'md.reciprocal %x, %q, %other_cell,')
        reject(name, expand(other),
               "unsupported reciprocal grid evaluator geometry: expected the potential's direct positions and cell")

    active_map = '''  %case_q = md.map_particles gather(%x, %q : !x, !q) {
  ^bb0(%case_x: vector<3xf64>, %case_qi: f64):
    %case_component = vector.extract %case_x[0] : f64 from vector<3xf64>
    %case_value = arith.addf %case_component, %case_qi : f64
    md.yield %case_value : f64
  } : !q
'''
    active = original.replace('  %u, %f, %w = md.reciprocal',
                              active_map+'  %u, %f, %w = md.reciprocal', 1)
    active = active.replace('md.reciprocal %x, %q,', 'md.reciprocal %x, %case_q,', 1)
    reject('active upstream amplitudes', expand(active),
           'position/cell-dependent assignment amplitudes require an upstream derivative rule')
    inactive_map = '''  %case_q = md.map_particles gather(%q : !q) {
  ^bb0(%case_qi: f64):
    %case_value = arith.mulf %case_qi, %case_qi : f64
    md.yield %case_value : f64
  } : !q
'''
    upstream = original.replace('  %u, %f, %w = md.reciprocal',
                                inactive_map+'  %u, %f, %w = md.reciprocal', 1)
    upstream = upstream.replace('md.reciprocal %x, %q,', 'md.reciprocal %x, %case_q,', 1)
    reject('upstream field VJP', expand(upstream), 'md.map_particles',
           'a field, and has no rule for the derivative in it')

    bare = '''    %case_relation = md.grid_stencil %arg0, %arg1 : !md.field<@atoms, 3 x f64> -> !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>
    %case_bare = md.scatter_stencil %case_relation, %arg0, %arg1, %arg2 {
    ^bb0(%case_distance: vector<3xf64>):
      %case_weight = md.bspline_window %case_distance order(4) : vector<3xf64>
      md.yield %case_weight : f64
    } : !md.grid_stencil<@atoms, [4, 5, 6], order(4), window("bspline")>, !md.field<@atoms, 3 x f64>, !md.field<@atoms, f64> -> tensor<4x5x6xf64>'''
    reject('proved plus bare', before_return(expanded, bare),
           'unsupported stencil VJP for active cell operand in ordinary virial composition')

    potential = original[:original.index('md.function @caller')]
    tangent = potential+'''md.function @caller(%x: !x, %h: !md.cell, %q: !q, %m: !m, %t: !x) -> f64 {
  %value = md.evaluate @energy(%x, %h, %q, %m, %t) request [tangent]
      : (!x, !md.cell, !q, !m, !x) -> f64
  md.return %value : f64
}
'''
    reject('higher composition', expand(tangent),
           'cannot form the tangent in the positions through this use')
    print(f'evaluator admission/refusal contracts: {passed} passed')


if __name__ == '__main__':
    main()

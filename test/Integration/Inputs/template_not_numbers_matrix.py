"""Lower or execute the approved D107 poison matrix exactly once per case."""
import argparse
import itertools
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('work', type=Path)
p.add_argument('--lower', required=True)
p.add_argument('--printer', required=True)
p.add_argument('--mlir-opt', required=True)
p.add_argument('--runner', required=True)
p.add_argument('--libs', default='')
p.add_argument('--lower-only', action='store_true')
a = p.parse_args()
a.work.mkdir(parents=True, exist_ok=True)
cases = list(itertools.product(['cold', 'writer'], ['oxygen', 'hydrogen', 'solute'],
                               ['nan', 'infinity', 'huge']))
cases += [(phase, 'two-hydrogens', 'nan') for phase in ['cold', 'writer']]
for phase, member, value in cases:
    stem = a.work / f'{phase}-{member}-{value}'
    source, gpu, llvm = [str(stem) + suffix for suffix in ['.mlir', '.gpu.mlir', '.llvm.mlir']]
    subprocess.run([sys.executable, str(Path(__file__).with_name('template_not_numbers.py')),
                    source, phase, member, value], check=True)
    subprocess.run([a.printer, source, '--convert-md-exec-to-gpu', '-o', gpu], check=True)
    subprocess.run([a.mlir_opt, gpu, *shlex.split(a.lower), '-o', llvm], check=True)
    if not a.lower_only:
        run = subprocess.run([a.runner, llvm, '-e', 'main', '--entry-point-result=void',
                              '--shared-libs=' + a.libs], capture_output=True, text=True, timeout=120)
        output = run.stdout + run.stderr
        Path(str(stem) + '.output').write_text(output)
        count = 2 if member == 'two-hydrogens' else 1
        expected = ('a position is not a number or is' if count == 1 else
                    '2 positions are not numbers or are')
        expected += ' far outside the cell at a build of the neighbor structure; the run has failed'
        assert run.returncode == 1 and expected in output, (phase, member, value, run.returncode, output)
    print(f'{phase}/{member}/{value}: ' + ('lowered' if a.lower_only else f'named D107 stop, count {count}'), flush=True)
print(f'D107 template matrix: {len(cases)} ' + ('lowerings passed' if a.lower_only else 'named stops passed'))

"""Registers, spills (local memory), and shared memory of the gathering
kernels of the prototype, from cuobjdump --dump-resource-usage, for the
(P, D, lanes) of the cases of P0 (#337).  resources.py BINARY ARCH"""
import os, re, subprocess, sys

# The CUDA toolkit from CUDA_ROOT (docs/environment.md, Scripts).
if 'CUDA_ROOT' not in os.environ:
    sys.exit('resources.py: set CUDA_ROOT to the CUDA toolkit')
tools = os.path.join(os.environ['CUDA_ROOT'], 'bin')
binary, arch = sys.argv[1], sys.argv[2]
out = subprocess.run([os.path.join(tools, 'cuobjdump'), '--dump-resource-usage', '--gpu-architecture', arch, binary],
                     capture_output=True, text=True).stdout
out = re.sub(r'(Function \S+):', r'\1', out)
names = subprocess.run([os.path.join(tools, 'cu++filt')], input=out, capture_output=True, text=True).stdout.splitlines()
wanted = {(4, 4), (4, 11), (5, 5), (5, 10), (6, 6), (6, 9), (8, 8)}
for i, line in enumerate(names):
    m = re.search(r'Function (?:void )?(gather|gatherPre)<float, ((?:\(int\)\d+(?:, )?)+)>', line)
    if not m:
        continue
    kind = m.group(1)
    ints = [int(v) for v in re.findall(r'\(int\)(\d+)', m.group(2))]
    if kind == 'gather':
        P, D, G = ints
    else:
        (P, G), D = ints, 0
    if kind == 'gather' and (P, D) not in wanted:
        continue
    if G not in (8, 32):
        continue
    res = names[i + 1]
    reg = re.search(r'REG:(\d+)', res).group(1)
    stack = re.search(r'STACK:(\d+)', res).group(1)
    local = re.search(r'LOCAL:(\d+)', res).group(1)
    print(f'{kind:10s} P {P} D {D:2d} lanes {G:2d}  registers {reg:>3s}  stack {stack:>4s}  local {local:>4s}')

# Writes JAC as flat binary for the tile prototype: n, box, x (f64), charge (f32, scaled by
# sqrt(332.0522)), type (i32), ntypes, A and B tables (f32), exclusions (CSR, both directions).
import re, struct, sys
top, crd, out = sys.argv[1:4]
flags = {}; cur = None; fmt = None
for line in open(top):
    if line.startswith('%FLAG'):
        cur = line.split()[1]; flags[cur] = []; continue
    if line.startswith('%FORMAT'):
        m = re.search(r'\((\d+)([aIEF])(\d+)', line); fmt = (int(m.group(1)), m.group(2), int(m.group(3))); continue
    if line.startswith('%') or cur is None: continue
    s = line.rstrip('\n'); w = fmt[2]
    for k in range(0, len(s), w):
        tok = s[k:k+w].strip()
        if tok: flags[cur].append(tok)
P = [int(v) for v in flags['POINTERS']]
n, ntypes = P[0], P[1]
q = [float(v) / 18.2223 * (332.0522 ** 0.5) for v in flags['CHARGE']]
t = [int(v) - 1 for v in flags['ATOM_TYPE_INDEX']]
nbi = [int(v) for v in flags['NONBONDED_PARM_INDEX']]
A = [float(v) for v in flags['LENNARD_JONES_ACOEF']]
B = [float(v) for v in flags['LENNARD_JONES_BCOEF']]
ta = []; tb = []
for i in range(ntypes):
    for j in range(ntypes):
        k = nbi[i * ntypes + j] - 1
        ta.append(A[k] if k >= 0 else 0.0); tb.append(B[k] if k >= 0 else 0.0)
nex = [int(v) for v in flags['NUMBER_EXCLUDED_ATOMS']]
exl = [int(v) - 1 for v in flags['EXCLUDED_ATOMS_LIST']]
ex = [set() for _ in range(n)]; p = 0
for i in range(n):
    for k in range(nex[i]):
        j = exl[p + k]
        if j >= 0: ex[i].add(j); ex[j].add(i)
    p += nex[i]
lines = open(crd).read().split('\n')
vals = []
for line in lines[2:]:
    for k in range(0, len(line), 12):
        tok = line[k:k+12].strip()
        if tok: vals.append(float(tok))
x = vals[:3 * n]; box = vals[-6:-3]
with open(out, 'wb') as f:
    f.write(struct.pack('ii', n, ntypes)); f.write(struct.pack('3d', *box))
    f.write(struct.pack(f'{3*n}d', *x)); f.write(struct.pack(f'{n}f', *q)); f.write(struct.pack(f'{n}i', *t))
    f.write(struct.pack(f'{ntypes*ntypes}f', *ta)); f.write(struct.pack(f'{ntypes*ntypes}f', *tb))
    off = [0]
    for s in ex: off.append(off[-1] + len(s))
    f.write(struct.pack(f'{n+1}i', *off))
    flat = [j for s in ex for j in sorted(s)]
    f.write(struct.pack(f'{len(flat)}i', *flat))
print(n, ntypes, box, len(flat))

"""Writes the coordinates of an Amber file moved by half the box along each
edge and folded into the box, [0, L), so that the molecules near its middle
lie across its faces.

    shift_inpcrd.py INPCRD > SHIFTED"""
import sys

lines = open(sys.argv[1]).read().split('\n')
n = int(lines[1].split()[0])
values = [float(lines[2 + i // 6][12 * (i % 6):12 * (i % 6) + 12]) for i in range(3 * n)]
box_line = [l for l in lines if l.strip()][-1]
box = [float(v) for v in box_line.split()[:3]]
shifted = [(v + 0.5 * box[i % 3]) % box[i % 3] for i, v in enumerate(values)]
out = [lines[0], '%6d' % n]
for k in range(0, len(shifted), 6):
    out.append(''.join('%12.7f' % v for v in shifted[k:k + 6]))
out.append(box_line)
print('\n'.join(out))

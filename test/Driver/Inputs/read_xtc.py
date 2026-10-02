"""Reads a trajectory in XTC and compares it with one in DCD of the same
run: the steps and times of the frames, the cell, and every position,
within half the precision of the file (1/1000 nm) and the rounding of
32-bit numbers.

    read_xtc.py XTC DCD PERIOD TIMESTEP

The decoder is written here on its own, from the format as GROMACS reads
it, so that it checks MDIR's writer (D141)."""
import math, struct, sys

MAGIC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
         80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
         1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
         16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
         131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
         832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
         4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216]

class Bits:
    def __init__(self, data):
        self.data = data; self.bit = 0
    def get(self, n):
        value = 0
        for _ in range(n):
            byte = self.data[self.bit // 8]
            value = (value << 1) | ((byte >> (7 - self.bit % 8)) & 1)
            self.bit += 1
        return value
    def ints(self, nbits, sizes):
        # A number of `nbits` bits whose bytes come least significant first,
        # each of 8 bits but the last.
        digits = []
        while nbits > 8:
            digits.append(self.get(8)); nbits -= 8
        if nbits > 0:
            digits.append(self.get(nbits))
        number = sum(d << (8 * i) for i, d in enumerate(digits))
        z = number % sizes[2]; number //= sizes[2]
        y = number % sizes[1]; number //= sizes[1]
        return [number, y, z]

def bits_of(size):
    n = 0
    while (1 << n) <= size and n < 32:
        n += 1
    return n

def bits_of3(sizes):
    product = sizes[0] * sizes[1] * sizes[2]
    return max(1, product.bit_length()) if product > 1 else 0

def read_frames(path):
    data = open(path, 'rb').read(); pos = 0; frames = []
    i32 = lambda p: struct.unpack_from('>i', data, p)[0]
    f32 = lambda p: struct.unpack_from('>f', data, p)[0]
    while pos < len(data):
        assert i32(pos) == 1995
        natoms, step, time = i32(pos + 4), i32(pos + 8), f32(pos + 12)
        box = [f32(pos + 16 + 4 * k) for k in range(9)]
        assert i32(pos + 52) == natoms
        pos += 56
        if natoms <= 9:
            x = [f32(pos + 4 * k) for k in range(3 * natoms)]; pos += 12 * natoms
        else:
            precision = f32(pos)
            least = [i32(pos + 4 + 4 * k) for k in range(3)]
            most = [i32(pos + 16 + 4 * k) for k in range(3)]
            small = i32(pos + 28); nbytes = i32(pos + 32)
            bits = Bits(data[pos + 36:pos + 36 + nbytes])
            pos += 36 + (nbytes + 3) // 4 * 4
            sizes = [most[k] - least[k] + 1 for k in range(3)]
            large = any(s > 0xffffff for s in sizes)
            each = [bits_of(s) for s in sizes]
            total = bits_of3(sizes)
            smaller = MAGIC[max(9, small - 1)] // 2
            smallnum = MAGIC[small] // 2
            out = []; i = 0; run = 0
            while i < natoms:
                v = [bits.get(each[k]) for k in range(3)] if large else bits.ints(total, sizes)
                cur = [v[k] + least[k] for k in range(3)]
                i += 1
                change = 0
                if bits.get(1):
                    run = bits.get(5); change = run % 3; run -= change; change -= 1
                if run > 0:
                    prev = cur
                    for k in range(0, run, 3):
                        d = bits.ints(small, [MAGIC[small]] * 3); i += 1
                        nxt = [d[c] + prev[c] - smallnum for c in range(3)]
                        if k == 0:
                            out += [nxt, prev]
                        else:
                            out.append(nxt)
                        prev = nxt
                else:
                    out.append(cur)
                small += change
                if change < 0:
                    smallnum = smaller; smaller = MAGIC[small - 1] // 2 if small > 9 else 0
                elif change > 0:
                    smaller = smallnum; smallnum = MAGIC[small] // 2
            x = [c / precision for p in out for c in p]
        frames.append((step, time, box, x))
    return frames

def read_dcd(path):
    data = open(path, 'rb').read(); pos = 0; out = []
    def rec():
        nonlocal pos
        n = struct.unpack_from('<i', data, pos)[0]; body = data[pos + 4:pos + 4 + n]
        pos += n + 8; return body
    rec(); rec(); n = struct.unpack('<i', rec())[0]
    while pos < len(data):
        cell = struct.unpack('<6d', rec())
        xyz = [struct.unpack('<%df' % n, rec()) for _ in range(3)]
        out.append((cell, [xyz[k][i] for i in range(n) for k in range(3)]))
    return out

xtc = read_frames(sys.argv[1]); dcd = read_dcd(sys.argv[2])
period, dt = int(sys.argv[3]), float(sys.argv[4])
print('frames %d %d' % (len(xtc), len(dcd)))
worst = 0.0; steps = True; cells = True
for f, ((step, time, box, x), (cell, y)) in enumerate(zip(xtc, dcd)):
    steps &= step == (f + 1) * period and abs(time - step * dt) < 1e-6
    cells &= all(abs(box[k] - v / 10) < 1e-6 for k, v in [(0, cell[0]), (4, cell[2]), (8, cell[5])])
    worst = max(worst, max(abs(a - b / 10) for a, b in zip(x, y)))
print('steps and times: %s' % ('ok' if steps else 'FAILED'))
print('cells: %s' % ('ok' if cells else 'FAILED'))
print('positions within 0.0005 nm: %s' % ('ok' if worst <= 0.0005 + 1e-6 else 'FAILED %.6f' % worst))

"""Checks that the trajectories of a run without a periodic cell hold no
cell (D142): a DCD whose header says its frames hold none, and whose
frames are the three records of the positions alone, and an XTC whose
frames have a cell of zeros.

    check_no_cell_frames.py DCD XTC"""
import struct, sys

data = open(sys.argv[1], 'rb').read()
pos = 0
def rec():
    global pos
    n = struct.unpack_from('<i', data, pos)[0]; body = data[pos + 4:pos + 4 + n]
    pos += n + 8; return body
head = rec()
flag = struct.unpack_from('<i', head, 4 + 4 * 10)[0]
rec(); natoms = struct.unpack('<i', rec())[0]
frames = 0; sizes = True
while pos < len(data):
    for _ in range(3):
        sizes &= len(rec()) == 4 * natoms
    frames += 1
print('DCD: cell flag %d, %d frames of positions alone: %s' % (flag, frames, 'ok' if sizes else 'FAILED'))

data = open(sys.argv[2], 'rb').read()
pos = 0; zeros = True; frames = 0
while pos < len(data):
    natoms = struct.unpack_from('>i', data, pos + 4)[0]
    zeros &= all(v == 0.0 for v in struct.unpack_from('>9f', data, pos + 16))
    pos += 56
    if natoms <= 9:
        pos += 12 * natoms
    else:
        nbytes = struct.unpack_from('>i', data, pos + 32)[0]
        pos += 36 + (nbytes + 3) // 4 * 4
    frames += 1
print('XTC: %d frames with a cell of zeros: %s' % (frames, 'ok' if zeros else 'FAILED'))

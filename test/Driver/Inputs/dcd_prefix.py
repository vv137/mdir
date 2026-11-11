"""Checks that a DCD trajectory holds the first frames of another, byte
for byte: the number of frames of its header and of its records, and
whether those frames are the first of the reference.

    dcd_prefix.py DCD REFERENCE"""
import struct
import sys


def read(path):
    data = open(path, 'rb').read()
    pos = 0
    records = []
    while pos < len(data):
        n = struct.unpack_from('<i', data, pos)[0]
        records.append(data[pos:pos + n + 8])
        pos += n + 8
    # The header, the title, and the number of particles; then the frames,
    # each with the record of its cell first if the header has one.
    body = records[0][4:-4]
    frames = struct.unpack_from('<i', body, 4)[0]
    per = 4 if struct.unpack_from('<i', body, 4 + 4 * 10)[0] & 1 else 3
    return frames, records[3:], per


frames, records, per = read(sys.argv[1])
_, reference, _ = read(sys.argv[2])
prefix = len(records) % per == 0 and records == reference[:len(records)]
print(f'{len(records) // per} frames, {frames} in the header, '
      f'{"the first of the reference" if prefix else "NOT the first of the reference"}')

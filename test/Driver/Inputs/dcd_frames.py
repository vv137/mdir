# Compares the frames of a trajectory in DCD with those of its parts: the
# frames of the parts, one after the other, must be those of the whole, bit
# for bit, and each part must count its frames in its header.
#
#   python dcd_frames.py <whole.dcd> <part.dcd> [<part.dcd> ...]

import struct
import sys


def read(path):
    """Returns the header numbers, the number of particles, and the frames,
    each the bytes of its records."""
    with open(path, "rb") as file:
        data = file.read()
    size, = struct.unpack_from("<i", data, 0)
    numbers = struct.unpack_from("<20i", data, 8)
    offset = 4 + size + 4
    size, = struct.unpack_from("<i", data, offset)
    offset += 4 + size + 4
    count, = struct.unpack_from("<i", data, offset + 4)
    offset += 12
    frame = (8 + 48) + 3 * (8 + 4 * count)
    frames = []
    while offset + frame <= len(data):
        frames.append(data[offset:offset + frame])
        offset += frame
    if offset != len(data):
        sys.exit(f"{path}: {len(data) - offset} bytes after the last frame")
    return numbers, count, frames


numbers, count, whole = read(sys.argv[1])
parts = []
for path in sys.argv[2:]:
    header, particles, frames = read(path)
    if particles != count:
        sys.exit(f"{path}: {particles} particles, the whole has {count}")
    if header[0] != len(frames):
        sys.exit(f"{path}: the header counts {header[0]} frames, the file "
                 f"holds {len(frames)}")
    print(f"{path.split('/')[-1]}: {len(frames)} frames from step {header[1]}")
    parts += frames
if parts != whole:
    sys.exit(f"the parts hold {len(parts)} frames that differ from the "
             f"{len(whole)} of the whole")
print(f"the {len(whole)} frames are identical")

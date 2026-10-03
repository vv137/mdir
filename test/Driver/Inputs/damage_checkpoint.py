"""Damages a checkpoint in place, for the tests of D173.

    damage_checkpoint.py value FILE PRINTED  changes one byte of the first
                                        position of the first particle in
                                        PRINTED, the output of `mdir
                                        checkpoint --print=positions FILE`
    damage_checkpoint.py format FILE N  sets the attribute `format` to N
    damage_checkpoint.py rename FILE A B  renames the attribute A to B, of
                                        the same length

The file is changed byte by byte, without an HDF5 library: the attribute
messages of the object headers that MDIR writes are of version 1 (H5MD
files written with the default settings of HDF5), in which the name, the
type, and the space are padded to 8 bytes and the data follows.
"""

import struct
import sys


def pad8(size):
    return (size + 7) // 8 * 8


def main():
    mode, path = sys.argv[1], sys.argv[2]
    data = bytearray(open(path, "rb").read())
    if mode == "value":
        first = open(sys.argv[3]).readline().split()
        target = struct.pack("<d", float(first[2]))
        at = data.find(target)
        if at < 0:
            sys.exit("value not found")
        data[at + 3] ^= 0x10
    elif mode == "format":
        name = b"format\x00"
        at = data.find(name)
        if at < 8 or data[at - 8] != 1:
            sys.exit("no attribute message of version 1 named 'format'")
        name_size, type_size, space_size = struct.unpack_from(
            "<HHH", data, at - 6)
        value = at + pad8(name_size) + pad8(type_size) + pad8(space_size)
        struct.pack_into("<i", data, value, int(sys.argv[3]))
    elif mode == "rename":
        old, new = sys.argv[3].encode(), sys.argv[4].encode()
        if len(old) != len(new):
            sys.exit("the names must be of the same length")
        at = data.find(old + b"\x00")
        if at < 0:
            sys.exit("attribute not found")
        data[at:at + len(old)] = new
    else:
        sys.exit("unknown mode " + mode)
    open(path, "wb").write(data)


if __name__ == "__main__":
    main()

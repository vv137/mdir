"""Whether the tilts of the cell that `mdir checkpoint` prints are the
given fractions of its diagonal, b_x / a_x, c_x / a_x, c_y / b_y, to the
digits it prints: the shape of the cell is kept.

Usage: mdir checkpoint end.h5 | python3 check_cell.py bx/ax cx/ax cy/by
"""
import sys


def main():
    box = tilts = None
    for line in sys.stdin:
        words = line.split()
        if words[:1] == ["box:"]:
            box = [float(w) for w in words[1:4]]
        elif words[:1] == ["tilts:"]:
            tilts = [float(w) for w in words[1:4]]
    want = [float(w) for w in sys.argv[1:4]]
    got = [tilts[0] / box[0], tilts[1] / box[0], tilts[2] / box[1]]
    if all(abs(g - w) < 1e-5 for g, w in zip(got, want)):
        print("the shape of the cell is kept")
    else:
        print("the shape of the cell changed:", got)


main()

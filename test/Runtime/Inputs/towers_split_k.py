"""Check all source splits against the same periodic all-pairs force words."""
from pathlib import Path
import random
import sys

from towers_triclinic_filter import cases, check, reference, rebuild


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rebuild.MDIR_OPT = sys.argv[2]
    rng = random.Random(1043)
    lattice = cases[1][0]
    # Four occupied layers share a column. A middle tower reaches all four
    # layers, so even the eighth part takes a full chunk. Uneven populations
    # put both ends of the home-target interval inside source chunks.
    populations = (53, 59, 61, 62, 65)
    fractional = [
        (rng.randrange(32, 64) / 256,
         rng.randrange(64, 96) / 256,
         rng.randrange(32 * z, 32 * (z + 1)) / 256)
        for z, count in enumerate(populations, 1)
        for _ in range(count)
    ]
    expected = reference(lattice, fractional, rounded=True)
    for split in (1, 2, 4, 8):
        for packed in (0, 1):
            pairs = check(directory, lattice, fractional, reach=2, workers=64,
                          packed=packed, capacity=96, expected=expected,
                          split=split)
        print(f"split {split}: {pairs} pairs, {len(expected[0])} force words match")


if __name__ == "__main__":
    main()

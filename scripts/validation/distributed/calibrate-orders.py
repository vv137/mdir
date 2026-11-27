#!/usr/bin/env python3
"""Generate ordinary MATRIX order controls and summarize their energy histories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from trajectory import QUANTITIES, trajectory

QUANTILE = 0.9995
MARGIN = 1.25
SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
SOURCE = Path("lib/Conversion/MDExecKernels/MDExecKernels.cpp")
ENTRY = """        Value narrow = memref::LoadOp::create(pair, loc, index,
                                              ValueRange{central, entry});"""
ORDERED_ENTRY = """        if (!explicitImages && !twoEnded)
          entry = calibrationSlot(pair, loc, central, entry, end);
""" + ENTRY
MARKER = "SmallVector<Value> kernels::emitPairKernel("
PERMUTATION = """// This helper is used only in the calibration build.
// Each round pairs i with (pivot - i) mod count. The smaller slot determines
// whether both partners swap. A composition of these involutions is bijective.
static Value calibrationSlot(OpBuilder &b, Location loc, Value row,
                             Value slot, Value count) {
  const char *order = std::getenv("MDIR_CALIBRATION_ORDER");
  if (!order || std::string_view(order) == "identity")
    return slot;
  Value one = createIndex(b, loc, 1);
  if (std::string_view(order) == "reverse")
    return arith::SubIOp::create(
        b, loc, arith::SubIOp::create(b, loc, count, one), slot);

  uint64_t seed = std::strtoull(order, nullptr, 10);
  for (int round = 0; round < 4; ++round) {
    // Use SplitMix64 to derive independent round constants.
    seed += 0x9e3779b97f4a7c15ULL;
    uint64_t bits = seed;
    bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
    bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
    bits ^= bits >> 31;
    Value key = createIndex(b, loc, bits & 0x3fffffff);
    Value pivot = arith::RemUIOp::create(
        b, loc, arith::AddIOp::create(b, loc, row, key), count);
    Value partner = arith::RemUIOp::create(
        b, loc, arith::SubIOp::create(
            b, loc, arith::AddIOp::create(b, loc, pivot, count), slot),
        count);
    Value lo = arith::MinUIOp::create(b, loc, slot, partner);
    Value word = arith::XOrIOp::create(
        b, loc, arith::AddIOp::create(b, loc, row, key),
        arith::MulIOp::create(b, loc, lo, createIndex(b, loc, 0x45d9f3b)));
    word = arith::MulIOp::create(
        b, loc, arith::XOrIOp::create(
            b, loc, word, arith::ShRUIOp::create(
                b, loc, word, createIndex(b, loc, 16))),
        createIndex(b, loc, 0x45d9f3b));
    Value bit = arith::AndIOp::create(
        b, loc, arith::ShRUIOp::create(
            b, loc, word, createIndex(b, loc, 16)), one);
    Value swap = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bit, one);
    slot = arith::SelectOp::create(b, loc, swap, partner, slot);
  }
  return slot;
}

"""


def overlay(source):
    path = source / SOURCE
    original = path.read_text()
    if original.count(ENTRY) != 1 or original.count(MARKER) != 1:
        raise ValueError(f"{path}: expected one unmodified MATRIX slot load")
    path.write_text(original.replace(ENTRY, ORDERED_ENTRY).replace(
        MARKER, PERMUTATION + MARKER).replace(
        "#include \"llvm/ADT/StringExtras.h\"",
        "#include \"llvm/ADT/StringExtras.h\"\n\n#include <cstdlib>\n#include <string_view>"))


def summarize(directory, repetitions):
    controls = ("identity", "reverse", *map(str, SEEDS))
    paths = sorted(directory.glob("existing/*/energy.dat"))
    if len(paths) != repetitions:
        raise ValueError(f"{directory}: expected {repetitions} existing repeats")
    for order in controls:
        paths.extend(directory / order / str(rep) / "energy.dat" for rep in (1, 2))
    histories = np.stack([trajectory(path) for path in paths])
    first, second = np.triu_indices(len(histories), k=1)
    differences = np.abs(histories[first] - histories[second]).reshape(-1, 2)
    quantiles = np.quantile(differences, QUANTILE, axis=0, method="higher")
    return dict(
        existing_repetitions=repetitions, new_repetitions_per_order=2,
        orders=controls, histories=len(histories), frame_differences=len(differences),
        quantile=QUANTILE, margin=MARGIN,
        quantities={
            name: dict(quantile=float(q), maximum=float(m), bound=float(MARGIN * q))
            for name, q, m in zip(QUANTITIES, quantiles, differences.max(axis=0))
        },
        history_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    edit = sub.add_parser("overlay", help="instrument a disposable source copy")
    edit.add_argument("source", type=Path)
    sub.add_parser("orders", help="print the run labels")
    summary = sub.add_parser("summarize", help="compute the frozen bounds")
    summary.add_argument("histories", type=Path)
    summary.add_argument("--existing", type=int, default=40)
    args = parser.parse_args()
    if args.command == "overlay":
        overlay(args.source)
    elif args.command == "orders":
        print("identity reverse", *SEEDS)
    else:
        print(json.dumps(summarize(args.histories, args.existing), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

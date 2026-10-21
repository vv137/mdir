// Ragged rows: a relation from each of N rows to a variable number of
// entries of E values, in one buffer of i32 (D314,
// docs/device-primitives.md, Section 4.7).
//
// The layout, for N rows and M entries:
//
//   [0]                          N
//   [1, N + 2)                   the offsets o_0 = 0 <= o_1 <= ... <= o_N = M,
//                                in entries
//   [N + 2 + E j, N + 2 + E(j+1))  entry j
//   the last E elements          an entry of zeros past the last row
//
// Row i is the entries o_i to o_{i+1}. This is the compressed sparse row
// form (CSR) of the incidence matrix [Saad2003, Chapter 3]. The entry of
// zeros at the end keeps a read of the first entry of an empty last row
// inside the buffer (a kernel that reads the first entry of every row and
// keeps it only where the row has one, #347). A buffer of no elements has
// no rows: it stands for an absent relation.
//
// The incidence structure of a tuple set is the first user: an entry is a
// tuple of the particle, E = a + 2 (the tuple, the place of the particle
// in it, its a members). Code reads a row only through emitRaggedRow and
// emitRaggedEntry; the templates call them as the device primitives
// `ragged_row` and `ragged_entry` (CompactionEmitter.h), which expand to
// the same code.

#ifndef MDIR_CONVERSION_RAGGEDROWS_H
#define MDIR_CONVERSION_RAGGEDROWS_H

#include "mlir/IR/Builders.h"

namespace mdir {
namespace kernels {

/// The number of elements before the entries of a buffer of `rows` rows:
/// the number of rows and the rows + 1 offsets.
inline int64_t getRaggedHeaderSize(int64_t rows) { return rows + 2; }

/// The number of elements of a buffer of `rows` rows and `entries` entries
/// of `entrySize` values, the entry of zeros at the end included.
mlir::Value emitRaggedSize(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value rows, mlir::Value entries,
                           int64_t entrySize);

/// A row of a buffer of ragged rows: the element where its first entry
/// begins, and its number of entries (both index).
struct RaggedRow {
  mlir::Value first, count;
};

/// The row `row` (index) of `buffer`. With `checked`, a row past the
/// buffer's rows, or any row of a buffer of no elements, is a row of no
/// entries; without it, `row` must be a row of the buffer.
RaggedRow emitRaggedRow(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value buffer, mlir::Value row,
                        mlir::Value entrySize, bool checked);
RaggedRow emitRaggedRow(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value buffer, mlir::Value row,
                        int64_t entrySize, bool checked);

/// The value `column` of the entry `k` of the row that begins at the
/// element `first` (emitRaggedRow), an i32; `k` and `column` are index.
mlir::Value emitRaggedEntry(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value buffer, mlir::Value first,
                            mlir::Value k, mlir::Value column,
                            mlir::Value entrySize);
mlir::Value emitRaggedEntry(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value buffer, mlir::Value first,
                            mlir::Value k, int64_t column, int64_t entrySize);

/// The rows of a relation on a device from the row of each of its `m`
/// items, `keys[j]` (i32, in [0, n)): a thread for each item counts it in
/// `counts` (emitDeviceAtomicAdd), the exclusive sums of the counts go to
/// `start` (n + 1) and `cursor` (emitDeviceScan), a thread for each item
/// takes the next slot of its row (emitCursorScatter), and a thread for
/// each row sorts its items ascending (emitSortBins). On return `items`
/// holds the items j of row i, ascending, from start[i] to start[i + 1],
/// whatever order the atomics took. `counts`, `cursor` (n) and `sums`
/// (`blocks` = max(1, ceil(n / 1024))) are scratch on the device; `block`
/// is the threads of a launch and `one` the index 1. Six launches.
void emitDeviceRaggedRows(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value keys, mlir::Value m, mlir::Value n,
                          mlir::Value counts, mlir::Value start,
                          mlir::Value cursor, mlir::Value items,
                          mlir::Value sums, mlir::Value blocks,
                          mlir::Value block, mlir::Value one);

} // namespace kernels
} // namespace mdir

#endif // MDIR_CONVERSION_RAGGEDROWS_H

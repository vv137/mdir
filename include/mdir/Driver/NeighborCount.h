//===- NeighborCount.h - The most neighbors of a particle -------*- C++ -*-===//
//
// The largest number of particles within a reach of a particle, from which
// the driver sizes the rows of a neighbor structure when no
// `neighbor_capacity` is given (D227). The count is exact: the same set of
// candidates and the same test of each pair as the serial search that it
// replaces (D316), so the width of the lowered program is the
// same integer.
//
//===----------------------------------------------------------------------===//

#ifndef MDIR_DRIVER_NEIGHBORCOUNT_H
#define MDIR_DRIVER_NEIGHBORCOUNT_H

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace mdir {
namespace driver {

/// The largest number of other particles j with |x_i - x_j|^2 < reach^2 in
/// the minimum image of the orthorhombic box `box` (the diagonal a_x, b_y,
/// c_z alone for a triclinic cell), over every particle i of `positions`
/// (x, y, z of each). Candidates are the members of the distinct cells
/// among the 27 around the cell of i, with max(1, floor(box_k / reach))
/// cells along each axis. `threads` bounds the threads of the search; 0
/// takes the cores that the process may run on, bounded by
/// MDIR_COMPILE_THREADS where set. The result does not depend on it.
int64_t countMostNeighbors(llvm::ArrayRef<double> positions,
                           const double box[3], double reach,
                           unsigned threads = 0);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_NEIGHBORCOUNT_H

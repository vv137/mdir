// The periodic cell of a run: the reduced lower-triangular matrix of the
// GROMACS manual (docs/triclinic-m2.md, Section 1).

#ifndef MDIR_DRIVER_CELL_H
#define MDIR_DRIVER_CELL_H

#include "llvm/Support/Error.h"

#include <array>

namespace mdir {
namespace driver {

/// The cell whose rows are a = (a_x, 0, 0), b = (b_x, b_y, 0), and
/// c = (c_x, c_y, c_z), in the units of the numbers it was made from.
struct Cell {
  /// a_x, b_y, c_z.
  std::array<double, 3> diagonal = {0.0, 0.0, 0.0};
  /// b_x, c_x, c_y.
  std::array<double, 3> tilt = {0.0, 0.0, 0.0};

  bool isOrthorhombic() const {
    return tilt[0] == 0.0 && tilt[1] == 0.0 && tilt[2] == 0.0;
  }
  double getVolume() const { return diagonal[0] * diagonal[1] * diagonal[2]; }
  /// The rows a, b, c.
  std::array<std::array<double, 3>, 3> getVectors() const;
  /// The lengths a, b, c and the angles α (b, c), β (a, c), γ (a, b) in
  /// degrees.
  std::array<double, 6> getLengthsAndAngles() const;
};

/// The cell of the lengths `a`, `b`, `c` and the angles `alpha` (between b
/// and c), `beta` (a and c), and `gamma` (a and b), in degrees, with a along
/// x and b in the x-y plane, reduced.
llvm::Expected<Cell> makeCell(double a, double b, double c, double alpha,
                              double beta, double gamma);

/// Takes `cell` to the reduced form, |b_x| <= a_x/2, |c_x| <= a_x/2,
/// |c_y| <= b_y/2, by adding whole lattice vectors where a bound is
/// exceeded by more than a relative 1e-6: the lattice stays the same. A
/// cell on a bound, as the truncated octahedron of Amber is, is left as it
/// is.
llvm::Error reduceCell(Cell &cell);

/// The rotation R, row-major, that takes a position of CHARMM, whose cell is
/// the symmetric square root of its metric, to the frame of `cell`:
/// x = x_s R^T. The identity for a rectangular cell.
std::array<double, 9> getSymmetricFrameRotation(const Cell &cell);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_CELL_H

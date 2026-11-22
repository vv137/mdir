// Setup geometry for a TOWERS grid. The input cell uses lower-triangular rows.
#pragma once

#include "mdir/Driver/System.h"
#include <array>
#include <cmath>

namespace mdir::tool {
struct TowersCell {
  explicit TowersCell(const driver::System &system)
      : triclinic(system.tilt[0] || system.tilt[1] || system.tilt[2]),
        lattice{system.box[0],
                0,
                0,
                system.tilt[0],
                system.box[1],
                0,
                system.tilt[1],
                system.tilt[2],
                system.box[2]},
        heights{system.box[0], system.box[1], system.box[2]} {
    if (triclinic) {
      // Each face height is the reciprocal norm of a column of H^-1.
      double ax = lattice[0], bx = lattice[3], by = lattice[4];
      double cx = lattice[6], cy = lattice[7], cz = lattice[8];
      heights[0] = ax / std::hypot(1., bx / by, (bx * cy / by - cx) / cz);
      heights[1] = by / std::hypot(1., cy / cz);
    }
  }

  // Rectangular entries take Cartesian positions and the triclinic entries
  // take fractional positions. Both retain the input image until insertion.
  std::array<double, 3> coordinates(const double *position) const {
    if (!triclinic)
      return {position[0], position[1], position[2]};
    double z = position[2] / lattice[8];
    double y = (position[1] - z * lattice[7]) / lattice[4];
    double x = (position[0] - y * lattice[3] - z * lattice[6]) / lattice[0];
    return {x, y, z};
  }

  bool triclinic;
  std::array<double, 9> lattice;
  std::array<double, 3> heights;
};
} // namespace mdir::tool

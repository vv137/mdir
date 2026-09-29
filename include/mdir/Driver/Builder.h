// Builds the program of a run: a module of md and dyn ops.
//
// See docs/driver-m0.md, Section 2.

#ifndef MDIR_DRIVER_BUILDER_H
#define MDIR_DRIVER_BUILDER_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"

#include <string>
#include <vector>

namespace mdir {
namespace driver {

/// A floating-point type that a buffer of the host holds.
enum class Element { F32, F64 };

/// The program of a run, and what the host must pass to it.
///
/// The entry function takes, in this order: the buffers of the positions,
/// of the velocities, and of the masses; one buffer for each field in
/// `fields`; the three edge lengths of the cell, in nm; and the time step,
/// in ps.
struct Program {
  /// The module, as text.
  std::string module;
  /// The name of the entry function.
  std::string entry;

  /// The types that the buffers hold.
  Element state;
  Element mass;
  Element parameter;

  /// The parameters that differ between the types, one value per particle,
  /// in the units of the control file.
  struct Field {
    std::string name;
    std::vector<double> values;
  };
  std::vector<Field> fields;

  /// The skin of the neighbor structures, in nm, and the number of
  /// neighbors that they hold per particle.
  double skin;
  int64_t neighborWidth;
};

llvm::Expected<Program> buildProgram(const Control &control,
                                     const System &system);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_BUILDER_H

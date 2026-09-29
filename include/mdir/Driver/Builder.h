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
/// The entry function takes, in this order: the buffers of the positions
/// and of the velocities; the buffer of the forces, if `takesForces` is
/// set; the buffer of the masses; one buffer for each field in `fields`;
/// the three edge lengths of the cell, in nm; the time step, in ps; and the
/// number of the step that the run begins after.
struct Program {
  /// The module, as text.
  std::string module;
  /// The name of the entry function.
  std::string entry;

  /// The types that the buffers hold.
  Element state;
  Element force;
  Element mass;
  Element parameter;

  /// Whether the run begins with forces that it is given: a run that
  /// continues an earlier one, with an integrator that carries forces.
  bool takesForces = false;
  /// Whether the state that a checkpoint holds has forces.
  bool writesForces = false;

  /// The parameters that differ between the types, one value per particle,
  /// in the units of the control file.
  struct Field {
    std::string name;
    std::vector<double> values;
    /// Whether the field holds whole numbers, which the program takes as
    /// i32: the types of the particles.
    bool isInteger = false;
  };
  std::vector<Field> fields;

  /// Parameters of the pairs of types that a pair term looks up, in the
  /// units of the control file: `values[a * count + b]` for the types `a`
  /// and `b`.
  struct Table {
    std::string name;
    unsigned count = 0;
    std::vector<double> values;
  };
  std::vector<Table> tables;

  /// The correction for the dispersion beyond the cutoff, in kJ/mol, at
  /// the volume of the run: what it adds to the potential energy and to
  /// the trace of the virial.
  double dispersionEnergy = 0.0;
  double dispersionVirial = 0.0;

  /// The skin of the neighbor structures, in nm, and the number of
  /// neighbors that they hold per particle.
  double skin;
  int64_t neighborWidth;

  /// Whether the program puts the particles in the order of their
  /// positions, and the width of the cells that it orders them by, in nm.
  bool reorders = false;
  double orderWidth = 0.0;
};

llvm::Expected<Program> buildProgram(const Control &control,
                                     const System &system);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_BUILDER_H

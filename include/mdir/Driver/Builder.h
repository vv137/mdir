// Builds the program of a run: a module of md and dyn ops.
//
// See docs/driver-m0.md, Section 2.

#ifndef MDIR_DRIVER_BUILDER_H
#define MDIR_DRIVER_BUILDER_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"

#include <algorithm>
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
  /// A table of `count` rows: of pairs of types, symmetric and square,
  /// or of `columns` columns.
  struct Table {
    std::string name;
    unsigned count = 0;
    std::vector<double> values;
    unsigned columns = 0;

    bool isSquare() const { return columns == 0; }
    unsigned getColumns() const { return isSquare() ? count : columns; }
    llvm::StringRef getType() const {
      return isSquare() ? "!table" : "!grid";
    }
  };
  std::vector<Table> tables;

  /// The tuples of a topology: bonds, angles, and the like. `members` has
  /// `arity` particles for each tuple, and each field one value for each
  /// tuple, in the units of MDIR.
  struct TupleSet {
    std::string name;
    unsigned arity = 0;
    std::vector<int32_t> members;
    std::vector<Field> fields;
    /// Whether a tuple is the same read backward, as the members of a bond
    /// or of a dihedral are. Those of a virtual site or a correction map
    /// are not.
    bool reversible = true;
    /// Whether a tuple of two members has an order, as the pairs of a
    /// particle and the anchor of its constraint group have (D110).
    bool oriented = false;

    size_t size() const { return arity ? members.size() / arity : 0; }
    /// Whether no particle is a member of two tuples, as in the groups of
    /// a constraint.
    bool isDisjoint() const {
      std::vector<int32_t> sorted(members);
      std::sort(sorted.begin(), sorted.end());
      return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
    }
    llvm::StringRef getOrientation() const {
      if (arity == 2 && !oriented)
        return "unordered";
      return reversible ? "reversal" : "ordered";
    }
  };
  std::vector<TupleSet> tupleSets;

  /// The correction for the dispersion beyond the cutoff, in kJ/mol, at
  /// the volume of the run: what it adds to the potential energy and to
  /// the trace of the virial.
  double dispersionEnergy = 0.0;
  double dispersionVirial = 0.0;

  /// Particle mesh Ewald or the reaction field: whether the program has
  /// it, and the constant energy of its Coulomb terms in kJ/mol: for
  /// particle mesh Ewald the self term and the background of a net charge,
  /// with what the background adds to the trace of the virial; for the
  /// reaction field its self term (D140).
  bool pme = false;
  bool reactionField = false;
  double coulombConstantEnergy = 0.0;
  double coulombConstantVirial = 0.0;
  double coulombSelfEnergy = 0.0;
  /// [free_energy] (D161): for each state, the constant energies that
  /// depend on λ, in kJ/mol at the volume of the file: those that do not
  /// depend on the volume (the self term of particle mesh Ewald) and those
  /// proportional to 1 / V (the background of a net charge and the
  /// correction for the dispersion); and for each component of λ their
  /// derivatives at the state of the run.
  std::vector<double> stateFixedEnergies, stateVolumeEnergies;
  std::vector<double> lambdaFixedDerivatives, lambdaVolumeDerivatives;
  /// β in nm⁻¹ and the numbers of points of the grid, for the log.
  double pmeBeta = 0.0;
  int64_t pmeGrid[3] = {0, 0, 0};
  /// Particle mesh Ewald of the dispersion (D162): whether the program has
  /// it, its self term in kJ/mol, which does not depend on the volume and
  /// which the program does not compute, and β in nm⁻¹ and the grid.
  bool ljpme = false;
  double ljpmeSelfEnergy = 0.0;
  double ljpmeBeta = 0.0;
  int64_t ljpmeGrid[3] = {0, 0, 0};

  /// The skin of the neighbor structures, in nm, and the number of
  /// neighbors that they hold per particle.
  double skin;
  int64_t neighborWidth;
  /// The skin of the inner list of a dual list, in nm; 0 keeps one list.
  double pruneSkin = 0.0;

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

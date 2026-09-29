// A system as a topology describes it, in the units and the forms of
// docs/conventions.md, whatever the format that it was read from.

#ifndef MDIR_DRIVER_TOPOLOGY_H
#define MDIR_DRIVER_TOPOLOGY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <vector>

namespace mdir {
namespace driver {

/// A system of particles with bonded terms, in nm, kJ/mol, ps, amu, e, and
/// radians. Particles are numbered from 0 in the order of the file.
struct Topology {
  //===--------------------------------------------------------------------===//
  // Particles
  //===--------------------------------------------------------------------===//

  std::vector<std::string> atomNames;
  std::vector<int> atomicNumbers;
  std::vector<double> masses;
  std::vector<double> charges;
  /// The type of Lennard-Jones of each particle, from 0.
  std::vector<unsigned> types;
  std::vector<std::string> typeNames;

  /// The residues: their names, and the particle that each begins with.
  std::vector<std::string> residueNames;
  std::vector<unsigned> residueStarts;
  /// The residue of each particle. Unlike `residueStarts`, it follows the
  /// particles when they are put in another order.
  std::vector<unsigned> residueOf;

  //===--------------------------------------------------------------------===//
  // Terms
  //===--------------------------------------------------------------------===//

  /// Lennard-Jones for each pair of types, `4 ε ((σ/r)¹² − (σ/r)⁶)`:
  /// `sigma[a * count + b]` and `epsilon[a * count + b]`. A pair with no
  /// interaction has both 0.
  std::vector<double> sigma;
  std::vector<double> epsilon;

  /// `½ k (r − r0)²`. `hydrogen` tells whether a member is a hydrogen, which
  /// the constraints of the bonds of hydrogen take.
  struct Bond {
    unsigned i, j;
    double k, r0;
    bool hydrogen;
  };
  /// `½ k (θ − θ0)²`, with the angle at `j`.
  struct Angle {
    unsigned i, j, k;
    double force, theta0;
  };
  /// `k (1 + cos(n φ − φ0))`, proper or improper.
  struct Dihedral {
    unsigned i, j, k, l;
    double force;
    int n;
    double phase;
    bool improper;
  };
  /// A pair three bonds apart: Lennard-Jones and Coulomb scaled by the
  /// two factors, with the σ and ε of the pair.
  struct Pair {
    unsigned i, j;
    double scaleLJ, scaleCoulomb;
    double sigma, epsilon;
  };

  std::vector<Bond> bonds;
  std::vector<Angle> angles;
  std::vector<Dihedral> dihedrals;
  std::vector<Pair> pairs;
  /// The pairs that the nonbonded terms leave out, `i < j`, sorted: one,
  /// two, and three bonds apart.
  std::vector<std::pair<unsigned, unsigned>> exclusions;

  //===--------------------------------------------------------------------===//
  // Configuration
  //===--------------------------------------------------------------------===//

  /// The edges of the orthorhombic cell.
  double box[3] = {0.0, 0.0, 0.0};
  std::vector<double> positions;
  /// Empty if the file has no velocities.
  std::vector<double> velocities;

  size_t getNumParticles() const { return masses.size(); }
  size_t getNumTypes() const { return typeNames.size(); }
};

/// Reads a topology in the format of Amber (`prmtop`, `parm7`).
llvm::Expected<Topology> readAmberTopology(llvm::StringRef path);

/// Reads the positions, the velocities if there are any, and the cell of
/// `topology` from a file of coordinates of Amber (`inpcrd`, `rst7`).
llvm::Error readAmberCoordinates(llvm::StringRef path, Topology &topology);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_TOPOLOGY_H

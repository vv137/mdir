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
/// A term over tuples of particles, given by an expression in their
/// internal coordinate, as the custom forces of OpenMM over bonds, angles,
/// and torsions [Eastman2017] (D136): `r` in Å between the two particles of
/// a tuple, `theta` in radians, the angle at the middle one of three and the
/// dihedral of four, in (-pi, pi]. The energy is in kcal/mol, the units of
/// the control file.
struct TupleTerm {
  std::string name;
  std::string expression;
  unsigned arity = 2;
  /// The particles of the tuples, from 0, `arity` for each.
  std::vector<unsigned> particles;
  /// The parameters that the expression uses, one value for each tuple.
  std::vector<std::pair<std::string, std::vector<double>>> parameters;

  size_t size() const { return arity ? particles.size() / arity : 0; }
  /// The name of the coordinate in the expression.
  llvm::StringRef getVariable() const { return arity == 2 ? "r" : "theta"; }
};

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

  /// `½ k (r − r0)²` between the outer atoms `i` and `k` of an angle: the
  /// term of Urey and Bradley that CHARMM force fields add to some angles.
  struct UreyBradley {
    unsigned i, k;
    double force, r0;
  };
  /// `½ k (ξ − ξ0)²` of the dihedral ξ of `i, j, k, l`, with the difference
  /// taken in [−π, π): a harmonic improper.
  struct HarmonicImproper {
    unsigned i, j, k, l;
    double force, xi0;
  };

  std::vector<Bond> bonds;
  std::vector<Angle> angles;
  std::vector<UreyBradley> ureyBradleys;
  std::vector<Dihedral> dihedrals;
  std::vector<HarmonicImproper> harmonicImpropers;
  std::vector<Pair> pairs;
  /// Rigid waters of three sites that SETTLE constrains: the first atom,
  /// the oxygen, and the two after it, with the distances O–H and H–H.
  struct Settle {
    unsigned oxygen;
    double distanceOH, distanceHH;
  };
  std::vector<Settle> settles;
  /// The bonds of hydrogen that SHAKE and RATTLE keep at their lengths, in
  /// groups of a heavy atom and the hydrogens bonded to it, one to three.
  struct Shake {
    unsigned center;
    std::vector<unsigned> hydrogens;
    std::vector<double> lengths;
  };
  std::vector<Shake> shakes;
  /// A correction map of two dihedrals, φ of `i, j, k, l` and ψ of
  /// `j, k, l, m`, on the map `map`.
  struct CMap {
    unsigned i, j, k, l, m;
    unsigned map;
  };
  std::vector<CMap> cmaps;
  /// Terms given by expressions over tuples (D136): those of the control
  /// file, and those that a reader gives so.
  std::vector<TupleTerm> tupleTerms;
  /// The maps: `resolution²` energies each, in kJ/mol, at φ and ψ from
  /// −180° in steps of 360° / `resolution`, φ the slower index.
  unsigned cmapResolution = 0;
  std::vector<std::vector<double>> cmapGrids;

  /// A particle of mass 0 whose position follows from those of three
  /// others, `i`, `j`, and `k`, and whose force goes to them.
  struct VirtualSite {
    enum Kind {
      /// The extra point of four-site water in Amber:
      /// `x_i + a (û + v̂) / |û + v̂|`, with `û` and `v̂` the unit vectors
      /// from `i` to `j` and to `k`. It depends only on their directions.
      AmberWater,
      /// `(1 − a − b) x_i + a x_j + b x_k` (`virtual_sites3`, function 1,
      /// of GROMACS).
      Linear,
    };
    Kind kind;
    unsigned site, i, j, k;
    double a, b;
  };
  std::vector<VirtualSite> virtualSites;
  /// The pairs that the nonbonded terms leave out, `i < j`, sorted: one,
  /// two, and three bonds apart.
  std::vector<std::pair<unsigned, unsigned>> exclusions;

  //===--------------------------------------------------------------------===//
  // Configuration
  //===--------------------------------------------------------------------===//

  /// The cell (docs/triclinic-m2.md): its diagonal a_x, b_y, c_z, which are
  /// the edges of an orthorhombic cell, and its tilts b_x, c_x, c_y, which
  /// are 0 for one.
  double box[3] = {0.0, 0.0, 0.0};
  double tilt[3] = {0.0, 0.0, 0.0};
  std::vector<double> positions;
  /// Empty if the file has no velocities.
  std::vector<double> velocities;

  size_t getNumParticles() const { return masses.size(); }
  size_t getNumTypes() const { return typeNames.size(); }
};

/// The coefficients of the bicubic patches of the maps of `topology`: for
/// each map, each cell (a, b) with φ index a and ψ index b, 16 numbers
/// `c[4 i + j]` of `E(t, u) = Σ c_ij t^i u^j`, with t and u the places of φ
/// and ψ in the cell, from 0 to 1. The derivatives at the points of a grid
/// are those of natural cubic splines through the grid repeated to twice
/// its length, as sander and GROMACS compute them.
std::vector<double> getCMapCoefficients(const Topology &topology);

/// Reads a topology in the format of Amber (`prmtop`, `parm7`).
llvm::Expected<Topology> readAmberTopology(llvm::StringRef path);

/// Reads the positions, the velocities if there are any, and the cell of
/// `topology` from a file of coordinates of Amber (`inpcrd`, `rst7`).
llvm::Error readAmberCoordinates(llvm::StringRef path, Topology &topology);

/// Reads a topology in the format of GROMACS (`.top` with its `.itp`
/// files). Included files are looked for next to the file that includes
/// them, then in `includePath`, then in the directories of `GMXLIB`.
/// `defines` are defined before the first line, as `-D` of grompp.
llvm::Expected<Topology>
readGromacsTopology(llvm::StringRef path,
                    llvm::ArrayRef<std::string> includePath,
                    llvm::ArrayRef<std::string> defines = {});

/// Reads the positions, the velocities if there are any, and the cell of
/// `topology` from a `.gro` file.
llvm::Error readGromacsCoordinates(llvm::StringRef path, Topology &topology);

/// Reads a protein structure file of CHARMM (PSF, with the types named, as
/// `write psf card xplor` and CHARMM-GUI write it) and assigns it the
/// parameters of `parameterFiles`: files of topology (RTF), of parameters
/// (PRM), and stream files, read in order (docs/charmm-m1.md).
llvm::Expected<Topology>
readCharmmTopology(llvm::StringRef path,
                   llvm::ArrayRef<std::string> parameterFiles);

/// Reads the positions of `topology` from a coordinate file of CHARMM
/// (CRD, standard or extended); it has no cell.
llvm::Error readCharmmCoordinates(llvm::StringRef path, Topology &topology);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_TOPOLOGY_H

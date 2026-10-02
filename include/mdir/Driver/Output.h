// What a run writes: the log, the trajectory, and checkpoints.

#ifndef MDIR_DRIVER_OUTPUT_H
#define MDIR_DRIVER_OUTPUT_H

#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace mdir {
namespace driver {

/// Writes trajectories in the DCD format: positions in Å, as 32-bit
/// floating-point numbers, with the cell.
class DCDWriter {
public:
  ~DCDWriter();

  /// `first` is the step of the first frame, `period` the number of steps
  /// between two frames, and `timestep` the time step in ps.
  llvm::Error open(const std::string &path, size_t numParticles,
                   int64_t first, int64_t period, double timestep,
                   const double box[3]);

  /// Writes a frame. `positions` holds three numbers per particle, in Å.
  void writeFrame(const float *positions);
  /// The edges of the cell of the frames that follow, in Å: of an
  /// orthorhombic cell, or the diagonal of a triclinic one, whose tilts
  /// scale with the columns of the cell (docs/triclinic-m2.md, I3).
  void setBox(const double edges[3]) {
    if (box[0] > 0.0) {
      tilt[0] *= edges[0] / box[0];
      tilt[1] *= edges[0] / box[0];
      tilt[2] *= edges[1] / box[1];
    }
    for (int i = 0; i != 3; ++i)
      box[i] = edges[i];
  }
  /// The tilts b_x, c_x, c_y of a triclinic cell, in Å.
  void setTilt(const double tilts[3]) {
    for (int i = 0; i != 3; ++i)
      tilt[i] = tilts[i];
  }

  void close();

private:
  void writeHeader();

  std::FILE *file = nullptr;
  size_t numParticles = 0;
  int64_t first = 0;
  int64_t period = 0;
  double timestep = 0.0;
  double box[3] = {0.0, 0.0, 0.0};
  double tilt[3] = {0.0, 0.0, 0.0};
  int32_t numFrames = 0;
};

/// The output of the run that is under way. The functions that compiled
/// code calls write to it.
struct Output {
  std::FILE *log = stdout;
  DCDWriter trajectory;
  bool hasTrajectory = false;

  /// The types that the buffers of the state and of the forces hold.
  Element state = Element::F64;
  Element force = Element::F64;

  /// The step and the time that the run begins with.
  int64_t firstStep = 0;
  double firstTime = 0.0;
  /// The steps between the rows of the log; energies computed between
  /// them, at more frequent frames, are not shown.
  int64_t energyPeriod = 0;
  double timestep = 0.0;
  double degreesOfFreedom = 0.0;
  /// The volume of the cell, in nm^3, and the edges, which a barostat
  /// changes (mdrtSetBox).
  double volume = 0.0;
  double box[3] = {0.0, 0.0, 0.0};
  /// The volume that the constants below are for; they are proportional to
  /// 1 / V but the self term.
  double firstVolume = 0.0;
  /// The correction for the dispersion beyond the cutoff, in kJ/mol: what
  /// it adds to the potential energy and to the trace of the virial.
  double dispersionEnergy = 0.0;
  double dispersionVirial = 0.0;
  /// Particle mesh Ewald: the self term and the background of a net
  /// charge, which the program does not compute.
  bool pme = false;
  double pmeConstantEnergy = 0.0;
  double pmeConstantVirial = 0.0;
  /// Of which the self term, which does not depend on the volume.
  double pmeSelfEnergy = 0.0;

  /// The constants at the volume `volume`.
  double getDispersionEnergy() const {
    return dispersionEnergy * firstVolume / volume;
  }
  double getDispersionVirial() const {
    return dispersionVirial * firstVolume / volume;
  }
  double getPMEConstantEnergy() const {
    return pmeSelfEnergy +
           (pmeConstantEnergy - pmeSelfEnergy) * firstVolume / volume;
  }
  double getPMEConstantVirial() const {
    return pmeConstantVirial * firstVolume / volume;
  }

  /// Whether the velocities are coupled, and the energy that the coupling
  /// has taken from the system so far, in kJ/mol. The total energy with it
  /// is conserved.
  bool couples = false;
  double bath = 0.0;
  /// Whether a barostat changes the cell, which the log then shows.
  bool changesCell = false;
  /// Whether the run minimizes the energy, whose log has the forces in
  /// place of the kinetic energy (mdrtWriteMinimization).
  bool minimizes = false;
  /// The shortest edge of the cell that the cutoff allows, twice it; a run
  /// whose barostat takes the cell below it stops.
  double leastEdge = 0.0;

  /// The energies at the first and at the last output, in kJ/mol. With
  /// coupling, those that are conserved.
  bool hasEnergies = false;
  double firstTotal = 0.0;
  double lastTotal = 0.0;

  /// The steps of the outputs of the energies and when each was written,
  /// in seconds of a steady clock: the rate of the run past its start
  /// (the set-up and the first build), as GROMACS reports it with
  /// `-resethway`.
  std::vector<std::pair<int64_t, double>> energyTimes;

  /// Where checkpoints go, and what they hold beside the state.
  std::string checkpointPath;
  Checkpoint checkpoint;
  int64_t numCheckpoints = 0;

  /// Takes the state when the run ends.
  System *system = nullptr;

  double getTime(int64_t step) const {
    return firstTime + static_cast<double>(step - firstStep) * timestep;
  }
};

/// Sets the output that the functions below write to.
void setOutput(Output *output);

void writeLogHeader(Output &output);

} // namespace driver
} // namespace mdir

/// The functions that compiled code calls. Buffers arrive as pointers to
/// descriptors, as the C interface of MLIR passes them.
///
/// The particles of a buffer are in the order that the run keeps them in.
/// `ids` holds the number of each: the place of the particle in the files
/// of the run. What is written is in the order of these numbers.
extern "C" {
void _mlir_ciface_mdrtWriteEnergies(int64_t step, double potential,
                                    double kinetic, double forceSquare,
                                    double virial);
void _mlir_ciface_mdrtWriteFrame(int64_t step, void *positions, void *ids);
/// The energy of each term of a topology at the start, in kJ/mol:
/// Lennard-Jones, Coulomb, bonds, angles, dihedrals, the pairs three bonds
/// apart, Lennard-Jones and Coulomb, and CMAP.
void _mlir_ciface_mdrtWriteTerms(void *terms);
/// The diagonal of the virial of the forces at the start, without those of
/// the constraints, in kJ/mol; the constant terms are added here.
void _mlir_ciface_mdrtWriteVirial(double xx, double yy, double zz);
/// The energy that a coupling of the velocities has just taken from the
/// system, in kJ/mol.
void _mlir_ciface_mdrtAddBath(double energy);
/// A step of a minimization: the potential energy and the length of the
/// next step, in kJ/mol and nm, and the forces, of which the log has the
/// root mean square over the particles with mass and the largest.
void _mlir_ciface_mdrtWriteMinimization(int64_t step, double energy,
                                        double size, void *forces,
                                        void *ids);
/// The cell after a barostat has changed it: its edges in nm.
void _mlir_ciface_mdrtSetBox(double lx, double ly, double lz);
/// The tilts b_x, c_x, c_y of a triclinic cell that a barostat has scaled,
/// after mdrtSetBox (docs/triclinic-m2.md).
void _mlir_ciface_mdrtSetTilt(double bx, double cx, double cy);
/// The state that the next scaling of a barostat that scales the cell every
/// step takes its pressure from (D92), for the checkpoints.
/// The state of the last scaling of a barostat that scales every step
/// (D92, D119): the diagonals of the virial and of the virial of the rigid
/// groups, and the kinetic energy of each axis without the center of mass.
void _mlir_ciface_mdrtSetBarostatState(double w0, double w1, double w2,
                                       double g0, double g1, double g2,
                                       double k0, double k1, double k2);
void _mlir_ciface_mdrtWriteCheckpoint(int64_t step, void *positions,
                                      void *velocities, void *ids);
void _mlir_ciface_mdrtWriteCheckpointWithForces(int64_t step,
                                                void *positions,
                                                void *velocities,
                                                void *forces, void *ids);
void _mlir_ciface_mdrtFinish(void *positions, void *velocities, void *ids);
}

#endif // MDIR_DRIVER_OUTPUT_H

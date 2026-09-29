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

  void close();

private:
  void writeHeader();

  std::FILE *file = nullptr;
  size_t numParticles = 0;
  int64_t first = 0;
  int64_t period = 0;
  double timestep = 0.0;
  double box[3] = {0.0, 0.0, 0.0};
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
  double timestep = 0.0;
  double degreesOfFreedom = 0.0;

  /// The energies at the first and at the last output, in kJ/mol.
  bool hasEnergies = false;
  double firstTotal = 0.0;
  double lastTotal = 0.0;

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
                                    double kinetic);
void _mlir_ciface_mdrtWriteFrame(int64_t step, void *positions, void *ids);
void _mlir_ciface_mdrtWriteCheckpoint(int64_t step, void *positions,
                                      void *velocities, void *ids);
void _mlir_ciface_mdrtWriteCheckpointWithForces(int64_t step,
                                                void *positions,
                                                void *velocities,
                                                void *forces, void *ids);
void _mlir_ciface_mdrtFinish(void *positions, void *velocities, void *ids);
}

#endif // MDIR_DRIVER_OUTPUT_H

// What a run writes: the log and the trajectory.

#ifndef MDIR_DRIVER_OUTPUT_H
#define MDIR_DRIVER_OUTPUT_H

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

  /// `period` is the number of steps between two frames, `timestep` the
  /// time step in ps.
  llvm::Error open(const std::string &path, size_t numParticles,
                   int64_t period, double timestep, const double box[3]);

  /// Writes a frame. `positions` holds three numbers per particle, in Å.
  void writeFrame(const float *positions);

  void close();

private:
  void writeHeader();

  std::FILE *file = nullptr;
  size_t numParticles = 0;
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

  double timestep = 0.0;
  double degreesOfFreedom = 0.0;

  /// The energies at the first and at the last output, in kJ/mol.
  bool hasEnergies = false;
  double firstTotal = 0.0;
  double lastTotal = 0.0;

  /// Takes the state when the run ends.
  System *system = nullptr;
};

/// Sets the output that the functions below write to.
void setOutput(Output *output);

void writeLogHeader(Output &output);

} // namespace driver
} // namespace mdir

/// The functions that compiled code calls. Buffers arrive as pointers to
/// descriptors, as the C interface of MLIR passes them.
extern "C" {
void _mlir_ciface_mdrtWriteEnergies(int64_t step, double potential,
                                    double kinetic);
void _mlir_ciface_mdrtWriteFrame_f32(int64_t step, void *positions);
void _mlir_ciface_mdrtWriteFrame_f64(int64_t step, void *positions);
void _mlir_ciface_mdrtFinish_f32(void *positions, void *velocities);
void _mlir_ciface_mdrtFinish_f64(void *positions, void *velocities);
}

#endif // MDIR_DRIVER_OUTPUT_H

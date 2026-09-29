// Checkpoints in the H5MD format.
//
// See docs/driver-m0.md, Section 2.6.

#ifndef MDIR_DRIVER_CHECKPOINT_H
#define MDIR_DRIVER_CHECKPOINT_H

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mdir {
namespace driver {

/// The state of a run, in the units inside MDIR: nm, ps, amu, kJ/mol. All
/// numbers are 64-bit, so that a value of any precision mode is stored
/// without loss.
struct Checkpoint {
  size_t getNumParticles() const { return masses.size(); }

  int64_t step = 0;
  double time = 0.0;

  /// Three numbers per particle.
  std::vector<double> positions;
  std::vector<double> velocities;
  /// The forces that the next step begins with, or nothing if the
  /// integrator computes them first.
  std::vector<double> forces;

  std::vector<double> masses;
  /// For every particle, the position of its type in the control file.
  std::vector<int32_t> species;
  double box[3] = {0.0, 0.0, 0.0};

  /// What is needed to continue the run as it was.
  std::string integrator;
  /// The time of the velocities relative to that of the positions, in
  /// time steps.
  double velocityOffset = 0.0;
  std::string precision;
  double timestep = 0.0;
  uint64_t seed = 0;
};

/// Returns true if the driver was built with the library that checkpoints
/// need.
bool hasCheckpointSupport();

/// Writes `checkpoint` to `path`. The file appears under its name only
/// when it is complete.
llvm::Error writeCheckpoint(const std::string &path,
                            const Checkpoint &checkpoint);

llvm::Expected<Checkpoint> readCheckpoint(const std::string &path);

/// Compares the states of two checkpoints. Returns an empty string if they
/// are identical, bit for bit, and a description of the first difference
/// otherwise.
std::string compareCheckpoints(const Checkpoint &first,
                               const Checkpoint &second);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_CHECKPOINT_H

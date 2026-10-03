// Checkpoints in the H5MD format: de Buyl et al., Comput. Phys. Commun. 185,
// 1546 (2014).
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
  /// The tilts b_x, c_x, c_y of a triclinic cell (docs/triclinic-m2.md).
  double tilt[3] = {0.0, 0.0, 0.0};
  /// Whether the cell is periodic; without one, `box` is the cell that the
  /// run placed around the particles (D142), which a continued run keeps.
  bool periodic = true;

  /// What is needed to continue the run as it was.
  std::string integrator;
  /// The time of the velocities relative to that of the positions, in
  /// time steps.
  double velocityOffset = 0.0;
  std::string precision;
  double timestep = 0.0;
  uint64_t seed = 0;
  /// The run that wrote the checkpoint (D129): the step it began at, which
  /// `mdir run --continue` counts its `steps` from; the part of the run,
  /// 1 for the first and one more for each continuation; and the file
  /// name of the trajectory that its frames went to, with the number of
  /// frames that file holds at this state (D130). Without a trajectory the
  /// name is empty. `outputsPart` is the part whose files the outputs of
  /// the run go to, `<name>.partNNNN<ext>`, or 0 for the names of the
  /// control file (D149).
  int64_t firstStep = 0;
  int64_t part = 1;
  int64_t outputsPart = 0;
  std::string trajectory;
  int64_t frames = 0;
  /// The energy that the coupling of the velocities has taken from the
  /// system since the run began, in kJ/mol, which the conserved energy of
  /// a continued run counts on from.
  double bath = 0.0;
  /// Whether the file holds the above; one written before D129 does not.
  bool hasRun = false;

  /// With a barostat that scales the cell every step (D92): the trace of
  /// the virial, that of the rigid groups, and the kinetic energy without
  /// the center of mass of the state, which the next scaling takes its
  /// pressure from, in kJ/mol; empty otherwise.
  std::vector<double> barostatState;
};

/// Returns true if the driver was built with the library that checkpoints
/// need.
bool hasCheckpointSupport();

/// Writes `checkpoint` to `path`. The file appears under its name only
/// when it is complete, and the one it replaces stays as `path` with
/// `.prev` appended (D132).
llvm::Error writeCheckpoint(const std::string &path,
                            const Checkpoint &checkpoint);

/// The name under which `writeCheckpoint` keeps the checkpoint before the
/// last.
std::string getPreviousCheckpointPath(const std::string &path);

llvm::Expected<Checkpoint> readCheckpoint(const std::string &path);

/// Compares the states of two checkpoints. Returns an empty string if they
/// are identical, bit for bit, and a description of the first difference
/// otherwise.
std::string compareCheckpoints(const Checkpoint &first,
                               const Checkpoint &second);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_CHECKPOINT_H

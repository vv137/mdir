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

/// An entry of the fingerprint of a run (D172): its
/// group, "physics", "coupling", or "execution"; its name, such as
/// "[energy] cutoff"; and its value, the canonical text of what the control
/// file writes, or the SHA-256 of what is too long to show.
struct FingerprintEntry {
  std::string group;
  std::string name;
  std::string value;
};
using Fingerprint = std::vector<FingerprintEntry>;

/// The format of the checkpoints that this MDIR writes and reads, the
/// contract of release 0.1.0 (D173). A later format comes
/// with a conversion from the one before it.
constexpr int checkpointFormat = 1;

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
  /// What defined the run (D172).
  Fingerprint fingerprint;
  /// The program that wrote the file: its version and commit.
  std::string creator = "MDIR";
  std::string creatorVersion;

  /// With a barostat that scales the cell every step (D92): the trace of
  /// the virial, that of the rigid groups, and the kinetic energy without
  /// the center of mass of the state, which the next scaling takes its
  /// pressure from, in kJ/mol; empty otherwise.
  std::vector<double> barostatState;
  /// With a Nose-Hoover chain (D163a): the positions of its thermostats,
  /// then their velocities, in the units of the run; empty otherwise.
  std::vector<double> thermostatState;
  /// [free_energy] (D161): the selection, the soft-core, and the values of
  /// every component at every state, as Control::FreeEnergy::describe
  /// gives them; the state of the run, and the components of λ there.
  /// Empty, -1, and empty without it.
  std::string freeEnergy;
  int64_t freeEnergyState = -1;
  std::vector<double> freeEnergyLambda;

  /// Additional entries of format 1 (D[python-checkpoints],
  /// docs/python-checkpoints.md), which a reader of release 0.1.0 ignores.
  /// They are outside the hash of the state and have a hash of their own,
  /// `extras_sha256`, which the reader checks when the file has it.
  /// The front end that wrote the file: "python", or empty for `mdir run`.
  std::string frontEnd;
  /// SHA-256 of the model (the physics and coupling of the fingerprint and
  /// the declarations of the tunables) and of the plan of the program.
  std::string modelHash, planHash;
  /// The tunable parameters (D213): their declarations as text, the
  /// values of each by name in MD units, the version of the values, and for
  /// each version the step after which it holds.
  std::string tunableDeclarations;
  std::vector<std::pair<std::string, std::vector<double>>> tunables;
  int64_t tunablesVersion = 0;
  std::vector<std::pair<int64_t, int64_t>> tunablesHistory;
  bool hasExtras() const {
    return !frontEnd.empty() || !modelHash.empty() || !planHash.empty() ||
           !tunableDeclarations.empty() || !tunables.empty();
  }
};

/// Returns true if the driver was built with the library that checkpoints
/// need.
bool hasCheckpointSupport();

/// Writes `checkpoint` to `path`. The file appears under its name only
/// when it is complete and on stable storage, and the one it replaces stays
/// as `path` with `.prev` appended (D132, D173).
llvm::Error writeCheckpoint(const std::string &path,
                            const Checkpoint &checkpoint);

/// The name under which `writeCheckpoint` keeps the checkpoint before the
/// last.
std::string getPreviousCheckpointPath(const std::string &path);

/// Reads a checkpoint of format `checkpointFormat`, and checks the hash of
/// its state. A newer format, a file of a development build before the
/// release, and a state whose hash differs are errors.
llvm::Expected<Checkpoint> readCheckpoint(const std::string &path);

/// Compares the states of two checkpoints. Returns an empty string if they
/// are identical, bit for bit, and a description of the first difference
/// otherwise.
std::string compareCheckpoints(const Checkpoint &first,
                               const Checkpoint &second);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_CHECKPOINT_H

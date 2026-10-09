// Trajectories in the H5MD format, version 1.1: de Buyl et al., Comput.
// Phys. Commun. 185, 1546 (2014). The frames of the state as the run has
// them: positions in f64 or f32 in nm, the cell, the step and the time,
// and optionally the velocities and the forces (D[h5md-reporter],
// docs/python-h5md.md).

#ifndef MDIR_DRIVER_H5MD_H
#define MDIR_DRIVER_H5MD_H

#include "mdir/Driver/Trajectory.h"

#include <limits>
#include <map>
#include <optional>

namespace mdir {
namespace driver {

/// What a trajectory in H5MD holds, fixed when the file is made; a
/// continued run must ask for the same.
struct H5MDOptions {
  /// The positions, velocities, and forces in f32 rather than f64.
  bool single = false;
  bool velocities = false, forces = false;
  /// `/observables/potential_energy`: every frame is a step of energy.
  bool energy = false;
  /// `/observables/tunables_version` (D213).
  bool tunables = false;
  /// The cell vectors as a matrix rather than the edges.
  bool triclinic = false;
  /// The time of the velocities after that of the positions, in time
  /// steps: -0.5 with leapfrog.
  double velocityOffset = 0.0;
  std::vector<double> masses;
  std::vector<int32_t> species;
  /// The program that writes the file, the front end ("python", or empty
  /// for `mdir run`), and the precision mode of the run.
  std::string creatorVersion, frontEnd, precision;
};

/// What a frame holds besides the state.
struct H5MDExtras {
  /// The potential energy of the step in kJ/mol; not a number if the step
  /// was not a step of energy.
  double potential = std::numeric_limits<double>::quiet_NaN();
  int64_t tunablesVersion = 0;
};

/// Writes the frames of a trajectory in H5MD. Every frame is flushed to the
/// operating system, so a run that is killed leaves a file that opens and
/// holds the frames written.
class H5MDWriter : public TrajectoryWriter {
public:
  explicit H5MDWriter(H5MDOptions options);
  ~H5MDWriter() override;

  /// The cell is that of `setExactEdges` and `setExactTilt`; `box` is not
  /// read.
  llvm::Error open(const std::string &path, size_t numParticles,
                   int64_t first, int64_t period, double timestep,
                   const double box[3]) override;
  /// Continues the file after its first `frames` frames (D130): it must be
  /// a trajectory of this writer with the same particles, type, elements,
  /// shape of the cell, and period, and hold at least `frames` frames.
  llvm::Expected<int64_t> append(const std::string &path,
                                 size_t numParticles, int64_t frames,
                                 int64_t period, double timestep,
                                 const double box[3]) override;
  /// Not the interface of this writer: frames arrive through `writeState`.
  void writeFrame(const float *positions, int64_t step,
                  double time) override;
  void close() override;
  bool isExact() const override { return true; }

  /// Writes a frame: three numbers per particle in the order of the input,
  /// in nm, nm/ps, and kJ/mol/nm. `velocities` and `forces` are read only
  /// if the file holds them. Returns false on a failure, which
  /// `getFailure` describes; nothing is written after one.
  bool writeState(const double *positions, const double *velocities,
                  const double *forces, int64_t step, double time,
                  const H5MDExtras &extras);
  const std::string &getFailure() const { return failure; }
  const H5MDOptions &getOptions() const { return options; }
  /// The step of the last frame of the file, or none without frames.
  std::optional<int64_t> getLastStep() const { return lastStep; }

private:
  struct File;
  H5MDOptions options;
  std::unique_ptr<File> h5;
  std::string failure;
  std::optional<int64_t> lastStep;
};

/// A frame read from a trajectory in H5MD, in the units of the file.
struct H5MDFrame {
  int64_t step = 0;
  /// Not a number if the file has no times.
  double time = 0.0;
  /// The width of an element of the positions, velocities, and forces as
  /// the file stores them: 4 or 8. The vectors hold that type, three per
  /// particle; those of the velocities and the forces are empty if the
  /// file has none.
  size_t width = 8;
  std::vector<char> positions, velocities, forces;
  /// Whether the frame has a cell, and the cell vectors in rows.
  bool hasCell = false;
  double cell[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  /// Whether the file stores the edges alone.
  bool orthorhombic = true;
  std::optional<double> potential;
  std::optional<int64_t> tunablesVersion;
};

/// Reads the frames of an H5MD file one at a time: the files of H5MDWriter,
/// and those of other writers with `particles/<group>/position` in explicit
/// or fixed step storage.
class H5MDReader {
public:
  /// `group` names the group of `/particles`; if empty, "all" or the only
  /// group of the file.
  static llvm::Expected<std::unique_ptr<H5MDReader>>
  open(const std::string &path, const std::string &group = "");
  ~H5MDReader();

  size_t getNumFrames() const { return steps.size(); }
  size_t getNumParticles() const { return numParticles; }
  size_t getWidth() const { return width; }
  const std::vector<int64_t> &getSteps() const { return steps; }
  /// Empty if the file has no times.
  const std::vector<double> &getTimes() const { return times; }
  bool hasVelocities() const { return velocities; }
  bool hasForces() const { return forces; }
  bool hasPotential() const { return potential; }
  bool hasTunablesVersion() const { return tunables; }
  /// The `unit` attributes by element: "position", "velocity", "force",
  /// "time", "box", "potential_energy".
  const std::map<std::string, std::string> &getUnits() const { return units; }
  const std::string &getCreator() const { return creator; }
  const std::string &getGroup() const { return group; }
  llvm::Expected<H5MDFrame> read(size_t frame);
  void close();
  bool isOpen() const { return h5 != nullptr; }

private:
  H5MDReader();
  struct File;
  std::unique_ptr<File> h5;
  std::string path, group, creator;
  size_t numParticles = 0, width = 8;
  std::vector<int64_t> steps;
  std::vector<double> times;
  bool velocities = false, forces = false, potential = false,
       tunables = false;
  std::map<std::string, std::string> units;
};

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_H5MD_H

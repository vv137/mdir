// Trajectories: the positions of the particles at intervals, in DCD or in
// XTC (D141), or the frames of the state without loss in H5MD
// (D[h5md-reporter], mdir/Driver/H5MD.h).

#ifndef MDIR_DRIVER_TRAJECTORY_H
#define MDIR_DRIVER_TRAJECTORY_H

#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mdir {
namespace driver {

enum class TrajectoryFormat { DCD, XTC, H5MD };

/// The name of a format, as `trajectory_format` writes it.
const char *getTrajectoryFormatName(TrajectoryFormat format);

/// Writes the frames of a trajectory: positions in Å, with the cell.
class TrajectoryWriter {
public:
  virtual ~TrajectoryWriter();

  /// `first` is the step of the first frame, `period` the number of steps
  /// between two frames, and `timestep` the time step in ps.
  virtual llvm::Error open(const std::string &path, size_t numParticles,
                           int64_t first, int64_t period, double timestep,
                           const double box[3]) = 0;
  /// Continues the trajectory at `path`, written by MDIR, after its first
  /// `frames` frames (D130): the frames after them, which a run wrote past
  /// its last checkpoint, are removed. Fails if the file holds fewer
  /// frames, other particles, or another period. Returns the number of
  /// frames removed.
  virtual llvm::Expected<int64_t> append(const std::string &path,
                                         size_t numParticles, int64_t frames,
                                         int64_t period, double timestep,
                                         const double box[3]) = 0;
  /// Writes a frame of the step `step`, at the time `time` in ps.
  /// `positions` holds three numbers per particle, in Å.
  virtual void writeFrame(const float *positions, int64_t step,
                          double time) = 0;
  virtual void close();

  /// The number of frames that the file holds.
  int64_t getNumFrames() const { return numFrames; }
  /// The step of the last frame of the file, where the format records the
  /// steps and the writer knows it.
  virtual std::optional<int64_t> getLastStep() const { return std::nullopt; }

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
  /// Without a periodic cell (D142) the frames hold none: the cell that the
  /// run placed around the particles is not one of the system. Before
  /// `open` or `append`.
  void setPeriodic(bool value) { periodic = value; }

  /// Whether the writer takes the frames of the state as the run has them,
  /// in nm and without rounding (H5MDWriter), rather than `writeFrame`'s
  /// positions in Å and f32.
  virtual bool isExact() const { return false; }
  /// The cell of the frames that follow as the run has it, in nm: the
  /// edges, and the tilts b_x, c_x, c_y. Only an exact writer reads them.
  void setExactEdges(const double edges[3]) {
    for (int i = 0; i != 3; ++i)
      exactBox[i] = edges[i];
  }
  void setExactTilt(const double tilts[3]) {
    for (int i = 0; i != 3; ++i)
      exactTilt[i] = tilts[i];
  }

protected:
  std::FILE *file = nullptr;
  size_t numParticles = 0;
  int64_t first = 0;
  int64_t period = 0;
  double timestep = 0.0;
  double box[3] = {0.0, 0.0, 0.0};
  double tilt[3] = {0.0, 0.0, 0.0};
  double exactBox[3] = {0.0, 0.0, 0.0};
  double exactTilt[3] = {0.0, 0.0, 0.0};
  int32_t numFrames = 0;
  bool periodic = true;
};

/// The format DCD: positions in Å as 32-bit floating-point numbers, with
/// the cell, and a header that counts the frames.
class DCDWriter : public TrajectoryWriter {
public:
  ~DCDWriter() override;
  llvm::Error open(const std::string &path, size_t numParticles,
                   int64_t first, int64_t period, double timestep,
                   const double box[3]) override;
  llvm::Expected<int64_t> append(const std::string &path,
                                 size_t numParticles, int64_t frames,
                                 int64_t period, double timestep,
                                 const double box[3]) override;
  void writeFrame(const float *positions, int64_t step,
                  double time) override;

private:
  void writeHeader();
};

/// The format XTC of GROMACS: frames in XDR, each with its step, its time,
/// the vectors of the cell in nm, and the positions in nm rounded to a
/// thousandth and compressed: the differences between neighbors in the
/// file, which are small where the particles of a molecule follow one
/// another, take fewer bits (D141).
class XTCWriter : public TrajectoryWriter {
public:
  ~XTCWriter() override;
  llvm::Error open(const std::string &path, size_t numParticles,
                   int64_t first, int64_t period, double timestep,
                   const double box[3]) override;
  llvm::Expected<int64_t> append(const std::string &path,
                                 size_t numParticles, int64_t frames,
                                 int64_t period, double timestep,
                                 const double box[3]) override;
  void writeFrame(const float *positions, int64_t step,
                  double time) override;
};

/// The writer of the format `format`, DCD or XTC; that of H5MD is made with
/// its options (mdir/Driver/H5MD.h).
std::unique_ptr<TrajectoryWriter> createTrajectoryWriter(
    TrajectoryFormat format);

/// The positions of a frame of XTC, compressed as GROMACS compresses them
/// with the precision `precision` (1000: to a thousandth of a nm), and the
/// inverse. Positions are in nm, three per particle; they are exposed for
/// the tests of the format.
std::vector<uint8_t> compressXTC(const float *positions, size_t count,
                                 float precision);
llvm::Error decompressXTC(const uint8_t *data, size_t size, size_t count,
                          std::vector<float> &positions);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_TRAJECTORY_H

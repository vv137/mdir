// Checkpoints in the H5MD format: de Buyl et al., Comput. Phys. Commun. 185,
// 1546 (2014).
//
// The layout of a file:
//
//   /h5md                         version, author, creator
//   /particles/all/box            dimension, boundary, edges
//   /particles/all/position       step, time, value
//   /particles/all/velocity       step, time, value
//   /particles/all/force          step, time, value    if the state has them
//   /particles/all/id             the numbers of the particles
//   /particles/all/species        the types of the particles
//   /particles/all/mass
//   /parameters/mdir              what MDIR needs to continue the run, and
//                                 the run that wrote it (D129, D130); the
//                                 format and the hash of the state
//                                 (D173)
//   /parameters/mdir/fingerprint  what defined the run
//                                 (D172)
//   /parameters/mdir/tunables     the tunable parameters of a Python
//                                 simulation (D213), if it has any
//
// Entries that release 0.1.0 does not know (front_end, model_sha256,
// plan_sha256, tunables) are additional entries of format 1 with a hash of
// their own, extras_sha256 (D[python-checkpoints]).
//
// A quantity that changes with time has one frame: the state that the
// checkpoint holds.

#include "mdir/Driver/Checkpoint.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <array>
#include <cstdio>
#include <cstring>

#if MDIR_HAS_HDF5
#include <fcntl.h>
#include <hdf5.h>
#include <unistd.h>
#endif

using namespace mdir::driver;

std::string mdir::driver::compareCheckpoints(const Checkpoint &first,
                                             const Checkpoint &second) {
  if (first.getNumParticles() != second.getNumParticles())
    return "the numbers of particles differ";
  if (first.step != second.step)
    return "the steps differ: " + std::to_string(first.step) + " and " +
           std::to_string(second.step);

  auto compare = [](const std::vector<double> &a,
                    const std::vector<double> &b,
                    const char *name) -> std::string {
    if (a.size() != b.size())
      return std::string("one state has ") + name + ", the other has not";
    double largest = 0.0;
    size_t count = 0;
    for (size_t i = 0, e = a.size(); i != e; ++i) {
      // Compare the bits, so that a value that is not a number counts as
      // equal to itself.
      if (std::memcmp(&a[i], &b[i], sizeof(double)) == 0)
        continue;
      ++count;
      double difference = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
      if (difference > largest)
        largest = difference;
    }
    if (count == 0)
      return "";
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer),
                  "%zu values of the %s differ, by up to %.3e", count, name,
                  largest);
    return buffer;
  };
  for (auto [a, b, name] :
       {std::make_tuple(&first.positions, &second.positions, "positions"),
        std::make_tuple(&first.velocities, &second.velocities, "velocities"),
        std::make_tuple(&first.forces, &second.forces, "forces")}) {
    std::string difference = compare(*a, *b, name);
    if (!difference.empty())
      return difference;
  }
  return "";
}

std::string mdir::driver::getPreviousCheckpointPath(const std::string &path) {
  return path + ".prev";
}

namespace {

/// The text of the entries of `group` of a fingerprint, a line for each:
/// the name, a tab, and the value.
std::string getFingerprintText(const Fingerprint &fingerprint,
                               llvm::StringRef group) {
  std::string text;
  for (const FingerprintEntry &entry : fingerprint)
    if (entry.group == group)
      text += entry.name + "\t" + entry.value + "\n";
  return text;
}

const char *const fingerprintGroups[] = {"physics", "coupling", "execution"};

/// SHA-256 of everything that a checkpoint holds, in a fixed order
/// (D173).
std::string hashState(const Checkpoint &state) {
  llvm::SHA256 hash;
  auto bytes = [&](const void *data, size_t size) {
    hash.update(llvm::ArrayRef<uint8_t>(static_cast<const uint8_t *>(data),
                                        size));
  };
  auto number = [&](const auto &value) { bytes(&value, sizeof(value)); };
  auto numbers = [&](const auto &values) {
    uint64_t size = values.size();
    number(size);
    bytes(values.data(), size * sizeof(values[0]));
  };
  auto text = [&](const std::string &value) {
    uint64_t size = value.size();
    number(size);
    bytes(value.data(), size);
  };
  number(state.step);
  number(state.time);
  numbers(state.positions);
  numbers(state.velocities);
  numbers(state.forces);
  numbers(state.masses);
  numbers(state.species);
  bytes(state.box, sizeof(state.box));
  bytes(state.tilt, sizeof(state.tilt));
  uint8_t periodic = state.periodic ? 1 : 0;
  number(periodic);
  text(state.integrator);
  number(state.velocityOffset);
  text(state.precision);
  number(state.timestep);
  number(state.seed);
  number(state.firstStep);
  number(state.part);
  number(state.outputsPart);
  text(state.trajectory);
  number(state.frames);
  number(state.bath);
  numbers(state.barostatState);
  numbers(state.thermostatState);
  for (const char *group : fingerprintGroups)
    text(getFingerprintText(state.fingerprint, group));
  std::array<uint8_t, 32> digest = hash.final();
  return llvm::toHex(digest, /*LowerCase=*/true);
}

/// SHA-256 of the additional entries of a checkpoint
/// (D[python-checkpoints]), apart from `state_sha256` so that a reader of
/// release 0.1.0, which recomputes that and does not know these, reads the
/// file.
std::string hashExtras(const Checkpoint &state) {
  llvm::SHA256 hash;
  auto bytes = [&](const void *data, size_t size) {
    hash.update(llvm::ArrayRef<uint8_t>(static_cast<const uint8_t *>(data),
                                        size));
  };
  auto number = [&](const auto &value) { bytes(&value, sizeof(value)); };
  auto text = [&](const std::string &value) {
    uint64_t size = value.size();
    number(size);
    bytes(value.data(), size);
  };
  text(state.frontEnd);
  text(state.modelHash);
  text(state.planHash);
  text(state.tunableDeclarations);
  uint64_t count = state.tunables.size();
  number(count);
  for (const auto &[name, values] : state.tunables) {
    text(name);
    uint64_t size = values.size();
    number(size);
    bytes(values.data(), size * sizeof(double));
  }
  number(state.tunablesVersion);
  count = state.tunablesHistory.size();
  number(count);
  for (const auto &[step, version] : state.tunablesHistory) {
    number(step);
    number(version);
  }
  std::array<uint8_t, 32> digest = hash.final();
  return llvm::toHex(digest, /*LowerCase=*/true);
}

} // namespace

#if !MDIR_HAS_HDF5

bool mdir::driver::hasCheckpointSupport() { return false; }

llvm::Error mdir::driver::writeCheckpoint(const std::string &,
                                          const Checkpoint &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "MDIR was built without HDF5, which "
                                 "checkpoints need");
}

llvm::Expected<Checkpoint> mdir::driver::readCheckpoint(const std::string &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "MDIR was built without HDF5, which "
                                 "checkpoints need");
}

#else

bool mdir::driver::hasCheckpointSupport() { return true; }

namespace {

/// An identifier of the library that is released when it goes out of scope.
class Handle {
public:
  using Close = herr_t (*)(hid_t);

  Handle(hid_t id, Close close) : id(id), close(close) {}
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  ~Handle() {
    if (id >= 0)
      close(id);
  }

  operator hid_t() const { return id; }
  bool isValid() const { return id >= 0; }

private:
  hid_t id;
  Close close;
};

/// Writes the parts of a file. The first failure is kept; what follows it
/// does nothing.
class Writer {
public:
  explicit Writer(hid_t file) : file(file) {}

  bool hasFailed() const { return failed; }

  Handle createGroup(hid_t parent, const char *name) {
    hid_t group = H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
    failed |= group < 0;
    return Handle(group, H5Gclose);
  }

  void writeAttribute(hid_t object, const char *name, hid_t type,
                      const void *data, hsize_t count = 0) {
    if (failed)
      return;
    Handle space(count == 0 ? H5Screate(H5S_SCALAR)
                            : H5Screate_simple(1, &count, nullptr),
                 H5Sclose);
    Handle attribute(
        H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT),
        H5Aclose);
    failed |= !attribute.isValid() || H5Awrite(attribute, type, data) < 0;
  }

  void writeText(hid_t object, const char *name, const std::string &text) {
    if (failed)
      return;
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type, text.size() + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    writeAttribute(object, name, type, text.c_str());
  }

  void writeReal(hid_t object, const char *name, double value) {
    writeAttribute(object, name, H5T_NATIVE_DOUBLE, &value);
  }

  void writeDataset(hid_t parent, const char *name, hid_t type,
                    llvm::ArrayRef<hsize_t> shape, const void *data,
                    const char *unit = nullptr) {
    if (failed)
      return;
    Handle space(H5Screate_simple(shape.size(), shape.data(), nullptr),
                 H5Sclose);
    Handle dataset(H5Dcreate2(parent, name, type, space, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT),
                   H5Dclose);
    failed |= !dataset.isValid() ||
              H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0;
    if (unit)
      writeText(dataset, "unit", unit);
  }

  /// A dataset of one string, which may be longer than an attribute holds.
  void writeTextDataset(hid_t parent, const char *name,
                        const std::string &text) {
    if (failed)
      return;
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type, text.size() + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle dataset(H5Dcreate2(parent, name, type, space, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT),
                   H5Dclose);
    failed |= !dataset.isValid() ||
              H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       text.c_str()) < 0;
  }

  /// A quantity that changes with time, with the one frame `values`, which
  /// is of the time `time`.
  void writeElement(hid_t parent, const char *name, const Checkpoint &state,
                    double time, const std::vector<double> &values,
                    const char *unit) {
    Handle group = createGroup(parent, name);
    int64_t step = state.step;
    writeDataset(group, "step", H5T_NATIVE_INT64, {1}, &step);
    writeDataset(group, "time", H5T_NATIVE_DOUBLE, {1}, &time, "ps");
    writeDataset(group, "value", H5T_NATIVE_DOUBLE,
                 {1, state.getNumParticles(), 3}, values.data(), unit);
  }

private:
  hid_t file;
  bool failed = false;
};

/// Reads the parts of a file. The first failure is kept, with what failed.
class Reader {
public:
  explicit Reader(hid_t file) : file(file) {}

  bool hasFailed() const { return !failure.empty(); }
  const std::string &getFailure() const { return failure; }

  bool has(const char *path) {
    return H5Lexists(file, path, H5P_DEFAULT) > 0;
  }
  bool hasAttribute(const char *path, const char *name) {
    return H5Aexists_by_name(file, path, name, H5P_DEFAULT) > 0;
  }

  /// Reads the dataset at `path`, which holds `count` values.
  template <typename T>
  void readDataset(const char *path, hid_t type, size_t count,
                   std::vector<T> &values) {
    if (hasFailed())
      return;
    Handle dataset(H5Dopen2(file, path, H5P_DEFAULT), H5Dclose);
    if (!dataset.isValid())
      return fail(path, "is missing");
    Handle space(H5Dget_space(dataset), H5Sclose);
    if (H5Sget_simple_extent_npoints(space) !=
        static_cast<hssize_t>(count))
      return fail(path, "does not have the size that the number of "
                        "particles asks for");
    values.resize(count);
    if (H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                values.data()) < 0)
      fail(path, "cannot be read");
  }

  /// The number of values of the dataset at `path`.
  size_t getSize(const char *path) {
    Handle dataset(H5Dopen2(file, path, H5P_DEFAULT), H5Dclose);
    if (!dataset.isValid()) {
      fail(path, "is missing");
      return 0;
    }
    Handle space(H5Dget_space(dataset), H5Sclose);
    return static_cast<size_t>(H5Sget_simple_extent_npoints(space));
  }

  template <typename T>
  void readAttribute(const char *path, const char *name, hid_t type,
                     T &value) {
    if (hasFailed())
      return;
    Handle attribute(
        H5Aopen_by_name(file, path, name, H5P_DEFAULT, H5P_DEFAULT),
        H5Aclose);
    if (!attribute.isValid() || H5Aread(attribute, type, &value) < 0)
      fail(path, (std::string("has no attribute '") + name + "'").c_str());
  }

  void readText(const char *path, const char *name, std::string &text) {
    if (hasFailed())
      return;
    Handle attribute(
        H5Aopen_by_name(file, path, name, H5P_DEFAULT, H5P_DEFAULT),
        H5Aclose);
    if (!attribute.isValid())
      return fail(path,
                  (std::string("has no attribute '") + name + "'").c_str());
    Handle stored(H5Aget_type(attribute), H5Tclose);
    size_t size = H5Tget_size(stored);
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type, size + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    std::vector<char> buffer(size + 1, '\0');
    if (H5Aread(attribute, type, buffer.data()) < 0)
      return fail(path, "has an attribute that cannot be read");
    text = buffer.data();
  }

  void readTextDataset(const char *path, std::string &text) {
    if (hasFailed())
      return;
    Handle dataset(H5Dopen2(file, path, H5P_DEFAULT), H5Dclose);
    if (!dataset.isValid())
      return fail(path, "is missing");
    Handle stored(H5Dget_type(dataset), H5Tclose);
    size_t size = H5Tget_size(stored);
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type, size + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    std::vector<char> buffer(size + 1, '\0');
    if (H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                buffer.data()) < 0)
      return fail(path, "cannot be read");
    text = buffer.data();
  }

private:
  void fail(const char *path, const char *what) {
    failure = std::string("'") + path + "' " + what;
  }

  hid_t file;
  std::string failure;
};

/// Writes what the system holds of a file, or of the names of a directory,
/// to stable storage.
bool synchronize(const std::string &path, bool directory) {
  int descriptor =
      ::open(path.c_str(), directory ? O_RDONLY | O_DIRECTORY : O_RDONLY);
  if (descriptor < 0)
    return false;
  bool synchronized = ::fsync(descriptor) == 0;
  ::close(descriptor);
  return synchronized;
}

} // namespace

llvm::Error mdir::driver::writeCheckpoint(const std::string &path,
                                          const Checkpoint &checkpoint) {
  // The library reports errors on its own. The driver reports them.
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

  std::string partial = path + ".partial";
  bool failed;
  {
    Handle file(
        H5Fcreate(partial.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
        H5Fclose);
    if (!file.isValid())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot write '%s'", partial.c_str());
    Writer writer(file);
    size_t count = checkpoint.getNumParticles();

    {
      Handle h5md = writer.createGroup(file, "h5md");
      int version[2] = {1, 1};
      writer.writeAttribute(h5md, "version", H5T_NATIVE_INT, version, 2);
      // H5MD asks for an author; MDIR does not know who runs it.
      Handle author = writer.createGroup(h5md, "author");
      writer.writeText(author, "name", "unknown");
      Handle creator = writer.createGroup(h5md, "creator");
      writer.writeText(creator, "name", checkpoint.creator);
      writer.writeText(creator, "version", checkpoint.creatorVersion.empty()
                                               ? "unknown"
                                               : checkpoint.creatorVersion);
    }

    {
      Handle particles = writer.createGroup(file, "particles");
      Handle all = writer.createGroup(particles, "all");

      {
        Handle box = writer.createGroup(all, "box");
        int dimension = 3;
        writer.writeAttribute(box, "dimension", H5T_NATIVE_INT, &dimension);
        Handle text(H5Tcopy(H5T_C_S1), H5Tclose);
        H5Tset_size(text, 9);
        H5Tset_strpad(text, H5T_STR_NULLTERM);
        char boundary[3][9] = {"periodic", "periodic", "periodic"};
        if (!checkpoint.periodic)
          for (auto &text : boundary)
            std::strcpy(text, "none");
        writer.writeAttribute(box, "boundary", text, boundary, 3);
        // H5MD: the edges of a rectangular cell, or the matrix of the
        // edge vectors of a triclinic one, a row each.
        bool triclinic = checkpoint.tilt[0] != 0.0 ||
                         checkpoint.tilt[1] != 0.0 ||
                         checkpoint.tilt[2] != 0.0;
        if (triclinic) {
          double matrix[9] = {checkpoint.box[0], 0.0, 0.0,
                              checkpoint.tilt[0], checkpoint.box[1], 0.0,
                              checkpoint.tilt[1], checkpoint.tilt[2],
                              checkpoint.box[2]};
          writer.writeDataset(box, "edges", H5T_NATIVE_DOUBLE, {3, 3},
                              matrix, "nm");
        } else {
          writer.writeDataset(box, "edges", H5T_NATIVE_DOUBLE, {3},
                              checkpoint.box, "nm");
        }
      }

      // With leapfrog the velocities are of another time than the
      // positions.
      double velocityTime =
          checkpoint.time + checkpoint.velocityOffset * checkpoint.timestep;
      writer.writeElement(all, "position", checkpoint, checkpoint.time,
                          checkpoint.positions, "nm");
      writer.writeElement(all, "velocity", checkpoint, velocityTime,
                          checkpoint.velocities, "nm ps-1");
      if (!checkpoint.forces.empty())
        writer.writeElement(all, "force", checkpoint, checkpoint.time,
                            checkpoint.forces, "kJ mol-1 nm-1");

      std::vector<int64_t> ids(count);
      for (size_t i = 0; i != count; ++i)
        ids[i] = static_cast<int64_t>(i);
      writer.writeDataset(all, "id", H5T_NATIVE_INT64, {count}, ids.data());
      writer.writeDataset(all, "species", H5T_NATIVE_INT32, {count},
                          checkpoint.species.data());
      writer.writeDataset(all, "mass", H5T_NATIVE_DOUBLE, {count},
                          checkpoint.masses.data(), "u");
    }

    {
      Handle parameters = writer.createGroup(file, "parameters");
      Handle mdir = writer.createGroup(parameters, "mdir");
      int format = checkpointFormat;
      writer.writeAttribute(mdir, "format", H5T_NATIVE_INT, &format);
      writer.writeText(mdir, "state_sha256", hashState(checkpoint));
      writer.writeText(mdir, "integrator", checkpoint.integrator);
      writer.writeReal(mdir, "velocity_offset", checkpoint.velocityOffset);
      writer.writeText(mdir, "precision", checkpoint.precision);
      writer.writeReal(mdir, "timestep", checkpoint.timestep);
      writer.writeAttribute(mdir, "seed", H5T_NATIVE_UINT64,
                            &checkpoint.seed);
      writer.writeAttribute(mdir, "first_step", H5T_NATIVE_INT64,
                            &checkpoint.firstStep);
      writer.writeAttribute(mdir, "part", H5T_NATIVE_INT64, &checkpoint.part);
      writer.writeAttribute(mdir, "outputs_part", H5T_NATIVE_INT64,
                            &checkpoint.outputsPart);
      int periodic = checkpoint.periodic ? 1 : 0;
      writer.writeAttribute(mdir, "periodic", H5T_NATIVE_INT, &periodic);
      writer.writeText(mdir, "trajectory", checkpoint.trajectory);
      writer.writeAttribute(mdir, "frames", H5T_NATIVE_INT64,
                            &checkpoint.frames);
      writer.writeReal(mdir, "bath", checkpoint.bath);
      // The states of [free_energy] and the one of the run (D161).
      if (!checkpoint.freeEnergy.empty()) {
        writer.writeText(mdir, "free_energy", checkpoint.freeEnergy);
        writer.writeAttribute(mdir, "free_energy_state", H5T_NATIVE_INT64,
                              &checkpoint.freeEnergyState);
        writer.writeDataset(mdir, "free_energy_lambda", H5T_NATIVE_DOUBLE,
                            {checkpoint.freeEnergyLambda.size()},
                            checkpoint.freeEnergyLambda.data());
      }
      if (!checkpoint.barostatState.empty())
        writer.writeDataset(mdir, "barostat_state", H5T_NATIVE_DOUBLE,
                            {checkpoint.barostatState.size()},
                            checkpoint.barostatState.data(), "kJ mol-1");
      if (!checkpoint.thermostatState.empty())
        writer.writeDataset(mdir, "thermostat_state", H5T_NATIVE_DOUBLE,
                            {checkpoint.thermostatState.size()},
                            checkpoint.thermostatState.data(), "");
      Handle fingerprint = writer.createGroup(mdir, "fingerprint");
      for (const char *group : fingerprintGroups)
        writer.writeTextDataset(
            fingerprint, group,
            getFingerprintText(checkpoint.fingerprint, group));
      // The additional entries (D[python-checkpoints]).
      if (checkpoint.hasExtras()) {
        writer.writeText(mdir, "extras_sha256", hashExtras(checkpoint));
        writer.writeText(mdir, "front_end", checkpoint.frontEnd);
        writer.writeText(mdir, "model_sha256", checkpoint.modelHash);
        writer.writeText(mdir, "plan_sha256", checkpoint.planHash);
        if (!checkpoint.tunables.empty() ||
            !checkpoint.tunableDeclarations.empty()) {
          Handle tunables = writer.createGroup(mdir, "tunables");
          writer.writeTextDataset(tunables, "declarations",
                                  checkpoint.tunableDeclarations);
          std::string names;
          for (const auto &[name, values] : checkpoint.tunables)
            names += name + "\n";
          writer.writeTextDataset(tunables, "names", names);
          for (size_t k = 0; k != checkpoint.tunables.size(); ++k) {
            const std::vector<double> &values = checkpoint.tunables[k].second;
            writer.writeDataset(tunables,
                                ("values" + std::to_string(k)).c_str(),
                                H5T_NATIVE_DOUBLE, {values.size()},
                                values.data());
          }
          writer.writeAttribute(tunables, "version", H5T_NATIVE_INT64,
                                &checkpoint.tunablesVersion);
          std::vector<int64_t> history;
          for (const auto &[step, version] : checkpoint.tunablesHistory) {
            history.push_back(step);
            history.push_back(version);
          }
          writer.writeDataset(tunables, "history", H5T_NATIVE_INT64,
                              {checkpoint.tunablesHistory.size(), 2},
                              history.data());
        }
      }
    }
    failed = writer.hasFailed() || H5Fflush(file, H5F_SCOPE_GLOBAL) < 0;
  }

  // On stable storage before it takes its name: a file system that delays
  // its writes (ext4 without auto_da_alloc, Lustre, NFS) may otherwise
  // leave an empty file under the name after a crash (D173).
  if (!failed)
    failed = !synchronize(partial, /*directory=*/false);
  if (failed) {
    std::remove(partial.c_str());
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write '%s'", path.c_str());
  }
  // The checkpoint before stays as `.prev` (D132). A second name for it is
  // made before the rename that replaces it, so that `path` holds a
  // complete state at every moment. Where the file system has no hard
  // links, the checkpoint before is renamed instead, and for a moment only
  // `.prev` holds a state; `mdir run --continue` then refuses rather than
  // begin the run anew.
  std::string previous = getPreviousCheckpointPath(path);
  if (llvm::sys::fs::exists(path)) {
    std::remove(previous.c_str());
    if (::link(path.c_str(), previous.c_str()) != 0 &&
        std::rename(path.c_str(), previous.c_str()) != 0) {
      std::remove(partial.c_str());
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot keep '%s' as '%s'", path.c_str(),
                                     previous.c_str());
    }
  }
  if (std::rename(partial.c_str(), path.c_str()) != 0) {
    std::remove(partial.c_str());
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write '%s'", path.c_str());
  }
  // The new names, too.
  llvm::SmallString<256> directory(path);
  llvm::sys::path::remove_filename(directory);
  if (directory.empty())
    directory = ".";
  if (!synchronize(directory.str().str(), /*directory=*/true))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot synchronize the directory of '%s'",
                                   path.c_str());
  return llvm::Error::success();
}

llvm::Expected<Checkpoint>
mdir::driver::readCheckpoint(const std::string &path) {
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

  Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.isValid())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read '%s' as an HDF5 file",
                                   path.c_str());
  Reader reader(file);
  if (!reader.has("/parameters/mdir"))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "'%s' is not a checkpoint of MDIR: it has no group "
        "'/parameters/mdir'",
        path.c_str());
  // The format, the contract of release 0.1.0 (D173). A
  // file of a development build before it records no fingerprint and no
  // hash of its state.
  int format = 0;
  if (reader.hasAttribute("/parameters/mdir", "format"))
    reader.readAttribute("/parameters/mdir", "format", H5T_NATIVE_INT,
                         format);
  if (format > checkpointFormat)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "'%s' is a checkpoint of format %d, written by a newer MDIR; this "
        "one reads format %d",
        path.c_str(), format, checkpointFormat);
  if (format != checkpointFormat ||
      !reader.hasAttribute("/parameters/mdir", "state_sha256") ||
      !reader.has("/parameters/mdir/fingerprint"))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "'%s' was written by a development build of MDIR before release "
        "0.1.0, whose checkpoints this one does not read; begin the run "
        "from the positions of its last frame",
        path.c_str());

  Checkpoint checkpoint;
  size_t count = reader.getSize("/particles/all/mass");
  reader.readDataset("/particles/all/mass", H5T_NATIVE_DOUBLE, count,
                     checkpoint.masses);
  reader.readDataset("/particles/all/species", H5T_NATIVE_INT32, count,
                     checkpoint.species);
  reader.readDataset("/particles/all/position/value", H5T_NATIVE_DOUBLE,
                     3 * count, checkpoint.positions);
  reader.readDataset("/particles/all/velocity/value", H5T_NATIVE_DOUBLE,
                     3 * count, checkpoint.velocities);
  if (reader.has("/particles/all/force"))
    reader.readDataset("/particles/all/force/value", H5T_NATIVE_DOUBLE,
                       3 * count, checkpoint.forces);

  std::vector<double> edges, time;
  std::vector<int64_t> step;
  // Three edges, or the matrix of a triclinic cell.
  int64_t edgeCount = reader.getSize("/particles/all/box/edges");
  reader.readDataset("/particles/all/box/edges", H5T_NATIVE_DOUBLE,
                     edgeCount == 9 ? 9 : 3, edges);
  reader.readDataset("/particles/all/position/step", H5T_NATIVE_INT64, 1,
                     step);
  reader.readDataset("/particles/all/position/time", H5T_NATIVE_DOUBLE, 1,
                     time);

  reader.readText("/parameters/mdir", "integrator", checkpoint.integrator);
  reader.readText("/parameters/mdir", "precision", checkpoint.precision);
  reader.readAttribute("/parameters/mdir", "velocity_offset",
                       H5T_NATIVE_DOUBLE, checkpoint.velocityOffset);
  reader.readAttribute("/parameters/mdir", "timestep", H5T_NATIVE_DOUBLE,
                       checkpoint.timestep);
  reader.readAttribute("/parameters/mdir", "seed", H5T_NATIVE_UINT64,
                       checkpoint.seed);
  // The run that wrote the checkpoint (D129, D130).
  int periodic = 1;
  reader.readAttribute("/parameters/mdir", "periodic", H5T_NATIVE_INT,
                       periodic);
  checkpoint.periodic = periodic != 0;
  reader.readAttribute("/parameters/mdir", "first_step", H5T_NATIVE_INT64,
                       checkpoint.firstStep);
  reader.readAttribute("/parameters/mdir", "part", H5T_NATIVE_INT64,
                       checkpoint.part);
  reader.readAttribute("/parameters/mdir", "outputs_part", H5T_NATIVE_INT64,
                       checkpoint.outputsPart);
  reader.readText("/parameters/mdir", "trajectory", checkpoint.trajectory);
  reader.readAttribute("/parameters/mdir", "frames", H5T_NATIVE_INT64,
                       checkpoint.frames);
  reader.readAttribute("/parameters/mdir", "bath", H5T_NATIVE_DOUBLE,
                       checkpoint.bath);
  // The state of the last scaling of a barostat that scales every step
  // (D92, D119): nine numbers.
  if (reader.has("/parameters/mdir/barostat_state"))
    reader.readDataset("/parameters/mdir/barostat_state", H5T_NATIVE_DOUBLE,
                       9, checkpoint.barostatState);
  // The states of [free_energy] (D161).
  if (reader.hasAttribute("/parameters/mdir", "free_energy_state")) {
    reader.readText("/parameters/mdir", "free_energy", checkpoint.freeEnergy);
    reader.readAttribute("/parameters/mdir", "free_energy_state",
                         H5T_NATIVE_INT64, checkpoint.freeEnergyState);
    reader.readDataset("/parameters/mdir/free_energy_lambda",
                       H5T_NATIVE_DOUBLE,
                       reader.getSize("/parameters/mdir/free_energy_lambda"),
                       checkpoint.freeEnergyLambda);
  }
  // The state of a Nose-Hoover chain (D163a).
  if (reader.has("/parameters/mdir/thermostat_state"))
    reader.readDataset("/parameters/mdir/thermostat_state",
                       H5T_NATIVE_DOUBLE,
                       reader.getSize("/parameters/mdir/thermostat_state"),
                       checkpoint.thermostatState);
  // What defined the run (D172).
  for (const char *group : fingerprintGroups) {
    std::string text;
    reader.readTextDataset(
        ("/parameters/mdir/fingerprint/" + std::string(group)).c_str(), text);
    llvm::SmallVector<llvm::StringRef, 16> lines;
    llvm::StringRef(text).split(lines, '\n', -1, /*KeepEmpty=*/false);
    for (llvm::StringRef line : lines) {
      auto [name, value] = line.split('\t');
      checkpoint.fingerprint.push_back({group, name.str(), value.str()});
    }
  }
  // The additional entries (D[python-checkpoints]), which are read only
  // with their hash.
  std::string extras;
  if (reader.hasAttribute("/parameters/mdir", "extras_sha256")) {
    reader.readText("/parameters/mdir", "extras_sha256", extras);
    reader.readText("/parameters/mdir", "front_end", checkpoint.frontEnd);
    reader.readText("/parameters/mdir", "model_sha256", checkpoint.modelHash);
    reader.readText("/parameters/mdir", "plan_sha256", checkpoint.planHash);
    if (reader.has("/parameters/mdir/tunables")) {
      const char *group = "/parameters/mdir/tunables";
      reader.readTextDataset("/parameters/mdir/tunables/declarations",
                             checkpoint.tunableDeclarations);
      std::string names;
      reader.readTextDataset("/parameters/mdir/tunables/names", names);
      llvm::SmallVector<llvm::StringRef, 8> lines;
      llvm::StringRef(names).split(lines, '\n', -1, /*KeepEmpty=*/false);
      for (size_t k = 0; k != lines.size() && !reader.hasFailed(); ++k) {
        std::string path =
            std::string(group) + "/values" + std::to_string(k);
        std::vector<double> values;
        size_t size = reader.getSize(path.c_str());
        reader.readDataset(path.c_str(), H5T_NATIVE_DOUBLE, size, values);
        checkpoint.tunables.emplace_back(lines[k].str(), std::move(values));
      }
      reader.readAttribute(group, "version", H5T_NATIVE_INT64,
                           checkpoint.tunablesVersion);
      std::vector<int64_t> history;
      size_t size = reader.getSize("/parameters/mdir/tunables/history");
      reader.readDataset("/parameters/mdir/tunables/history",
                         H5T_NATIVE_INT64, size, history);
      for (size_t k = 0; k + 1 < history.size(); k += 2)
        checkpoint.tunablesHistory.push_back({history[k], history[k + 1]});
    }
  }
  reader.readText("/h5md/creator", "name", checkpoint.creator);
  reader.readText("/h5md/creator", "version", checkpoint.creatorVersion);
  std::string stored;
  reader.readText("/parameters/mdir", "state_sha256", stored);

  if (reader.hasFailed())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "in '%s': %s", path.c_str(),
                                   reader.getFailure().c_str());
  if (edges.size() == 9) {
    checkpoint.box[0] = edges[0];
    checkpoint.box[1] = edges[4];
    checkpoint.box[2] = edges[8];
    checkpoint.tilt[0] = edges[3];
    checkpoint.tilt[1] = edges[6];
    checkpoint.tilt[2] = edges[7];
  } else {
    for (int i = 0; i != 3; ++i)
      checkpoint.box[i] = edges[i];
  }
  checkpoint.step = step[0];
  checkpoint.time = time[0];
  if (hashState(checkpoint) != stored)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "the state in '%s' does not match the hash it was written with: the "
        "file was changed or damaged after it was written",
        path.c_str());
  if (!extras.empty() && hashExtras(checkpoint) != extras)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "the entries of the model, the plan, or the tunables in '%s' do not "
        "match the hash they were written with: the file was changed or "
        "damaged after it was written",
        path.c_str());
  return std::move(checkpoint);
}

#endif // MDIR_HAS_HDF5

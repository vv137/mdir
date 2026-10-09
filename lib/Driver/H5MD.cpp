// Trajectories in the H5MD format, version 1.1: de Buyl et al., Comput.
// Phys. Commun. 185, 1546 (2014), and the units module, version 1.0, of
// the same specification.
//
// The layout of a file (D[h5md-reporter], docs/python-h5md.md):
//
//   /h5md                            version, author, creator, modules/units
//   /particles/all/box               dimension, boundary
//   /particles/all/box/edges         step, time (links), value [K][3] or
//                                    [K][3][3], the cell vectors in rows
//   /particles/all/position          step [K], time [K], value [K][N][3]
//   /particles/all/velocity          step, time, value          if asked
//   /particles/all/force             step, time, value          if asked
//   /particles/all/mass, species     [N]
//   /observables/potential_energy    step, time, value [K]      if asked
//   /observables/tunables_version    step, time, value [K]      if asked
//   /parameters/mdir                 trajectory_format, precision, timestep,
//                                    period, velocity_offset, front_end
//
// Elements sampled together share `step` and `time` through hard links.
// The numbers are those of the state, in nm, ps, and kJ/mol; a row of a
// value is a particle in the order of the input.

#include "mdir/Driver/H5MD.h"

#include "llvm/ADT/Twine.h"

#include <cstring>
#include <mutex>

#if MDIR_HAS_HDF5
#include <hdf5.h>
#endif

using namespace mdir::driver;

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 message.str().c_str());
}

} // namespace

#if !MDIR_HAS_HDF5

static const char noHDF5[] = "MDIR was built without HDF5, which "
                             "trajectories in H5MD need";

struct H5MDWriter::File {};
struct H5MDReader::File {};

H5MDWriter::H5MDWriter(H5MDOptions options) : options(std::move(options)) {}
H5MDWriter::~H5MDWriter() = default;
llvm::Error H5MDWriter::open(const std::string &, size_t, int64_t, int64_t,
                             double, const double[3]) {
  return makeError(noHDF5);
}
llvm::Expected<int64_t> H5MDWriter::append(const std::string &, size_t,
                                           int64_t, int64_t, double,
                                           const double[3]) {
  return makeError(noHDF5);
}
void H5MDWriter::writeFrame(const float *, int64_t, double) {}
void H5MDWriter::close() {}
bool H5MDWriter::writeState(const double *, const double *, const double *,
                            int64_t, double, const H5MDExtras &) {
  failure = noHDF5;
  return false;
}

H5MDReader::H5MDReader() = default;
H5MDReader::~H5MDReader() = default;
llvm::Expected<std::unique_ptr<H5MDReader>>
H5MDReader::open(const std::string &, const std::string &) {
  return makeError(noHDF5);
}
llvm::Expected<H5MDFrame> H5MDReader::read(size_t) {
  return makeError(noHDF5);
}
void H5MDReader::close() {}

#else

namespace {

/// The library is not thread-safe unless built so: a frame written by a run
/// and a file read by another thread take their turns.
std::mutex &getLibraryMutex() {
  static std::mutex mutex;
  return mutex;
}

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

/// A string type of fixed length for `text`, in ASCII, as H5MD asks.
hid_t makeTextType(size_t size) {
  hid_t type = H5Tcopy(H5T_C_S1);
  H5Tset_size(type, size);
  H5Tset_strpad(type, H5T_STR_NULLTERM);
  H5Tset_cset(type, H5T_CSET_ASCII);
  return type;
}

bool writeAttribute(hid_t object, const char *name, hid_t fileType,
                    hid_t memoryType, const void *data, hsize_t count = 0) {
  Handle space(count == 0 ? H5Screate(H5S_SCALAR)
                          : H5Screate_simple(1, &count, nullptr),
               H5Sclose);
  Handle attribute(
      H5Acreate2(object, name, fileType, space, H5P_DEFAULT, H5P_DEFAULT),
      H5Aclose);
  return attribute.isValid() && H5Awrite(attribute, memoryType, data) >= 0;
}

bool writeText(hid_t object, const char *name, const std::string &text) {
  Handle type(makeTextType(text.size() + 1), H5Tclose);
  return writeAttribute(object, name, type, type, text.c_str());
}

/// Reads a string attribute of fixed or variable length; empty if absent.
std::string readText(hid_t object, const char *name) {
  if (H5Aexists(object, name) <= 0)
    return "";
  Handle attribute(H5Aopen(object, name, H5P_DEFAULT), H5Aclose);
  if (!attribute.isValid())
    return "";
  Handle stored(H5Aget_type(attribute), H5Tclose);
  if (H5Tget_class(stored) != H5T_STRING)
    return "";
  Handle space(H5Aget_space(attribute), H5Sclose);
  if (H5Sget_simple_extent_npoints(space) != 1)
    return "";
  if (H5Tis_variable_str(stored) > 0) {
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type, H5T_VARIABLE);
    H5Tset_cset(type, H5Tget_cset(stored));
    char *value = nullptr;
    std::string text;
    if (H5Aread(attribute, type, &value) >= 0 && value) {
      text = value;
      H5free_memory(value);
    }
    return text;
  }
  size_t size = H5Tget_size(stored);
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
  H5Tset_size(type, size + 1);
  H5Tset_strpad(type, H5T_STR_NULLTERM);
  H5Tset_cset(type, H5Tget_cset(stored));
  std::vector<char> buffer(size + 1, '\0');
  if (H5Aread(attribute, type, buffer.data()) < 0)
    return "";
  return buffer.data();
}

bool exists(hid_t parent, const char *name) {
  return H5Lexists(parent, name, H5P_DEFAULT) > 0;
}

/// Whether every link of the absolute path `path` exists.
bool existsPath(hid_t file, const std::string &path) {
  for (size_t at = path.find('/', 1);; at = path.find('/', at + 1)) {
    std::string prefix = path.substr(0, at);
    if (H5Lexists(file, prefix.c_str(), H5P_DEFAULT) <= 0)
      return false;
    if (at == std::string::npos)
      return true;
  }
}

/// Whether the attribute `boundary` of a box says that no direction is
/// periodic.
bool hasNoBoundary(hid_t box) {
  if (H5Aexists(box, "boundary") <= 0)
    return false;
  Handle attribute(H5Aopen(box, "boundary", H5P_DEFAULT), H5Aclose);
  Handle stored(H5Aget_type(attribute), H5Tclose);
  Handle space(H5Aget_space(attribute), H5Sclose);
  hssize_t count = H5Sget_simple_extent_npoints(space);
  if (!attribute.isValid() || H5Tget_class(stored) != H5T_STRING ||
      H5Tis_variable_str(stored) > 0 || count <= 0)
    return false;
  size_t size = H5Tget_size(stored);
  std::vector<char> buffer(size * count + 1, '\0');
  if (H5Aread(attribute, stored, buffer.data()) < 0)
    return false;
  for (hssize_t k = 0; k != count; ++k)
    if (std::strncmp(buffer.data() + k * size, "none", size) != 0)
      return false;
  return true;
}

/// The extents of a dataset.
std::vector<hsize_t> getShape(hid_t dataset) {
  Handle space(H5Dget_space(dataset), H5Sclose);
  int rank = H5Sget_simple_extent_ndims(space);
  std::vector<hsize_t> shape(rank > 0 ? rank : 0);
  if (rank > 0)
    H5Sget_simple_extent_dims(space, shape.data(), nullptr);
  return shape;
}

/// A dataset of rows that grows along its first extent, with `rows` rows
/// in a chunk. `shape[0]` is not read.
hid_t createSeries(hid_t parent, const char *name, hid_t type,
                   std::vector<hsize_t> shape, hsize_t rows) {
  std::vector<hsize_t> most = shape, chunk = shape;
  shape[0] = 0;
  most[0] = H5S_UNLIMITED;
  chunk[0] = rows;
  Handle space(H5Screate_simple(shape.size(), shape.data(), most.data()),
               H5Sclose);
  Handle create(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
  H5Pset_chunk(create, chunk.size(), chunk.data());
  return H5Dcreate2(parent, name, type, space, H5P_DEFAULT, create,
                    H5P_DEFAULT);
}

/// Writes `data` as the row `row` of a dataset of rows, which grows to
/// hold it.
bool writeRow(hid_t dataset, hid_t memoryType, hsize_t row, const void *data) {
  std::vector<hsize_t> shape = getShape(dataset);
  if (shape.empty())
    return false;
  if (shape[0] < row + 1) {
    shape[0] = row + 1;
    if (H5Dset_extent(dataset, shape.data()) < 0)
      return false;
  }
  Handle space(H5Dget_space(dataset), H5Sclose);
  std::vector<hsize_t> start(shape.size(), 0), count = shape;
  start[0] = row;
  count[0] = 1;
  if (H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), nullptr,
                          count.data(), nullptr) < 0)
    return false;
  Handle memory(H5Screate_simple(count.size(), count.data(), nullptr),
                H5Sclose);
  return H5Dwrite(dataset, memoryType, memory, space, H5P_DEFAULT, data) >= 0;
}

/// Reads the row `row` of a dataset of rows.
bool readRow(hid_t dataset, hid_t memoryType, hsize_t row, void *data) {
  std::vector<hsize_t> shape = getShape(dataset);
  if (shape.empty() || shape[0] <= row)
    return false;
  Handle space(H5Dget_space(dataset), H5Sclose);
  std::vector<hsize_t> start(shape.size(), 0), count = shape;
  start[0] = row;
  count[0] = 1;
  if (H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), nullptr,
                          count.data(), nullptr) < 0)
    return false;
  Handle memory(H5Screate_simple(count.size(), count.data(), nullptr),
                H5Sclose);
  return H5Dread(dataset, memoryType, memory, space, H5P_DEFAULT, data) >= 0;
}

/// The rows of a chunk: whole frames, one if a frame is 64 KiB or more,
/// otherwise as many as fill 64 KiB.
hsize_t getFrameRows(size_t bytes) {
  const size_t target = 64 * 1024;
  if (bytes == 0)
    return 1024;
  if (bytes >= target)
    return 1;
  return std::min<size_t>(1024, (target + bytes - 1) / bytes);
}

constexpr int trajectoryFormat = 1;

} // namespace

//===----------------------------------------------------------------------===//
// The writer
//===----------------------------------------------------------------------===//

struct H5MDWriter::File {
  hid_t file = -1, position = -1, velocity = -1, force = -1, edges = -1,
        step = -1, time = -1, velocityTime = -1, energy = -1, version = -1;
  ~File() {
    for (hid_t dataset : {position, velocity, force, edges, step, time,
                          velocityTime, energy, version})
      if (dataset >= 0)
        H5Dclose(dataset);
    if (file >= 0)
      H5Fclose(file);
  }
  /// The datasets of one row per frame.
  std::vector<hid_t> getSeries() const {
    std::vector<hid_t> all;
    for (hid_t dataset : {position, velocity, force, edges, step, time,
                          velocityTime, energy, version})
      if (dataset >= 0)
        all.push_back(dataset);
    return all;
  }
};

H5MDWriter::H5MDWriter(H5MDOptions options) : options(std::move(options)) {}

H5MDWriter::~H5MDWriter() { close(); }

void H5MDWriter::close() {
  // Nothing to close, and the mutex may be held: a reader that `open`
  // refuses is destroyed inside it.
  if (!h5)
    return;
  std::lock_guard<std::mutex> lock(getLibraryMutex());
  h5.reset();
}

void H5MDWriter::writeFrame(const float *, int64_t, double) {
  failure = "the writer of H5MD takes the frames of the state; this is a "
            "defect of mdir";
}

llvm::Error H5MDWriter::open(const std::string &path, size_t count,
                             int64_t firstStep, int64_t framePeriod,
                             double dt, const double[3]) {
  std::lock_guard<std::mutex> lock(getLibraryMutex());
  // The library reports errors on its own. The driver reports them.
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  numParticles = count;
  first = firstStep;
  period = framePeriod;
  timestep = dt;
  numFrames = 0;
  lastStep.reset();
  failure.clear();
  if (!options.masses.empty() && options.masses.size() != count)
    return makeError("the masses of an H5MD trajectory are not those of "
                     "its particles; this is a defect of mdir");

  // The earliest format of the library: its superblock has no mark of a
  // writer, so a file that a killed run leaves opens as it is.
  auto h = std::make_unique<File>();
  h->file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (h->file < 0)
    return makeError("cannot write '" + path + "'");
  bool ok = true;
  hid_t real = options.single ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
  size_t width = options.single ? 4 : 8;
  hsize_t rows = getFrameRows(3 * count * width);
  {
    Handle h5md(H5Gcreate2(h->file, "h5md", H5P_DEFAULT, H5P_DEFAULT,
                           H5P_DEFAULT),
                H5Gclose);
    int version[2] = {1, 1};
    ok &= writeAttribute(h5md, "version", H5T_STD_I32LE, H5T_NATIVE_INT,
                         version, 2);
    // H5MD asks for an author; MDIR does not know who runs it.
    Handle author(H5Gcreate2(h5md, "author", H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT),
                  H5Gclose);
    ok &= writeText(author, "name", "unknown");
    Handle creator(H5Gcreate2(h5md, "creator", H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT),
                   H5Gclose);
    ok &= writeText(creator, "name", "MDIR");
    ok &= writeText(creator, "version",
                    options.creatorVersion.empty() ? "unknown"
                                                   : options.creatorVersion);
    Handle modules(H5Gcreate2(h5md, "modules", H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT),
                   H5Gclose);
    Handle unitsModule(H5Gcreate2(modules, "units", H5P_DEFAULT, H5P_DEFAULT,
                                  H5P_DEFAULT),
                       H5Gclose);
    int unitsVersion[2] = {1, 0};
    ok &= writeAttribute(unitsModule, "version", H5T_STD_I32LE,
                         H5T_NATIVE_INT, unitsVersion, 2);
    ok &= writeText(unitsModule, "system", "SI");
  }
  {
    Handle particles(H5Gcreate2(h->file, "particles", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT),
                     H5Gclose);
    Handle all(H5Gcreate2(particles, "all", H5P_DEFAULT, H5P_DEFAULT,
                          H5P_DEFAULT),
               H5Gclose);
    {
      Handle position(H5Gcreate2(all, "position", H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT),
                      H5Gclose);
      h->step = createSeries(position, "step", H5T_STD_I64LE, {0}, 1024);
      h->time = createSeries(position, "time", H5T_IEEE_F64LE, {0}, 1024);
      h->position = createSeries(position, "value", real, {0, count, 3}, rows);
      ok &= h->step >= 0 && h->time >= 0 && h->position >= 0;
      ok = ok && writeText(h->time, "unit", "ps") &&
           writeText(h->position, "unit", "nm");
    }
    // Elements sampled with the positions share their steps and times.
    auto link = [&](hid_t group, bool withTime = true) {
      ok = ok && H5Lcreate_hard(all, "position/step", group, "step",
                                H5P_DEFAULT, H5P_DEFAULT) >= 0;
      if (withTime)
        ok = ok && H5Lcreate_hard(all, "position/time", group, "time",
                                  H5P_DEFAULT, H5P_DEFAULT) >= 0;
    };
    {
      Handle box(H5Gcreate2(all, "box", H5P_DEFAULT, H5P_DEFAULT,
                            H5P_DEFAULT),
                 H5Gclose);
      int dimension = 3;
      ok &= writeAttribute(box, "dimension", H5T_STD_I32LE, H5T_NATIVE_INT,
                           &dimension);
      Handle text(makeTextType(9), H5Tclose);
      char boundary[3][9] = {"periodic", "periodic", "periodic"};
      if (!periodic)
        for (auto &name : boundary)
          std::strcpy(name, "none");
      ok &= writeAttribute(box, "boundary", text, text, boundary, 3);
      // Without a periodic cell there are no edges: the cell that the run
      // places around the particles is not one of the system (D142).
      if (periodic) {
        Handle edges(H5Gcreate2(box, "edges", H5P_DEFAULT, H5P_DEFAULT,
                                H5P_DEFAULT),
                     H5Gclose);
        link(edges);
        h->edges = options.triclinic
                       ? createSeries(edges, "value", H5T_IEEE_F64LE,
                                      {0, 3, 3}, 1024)
                       : createSeries(edges, "value", H5T_IEEE_F64LE, {0, 3},
                                      1024);
        ok = ok && h->edges >= 0 && writeText(h->edges, "unit", "nm");
      }
    }
    if (options.velocities) {
      Handle velocity(H5Gcreate2(all, "velocity", H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT),
                      H5Gclose);
      // Leapfrog's velocities are of another time than the positions.
      bool shifted = options.velocityOffset != 0.0;
      link(velocity, !shifted);
      if (shifted) {
        h->velocityTime =
            createSeries(velocity, "time", H5T_IEEE_F64LE, {0}, 1024);
        ok = ok && h->velocityTime >= 0 &&
             writeText(h->velocityTime, "unit", "ps");
      }
      h->velocity = createSeries(velocity, "value", real, {0, count, 3}, rows);
      ok = ok && h->velocity >= 0 && writeText(h->velocity, "unit", "nm ps-1");
    }
    if (options.forces) {
      Handle force(H5Gcreate2(all, "force", H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT),
                   H5Gclose);
      link(force);
      h->force = createSeries(force, "value", real, {0, count, 3}, rows);
      ok = ok && h->force >= 0 &&
           writeText(h->force, "unit", "kJ mol-1 nm-1");
    }
    auto fixed = [&](const char *name, hid_t fileType, hid_t memoryType,
                     const void *data, const char *unit) {
      hsize_t extent = count;
      Handle space(H5Screate_simple(1, &extent, nullptr), H5Sclose);
      Handle dataset(H5Dcreate2(all, name, fileType, space, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT),
                     H5Dclose);
      ok = ok && dataset.isValid() &&
           H5Dwrite(dataset, memoryType, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    data) >= 0;
      if (unit)
        ok = ok && writeText(dataset, "unit", unit);
    };
    // The atomic mass unit is a gram per mole.
    if (!options.masses.empty() && count > 0)
      fixed("mass", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, options.masses.data(),
            "g mol-1");
    if (options.species.size() == count && count > 0)
      fixed("species", H5T_STD_I32LE, H5T_NATIVE_INT32,
            options.species.data(), nullptr);

    if (options.energy || options.tunables) {
      Handle observables(H5Gcreate2(h->file, "observables", H5P_DEFAULT,
                                    H5P_DEFAULT, H5P_DEFAULT),
                         H5Gclose);
      if (options.energy) {
        Handle group(H5Gcreate2(observables, "potential_energy", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT),
                     H5Gclose);
        link(group);
        h->energy = createSeries(group, "value", H5T_IEEE_F64LE, {0}, 1024);
        ok = ok && h->energy >= 0 && writeText(h->energy, "unit", "kJ mol-1");
      }
      if (options.tunables) {
        Handle group(H5Gcreate2(observables, "tunables_version", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT),
                     H5Gclose);
        link(group);
        h->version = createSeries(group, "value", H5T_STD_I64LE, {0}, 1024);
        ok = ok && h->version >= 0;
      }
    }
  }
  {
    Handle parameters(H5Gcreate2(h->file, "parameters", H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT),
                      H5Gclose);
    Handle mdir(H5Gcreate2(parameters, "mdir", H5P_DEFAULT, H5P_DEFAULT,
                           H5P_DEFAULT),
                H5Gclose);
    int format = trajectoryFormat;
    ok &= writeAttribute(mdir, "trajectory_format", H5T_STD_I32LE,
                         H5T_NATIVE_INT, &format);
    ok &= writeAttribute(mdir, "period", H5T_STD_I64LE, H5T_NATIVE_INT64,
                         &period);
    ok &= writeAttribute(mdir, "timestep", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                         &timestep);
    ok &= writeAttribute(mdir, "velocity_offset", H5T_IEEE_F64LE,
                         H5T_NATIVE_DOUBLE, &options.velocityOffset);
    if (!options.precision.empty())
      ok &= writeText(mdir, "precision", options.precision);
    if (!options.frontEnd.empty())
      ok &= writeText(mdir, "front_end", options.frontEnd);
  }
  ok = ok && H5Fflush(h->file, H5F_SCOPE_GLOBAL) >= 0;
  if (!ok)
    return makeError("cannot write '" + path + "'");
  h5 = std::move(h);
  return llvm::Error::success();
}

llvm::Expected<int64_t> H5MDWriter::append(const std::string &path,
                                           size_t count, int64_t frames,
                                           int64_t framePeriod, double dt,
                                           const double[3]) {
  std::lock_guard<std::mutex> lock(getLibraryMutex());
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  numParticles = count;
  period = framePeriod;
  timestep = dt;
  failure.clear();
  auto refuse = [&](const llvm::Twine &what) {
    return makeError("'" + path + "' " + what + "; it cannot be continued");
  };
  auto h = std::make_unique<File>();
  h->file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (h->file < 0)
    return refuse("cannot be opened as an HDF5 file for writing");
  int format = 0;
  {
    Handle mdir(H5Gopen2(h->file, "/parameters/mdir", H5P_DEFAULT), H5Gclose);
    Handle attribute(mdir.isValid() && H5Aexists(mdir, "trajectory_format") > 0
                         ? H5Aopen(mdir, "trajectory_format", H5P_DEFAULT)
                         : -1,
                     H5Aclose);
    if (!attribute.isValid() ||
        H5Aread(attribute, H5T_NATIVE_INT, &format) < 0)
      return refuse("is not a trajectory in H5MD that MDIR wrote");
    if (format > trajectoryFormat)
      return refuse("is a trajectory of format " + llvm::Twine(format) +
                    ", written by a newer MDIR");
    int64_t stored = 0;
    Handle periodAttribute(H5Aexists(mdir, "period") > 0
                               ? H5Aopen(mdir, "period", H5P_DEFAULT)
                               : -1,
                           H5Aclose);
    if (!periodAttribute.isValid() ||
        H5Aread(periodAttribute, H5T_NATIVE_INT64, &stored) < 0)
      return refuse("has no period of its frames");
    if (stored != framePeriod)
      return refuse("has a frame every " + llvm::Twine(stored) +
                    " steps, and the run writes one every " +
                    llvm::Twine(framePeriod));
  }
  auto openSeries = [&](const char *name) -> hid_t {
    return exists(h->file, name) ? H5Dopen2(h->file, name, H5P_DEFAULT) : -1;
  };
  h->position = openSeries("/particles/all/position/value");
  h->step = openSeries("/particles/all/position/step");
  h->time = openSeries("/particles/all/position/time");
  if (h->position < 0 || h->step < 0 || h->time < 0)
    return refuse("has no positions with their steps and times");
  std::vector<hsize_t> shape = getShape(h->position);
  if (shape.size() != 3 || shape[2] != 3)
    return refuse("has positions of another shape");
  if (shape[1] != count)
    return refuse("holds " + llvm::Twine(shape[1]) +
                  " particles, and the run has " + llvm::Twine(count));
  {
    Handle type(H5Dget_type(h->position), H5Tclose);
    bool single = H5Tget_size(type) == 4;
    if (single != options.single)
      return refuse(llvm::Twine("holds its frames in ") +
                    (single ? "f32" : "f64") + ", and the run writes " +
                    (options.single ? "f32" : "f64"));
  }
  // What the file holds must be what the run writes.
  struct Element {
    const char *path, *name;
    bool asked;
    hid_t *dataset;
  } elements[] = {
      {"/particles/all/velocity/value", "velocities", options.velocities,
       &h->velocity},
      {"/particles/all/force/value", "forces", options.forces, &h->force},
      {"/particles/all/box/edges/value", "a periodic cell", periodic,
       &h->edges},
      {"/observables/potential_energy/value", "the potential energy",
       options.energy, &h->energy},
      {"/observables/tunables_version/value", "the version of the tunables",
       options.tunables, &h->version},
  };
  for (const Element &element : elements) {
    bool has = existsPath(h->file, element.path);
    if (has != element.asked)
      return refuse(llvm::Twine(has ? "holds " : "does not hold ") +
                    element.name + ", and the run " +
                    (element.asked ? "writes them" : "does not write them"));
    if (has) {
      *element.dataset = H5Dopen2(h->file, element.path, H5P_DEFAULT);
      if (*element.dataset < 0)
        return refuse(llvm::Twine("has ") + element.name +
                      " that cannot be opened");
    }
  }
  if (h->edges >= 0 &&
      (getShape(h->edges).size() == 3) != options.triclinic)
    return refuse(llvm::Twine("holds ") +
                  (options.triclinic ? "an orthorhombic" : "a triclinic") +
                  " cell, and the run has " +
                  (options.triclinic ? "a triclinic one"
                                     : "an orthorhombic one"));
  if (options.velocities) {
    // A time of the velocities of its own is another object than that of
    // the positions.
    H5O_info2_t a, b;
    if (H5Oget_info_by_name3(h->file, "/particles/all/velocity/time", &a,
                             H5O_INFO_BASIC, H5P_DEFAULT) < 0 ||
        H5Oget_info_by_name3(h->file, "/particles/all/position/time", &b,
                             H5O_INFO_BASIC, H5P_DEFAULT) < 0)
      return refuse("has velocities without their times");
    int same = 0;
    H5Otoken_cmp(h->file, &a.token, &b.token, &same);
    bool shifted = same != 0;
    if (shifted != (options.velocityOffset != 0.0))
      return refuse("holds velocities of another time relative to the "
                    "positions than the run writes");
    if (shifted)
      h->velocityTime =
          H5Dopen2(h->file, "/particles/all/velocity/time", H5P_DEFAULT);
  }

  // The frames that every dataset holds: a run that was killed inside a
  // frame may have left some longer.
  hsize_t held = shape[0];
  for (hid_t dataset : h->getSeries())
    held = std::min(held, getShape(dataset)[0]);
  if (frames < 0 || held < static_cast<hsize_t>(frames))
    return makeError("'" + path + "' holds " + llvm::Twine(held) +
                     " frames, fewer than the " + llvm::Twine(frames) +
                     " that the checkpoint counts");
  for (hid_t dataset : h->getSeries()) {
    std::vector<hsize_t> extent = getShape(dataset);
    if (extent[0] == static_cast<hsize_t>(frames))
      continue;
    extent[0] = frames;
    if (H5Dset_extent(dataset, extent.data()) < 0)
      return refuse("cannot be cut to the frames of the checkpoint");
  }
  lastStep.reset();
  if (frames > 0) {
    int64_t last = 0;
    if (!readRow(h->step, H5T_NATIVE_INT64, frames - 1, &last))
      return refuse("has steps that cannot be read");
    lastStep = last;
  }
  if (H5Fflush(h->file, H5F_SCOPE_GLOBAL) < 0)
    return refuse("cannot be written");
  numFrames = static_cast<int32_t>(frames);
  h5 = std::move(h);
  return static_cast<int64_t>(held) - frames;
}

bool H5MDWriter::writeState(const double *positions, const double *velocities,
                            const double *forces, int64_t step, double time,
                            const H5MDExtras &extras) {
  if (!failure.empty())
    return false;
  std::lock_guard<std::mutex> lock(getLibraryMutex());
  if (!h5) {
    failure = "the trajectory in H5MD is not open";
    return false;
  }
  if ((options.velocities && !velocities) || (options.forces && !forces)) {
    failure = "a frame of the trajectory in H5MD lacks the velocities or "
              "the forces that the file holds; this is a defect of mdir";
    return false;
  }
  File &h = *h5;
  hsize_t row = numFrames;
  bool ok = true;
  // The values of the frame first and its step last, then everything to
  // the operating system: a reader counts the frames by the steps.
  std::vector<float> narrow;
  auto vectors = [&](hid_t dataset, const double *values) {
    if (!options.single) {
      ok = ok && writeRow(dataset, H5T_NATIVE_DOUBLE, row, values);
      return;
    }
    narrow.assign(values, values + 3 * numParticles);
    ok = ok && writeRow(dataset, H5T_NATIVE_FLOAT, row, narrow.data());
  };
  vectors(h.position, positions);
  if (h.velocity >= 0)
    vectors(h.velocity, velocities);
  if (h.force >= 0)
    vectors(h.force, forces);
  if (h.edges >= 0) {
    if (options.triclinic) {
      double matrix[9] = {exactBox[0],  0.0,          0.0,
                          exactTilt[0], exactBox[1],  0.0,
                          exactTilt[1], exactTilt[2], exactBox[2]};
      ok = ok && writeRow(h.edges, H5T_NATIVE_DOUBLE, row, matrix);
    } else {
      ok = ok && writeRow(h.edges, H5T_NATIVE_DOUBLE, row, exactBox);
    }
  }
  if (h.energy >= 0)
    ok = ok && writeRow(h.energy, H5T_NATIVE_DOUBLE, row, &extras.potential);
  if (h.version >= 0)
    ok = ok &&
         writeRow(h.version, H5T_NATIVE_INT64, row, &extras.tunablesVersion);
  if (h.velocityTime >= 0) {
    double shifted = time + options.velocityOffset * timestep;
    ok = ok && writeRow(h.velocityTime, H5T_NATIVE_DOUBLE, row, &shifted);
  }
  ok = ok && writeRow(h.time, H5T_NATIVE_DOUBLE, row, &time);
  ok = ok && writeRow(h.step, H5T_NATIVE_INT64, row, &step);
  ok = ok && H5Fflush(h.file, H5F_SCOPE_GLOBAL) >= 0;
  if (!ok) {
    failure = "the frame of step " + std::to_string(step) +
              " could not be written to the trajectory in H5MD (a full "
              "disk?)";
    return false;
  }
  ++numFrames;
  lastStep = step;
  return true;
}

//===----------------------------------------------------------------------===//
// The reader
//===----------------------------------------------------------------------===//

struct H5MDReader::File {
  hid_t file = -1, position = -1, velocity = -1, force = -1, edges = -1,
        energy = -1, version = -1;
  /// Whether the edges have a row per frame, and whether they are a
  /// matrix.
  bool movingCell = false, matrix = false;
  ~File() {
    for (hid_t dataset : {position, velocity, force, edges, energy, version})
      if (dataset >= 0)
        H5Dclose(dataset);
    if (file >= 0)
      H5Fclose(file);
  }
};

H5MDReader::H5MDReader() = default;

H5MDReader::~H5MDReader() { close(); }

void H5MDReader::close() {
  // Nothing to close, and the mutex may be held: a reader that `open`
  // refuses is destroyed inside it.
  if (!h5)
    return;
  std::lock_guard<std::mutex> lock(getLibraryMutex());
  h5.reset();
}

llvm::Expected<std::unique_ptr<H5MDReader>>
H5MDReader::open(const std::string &path, const std::string &asked) {
  std::lock_guard<std::mutex> lock(getLibraryMutex());
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  auto refuse = [&](const llvm::Twine &what) {
    return makeError("'" + path + "' " + what);
  };
  std::unique_ptr<H5MDReader> reader(new H5MDReader());
  reader->path = path;
  auto h = std::make_unique<File>();
  h->file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (h->file < 0)
    return refuse("cannot be read as an HDF5 file");
  if (!exists(h->file, "h5md"))
    return refuse("is not an H5MD file: it has no group 'h5md'");
  if (exists(h->file, "/h5md/creator")) {
    Handle creator(H5Gopen2(h->file, "/h5md/creator", H5P_DEFAULT), H5Gclose);
    reader->creator = readText(creator, "name");
    std::string version = readText(creator, "version");
    if (!version.empty())
      reader->creator += " " + version;
  }
  if (!exists(h->file, "particles"))
    return refuse("has no group 'particles'");
  std::string group = asked;
  if (group.empty()) {
    Handle particles(H5Gopen2(h->file, "/particles", H5P_DEFAULT), H5Gclose);
    H5G_info_t info;
    if (H5Gget_info(particles, &info) < 0 || info.nlinks == 0)
      return refuse("has no group of particles");
    if (exists(particles, "all")) {
      group = "all";
    } else if (info.nlinks == 1) {
      char name[256];
      if (H5Lget_name_by_idx(particles, ".", H5_INDEX_NAME, H5_ITER_INC, 0,
                             name, sizeof name, H5P_DEFAULT) < 0)
        return refuse("has a group of particles whose name cannot be read");
      group = name;
    } else {
      return refuse("has " + llvm::Twine(info.nlinks) +
                    " groups of particles and none named 'all'; name one "
                    "with group=");
    }
  }
  reader->group = group;
  std::string base = "/particles/" + group;
  if (H5Lexists(h->file, base.c_str(), H5P_DEFAULT) <= 0)
    return refuse("has no group '" + base + "'");
  Handle particles(H5Gopen2(h->file, base.c_str(), H5P_DEFAULT), H5Gclose);
  if (!particles.isValid() || !exists(particles, "position") ||
      H5Lexists(particles, "position/value", H5P_DEFAULT) <= 0)
    return refuse("has no positions that change with time in '" + base +
                  "'");
  h->position = H5Dopen2(particles, "position/value", H5P_DEFAULT);
  std::vector<hsize_t> shape =
      h->position >= 0 ? getShape(h->position) : std::vector<hsize_t>();
  if (shape.size() != 3 || shape[2] != 3)
    return refuse("has positions that are not [frames][particles][3]");
  {
    Handle type(H5Dget_type(h->position), H5Tclose);
    if (H5Tget_class(type) != H5T_FLOAT)
      return refuse("has positions that are not floating-point numbers");
    reader->width = H5Tget_size(type) == 4 ? 4 : 8;
  }
  reader->numParticles = shape[1];
  hsize_t frames = shape[0];
  auto unit = [&](hid_t dataset, const char *name) {
    std::string text = readText(dataset, "unit");
    if (!text.empty())
      reader->units[name] = text;
  };
  unit(h->position, "position");

  // The steps and the times: a value for each frame, or the increment and
  // the offset of fixed storage.
  auto series = [&](const char *name, bool integer, std::vector<int64_t> &whole,
                    std::vector<double> &real) -> bool {
    Handle dataset(H5Dopen2(particles, name, H5P_DEFAULT), H5Dclose);
    if (!dataset.isValid())
      return false;
    if (!integer)
      unit(dataset, "time");
    Handle space(H5Dget_space(dataset), H5Sclose);
    if (H5Sget_simple_extent_type(space) == H5S_SCALAR) {
      double increment = 0.0, offset = 0.0;
      if (H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  &increment) < 0)
        return false;
      if (H5Aexists(dataset, "offset") > 0) {
        Handle attribute(H5Aopen(dataset, "offset", H5P_DEFAULT), H5Aclose);
        H5Aread(attribute, H5T_NATIVE_DOUBLE, &offset);
      }
      for (hsize_t i = 0; i != frames; ++i) {
        if (integer)
          whole.push_back(static_cast<int64_t>(offset + i * increment));
        else
          real.push_back(offset + i * increment);
      }
      return true;
    }
    std::vector<hsize_t> extent = getShape(dataset);
    if (extent.size() != 1)
      return false;
    // A run that was killed inside a frame may have left values without
    // their step: the frames are those that have one.
    frames = std::min(frames, extent[0]);
    if (integer) {
      whole.resize(extent[0]);
      if (extent[0] > 0 &&
          H5Dread(dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  whole.data()) < 0)
        return false;
    } else {
      real.resize(extent[0]);
      if (extent[0] > 0 &&
          H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  real.data()) < 0)
        return false;
    }
    return true;
  };
  std::vector<double> none;
  std::vector<int64_t> noSteps;
  if (H5Lexists(particles, "position/step", H5P_DEFAULT) <= 0 ||
      !series("position/step", true, reader->steps, none))
    return refuse("has positions without readable steps");
  if (H5Lexists(particles, "position/time", H5P_DEFAULT) > 0 &&
      !series("position/time", false, noSteps, reader->times))
    return refuse("has times of the positions that cannot be read");

  auto vectors = [&](const char *name, const char *label, hid_t &dataset) {
    std::string value = std::string(name) + "/value";
    if (!exists(particles, name) ||
        H5Lexists(particles, value.c_str(), H5P_DEFAULT) <= 0)
      return false;
    hid_t opened = H5Dopen2(particles, value.c_str(), H5P_DEFAULT);
    if (opened < 0)
      return false;
    std::vector<hsize_t> extent = getShape(opened);
    Handle type(H5Dget_type(opened), H5Tclose);
    if (extent.size() != 3 || extent[1] != reader->numParticles ||
        extent[2] != 3 || H5Tget_class(type) != H5T_FLOAT) {
      H5Dclose(opened);
      return false;
    }
    frames = std::min(frames, extent[0]);
    dataset = opened;
    unit(opened, label);
    return true;
  };
  reader->velocities = vectors("velocity", "velocity", h->velocity);
  reader->forces = vectors("force", "force", h->force);

  // The cell: none where no boundary is periodic or the edges are absent.
  bool open = false;
  if (exists(particles, "box")) {
    Handle box(H5Gopen2(particles, "box", H5P_DEFAULT), H5Gclose);
    open = box.isValid() && hasNoBoundary(box);
  }
  if (!open && exists(particles, "box") &&
      H5Lexists(particles, "box/edges", H5P_DEFAULT) > 0) {
    H5O_info2_t info;
    if (H5Oget_info_by_name3(particles, "box/edges", &info, H5O_INFO_BASIC,
                             H5P_DEFAULT) < 0)
      return refuse("has a cell that cannot be read");
    if (info.type == H5O_TYPE_GROUP) {
      if (H5Lexists(particles, "box/edges/value", H5P_DEFAULT) <= 0)
        return refuse("has a cell without values");
      h->edges = H5Dopen2(particles, "box/edges/value", H5P_DEFAULT);
      h->movingCell = true;
    } else {
      h->edges = H5Dopen2(particles, "box/edges", H5P_DEFAULT);
    }
    std::vector<hsize_t> extent =
        h->edges >= 0 ? getShape(h->edges) : std::vector<hsize_t>();
    size_t rank = extent.size() - (h->movingCell ? 1 : 0);
    if (h->edges < 0 || extent.empty() || (rank != 1 && rank != 2) ||
        extent.back() != 3 || (rank == 2 && extent[extent.size() - 2] != 3))
      return refuse("has a cell that is neither [3] nor [3][3]");
    h->matrix = rank == 2;
    if (h->movingCell)
      frames = std::min(frames, extent[0]);
    unit(h->edges, "box");
  }
  auto observable = [&](const char *name, hid_t &dataset) {
    std::string value = std::string("/observables/") + name + "/value";
    std::string parent = std::string("/observables/") + name;
    if (!exists(h->file, "observables") ||
        H5Lexists(h->file, parent.c_str(), H5P_DEFAULT) <= 0 ||
        H5Lexists(h->file, value.c_str(), H5P_DEFAULT) <= 0)
      return false;
    hid_t opened = H5Dopen2(h->file, value.c_str(), H5P_DEFAULT);
    if (opened < 0)
      return false;
    std::vector<hsize_t> extent = getShape(opened);
    if (extent.size() != 1) {
      H5Dclose(opened);
      return false;
    }
    frames = std::min(frames, extent[0]);
    dataset = opened;
    return true;
  };
  reader->potential = observable("potential_energy", h->energy);
  if (reader->potential)
    unit(h->energy, "potential_energy");
  reader->tunables = observable("tunables_version", h->version);

  reader->steps.resize(std::min<size_t>(reader->steps.size(), frames));
  if (!reader->times.empty())
    reader->times.resize(std::min<size_t>(reader->times.size(), frames));
  if (!reader->times.empty() && reader->times.size() != reader->steps.size())
    return refuse("has another number of times than of steps");
  reader->h5 = std::move(h);
  return std::move(reader);
}

llvm::Expected<H5MDFrame> H5MDReader::read(size_t index) {
  std::lock_guard<std::mutex> lock(getLibraryMutex());
  if (!h5)
    return makeError("'" + path + "' is closed");
  if (index >= steps.size())
    return makeError("'" + path + "' has no frame " + llvm::Twine(index) +
                     ": it holds " + llvm::Twine(steps.size()));
  auto fail = [&](const char *what) {
    return makeError("the " + llvm::Twine(what) + " of frame " +
                     llvm::Twine(index) + " of '" + path +
                     "' cannot be read");
  };
  File &h = *h5;
  H5MDFrame frame;
  frame.step = steps[index];
  frame.time = times.empty() ? std::numeric_limits<double>::quiet_NaN()
                             : times[index];
  frame.width = width;
  hid_t type = width == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
  size_t bytes = 3 * numParticles * width;
  frame.positions.resize(bytes);
  if (bytes && !readRow(h.position, type, index, frame.positions.data()))
    return fail("positions");
  if (h.velocity >= 0) {
    frame.velocities.resize(bytes);
    if (bytes && !readRow(h.velocity, type, index, frame.velocities.data()))
      return fail("velocities");
  }
  if (h.force >= 0) {
    frame.forces.resize(bytes);
    if (bytes && !readRow(h.force, type, index, frame.forces.data()))
      return fail("forces");
  }
  if (h.edges >= 0) {
    double values[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool ok = h.movingCell
                  ? readRow(h.edges, H5T_NATIVE_DOUBLE, index, values)
                  : H5Dread(h.edges, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, values) >= 0;
    if (!ok)
      return fail("cell");
    frame.hasCell = true;
    frame.orthorhombic = !h.matrix;
    if (h.matrix) {
      std::memcpy(frame.cell, values, sizeof values);
    } else {
      frame.cell[0] = values[0];
      frame.cell[4] = values[1];
      frame.cell[8] = values[2];
    }
  }
  if (h.energy >= 0) {
    double value = 0.0;
    if (!readRow(h.energy, H5T_NATIVE_DOUBLE, index, &value))
      return fail("potential energy");
    frame.potential = value;
  }
  if (h.version >= 0) {
    int64_t value = 0;
    if (!readRow(h.version, H5T_NATIVE_INT64, index, &value))
      return fail("version of the tunables");
    frame.tunablesVersion = value;
  }
  return frame;
}

#endif // MDIR_HAS_HDF5

// The handlers of the XLA FFI for the JAX adapter of the frame evaluator
// (D371, docs/python-frames.md, `mdir.jax`).
//
// `mdir_frames` evaluates the K frames of a call in the simulation of a
// frame evaluator, one frame after the other on its existing loop
// (`Simulation::evaluateFrame`, D251), and writes the energies, the
// gradients in the positions and in the cell, and the rows of the
// Jacobian in the tunables into the buffers that XLA gives it. The
// handler is typed FFI (custom call API version 4) with two stages:
// `instantiate`, once per executable, resolves the key of the program and
// keeps the simulation in the state of the executable; `execute` runs the
// frames. It never touches Python: XLA calls it from its own threads.
#include "XlaFfi.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <pybind11/numpy.h>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string_view>
#include <vector>

namespace ffi = ::xla::ffi;
namespace py = pybind11;

namespace mdir::python::xla {
namespace {

/// A program that the handlers can evaluate.
struct Entry {
  std::shared_ptr<compiler::Simulation> simulation;
  std::string fingerprint;
  size_t particles = 0, entries = 0;
  bool periodic = false, forcesDouble = true;
  std::vector<std::string> names;
  std::vector<size_t> sizes;
  /// The values of the tunables that the simulation is at, flattened in
  /// the order of the plan: those of the program at first.
  std::vector<double> theta;
  /// Calls of several executables on one program are serialized here: the
  /// simulation takes one operation at a time.
  std::mutex mutex;
};

std::mutex registryMutex;
std::map<int64_t, std::shared_ptr<Entry>> registry;
int64_t nextKey = 1;

/// The state of an executable: the program that its calls evaluate.
struct FrameState {
  static ffi::TypeId id;
  std::shared_ptr<Entry> entry;
};
ffi::TypeId FrameState::id = {};
XLA_FFI_TypeInfo frameStateInfo = ffi::MakeTypeInfo<FrameState>();

/// The text of a program that its key is checked against: everything that
/// defined its model and its execution (the fingerprint of a checkpoint,
/// D172, D223), the declarations of its tunables, and the shapes and types
/// of the buffers of a call. A key of another process, from a persistent
/// cache of compilations, names another program or none.
std::string fingerprintOf(const compiler::Simulation &simulation,
                          const model::PreparedModel &prepared) {
  std::string text;
  for (const driver::FingerprintEntry &entry : prepared.fingerprint)
    text += entry.group + "\t" + entry.name + "\t" + entry.value + "\n";
  text += model::describeTunables(prepared.tunables);
  text += "particles=" + std::to_string(simulation.getParticleCount()) +
          "\nforce=" + (simulation.keepsForcesInDouble() ? "float64" : "float32") +
          "\nperiodic=" + (prepared.control.periodic ? "true" : "false") + "\n";
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),
                     /*LowerCase=*/true);
}

ffi::Error invalid(std::string message) {
  return ffi::Error::InvalidArgument("mdir_frames: " + std::move(message));
}

/// An error of the data of a call (a frame that is refused or whose
/// evaluation fails, values of the tunables that are refused): a failed
/// precondition. JAX raises it as `JaxRuntimeError` where the result is
/// awaited, and as `ValueError` where the call runs as it is made.
ffi::Error failed(std::string message) {
  return ffi::Error(ffi::ErrorCode::kFailedPrecondition, "mdir_frames: " + std::move(message));
}

ffi::ErrorOr<std::unique_ptr<FrameState>> instantiate(int64_t program,
                                                      std::string_view fingerprint) {
  std::shared_ptr<Entry> entry;
  {
    std::lock_guard<std::mutex> lock(registryMutex);
    auto found = registry.find(program);
    if (found != registry.end())
      entry = found->second;
  }
  if (!entry)
    return ffi::Unexpected(invalid(
        "the program of this call (key " + std::to_string(program) +
        ") was released, or is not one of this process: make the energy "
        "function again with mdir.jax.energy_fn"));
  if (entry->fingerprint != fingerprint)
    return ffi::Unexpected(invalid(
        "the key " + std::to_string(program) + " names another program in "
        "this process than the one this call was compiled for (a "
        "compilation of another process, from a persistent cache?)"));
  auto state = std::make_unique<FrameState>();
  state->entry = std::move(entry);
  return state;
}

/// The gradient of the energy in the cell vectors h (in rows, lower
/// triangular) at fixed Cartesian positions, from G = dU/dx and
/// S = dU/d(epsilon) under x -> (1 + epsilon) x, h -> h (1 + epsilon)^T:
/// S = sum_i G_i (x) x_i + (dU/dh)^T h gives
/// dU/dh = h^-T (S - sum_i G_i (x) x_i)^T (D252), the nine entries row by
/// row. The sum over the particles is taken in f64 in the order of the
/// input; h^T is upper triangular, so the solve is a back substitution.
/// Both adapters form it here (`_frame_cell_gradient` for `mdir.torch`), so
/// that they give the same bits.
template <class Force>
void cellGradient(const double *x, const Force *g, size_t particles,
                  const std::array<double, 9> &strain, const std::array<double, 6> &cell,
                  double *out) {
  double moment[3][3] = {};
  for (size_t i = 0; i != particles; ++i)
    for (int a = 0; a != 3; ++a)
      for (int b = 0; b != 3; ++b)
        moment[a][b] += static_cast<double>(g[3 * i + a]) * x[3 * i + b];
  // B = (S - M)^T; solve h^T D = B, with h^T = [[ax, bx, cx], [0, by, cy],
  // [0, 0, cz]].
  double rhs[3][3];
  for (int a = 0; a != 3; ++a)
    for (int b = 0; b != 3; ++b)
      rhs[a][b] = strain[3 * b + a] - moment[b][a];
  const double ax = cell[0], by = cell[1], cz = cell[2], bx = cell[3], cx = cell[4],
               cy = cell[5];
  for (int b = 0; b != 3; ++b) {
    double d2 = rhs[2][b] / cz;
    double d1 = (rhs[1][b] - cy * d2) / by;
    double d0 = (rhs[0][b] - bx * d1 - cx * d2) / ax;
    out[b] = d0;
    out[3 + b] = d1;
    out[6 + b] = d2;
  }
}

ffi::Error execute(FrameState *state, ffi::Buffer<ffi::F64> theta,
                   ffi::Buffer<ffi::F64> positions, ffi::Buffer<ffi::F64> cells,
                   ffi::ResultBuffer<ffi::F64> energies, ffi::Result<ffi::AnyBuffer> gradient,
                   ffi::ResultBuffer<ffi::F64> cellGradients,
                   ffi::ResultBuffer<ffi::F64> rows) {
  if (!state || !state->entry)
    return invalid("the call has no program; this is a defect of mdir");
  Entry &entry = *state->entry;
  auto dims = positions.dimensions();
  if (dims.size() != 3 || static_cast<size_t>(dims[1]) != entry.particles || dims[2] != 3)
    return invalid("expected positions of shape (K, " + std::to_string(entry.particles) +
                   ", 3)");
  const size_t frames = dims[0], n = entry.particles;
  if (theta.element_count() != entry.entries)
    return invalid("expected " + std::to_string(entry.entries) + " values of the tunables");
  auto cellDims = cells.dimensions();
  if (cellDims.size() != 2 || static_cast<size_t>(cellDims[0]) != frames ||
      cellDims[1] != (entry.periodic ? 6 : 0))
    return invalid("expected cells of shape (K, " + std::string(entry.periodic ? "6" : "0") +
                   ")");
  // What is asked is what has rows: an output of shape (0, ...) is not.
  auto asked = [&](auto dimensions) {
    return dimensions.size() != 0 && static_cast<size_t>(dimensions[0]) == frames &&
           frames != 0;
  };
  bool wantsForces = asked(gradient->dimensions());
  bool wantsCell = asked(cellGradients->dimensions());
  bool wantsRows = asked(rows->dimensions());
  if (energies->element_count() != frames)
    return invalid("expected energies of shape (K,)");
  ffi::DataType forceType = entry.forcesDouble ? ffi::DataType::F64 : ffi::DataType::F32;
  if (wantsForces && gradient->element_type() != forceType)
    return invalid("the gradient in the positions is in the type of the forces of the "
                   "program");
  if (wantsRows && rows->element_count() != frames * entry.entries)
    return invalid("expected rows of shape (K, M)");
  if (wantsCell && !entry.periodic)
    return invalid("the program has no periodic cell");

  std::lock_guard<std::mutex> lock(entry.mutex);
  compiler::Simulation &simulation = *entry.simulation;
  // The values of the call: the tunables whose bits differ are updated
  // (D213), which ends the activation (D251).
  const double *values = theta.typed_data();
  if (std::memcmp(values, entry.theta.data(), entry.entries * sizeof(double)) != 0) {
    std::vector<std::pair<std::string, std::vector<double>>> changes;
    size_t at = 0;
    for (size_t k = 0; k != entry.names.size(); ++k) {
      size_t size = entry.sizes[k];
      if (std::memcmp(values + at, entry.theta.data() + at, size * sizeof(double)) != 0)
        changes.emplace_back(entry.names[k], std::vector<double>(values + at, values + at + size));
      at += size;
    }
    if (llvm::Error error = simulation.updateTunables(changes))
      return failed("the values of the tunables: " + llvm::toString(std::move(error)));
    std::memcpy(entry.theta.data(), values, entry.entries * sizeof(double));
  }
  const double *x = positions.typed_data();
  const double *c = cells.typed_data();
  size_t width = entry.forcesDouble ? sizeof(double) : sizeof(float);
  // The gradient in the positions where the cell asks for it and the
  // positions do not.
  std::vector<char> scratch(wantsCell && !wantsForces ? 3 * n * width : 0);
  char *g = wantsForces ? static_cast<char *>(gradient->untyped_data()) : nullptr;
  double *out = energies->typed_data();
  for (size_t k = 0; k != frames; ++k) {
    std::optional<std::array<double, 6>> cell;
    if (entry.periodic)
      cell = std::array<double, 6>{c[6 * k], c[6 * k + 1], c[6 * k + 2],
                                   c[6 * k + 3], c[6 * k + 4], c[6 * k + 5]};
    char *frameGradient = g ? g + 3 * n * width * k : scratch.empty() ? nullptr : scratch.data();
    auto result = simulation.evaluateFrame(x + 3 * n * k, cell, wantsRows, frameGradient,
                                           wantsCell);
    if (!result)
      return failed("frame " + std::to_string(k) + " of the call: " +
                     llvm::toString(result.takeError()));
    out[k] = result->energy;
    if (wantsRows) {
      double *row = rows->typed_data() + k * entry.entries;
      for (const auto &part : result->gradient) {
        std::copy(part.begin(), part.end(), row);
        row += part.size();
      }
    }
    if (wantsCell) {
      double *target = cellGradients->typed_data() + 9 * k;
      if (entry.forcesDouble)
        cellGradient(x + 3 * n * k, reinterpret_cast<const double *>(frameGradient), n,
                     result->strain, *cell, target);
      else
        cellGradient(x + 3 * n * k, reinterpret_cast<const float *>(frameGradient), n,
                     result->strain, *cell, target);
    }
  }
  return ffi::Error::Success();
}

/// Refuses an XLA whose FFI is not the one of the headers that MDIR was
/// built with (`third_party/xla_ffi`): the same major version and at
/// least the minor one.
XLA_FFI_Error *checkVersion(XLA_FFI_CallFrame *frame) {
  const XLA_FFI_Api *api = frame->api;
  if (!api)
    return nullptr;
  int major = api->api_version.major_version, minor = api->api_version.minor_version;
  if (major == XLA_FFI_API_MAJOR && minor >= XLA_FFI_API_MINOR)
    return nullptr;
  std::string message = "mdir_frames: the XLA FFI of this JAX is version " +
                        std::to_string(major) + "." + std::to_string(minor) +
                        "; MDIR was built with the headers of version " +
                        std::to_string(XLA_FFI_API_MAJOR) + "." +
                        std::to_string(XLA_FFI_API_MINOR) + " (third_party/xla_ffi)";
  XLA_FFI_Error_Create_Args args;
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.message = message.c_str();
  args.errc = XLA_FFI_Error_Code_FAILED_PRECONDITION;
  return api->XLA_FFI_Error_Create(&args);
}

XLA_FFI_DEFINE_HANDLER(framesInstantiate, instantiate,
                       ffi::Ffi::BindInstantiate()
                           .Attr<int64_t>("program")
                           .Attr<std::string_view>("fingerprint"));

XLA_FFI_DEFINE_HANDLER(framesExecute, execute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::State<FrameState>>()
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_Error *mdirFramesInstantiate(XLA_FFI_CallFrame *frame) {
  if (XLA_FFI_Error *error = checkVersion(frame))
    return error;
  return framesInstantiate(frame);
}

XLA_FFI_Error *mdirFramesExecute(XLA_FFI_CallFrame *frame) {
  if (XLA_FFI_Error *error = checkVersion(frame))
    return error;
  return framesExecute(frame);
}

py::capsule capsule(void *pointer) { return py::capsule(pointer); }
} // namespace

std::pair<int64_t, std::string>
registerProgram(std::shared_ptr<compiler::Simulation> simulation,
                const model::PreparedModel &prepared) {
  auto entry = std::make_shared<Entry>();
  entry->fingerprint = fingerprintOf(*simulation, prepared);
  entry->particles = simulation->getParticleCount();
  entry->forcesDouble = simulation->keepsForcesInDouble();
  entry->periodic = prepared.control.periodic;
  const model::TunableSet &set = simulation->getTunables();
  for (size_t k = 0; k != set.tunables.size(); ++k) {
    entry->names.push_back(set.tunables[k].name);
    entry->sizes.push_back(set.tunables[k].entries);
    entry->entries += set.tunables[k].entries;
    entry->theta.insert(entry->theta.end(), set.values[k].begin(), set.values[k].end());
  }
  entry->simulation = std::move(simulation);
  std::lock_guard<std::mutex> lock(registryMutex);
  int64_t key = nextKey++;
  std::string fingerprint = entry->fingerprint;
  registry.emplace(key, std::move(entry));
  return {key, fingerprint};
}

void addBindings(py::module_ &m) {
  // The handlers of the targets, by stage, as `jax.ffi.register_ffi_target`
  // takes them.
  m.def("_xla_ffi_targets", []() {
    py::dict targets, frames;
    frames["instantiate"] = capsule(reinterpret_cast<void *>(&mdirFramesInstantiate));
    frames["execute"] = capsule(reinterpret_cast<void *>(&mdirFramesExecute));
    targets["mdir_frames"] = frames;
    return targets;
  });
  // The type of the state of an executable, which is registered before the
  // handlers (`jax.ffi.register_ffi_type`).
  m.def("_xla_ffi_types", []() {
    py::dict types, state;
    state["type_id"] = capsule(&FrameState::id);
    state["type_info"] = capsule(&frameStateInfo);
    types["mdir_frames_state"] = state;
    return types;
  });
  // The gradient in the cell vectors of `cellGradient`, for the PyTorch
  // adapter, so that both adapters form it with the same arithmetic:
  // positions (K, N, 3) float64, the gradient in them (K, N, 3) in the type
  // of the forces, the strain (K, 3, 3), and the cells (K, 6) give
  // dU/dh (K, 3, 3).
  m.def("_frame_cell_gradient", [](py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                   py::array gradient,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> strain,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> cells) {
    if (x.ndim() != 3 || x.shape(2) != 3)
      throw py::value_error("_frame_cell_gradient: positions (K, N, 3)");
    py::ssize_t frames = x.shape(0), particles = x.shape(1);
    if (gradient.ndim() != 3 || gradient.shape(0) != frames || gradient.shape(1) != particles ||
        gradient.shape(2) != 3 || strain.ndim() != 3 || strain.shape(0) != frames ||
        cells.ndim() != 2 || cells.shape(0) != frames || cells.shape(1) != 6)
      throw py::value_error("_frame_cell_gradient: shapes (K, N, 3), (K, 3, 3), (K, 6)");
    bool single = gradient.dtype().is(py::dtype::of<float>());
    if (!single && !gradient.dtype().is(py::dtype::of<double>()))
      throw py::value_error("_frame_cell_gradient: the gradient is float32 or float64");
    py::array g = single ? py::array(py::array_t<float, py::array::c_style | py::array::forcecast>(gradient))
                         : py::array(py::array_t<double, py::array::c_style | py::array::forcecast>(gradient));
    py::array_t<double> result({frames, py::ssize_t(3), py::ssize_t(3)});
    const double *xs = x.data(), *s = strain.data(), *c = cells.data();
    double *out = result.mutable_data();
    for (py::ssize_t k = 0; k != frames; ++k) {
      std::array<double, 9> frameStrain;
      std::copy(s + 9 * k, s + 9 * k + 9, frameStrain.begin());
      std::array<double, 6> cell;
      std::copy(c + 6 * k, c + 6 * k + 6, cell.begin());
      size_t at = 3 * static_cast<size_t>(particles) * k;
      if (single)
        cellGradient(xs + at, static_cast<const float *>(g.data()) + at, particles, frameStrain,
                     cell, out + 9 * k);
      else
        cellGradient(xs + at, static_cast<const double *>(g.data()) + at, particles, frameStrain,
                     cell, out + 9 * k);
    }
    return result;
  });
  m.def("_xla_ffi_api_version",
        []() { return py::make_tuple(XLA_FFI_API_MAJOR, XLA_FFI_API_MINOR); });
  // Removes the program from the registry; an executable that was
  // instantiated with it keeps its own reference.
  m.def("_xla_ffi_release", [](int64_t key) {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(registryMutex);
      auto found = registry.find(key);
      if (found == registry.end())
        return;
      entry = std::move(found->second);
      registry.erase(found);
    }
    // The simulation is freed here if nothing else holds it, without the GIL.
    py::gil_scoped_release release;
    entry.reset();
  });
}

} // namespace mdir::python::xla

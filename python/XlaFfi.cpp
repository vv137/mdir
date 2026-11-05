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
//
// The handler of the platform "CUDA" (built where MDIR is built with CUDA,
// MDIR_XLA_FFI_CUDA) runs the same loop on buffers of the host: it copies
// the values of the tunables, the positions, and the cells of the call from
// XLA's buffers on the device on XLA's stream, waits for them, evaluates
// the frames, and copies the outputs back on that stream, waiting for them
// too. The work of XLA before the call and after it is therefore ordered
// with it, through the waits on the host (D[frame-evaluator-jax-cuda]). The
// boundary of a batch on the device (PR 1 of #138) replaces these copies.
#include "XlaFfi.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <pybind11/numpy.h>

#ifdef MDIR_XLA_FFI_CUDA
#include <cuda.h>
#include "llvm/Support/DynamicLibrary.h"
#endif

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
  /// Whether the program runs on the GPU (`Execution.target`), whose
  /// device a call on the platform "CUDA" must be on.
  bool onDevice = false;
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

#ifdef MDIR_XLA_FFI_CUDA
/// The same for an executable of the platform "CUDA". The plugin of CUDA
/// keeps a registry of types of its own, which writes the identifier it
/// gives a type into its `TypeId`; a type of its own here keeps that of
/// the CPU as the CPU's registry gave it.
struct CudaFrameState {
  static ffi::TypeId id;
  std::shared_ptr<Entry> entry;
};
ffi::TypeId CudaFrameState::id = {};
XLA_FFI_TypeInfo cudaFrameStateInfo = ffi::MakeTypeInfo<CudaFrameState>();
#endif

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

/// The program of the key `program`, if its fingerprint is `fingerprint`.
ffi::ErrorOr<std::shared_ptr<Entry>> resolve(int64_t program, std::string_view fingerprint) {
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
  return entry;
}

ffi::ErrorOr<std::unique_ptr<FrameState>> instantiate(int64_t program,
                                                      std::string_view fingerprint) {
  auto entry = resolve(program, fingerprint);
  if (entry.has_error())
    return ffi::Unexpected(entry.error());
  auto state = std::make_unique<FrameState>();
  state->entry = std::move(*entry);
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

/// What a call asks of its frames, from the shapes of its buffers.
struct Call {
  size_t frames = 0;
  bool wantsForces = false, wantsCell = false, wantsRows = false;
};

/// Checks the shapes and types of the buffers of a call against its
/// program. Only the metadata of the buffers is read, which is on the host
/// on every platform.
ffi::ErrorOr<Call> checkCall(const Entry &entry, const ffi::Buffer<ffi::F64> &theta,
                             const ffi::Buffer<ffi::F64> &positions,
                             const ffi::Buffer<ffi::F64> &cells,
                             const ffi::ResultBuffer<ffi::F64> &energies,
                             const ffi::Result<ffi::AnyBuffer> &gradient,
                             const ffi::ResultBuffer<ffi::F64> &cellGradients,
                             const ffi::ResultBuffer<ffi::F64> &rows) {
  auto dims = positions.dimensions();
  if (dims.size() != 3 || static_cast<size_t>(dims[1]) != entry.particles || dims[2] != 3)
    return ffi::Unexpected(invalid("expected positions of shape (K, " +
                                   std::to_string(entry.particles) + ", 3)"));
  Call call;
  call.frames = dims[0];
  const size_t frames = call.frames;
  if (theta.element_count() != entry.entries)
    return ffi::Unexpected(
        invalid("expected " + std::to_string(entry.entries) + " values of the tunables"));
  auto cellDims = cells.dimensions();
  if (cellDims.size() != 2 || static_cast<size_t>(cellDims[0]) != frames ||
      cellDims[1] != (entry.periodic ? 6 : 0))
    return ffi::Unexpected(invalid("expected cells of shape (K, " +
                                   std::string(entry.periodic ? "6" : "0") + ")"));
  // What is asked is what has rows: an output of shape (0, ...) is not.
  auto asked = [&](auto dimensions) {
    return dimensions.size() != 0 && static_cast<size_t>(dimensions[0]) == frames &&
           frames != 0;
  };
  call.wantsForces = asked(gradient->dimensions());
  call.wantsCell = asked(cellGradients->dimensions());
  call.wantsRows = asked(rows->dimensions());
  if (energies->element_count() != frames)
    return ffi::Unexpected(invalid("expected energies of shape (K,)"));
  ffi::DataType forceType = entry.forcesDouble ? ffi::DataType::F64 : ffi::DataType::F32;
  if (call.wantsForces && gradient->element_type() != forceType)
    return ffi::Unexpected(invalid("the gradient in the positions is in the type of the "
                                   "forces of the program"));
  if (call.wantsRows && rows->element_count() != frames * entry.entries)
    return ffi::Unexpected(invalid("expected rows of shape (K, M)"));
  if (call.wantsCell && !entry.periodic)
    return ffi::Unexpected(invalid("the program has no periodic cell"));
  return call;
}

/// Evaluates the frames of a call from buffers of the host into buffers of
/// the host: the handler of the CPU on XLA's buffers, that of CUDA on its
/// copies of them. An output that is not asked is null.
ffi::Error evaluateCall(Entry &entry, const Call &call, const double *values, const double *x,
                        const double *c, double *out, char *g, double *cellOut,
                        double *rowsOut) {
  const size_t frames = call.frames, n = entry.particles;
  std::lock_guard<std::mutex> lock(entry.mutex);
  compiler::Simulation &simulation = *entry.simulation;
  // The values of the call: the tunables whose bits differ are updated
  // (D213), which ends the activation (D251).
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
  size_t width = entry.forcesDouble ? sizeof(double) : sizeof(float);
  // The gradient in the positions where the cell asks for it and the
  // positions do not.
  std::vector<char> scratch(call.wantsCell && !call.wantsForces ? 3 * n * width : 0);
  if (!call.wantsForces)
    g = nullptr;
  for (size_t k = 0; k != frames; ++k) {
    std::optional<std::array<double, 6>> cell;
    if (entry.periodic)
      cell = std::array<double, 6>{c[6 * k], c[6 * k + 1], c[6 * k + 2],
                                   c[6 * k + 3], c[6 * k + 4], c[6 * k + 5]};
    char *frameGradient = g ? g + 3 * n * width * k : scratch.empty() ? nullptr : scratch.data();
    auto result = simulation.evaluateFrame(x + 3 * n * k, cell, call.wantsRows, frameGradient,
                                           call.wantsCell);
    if (!result)
      return failed("frame " + std::to_string(k) + " of the call: " +
                     llvm::toString(result.takeError()));
    out[k] = result->energy;
    if (call.wantsRows) {
      double *row = rowsOut + k * entry.entries;
      for (const auto &part : result->gradient) {
        std::copy(part.begin(), part.end(), row);
        row += part.size();
      }
    }
    if (call.wantsCell) {
      double *target = cellOut + 9 * k;
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

ffi::Error execute(FrameState *state, ffi::Buffer<ffi::F64> theta,
                   ffi::Buffer<ffi::F64> positions, ffi::Buffer<ffi::F64> cells,
                   ffi::ResultBuffer<ffi::F64> energies, ffi::Result<ffi::AnyBuffer> gradient,
                   ffi::ResultBuffer<ffi::F64> cellGradients,
                   ffi::ResultBuffer<ffi::F64> rows) {
  if (!state || !state->entry)
    return invalid("the call has no program; this is a defect of mdir");
  Entry &entry = *state->entry;
  auto call = checkCall(entry, theta, positions, cells, energies, gradient, cellGradients, rows);
  if (call.has_error())
    return call.error();
  return evaluateCall(entry, *call, theta.typed_data(), positions.typed_data(),
                      cells.typed_data(), energies->typed_data(),
                      static_cast<char *>(gradient->untyped_data()),
                      cellGradients->typed_data(), rows->typed_data());
}

#ifdef MDIR_XLA_FFI_CUDA
/// The functions of the driver of CUDA that the handler of CUDA calls,
/// from libcuda.so.1 as the process has loaded it (XLA has), resolved once:
/// the extension does not link the driver, so that it imports on a machine
/// without one (as lib/Compiler/Skins.cpp reaches it).
struct Driver {
  decltype(&::cuMemcpyDtoHAsync) copyToHost = nullptr;
  decltype(&::cuMemcpyHtoDAsync) copyToDevice = nullptr;
  decltype(&::cuStreamSynchronize) synchronize = nullptr;
  decltype(&::cuStreamGetCtx) streamContext = nullptr;
  decltype(&::cuCtxGetCurrent) getCurrent = nullptr;
  decltype(&::cuCtxSetCurrent) setCurrent = nullptr;
  decltype(&::cuCtxGetDevice) contextDevice = nullptr;
  decltype(&::cuDeviceGet) device = nullptr;
  decltype(&::cuGetErrorString) errorString = nullptr;
  std::string missing;
};

const Driver &getDriver() {
  static const Driver driver = [] {
    Driver d;
    auto library = llvm::sys::DynamicLibrary::getPermanentLibrary("libcuda.so.1");
    if (!library.isValid()) {
      d.missing = "libcuda.so.1";
      return d;
    }
    auto find = [&](auto &slot, const char *name) {
      slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(
          library.getAddressOfSymbol(name));
      if (!slot && d.missing.empty())
        d.missing = name;
    };
    find(d.copyToHost, "cuMemcpyDtoHAsync_v2");
    find(d.copyToDevice, "cuMemcpyHtoDAsync_v2");
    find(d.synchronize, "cuStreamSynchronize");
    find(d.streamContext, "cuStreamGetCtx");
    find(d.getCurrent, "cuCtxGetCurrent");
    find(d.setCurrent, "cuCtxSetCurrent");
    find(d.contextDevice, "cuCtxGetDevice");
    find(d.device, "cuDeviceGet");
    find(d.errorString, "cuGetErrorString");
    return d;
  }();
  return driver;
}

/// The device that the runtime of MDIR runs a program of the GPU on: the
/// visible one that MDRT_DEVICE names, else the first (runtime/mdrt_cuda.c).
int runtimeDevice() {
  if (const char *chosen = std::getenv("MDRT_DEVICE"); chosen && *chosen)
    return static_cast<int>(std::strtol(chosen, nullptr, 10));
  return 0;
}

ffi::ErrorOr<std::unique_ptr<CudaFrameState>> instantiateCuda(int64_t program,
                                                              std::string_view fingerprint) {
  const Driver &driver = getDriver();
  if (!driver.missing.empty())
    return ffi::Unexpected(invalid("the driver of CUDA does not give " + driver.missing));
  auto entry = resolve(program, fingerprint);
  if (entry.has_error())
    return ffi::Unexpected(entry.error());
  auto state = std::make_unique<CudaFrameState>();
  state->entry = std::move(*entry);
  return state;
}

/// Copies of a call between XLA's buffers on the device and buffers of the
/// host, on XLA's stream, in the context of that stream.
class Staging {
public:
  Staging(const Driver &driver, CUstream stream) : driver(driver), stream(stream) {}
  ~Staging() {
    if (saved)
      driver.setCurrent(previous);
  }
  /// Makes the context of the stream current, keeping the one before.
  ffi::Error enter() {
    if (!saved) {
      if (ffi::Error error = check(driver.getCurrent(&previous), "cuCtxGetCurrent");
          error.failure())
        return error;
      saved = true;
      if (ffi::Error error = check(driver.streamContext(stream, &context), "cuStreamGetCtx");
          error.failure())
        return error;
    }
    return check(driver.setCurrent(context), "cuCtxSetCurrent");
  }
  ffi::Error toHost(void *host, const void *device, size_t bytes) {
    if (!bytes)
      return ffi::Error::Success();
    return check(driver.copyToHost(host, reinterpret_cast<CUdeviceptr>(device), bytes, stream),
                 "cuMemcpyDtoHAsync");
  }
  ffi::Error toDevice(void *device, const void *host, size_t bytes) {
    if (!bytes)
      return ffi::Error::Success();
    return check(driver.copyToDevice(reinterpret_cast<CUdeviceptr>(device), host, bytes, stream),
                 "cuMemcpyHtoDAsync");
  }
  ffi::Error wait() { return check(driver.synchronize(stream), "cuStreamSynchronize"); }
  /// Refuses a stream on another device than `ordinal`, after `enter`.
  ffi::Error checkDevice(int ordinal) {
    CUdevice wanted = 0, found = 0;
    if (ffi::Error error = check(driver.device(&wanted, ordinal), "cuDeviceGet"); error.failure())
      return error;
    if (ffi::Error error = check(driver.contextDevice(&found), "cuCtxGetDevice"); error.failure())
      return error;
    if (found == wanted)
      return ffi::Error::Success();
    return invalid("the call is on device " + std::to_string(found) +
                   ", and the program runs on device " + std::to_string(ordinal) +
                   " (MDRT_DEVICE, else the first visible device): place the arrays of the "
                   "call on that device");
  }

private:
  ffi::Error check(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS)
      return ffi::Error::Success();
    const char *text = nullptr;
    driver.errorString(result, &text);
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("mdir_frames: ") + what + ": " +
                                                     (text ? text : "an error of CUDA"));
  }
  const Driver &driver;
  CUstream stream;
  CUcontext previous = nullptr, context = nullptr;
  bool saved = false;
};

/// The handler of the platform "CUDA": the buffers of XLA are on the
/// device. The inputs are copied to the host on XLA's stream after the work
/// of XLA that wrote them, and waited for; the frames are evaluated as on
/// the CPU (a program of the GPU on its own stream, which `evaluateFrame`
/// waits for); and the outputs are copied to XLA's buffers on XLA's stream,
/// and waited for, before the work of XLA that reads them. With the boundary
/// of a batch on the device (PR 1 of #138), the program will read and write
/// XLA's buffers on XLA's stream instead.
ffi::Error executeCuda(CUstream stream, CudaFrameState *state, ffi::Buffer<ffi::F64> theta,
                       ffi::Buffer<ffi::F64> positions, ffi::Buffer<ffi::F64> cells,
                       ffi::ResultBuffer<ffi::F64> energies,
                       ffi::Result<ffi::AnyBuffer> gradient,
                       ffi::ResultBuffer<ffi::F64> cellGradients,
                       ffi::ResultBuffer<ffi::F64> rows) {
  if (!state || !state->entry)
    return invalid("the call has no program; this is a defect of mdir");
  Entry &entry = *state->entry;
  auto checked = checkCall(entry, theta, positions, cells, energies, gradient, cellGradients,
                           rows);
  if (checked.has_error())
    return checked.error();
  const Call &call = *checked;
  const size_t frames = call.frames, n = entry.particles;
  size_t width = entry.forcesDouble ? sizeof(double) : sizeof(float);
  std::vector<double> values(entry.entries), x(3 * n * frames),
      c(frames * (entry.periodic ? 6 : 0)), out(frames),
      cellOut(call.wantsCell ? 9 * frames : 0), rowsOut(call.wantsRows ? frames * entry.entries : 0);
  std::vector<char> g(call.wantsForces ? 3 * n * frames * width : 0);
  Staging staging(getDriver(), stream);
  // Each step runs once the one before it succeeded. MDIR's runtime runs
  // on one device of the process (#138, design part 3, Sharding): a call of
  // a program of the GPU on another is refused before anything runs. XLA
  // gives the device of a call to `execute`, not to `instantiate`.
  ffi::Error error = staging.enter();
  if (error.success() && entry.onDevice)
    error = staging.checkDevice(runtimeDevice());
  if (error.success())
    error = staging.toHost(values.data(), theta.typed_data(), values.size() * sizeof(double));
  if (error.success())
    error = staging.toHost(x.data(), positions.typed_data(), x.size() * sizeof(double));
  if (error.success())
    error = staging.toHost(c.data(), cells.typed_data(), c.size() * sizeof(double));
  if (error.success())
    error = staging.wait();
  if (error.success())
    error = evaluateCall(entry, call, values.data(), x.data(), c.data(), out.data(), g.data(),
                         cellOut.data(), rowsOut.data());
  // The runtime of a program of the GPU makes its own context current.
  if (error.success())
    error = staging.enter();
  if (error.success())
    error = staging.toDevice(energies->typed_data(), out.data(), out.size() * sizeof(double));
  if (error.success())
    error = staging.toDevice(gradient->untyped_data(), g.data(), g.size());
  if (error.success())
    error = staging.toDevice(cellGradients->typed_data(), cellOut.data(),
                             cellOut.size() * sizeof(double));
  if (error.success())
    error = staging.toDevice(rows->typed_data(), rowsOut.data(), rowsOut.size() * sizeof(double));
  if (error.success())
    error = staging.wait();
  return error;
}
#endif

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

#ifdef MDIR_XLA_FFI_CUDA
XLA_FFI_DEFINE_HANDLER(framesInstantiateCuda, instantiateCuda,
                       ffi::Ffi::BindInstantiate()
                           .Attr<int64_t>("program")
                           .Attr<std::string_view>("fingerprint"));

XLA_FFI_DEFINE_HANDLER(framesExecuteCuda, executeCuda,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<CUstream>>()
                           .Ctx<ffi::State<CudaFrameState>>()
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_Error *mdirFramesInstantiateCuda(XLA_FFI_CallFrame *frame) {
  if (XLA_FFI_Error *error = checkVersion(frame))
    return error;
  return framesInstantiateCuda(frame);
}

XLA_FFI_Error *mdirFramesExecuteCuda(XLA_FFI_CallFrame *frame) {
  if (XLA_FFI_Error *error = checkVersion(frame))
    return error;
  return framesExecuteCuda(frame);
}
#endif

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
  entry->onDevice = prepared.control.target == driver::Target::GPU;
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
  // `platform` is "cpu", or "CUDA" where MDIR was built with CUDA
  // (`_xla_ffi_platforms`).
  m.def("_xla_ffi_targets", [](const std::string &platform) {
    py::dict targets, frames;
    if (platform == "cpu") {
      frames["instantiate"] = capsule(reinterpret_cast<void *>(&mdirFramesInstantiate));
      frames["execute"] = capsule(reinterpret_cast<void *>(&mdirFramesExecute));
#ifdef MDIR_XLA_FFI_CUDA
    } else if (platform == "CUDA") {
      frames["instantiate"] = capsule(reinterpret_cast<void *>(&mdirFramesInstantiateCuda));
      frames["execute"] = capsule(reinterpret_cast<void *>(&mdirFramesExecuteCuda));
#endif
    } else {
      throw py::value_error("_xla_ffi_targets: no handlers for the platform " + platform);
    }
    targets["mdir_frames"] = frames;
    return targets;
  }, py::arg("platform") = "cpu");
  // The type of the state of an executable, which is registered before the
  // handlers (`jax.ffi.register_ffi_type`), for each platform.
  m.def("_xla_ffi_types", [](const std::string &platform) {
    py::dict types, state;
    if (platform == "cpu") {
      state["type_id"] = capsule(&FrameState::id);
      state["type_info"] = capsule(&frameStateInfo);
#ifdef MDIR_XLA_FFI_CUDA
    } else if (platform == "CUDA") {
      state["type_id"] = capsule(&CudaFrameState::id);
      state["type_info"] = capsule(&cudaFrameStateInfo);
#endif
    } else {
      throw py::value_error("_xla_ffi_types: no handlers for the platform " + platform);
    }
    types["mdir_frames_state"] = state;
    return types;
  }, py::arg("platform") = "cpu");
  // The platforms that the extension has handlers for.
  m.def("_xla_ffi_platforms", []() {
    py::list platforms;
    platforms.append("cpu");
#ifdef MDIR_XLA_FFI_CUDA
    platforms.append("CUDA");
#endif
    return platforms;
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

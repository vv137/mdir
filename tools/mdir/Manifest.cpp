// Optional execution provenance, independent of the compiled schedule (D168).
#include "Manifest.h"
#include "BuildInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include <cerrno>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <limits>
#include <set>

using namespace mdir;
using namespace mdir::tool;
using llvm::StringRef;
using llvm::json::Object;
using llvm::json::Value;

llvm::json::Object mdir::tool::getManifestBuild() {
  Value dirty = StringRef(MDIR_GIT_DIRTY) == "unknown"
                    ? Value(nullptr) : Value(StringRef(MDIR_GIT_DIRTY) == "yes");
  return Object{{"version", MDIR_VERSION}, {"git_commit", MDIR_GIT_COMMIT},
                {"dirty", std::move(dirty)}, {"llvm_version", LLVM_VERSION_STRING},
                {"mlir_version", LLVM_VERSION_STRING},
                {"cuda_toolkit_version", MDIR_CUDA_VERSION}};
}

std::string mdir::tool::hashInput(StringRef data) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(data)), true);
}

InputPaths mdir::tool::getManifestInputs(StringRef controlFile,
                                        const driver::Control &c,
                                        const driver::System &s) {
  InputPaths inputs = {{"control", controlFile.str()}, {"coordinates", c.pdbFile},
      {"amber_topology", c.prmtopFile}, {"amber_coordinates", c.amberCoordinateFile},
      {"gromacs_topology", c.gromacsTopologyFile},
      {"gromacs_coordinates", c.gromacsCoordinateFile},
      {"charmm_structure", c.charmmStructureFile},
      {"charmm_coordinates", c.charmmCoordinateFile}, {"checkpoint", c.restartInput}};
  for (const auto &path : c.charmmParameterFiles)
    inputs.push_back({"charmm_parameters", path});
  if (s.topology)
    for (const auto &path : s.topology->sourceFiles)
      inputs.push_back({"gromacs_topology", path});
  return inputs;
}

static std::string absolutePath(StringRef path) {
  // real_path cannot resolve a dangling final symlink. Follow its target
  // first, even when the output that it will name has not been created.
  std::filesystem::path candidate(path.str());
  for (unsigned links = 0; links != 40; ++links) {
    std::error_code error;
    auto target = std::filesystem::read_symlink(candidate, error);
    if (error)
      break;
    candidate = target.is_absolute() ? target : candidate.parent_path() / target;
  }
  std::string spelling = candidate.string();
  path = spelling;
  llvm::SmallString<256> result;
  if (!llvm::sys::fs::real_path(path, result))
    return result.str().str();
  result = path;
  llvm::sys::fs::make_absolute(result);
  // Outputs need not exist yet. Resolve their containing directory so
  // two spellings through a directory symlink cannot overwrite one file.
  llvm::SmallString<256> parent;
  if (!llvm::sys::fs::real_path(llvm::sys::path::parent_path(result), parent)) {
    llvm::sys::path::append(parent, llvm::sys::path::filename(result));
    return parent.str().str();
  }
  llvm::sys::path::remove_dots(result, true);
  return result.str().str();
}

static bool sameFile(StringRef a, StringRef b) {
  return absolutePath(a) == absolutePath(b) || llvm::sys::fs::equivalent(a, b);
}

llvm::Error mdir::tool::checkManifestPath(StringRef path,
                                          const InputPaths &inputs,
                                          llvm::ArrayRef<std::string> outputs) {
  if (path.empty())
    return llvm::Error::success();
  if (llvm::sys::fs::is_directory(path))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "manifest '%s' is a directory", path.str().c_str());
  for (const auto &[role, input] : inputs)
    if (!input.empty() && sameFile(path, input))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "manifest '%s' names an input of the run, '%s'", path.str().c_str(), input.c_str());
  for (const auto &output : outputs)
    if (!output.empty() && sameFile(path, output))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
          "manifest '%s' names another output of the run, '%s'",
          path.str().c_str(), output.c_str());
  return llvm::Error::success();
}

llvm::Expected<llvm::json::Array>
mdir::tool::hashManifestInputs(const InputPaths &inputs) {
  llvm::json::Array result;
  std::set<std::string> seen;
  for (const auto &[role, path] : inputs) {
    if (path.empty())
      continue;
    std::string absolute = absolutePath(path);
    if (!seen.insert(absolute).second)
      continue;
    auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
    if (!buffer)
      return llvm::createStringError(buffer.getError(), "cannot hash input '%s'", path.c_str());
    StringRef data = (*buffer)->getBuffer();
    result.push_back(Object{{"role", role}, {"path", absolute},
        {"bytes", static_cast<int64_t>(data.size())}, {"sha256", hashInput(data)}});
  }
  return result;
}

llvm::Expected<Value> mdir::tool::getManifestDevice(driver::Target target) {
  if (target == driver::Target::CPU)
    return Value(nullptr);
  // The JIT has loaded the kernels and made the runtime's selected context
  // current. Query that context, respecting both device-selection variables.
  auto library = llvm::sys::DynamicLibrary::getPermanentLibrary("libcuda.so.1");
  if (!library.isValid())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open the CUDA driver for the manifest");
  auto version = reinterpret_cast<int (*)(int *)>(
      library.getAddressOfSymbol("cuDriverGetVersion"));
  auto capability = reinterpret_cast<int (*)(int *, int *, int)>(
      library.getAddressOfSymbol("cuDeviceComputeCapability"));
  auto current = reinterpret_cast<int (*)(int *)>(
      library.getAddressOfSymbol("cuCtxGetDevice"));
  auto name = reinterpret_cast<int (*)(char *, int, int)>(
      library.getAddressOfSymbol("cuDeviceGetName"));
  struct UUID { unsigned char bytes[16]; };
  auto uuid = reinterpret_cast<int (*)(UUID *, int)>(
      library.getAddressOfSymbol("cuDeviceGetUuid"));
  int device = -1, driverVersion = 0, major = 0, minor = 0;
  char deviceName[256] = {};
  UUID id{};
  if (!current || !name || !uuid || !version || !capability || current(&device) ||
      version(&driverVersion) || capability(&major, &minor, device) ||
      name(deviceName, sizeof(deviceName), device) || uuid(&id, device))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot query the CUDA device for the manifest");
  // MDIR uses the driver API and does not require libcudart. Report a
  // runtime API version only if a runtime has actually been loaded.
  auto runtimeVersion = reinterpret_cast<int (*)(int *)>(
      llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("cudaRuntimeGetVersion"));
  int runtime = 0;
  Value runtimeAPI = runtimeVersion && runtimeVersion(&runtime) == 0
                         ? Value(runtime) : Value(nullptr);
  return Value(Object{{"api", "driver"}, {"driver_api_version", driverVersion},
      {"runtime_api_version", std::move(runtimeAPI)},
      {"compute_capability", llvm::json::Array{major, minor}},
      {"visible_index", device}, {"name", std::string(deviceName)},
      {"uuid_hex", llvm::toHex(llvm::ArrayRef<uint8_t>(id.bytes, 16), true)}});
}

static std::string timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char text[32];
  std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return text;
}

Manifest::~Manifest() {
  if (file)
    std::fclose(file);
}

llvm::Error Manifest::write(Object event) {
  event["schema_version"] = 1;
  event["invocation"] = invocation;
  event["timestamp_utc"] = timestamp();
  std::string line = llvm::formatv("{0}\n", Value(std::move(event))).str();
  if (std::fwrite(line.data(), 1, line.size(), file) != line.size() ||
      std::fflush(file) != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write manifest '%s'", path.c_str());
  return llvm::Error::success();
}

llvm::Error Manifest::start(const std::string &given, bool append, Object metadata) {
  path = given;
  if (append && llvm::sys::fs::exists(path)) {
    std::ifstream previous(path);
    if (!previous)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot read manifest '%s'", path.c_str());
    int64_t last = 0;
    bool ended = false;
    for (std::string line; std::getline(previous, line);) {
      auto json = llvm::json::parse(line);
      const Object *record = json ? json->getAsObject() : nullptr;
      auto number = record ? record->getInteger("invocation") : std::nullopt;
      auto event = record ? record->getString("event") : std::nullopt;
      bool start = event && *event == "start";
      bool end = event && *event == "end";
      bool valid = record && record->getInteger("schema_version") == 1 && number &&
          record->getString("timestamp_utc") &&
          ((start && *number == last + 1) ||
           (end && last > 0 && *number == last && !ended));
      if (end && record) {
        auto status = record->getString("status");
        valid &= status && (*status == "completed" || *status == "stopped");
      }
      if (!json)
        llvm::consumeError(json.takeError());
      // A partial last line is kept for diagnosis, never silently repaired.
      if (!valid || previous.eof() || *number == std::numeric_limits<int64_t>::max())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
            "manifest '%s' has malformed or unsupported history; use a new manifest path",
            path.c_str());
      last = *number;
      ended = end;
    }
    if (previous.bad())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot read manifest '%s'", path.c_str());
    invocation = last + 1;
  }
  file = std::fopen(path.c_str(), append ? "a" : "w");
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open manifest '%s'", path.c_str());
  began = std::chrono::steady_clock::now();
  metadata["event"] = "start";
  return write(std::move(metadata));
}

llvm::Error Manifest::finish(StringRef status, int64_t step, StringRef reason) {
  if (!file)
    return llvm::Error::success();
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  return write(Object{{"event", "end"}, {"status", status}, {"step", step},
                      {"reason", reason}, {"elapsed_seconds", elapsed}});
}

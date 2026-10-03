// Installation checks use the same executable and runtime as a real run (D155).

#include "Commands.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <cmath>
#include <cstdlib>
#include <optional>
#include <tuple>
#include <vector>

extern char **environ;

using llvm::StringRef;

namespace {

// Resolve the CUDA driver dynamically: the doctor must still run when the
// driver is missing. These entry points use only integers and strings from
// the stable CUDA driver ABI; no context or allocation is created here.
bool probeGPU() {
  std::string error;
  auto library = llvm::sys::DynamicLibrary::getPermanentLibrary(
      "libcuda.so.1", &error);
  if (!library.isValid()) {
    llvm::outs() << "FAIL CUDA driver: " << error
                 << "; install or expose the NVIDIA driver, or use "
                    "--target=cpu\n";
    return false;
  }
  auto init = reinterpret_cast<int (*)(unsigned)>(
      library.getAddressOfSymbol("cuInit"));
  auto version = reinterpret_cast<int (*)(int *)>(
      library.getAddressOfSymbol("cuDriverGetVersion"));
  auto count = reinterpret_cast<int (*)(int *)>(
      library.getAddressOfSymbol("cuDeviceGetCount"));
  auto device = reinterpret_cast<int (*)(int *, int)>(
      library.getAddressOfSymbol("cuDeviceGet"));
  auto name = reinterpret_cast<int (*)(char *, int, int)>(
      library.getAddressOfSymbol("cuDeviceGetName"));
  auto capability = reinterpret_cast<int (*)(int *, int *, int)>(
      library.getAddressOfSymbol("cuDeviceComputeCapability"));
  auto errorName = reinterpret_cast<int (*)(int, const char **)>(
      library.getAddressOfSymbol("cuGetErrorName"));
  if (!init || !version || !count || !device || !name || !capability) {
    llvm::outs() << "FAIL CUDA driver: required driver entry points are missing; "
                    "check the NVIDIA driver installation\n";
    return false;
  }
  auto checked = [&](int result, StringRef operation) {
    if (result == 0)
      return true;
    const char *description = nullptr;
    if (errorName)
      errorName(result, &description);
    llvm::outs() << "FAIL CUDA driver: " << operation << " returned " << result;
    if (description)
      llvm::outs() << " (" << description << ")";
    llvm::outs() << "; check the driver and device access, or use --target=cpu\n";
    return false;
  };
  int driver = 0, devices = 0;
  if (!checked(init(0), "cuInit") ||
      !checked(version(&driver), "cuDriverGetVersion") ||
      !checked(count(&devices), "cuDeviceGetCount"))
    return false;
  llvm::outs() << "CUDA driver API: " << driver / 1000 << "."
               << (driver % 1000) / 10 << "\n";
  if (devices == 0) {
    llvm::outs() << "FAIL CUDA devices: no visible device; check "
                    "CUDA_VISIBLE_DEVICES and GPU access, or use --target=cpu\n";
    return false;
  }
  int selected = 0;
  if (const char *value = std::getenv("MDRT_DEVICE"))
    if (StringRef(value).getAsInteger(10, selected) || selected < 0 ||
        selected >= devices) {
      llvm::outs() << "FAIL CUDA device: MDRT_DEVICE must name an index from 0 "
                     "to " << devices - 1 << " among the visible devices\n";
      return false;
    }
  for (int index = 0; index != devices; ++index) {
    int id = 0, major = 0, minor = 0;
    char deviceName[256] = {};
    if (!checked(device(&id, index), "cuDeviceGet") ||
        !checked(name(deviceName, sizeof(deviceName), id), "cuDeviceGetName") ||
        !checked(capability(&major, &minor, id), "cuDeviceComputeCapability"))
      return false;
    llvm::outs() << "CUDA device " << index << ": " << deviceName
                 << " (compute capability " << major << "." << minor << ")"
                 << (index == selected ? " [selected]" : "") << "\n";
  }
  return true;
}

std::string getPath(StringRef directory, StringRef name) {
  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, name);
  return std::string(path);
}

bool writeFile(StringRef path, StringRef text) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::outs() << "FAIL doctor files: cannot write '" << path
                 << "': " << error.message() << "\n";
    return false;
  }
  output << text;
  output.close();
  if (output.has_error()) {
    llvm::outs() << "FAIL doctor files: writing '" << path << "' failed\n";
    output.clear_error();
    return false;
  }
  return true;
}

std::string controlFile(bool gpu) {
  std::string text = R"TOML([input]
coordinates = "atoms.pdb"
[output]
energy_interval = 2
[energy]
cutoff = 8.0
pairlist_distance = 9.0
lennard_jones_modifier = "NONE"
[[energy.pair]]
name = "lj"
expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
mixing = "lorentz-berthelot"
[[energy.type]]
name = "AR"
mass = 39.95
epsilon = 0.2385
sigma = 3.4
[dynamics]
steps = 2
time_step = 0.001
[ensemble]
ensemble = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
box = [20.0, 20.0, 20.0]
[execution]
)TOML";
  text += gpu ? "target = \"GPU\"\nprecision = \"MIXED\"\n"
              : "target = \"CPU\"\nprecision = \"DOUBLE\"\nthreads = 2\n";
  return text;
}

// Check execution as well as process success: two energy rows, all finite,
// and the starting pair energy from its Lennard-Jones expression. The log
// prints four decimals, so 1e-4 covers its rounding on either target.
bool checkResults(StringRef path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return false;
  bool initial = false, final = false;
  StringRef text = (*file)->getBuffer();
  while (!text.empty()) {
    StringRef line;
    std::tie(line, text) = text.split('\n');
    if (!line.consume_front("INFO:"))
      continue;
    llvm::SmallVector<StringRef> columns;
    line.split(columns, ' ', -1, /*KeepEmpty=*/false);
    int64_t step;
    if (columns.empty() || columns[0].getAsInteger(10, step))
      continue;
    if (columns.size() != 8)
      return false;
    double values[7];
    for (int k = 0; k != 7; ++k)
      if (columns[k + 1].getAsDouble(values[k]) || !std::isfinite(values[k]))
        return false;
    if (step == 0) {
      double ratio6 = std::pow(3.4 / 4.0, 6);
      double expected = 4 * 0.2385 * (ratio6 * ratio6 - ratio6);
      if (std::abs(values[2] - expected) > 1e-4)
        return false;
      initial = true;
    }
    if (step == 2)
      final = true;
  }
  return initial && final;
}

bool smokeRun(StringRef executable, StringRef directory, bool gpu) {
  StringRef target = gpu ? "gpu" : "cpu";
  std::string control = getPath(directory, (target + ".toml").str());
  std::string out = getPath(directory, (target + ".out").str());
  std::string err = getPath(directory, (target + ".err").str());
  if (!writeFile(control, controlFile(gpu)))
    return false;
  llvm::outs() << "Checking " << target << ": compile and run two steps...\n";
  llvm::outs().flush();
  StringRef arguments[] = {executable, "run", control};
  std::optional<StringRef> redirects[] = {StringRef(""), StringRef(out),
                                         StringRef(err)};
  // Keep compiler failure artifacts beside the logs, even when the caller
  // requests a reproducer in the working directory.
  std::vector<std::string> environment;
  for (char **entry = environ; *entry; ++entry)
    if (!StringRef(*entry).starts_with("MDIR_REPRODUCER="))
      environment.push_back(*entry);
  environment.push_back("MDIR_REPRODUCER=" +
                        getPath(directory, (target + "-reproducer.mlir").str()));
  std::vector<StringRef> environmentRefs(environment.begin(), environment.end());
  std::string error;
  int status = llvm::sys::ExecuteAndWait(
      executable, arguments, environmentRefs, redirects,
      /*SecondsToWait=*/120, /*MemoryLimit=*/0, &error);
  if (status != 0 || !checkResults(out)) {
    llvm::outs() << "FAIL " << target << ": ";
    if (status != 0)
      llvm::outs() << "run exited with status " << status;
    else
      llvm::outs() << "the run did not produce the expected finite energies";
    if (!error.empty())
      llvm::outs() << " (" << error << ")";
    llvm::outs() << "\n  Read " << err << " and " << out
                 << "; rerun with:\n    mdir run '";
    for (char c : control)
      if (c == '\'')
        llvm::outs() << "'\\''";
      else
        llvm::outs() << c;
    llvm::outs() << "'\n";
    return false;
  }
  llvm::outs() << "PASS " << target << ": compiled and ran two steps with "
                  "finite energies\n";
  return true;
}
} // namespace

int mdir::tool::doctor(StringRef target, const char *argv0) {
  if (target != "all" && target != "cpu" && target != "gpu") {
    llvm::errs() << "mdir: doctor expects --target=all, cpu, or gpu, got '"
                 << target << "'\n";
    return 1;
  }
  printVersion(llvm::outs());
  bool hasGPU = MDIR_HAS_CUDA;
  if (target == "gpu" && !hasGPU) {
    llvm::outs() << "FAIL gpu: this build has no CUDA target; use a CUDA build "
                    "or --target=cpu\n";
    return 1;
  }
  llvm::SmallString<256> prefix, directory;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, prefix);
  llvm::sys::path::append(prefix, "mdir-doctor");
  if (auto error = llvm::sys::fs::createUniqueDirectory(prefix, directory)) {
    llvm::outs() << "FAIL doctor files: cannot create a temporary directory: "
                 << error.message() << "\n";
    return 1;
  }
  bool ok = writeFile(
      getPath(directory, "atoms.pdb"),
      "HETATM    1  AR  AR  A   1       1.000   1.000   1.000  1.00  0.00          AR\n"
      "HETATM    2  AR  AR  A   2       5.000   1.000   1.000  1.00  0.00          AR\n");
  if (ok) {
    static int anchor;
    std::string executable = llvm::sys::fs::getMainExecutable(argv0, &anchor);
    if (target != "gpu")
      ok = smokeRun(executable, directory, /*gpu=*/false);
    if (target != "cpu") {
      if (hasGPU) {
        // Do not short-circuit a GPU check after a CPU failure.
        bool gpuOK = probeGPU() && smokeRun(executable, directory, /*gpu=*/true);
        ok = gpuOK && ok;
      } else {
        llvm::outs() << "SKIP gpu: this build has no CUDA target\n";
      }
    }
  }
  if (!ok) {
    llvm::outs() << "doctor: checks failed; diagnostics kept in " << directory
                 << "\n";
    return 1;
  }
  if (auto error = llvm::sys::fs::remove_directories(directory)) {
    llvm::outs() << "FAIL doctor cleanup: " << error.message()
                 << "; temporary files remain in " << directory << "\n";
    return 1;
  }
  llvm::outs() << "doctor: all requested checks passed\n";
  return 0;
}

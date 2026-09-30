// `mdir version` and `mdir bug-report`: what a report of a defect needs to
// reproduce it. See docs/debugging.md.

#include "Commands.h"

#include "BuildInfo.h"

#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Control.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <optional>

using namespace mdir;
using namespace mdir::tool;
using llvm::StringRef;

#ifndef MDIR_VERSION
#define MDIR_VERSION "unknown"
#endif

extern char **environ;

/// Inputs no larger than this are copied into a report; larger ones are
/// named with their sizes and hashes.
static constexpr uint64_t maxCopiedInput = 16 << 20;

/// The version of the CUDA toolkit at `root`, from its version.json.
static std::string getToolkitVersion(StringRef root) {
  llvm::SmallString<256> path(root);
  llvm::sys::path::append(path, "version.json");
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return "unknown";
  auto json = llvm::json::parse((*buffer)->getBuffer());
  if (!json) {
    llvm::consumeError(json.takeError());
    return "unknown";
  }
  if (const auto *object = json->getAsObject())
    if (const auto *cuda = object->getObject("cuda"))
      if (auto version = cuda->getString("version"))
        return version->str();
  return "unknown";
}

void mdir::tool::printVersion(llvm::raw_ostream &os) {
  os << "MDIR " << MDIR_VERSION << "\n";
  os << "commit: " << MDIR_GIT_COMMIT << "\n";
  os << "uncommitted changes: " << MDIR_GIT_DIRTY << "\n";
  os << "LLVM " << LLVM_VERSION_STRING << "\n";
  os << "targets: cpu";
  if (StringRef(MDIR_CUDA_ROOT) != "")
    os << ", gpu (CUDA " << getToolkitVersion(MDIR_CUDA_ROOT) << " at "
       << MDIR_CUDA_ROOT << ")";
  os << "\n";
  os << "checkpoints: "
     << (driver::hasCheckpointSupport() ? "yes (HDF5)" : "no") << "\n";
}

namespace {

/// Writes the files of a report into a directory.
class Report {
public:
  explicit Report(StringRef directory) : directory(directory) {}

  std::string getPath(StringRef name) const {
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, name);
    return std::string(path);
  }

  /// Opens a file of the report for writing.
  std::unique_ptr<llvm::raw_fd_ostream> open(StringRef name) {
    std::error_code error;
    auto os = std::make_unique<llvm::raw_fd_ostream>(
        getPath(name), error, llvm::sys::fs::OF_Text);
    if (error) {
      llvm::errs() << "mdir: cannot write '" << getPath(name)
                   << "': " << error.message() << "\n";
      return nullptr;
    }
    return os;
  }

  /// Runs `program` with `arguments`, its output to `name`.out and its
  /// errors to `name`.err. Returns its exit status, or -1 if it did not
  /// run.
  int run(StringRef program, llvm::ArrayRef<StringRef> arguments,
          StringRef name,
          std::optional<llvm::ArrayRef<StringRef>> environment = {}) {
    std::string out = getPath((name + ".out").str());
    std::string err = getPath((name + ".err").str());
    std::optional<StringRef> redirects[] = {StringRef(""), StringRef(out),
                                            StringRef(err)};
    std::string message;
    int status = llvm::sys::ExecuteAndWait(program, arguments, environment,
                                           redirects, /*SecondsToWait=*/0,
                                           /*MemoryLimit=*/0, &message);
    if (status < 0 && !message.empty())
      if (auto os = open((name + ".err").str()))
        *os << message << "\n";
    return status;
  }

  std::string directory;
};

} // namespace

/// Adds to `os` the size and SHA-256 of `path`, and copies the file into
/// the report if it is small.
static void describeInput(Report &report, llvm::raw_ostream &os,
                          StringRef role, StringRef path) {
  if (path.empty())
    return;
  os << role << ": " << path << "\n";
  auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
  if (!buffer) {
    os << "  cannot be read: " << buffer.getError().message() << "\n";
    return;
  }
  StringRef data = (*buffer)->getBuffer();
  os << "  bytes: " << data.size() << "\n";
  os << "  sha256: "
     << llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(data)),
                    /*LowerCase=*/true)
     << "\n";
  if (data.size() > maxCopiedInput) {
    os << "  not copied: larger than " << (maxCopiedInput >> 20)
       << " MiB; attach it if you can\n";
    return;
  }
  std::string copy = report.getPath(
      ("inputs/" + role + "-" + llvm::sys::path::filename(path)).str());
  if (std::error_code error = llvm::sys::fs::copy_file(path, copy))
    os << "  not copied: " << error.message() << "\n";
  else
    os << "  copied to: inputs/" << llvm::sys::path::filename(copy) << "\n";
}

int mdir::tool::writeBugReport(StringRef controlFile, StringRef directory,
                               bool runs, const char *argv0) {
  if (std::error_code error = llvm::sys::fs::create_directories(
          (directory + "/inputs").str())) {
    llvm::errs() << "mdir: cannot create '" << directory
                 << "': " << error.message() << "\n";
    return 1;
  }
  Report report(directory);
  std::string executable =
      llvm::sys::fs::getMainExecutable(argv0, (void *)&printVersion);

  // The build and the machine.
  if (auto os = report.open("version.txt")) {
    printVersion(*os);
    *os << "executable: " << executable << "\n";
  }
  if (auto nvidiaSmi = llvm::sys::findProgramByName("nvidia-smi")) {
    StringRef arguments[] = {
        *nvidiaSmi, "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv"};
    report.run(*nvidiaSmi, arguments, "gpus");
  }
  if (auto uname = llvm::sys::findProgramByName("uname")) {
    StringRef arguments[] = {*uname, "-a"};
    report.run(*uname, arguments, "host");
  }

  // The environment that the compiler and the runtime read.
  if (auto os = report.open("environment.txt"))
    for (char **entry = environ; *entry; ++entry) {
      StringRef variable(*entry);
      if (variable.starts_with("MDIR_") || variable.starts_with("MDRT_") ||
          variable.starts_with("CUDA_") || variable.starts_with("OMP_") ||
          variable.starts_with("LD_LIBRARY_PATH="))
        *os << variable << "\n";
    }

  // The control file and the inputs that it names.
  {
    auto os = report.open("inputs.txt");
    if (!os)
      return 1;
    describeInput(report, *os, "control", controlFile);
    auto control = driver::readControl(controlFile);
    if (!control) {
      *os << "the control file cannot be read: "
          << llvm::toString(control.takeError()) << "\n";
    } else {
      describeInput(report, *os, "pdb", control->pdbFile);
      describeInput(report, *os, "prmtop", control->prmtopFile);
      describeInput(report, *os, "inpcrd", control->amberCoordinateFile);
      describeInput(report, *os, "top", control->gromacsTopologyFile);
      describeInput(report, *os, "gro", control->gromacsCoordinateFile);
      describeInput(report, *os, "restart", control->restartInput);
      for (const std::string &include : control->gromacsIncludes)
        *os << "include directory: " << include << "\n";
    }
  }

  // What the driver makes of the run, each step in a process of its own so
  // that a crash of one still leaves the others. The passes write a
  // reproducer if they fail.
  std::string reproducer = report.getPath("reproducer.mlir");
  std::string reproducerVariable = "MDIR_REPRODUCER=" + reproducer;
  std::vector<std::string> environment;
  for (char **entry = environ; *entry; ++entry)
    if (!StringRef(*entry).starts_with("MDIR_REPRODUCER="))
      environment.push_back(*entry);
  environment.push_back(reproducerVariable);
  std::vector<StringRef> environmentRefs(environment.begin(),
                                         environment.end());

  llvm::SmallVector<std::pair<std::string, int>> steps;
  auto step = [&](StringRef name, llvm::ArrayRef<StringRef> arguments) {
    int status = report.run(executable, arguments, name, environmentRefs);
    steps.push_back({name.str(), status});
  };
  {
    StringRef arguments[] = {executable, "check", controlFile};
    step("check", arguments);
  }
  {
    StringRef arguments[] = {executable, "emit", controlFile};
    step("module", arguments);
  }
  {
    StringRef arguments[] = {executable, "emit", "--stage=pipeline",
                             controlFile};
    step("pipeline", arguments);
  }
  {
    StringRef arguments[] = {executable, "emit", "--stage=lowered",
                             controlFile};
    step("lowered", arguments);
  }
  if (runs) {
    // Each launch is waited for, so that a failure names its kernel.
    std::vector<StringRef> runEnvironment = environmentRefs;
    runEnvironment.push_back("MDRT_WAIT=1");
    StringRef arguments[] = {executable, "run", controlFile};
    int status = report.run(executable, arguments, "run", runEnvironment);
    steps.push_back({"run", status});
  }

  if (auto os = report.open("README.txt")) {
    *os << "A report of MDIR, written by `mdir bug-report`.\n\n"
           "version.txt, gpus.out, host.out   the build and the machine\n"
           "environment.txt                   variables that MDIR reads\n"
           "inputs.txt, inputs/               the control file and its "
           "inputs, with hashes\n"
           "check.out                         what `mdir check` reads\n"
           "module.out                        the program of the run "
           "(`mdir emit`)\n"
           "pipeline.out                      the passes that compile it\n"
           "lowered.out                       the program after the passes\n"
           "reproducer.mlir                   if the passes failed: the "
           "input of the pass that failed\n"
           "run.out, run.err                  with --run: the log of the run "
           "under MDRT_WAIT=1\n"
           "*.err                             the errors of each step\n\n";
    *os << "exit status of each step (-1: did not start; -2: crashed or aborted):\n";
    for (auto &[name, status] : steps)
      *os << "  " << name << ": " << status << "\n";
  }

  // Steps that wrote no errors leave no files of errors.
  for (StringRef name : {"gpus", "host", "check", "module", "pipeline",
                         "lowered", "run"}) {
    std::string err = report.getPath((name + ".err").str());
    uint64_t size = 0;
    if (!llvm::sys::fs::file_size(err, size) && size == 0)
      llvm::sys::fs::remove(err);
  }

  llvm::outs() << "mdir: wrote a report to '" << directory << "'\n";
  for (auto &[name, status] : steps)
    llvm::outs() << "  " << name << ": "
                 << (status == 0 ? "ok" : "failed (" + std::to_string(status) +
                                              ")")
                 << "\n";
  llvm::outs() << "Attach it to an issue as an archive:\n  tar czf "
               << llvm::sys::path::filename(directory) << ".tar.gz -C "
               << (llvm::sys::path::parent_path(directory).empty()
                       ? StringRef(".")
                       : llvm::sys::path::parent_path(directory))
               << " " << llvm::sys::path::filename(directory) << "\n";
  return 0;
}

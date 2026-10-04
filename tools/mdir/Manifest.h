// Optional execution provenance (D168).
#ifndef MDIR_TOOL_MANIFEST_H
#define MDIR_TOOL_MANIFEST_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"
#include "llvm/Support/JSON.h"
#include <chrono>
#include <cstdio>

namespace mdir::tool {
llvm::json::Object getManifestBuild();
/// The version of MDIR and its commit, "0.1.0 (<commit>)", as a checkpoint
/// records its creator (D173).
std::string getBuildVersion();
std::string hashInput(llvm::StringRef data);
using InputPaths = std::vector<std::pair<std::string, std::string>>;
InputPaths getManifestInputs(llvm::StringRef controlFile,
                             const driver::Control &control,
                             const driver::System &system);
llvm::Error checkManifestPath(llvm::StringRef path, const InputPaths &inputs,
                              llvm::ArrayRef<std::string> outputs);
/// Protect tabulated inputs against resolved output and checkpoint-backup
/// paths, including the part names of --no-append (D178).
llvm::Error checkTabulatedInputs(const InputPaths &inputs,
                                 llvm::ArrayRef<std::string> outputs);
llvm::Expected<llvm::json::Array> hashManifestInputs(const InputPaths &inputs);
llvm::Expected<llvm::json::Value> getManifestDevice(driver::Target target);

class Manifest {
public:
  Manifest() = default;
  Manifest(const Manifest &) = delete;
  Manifest &operator=(const Manifest &) = delete;
  ~Manifest();
  llvm::Error start(const std::string &path, bool append,
                     llvm::json::Object metadata);
  llvm::Error finish(llvm::StringRef status, int64_t step,
                      llvm::StringRef reason = "");
private:
  llvm::Error write(llvm::json::Object event);
  std::FILE *file = nullptr;
  std::string path;
  int64_t invocation = 1;
  std::chrono::steady_clock::time_point began;
};
} // namespace mdir::tool
#endif

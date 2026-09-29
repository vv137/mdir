// The subcommands of `mdir`.

#ifndef MDIR_TOOLS_MDIR_COMMANDS_H
#define MDIR_TOOLS_MDIR_COMMANDS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mdir {
namespace tool {

/// What `runControl` does with the program of the run.
enum class Emit {
  /// Print it as it is built.
  Module,
  /// Print it as it is executed, in the LLVM dialect.
  Lowered,
  /// Execute it.
  Run,
};

/// Reads the control file `controlFile`, builds the program of the run,
/// and compiles and executes it or prints it. `argv0` is the name that the
/// program was started with, which tells where the runtime is. Returns the
/// exit status.
int runControl(llvm::StringRef controlFile, Emit emit, const char *argv0);

/// Reads the control file and the input that it names, and prints what
/// they describe. Returns the exit status.
int checkControl(llvm::StringRef controlFile);

/// Describes a checkpoint, or compares the states of two. Returns the exit
/// status: 0 if the states are identical, 1 if they differ, 2 on an error.
int describeCheckpoints(llvm::ArrayRef<std::string> files);

} // namespace tool
} // namespace mdir

#endif // MDIR_TOOLS_MDIR_COMMANDS_H

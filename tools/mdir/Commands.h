// The subcommands of `mdir`.

#ifndef MDIR_TOOLS_MDIR_COMMANDS_H
#define MDIR_TOOLS_MDIR_COMMANDS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mdir {
namespace tool {

/// What `runControl` does with the program of the run.
enum class Emit {
  /// Print it as it is built.
  Module,
  /// Print it as it is executed, in the LLVM dialect.
  Lowered,
  /// Print the pipeline of passes that lowers it.
  Pipeline,
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

/// Prints the version of MDIR, the commit it was built from, and what the
/// build supports.
void printVersion(llvm::raw_ostream &os);

/// Writes into `directory` what a report of a defect in the run of
/// `controlFile` needs: the versions, the environment, the inputs with
/// their hashes, and the program at each stage, each made by a process of
/// its own. With `runs`, runs it as well, waiting for each kernel. Returns
/// the exit status.
int writeBugReport(llvm::StringRef controlFile, llvm::StringRef directory,
                   bool runs, const char *argv0);

} // namespace tool
} // namespace mdir

#endif // MDIR_TOOLS_MDIR_COMMANDS_H

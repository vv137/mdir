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

/// How `mdir run` treats a run that one job may not finish (D129 to D131).
struct RunOptions {
  /// Continue the run from its own checkpoint, if there is one, until it
  /// has taken its `steps`; begin it if there is none.
  bool continues = false;
  /// Append the frames of a continued run to its trajectory, rather than
  /// write them to a part of their own.
  bool appends = true;
  /// The wall time that the run may take, in s, or 0 for no limit. The run
  /// stops at the last checkpoint that leaves time for the next interval.
  double maxWalltime = 0.0;
};

/// Reads the control file `controlFile`, builds the program of the run,
/// and compiles and executes it or prints it. `argv0` is the name that the
/// program was started with, which tells where the runtime is. Returns the
/// exit status.
int runControl(llvm::StringRef controlFile, Emit emit, const char *argv0,
               const RunOptions &options = {});

/// Reads a wall time: a number of hours, or hours and minutes, and
/// optionally seconds, as `H:MM[:SS]`. Returns the seconds, or a negative
/// number if `text` is neither.
double parseWalltime(llvm::StringRef text);

/// Reads the control file and the input that it names, and prints what
/// they describe. Returns the exit status.
int checkControl(llvm::StringRef controlFile);

/// Describes a checkpoint, or compares the states of two. Returns the exit
/// status: 0 if the states are identical, 1 if they differ, 2 on an error.
/// Describes a checkpoint, compares two, or prints a field of one: the
/// mass and the three numbers of `field` of each particle.
int describeCheckpoints(llvm::ArrayRef<std::string> files,
                        llvm::StringRef field = "");

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

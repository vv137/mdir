// mdir: the command line of MDIR. See docs/design-m1.md, Section 14.

#include "Commands.h"

#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Control.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mdir;
using namespace mdir::tool;

static llvm::cl::SubCommand runCommand("run",
                                       "Compile a run and execute it");
static llvm::cl::SubCommand emitCommand(
    "emit", "Print the program of a run instead of executing it");
static llvm::cl::SubCommand checkCommand(
    "check", "Read the input of a run and print what it describes");
static llvm::cl::SubCommand templateCommand(
    "template", "Print a control file with every keyword");
static llvm::cl::SubCommand checkpointCommand(
    "checkpoint", "Describe a checkpoint, or compare the states of two");
static llvm::cl::SubCommand versionCommand(
    "version", "Print the version and what this build supports");

static llvm::cl::opt<std::string>
    controlFile(llvm::cl::Positional, llvm::cl::desc("<control file>"),
                llvm::cl::Required, llvm::cl::sub(runCommand),
                llvm::cl::sub(emitCommand), llvm::cl::sub(checkCommand));

static llvm::cl::opt<Emit> stage(
    "stage", llvm::cl::desc("Which form of the program to print"),
    llvm::cl::values(
        clEnumValN(Emit::Module, "module", "As it is built (default)"),
        clEnumValN(Emit::Lowered, "lowered",
                   "As it is executed, in the LLVM dialect")),
    llvm::cl::init(Emit::Module), llvm::cl::sub(emitCommand));

static llvm::cl::opt<std::string>
    templateName(llvm::cl::Positional, llvm::cl::desc("<kind>: md"),
                 llvm::cl::Required, llvm::cl::sub(templateCommand));

static llvm::cl::list<std::string> checkpointFiles(
    llvm::cl::Positional, llvm::cl::desc("<checkpoint> [<checkpoint>]"),
    llvm::cl::OneOrMore, llvm::cl::sub(checkpointCommand));

#ifndef MDIR_VERSION
#define MDIR_VERSION "unknown"
#endif

static int printVersion() {
  llvm::outs() << "MDIR " << MDIR_VERSION << "\n";
  llvm::outs() << "LLVM " << LLVM_VERSION_STRING << "\n";
  llvm::outs() << "targets: cpu";
  if (llvm::StringRef(MDIR_CUDA_ROOT) != "")
    llvm::outs() << ", gpu (CUDA)";
  llvm::outs() << "\n";
  llvm::outs() << "checkpoints: "
               << (driver::hasCheckpointSupport() ? "yes (HDF5)" : "no")
               << "\n";
  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "MDIR: compiles and runs molecular dynamics\n\n"
      "  mdir run <control file>\n"
      "  mdir emit <control file> [--stage=module|lowered]\n"
      "  mdir check <control file>\n"
      "  mdir template md\n"
      "  mdir checkpoint <checkpoint> [<checkpoint>]\n"
      "  mdir version\n");

  if (runCommand)
    return runControl(controlFile, Emit::Run, argv[0]);
  if (emitCommand)
    return runControl(controlFile, stage, argv[0]);
  if (checkCommand)
    return checkControl(controlFile);
  if (templateCommand) {
    if (templateName != "md") {
      llvm::errs() << "mdir: expected the template 'md', got '"
                   << templateName << "'\n";
      return 1;
    }
    llvm::outs() << driver::getControlTemplate();
    return 0;
  }
  if (checkpointCommand)
    return describeCheckpoints(checkpointFiles);
  if (versionCommand)
    return printVersion();

  llvm::errs() << "mdir: expected a subcommand; see 'mdir --help'\n";
  return 1;
}

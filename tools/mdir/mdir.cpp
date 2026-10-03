// mdir: the command line of MDIR. See docs/design-m1.md, Section 14.

#include "Commands.h"

#include "mdir/Driver/Control.h"

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
    "template", "Print a reference control file or a standard pipeline stage");
static llvm::cl::SubCommand checkpointCommand(
    "checkpoint", "Describe a checkpoint, or compare the states of two");
static llvm::cl::SubCommand bugReportCommand(
    "bug-report", "Collect what a report of a defect in a run needs");
static llvm::cl::SubCommand versionCommand(
    "version", "Print the version and what this build supports");

static llvm::cl::opt<std::string>
    controlFile(llvm::cl::Positional, llvm::cl::desc("<control file>"),
                llvm::cl::Required, llvm::cl::sub(runCommand),
                llvm::cl::sub(emitCommand), llvm::cl::sub(checkCommand),
                llvm::cl::sub(bugReportCommand));

static llvm::cl::opt<bool> continueRun(
    "continue",
    llvm::cl::desc("Continue the run from the checkpoint of its [output] "
                   "until it has taken its 'steps', or begin it if there is "
                   "none; a complete run exits with 0"),
    llvm::cl::sub(runCommand));

static llvm::cl::opt<bool> noAppend(
    "no-append",
    llvm::cl::desc("With --continue, write the frames that follow to a part "
                   "of their own, <trajectory>.partNNNN.dcd (or .xtc)"),
    llvm::cl::sub(runCommand));

static llvm::cl::opt<std::string> maxWalltime(
    "max-walltime",
    llvm::cl::desc("Stop at the last checkpoint that leaves time for the "
                   "next interval within this wall time, given in hours or "
                   "as H:MM[:SS]; a run that stops exits with 75"),
    llvm::cl::value_desc("time"), llvm::cl::sub(runCommand));

static llvm::cl::opt<std::string>
    reportDirectory("o", llvm::cl::desc("The directory of the report"),
                    llvm::cl::value_desc("directory"),
                    llvm::cl::init("mdir-report"),
                    llvm::cl::sub(bugReportCommand));

static llvm::cl::opt<bool>
    reportRuns("run",
               llvm::cl::desc("Run as well, waiting for each kernel, and "
                              "keep the log"),
               llvm::cl::sub(bugReportCommand));

static llvm::cl::opt<Emit> stage(
    "stage", llvm::cl::desc("Which form of the program to print"),
    llvm::cl::values(
        clEnumValN(Emit::Module, "module", "As it is built (default)"),
        clEnumValN(Emit::Lowered, "lowered",
                   "As it is executed, in the LLVM dialect"),
        clEnumValN(Emit::Pipeline, "pipeline",
                   "The passes that lower it")),
    llvm::cl::init(Emit::Module), llvm::cl::sub(emitCommand));

static llvm::cl::opt<std::string>
    templateName(llvm::cl::Positional,
                 llvm::cl::desc("<kind>: md, amber, minimize, nvt, npt, production"),
                 llvm::cl::Required, llvm::cl::sub(templateCommand));

static llvm::cl::list<std::string> checkpointFiles(
    llvm::cl::Positional, llvm::cl::desc("<checkpoint> [<checkpoint>]"),
    llvm::cl::OneOrMore, llvm::cl::sub(checkpointCommand));
static llvm::cl::opt<std::string> checkpointField(
    "print",
    llvm::cl::desc("Print a field of each particle in the order of the "
                   "input: its mass and its positions (nm), velocities "
                   "(nm/ps), or forces (kJ/mol/nm)"),
    llvm::cl::value_desc("positions|velocities|forces"),
    llvm::cl::sub(checkpointCommand));

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "MDIR: compiles and runs molecular dynamics\n\n"
      "  mdir run <control file> [--continue [--no-append]] "
      "[--max-walltime=<time>]\n"
      "  mdir emit <control file> [--stage=module|lowered|pipeline]\n"
      "  mdir check <control file>\n"
      "  mdir template md|amber|minimize|nvt|npt|production\n"
      "  mdir checkpoint <checkpoint> [<checkpoint>] "
      "[--print=positions|velocities|forces]\n"
      "  mdir bug-report <control file> [-o <directory>] [--run]\n"
      "  mdir version\n");

  if (runCommand) {
    RunOptions options;
    options.continues = continueRun;
    options.appends = !noAppend;
    if (noAppend && !continueRun) {
      llvm::errs() << "mdir: --no-append goes with --continue\n";
      return 1;
    }
    if (!maxWalltime.empty()) {
      options.maxWalltime = parseWalltime(maxWalltime);
      if (options.maxWalltime <= 0.0) {
        llvm::errs() << "mdir: expected a wall time in hours or as "
                        "H:MM[:SS], got '"
                     << maxWalltime << "'\n";
        return 1;
      }
    }
    return runControl(controlFile, Emit::Run, argv[0], options);
  }
  if (emitCommand)
    return runControl(controlFile, stage, argv[0]);
  if (checkCommand)
    return checkControl(controlFile);
  if (templateCommand) {
    if (templateName == "md") {
      llvm::outs() << driver::getControlTemplate();
      return 0;
    }
    if (templateName == "amber") {
      llvm::outs() << driver::getAmberControlTemplate();
      return 0;
    }
    std::string pipeline = driver::getPipelineControlTemplate(templateName);
    if (!pipeline.empty()) {
      llvm::outs() << pipeline;
      return 0;
    }
    llvm::errs() << "mdir: expected a template: md, amber, minimize, nvt, "
                    "npt, production; got '"
                 << templateName << "'\n";
    return 1;
  }
  if (checkpointCommand)
    return describeCheckpoints(checkpointFiles, checkpointField);
  if (bugReportCommand)
    return writeBugReport(controlFile, reportDirectory, reportRuns, argv[0]);
  if (versionCommand) {
    printVersion(llvm::outs());
    return 0;
  }

  llvm::errs() << "mdir: expected a subcommand; see 'mdir --help'\n";
  return 1;
}

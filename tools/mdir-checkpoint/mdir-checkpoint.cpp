// mdir-checkpoint: describes and compares checkpoints.

#include "mdir/Driver/Checkpoint.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>

using namespace mdir::driver;

static llvm::cl::list<std::string>
    files(llvm::cl::Positional, llvm::cl::desc("<checkpoint> [<checkpoint>]"),
          llvm::cl::OneOrMore);

static int fail(llvm::Error error) {
  llvm::errs() << "mdir-checkpoint: " << llvm::toString(std::move(error))
               << "\n";
  return 2;
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "MDIR: describes a checkpoint, or compares the states of two\n");
  if (files.size() > 2) {
    llvm::errs() << "mdir-checkpoint: expected one checkpoint or two\n";
    return 2;
  }

  auto first = readCheckpoint(files[0]);
  if (!first)
    return fail(first.takeError());

  if (files.size() == 1) {
    std::printf("particles:       %zu\n", first->getNumParticles());
    std::printf("step:            %lld\n",
                static_cast<long long>(first->step));
    std::printf("time:            %g ps\n", first->time);
    std::printf("box:             %g %g %g nm\n", first->box[0],
                first->box[1], first->box[2]);
    std::printf("forces:          %s\n", first->forces.empty() ? "no" : "yes");
    std::printf("integrator:      %s\n", first->integrator.c_str());
    std::printf("velocity offset: %g time steps\n", first->velocityOffset);
    std::printf("precision:       %s\n", first->precision.c_str());
    std::printf("time step:       %g ps\n", first->timestep);
    return 0;
  }

  auto second = readCheckpoint(files[1]);
  if (!second)
    return fail(second.takeError());
  std::string difference = compareCheckpoints(*first, *second);
  if (difference.empty()) {
    std::printf("the states are identical\n");
    return 0;
  }
  std::printf("%s\n", difference.c_str());
  return 1;
}

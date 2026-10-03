// `mdir checkpoint`: describes and compares checkpoints.

#include "Commands.h"

#include "mdir/Driver/Checkpoint.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>

using namespace mdir::driver;

static int fail(llvm::Error error) {
  llvm::errs() << "mdir: " << llvm::toString(std::move(error)) << "\n";
  return 2;
}

int mdir::tool::describeCheckpoints(llvm::ArrayRef<std::string> files,
                                    llvm::StringRef field) {
  if (files.empty() || files.size() > 2) {
    llvm::errs() << "mdir: expected one checkpoint or two\n";
    return 2;
  }

  auto first = readCheckpoint(files[0]);
  if (!first)
    return fail(first.takeError());

  if (!field.empty()) {
    if (files.size() != 1) {
      llvm::errs() << "mdir: --print takes one checkpoint\n";
      return 2;
    }
    // What defined the run (D[checkpoint-fingerprint]), an entry a line.
    if (field == "fingerprint") {
      for (const FingerprintEntry &entry : first->fingerprint)
        std::printf("%s %s = %s\n", entry.group.c_str(), entry.name.c_str(),
                    entry.value.c_str());
      return 0;
    }
    const std::vector<double> *values =
        field == "positions"    ? &first->positions
        : field == "velocities" ? &first->velocities
        : field == "forces"     ? &first->forces
                                : nullptr;
    if (!values) {
      llvm::errs() << "mdir: --print takes positions, velocities, forces, "
                      "or fingerprint, not '"
                   << field << "'\n";
      return 2;
    }
    if (values->empty()) {
      llvm::errs() << "mdir: the checkpoint holds no " << field << "\n";
      return 2;
    }
    for (size_t i = 0, e = first->getNumParticles(); i != e; ++i)
      std::printf("%zu %.17g %.17g %.17g %.17g\n", i, first->masses[i],
                  (*values)[3 * i], (*values)[3 * i + 1],
                  (*values)[3 * i + 2]);
    return 0;
  }

  if (files.size() == 1) {
    std::printf("format:          %d, written by %s %s\n", checkpointFormat,
                first->creator.c_str(), first->creatorVersion.c_str());
    std::printf("particles:       %zu\n", first->getNumParticles());
    std::printf("step:            %lld\n",
                static_cast<long long>(first->step));
    std::printf("time:            %g ps\n", first->time);
    std::printf("box:             %g %g %g nm\n", first->box[0],
                first->box[1], first->box[2]);
    if (first->tilt[0] != 0.0 || first->tilt[1] != 0.0 ||
        first->tilt[2] != 0.0)
      std::printf("tilts:           %g %g %g nm (b_x, c_x, c_y)\n",
                  first->tilt[0], first->tilt[1], first->tilt[2]);
    std::printf("forces:          %s\n", first->forces.empty() ? "no" : "yes");
    std::printf("integrator:      %s\n", first->integrator.c_str());
    std::printf("velocity offset: %g time steps\n", first->velocityOffset);
    std::printf("precision:       %s\n", first->precision.c_str());
    std::printf("time step:       %g ps\n", first->timestep);
    std::printf("run began at:    step %lld\n",
                static_cast<long long>(first->firstStep));
    std::printf("part:            %lld\n", static_cast<long long>(first->part));
    if (first->outputsPart > 0)
      std::printf("outputs to part: %lld\n",
                  static_cast<long long>(first->outputsPart));
    if (!first->trajectory.empty())
      std::printf("trajectory:      %s, %lld frames\n",
                  first->trajectory.c_str(),
                  static_cast<long long>(first->frames));
    std::printf("fingerprint:     %zu entries (--print=fingerprint)\n",
                first->fingerprint.size());
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

// Rule (B) in written frames must not replace the dynamics' rule (A).
// This tests reporter isolation against the existing general no-step API;
// the independent physical oracle is a separate template-kernel gate.
#include "mdir/Compiler/Simulation.h"
#include "mdir/Driver/H5MD.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace mdir;
template <class T> static T take(llvm::Expected<T> value) {
  if (!value) {
    llvm::errs() << llvm::toString(value.takeError()) << '\n';
    std::exit(1);
  }
  return std::move(*value);
}
static void check(bool value, const char *message) {
  if (!value) { llvm::errs() << message << '\n'; std::exit(1); }
}
static void succeed(llvm::Error error) {
  if (error) { llvm::errs() << llvm::toString(std::move(error)) << '\n'; std::exit(1); }
}
static bool bits(const std::vector<double> &a, const std::vector<double> &b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}
int main(int argc, char **argv) {
  check(argc == 3, "expected control and output trajectory");
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  model::PreparedModel prepared;
  prepared.control = take(driver::readControl(argv[1]));
  prepared.system = take(driver::readSystem(prepared.control));
  driver::resolveNeighborStructure(prepared.control, prepared.system);
  prepared.execution.target = prepared.control.target;
  prepared.execution.precision = prepared.control.precision;
  auto simulation = take(compiler::Simulation::create(prepared, false));
  compiler::Simulation::Reports reports;
  reports.trajectoryPath = argv[2];
  reports.trajectoryFormat = driver::TrajectoryFormat::H5MD;
  reports.framePeriod = 1;
  reports.frameForces = reports.frameVelocities = true;
  succeed(simulation->setReports(reports));
  std::vector<compiler::SimulationState> states;
  for (unsigned frame = 0; frame != 2; ++frame) {
    check(take(simulation->run(1, {}, true)) == 1, "run did not take one step");
    auto before = take(simulation->getState());
    uint64_t generation = simulation->getGeneration();
    succeed(simulation->writeStateFrame());
    auto after = take(simulation->getState());
    check(bits(before.positions, after.positions) &&
          bits(before.velocities, after.velocities) &&
          bits(before.forces, after.forces) && before.step == after.step &&
          generation == simulation->getGeneration(),
          "writing a force frame changed the carried dynamics state");
    states.push_back(std::move(before));
  }
  simulation->closeReports();
  // Keep the same physics but select the general entry for the reference.
  prepared.control.waterTemplate = driver::WaterTemplateChoice::Off;
  auto reader = take(driver::H5MDReader::open(argv[2]));
  check(reader->getNumFrames() == 2 && reader->getWidth() == 8,
        "wrong force-frame layout");
  double maxDifference = 0.0, maxScale = 0.0, ruleDifference = 0.0;
  for (unsigned k = 0; k != 2; ++k) {
    auto frame = take(reader->read(k));
    const auto &state = states[k];
    check(frame.positions.size() == state.positions.size() * sizeof(double) &&
          std::memcmp(frame.positions.data(), state.positions.data(), frame.positions.size()) == 0 &&
          frame.velocities.size() == state.velocities.size() * sizeof(double) &&
          std::memcmp(frame.velocities.data(), state.velocities.data(), frame.velocities.size()) == 0,
          "report changed positions or velocities");
    std::array<double, 6> cell = {state.box[0], state.box[1], state.box[2],
                                 state.tilt[0], state.tilt[1], state.tilt[2]};
    prepared.system.positions = state.positions;
    prepared.system.velocities = state.velocities;
    for (unsigned axis = 0; axis != 3; ++axis) {
      prepared.system.box[axis] = cell[axis];
      prepared.system.tilt[axis] = cell[3 + axis];
    }
    auto general = take(compiler::Simulation::create(prepared, false));
    succeed(general->evaluate());
    auto referenceState = take(general->getState());
    check(frame.forces.size() == referenceState.forces.size() * sizeof(double), "missing frame forces");
    for (size_t i = 0; i != referenceState.forces.size(); ++i) {
      double observed;
      std::memcpy(&observed, frame.forces.data() + i * sizeof(double), sizeof(double));
      double reference = referenceState.forces[i];
      check(std::isfinite(observed) && std::isfinite(reference), "nonfinite force");
      maxDifference = std::max(maxDifference, std::abs(observed - reference));
      maxScale = std::max(maxScale, std::abs(reference));
      ruleDifference = std::max(ruleDifference, std::abs(observed - state.forces[i]));
    }
  }
  check(maxDifference <= 1e-5 * maxScale, "written force differs from general rule (B)");
  check(ruleDifference > 1e-5 * maxScale, "fixture does not distinguish rules (A) and (B)");
  llvm::outs() << "two rule-(B) force frames; carried state unchanged; continued dynamics\n"
               << "general-entry max difference " << maxDifference << ", scale " << maxScale
               << ", relative bound 1e-5; rule-(A)/(B) difference " << ruleDifference << '\n';
}

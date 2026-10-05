// A simulation that persists across runs of any number of steps, for an
// embedding front end (D[python-segments], docs/python-segments.md).
#ifndef MDIR_COMPILER_SIMULATION_H
#define MDIR_COMPILER_SIMULATION_H
#include "mdir/Driver/Model.h"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace mdir { namespace driver { struct Output; } }
namespace mdir { namespace compiler {

/// A failure while a simulation runs, or an operation that another one under
/// way on the same simulation excludes.
class SimulationError : public llvm::ErrorInfo<SimulationError> {
public:
  static char ID;
  explicit SimulationError(std::string message) : message(std::move(message)) {}
  void log(llvm::raw_ostream &os) const override { os << message; }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
  std::string message;
};

/// The energies of a step of energy, as a row of the log of `mdir run`
/// holds them: kJ/mol, K, bar, nm^3. `total` is potential and kinetic;
/// `conserved` adds the energy that the coupling has taken (D[python-segments]).
struct SimulationEnergies {
  double potential = 0.0, kinetic = 0.0, total = 0.0, conserved = 0.0,
         temperature = 0.0, virial = 0.0, pressure = 0.0, volume = 0.0;
};

/// The state of a simulation, in the order of the input and in MD units.
struct SimulationState {
  int64_t step = 0;
  double time = 0.0;
  std::vector<double> positions, velocities;
  /// Those that the next step begins with; empty before the first run.
  std::vector<double> forces;
  /// The time of the velocities after `time`, in time steps: -0.5 for
  /// leapfrog once it has run.
  double velocityOffset = 0.0;
  std::array<double, 3> box{}, tilt{};
  /// Those of the step `step`, if the run that ended there ended with a
  /// step of energy.
  std::optional<SimulationEnergies> energies;
};

class Simulation {
public:
  /// Builds the program of the first segment of `prepared` and compiles it.
  static llvm::Expected<std::unique_ptr<Simulation>>
  create(const model::PreparedModel &prepared);
  ~Simulation();
  Simulation(const Simulation &) = delete;
  Simulation &operator=(const Simulation &) = delete;

  /// Takes `count` more steps in parts of about `partSeconds` at most.
  /// Between parts it stops if a stop was requested or `poll` returns
  /// true. Returns the steps taken.
  /// With `energy`, the last step of the run is a step of energy, as a
  /// row of the log of `mdir run` is, whose energies getState returns.
  llvm::Expected<int64_t> run(int64_t count,
                              const std::function<bool()> &poll = {},
                              bool energy = false);
  /// Asks a run under way to stop after its part; from any thread.
  void requestStop() { stopRequested = true; }

  /// The state after the last part that succeeded.
  llvm::Expected<SimulationState> getState() const;
  int64_t getStep() const { return step; }
  double getTime() const;
  bool hasFailed() const { return failed; }
  /// The longest a part may take, in s, which bounds the latency of a stop.
  double partSeconds = 0.5;

  /// A compiled program of segments, with what its entry takes besides the
  /// state.
  struct Engine;

private:
  Simulation() = default;
  llvm::Expected<Engine *> getEngine();
  /// The counts of a part: periods (or steps) of the first nest, its plain
  /// steps per period, plain steps after it, a plain step of energy (0 or
  /// 1), and a period that closes with a step of energy (0 or 1) with its
  /// plain steps.
  struct Part {
    int64_t outer = 0, inner = 0, tail = 0, plain = 0, close = 0,
            closeInner = 0;
  };
  llvm::Error runPart(Engine &engine, Part part);

  model::PreparedModel prepared;
  std::unique_ptr<Engine> first, continued;
  /// What the compiled code reports to: the cell, the bath, the state at
  /// the end of a part.
  std::unique_ptr<driver::Output> output;
  /// The system that the programs are built from, at its first step, and
  /// the state between parts.
  driver::System initial, system;
  std::vector<double> forces;
  int64_t step = 0;
  bool hasRun = false;
  bool failed = false;
  /// The time that a step took in the last part long enough to tell, in
  /// s, which sets the length of the next part; 0 until then.
  double secondsPerStep = 0.0;
  std::atomic<bool> stopRequested{false};
  mutable std::atomic<bool> busy{false};
};

} }
#endif

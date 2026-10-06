// A simulation that persists across runs of any number of steps, for an
// embedding front end (D196, docs/python-segments.md).
#ifndef MDIR_COMPILER_SIMULATION_H
#define MDIR_COMPILER_SIMULATION_H
#include "mdir/Compiler/CompileCache.h"
#include "mdir/Driver/Model.h"
#include "mdir/Driver/Trajectory.h"
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
/// `conserved` adds the energy that the coupling has taken (D196).
struct SimulationEnergies {
  double potential = 0.0, kinetic = 0.0, total = 0.0, conserved = 0.0,
         temperature = 0.0, virial = 0.0, pressure = 0.0, volume = 0.0;
};

/// The row that `mdir run` logs at a step of a minimization, in kJ/mol and
/// nm (D202): the energy with its constant parts, the RMS and
/// the largest force without their parts along the constraints, the
/// particle of the largest (zero-based), and the length of the next step.
struct SimulationMinimization {
  double energy = 0.0, rmsForce = 0.0, maxForce = 0.0, stepSize = 0.0;
  int64_t maxForceParticle = 0;
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
  /// That of the last step of a minimization that has taken steps.
  std::optional<SimulationMinimization> minimization;
  /// The version of the values of the tunable parameters that the forces
  /// and the energies are of (D[python-tunable]).
  int64_t tunablesVersion = 0;
};

class Simulation {
public:
  /// Builds the program of the segments of `prepared` and compiles it, once
  /// (D211).
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
  /// Evaluates the forces at the state, and the energies of a step of
  /// energy, without taking a step (D[python-tunable]): the start of the
  /// run before the first, and the forces of the state anew after it.
  /// After the first run, leapfrog's energies are not those of the step
  /// (its velocities are half a step behind) and stay unset; a program
  /// without tunables refuses leapfrog then.
  llvm::Error evaluate();
  /// Takes `count` more steps of the minimization of `mdir run`, or the
  /// steps of its schedule if none, in parts as `run` does; a simulation
  /// of a program that minimizes takes only these (D202).
  llvm::Expected<int64_t> minimize(std::optional<int64_t> count = {},
                                   const std::function<bool()> &poll = {});
  bool isMinimization() const { return prepared.control.minimize; }
  /// The built-in reports of `mdir run`'s outputs (D207):
  /// the columns file of the energies every `energyPeriod` steps and a
  /// trajectory every `framePeriod` steps, 0 for none. Files are backed up
  /// as `mdir run` backs up its outputs and opened at once; they stay open
  /// across runs, and the reports are written inside the parts of a run.
  struct Reports {
    std::string energyPath;
    int64_t energyPeriod = 0;
    std::string trajectoryPath;
    driver::TrajectoryFormat trajectoryFormat = driver::TrajectoryFormat::DCD;
    int64_t framePeriod = 0;
  };
  llvm::Error setReports(const Reports &reports);
  /// Flushes and closes the files of the reports.
  void closeReports();
  const Reports &getReports() const { return reports; }

  /// Asks a run under way to stop after its part; from any thread.
  void requestStop() { stopRequested = true; }

  /// The state after the last part that succeeded.
  llvm::Expected<SimulationState> getState() const;

  /// The tunable parameters of the program (D[python-tunable],
  /// docs/python-tunable.md), their values, the version of the values (0
  /// at the creation, one more for each update), and for each version the
  /// step after which it holds.
  const model::TunableSet &getTunables() const { return prepared.tunables; }
  const std::vector<std::vector<double>> &getTunableValues() const {
    return tunableValues;
  }
  int64_t getTunablesVersion() const { return tunablesVersion; }
  const std::vector<std::pair<int64_t, int64_t>> &getTunablesHistory() const {
    return tunablesHistory;
  }
  /// Gives the tunables named in `changes` the values given, at once: the
  /// values of the program are built anew from the model with them, which
  /// must give the program compiled (a structural change is refused), and
  /// the forces of the state are evaluated anew. On any failure nothing
  /// changes.
  llvm::Error updateTunables(
      const std::vector<std::pair<std::string, std::vector<double>>> &changes);
  int64_t getStep() const { return step; }
  double getTime() const;
  bool hasFailed() const { return failed; }
  /// What compiling its programs cost, and what the compile cache saved
  /// (D212).
  CompileStats getCompileStats() const;
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
    /// The periods of coupling of an interval of the second nest, less one
    /// (D207).
    int64_t closePeriods = 0;
  };
  llvm::Error runPart(Engine &engine, Part part);
  /// A part of no steps: the evaluation of `evaluate`.
  llvm::Error evaluatePart();
  /// The steps of the next part, in multiples of `unit`.
  int64_t getPartSteps(int64_t unit) const;
  /// The next step after `step` at which a built-in report is due, or -1.
  int64_t getNextReport(int64_t step) const;
  Reports reports;

  model::PreparedModel prepared;
  /// The one program of the simulation, whose entry begins the run on the
  /// first call and continues the last segment on the others
  /// (D211).
  std::unique_ptr<Engine> compiled;
  /// What the compiled code reports to: the cell, the bath, the state at
  /// the end of a part.
  std::unique_ptr<driver::Output> output;
  /// The state between parts; the program is built from it at the first
  /// step.
  driver::System system;
  std::vector<double> forces;
  int64_t step = 0;
  /// A minimization: the steps of its schedule, and the length of its next
  /// step in nm.
  int64_t minimizationSteps = 0;
  double minimizationSize = 0.0;
  bool hasRun = false;
  bool failed = false;
  /// The system that the program was built from, at its first step, which
  /// the values of the tunables are put into to build them anew; their
  /// values, version, and history (D[python-tunable]).
  driver::System compiledSystem;
  std::vector<std::vector<double>> tunableValues;
  int64_t tunablesVersion = 0;
  std::vector<std::pair<int64_t, int64_t>> tunablesHistory;
  /// Whether the part under way evaluates the forces anew after an update:
  /// a call of the entry with `%first_call` 2 and no steps.
  bool refreshing = false;
  /// The time that a step took in the last part long enough to tell, in
  /// s, which sets the length of the next part; 0 until then.
  double secondsPerStep = 0.0;
  std::atomic<bool> stopRequested{false};
  mutable std::atomic<bool> busy{false};
};

} }
#endif

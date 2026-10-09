// A simulation that persists across runs of any number of steps, for an
// embedding front end (D196, docs/python-segments.md).
#ifndef MDIR_COMPILER_SIMULATION_H
#define MDIR_COMPILER_SIMULATION_H
#include "mdir/Compiler/CompileCache.h"
#include "mdir/Driver/Checkpoint.h"
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
/// The activation of the entry that runs the parts of a simulation
/// (D215).
struct Activation;

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
  /// Whether `maxForce` is below the tolerance of the call of `minimize`
  /// that wrote the row; none if that call had no tolerance
  /// (D219).
  std::optional<bool> converged;
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
  /// and the energies are of (D213).
  int64_t tunablesVersion = 0;
  /// The observed energies of terms and their derivatives at the step
  /// `step` (D189, D232), by the names of their columns, in
  /// kJ/mol and kJ/mol per unit of the constant, where `energies` is set
  /// and the program observes.
  std::optional<std::vector<std::pair<std::string, double>>> observables;
};

/// The buffers of the state of a simulation where its program keeps them,
/// in the order of the program and the types that it stores
/// (D220, docs/python-dlpack.md).
struct SimulationView {
  const void *positions = nullptr, *velocities = nullptr, *forces = nullptr;
  /// The index in the input of the particle at each place.
  const int32_t *ids = nullptr;
  size_t count = 0;
  /// The width of an element of the state and of the forces, 4 or 8.
  size_t stateWidth = 8, forceWidth = 8;
  /// Whether the buffers are memory of the device, and its ordinal.
  bool onDevice = false;
  int device = 0;
  int64_t step = 0;
  double time = 0.0, velocityOffset = 0.0;
  /// What `getGeneration` returned when the view was taken.
  uint64_t generation = 0;
};

class Simulation {
public:
  /// Builds the program of the segments of `prepared` and compiles it, once
  /// (D211). Without `cache`, the compilation reads and writes no entry of
  /// the compile cache (D217). With `store`, the simulation takes the code
  /// that an earlier one left there under the same key, or leaves its own
  /// (D236); without `cache` it does neither.
  static llvm::Expected<std::unique_ptr<Simulation>>
  create(const model::PreparedModel &prepared, bool cache = true,
         CodeStore *store = nullptr);
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
  /// energy, without taking a step (D213): the start of the
  /// run before the first, and the forces of the state anew after it.
  /// After the first run, leapfrog's energies are not those of the step
  /// (its velocities are half a step behind) and stay unset; a program
  /// without tunables refuses leapfrog then.
  llvm::Error evaluate();
  /// Takes `count` more steps of the minimization of `mdir run`, or the
  /// steps of its schedule if none, in parts as `run` does; a simulation
  /// of a program that minimizes takes only these (D202). With a
  /// `tolerance` in kJ/mol/nm, it stops at the first row whose largest
  /// force is below it, checked every energy period of the program and at
  /// the end, as `[minimize] force_tolerance` (D219).
  llvm::Expected<int64_t> minimize(std::optional<int64_t> count = {},
                                   const std::function<bool()> &poll = {},
                                   std::optional<double> tolerance = {});
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
    /// The file of `[output] observables` (D189) every `observablesPeriod`
    /// steps, of a program whose terms observe (D232).
    std::string observablesPath;
    int64_t observablesPeriod = 0;
  };
  /// The names of the columns that the program observes, in their order.
  std::vector<std::string> getObservableNames() const;
  llvm::Error setReports(const Reports &reports);
  /// Flushes and closes the files of the reports.
  void closeReports();
  const Reports &getReports() const { return reports; }

  /// Asks a run under way to stop after its part; from any thread.
  void requestStop() { stopRequested = true; }

  /// The state after the last part that succeeded, copied from where the
  /// program keeps it.
  llvm::Expected<SimulationState> getState() const;

  /// The tunable parameters of the program (D213,
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
  /// The derivative of the potential energy in the tunables at the state
  /// (D230, docs/python-gradient.md): one array per
  /// tunable, of the shape of its values, in kJ/mol per unit of the
  /// tunable; the potential energy that it is the derivative of, that of
  /// the potential that the forces sample (D210): the energy that the run
  /// reports, with the pair terms that the run cuts at the cutoff shifted
  /// to 0 there and the estimate of that shift; the version of the values
  /// and the step; and for each tunable whether its derivative is zero by
  /// proof.
  struct TunableGradient {
    std::vector<std::vector<double>> values;
    std::vector<bool> zero;
    double energy = 0.0;
    int64_t version = 0, step = 0;
  };
  /// Evaluates the state without a step, as `evaluate` does, with the
  /// derivative in the tunables, which only a program compiled with it
  /// carries.
  llvm::Expected<TunableGradient> evaluateTunableGradient();
  /// Writes the checkpoint of `mdir run` (H5MD format 1, D173) of the state
  /// after the last run, with `.prev` rotation and durable replacement, and
  /// the additional entries of a Python simulation
  /// (D223, docs/python-checkpoints.md). A simulation that
  /// has not run evaluates its start first. `creatorVersion` names the
  /// program that writes it.
  llvm::Error saveCheckpoint(const std::string &path,
                             const std::string &creatorVersion);
  /// Takes the state of `checkpoint` before the first run. Without
  /// `stage`, the same run goes on, as `mdir run --continue` continues it:
  /// a checkpoint of other physics or coupling is refused. With `stage`, a
  /// new stage begins at the checkpoint, as `[input] checkpoint` begins
  /// one: its forces are taken only if physics and coupling are the same,
  /// and evaluated at the first step otherwise. With `append` the files of
  /// the reporters of the same run continue those of the checkpoint;
  /// otherwise they are those of a part of their own (D149). Returns the
  /// notes for the front end: execution that differs, or what a stage
  /// evaluates anew.
  llvm::Expected<std::vector<std::string>>
  continueFrom(const driver::Checkpoint &checkpoint, bool stage, bool append);
  /// The file that a built-in report writes: `path`, or that of the part of
  /// the outputs of a continued run (D149).
  std::string getReportPath(const std::string &path) const;
  int64_t getStep() const { return step; }
  double getTime() const;

  /// The buffers of the state at the end of the last part, with a lease on
  /// them (D220): while a lease is alive, `run`, `minimize`,
  /// `evaluate`, and `updateTunables` are refused, and the buffers hold the
  /// state of the view. Refused when there is no activation: before the
  /// first part and after a failure.
  llvm::Expected<SimulationView> takeView();
  /// Takes another lease, of a view that a consumer was given, and
  /// releases one; from any thread.
  void acquireLease() { ++leases; }
  void releaseLease() { --leases; }
  int64_t getLeases() const { return leases; }
  /// Advances whenever the buffers of the state may change: a part, the
  /// beginning or the end of an activation.
  uint64_t getGeneration() const { return generation; }
  /// Makes the stream `consumer` of the device (1 the legacy default
  /// stream, 2 the per-thread default stream) wait for the work issued so
  /// far, and records that the buffers were given to a consumer, whose work
  /// the next part or the end of the activation waits for.
  void handOff(uint64_t consumer);
  /// Records that a consumer was given the buffers of the device without a
  /// stream to wait.
  void markExported() { exported = true; }

  /// A writable borrow of the buffers of the state (D229,
  /// docs/python-dlpack.md): the view of `takeView`, with a lease, for a
  /// consumer that writes the positions and the velocities. It excludes
  /// every other operation on the state, views included, until it is
  /// committed or, once no lease is left, abandoned. Refused without an
  /// activation, under a lease, and for a minimization.
  llvm::Expected<SimulationView> takeBorrow();
  /// The fields of the state that a consumer was given to write.
  enum : unsigned { WrittenPositions = 1, WrittenVelocities = 2 };
  void markWritten(unsigned fields) { borrowWritten |= fields; }
  unsigned getWritten() const { return borrowWritten; }
  bool isBorrowed() const { return borrowed; }
  /// Takes what was written through the borrow: the positions and the
  /// velocities marked as written, from the buffers; the cell `cell`, if
  /// given, as its diagonal a_x, b_y, c_z and its tilts b_x, c_x, c_y (the
  /// tilts of an orthorhombic cell are zero, and those of a triclinic one
  /// are not all zero: which of the two a program takes is fixed when it
  /// is compiled); and the values `tunables`, as `updateTunables`
  /// takes them. Everything is checked before anything changes; a refusal
  /// (InputError) leaves the borrow live. Then the versions of the changed
  /// fields advance and an activation begins from the committed state,
  /// whose forces are evaluated anew; if that fails, the commit is undone
  /// and the borrow ends. Only the lease of the borrow itself may be alive.
  /// Returns the names of the fields changed.
  llvm::Expected<std::vector<std::string>> commitBorrow(
      const std::optional<std::array<double, 6>> &cell,
      const std::vector<std::pair<std::string, std::vector<double>>> &tunables);
  /// The cell, in nm: its diagonal a_x, b_y, c_z, the edges of an
  /// orthorhombic cell, and its tilts b_x, c_x, c_y.
  std::array<double, 6> getCell() const;
  /// Whether the program was compiled for a triclinic cell, whose tilts a
  /// borrow may write (D[borrow-tilts]).
  bool hasTriclinicCell() const;
  bool isPeriodic() const { return prepared.control.periodic; }
  /// The number of commits that changed the positions, the velocities, and
  /// the cell from the host (P16), and for each commit that changed
  /// anything the step and the names of its fields.
  struct StateVersions {
    int64_t positions = 0, velocities = 0, cell = 0;
  };
  StateVersions getStateVersions() const { return stateVersions; }
  const std::vector<std::pair<int64_t, std::vector<std::string>>> &
  getCommits() const {
    return commits;
  }
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
  /// The program built anew with the values `values` of the tunables, or
  /// an InputError if its text would change.
  llvm::Expected<driver::Program>
  rebuildTunables(const std::vector<std::vector<double>> &values);
  /// A part of no steps: the evaluation of `evaluate`, the start of an
  /// activation. With `committed`, the evaluation of a state that a borrow
  /// committed, which every program of segments takes with leapfrog too
  /// (its half kick back is a select on `%first_call`, D223).
  llvm::Error evaluatePart(bool committed = false);
  /// Begins an activation of the entry from the state of the host
  /// (`%first_call` as given) and runs it to the end of its start.
  llvm::Error startActivation(int64_t firstCall);
  /// Runs the activation to the end of the next part, whose step and counts
  /// it is given.
  void resumeActivation();
  /// Ends the activation: what it allocated is freed (#110), and its state,
  /// if not copied, is lost.
  void endActivation();
  /// Makes the state of the host that of the activation, if it is not.
  void downloadState() const;
  /// Copies the state of the activation, where it is, as that which a part
  /// that fails returns to (D196), and makes the state of the host that copy.
  void takeSnapshot();
  void restoreSnapshot();
  void freeSnapshot();
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
  /// The activation of its entry that runs the parts, waiting at the end of
  /// the last one, with the state in its buffers (D215);
  /// none before the first part, after a failure, and while the values of
  /// the program change.
  std::unique_ptr<Activation> activation;
  /// Whether `system` and `forces` hold the state of the activation, or of
  /// the simulation if there is none.
  mutable bool hostCurrent = true;
  /// The state at the end of the last part that succeeded, where the
  /// activation keeps it and in its order: on a device, memory of the
  /// device; on the CPU, of the host.
  struct Snapshot {
    void *positions = nullptr, *velocities = nullptr, *forces = nullptr;
    size_t stateBytes = 0, forceBytes = 0;
    bool onDevice = false, valid = false;
    std::vector<char> host;
  } snapshot;
  /// What the compiled code reports to: the cell, the bath, the state at
  /// the end of a part.
  std::unique_ptr<driver::Output> output;
  /// The state between parts, when `hostCurrent`; the program is built from
  /// it at the first step.
  mutable driver::System system;
  mutable std::vector<double> forces;
  int64_t step = 0;
  /// A minimization: the steps of its schedule, and the length of its next
  /// step in nm.
  int64_t minimizationSteps = 0;
  double minimizationSize = 0.0;
  /// The steps between the checks of a tolerance: the energy period of
  /// the program (D219).
  int64_t minimizationCheckPeriod = 0;
  /// Whether the row at `minimizationCheckedStep` is below the tolerance
  /// of the call that wrote it; none without a tolerance.
  std::optional<bool> minimizationConverged;
  int64_t minimizationCheckedStep = -1;
  bool hasRun = false;
  bool failed = false;
  /// The run that a checkpoint continues or a stage begins from
  /// (D223): the step it began at, its part, the part of
  /// its outputs (0 for the names given), and for the same run whose
  /// files continue, the step through which the energy file keeps its rows
  /// and the trajectory and frames the checkpoint counts.
  int64_t runFirstStep = 0, runPartNumber = 1, outputsPart = 0;
  std::optional<int64_t> keepThrough;
  std::string continuedTrajectory;
  int64_t continuedFrames = 0;
  /// Whether the first part evaluates the forces of the state given, a
  /// stage of other physics.
  bool startRefresh = false;
  /// The system that the program was built from, at its first step, which
  /// the values of the tunables are put into to build them anew; their
  /// values, version, and history (D213).
  driver::System compiledSystem;
  std::vector<std::vector<double>> tunableValues;
  int64_t tunablesVersion = 0;
  std::vector<std::pair<int64_t, int64_t>> tunablesHistory;
  /// Whether the activation that begins takes the derivative in the
  /// tunables (the start value `tunable_gradient`).
  bool gradientAsked = false;
  /// Whether the part under way evaluates the forces anew after an update:
  /// a call of the entry with `%first_call` 2 and no steps.
  bool refreshing = false;
  /// The time that a step took in the last part long enough to tell, in
  /// s, which sets the length of the next part; 0 until then.
  double secondsPerStep = 0.0;
  std::atomic<bool> stopRequested{false};
  /// The live leases of views, the generation of the buffers, and whether
  /// buffers of the device were given to a consumer since the last wait
  /// for all the work of the device (D220).
  std::atomic<int64_t> leases{0};
  std::atomic<uint64_t> generation{0};
  std::atomic<bool> exported{false};
  /// Refuses an operation that would write or free the buffers of views,
  /// and every operation under a writable borrow; a borrow that was
  /// abandoned and has no lease left is undone first.
  llvm::Error checkLeases(const char *operation);
  /// Whether a writable borrow has not been committed or undone, the
  /// fields of the state that it gave out to write, the versions, and the
  /// commits (D229).
  std::atomic<bool> borrowed{false};
  std::atomic<unsigned> borrowWritten{0};
  StateVersions stateVersions;
  std::vector<std::pair<int64_t, std::vector<std::string>>> commits;
  /// Copies the fields that an abandoned borrow gave out back from the
  /// snapshot, so that the activation continues as if it had not been.
  void undoBorrow();
  /// Makes `cell` (the diagonal, then the tilts) the cell of the host: that
  /// of the state, of the next activation, of the frames, and of a
  /// checkpoint.
  void setCell(const std::array<double, 6> &cell);
  /// What a commit asks of a cell that it is given (D[borrow-tilts]).
  llvm::Error checkCommittedCell(const std::array<double, 6> &cell) const;
  SimulationView describeView() const;
  /// Waits for the work of consumers of views before the buffers are
  /// written or freed.
  void waitForConsumers();
  mutable std::atomic<bool> busy{false};
};

} }
#endif

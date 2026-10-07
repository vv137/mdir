// Native object model shared by the Python and control-file front ends.
#ifndef MDIR_DRIVER_MODEL_H
#define MDIR_DRIVER_MODEL_H
#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Cell.h"
namespace mdir {
namespace model {

/// Bindings map these categories to Python InputError and UnsupportedError.
class ModelError : public llvm::ErrorInfo<ModelError> {
public:
  static char ID;
  enum Kind { Input, Unsupported };
  ModelError(Kind kind, std::string message) : kind(kind), message(std::move(message)) {}
  void log(llvm::raw_ostream &os) const override { os << message; }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
  Kind kind;
  std::string message;
};
enum class Format { Amber, Gromacs, Charmm };
enum class Electrostatics { Cutoff, PME };
/// The `[energy] coulomb_modifier` of the control file: whether the
/// real-space Coulomb term of PME is shifted to zero at the cutoff
/// (Control::pmeShift; D205).
enum class CoulombModifier { None, PotentialShift };
enum class EnsembleKind { NVE, NVT, NPT };

/// A tunable parameter (D213, docs/python-tunable.md): a
/// vector θ of M entries and a map from the sites of `parameter` to them.
/// `parameter` is "charge" (sites: the particles), "sigma" or "epsilon"
/// (the Lennard-Jones types, with the combining rule), or, with `term`, a
/// constant of that pair term (one site) or a parameter of that tuple term
/// (its tuples). `map` has an entry for each site, the entry of θ it takes
/// or -1 to keep its value; empty, each site is an entry of its own.
/// `values` are the initial θ; empty, those of the model.
struct Tunable {
  std::string name, parameter, term;
  std::vector<int64_t> map;
  std::vector<double> values;
  /// The rule of σ of a pair of types: Arithmetic (Lorentz) or Geometric.
  driver::Mixing mixing = driver::Mixing::Arithmetic;
};
/// The tunables of a prepared model, resolved against it.
struct TunableSet {
  struct Entry {
    enum Kind { Charge, Sigma, Epsilon, PairConstant, TupleParameter };
    std::string name, parameter, term;
    Kind kind = Charge;
    /// The index of the term among the pair terms of the control or the
    /// tuple terms of the topology.
    unsigned termIndex = 0;
    /// One entry per site, -1 for a site that keeps its value.
    std::vector<int64_t> map;
    size_t entries = 0;
    driver::Mixing mixing = driver::Mixing::Arithmetic;
    /// The unit of the values, for messages and the plan.
    std::string unit;
  };
  std::vector<Entry> tunables;
  /// The values of each tunable, in MD units.
  std::vector<std::vector<double>> values;
  /// σ and ε of each type as the model gives them, which a type that a map
  /// leaves out keeps.
  std::vector<double> typeSigma, typeEpsilon;
  /// For each pair of types (`a * T + b`), whether its σ and ε are an
  /// override of the combining rule in the model (an NBFIX), which keeps
  /// its values when per-type σ or ε are tunable.
  std::vector<bool> fixedPairs;
  /// Whether each pair term's tail is in the correction for the
  /// dispersion at the values of the compile (D209), which an update keeps.
  std::vector<bool> pairTails;
  bool empty() const { return tunables.empty(); }
  /// The place of the tunable `name`, or -1.
  int find(llvm::StringRef name) const {
    for (size_t k = 0; k != tunables.size(); ++k)
      if (tunables[k].name == name)
        return static_cast<int>(k);
    return -1;
  }
};

/// Input order, nm and nm/ps. Empty velocities stay absent until execution.
struct InitialState {
  std::vector<double> positions, velocities;
  driver::Cell cell;
};
/// Physics only; topology coordinate/cell fields must remain empty/zero.
/// These are value objects. A prepared run takes a deep copy.
struct System {
  driver::Topology topology;
  Format format = Format::Amber;
  bool periodic = true;
  double cutoff = 1.2, pairlistDistance = 1.35, switchDistance = 1.0;
  driver::Truncation truncation = driver::Truncation::Switch;
  Electrostatics electrostatics = Electrostatics::Cutoff;
  /// For PME only; as the control file, none by default.
  CoulombModifier coulombModifier = CoulombModifier::None;
  /// The correction for the dispersion beyond the cutoff, as
  /// `[energy] dispersion_correction` (D209). `dispersionGiven` says
  /// whether it was set, as whether the control file gives the key
  /// (D[python-dispersion]): a pair term whose tail diverges is refused
  /// when it was, and left out with the warning `pair_tail_left_out` when
  /// not; without a periodic cell the default is off.
  driver::DispersionCorrection dispersion = driver::DispersionCorrection::EnergyPressure;
  bool dispersionGiven = false;
  double pmeAlpha = 0, pmeTolerance = 1.e-5, pmeSpacing = 0.12;
  std::array<int64_t, 3> pmeGrid = {0,0,0};
  int64_t pmeOrder = 4;
  bool rigidHydrogenBonds = false, rigidWater = false;
  /// Explicitly permit flexible water when importing GROMACS SETTLE.
  bool flexibleWater = false;
  std::vector<std::string> waterResidues;
  /// Expressions yield kJ/mol, r is nm, theta radians. Pair terms in this
  /// subset accept constants only; tuple parameters have one value/tuple.
  /// A pair term's `dispersion`, when `dispersionGiven`, is that of its
  /// `dispersion_correction`: None leaves it out of the correction, and
  /// EnergyPressure asks for its tail (D209, D[python-dispersion]).
  std::vector<driver::PairTerm> pairTerms;
  std::vector<driver::TupleTerm> tupleTerms;
  /// One `[[restraints]]` table (D74, D124), with its constant in
  /// kJ/mol/nm^2 (D198).
  struct Restraint {
    std::string selection;
    double forceConstant = 0;
    driver::ReferenceScaling scaling = driver::ReferenceScaling::Center;
  };
  std::vector<Restraint> restraints;
  /// The reference of the restraints, (N, 3) nm in input order, the
  /// coordinates file of the control file; empty: the positions of the
  /// initial state that is prepared.
  std::vector<double> restraintReference;
  /// The tunable parameters (D213).
  std::vector<Tunable> tunables;
};
struct LoadedData {
  driver::Topology topology;
  Format format;
  /// Files actually read, including active includes, and their owned bytes.
  std::vector<std::pair<std::string, std::string>> sources;
  System makeSystem() const;
  InitialState makeState() const;
};
llvm::Expected<LoadedData> loadAmber(llvm::StringRef topology, llvm::StringRef coordinates);
llvm::Expected<LoadedData> loadGromacs(llvm::StringRef topology, llvm::StringRef coordinates,
                                     llvm::ArrayRef<std::string> includes = {},
                                     llvm::ArrayRef<std::string> defines = {});
/// CHARMM CRD has no cell. Supply a reduced cell in nm for periodic input;
/// positions are rotated from the symmetric frame exactly as in the CLI.
/// A zero cell means nonperiodic input, with no rotation.
llvm::Expected<LoadedData> loadCharmm(llvm::StringRef structure,
                                    llvm::StringRef coordinates,
                                    llvm::ArrayRef<std::string> parameters,
                                    const driver::Cell &cell = {});
struct Integrator {
  driver::Integrator method = driver::Integrator::VelocityVerlet;
  double timestep = 0.001;
  bool minimize = false;
  double minimizeStep = 0.01;
};
struct Ensemble {
  EnsembleKind kind = EnsembleKind::NVE;
  double temperature = 298.15, tauT = 1.0;
  double pressure = 1.01325, tauP = 5.0, compressibility = 4.5e-5;
  int64_t couplingPeriod = 10, comPeriod = 0;
  uint64_t seed = 314159;
};
struct Execution {
  driver::Target target = driver::Target::CPU;
  driver::Precision precision = driver::Precision::Double;
  /// Logical device under CUDA_VISIBLE_DEVICES, resolved by the next service.
  int64_t device = 0;
  int64_t threads = 1;
  bool deterministic = false, reorder = true, fastMath = true;
};
/// A temporary fixed schedule for shared-builder parity. No runtime ownership,
/// JIT, output writers or public Python simulation is implemented here.
struct Schedule { int64_t steps = 100, energyPeriod = 10; };
struct PreparedModel {
  Execution execution;
  driver::Control control;
  driver::System system;
  /// The tunables, whose values `control` and `system` hold.
  TunableSet tunables;
  llvm::Expected<driver::Program> build() const;
};
/// Resolves the tunables of `model` against its prepared `control` and
/// `system`, and puts their initial values into them (D213).
llvm::Expected<TunableSet> resolveTunables(const System &model,
                                           driver::Control &control,
                                           driver::System &system);
/// Puts the values `values` of the tunables `set` into `control` and
/// `system` (a copy of the topology), and collects the tails of the pair
/// terms anew; an InputError if a value is not allowed or a pair term's
/// tail enters or leaves the correction for the dispersion
/// (D213).
llvm::Error applyTunables(const TunableSet &set,
                          const std::vector<std::vector<double>> &values,
                          driver::Control &control, driver::System &system);
/// Checks new values of the tunables `set`: one array of the right shape
/// for each, finite, σ and ε not negative.
llvm::Error checkTunableValues(const TunableSet &set,
                               const std::vector<std::vector<double>> &values);
llvm::Expected<PreparedModel> prepare(const System &, const InitialState &,
                                     const Integrator &, const Ensemble &,
                                     const Execution &, const Schedule &);
/// A copy of `state` with the velocities that `mdir run` draws for the same
/// system, temperature (K), and seed: `system` is prepared as `prepare`
/// prepares it (NVE at `temperature`), then driver::assignVelocities draws.
llvm::Expected<InitialState> drawVelocities(const System &system,
                                            const InitialState &state,
                                            double temperature, uint64_t seed);
} // namespace model
} // namespace mdir
#endif

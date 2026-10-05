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
enum class EnsembleKind { NVE, NVT, NPT };

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
  driver::DispersionCorrection dispersion = driver::DispersionCorrection::EnergyPressure;
  double pmeAlpha = 0, pmeTolerance = 1.e-5, pmeSpacing = 0.12;
  std::array<int64_t, 3> pmeGrid = {0,0,0};
  int64_t pmeOrder = 4;
  bool rigidHydrogenBonds = false, rigidWater = false;
  /// Explicitly permit flexible water when importing GROMACS SETTLE.
  bool flexibleWater = false;
  std::vector<std::string> waterResidues;
  /// Expressions yield kJ/mol, r is nm, theta radians. Pair terms in this
  /// subset accept constants only; tuple parameters have one value/tuple.
  std::vector<driver::PairTerm> pairTerms;
  std::vector<driver::TupleTerm> tupleTerms;
  /// One `[[restraints]]` table (D74, D124), with its constant in
  /// kJ/mol/nm^2 (D[python-velocities-restraints]).
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
  llvm::Expected<driver::Program> build() const;
};
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

#include "mdir/Driver/Model.h"
#include "mdir/Driver/Fingerprint.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cmath>
#include <limits>
#include <set>
using namespace mdir;
using namespace mdir::model;
char ModelError::ID;
static llvm::Error input(const llvm::Twine &s) {
  return llvm::make_error<ModelError>(ModelError::Input, s.str());
}
static llvm::Error unsupported(const llvm::Twine &s) {
  return llvm::make_error<ModelError>(ModelError::Unsupported, s.str());
}
static llvm::Error typed(llvm::Error e) {
  return input(llvm::toString(std::move(e)));
}
/// A message of the builder about the tail of a pair term, in the words of
/// the Python model (D222).
static std::string pythonWords(std::string message) {
  static const std::string key = "give 'dispersion_correction = \"NONE\"' in the term";
  for (size_t at = message.find(key); at != std::string::npos; at = message.find(key, at))
    message.replace(at, key.size(), "set PairTerm.dispersion to DispersionCorrection.None_");
  return message;
}

System LoadedData::makeSystem() const {
  System s;
  s.topology = topology;
  s.format = format;
  s.topologyFilesHash = topologyFilesHash;
  s.topology.positions.clear();
  s.topology.velocities.clear();
  for (int k = 0; k != 3; ++k)
    s.topology.box[k] = s.topology.tilt[k] = 0;
  return s;
}
InitialState LoadedData::makeState() const {
  InitialState s;
  s.positions = topology.positions;
  s.velocities = topology.velocities;
  for (int k = 0; k != 3; ++k) {
    s.cell.diagonal[k] = topology.box[k];
    s.cell.tilt[k] = topology.tilt[k];
  }
  return s;
}
static llvm::Expected<LoadedData> loaded(driver::Topology t, Format f,
                                         llvm::StringRef path, llvm::StringRef coords) {
  if (llvm::Error e = driver::validateTopology(t))
    return typed(std::move(e));
  LoadedData data{std::move(t), f, {}};
  std::vector<std::string> files = data.topology.sourceFiles;
  for (llvm::StringRef file : {path, coords})
    if (!llvm::is_contained(files, file.str())) files.push_back(file.str());
  for (const auto &file : files) {
    auto buffer = llvm::MemoryBuffer::getFile(file);
    if (!buffer) return input("cannot capture source '" + file + "'");
    data.sources.emplace_back(file, (*buffer)->getBuffer().str());
  }
  // The files of the topology as `mdir run` hashes them for its
  // fingerprint (D172): the topology, then the files it read, but not the
  // coordinates (D223).
  std::vector<std::string> topologyFiles = {path.str()};
  for (const auto &file : data.topology.sourceFiles)
    if (!llvm::is_contained(topologyFiles, file))
      topologyFiles.push_back(file);
  std::vector<std::string> contents;
  for (const auto &file : topologyFiles)
    for (const auto &[name, bytes] : data.sources)
      if (name == file) {
        contents.push_back(bytes);
        break;
      }
  data.topologyFilesHash = driver::getTopologyFilesHash(contents);
  return data;
}
llvm::Expected<LoadedData> mdir::model::loadAmber(llvm::StringRef path, llvm::StringRef coords) {
  auto t = driver::readAmberTopology(path);
  if (!t) return typed(t.takeError());
  if (llvm::Error e = driver::readAmberCoordinates(coords, *t)) return typed(std::move(e));
  return loaded(std::move(*t), Format::Amber, path, coords);
}
llvm::Expected<LoadedData> mdir::model::loadGromacs(
    llvm::StringRef path, llvm::StringRef coords,
    llvm::ArrayRef<std::string> includes, llvm::ArrayRef<std::string> defines) {
  auto t = driver::readGromacsTopology(path, includes, defines);
  if (!t) return typed(t.takeError());
  if (llvm::Error e = driver::readGromacsCoordinates(coords, *t)) return typed(std::move(e));
  return loaded(std::move(*t), Format::Gromacs, path, coords);
}

llvm::Expected<LoadedData> mdir::model::loadCharmm(
    llvm::StringRef path, llvm::StringRef coords,
    llvm::ArrayRef<std::string> parameters, const driver::Cell &cell) {
  auto t = driver::readCharmmTopology(path, parameters);
  if (!t) return typed(t.takeError());
  if (llvm::Error e = driver::readCharmmCoordinates(coords, *t)) return typed(std::move(e));
  bool zero = true;
  for (int k = 0; k != 3; ++k) {
    if (!std::isfinite(cell.diagonal[k]) || !std::isfinite(cell.tilt[k]))
      return input("CHARMM cell entries must be finite");
    zero &= cell.diagonal[k] == 0 && cell.tilt[k] == 0;
  }
  if (!zero) {
    for (double edge : cell.diagonal)
      if (!(edge > 0)) return input("CHARMM periodic cell needs three positive diagonals");
    auto reduced = cell;
    if (llvm::Error e = driver::reduceCell(reduced)) return typed(std::move(e));
    if (reduced.diagonal != cell.diagonal || reduced.tilt != cell.tilt)
      return input("CHARMM periodic cell must be reduced");
    driver::applyCharmmCell(*t, cell);
  }
  t->sourceFiles.assign(parameters.begin(), parameters.end());
  return loaded(std::move(*t), Format::Charmm, path, coords);
}

// Coordinate and energy conversion at the existing builder boundary.
// Token replacement preserves parameter names and scientific exponents.
static std::string convertExpression(llvm::StringRef expr, bool distance,
                                     bool position = false) {
  std::string out = "(";
  while (!expr.empty()) {
    if (llvm::isAlpha(expr.front()) || expr.front() == '_') {
      size_t n = 1;
      while (n < expr.size() && (llvm::isAlnum(expr[n]) || expr[n] == '_')) ++n;
      llvm::StringRef word = expr.take_front(n);
      bool length = (distance && word == "r") ||
                    (position && (word == "x" || word == "y" || word == "z"));
      out += length ? "(" + word.str() + "*0.1)" : word.str();
      expr = expr.drop_front(n);
    } else {
      out += expr.front(); expr = expr.drop_front();
    }
  }
  size_t definitions = out.find(';');
  if (definitions == std::string::npos) return out + ")/4.184";
  return out.substr(0, definitions) + ")/4.184" + out.substr(definitions);
}
static bool validParameter(llvm::StringRef name) {
  if (name.empty() || !(llvm::isAlpha(name.front()) || name.front() == '_') ||
      driver::Expression::isFunction(name)) return false;
  return llvm::all_of(name, [](char ch) { return llvm::isAlnum(ch) || ch == '_'; });
}
static llvm::Error checkExpression(llvm::StringRef name, llvm::StringRef text,
                                   llvm::StringRef coordinate,
                                   const std::set<std::string> &parameters,
                                   const std::set<std::string> &variables = {}) {
  if (name.empty()) return input("a custom term needs a name");
  llvm::StringRef definitions = text.split(';').second;
  while (!definitions.empty()) {
    auto part = definitions.split(';');
    llvm::StringRef local = part.first.split('=').first.trim();
    if (local == "r" || local == "theta" || parameters.count(local.str()) ||
        variables.count(local.str()))
      return input("term '" + name + "': local definition shadows supplied name '" + local + "'");
    definitions = part.second;
  }
  auto e = driver::Expression::parse(text);
  if (!e) return input("term '" + name + "': " + llvm::toString(e.takeError()));
  for (const auto &n : e->getNames())
    if (n != coordinate && !parameters.count(n) && !variables.count(n))
      return input("term '" + name + "': undeclared parameter '" + n + "'");
  return llvm::Error::success();
}
llvm::Expected<PreparedModel> mdir::model::prepare(
    const System &s, const InitialState &state, const Integrator &integrator,
    const Ensemble &ensemble, const Execution &execution, const Schedule &schedule) {
  driver::Control c;
  c.inMemoryTopology = true;
  c.inMemoryCharmm = s.format == Format::Charmm;
  if (s.format != Format::Amber && s.format != Format::Gromacs && s.format != Format::Charmm)
    return unsupported("unsupported topology format");
  if (execution.target != driver::Target::CPU && execution.target != driver::Target::GPU)
    return unsupported("unsupported execution target");
  if (execution.device < 0 || (execution.target == driver::Target::CPU && execution.device != 0))
    return input("device must be nonnegative; CPU execution uses device 0");
  if (execution.precision != driver::Precision::Mixed && execution.precision != driver::Precision::Double)
    return unsupported("the initial object subset supports mixed and double precision");
  if (integrator.method != driver::Integrator::VelocityVerlet && integrator.method != driver::Integrator::Leapfrog)
    return unsupported("the initial object subset supports velocity Verlet and leapfrog");
  if (ensemble.kind != EnsembleKind::NVE && ensemble.kind != EnsembleKind::NVT && ensemble.kind != EnsembleKind::NPT)
    return unsupported("unsupported ensemble");
  if (ensemble.barostatCoupling != BarostatCoupling::Isotropic && ensemble.barostatCoupling != BarostatCoupling::SemiIsotropic &&
      ensemble.barostatCoupling != BarostatCoupling::Anisotropic)
    return unsupported("unsupported coupling of the barostat");
  if (ensemble.barostatWork != driver::BarostatWork::Trotter && ensemble.barostatWork != driver::BarostatWork::TrotterFirstOrder &&
      ensemble.barostatWork != driver::BarostatWork::Exact && ensemble.barostatWork != driver::BarostatWork::FirstOrder)
    return unsupported("unsupported work of the barostat");
  if (s.electrostatics != Electrostatics::Cutoff && s.electrostatics != Electrostatics::PME)
    return unsupported("unsupported electrostatics");
  // As the control file: the modifier is of the real-space term of PME.
  if (s.coulombModifier != CoulombModifier::None && s.coulombModifier != CoulombModifier::PotentialShift)
    return input("unknown Coulomb modifier");
  if (s.coulombModifier != CoulombModifier::None && s.electrostatics != Electrostatics::PME)
    return input("the Coulomb modifier is for PME electrostatics");
  if (s.truncation != driver::Truncation::None && s.truncation != driver::Truncation::Shift &&
      s.truncation != driver::Truncation::Switch && s.truncation != driver::Truncation::ForceSwitch &&
      s.truncation != driver::Truncation::PowerForceSwitch && s.truncation != driver::Truncation::SquaredDistanceSwitch)
    return unsupported("unsupported truncation");
  if (s.dispersion != driver::DispersionCorrection::None && s.dispersion != driver::DispersionCorrection::EnergyPressure)
    return unsupported("unsupported dispersion correction");
  auto positive = [](double v) { return std::isfinite(v) && v > 0; };
  const double pairlistDistance = s.getPairlistDistance(), switchDistance = s.getSwitchDistance();
  if (!positive(s.cutoff) || !positive(pairlistDistance) || pairlistDistance <= s.cutoff ||
      !std::isfinite(switchDistance) || switchDistance < 0 ||
      (s.truncation != driver::Truncation::None && s.truncation != driver::Truncation::Shift && switchDistance >= s.cutoff))
    return input("cutoff/list/switch distances are inconsistent or nonfinite");
  if (!positive(integrator.timestep) || !positive(integrator.minimizeStep) ||
      (!std::isfinite(ensemble.temperature) || ensemble.temperature < 0 ||
       (ensemble.kind != EnsembleKind::NVE && ensemble.temperature == 0)) || !positive(ensemble.tauT) || !positive(ensemble.tauP) ||
      !positive(ensemble.compressibility) || !std::isfinite(ensemble.pressure))
    return input("integrator and bath parameters must be finite, with positive time scales and temperature");
  if (schedule.steps < 0 || schedule.energyPeriod < 0 || execution.threads < 1 ||
      ensemble.couplingPeriod < 1 || ensemble.comPeriod.value_or(0) < 0)
    return input("invalid step, thread or coupling count");
  if (execution.neighborCapacity < 0)
    return input("Execution.neighbor_capacity must be positive, or 0 for the estimate");
  if (integrator.minimize && ensemble.kind != EnsembleKind::NVE)
    return input("minimization takes no bath");
  // The program of a minimization loops over the intervals between its
  // energies, as [minimize] of the control file does (emitMinimization).
  if (integrator.minimize && schedule.energyPeriod == 0)
    return input("a minimization writes energies: Schedule.energy_period may not be 0");
  if (integrator.minimize && schedule.steps == 0)
    return input("a minimization takes steps: Schedule.steps may not be 0");
  if (s.rigidWater && s.flexibleWater)
    return input("rigid and flexible water are mutually exclusive");
  if (!s.topology.positions.empty() || !s.topology.velocities.empty())
    return input("positions and velocities belong to InitialState, not System");
  for (int k = 0; k != 3; ++k)
    if (s.topology.box[k] != 0 || s.topology.tilt[k] != 0)
      return input("the cell belongs to InitialState, not System");
  if (!s.topology.tupleTerms.empty() || !s.topology.externalTerms.empty())
    return unsupported("imported expression terms are outside the initial subset; use typed model terms");
  // As the control file: without a periodic cell the correction is off
  // unless it is asked for (D222).
  driver::DispersionCorrection dispersion =
      s.periodic || s.dispersionGiven ? s.dispersion : driver::DispersionCorrection::None;
  // With a switch the default correction is off, with a warning: the
  // switch takes part of the potential below the cutoff, which the
  // correction would leave out (D210, D222).
  bool switched = !s.dispersionGiven && dispersion != driver::DispersionCorrection::None &&
                  s.truncation != driver::Truncation::None &&
                  s.truncation != driver::Truncation::Shift;
  if (switched) dispersion = driver::DispersionCorrection::None;
  if (!s.periodic && (s.electrostatics == Electrostatics::PME || ensemble.kind == EnsembleKind::NPT ||
                     dispersion != driver::DispersionCorrection::None))
    return input("PME, pressure coupling and dispersion correction require periodic boundaries");
  // The orders of the control file, with its words.
  if (s.pmeOrder != 4 && s.pmeOrder != 6 && s.pmeOrder != 8)
    return input("model: expected 4, 6, or 8 for 'order'");
  if (s.pmeInfluence != PMEInfluence::SPME && s.pmeInfluence != PMEInfluence::Optimal)
    return unsupported("unsupported influence function of PME");
  if (!positive(s.pmeSpacing) || !std::isfinite(s.pmeAlpha) || s.pmeAlpha < 0 ||
      !positive(s.pmeTolerance) || s.pmeTolerance >= 1)
    return input("invalid PME settings");
  bool anyGrid = false, allGrid = true;
  for (auto n : s.pmeGrid) { anyGrid |= n != 0; allGrid &= n >= 8; }
  if (anyGrid && !allGrid) return input("PME grid must be automatic or have three positive dimensions at least the spline order");

  c.periodic = s.periodic;
  c.cutoffDistance = s.cutoff / driver::units::length;
  // Not given, the two are the control file's to the bit: 1.5 Å beyond its
  // cutoff, and its cutoff (D[python-defaults]).
  c.pairlistDistance = s.pairlistDistance ? *s.pairlistDistance / driver::units::length
                                          : c.cutoffDistance + 1.5;
  c.switchDistance = s.switchDistance ? *s.switchDistance / driver::units::length
                                      : c.cutoffDistance;
  // A dual list, with the refusals and the words of the control file
  // (Reader::readEnergy; D114, D245).
  if (!std::isfinite(s.prunedDistance) || s.prunedDistance < 0)
    return input("System.pruned_distance must be positive, or 0 for one list");
  c.prunedDistance = s.prunedDistance / driver::units::length;
  if (c.prunedDistance != 0.0 &&
      !(c.prunedDistance > c.cutoffDistance && c.prunedDistance < c.pairlistDistance))
    return input("model: 'pruned_distance' is not between 'cutoff' and "
                 "'pairlist_distance'");
  c.truncation = s.truncation;
  c.topologyDispersion = dispersion;
  c.topologyDispersionGiven = s.dispersionGiven;
  // As the control file (D210): the correction takes a plain cutoff or the
  // shift; a switch would leave out what it removes below the cutoff.
  if (s.dispersionGiven && dispersion != driver::DispersionCorrection::None &&
      s.truncation != driver::Truncation::None && s.truncation != driver::Truncation::Shift)
    return input("the correction for the dispersion needs a plain cutoff or the shift "
                 "(Truncation.None_ or Truncation.Shift); with a switch, set "
                 "System.dispersion to DispersionCorrection.None_");
  c.pme = s.electrostatics == Electrostatics::PME;
  c.pmeShift = s.coulombModifier == CoulombModifier::PotentialShift;
  c.pmeAlpha = s.pmeAlpha * driver::units::length;
  c.pmeAlphaTolerance = s.pmeTolerance;
  c.pmeMaxSpacing = s.pmeSpacing / driver::units::length;
  c.pmeOrder = s.pmeOrder;
  c.pmeOptimal = s.pmeInfluence == PMEInfluence::Optimal;
  c.analyticBonds = s.analyticBonds;
  for (int k = 0; k != 3; ++k) c.pmeGrid[k] = s.pmeGrid[k];
  c.rigidBonds = s.rigidHydrogenBonds;
  c.fastWater = s.rigidWater;
  c.statesFlexible = s.flexibleWater;
  c.settleResidues = s.waterResidues;
  c.integrator = integrator.method;
  c.minimize = integrator.minimize;
  c.minimizeStep = integrator.minimizeStep / driver::units::length;
  c.timestep = integrator.timestep;
  c.numSteps = schedule.steps;
  c.energyPeriod = schedule.energyPeriod;
  c.thermostat = ensemble.kind != EnsembleKind::NVE;
  c.barostat = ensemble.kind == EnsembleKind::NPT;
  c.temperature = ensemble.temperature;
  c.tauT = ensemble.tauT;
  c.pressure = ensemble.pressure / 1.01325;
  c.tauP = ensemble.tauP;
  c.compressibility = ensemble.compressibility * 1.01325;
  // The coupling and the work of the barostat, as Reader::readBarostat
  // takes them (D[python-barostat]): one compressibility, that of every
  // axis and of z, no surface tension. They are of a barostat: without
  // one they are not read, as `tau_p` is not.
  if (c.barostat) {
    c.barostatWork = ensemble.barostatWork;
    c.semiIsotropic = ensemble.barostatCoupling == BarostatCoupling::SemiIsotropic;
    c.anisotropic = ensemble.barostatCoupling == BarostatCoupling::Anisotropic;
    for (double &value : c.compressibilities) value = c.compressibility;
    c.compressibilityZ = c.compressibility;
    // A compressibility of each axis, with the words of the control file.
    if (!ensemble.compressibilities.empty()) {
      if (!c.anisotropic)
        return input("model: a 'compressibility' of each axis needs 'coupling = "
                     "\"ANISOTROPIC\"'; give one number");
      bool any = false;
      if (ensemble.compressibilities.size() != 3)
        return input("Ensemble.compressibility: expected one number, or three that are not "
                     "negative, those of x, y, and z, in 1/bar");
      for (size_t k = 0; k != 3; ++k) {
        double value = ensemble.compressibilities[k];
        if (!std::isfinite(value) || value < 0.0)
          return input("Ensemble.compressibility: expected one number, or three that are not "
                       "negative, those of x, y, and z, in 1/bar");
        c.compressibilities[k] = value * 1.01325;
        any |= value > 0.0;
      }
      if (!any)
        return input("model: a barostat whose compressibilities are all 0 keeps the "
                     "cell; give one that is not 0");
    }
    // The keys of semi-isotropic coupling (D119).
    if (!c.semiIsotropic) {
      const char *key = ensemble.compressibilityZ ? "compressibility_z"
                        : ensemble.surfaceTension != 0.0 ? "surface_tension"
                        : ensemble.surfaces != 2 ? "surfaces" : nullptr;
      if (key)
        return input("model: '" + std::string(key) + "' needs 'coupling = \"SEMI_ISOTROPIC\"'");
    }
    if (ensemble.compressibilityZ) {
      if (!std::isfinite(*ensemble.compressibilityZ) || *ensemble.compressibilityZ < 0.0)
        return input("model: expected 0 or a positive number for 'compressibility_z'");
      c.compressibilityZ = *ensemble.compressibilityZ * 1.01325;
    }
    if (!std::isfinite(ensemble.surfaceTension))
      return input("Ensemble.surface_tension must be finite");
    if (ensemble.surfaces < 1)
      return input("Ensemble.surfaces must be at least 1");
    c.surfaceTension = ensemble.surfaceTension / driver::units::dynePerCmToBarNm;
    c.surfaces = ensemble.surfaces;
    if ((c.semiIsotropic || c.anisotropic) && c.barostatWork == driver::BarostatWork::FirstOrder)
      return input("model: 'work = \"FIRST_ORDER\"' counts the work from the trace of "
                   "the virial of the step with twice the internal kinetic "
                   "energy, which holds for the trace only; with "
                   "'coupling = \"SEMI_ISOTROPIC\"' or \"ANISOTROPIC\" use "
                   "\"TROTTER\", "
                   "\"TROTTER_FIRST_ORDER\", or \"EXACT\"");
  }
  c.thermostatPeriod = c.thermostat ? ensemble.couplingPeriod : 0;
  c.barostatPeriod = c.barostat ? ensemble.couplingPeriod : 0;
  // Not given: as the control file without `center_of_mass_interval`,
  // which `resolveControlCoupling` settles, with the thermostat or never
  // (D[python-defaults]); a minimization removes nothing.
  c.comPeriod = ensemble.comPeriod ? *ensemble.comPeriod : integrator.minimize ? 0 : -1;
  c.seed = ensemble.seed;
  c.target = execution.target;
  c.precision = execution.precision;
  c.threads = execution.threads;
  c.deterministic = execution.deterministic;
  c.reorder = execution.reorder;
  c.fastMath = execution.fastMath;
  c.neighborWidth = execution.neighborCapacity;
  // The groups, with the refusal and the words of the control file
  // (Reader::readExecution; D89, D245).
  if (execution.neighborStructure != driver::NeighborStructure::Matrix &&
      execution.neighborStructure != driver::NeighborStructure::Groups)
    return unsupported("unsupported neighbor structure");
  c.neighborStructure = execution.neighborStructure;
  if (c.neighborStructure == driver::NeighborStructure::Groups &&
      (c.target != driver::Target::GPU || c.deterministic))
    return input("model: 'neighbor_structure = \"GROUPS\"' needs 'target = "
                 "\"GPU\"' and not 'deterministic'");
  std::set<std::string> termNames;
  // `observe` of a term (D189, D232): the columns follow the
  // pair terms, the tuple terms, and the terms of the positions, each in
  // its order; the control checks the constants as it does those of the
  // control file.
  auto observe = [&](const std::string &term, bool observed,
                     const std::vector<std::string> &constants) -> llvm::Error {
    if (!observed) return llvm::Error::success();
    int64_t line = static_cast<int64_t>(c.observables.size());
    c.observables.push_back({term, "", line});
    std::set<std::string> seen;
    for (const std::string &name : constants) {
      if (name.empty()) return input("the term '" + term + "' observes a constant without a name");
      if (!seen.insert(name).second)
        return input("the term '" + term + "': '" + name + "' is observed twice");
      c.observables.push_back({term, name, line});
    }
    return llvm::Error::success();
  };
  for (auto term : s.pairTerms) {
    if (!termNames.insert(term.name).second) return input("duplicate custom term name");
    if (!term.mixing.empty())
      return unsupported("custom pairs initially support constants, not mixing");
    // The term's own correction, as `dispersion_correction` of a pair term
    // with a topology (D209, D222): None leaves it out, and
    // EnergyPressure follows the system's, which must be on.
    if (term.dispersionGiven && term.dispersion != driver::DispersionCorrection::None &&
        term.dispersion != driver::DispersionCorrection::EnergyPressure)
      return unsupported("unsupported dispersion correction of the pair term '" + term.name + "'");
    if (!term.dispersionGiven) term.dispersion = driver::DispersionCorrection::None;
    if (term.dispersion != driver::DispersionCorrection::None &&
        dispersion == driver::DispersionCorrection::None)
      return input("the pair term '" + term.name + "' asks for the correction for the "
                   "dispersion, which follows that of the system, which is off; a term can "
                   "only leave it with DispersionCorrection.None_");
    if (!term.groups.empty() && term.groups.size() != 2)
      return input("custom pair groups must contain two selections");
    std::set<std::string> names;
    for (auto &[n,v] : term.constants)
      if (!validParameter(n) || n == "r" || n == "coulomb" || !names.insert(n).second || !std::isfinite(v))
        return input("invalid custom pair parameter name or value");
    if (llvm::Error e = checkExpression(term.name, term.expression, "r", names)) return std::move(e);
    term.expression = convertExpression(term.expression, true);
    if (llvm::Error e = observe(term.name, term.observed, term.observe)) return std::move(e);
    c.pairs.push_back(std::move(term));
  }
  for (auto term : s.tupleTerms) {
    if (!termNames.insert(term.name).second) return input("duplicate custom term name");
    if (term.isCentroid() || term.isCompound() || !term.centers.empty())
      return unsupported("centroid and compound terms follow later");
    if (term.arity < 2 || term.arity > 4 || term.particles.size() % term.arity)
      return input("custom tuples need arity 2, 3 or 4 and complete tuples");
    std::set<std::string> names;
    for (auto &[n,v] : term.parameters)
      if (!validParameter(n) || n == "r" || n == "theta" || n == term.getVariable() || !names.insert(n).second || v.size() != term.size() ||
          llvm::any_of(v, [](double x) { return !std::isfinite(x); }))
        return input("invalid custom tuple parameter name, shape or value");
    if (llvm::Error e = checkExpression(term.name, term.expression, term.getVariable(), names)) return std::move(e);
    for (size_t k = 0; k != term.particles.size(); k += term.arity) {
      std::set<unsigned> members;
      for (unsigned j = 0; j != term.arity; ++j)
        if (term.particles[k+j] >= s.topology.getNumParticles() || !members.insert(term.particles[k+j]).second)
          return input("invalid custom tuple particle identities");
    }
    term.expression = convertExpression(term.expression, term.arity == 2);
    if (llvm::Error e = observe(term.name, term.observed, term.observe)) return std::move(e);
    c.tupleTerms.push_back(std::move(term));
  }
  // The terms of the absolute positions (D148, D233), which
  // the system resolves as it does those of the control file.
  for (auto term : s.externalTerms) {
    if (!termNames.insert(term.name).second) return input("duplicate custom term name");
    if (!term.parameters.empty())
      return unsupported("the external term '" + term.name + "': a parameter with a value "
                         "for each particle follows later; give constants");
    if (term.selection.empty() == term.particles.empty())
      return input("the external term '" + term.name + "' takes 'selection', a mask of "
                   "particles, or 'particles', their indices, and not both");
    std::set<unsigned> members;
    for (unsigned particle : term.particles)
      if (particle >= s.topology.getNumParticles() || !members.insert(particle).second)
        return input("the external term '" + term.name + "': invalid particle identities");
    static const std::set<std::string> variables = {"x", "y", "z", "q"};
    std::set<std::string> names;
    for (auto &[n,v] : term.constants)
      if (!validParameter(n) || variables.count(n) || n == "t" || !names.insert(n).second || !std::isfinite(v))
        return input("invalid external term parameter name or value");
    if (llvm::Error e = checkExpression(term.name, term.expression, "", names, variables)) return std::move(e);
    if (c.barostat && term.scaling == driver::ExternalTerm::Scaling::Unset)
      return input("the external term '" + term.name + "' needs 'scaling' under a barostat: "
                   "ExternalScaling.None_ keeps it fixed in space while the cell scales about "
                   "the origin, ExternalScaling.Cell takes its positions in the frame of the "
                   "cell, scaled to the cell of the initial state");
    term.expression = convertExpression(term.expression, false, true);
    if (llvm::Error e = observe(term.name, term.observed, term.observe)) return std::move(e);
    c.externalTerms.push_back(std::move(term));
  }
  // The restraints become those of the control structure, whose constants
  // are in kJ/mol/nm^2, and are prepared by the same code (D74, D124).
  for (const auto &r : s.restraints) {
    if (r.selection.empty()) return input("a restraint needs a selection");
    if (!positive(r.forceConstant))
      return input("the restraint of '" + r.selection +
                   "': the force constant must be positive and finite (kJ/mol/nm^2)");
    if (r.scaling != driver::ReferenceScaling::Center && r.scaling != driver::ReferenceScaling::All)
      return input("the restraint of '" + r.selection + "': unknown reference scaling");
    c.restraints.push_back({r.selection, r.forceConstant, r.scaling});
  }
  if (!s.restraintReference.empty() &&
      (s.restraintReference.size() != 3 * s.topology.getNumParticles() ||
       llvm::any_of(s.restraintReference, [](double x) { return !std::isfinite(x); })))
    return input("the restraint reference needs finite (N, 3) positions of every particle");
  // As `mdir run`, which refuses `observables` under [minimize]: the
  // columns are those of the energies of a run of dynamics (D189).
  if (c.minimize && !c.observables.empty())
    return input("the term '" + c.observables.front().term + "' gives 'observe', which is "
                 "evaluated at the energies of a run of dynamics; a minimization does not "
                 "take it: set its observe to None for the program that minimizes");
  if (!c.minimize)
    if (llvm::Error e = driver::resolveControlCoupling(c, "model")) return typed(std::move(e));
  if (llvm::Error e = driver::validateControl(c, "model")) return typed(std::move(e));
  driver::Topology topology = s.topology;
  topology.positions = state.positions;
  topology.velocities = state.velocities;
  auto cell = state.cell;
  if (s.periodic) {
    for (double edge : cell.diagonal)
      if (!positive(edge)) return input("periodic cell needs positive finite diagonal entries");
    if (llvm::Error e = driver::reduceCell(cell)) return typed(std::move(e));
    if (cell.diagonal != state.cell.diagonal || cell.tilt != state.cell.tilt)
      return input("InitialState cell must already be reduced");
  }
  for (int k = 0; k != 3; ++k) {
    topology.box[k] = cell.diagonal[k]; topology.tilt[k] = cell.tilt[k];
  }
  auto prepared = driver::prepareTopologySystem(c, std::move(topology), s.format != Format::Gromacs);
  if (!prepared) return input(pythonWords(llvm::toString(prepared.takeError())));
  for (auto &[code, message] : prepared->warnings)
    if (code == "pair_tail_left_out") message = pythonWords(std::move(message));
  // The reference of the restraints, as the CLI takes the positions of its
  // coordinates file; the cell it scales from is that of the state.
  if (switched)
    prepared->warnings.push_back(
        {"dispersion_switched",
         "the switch of the truncation turns the correction for the dispersion off; it "
         "takes a plain cutoff or the shift (Truncation.None_ or Truncation.Shift). Set "
         "System.dispersion to DispersionCorrection.None_ to say so"});
  prepared->referencePositions =
      s.restraintReference.empty() ? prepared->positions : s.restraintReference;
  // A state without velocities takes those that `mdir run` draws when its
  // coordinates give none: at the temperature of the ensemble, with its
  // seed (D[python-defaults]). A minimization begins at rest, as there.
  // Velocities that are given, zeros among them, are kept.
  if (state.velocities.empty() && !c.minimize)
    driver::assignVelocities(c, *prepared);
  // The tunable parameters, whose initial values the prepared model takes
  // (D213).
  auto tunables = resolveTunables(s, c, *prepared);
  if (!tunables) return tunables.takeError();
  // An observed constant may be a tunable with one entry that every site
  // takes: its value reaches the program as a value of its entry
  // (D232). The derivative in a tunable with several entries
  // is that of D230.
  for (const TunableSet::Entry &entry : tunables->tunables)
    for (const driver::Control::Observable &observable : c.observables)
      if (!entry.term.empty() && entry.term == observable.term &&
          entry.parameter == observable.constant) {
        if (entry.entries != 1 || llvm::any_of(entry.map, [](int64_t m) { return m != 0; }))
          return input("the term '" + entry.term + "' observes '" + entry.parameter +
                       "', which the tunable '" + entry.name + "' takes with several entries "
                       "or leaves to some of its sites; an observed constant is one number for "
                       "the whole term. The derivative in each entry at a state is "
                       "Simulation.tunables.gradient()['" + entry.name +
                       "'] (System.tunable_gradient), of the same potential");
        c.observedTunables.push_back({entry.term, entry.parameter});
      }
  return PreparedModel{execution, std::move(c), std::move(*prepared), std::move(*tunables)};
}
llvm::Expected<InitialState> mdir::model::drawVelocities(
    const System &s, const InitialState &state, double temperature, uint64_t seed) {
  if (!std::isfinite(temperature) || temperature < 0)
    return input("the temperature of drawn velocities must be finite and at least 0 K");
  if (seed > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return input("the seed must be from 0 to 2^63-1, as [dynamics] seed");
  Ensemble ensemble;
  ensemble.kind = EnsembleKind::NVE;
  ensemble.temperature = temperature;
  ensemble.seed = seed;
  // The draw does not read the neighbor structures: the system is prepared
  // for the default execution, which has the matrix, so without the dual
  // list, which is of the groups (D245). A value that
  // `compile` would refuse is refused there. The copy of the system is
  // the price of preparing it as `prepare` does.
  System physics = s;
  physics.prunedDistance = 0.0;
  auto prepared = prepare(physics, state, Integrator{}, ensemble, Execution{}, Schedule{});
  if (!prepared) return prepared.takeError();
  // The draw of `mdir run` when its coordinates give no velocities.
  driver::assignVelocities(prepared->control, prepared->system);
  InitialState result = state;
  result.velocities = prepared->system.velocities;
  return result;
}
llvm::Expected<driver::Program> PreparedModel::build() const {
  auto result = driver::buildProgram(control, system);
  if (!result) return typed(result.takeError());
  return result;
}

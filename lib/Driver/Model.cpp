#include "mdir/Driver/Model.h"
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

System LoadedData::makeSystem() const {
  System s;
  s.topology = topology;
  s.format = format;
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
static std::string convertExpression(llvm::StringRef expr, bool distance) {
  std::string out = "(";
  while (!expr.empty()) {
    if (llvm::isAlpha(expr.front()) || expr.front() == '_') {
      size_t n = 1;
      while (n < expr.size() && (llvm::isAlnum(expr[n]) || expr[n] == '_')) ++n;
      llvm::StringRef word = expr.take_front(n);
      out += distance && word == "r" ? "(r*0.1)" : word.str();
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
                                   const std::set<std::string> &parameters) {
  if (name.empty()) return input("a custom term needs a name");
  llvm::StringRef definitions = text.split(';').second;
  while (!definitions.empty()) {
    auto part = definitions.split(';');
    llvm::StringRef local = part.first.split('=').first.trim();
    if (local == "r" || local == "theta" || parameters.count(local.str()))
      return input("term '" + name + "': local definition shadows supplied name '" + local + "'");
    definitions = part.second;
  }
  auto e = driver::Expression::parse(text);
  if (!e) return input("term '" + name + "': " + llvm::toString(e.takeError()));
  for (const auto &n : e->getNames())
    if (n != coordinate && !parameters.count(n))
      return input("term '" + name + "': undeclared parameter '" + n + "'");
  return llvm::Error::success();
}
// A restraint constant in kJ/mol/nm^2 as the control file's kcal/mol/A^2:
// the value that prepareTopologySystem converts back to `k` exactly where
// one exists (always for a constant converted from a control file), or else
// the nearest of the neighbors tried.
static double toControlConstant(double k) {
  auto back = [](double c) {
    return c * driver::units::energy / (driver::units::length * driver::units::length);
  };
  double guess = k * (driver::units::length * driver::units::length) / driver::units::energy;
  double best = guess;
  for (double direction : {-HUGE_VAL, HUGE_VAL}) {
    double c = guess;
    for (int step = 0; step != 3; ++step, c = std::nextafter(c, direction))
      if (std::abs(back(c) - k) < std::abs(back(best) - k)) best = c;
  }
  return best;
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
  if (s.electrostatics != Electrostatics::Cutoff && s.electrostatics != Electrostatics::PME)
    return unsupported("unsupported electrostatics");
  if (s.truncation != driver::Truncation::None && s.truncation != driver::Truncation::Shift &&
      s.truncation != driver::Truncation::Switch && s.truncation != driver::Truncation::ForceSwitch &&
      s.truncation != driver::Truncation::PowerForceSwitch && s.truncation != driver::Truncation::SquaredDistanceSwitch)
    return unsupported("unsupported truncation");
  if (s.dispersion != driver::DispersionCorrection::None && s.dispersion != driver::DispersionCorrection::EnergyPressure)
    return unsupported("unsupported dispersion correction");
  auto positive = [](double v) { return std::isfinite(v) && v > 0; };
  if (!positive(s.cutoff) || !positive(s.pairlistDistance) || s.pairlistDistance <= s.cutoff ||
      !std::isfinite(s.switchDistance) || s.switchDistance < 0 ||
      (s.truncation != driver::Truncation::None && s.truncation != driver::Truncation::Shift && s.switchDistance >= s.cutoff))
    return input("cutoff/list/switch distances are inconsistent or nonfinite");
  if (!positive(integrator.timestep) || !positive(integrator.minimizeStep) ||
      (!std::isfinite(ensemble.temperature) || ensemble.temperature < 0 ||
       (ensemble.kind != EnsembleKind::NVE && ensemble.temperature == 0)) || !positive(ensemble.tauT) || !positive(ensemble.tauP) ||
      !positive(ensemble.compressibility) || !std::isfinite(ensemble.pressure))
    return input("integrator and bath parameters must be finite, with positive time scales and temperature");
  if (schedule.steps < 0 || schedule.energyPeriod < 0 || execution.threads < 1 ||
      ensemble.couplingPeriod < 1 || ensemble.comPeriod < 0)
    return input("invalid step, thread or coupling count");
  if (integrator.minimize && ensemble.kind != EnsembleKind::NVE)
    return input("minimization takes no bath");
  if (s.rigidWater && s.flexibleWater)
    return input("rigid and flexible water are mutually exclusive");
  if (!s.topology.positions.empty() || !s.topology.velocities.empty())
    return input("positions and velocities belong to InitialState, not System");
  for (int k = 0; k != 3; ++k)
    if (s.topology.box[k] != 0 || s.topology.tilt[k] != 0)
      return input("the cell belongs to InitialState, not System");
  if (!s.topology.tupleTerms.empty() || !s.topology.externalTerms.empty())
    return unsupported("imported expression terms are outside the initial subset; use typed model terms");
  if (!s.periodic && (s.electrostatics == Electrostatics::PME || ensemble.kind == EnsembleKind::NPT ||
                     s.dispersion != driver::DispersionCorrection::None))
    return input("PME, pressure coupling and dispersion correction require periodic boundaries");
  if (s.pmeOrder != 4 || !positive(s.pmeSpacing) || !std::isfinite(s.pmeAlpha) || s.pmeAlpha < 0 ||
      !positive(s.pmeTolerance) || s.pmeTolerance >= 1)
    return input("invalid PME settings (the initial subset uses order 4)");
  bool anyGrid = false, allGrid = true;
  for (auto n : s.pmeGrid) { anyGrid |= n != 0; allGrid &= n >= 8; }
  if (anyGrid && !allGrid) return input("PME grid must be automatic or have three positive dimensions at least the spline order");

  c.periodic = s.periodic;
  c.cutoffDistance = s.cutoff / driver::units::length;
  c.pairlistDistance = s.pairlistDistance / driver::units::length;
  c.switchDistance = s.switchDistance / driver::units::length;
  c.truncation = s.truncation;
  c.topologyDispersion = s.dispersion;
  c.pme = s.electrostatics == Electrostatics::PME;
  c.pmeAlpha = s.pmeAlpha * driver::units::length;
  c.pmeAlphaTolerance = s.pmeTolerance;
  c.pmeMaxSpacing = s.pmeSpacing / driver::units::length;
  c.pmeOrder = s.pmeOrder;
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
  c.thermostatPeriod = c.thermostat ? ensemble.couplingPeriod : 0;
  c.barostatPeriod = c.barostat ? ensemble.couplingPeriod : 0;
  c.comPeriod = ensemble.comPeriod;
  c.seed = ensemble.seed;
  c.target = execution.target;
  c.precision = execution.precision;
  c.threads = execution.threads;
  c.deterministic = execution.deterministic;
  c.reorder = execution.reorder;
  c.fastMath = execution.fastMath;
  std::set<std::string> termNames;
  for (auto term : s.pairTerms) {
    if (!termNames.insert(term.name).second) return input("duplicate custom term name");
    if (!term.mixing.empty() || term.dispersion != driver::DispersionCorrection::None)
      return unsupported("custom pairs initially support constants and no dispersion correction");
    if (!term.groups.empty() && term.groups.size() != 2)
      return input("custom pair groups must contain two selections");
    std::set<std::string> names;
    for (auto &[n,v] : term.constants)
      if (!validParameter(n) || n == "r" || n == "coulomb" || !names.insert(n).second || !std::isfinite(v))
        return input("invalid custom pair parameter name or value");
    if (llvm::Error e = checkExpression(term.name, term.expression, "r", names)) return std::move(e);
    term.expression = convertExpression(term.expression, true);
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
    c.tupleTerms.push_back(std::move(term));
  }
  // The restraints become those of the control file, whose constants are
  // in kcal/mol/A^2, and are prepared by the same code (D74, D124).
  for (const auto &r : s.restraints) {
    if (r.selection.empty()) return input("a restraint needs a selection");
    if (!positive(r.forceConstant))
      return input("the restraint of '" + r.selection +
                   "': the force constant must be positive and finite (kJ/mol/nm^2)");
    if (r.scaling != driver::ReferenceScaling::Center && r.scaling != driver::ReferenceScaling::All)
      return input("the restraint of '" + r.selection + "': unknown reference scaling");
    c.restraints.push_back({r.selection, toControlConstant(r.forceConstant), r.scaling});
  }
  if (!s.restraintReference.empty() &&
      (s.restraintReference.size() != 3 * s.topology.getNumParticles() ||
       llvm::any_of(s.restraintReference, [](double x) { return !std::isfinite(x); })))
    return input("the restraint reference needs finite (N, 3) positions of every particle");
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
  if (!prepared) return typed(prepared.takeError());
  // The reference of the restraints, as the CLI takes the positions of its
  // coordinates file; the cell it scales from is that of the state.
  prepared->referencePositions =
      s.restraintReference.empty() ? prepared->positions : s.restraintReference;
  return PreparedModel{execution, std::move(c), std::move(*prepared)};
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
  auto prepared = prepare(s, state, Integrator{}, ensemble, Execution{}, Schedule{});
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

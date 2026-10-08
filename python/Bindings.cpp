// Owned Python inputs; collections cross the boundary by value.
#include "mdir/Compiler/Compile.h"
#include "mdir/Compiler/Simulation.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <optional>
#include <set>
namespace py = pybind11;
using namespace mdir;
struct InputError : std::runtime_error { using std::runtime_error::runtime_error; };
struct UnsupportedError : std::runtime_error { using std::runtime_error::runtime_error; };
struct CompileError : std::runtime_error { using std::runtime_error::runtime_error; };
struct StaleProgramError : std::runtime_error { using std::runtime_error::runtime_error; };
struct SimulationError : std::runtime_error { using std::runtime_error::runtime_error; };
[[noreturn]] static void raise(llvm::Error error) {
  bool unsupported = false, compile = false, simulation = false;
  std::string message;
  llvm::handleAllErrors(std::move(error),
    [&](const model::ModelError &e) {
      unsupported = e.kind == model::ModelError::Unsupported; message = e.message;
    }, [&](const compiler::CompileError &e) {
      compile = true; message = e.diagnostic;
    }, [&](const compiler::SimulationError &e) {
      simulation = true; message = e.message;
    }, [&](const llvm::ErrorInfoBase &e) {
      llvm::raw_string_ostream os(message); e.log(os);
    });
  if (unsupported) throw UnsupportedError(message);
  if (compile) throw CompileError(message);
  if (simulation) throw SimulationError(message);
  throw InputError(message);
}
template <class T> static T unwrap(llvm::Expected<T> value) {
  if (!value) raise(value.takeError());
  return std::move(*value);
}
#include "Units.h"
#include "HostArrays.h"
#include "Tunables.h"
#include "DLPack.h"
#include "Topology.h"

struct Version {
  uint64_t version = 0;
  /// The attributes given explicitly, which the fingerprint of a checkpoint
  /// records (D223).
  std::set<std::string> given;
  virtual ~Version() = default;
};
template <class T> struct Input : Version { T value; std::optional<size_t> particleCount; };
template <class T, class V>
static void property(py::class_<Input<T>, std::shared_ptr<Input<T>>> &c,
                     const char *name, V T::*member) {
  c.def_property(name, [member](const Input<T> &o) { return o.value.*member; },
    [member, name](Input<T> &o, V value) {
      o.value.*member = std::move(value); ++o.version; o.given.insert(name);
    });
}
/// A real-valued property in `unit`, which also takes a unit quantity.
template <class T>
static void property(py::class_<Input<T>, std::shared_ptr<Input<T>>> &c,
                     const char *name, double T::*member, units::Unit unit) {
  std::string qualified = py::cast<std::string>(c.attr("__name__")) + "." + name;
  c.def_property(name, [member](const Input<T> &o) { return o.value.*member; },
    [member, qualified, unit, name](Input<T> &o, py::object value) {
      o.value.*member = units::scalar(value, qualified, unit); ++o.version;
      o.given.insert(name);
    });
}
template <class T> static auto input(py::module_ &m, const char *name) {
  return py::class_<Input<T>, std::shared_ptr<Input<T>>>(m, name).def(py::init<>());
}
struct Program {
  /// The built program and its pipeline; not lowered (D224).
  compiler::CompiledProgram compiled;
  /// The lowered text, made on the first read of `lowered_ir`, under the
  /// mutex, by the thread that reads it first.
  struct Lowering {
    std::mutex mutex;
    std::optional<std::string> text;
  };
  std::shared_ptr<Lowering> lowering = std::make_shared<Lowering>();
  /// What a simulation compiles its programs from (D196).
  std::shared_ptr<const model::PreparedModel> prepared;
  /// The code that its simulations left, which later ones take
  /// (D[program-reuse]); freed with the program.
  std::shared_ptr<compiler::CodeStore> code =
      std::make_shared<compiler::CodeStore>();
  /// Whether its compilations use the compile cache, the default of its
  /// simulations (D217).
  bool cache = true;
  std::vector<std::pair<std::shared_ptr<Version>, uint64_t>> inputs;
  template <class T> void track(const std::shared_ptr<Input<T>> &o) {
    inputs.emplace_back(o, o->version);
  }
  bool stale() const {
    for (const auto &o : inputs) if (o.first->version != o.second) return true;
    return false;
  }
  void checkCurrent() const {
    if (stale()) throw StaleProgramError("compile inputs changed; explicitly recompile the program");
  }
};
// Runs `call` with the GIL released. Signals reach Python between parts; a
// KeyboardInterrupt ends the call there and is raised once the GIL is held
// again.
template <class Call> static int64_t withSignals(Call call) {
  std::optional<py::error_already_set> interrupt;
  std::optional<llvm::Expected<int64_t>> taken;
  {
    py::gil_scoped_release release;
    taken.emplace(call([&] {
      py::gil_scoped_acquire acquire;
      if (PyErr_CheckSignals() != 0) {
        interrupt.emplace();
        return true;
      }
      return false;
    }));
  }
  if (interrupt) {
    if (!*taken) llvm::consumeError(taken->takeError());
    throw *interrupt;
  }
  return unwrap(std::move(*taken));
}
/// Reads a checkpoint for a simulation (D223): without
/// HDF5 an UnsupportedError, and a file that is not one, of a newer format,
/// or whose hashes differ an InputError with the pointer to `.prev` that
/// `mdir run` gives.
static driver::Checkpoint readCheckpointFile(const std::string &path) {
  if (!driver::hasCheckpointSupport())
    throw UnsupportedError("this build of MDIR has no HDF5, which checkpoints need");
  auto read = driver::readCheckpoint(path);
  if (!read)
    throw InputError(llvm::toString(read.takeError()) + "; '" +
                     driver::getPreviousCheckpointPath(path) +
                     "' holds the checkpoint before it, if there is one");
  return std::move(*read);
}
static py::array_t<double> vectors(const std::vector<double> &values) {
  return host::copy(values.data(), values.size(),
                    {static_cast<py::ssize_t>(values.size() / 3), 3});
}
/// `observe` of a term (D189, D232): None, the term is not
/// observed, or the names of the constants whose derivatives are, with its
/// energy.
template <class Term> static py::object observeOf(const Term &term) {
  if (!term.observed) return py::none();
  return py::cast(term.observe);
}
template <class Term>
static void setObserve(Term &term, py::object value, const char *name) {
  if (value.is_none()) {
    term.observed = false;
    term.observe.clear();
    return;
  }
  std::string message = std::string(name) + ".observe takes None or a list of the names "
                        "of constants of the term";
  if (py::isinstance<py::str>(value) || !py::isinstance<py::sequence>(value))
    throw py::type_error(message);
  std::vector<std::string> names;
  for (py::handle item : value) {
    if (!py::isinstance<py::str>(item)) throw py::type_error(message);
    names.push_back(item.cast<std::string>());
  }
  term.observed = true;
  term.observe = std::move(names);
}
PYBIND11_MODULE(_core, m) {
  // Required at import as well as configuration, including installed modules.
  try {
    auto numpy = py::module_::import("numpy");
    auto version = numpy.attr("lib").attr("NumpyVersion")(numpy.attr("__version__"));
    if (!py::cast<bool>(version.attr("__ge__")("1.23.0")))
      throw py::import_error("The MDIR Python interface requires NumPy >=1.23");
  } catch (const py::error_already_set &e) {
    throw py::import_error(std::string("The MDIR Python interface requires NumPy >=1.23: ") + e.what());
  }
  m.attr("__version__") = MDIR_VERSION;
  py::register_exception<InputError>(m, "InputError", PyExc_ValueError);
  py::register_exception<UnsupportedError>(m, "UnsupportedError");
  py::register_exception<CompileError>(m, "CompileError");
  py::register_exception<StaleProgramError>(m, "StaleProgramError");
  py::register_exception<SimulationError>(m, "SimulationError", PyExc_RuntimeError);
  py::enum_<driver::Target>(m, "Target")
    .value("CPU", driver::Target::CPU)
    .value("GPU", driver::Target::GPU)
    ;
  py::enum_<driver::Precision>(m, "Precision")
    .value("Single", driver::Precision::Single)
    .value("Mixed", driver::Precision::Mixed)
    .value("Double", driver::Precision::Double)
    ;
  py::enum_<driver::Integrator>(m, "IntegratorMethod")
    .value("VelocityVerlet", driver::Integrator::VelocityVerlet)
    .value("Leapfrog", driver::Integrator::Leapfrog)
    .value("Brownian", driver::Integrator::Brownian)
    ;
  py::enum_<model::EnsembleKind>(m, "EnsembleKind")
    .value("NVE", model::EnsembleKind::NVE)
    .value("NVT", model::EnsembleKind::NVT)
    .value("NPT", model::EnsembleKind::NPT)
    ;
  py::enum_<model::Electrostatics>(m, "Electrostatics")
    .value("Cutoff", model::Electrostatics::Cutoff)
    .value("PME", model::Electrostatics::PME)
    ;
  py::enum_<model::CoulombModifier>(m, "CoulombModifier")
    .value("None_", model::CoulombModifier::None)
    .value("PotentialShift", model::CoulombModifier::PotentialShift)
    ;
  py::enum_<driver::Truncation>(m, "Truncation")
    .value("None_", driver::Truncation::None)
    .value("Shift", driver::Truncation::Shift)
    .value("Switch", driver::Truncation::Switch)
    .value("ForceSwitch", driver::Truncation::ForceSwitch)
    .value("PowerForceSwitch", driver::Truncation::PowerForceSwitch)
    .value("SquaredDistanceSwitch", driver::Truncation::SquaredDistanceSwitch)
    ;
  py::enum_<driver::DispersionCorrection>(m, "DispersionCorrection")
    .value("None_", driver::DispersionCorrection::None)
    .value("EnergyPressure", driver::DispersionCorrection::EnergyPressure)
    ;
  py::enum_<model::Format>(m, "Format")
    .value("Amber", model::Format::Amber)
    .value("Gromacs", model::Format::Gromacs)
    .value("Charmm", model::Format::Charmm)
    ;
  auto cell = py::class_<driver::Cell>(m, "Cell").def(py::init<>());
  for (auto item : {std::make_pair("diagonal", &driver::Cell::diagonal),
                    std::make_pair("tilt", &driver::Cell::tilt)}) {
    auto member = item.second;
    std::string name = std::string("Cell.") + item.first;
    cell.def_property(item.first, [member](const driver::Cell &c) {
      return host::copy((c.*member).data(), 3, {3});
    }, [member, name](driver::Cell &c, py::object input) {
      auto values = host::doubles(input, name, 3, {}, units::nm);
      std::copy(values.begin(), values.end(), (c.*member).begin());
    });
  }
  cell.def_property_readonly("vectors", [](const driver::Cell &c) {
    auto vectors = c.getVectors();
    std::array<double, 9> values;
    for (size_t i = 0; i < 3; ++i)
      std::copy(vectors[i].begin(), vectors[i].end(), values.begin() + 3 * i);
    return host::copy(values.data(), values.size(), {3, 3});
  });
  py::class_<driver::PairTerm>(m, "PairTerm").def(py::init<>())
    .def_readwrite("name", &driver::PairTerm::name)
    .def_readwrite("expression", &driver::PairTerm::expression)
    .def_readwrite("constants", &driver::PairTerm::constants)
    .def_readwrite("groups", &driver::PairTerm::groups)
    .def_property("observe", [](const driver::PairTerm &t) { return observeOf(t); },
                  [](driver::PairTerm &t, py::object value) { setObserve(t, value, "PairTerm"); })
    // `dispersion_correction` of the term (D209, D222):
    // None (Python's) follows the system, DispersionCorrection.None_ leaves
    // the term out of the correction, EnergyPressure asks for its tail.
    .def_property("dispersion", [](const driver::PairTerm &t) -> py::object {
      if (!t.dispersionGiven) return py::none();
      return py::cast(t.dispersion);
    }, [](driver::PairTerm &t, py::object value) {
      if (!value.is_none() && !py::isinstance<driver::DispersionCorrection>(value))
        throw py::type_error("PairTerm.dispersion takes a DispersionCorrection or None");
      t.dispersionGiven = !value.is_none();
      t.dispersion = t.dispersionGiven ? value.cast<driver::DispersionCorrection>()
                                       : driver::DispersionCorrection::None;
    });
  py::class_<driver::TupleTerm>(m, "TupleTerm").def(py::init<>())
    .def_readwrite("name", &driver::TupleTerm::name)
    .def_readwrite("expression", &driver::TupleTerm::expression)
    .def_readwrite("arity", &driver::TupleTerm::arity)
    .def_property("particles", [](const driver::TupleTerm &t) { return host::particles(t); },
                  [](driver::TupleTerm &t, py::object value) { host::particles(t, value); })
    .def_property("parameters", [](const driver::TupleTerm &t) { return host::parameters(t); },
                  [](driver::TupleTerm &t, py::sequence value) { host::parameters(t, value); })
    .def_property("observe", [](const driver::TupleTerm &t) { return observeOf(t); },
                  [](driver::TupleTerm &t, py::object value) { setObserve(t, value, "TupleTerm"); });
  // Terms of the absolute positions, `[[energy.external]]` (D148,
  // D233): x, y, z in nm, the charge q, kJ/mol.
  py::enum_<driver::ExternalTerm::Scaling>(m, "ExternalScaling")
    .value("None_", driver::ExternalTerm::Scaling::None)
    .value("Cell", driver::ExternalTerm::Scaling::Cell)
    ;
  py::class_<driver::ExternalTerm>(m, "ExternalTerm").def(py::init<>())
    .def_readwrite("name", &driver::ExternalTerm::name)
    .def_readwrite("expression", &driver::ExternalTerm::expression)
    .def_readwrite("selection", &driver::ExternalTerm::selection)
    .def_readwrite("constants", &driver::ExternalTerm::constants)
    .def_property("particles", [](const driver::ExternalTerm &t) {
      std::vector<int64_t> values(t.particles.begin(), t.particles.end());
      return host::copy(values.data(), values.size(), {static_cast<py::ssize_t>(values.size())});
    }, [](driver::ExternalTerm &t, py::object source) {
      const std::string name = "ExternalTerm.particles";
      auto values = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(source);
      if (!values || values.ndim() != 1)
        throw InputError(name + ": expected a one-dimensional array of particle indices, from 0");
      std::vector<unsigned> particles;
      for (py::ssize_t k = 0; k != values.shape(0); ++k) {
        int64_t value = values.at(k);
        if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<unsigned>::max())
          throw InputError(name + ": a particle index is negative or exceeds the native range");
        particles.push_back(static_cast<unsigned>(value));
      }
      t.particles = std::move(particles);
    })
    // How the term follows a barostat (D154): None (Python's) leaves it
    // unset, which a model with a barostat refuses.
    .def_property("scaling", [](const driver::ExternalTerm &t) -> py::object {
      if (t.scaling == driver::ExternalTerm::Scaling::Unset) return py::none();
      return py::cast(t.scaling);
    }, [](driver::ExternalTerm &t, py::object value) {
      if (!value.is_none() && !py::isinstance<driver::ExternalTerm::Scaling>(value))
        throw py::type_error("ExternalTerm.scaling takes an ExternalScaling or None");
      t.scaling = value.is_none() ? driver::ExternalTerm::Scaling::Unset
                                  : value.cast<driver::ExternalTerm::Scaling>();
    })
    .def_property("observe", [](const driver::ExternalTerm &t) { return observeOf(t); },
                  [](driver::ExternalTerm &t, py::object value) { setObserve(t, value, "ExternalTerm"); });
  // Tunable parameters (D213).
  tunables::bindTunable(m);
  // Read-only topology views (D221).
  topology::bind(m);
  // Restraints (D74, D124; D198).
  py::enum_<driver::ReferenceScaling>(m, "ReferenceScaling")
    .value("Center", driver::ReferenceScaling::Center)
    .value("All", driver::ReferenceScaling::All)
    ;
  py::class_<model::System::Restraint>(m, "Restraint")
    .def(py::init<>())
    .def(py::init([](std::string selection, py::object forceConstant,
                     driver::ReferenceScaling scaling) {
      return model::System::Restraint{
          std::move(selection),
          units::scalar(forceConstant, "Restraint.force_constant", units::springConstant), scaling};
    }), py::arg("selection"), py::arg("force_constant"),
        py::arg("reference_scaling") = driver::ReferenceScaling::Center)
    .def_readwrite("selection", &model::System::Restraint::selection)
    .def_property("force_constant",
                  [](const model::System::Restraint &r) { return r.forceConstant; },
                  [](model::System::Restraint &r, py::object value) {
                    r.forceConstant = units::scalar(value, "Restraint.force_constant",
                                                    units::springConstant);
                  })
    .def_readwrite("reference_scaling", &model::System::Restraint::scaling);
  auto system = input<model::System>(m, "System");
  property(system, "periodic", &model::System::periodic);
  property(system, "cutoff", &model::System::cutoff, units::nm);
  property(system, "pairlist_distance", &model::System::pairlistDistance, units::nm);
  property(system, "switch_distance", &model::System::switchDistance, units::nm);
  property(system, "truncation", &model::System::truncation);
  property(system, "electrostatics", &model::System::electrostatics);
  property(system, "coulomb_modifier", &model::System::coulombModifier);
  // Setting the correction makes it explicit, as the key of the control
  // file; None restores the default (D222).
  system.def_property("dispersion", [](const Input<model::System> &o) {
    return o.value.dispersion;
  }, [](Input<model::System> &o, py::object value) {
    if (!value.is_none() && !py::isinstance<driver::DispersionCorrection>(value))
      throw py::type_error("System.dispersion takes a DispersionCorrection or None");
    o.value.dispersionGiven = !value.is_none();
    o.value.dispersion = o.value.dispersionGiven
                             ? value.cast<driver::DispersionCorrection>()
                             : driver::DispersionCorrection::EnergyPressure;
    ++o.version;
  });
  system.def_property_readonly("dispersion_given", [](const Input<model::System> &o) {
    return o.value.dispersionGiven;
  });
  property(system, "pme_alpha", &model::System::pmeAlpha, units::inverseNm);
  property(system, "pme_tolerance", &model::System::pmeTolerance, units::none);
  property(system, "pme_spacing", &model::System::pmeSpacing, units::nm);
  property(system, "pme_grid", &model::System::pmeGrid);
  property(system, "pme_order", &model::System::pmeOrder);
  property(system, "rigid_hydrogen_bonds", &model::System::rigidHydrogenBonds);
  property(system, "rigid_water", &model::System::rigidWater);
  property(system, "flexible_water", &model::System::flexibleWater);
  property(system, "water_residues", &model::System::waterResidues);
  property(system, "pair_terms", &model::System::pairTerms);
  property(system, "tuple_terms", &model::System::tupleTerms);
  property(system, "external_terms", &model::System::externalTerms);
  property(system, "restraints", &model::System::restraints);
  property(system, "tunables", &model::System::tunables);
  // Whether the program carries the derivative of the energy in the
  // tunables (D230).
  property(system, "tunable_gradient", &model::System::tunableGradient);
  system.def_property("restraint_reference", [](const Input<model::System> &o) {
    const auto &v = o.value.restraintReference;
    return host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(v.size() / 3), 3});
  }, [](Input<model::System> &o, py::object source) {
    const std::string name = "System.restraint_reference";
    auto values = host::doubles(source, name, {}, 3, units::nm);
    size_t count = values.size() / 3, particles = o.value.topology.getNumParticles();
    if (count != 0 && count != particles)
      throw InputError(name + ": expected shape (" + std::to_string(particles) +
                       ", 3) or (0, 3); found (" + std::to_string(count) + ", 3)");
    o.value.restraintReference = std::move(values);
    ++o.version;
  });
  auto initialstate = input<model::InitialState>(m, "InitialState");
  for (auto item : {std::make_pair("positions", &model::InitialState::positions),
                    std::make_pair("velocities", &model::InitialState::velocities)}) {
    auto member = item.second;
    bool velocity = member == &model::InitialState::velocities;
    std::string name = std::string("InitialState.") + item.first;
    initialstate.def_property(item.first, [member](const Input<model::InitialState> &o) {
      const auto &v = o.value.*member;
      return host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(v.size() / 3), 3});
    }, [member, velocity, name](Input<model::InitialState> &o, py::object source) {
      auto values = host::doubles(source, name, {}, 3, velocity ? units::nmPerPs : units::nm);
      size_t count = values.size() / 3;
      if (o.particleCount && count != *o.particleCount && !(velocity && count == 0))
        throw InputError(name + ": expected shape (" + std::to_string(*o.particleCount) +
                         ", 3); found (" + std::to_string(count) + ", 3)");
      o.value.*member = std::move(values);
      if (!o.particleCount && count) o.particleCount = count;
      ++o.version;
    });
  }
  property(initialstate, "cell", &model::InitialState::cell);
  // The velocities that `mdir run` draws for the same system, temperature,
  // and seed, in a new state (D198).
  initialstate.def("draw_velocities", [](const Input<model::InitialState> &o,
                                         std::shared_ptr<Input<model::System>> system,
                                         py::object temperatureValue, py::int_ seed) {
    if (!system) throw InputError("draw_velocities takes a System");
    double temperature = units::scalar(temperatureValue, "draw_velocities: temperature",
                                       units::kelvin);
    int overflow = 0;
    long long value = PyLong_AsLongLongAndOverflow(seed.ptr(), &overflow);
    if (overflow || value < 0)
      throw InputError("draw_velocities: the seed must be from 0 to 2^63-1, as [dynamics] seed");
    auto result = std::make_shared<Input<model::InitialState>>();
    result->value = unwrap(model::drawVelocities(system->value, o.value, temperature,
                                                 static_cast<uint64_t>(value)));
    result->particleCount = o.particleCount;
    if (!result->particleCount && !o.value.positions.empty())
      result->particleCount = o.value.positions.size() / 3;
    return result;
  }, py::arg("system"), py::arg("temperature"), py::arg("seed"));
  // A new initial state from the state that a simulation reached: copies of
  // its positions, its cell, and, unless velocities=False or it has none, its
  // velocities. A new stage begins there; the baths, the step, and the
  // random streams are not carried (a continuation is a checkpoint's, #132).
  initialstate.def_static("from_state", [](const compiler::SimulationState &state,
                                           bool velocities) {
    auto result = std::make_shared<Input<model::InitialState>>();
    result->value.positions = state.positions;
    if (velocities && !state.velocities.empty()) {
      if (state.velocityOffset != 0.0)
        throw InputError("InitialState.from_state: the velocities of this state are " +
                         std::to_string(state.velocityOffset) +
                         " of a step from its positions (leapfrog); an initial state takes "
                         "velocities at the time of its positions. Give velocities=False "
                         "and draw them, or continue the simulation itself");
      result->value.velocities = state.velocities;
    }
    for (int k = 0; k != 3; ++k) {
      result->value.cell.diagonal[k] = state.box[k];
      result->value.cell.tilt[k] = state.tilt[k];
    }
    if (!state.positions.empty())
      result->particleCount = state.positions.size() / 3;
    return result;
  }, py::arg("state"), py::arg("velocities") = true);
  auto integrator = input<model::Integrator>(m, "Integrator");
  property(integrator, "method", &model::Integrator::method);
  property(integrator, "timestep", &model::Integrator::timestep, units::ps);
  property(integrator, "minimize", &model::Integrator::minimize);
  property(integrator, "minimize_step", &model::Integrator::minimizeStep, units::nm);
  auto ensemble = input<model::Ensemble>(m, "Ensemble");
  property(ensemble, "kind", &model::Ensemble::kind);
  property(ensemble, "temperature", &model::Ensemble::temperature, units::kelvin);
  property(ensemble, "tau_t", &model::Ensemble::tauT, units::ps);
  property(ensemble, "pressure", &model::Ensemble::pressure, units::bar);
  property(ensemble, "tau_p", &model::Ensemble::tauP, units::ps);
  property(ensemble, "compressibility", &model::Ensemble::compressibility, units::inverseBar);
  property(ensemble, "coupling_period", &model::Ensemble::couplingPeriod);
  property(ensemble, "com_period", &model::Ensemble::comPeriod);
  property(ensemble, "seed", &model::Ensemble::seed);
  auto execution = input<model::Execution>(m, "Execution");
  property(execution, "target", &model::Execution::target);
  property(execution, "precision", &model::Execution::precision);
  property(execution, "device", &model::Execution::device);
  property(execution, "threads", &model::Execution::threads);
  property(execution, "deterministic", &model::Execution::deterministic);
  property(execution, "reorder", &model::Execution::reorder);
  property(execution, "fast_math", &model::Execution::fastMath);
  property(execution, "neighbor_capacity", &model::Execution::neighborCapacity);
  auto schedule = input<model::Schedule>(m, "Schedule");
  property(schedule, "steps", &model::Schedule::steps);
  property(schedule, "energy_period", &model::Schedule::energyPeriod);
  system.def_property_readonly("particle_count", [](const Input<model::System> &s) {
    return s.value.topology.masses.size();
  });
  system.def_property_readonly("topology", [](const Input<model::System> &s) {
    return topology::of(s.value.topology, false);
  });
  py::class_<model::LoadedData>(m, "LoadedData")
    .def_property_readonly("format", [](const model::LoadedData &d) { return d.format; })
    .def_property_readonly("sources", [](const model::LoadedData &d) { return d.sources; })
    .def_property_readonly("topology", [](const model::LoadedData &d) {
      return topology::of(d.topology, false);
    })
    .def("make_system", [](const model::LoadedData &d) {
      auto result = std::make_shared<Input<model::System>>(); result->value = d.makeSystem(); return result;
    })
    .def("make_state", [](const model::LoadedData &d) {
      auto result = std::make_shared<Input<model::InitialState>>(); result->value = d.makeState();
      result->particleCount = d.topology.getNumParticles(); return result;
    });
  m.def("load_amber", [](const std::string &top, const std::string &coordinates) {
    return unwrap(model::loadAmber(top, coordinates));
  }, py::arg("topology"), py::arg("coordinates"));
  m.def("load_gromacs", [](const std::string &top, const std::string &coordinates,
                            const std::vector<std::string> &includes,
                            const std::vector<std::string> &defines) {
    return unwrap(model::loadGromacs(top, coordinates, includes, defines));
  }, py::arg("topology"), py::arg("coordinates"),
     py::arg("includes") = std::vector<std::string>{}, py::arg("defines") = std::vector<std::string>{});
  m.def("load_charmm", [](const std::string &structure, const std::string &coordinates,
                           const std::vector<std::string> &parameters, const driver::Cell &cell) {
    return unwrap(model::loadCharmm(structure, coordinates, parameters, cell));
  }, py::arg("structure"), py::arg("coordinates"), py::arg("parameters"), py::arg("cell") = driver::Cell{});
  py::class_<Program, std::shared_ptr<Program>>(m, "Program")
    .def_property_readonly("ir", [](const Program &p) { return p.compiled.program.module; })
    .def_property_readonly("lowered_ir", [](const Program &p) {
      // Lowered when first asked for, with the GIL released: a simulation
      // lowers programs of its own (D224).
      std::optional<llvm::Error> failed;
      {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(p.lowering->mutex);
        if (!p.lowering->text) {
          auto lowered = compiler::lowerToText(p.prepared->control, p.compiled);
          if (lowered)
            p.lowering->text = std::move(*lowered);
          else
            failed.emplace(lowered.takeError());
        }
      }
      if (failed) raise(std::move(*failed));
      return *p.lowering->text;
    })
    .def_property_readonly("pipeline", [](const Program &p) { return p.compiled.pipeline; })
    .def_property_readonly("stale", &Program::stale)
    // The topology as compiled, with its constraints resolved.
    .def_property_readonly("topology", [](const Program &p) {
      return topology::of(*p.prepared->system.topology, true);
    })
    .def("check_current", &Program::checkCurrent)
    .def_property_readonly("plan", [](const Program &p) {
      py::dict d;
      const auto &c = p.compiled;
      d["target"] = c.execution.target; d["precision"] = c.execution.precision;
      d["device"] = c.execution.device; d["threads"] = c.execution.threads;
      d["deterministic"] = c.execution.deterministic;
      d["reorders"] = c.program.reorders;
      d["neighbor_capacity"] = c.program.neighborWidth;
      d["entry"] = c.program.entry;
      d["state_dtype"] = c.program.state == driver::Element::F64 ? "float64" : "float32";
      d["force_dtype"] = c.program.force == driver::Element::F64 ? "float64" : "float32";
      d["pme"] = c.program.pme;
      d["pme_grid"] = std::array<int64_t, 3>{c.program.pmeGrid[0], c.program.pmeGrid[1], c.program.pmeGrid[2]};
      d["tunables"] = tunables::describe(p.prepared->tunables, c.program);
      // The correction for the dispersion (D209): whether it is on and was
      // given, and for each pair term whether its tail is in it
      // (D222).
      const auto &control = p.prepared->control;
      const auto &tails = p.prepared->system.pairTails;
      py::dict dispersion, terms;
      bool on = control.topologyDispersion != driver::DispersionCorrection::None;
      dispersion["correction"] = control.topologyDispersion;
      dispersion["given"] = control.topologyDispersionGiven;
      for (size_t k = 0; k != control.pairs.size(); ++k)
        terms[py::str(control.pairs[k].name)] = on && k < tails.size() && !tails[k].empty();
      dispersion["pair_terms"] = terms;
      d["dispersion"] = dispersion;
      // The columns that the terms observe, in their order
      // (D232).
      std::vector<std::string> observed;
      for (const auto &[name, unit] : control.getObservableColumns()) observed.push_back(name);
      d["observables"] = observed;
      return d;
    });
  m.def("compile", [](std::shared_ptr<Input<model::System>> system,
                        std::shared_ptr<Input<model::InitialState>> state,
                        std::shared_ptr<Input<model::Integrator>> integrator,
                        std::shared_ptr<Input<model::Ensemble>> ensemble,
                        std::shared_ptr<Input<model::Execution>> execution,
                        std::shared_ptr<Input<model::Schedule>> schedule,
                        bool cache) {
    if (!system || !state || !integrator || !ensemble || !execution || !schedule)
      throw InputError("compile inputs must not be None");
    // Keep the GIL while copying and building: concurrent mutation cannot
    // race version capture. GIL release belongs to persistent execution.
    auto prepared = unwrap(model::prepare(system->value, state->value, integrator->value,
                                          ensemble->value, execution->value, schedule->value));
    // A restraint that selects nothing restrains nothing: the control file
    // warns of it, and so does compile.
    for (const auto &[code, message] : prepared.system.warnings)
      if ((code == "empty_selection" && llvm::StringRef(message).starts_with("the restraint of")) ||
          code == "tunable_fixed_pairs" || code == "pair_tail_left_out" ||
          code == "dispersion_switched")
        if (PyErr_WarnEx(PyExc_UserWarning, message.c_str(), 1) != 0)
          throw py::error_already_set();
    // What defined the model, for its checkpoints (D223).
    prepared.fingerprint = model::getFingerprint(
        system->value, integrator->value, ensemble->value, execution->value,
        model::GivenSettings{system->given, integrator->given, ensemble->given,
                             execution->given},
        prepared);
    Program result;
    // Built, not lowered: a simulation lowers programs of its own, and
    // `lowered_ir` lowers on its first read (D224, #151).
    result.compiled = unwrap(compiler::plan(prepared, cache));
    result.cache = cache;
    result.prepared = std::make_shared<const model::PreparedModel>(std::move(prepared));
    result.track(system); result.track(state); result.track(integrator);
    result.track(ensemble); result.track(execution); result.track(schedule);
    return std::make_shared<Program>(std::move(result));
  }, py::arg("system"), py::arg("state"), py::arg("integrator"),
     py::arg("ensemble"), py::arg("execution"), py::arg("schedule"),
     py::kw_only(), py::arg("cache") = true);

  // Removes the entries of the compile cache (D217):
  // those of this format in `directory`, else in MDIR_COMPILE_CACHE_DIR,
  // which MDIR_COMPILE_CACHE=off does not hide from it.
  m.def("clear_compile_cache", [](std::optional<std::string> directory) {
    if (!directory)
      if (const char *named = std::getenv("MDIR_COMPILE_CACHE_DIR"); named && *named)
        directory = named;
    compiler::ClearedCache cleared;
    if (directory && !directory->empty()) {
      py::gil_scoped_release release;
      cleared = compiler::clearCache(*directory);
    }
    py::dict d;
    d["directory"] = directory && !directory->empty() ? py::cast(*directory) : py::none();
    d["host_entries"] = cleared.hostEntries;
    d["gpu_entries"] = cleared.gpuEntries;
    d["bytes"] = cleared.bytes;
    return d;
  }, py::arg("directory") = py::none());

  // Persistent simulations (D196, docs/python-segments.md).
  py::class_<compiler::SimulationState>(m, "State")
    .def_property_readonly("step", [](const compiler::SimulationState &s) { return s.step; })
    .def_property_readonly("time", [](const compiler::SimulationState &s) { return s.time; })
    .def_property_readonly("positions", [](const compiler::SimulationState &s) {
      return host::copy(s.positions.data(), s.positions.size(),
                        {static_cast<py::ssize_t>(s.positions.size() / 3), 3});
    })
    .def_property_readonly("velocities", [](const compiler::SimulationState &s) {
      return host::copy(s.velocities.data(), s.velocities.size(),
                        {static_cast<py::ssize_t>(s.velocities.size() / 3), 3});
    })
    .def_property_readonly("tunables_version",
                           [](const compiler::SimulationState &s) { return s.tunablesVersion; })
    .def_property_readonly("velocity_offset",
                           [](const compiler::SimulationState &s) { return s.velocityOffset; })
    .def_property_readonly("forces", [](const compiler::SimulationState &s) -> py::object {
      if (s.forces.empty()) return py::none();
      return host::copy(s.forces.data(), s.forces.size(),
                        {static_cast<py::ssize_t>(s.forces.size() / 3), 3});
    })
    .def_property_readonly("energies", [](const compiler::SimulationState &s) -> py::object {
      // Those of a run that ended with a step of energy (run(n, energy=True)),
      // at this step: kJ/mol, K, bar, nm^3.
      if (!s.energies) return py::none();
      const auto &e = *s.energies;
      py::dict d;
      d["potential"] = e.potential; d["kinetic"] = e.kinetic; d["total"] = e.total;
      d["conserved"] = e.conserved; d["temperature"] = e.temperature;
      d["virial"] = e.virial; d["pressure"] = e.pressure; d["volume"] = e.volume;
      return d;
    })
    .def_property_readonly("observables", [](const compiler::SimulationState &s) -> py::object {
      // The columns of `observe` at this step, where `energies` is set:
      // kJ/mol and kJ/mol per unit of the constant (D232).
      if (!s.observables) return py::none();
      py::dict d;
      for (const auto &[name, value] : *s.observables) d[py::str(name)] = value;
      return d;
    })
    .def_property_readonly("minimization", [](const compiler::SimulationState &s) -> py::object {
      // The row of the log of `mdir run` at the last step of a minimization,
      // in kJ/mol and nm (D202).
      if (!s.minimization) return py::none();
      const auto &m = *s.minimization;
      py::dict d;
      d["energy"] = m.energy; d["rms_force"] = m.rmsForce; d["max_force"] = m.maxForce;
      d["max_force_particle"] = m.maxForceParticle; d["step_size"] = m.stepSize;
      d["converged"] = m.converged ? py::cast(*m.converged) : py::none();
      return d;
    })
    .def_property_readonly("cell", [](const compiler::SimulationState &s) {
      driver::Cell cell;
      for (int k = 0; k != 3; ++k) { cell.diagonal[k] = s.box[k]; cell.tilt[k] = s.tilt[k]; }
      return cell;
    });
  // A read-only view of a checkpoint (D223), in MD units.
  using driver::Checkpoint;
  py::class_<Checkpoint>(m, "Checkpoint")
    .def_property_readonly("step", [](const Checkpoint &c) { return c.step; })
    .def_property_readonly("time", [](const Checkpoint &c) { return c.time; })
    .def_property_readonly("positions", [](const Checkpoint &c) { return vectors(c.positions); })
    .def_property_readonly("velocities", [](const Checkpoint &c) { return vectors(c.velocities); })
    .def_property_readonly("forces", [](const Checkpoint &c) -> py::object {
      if (c.forces.empty()) return py::none();
      return vectors(c.forces);
    })
    .def_property_readonly("masses", [](const Checkpoint &c) {
      return host::copy(c.masses.data(), c.masses.size(), {static_cast<py::ssize_t>(c.masses.size())});
    })
    .def_property_readonly("cell", [](const Checkpoint &c) {
      driver::Cell cell;
      for (int k = 0; k != 3; ++k) { cell.diagonal[k] = c.box[k]; cell.tilt[k] = c.tilt[k]; }
      return cell;
    })
    .def_property_readonly("periodic", [](const Checkpoint &c) { return c.periodic; })
    .def_property_readonly("integrator", [](const Checkpoint &c) { return c.integrator; })
    .def_property_readonly("velocity_offset", [](const Checkpoint &c) { return c.velocityOffset; })
    .def_property_readonly("precision", [](const Checkpoint &c) { return c.precision; })
    .def_property_readonly("timestep", [](const Checkpoint &c) { return c.timestep; })
    .def_property_readonly("seed", [](const Checkpoint &c) { return c.seed; })
    .def_property_readonly("first_step", [](const Checkpoint &c) { return c.firstStep; })
    .def_property_readonly("part", [](const Checkpoint &c) { return c.part; })
    .def_property_readonly("outputs_part", [](const Checkpoint &c) { return c.outputsPart; })
    .def_property_readonly("trajectory", [](const Checkpoint &c) { return c.trajectory; })
    .def_property_readonly("frames", [](const Checkpoint &c) { return c.frames; })
    .def_property_readonly("bath", [](const Checkpoint &c) { return c.bath; })
    .def_property_readonly("fingerprint", [](const Checkpoint &c) {
      py::list entries;
      for (const auto &e : c.fingerprint) entries.append(py::make_tuple(e.group, e.name, e.value));
      return entries;
    })
    .def_property_readonly("creator", [](const Checkpoint &c) {
      return c.creator + (c.creatorVersion.empty() ? "" : " " + c.creatorVersion);
    })
    .def_property_readonly("front_end", [](const Checkpoint &c) {
      return c.frontEnd.empty() ? std::string("mdir run") : c.frontEnd;
    })
    .def_property_readonly("model_sha256", [](const Checkpoint &c) -> py::object {
      if (c.modelHash.empty()) return py::none();
      return py::str(c.modelHash);
    })
    .def_property_readonly("plan_sha256", [](const Checkpoint &c) -> py::object {
      if (c.planHash.empty()) return py::none();
      return py::str(c.planHash);
    })
    .def_property_readonly("tunables", [](const Checkpoint &c) {
      py::dict d;
      for (const auto &[name, values] : c.tunables)
        d[py::str(name)] = host::copy(values.data(), values.size(),
                                      {static_cast<py::ssize_t>(values.size())});
      return d;
    })
    .def_property_readonly("tunables_version", [](const Checkpoint &c) { return c.tunablesVersion; })
    .def_property_readonly("tunables_history", [](const Checkpoint &c) { return c.tunablesHistory; })
    .def("__repr__", [](const Checkpoint &c) {
      return "Checkpoint(step=" + std::to_string(c.step) + ", particles=" +
             std::to_string(c.getNumParticles()) + ", integrator=" + c.integrator + ")";
    });
  m.def("read_checkpoint", [](const std::string &path) {
    return readCheckpointFile(path);
  }, py::arg("path"));
  // Reporters (D207, docs/python-reporters.md): the files
  // of `mdir run` written inside the parts of a run, and Python functions
  // called after the part that ends at their step.
  py::enum_<driver::TrajectoryFormat>(m, "TrajectoryFormat")
    .value("DCD", driver::TrajectoryFormat::DCD)
    .value("XTC", driver::TrajectoryFormat::XTC)
    ;
  struct EnergyReporter { std::string file; int64_t period; };
  struct TrajectoryReporter { std::string file; int64_t period; driver::TrajectoryFormat format; };
  struct CallbackReporter { py::object function; int64_t period; };
  struct ObservablesReporter { std::string file; int64_t period; };
  auto positive = [](int64_t period) {
    if (period <= 0) throw InputError("a reporter's period must be a positive number of steps");
    return period;
  };
  py::class_<EnergyReporter>(m, "EnergyReporter")
    .def(py::init([positive](std::string file, int64_t period) {
      return EnergyReporter{std::move(file), positive(period)};
    }), py::arg("file"), py::arg("period"))
    .def_readonly("file", &EnergyReporter::file)
    .def_readonly("period", &EnergyReporter::period);
  // The file of `[output] observables` (D189, D232).
  py::class_<ObservablesReporter>(m, "ObservablesReporter")
    .def(py::init([positive](std::string file, int64_t period) {
      if (file.empty()) throw InputError("ObservablesReporter takes the name of a file");
      return ObservablesReporter{std::move(file), positive(period)};
    }), py::arg("file"), py::arg("period"))
    .def_readonly("file", &ObservablesReporter::file)
    .def_readonly("period", &ObservablesReporter::period);
  py::class_<TrajectoryReporter>(m, "TrajectoryReporter")
    .def(py::init([positive](std::string file, int64_t period,
                             std::optional<driver::TrajectoryFormat> format) {
      if (!format) {
        llvm::StringRef name(file);
        if (name.ends_with_insensitive(".dcd")) format = driver::TrajectoryFormat::DCD;
        else if (name.ends_with_insensitive(".xtc")) format = driver::TrajectoryFormat::XTC;
        else throw InputError("TrajectoryReporter: cannot tell the format of '" + file +
                              "'; name a .dcd or .xtc file, or give format=");
      }
      return TrajectoryReporter{std::move(file), positive(period), *format};
    }), py::arg("file"), py::arg("period"), py::arg("format") = py::none())
    .def_readonly("file", &TrajectoryReporter::file)
    .def_readonly("period", &TrajectoryReporter::period)
    .def_readonly("format", &TrajectoryReporter::format);
  // Checkpoints (D223, docs/python-checkpoints.md).
  struct CheckpointReporter { std::string file; int64_t period; };
  py::class_<CheckpointReporter>(m, "CheckpointReporter")
    .def(py::init([positive](std::string file, int64_t period) {
      if (file.empty()) throw InputError("CheckpointReporter takes the name of a file");
      return CheckpointReporter{std::move(file), positive(period)};
    }), py::arg("file"), py::arg("period"))
    .def_readonly("file", &CheckpointReporter::file)
    .def_readonly("period", &CheckpointReporter::period);
  py::class_<CallbackReporter>(m, "CallbackReporter")
    .def(py::init([positive](py::object function, int64_t period) {
      if (!PyCallable_Check(function.ptr())) throw InputError("CallbackReporter takes a callable");
      return CallbackReporter{std::move(function), positive(period)};
    }), py::arg("function"), py::arg("period"))
    .def_readonly("function", &CallbackReporter::function)
    .def_readonly("period", &CallbackReporter::period);
  struct PySimulation {
    std::shared_ptr<Program> program;
    /// Shared with the leases of views, which keep it alive
    /// (D220).
    std::shared_ptr<compiler::Simulation> simulation;
    py::list reporters;
    /// The checkpoint reporter of the list, if any.
    std::optional<CheckpointReporter> checkpoints;
    /// Gives the simulation the built-in reporters of the list; returns the
    /// callbacks.
    std::vector<CallbackReporter> sync() {
      compiler::Simulation::Reports given;
      std::vector<CallbackReporter> callbacks;
      checkpoints.reset();
      for (py::handle item : reporters) {
        if (py::isinstance<CheckpointReporter>(item)) {
          if (checkpoints) throw InputError("a simulation takes one CheckpointReporter");
          checkpoints = item.cast<CheckpointReporter>();
          continue;
        }
        if (py::isinstance<EnergyReporter>(item)) {
          if (given.energyPeriod) throw InputError("a simulation takes one EnergyReporter");
          const auto &r = item.cast<const EnergyReporter &>();
          given.energyPath = r.file; given.energyPeriod = r.period;
        } else if (py::isinstance<TrajectoryReporter>(item)) {
          if (given.framePeriod) throw InputError("a simulation takes one TrajectoryReporter");
          const auto &r = item.cast<const TrajectoryReporter &>();
          given.trajectoryPath = r.file; given.framePeriod = r.period;
          given.trajectoryFormat = r.format;
        } else if (py::isinstance<ObservablesReporter>(item)) {
          if (given.observablesPeriod) throw InputError("a simulation takes one ObservablesReporter");
          const auto &r = item.cast<const ObservablesReporter &>();
          given.observablesPath = r.file; given.observablesPeriod = r.period;
        } else if (py::isinstance<CallbackReporter>(item)) {
          callbacks.push_back(item.cast<CallbackReporter>());
        } else {
          throw InputError("Simulation.reporters holds EnergyReporter, TrajectoryReporter, "
                           "ObservablesReporter, CheckpointReporter, and CallbackReporter values");
        }
      }
      const auto &now = simulation->getReports();
      if (given.energyPath != now.energyPath || given.energyPeriod != now.energyPeriod ||
          given.trajectoryPath != now.trajectoryPath || given.framePeriod != now.framePeriod ||
          given.trajectoryFormat != now.trajectoryFormat ||
          given.observablesPath != now.observablesPath ||
          given.observablesPeriod != now.observablesPeriod)
        if (llvm::Error error = simulation->setReports(given)) raise(std::move(error));
      return callbacks;
    }
  };
  // The values of a simulation's tunables, a mapping of names to arrays
  // (D213).
  struct TunableValues { py::object owner; };
  static auto simulationOf = [](const TunableValues &t) -> compiler::Simulation & {
    return *t.owner.cast<PySimulation &>().simulation;
  };
  static auto valueOf = [](const TunableValues &t, const std::string &name) {
    auto &sim = simulationOf(t);
    int k = sim.getTunables().find(name);
    if (k < 0) throw py::key_error(name);
    const auto &v = sim.getTunableValues()[k];
    return host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(v.size())});
  };
  static auto namesOf = [](const TunableValues &t) {
    std::vector<std::string> names;
    for (const auto &entry : simulationOf(t).getTunables().tunables) names.push_back(entry.name);
    return names;
  };
  static auto updateOf = [](TunableValues &t, py::handle mapping) {
    auto &sim = simulationOf(t);
    const auto &set = sim.getTunables();
    if (set.empty())
      throw InputError("the program of this simulation declares no tunable parameters "
                       "(System.tunables)");
    std::vector<std::pair<std::string, std::vector<double>>> changes;
    auto items = py::reinterpret_borrow<py::object>(mapping).attr("items")();
    for (py::handle item : items) {
      auto pair = py::reinterpret_borrow<py::tuple>(item);
      if (!py::isinstance<py::str>(pair[0]))
        throw InputError("Simulation.tunables: the names of tunables are strings");
      auto name = py::cast<std::string>(pair[0]);
      int k = set.find(name);
      if (k < 0) throw InputError("this simulation has no tunable named '" + name + "'");
      const auto &entry = set.tunables[k];
      changes.emplace_back(name, host::doubles(pair[1], "Simulation.tunables['" + name + "']",
                                               entry.entries, {},
                                               tunables::unitOf(entry.parameter, entry.term)));
    }
    std::optional<llvm::Error> error;
    {
      py::gil_scoped_release release;
      error.emplace(sim.updateTunables(changes));
    }
    if (*error) raise(std::move(*error));
    return sim.getTunablesVersion();
  };
  // The derivative of the energy in the tunables at a state
  // (D230): a read-only mapping of the names to arrays of
  // the shape of the values.
  struct TunableGradient {
    py::dict values;
    py::frozenset zero;
    double energy;
    int64_t version, step;
  };
  py::class_<TunableGradient>(m, "TunableGradient")
    .def("__getitem__", [](const TunableGradient &g, const std::string &name) -> py::object {
      if (!g.values.contains(name)) throw py::key_error(name);
      return g.values[py::str(name)];
    })
    .def("__len__", [](const TunableGradient &g) { return g.values.size(); })
    .def("__contains__", [](const TunableGradient &g, const std::string &name) {
      return g.values.contains(name);
    })
    .def("__iter__", [](const TunableGradient &g) { return py::iter(g.values); })
    .def("keys", [](const TunableGradient &g) { return g.values.attr("keys")(); })
    .def("values", [](const TunableGradient &g) { return g.values.attr("values")(); })
    .def("items", [](const TunableGradient &g) { return g.values.attr("items")(); })
    .def_property_readonly("energy", [](const TunableGradient &g) { return g.energy; })
    .def_property_readonly("version", [](const TunableGradient &g) { return g.version; })
    .def_property_readonly("step", [](const TunableGradient &g) { return g.step; })
    .def_property_readonly("zero", [](const TunableGradient &g) { return g.zero; })
    .def("__repr__", [](const TunableGradient &g) {
      return "TunableGradient(step=" + std::to_string(g.step) + ", version=" +
             std::to_string(g.version) + ", " + py::str(py::list(g.values)).cast<std::string>() + ")";
    });
  py::class_<TunableValues>(m, "TunableValues")
    .def("gradient", [](TunableValues &t) {
      auto &sim = simulationOf(t);
      std::optional<llvm::Expected<compiler::Simulation::TunableGradient>> result;
      {
        py::gil_scoped_release release;
        result.emplace(sim.evaluateTunableGradient());
      }
      if (!*result) raise(result->takeError());
      const auto &given = **result;
      TunableGradient g{py::dict(), py::frozenset(py::set()), given.energy, given.version, given.step};
      py::set zero;
      const auto &set = sim.getTunables();
      for (size_t k = 0; k != set.tunables.size(); ++k) {
        const auto &v = given.values[k];
        py::array a = host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(v.size())});
        a.attr("setflags")(py::arg("write") = false);
        g.values[py::str(set.tunables[k].name)] = a;
        if (given.zero[k]) zero.add(py::str(set.tunables[k].name));
      }
      g.zero = py::frozenset(zero);
      return g;
    })
    .def("__getitem__", [](const TunableValues &t, const std::string &name) { return valueOf(t, name); })
    .def("__setitem__", [](TunableValues &t, const std::string &name, py::object value) {
      py::dict d; d[py::str(name)] = value; updateOf(t, d);
    })
    .def("__len__", [](const TunableValues &t) { return simulationOf(t).getTunables().tunables.size(); })
    .def("__contains__", [](const TunableValues &t, const std::string &name) {
      return simulationOf(t).getTunables().find(name) >= 0;
    })
    .def("__iter__", [](const TunableValues &t) { return py::iter(py::cast(namesOf(t))); })
    .def("keys", [](const TunableValues &t) { return namesOf(t); })
    .def("values", [](const TunableValues &t) {
      py::list result;
      for (const auto &name : namesOf(t)) result.append(valueOf(t, name));
      return result;
    })
    .def("items", [](const TunableValues &t) {
      py::list result;
      for (const auto &name : namesOf(t)) result.append(py::make_tuple(name, valueOf(t, name)));
      return result;
    })
    .def("update", [](TunableValues &t, py::object mapping, py::kwargs more) {
      py::dict all;
      if (!mapping.is_none()) all = py::dict(mapping);
      for (auto item : more) all[item.first] = item.second;
      return updateOf(t, all);
    }, py::arg("values") = py::none())
    .def_property_readonly("version", [](const TunableValues &t) { return simulationOf(t).getTunablesVersion(); })
    .def_property_readonly("history", [](const TunableValues &t) { return simulationOf(t).getTunablesHistory(); })
    .def_property_readonly("units", [](const TunableValues &t) {
      py::dict d;
      for (const auto &entry : simulationOf(t).getTunables().tunables) d[py::str(entry.name)] = entry.unit;
      return d;
    })
    .def("__repr__", [](const TunableValues &t) {
      std::string text = "TunableValues(version=" + std::to_string(simulationOf(t).getTunablesVersion()) + ", [";
      bool first = true;
      for (const auto &name : namesOf(t)) { text += (first ? "'" : ", '") + name + "'"; first = false; }
      return text + "])";
    });
  views::bind(m);
  py::class_<PySimulation>(m, "Simulation")
    .def(py::init([](std::shared_ptr<Program> program, std::optional<bool> cache,
                     std::optional<std::string> checkpoint, bool stage, bool append) {
      if (!program) throw InputError("Simulation takes a compiled program");
      program->checkCurrent();
      if (!checkpoint && stage)
        throw InputError("Simulation: stage=True begins a stage from a checkpoint; give checkpoint=");
      // A checkpoint is read before anything is compiled.
      std::optional<driver::Checkpoint> taken;
      if (checkpoint) {
        taken = readCheckpointFile(*checkpoint);
      }
      // The program's choice unless one is given (D217).
      bool useCache = cache.value_or(program->cache);
      auto prepared = program->prepared;
      std::optional<llvm::Expected<std::unique_ptr<compiler::Simulation>>> created;
      {
        // The inputs were copied at compilation; nothing here touches Python.
        py::gil_scoped_release release;
        created.emplace(compiler::Simulation::create(*prepared, useCache,
                                                     program->code.get()));
      }
      PySimulation result;
      result.program = std::move(program);
      result.simulation = unwrap(std::move(*created));
      // The same run goes on, or a stage begins (D223);
      // what differs in execution, or is evaluated anew, is a warning.
      if (taken) {
        auto notes = unwrap(result.simulation->continueFrom(*taken, stage, append));
        for (const std::string &note : notes)
          if (PyErr_WarnEx(PyExc_UserWarning, ("Simulation: '" + *checkpoint + "': " + note).c_str(), 1) != 0)
            throw py::error_already_set();
      }
      return result;
    }), py::arg("program"), py::kw_only(), py::arg("cache") = py::none(),
       py::arg("checkpoint") = py::none(), py::arg("stage") = false,
       py::arg("append") = true)
    .def("save_checkpoint", [](PySimulation &s, const std::string &path) {
      std::optional<llvm::Error> error;
      {
        py::gil_scoped_release release;
        error.emplace(s.simulation->saveCheckpoint(path, MDIR_VERSION));
      }
      if (*error) raise(std::move(*error));
    }, py::arg("path"))
    .def("run", [](py::object self, int64_t steps, bool energy) {
      auto &s = self.cast<PySimulation &>();
      if (steps < 0) throw InputError("run takes a nonnegative number of steps");
      std::vector<CallbackReporter> callbacks = s.sync();
      // No step, and the energies of the state (D213).
      if (steps == 0 && energy) {
        std::optional<llvm::Error> error;
        {
          py::gil_scoped_release release;
          error.emplace(s.simulation->evaluate());
        }
        if (*error) raise(std::move(*error));
        return int64_t(0);
      }
      // The parts end at the steps of the callbacks, whose state they take;
      // the files of the built-in reporters are written inside the parts.
      int64_t taken = 0, end = s.simulation->getStep() + steps;
      // A checkpoint reporter ends a part at its steps, as a callback does
      // (D223).
      std::optional<CheckpointReporter> saves = s.checkpoints;
      while (taken < steps) {
        int64_t now = s.simulation->getStep(), next = end;
        for (const auto &c : callbacks) next = std::min(next, (now / c.period + 1) * c.period);
        if (saves) next = std::min(next, (now / saves->period + 1) * saves->period);
        bool atCallback = next < end || (!callbacks.empty() && llvm::any_of(callbacks,
            [&](const CallbackReporter &c) { return end % c.period == 0; }));
        int64_t leg = withSignals([&](const std::function<bool()> &poll) {
          return s.simulation->run(next - now, poll, energy || atCallback);
        });
        taken += leg;
        if (leg < next - now) break;  // A stop.
        int64_t at = s.simulation->getStep();
        if (saves && at % saves->period == 0) {
          std::optional<llvm::Error> error;
          {
            py::gil_scoped_release release;
            error.emplace(s.simulation->saveCheckpoint(saves->file, MDIR_VERSION));
          }
          if (*error) raise(std::move(*error));
        }
        bool due = false;
        for (const auto &c : callbacks) due = due || at % c.period == 0;
        if (!due) continue;
        py::object state = py::cast(unwrap(s.simulation->getState()));
        for (const auto &c : callbacks)
          if (at % c.period == 0) c.function(self, state);
      }
      return taken;
    }, py::arg("steps"), py::arg("energy") = false)
    .def_property("reporters", [](PySimulation &s) { return s.reporters; },
                  [](PySimulation &s, py::list reporters) { s.reporters = std::move(reporters); })
    .def("close_reporters", [](PySimulation &s) {
      s.simulation->closeReports();
      s.reporters = py::list();
    })
    .def("minimize", [](PySimulation &s, std::optional<int64_t> steps, py::object given) {
      // The largest force at which it stops, in kJ/mol/nm (D219).
      std::optional<double> tolerance;
      if (!given.is_none())
        tolerance = units::scalar(given, "minimize: tolerance", units::force);
      return withSignals([&](const std::function<bool()> &poll) {
        return s.simulation->minimize(steps, poll, tolerance);
      });
    }, py::arg("steps") = py::none(), py::arg("tolerance") = py::none())
    .def("request_stop", [](PySimulation &s) { s.simulation->requestStop(); })
    .def("state", [](PySimulation &s) { return unwrap(s.simulation->getState()); })
    // Read-only DLPack views of the buffers (D220).
    .def("view", [](PySimulation &s) { return views::take(s.simulation); })
    // Writable borrows with a commit (D229).
    .def("borrow", [](PySimulation &s) { return views::borrow(s.simulation); })
    .def_property_readonly("versions", [](const PySimulation &s) {
      auto v = s.simulation->getStateVersions();
      py::dict d;
      d["positions"] = v.positions;
      d["velocities"] = v.velocities;
      d["cell"] = v.cell;
      d["tunables"] = s.simulation->getTunablesVersion();
      return d;
    })
    .def_property_readonly("commits", [](const PySimulation &s) {
      py::list result;
      for (const auto &[step, fields] : s.simulation->getCommits()) {
        py::list names;
        for (const std::string &name : fields) names.append(name);
        result.append(py::make_tuple(step, py::tuple(names)));
      }
      return result;
    })
    .def_property_readonly("leases", [](const PySimulation &s) { return s.simulation->getLeases(); })
    .def_property_readonly("step", [](const PySimulation &s) { return s.simulation->getStep(); })
    .def_property_readonly("time", [](const PySimulation &s) { return s.simulation->getTime(); })
    .def_property_readonly("failed", [](const PySimulation &s) { return s.simulation->hasFailed(); })
    .def_property_readonly("program", [](const PySimulation &s) { return s.program; })
    // What compiling the programs cost, and what the compile cache saved
    // (D212).
    .def_property_readonly("compile_stats", [](const PySimulation &s) {
      compiler::CompileStats c = s.simulation->getCompileStats();
      py::dict d;
      d["programs"] = c.programs;
      d["pipeline_seconds"] = c.pipelineSeconds;
      d["engine_seconds"] = c.engineSeconds;
      d["host_compiled"] = c.compiled;
      d["host_compile_seconds"] = c.compileSeconds;
      d["cache_hits"] = c.hits;
      d["cache_saved_seconds"] = c.savedSeconds;
      d["cache_rejected"] = c.rejected;
      d["cache_stored"] = c.stored;
      d["cache_unstored"] = c.unstored;
      d["cache_lookup_seconds"] = c.lookupSeconds;
      // The GPU modules (D214).
      d["gpu_modules"] = c.gpuModules;
      d["gpu_serialize_seconds"] = c.gpuSerializeSeconds;
      d["gpu_ptx_compiled"] = c.gpuPtxCompiled;
      d["gpu_ptx_hits"] = c.gpuPtxHits;
      d["gpu_cubin_compiled"] = c.gpuCubinCompiled;
      d["gpu_cubin_hits"] = c.gpuCubinHits;
      d["gpu_compile_seconds"] = c.gpuCompileSeconds;
      d["gpu_cache_saved_seconds"] = c.gpuSavedSeconds;
      d["gpu_cache_lookup_seconds"] = c.gpuLookupSeconds;
      d["gpu_cache_rejected"] = c.gpuRejected;
      d["gpu_cache_stored"] = c.gpuStored;
      d["gpu_cache_unstored"] = c.gpuUnstored;
      // Programs compiled with cache=False (D217).
      d["cache_bypassed"] = c.bypassed;
      // Programs that took the code kept with their Program
      // (D[program-reuse]).
      d["program_reused"] = c.reused;
      d["reuse_saved_seconds"] = c.reuseSavedSeconds;
      return d;
    })
    .def_property_readonly("tunables", [](py::object self) { return TunableValues{self}; })
    .def_property("part_seconds",
                  [](const PySimulation &s) { return s.simulation->partSeconds; },
                  [](PySimulation &s, py::object value) {
                    // Wall-clock time: a quantity converts to seconds.
                    double seconds = units::scalar(value, "Simulation.part_seconds", units::second);
                    if (!(seconds > 0) || !std::isfinite(seconds))
                      throw InputError("part_seconds must be positive and finite");
                    s.simulation->partSeconds = seconds;
                  });
}

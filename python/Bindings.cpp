// Owned Python inputs; collections cross the boundary by value.
#include "mdir/Compiler/Compile.h"
#include "mdir/Compiler/Simulation.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>
#include <optional>
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
#include "HostArrays.h"

struct Version { uint64_t version = 0; virtual ~Version() = default; };
template <class T> struct Input : Version { T value; std::optional<size_t> particleCount; };
template <class T, class V>
static void property(py::class_<Input<T>, std::shared_ptr<Input<T>>> &c,
                     const char *name, V T::*member) {
  c.def_property(name, [member](const Input<T> &o) { return o.value.*member; },
    [member](Input<T> &o, V value) {
      o.value.*member = std::move(value); ++o.version;
    });
}
template <class T> static auto input(py::module_ &m, const char *name) {
  return py::class_<Input<T>, std::shared_ptr<Input<T>>>(m, name).def(py::init<>());
}
struct Program {
  compiler::CompiledProgram compiled;
  /// What a simulation compiles its programs from (D196).
  std::shared_ptr<const model::PreparedModel> prepared;
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
PYBIND11_MODULE(mdir, m) {
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
      auto values = host::doubles(input, name, 3);
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
    .def_readwrite("groups", &driver::PairTerm::groups);
  py::class_<driver::TupleTerm>(m, "TupleTerm").def(py::init<>())
    .def_readwrite("name", &driver::TupleTerm::name)
    .def_readwrite("expression", &driver::TupleTerm::expression)
    .def_readwrite("arity", &driver::TupleTerm::arity)
    .def_property("particles", [](const driver::TupleTerm &t) { return host::particles(t); },
                  [](driver::TupleTerm &t, py::object value) { host::particles(t, value); })
    .def_property("parameters", [](const driver::TupleTerm &t) { return host::parameters(t); },
                  [](driver::TupleTerm &t, py::sequence value) { host::parameters(t, value); });
  // Restraints (D74, D124; D[python-velocities-restraints]).
  py::enum_<driver::ReferenceScaling>(m, "ReferenceScaling")
    .value("Center", driver::ReferenceScaling::Center)
    .value("All", driver::ReferenceScaling::All)
    ;
  py::class_<model::System::Restraint>(m, "Restraint")
    .def(py::init<>())
    .def(py::init([](std::string selection, double forceConstant, driver::ReferenceScaling scaling) {
      return model::System::Restraint{std::move(selection), forceConstant, scaling};
    }), py::arg("selection"), py::arg("force_constant"),
        py::arg("reference_scaling") = driver::ReferenceScaling::Center)
    .def_readwrite("selection", &model::System::Restraint::selection)
    .def_readwrite("force_constant", &model::System::Restraint::forceConstant)
    .def_readwrite("reference_scaling", &model::System::Restraint::scaling);
  auto system = input<model::System>(m, "System");
  property(system, "periodic", &model::System::periodic);
  property(system, "cutoff", &model::System::cutoff);
  property(system, "pairlist_distance", &model::System::pairlistDistance);
  property(system, "switch_distance", &model::System::switchDistance);
  property(system, "truncation", &model::System::truncation);
  property(system, "electrostatics", &model::System::electrostatics);
  property(system, "dispersion", &model::System::dispersion);
  property(system, "pme_alpha", &model::System::pmeAlpha);
  property(system, "pme_tolerance", &model::System::pmeTolerance);
  property(system, "pme_spacing", &model::System::pmeSpacing);
  property(system, "pme_grid", &model::System::pmeGrid);
  property(system, "pme_order", &model::System::pmeOrder);
  property(system, "rigid_hydrogen_bonds", &model::System::rigidHydrogenBonds);
  property(system, "rigid_water", &model::System::rigidWater);
  property(system, "flexible_water", &model::System::flexibleWater);
  property(system, "water_residues", &model::System::waterResidues);
  property(system, "pair_terms", &model::System::pairTerms);
  property(system, "tuple_terms", &model::System::tupleTerms);
  property(system, "restraints", &model::System::restraints);
  system.def_property("restraint_reference", [](const Input<model::System> &o) {
    const auto &v = o.value.restraintReference;
    return host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(v.size() / 3), 3});
  }, [](Input<model::System> &o, py::object source) {
    const std::string name = "System.restraint_reference";
    auto values = host::doubles(source, name, {}, 3);
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
      auto values = host::doubles(source, name, {}, 3);
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
  // and seed, in a new state (D[python-velocities-restraints]).
  initialstate.def("draw_velocities", [](const Input<model::InitialState> &o,
                                         std::shared_ptr<Input<model::System>> system,
                                         double temperature, py::int_ seed) {
    if (!system) throw InputError("draw_velocities takes a System");
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
  auto integrator = input<model::Integrator>(m, "Integrator");
  property(integrator, "method", &model::Integrator::method);
  property(integrator, "timestep", &model::Integrator::timestep);
  property(integrator, "minimize", &model::Integrator::minimize);
  property(integrator, "minimize_step", &model::Integrator::minimizeStep);
  auto ensemble = input<model::Ensemble>(m, "Ensemble");
  property(ensemble, "kind", &model::Ensemble::kind);
  property(ensemble, "temperature", &model::Ensemble::temperature);
  property(ensemble, "tau_t", &model::Ensemble::tauT);
  property(ensemble, "pressure", &model::Ensemble::pressure);
  property(ensemble, "tau_p", &model::Ensemble::tauP);
  property(ensemble, "compressibility", &model::Ensemble::compressibility);
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
  auto schedule = input<model::Schedule>(m, "Schedule");
  property(schedule, "steps", &model::Schedule::steps);
  property(schedule, "energy_period", &model::Schedule::energyPeriod);
  system.def_property_readonly("particle_count", [](const Input<model::System> &s) {
    return s.value.topology.masses.size();
  });
  py::class_<model::LoadedData>(m, "LoadedData")
    .def_property_readonly("format", [](const model::LoadedData &d) { return d.format; })
    .def_property_readonly("sources", [](const model::LoadedData &d) { return d.sources; })
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
    .def_property_readonly("lowered_ir", [](const Program &p) { return p.compiled.loweredIR; })
    .def_property_readonly("pipeline", [](const Program &p) { return p.compiled.pipeline; })
    .def_property_readonly("stale", &Program::stale)
    .def("check_current", &Program::checkCurrent)
    .def_property_readonly("plan", [](const Program &p) {
      py::dict d;
      const auto &c = p.compiled;
      d["target"] = c.execution.target; d["precision"] = c.execution.precision;
      d["device"] = c.execution.device; d["threads"] = c.execution.threads;
      d["deterministic"] = c.execution.deterministic;
      d["reorders"] = c.program.reorders;
      d["entry"] = c.program.entry;
      d["state_dtype"] = c.program.state == driver::Element::F64 ? "float64" : "float32";
      d["force_dtype"] = c.program.force == driver::Element::F64 ? "float64" : "float32";
      d["pme"] = c.program.pme;
      d["pme_grid"] = std::array<int64_t, 3>{c.program.pmeGrid[0], c.program.pmeGrid[1], c.program.pmeGrid[2]};
      return d;
    });
  m.def("compile", [](std::shared_ptr<Input<model::System>> system,
                        std::shared_ptr<Input<model::InitialState>> state,
                        std::shared_ptr<Input<model::Integrator>> integrator,
                        std::shared_ptr<Input<model::Ensemble>> ensemble,
                        std::shared_ptr<Input<model::Execution>> execution,
                        std::shared_ptr<Input<model::Schedule>> schedule) {
    if (!system || !state || !integrator || !ensemble || !execution || !schedule)
      throw InputError("compile inputs must not be None");
    // Keep the GIL while copying and lowering: concurrent mutation cannot race
    // version capture. GIL release belongs to persistent execution.
    auto prepared = unwrap(model::prepare(system->value, state->value, integrator->value,
                                          ensemble->value, execution->value, schedule->value));
    // A restraint that selects nothing restrains nothing: the control file
    // warns of it, and so does compile.
    for (const auto &[code, message] : prepared.system.warnings)
      if (code == "empty_selection" && llvm::StringRef(message).starts_with("the restraint of"))
        if (PyErr_WarnEx(PyExc_UserWarning, message.c_str(), 1) != 0)
          throw py::error_already_set();
    Program result;
    result.compiled = unwrap(compiler::compile(prepared));
    result.prepared = std::make_shared<const model::PreparedModel>(std::move(prepared));
    result.track(system); result.track(state); result.track(integrator);
    result.track(ensemble); result.track(execution); result.track(schedule);
    return std::make_shared<Program>(std::move(result));
  }, py::arg("system"), py::arg("state"), py::arg("integrator"),
     py::arg("ensemble"), py::arg("execution"), py::arg("schedule"));

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
    .def_property_readonly("cell", [](const compiler::SimulationState &s) {
      driver::Cell cell;
      for (int k = 0; k != 3; ++k) { cell.diagonal[k] = s.box[k]; cell.tilt[k] = s.tilt[k]; }
      return cell;
    });
  struct PySimulation {
    std::shared_ptr<Program> program;
    std::unique_ptr<compiler::Simulation> simulation;
  };
  py::class_<PySimulation>(m, "Simulation")
    .def(py::init([](std::shared_ptr<Program> program) {
      if (!program) throw InputError("Simulation takes a compiled program");
      program->checkCurrent();
      auto prepared = program->prepared;
      std::optional<llvm::Expected<std::unique_ptr<compiler::Simulation>>> created;
      {
        // The inputs were copied at compilation; nothing here touches Python.
        py::gil_scoped_release release;
        created.emplace(compiler::Simulation::create(*prepared));
      }
      PySimulation result;
      result.program = std::move(program);
      result.simulation = unwrap(std::move(*created));
      return result;
    }), py::arg("program"))
    .def("run", [](PySimulation &s, int64_t steps, bool energy) {
      std::optional<py::error_already_set> interrupt;
      std::optional<llvm::Expected<int64_t>> taken;
      {
        py::gil_scoped_release release;
        // Signals reach Python between parts; a KeyboardInterrupt ends the
        // run there and is raised once the GIL is held again.
        taken.emplace(s.simulation->run(steps, [&] {
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() != 0) {
            interrupt.emplace();
            return true;
          }
          return false;
        }, energy));
      }
      if (interrupt) {
        if (!*taken) llvm::consumeError(taken->takeError());
        throw *interrupt;
      }
      return unwrap(std::move(*taken));
    }, py::arg("steps"), py::arg("energy") = false)
    .def("request_stop", [](PySimulation &s) { s.simulation->requestStop(); })
    .def("state", [](PySimulation &s) { return unwrap(s.simulation->getState()); })
    .def_property_readonly("step", [](const PySimulation &s) { return s.simulation->getStep(); })
    .def_property_readonly("time", [](const PySimulation &s) { return s.simulation->getTime(); })
    .def_property_readonly("failed", [](const PySimulation &s) { return s.simulation->hasFailed(); })
    .def_property_readonly("program", [](const PySimulation &s) { return s.program; })
    .def_property("part_seconds",
                  [](const PySimulation &s) { return s.simulation->partSeconds; },
                  [](PySimulation &s, double seconds) {
                    if (!(seconds > 0) || !std::isfinite(seconds))
                      throw InputError("part_seconds must be positive and finite");
                    s.simulation->partSeconds = seconds;
                  });
}

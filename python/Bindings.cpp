// Owned Python inputs; collections cross the boundary by value.
#include "mdir/Compiler/Compile.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>
namespace py = pybind11;
using namespace mdir;
struct InputError : std::runtime_error { using std::runtime_error::runtime_error; };
struct UnsupportedError : std::runtime_error { using std::runtime_error::runtime_error; };
struct CompileError : std::runtime_error { using std::runtime_error::runtime_error; };
struct StaleProgramError : std::runtime_error { using std::runtime_error::runtime_error; };
[[noreturn]] static void raise(llvm::Error error) {
  bool unsupported = false, compile = false;
  std::string message;
  llvm::handleAllErrors(std::move(error),
    [&](const model::ModelError &e) {
      unsupported = e.kind == model::ModelError::Unsupported; message = e.message;
    }, [&](const compiler::CompileError &e) {
      compile = true; message = e.diagnostic;
    }, [&](const llvm::ErrorInfoBase &e) {
      llvm::raw_string_ostream os(message); e.log(os);
    });
  if (unsupported) throw UnsupportedError(message);
  if (compile) throw CompileError(message);
  throw InputError(message);
}
template <class T> static T unwrap(llvm::Expected<T> value) {
  if (!value) raise(value.takeError());
  return std::move(*value);
}
struct Version { uint64_t version = 0; virtual ~Version() = default; };
template <class T> struct Input : Version { T value; };
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
  m.attr("__version__") = MDIR_VERSION;
  py::register_exception<InputError>(m, "InputError", PyExc_ValueError);
  py::register_exception<UnsupportedError>(m, "UnsupportedError");
  py::register_exception<CompileError>(m, "CompileError");
  py::register_exception<StaleProgramError>(m, "StaleProgramError");
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
  py::class_<driver::Cell>(m, "Cell")
    .def(py::init<>()).def_readwrite("diagonal", &driver::Cell::diagonal)
    .def_readwrite("tilt", &driver::Cell::tilt)
    .def_property_readonly("vectors", &driver::Cell::getVectors);
  py::class_<driver::PairTerm>(m, "PairTerm").def(py::init<>())
    .def_readwrite("name", &driver::PairTerm::name)
    .def_readwrite("expression", &driver::PairTerm::expression)
    .def_readwrite("constants", &driver::PairTerm::constants)
    .def_readwrite("groups", &driver::PairTerm::groups);
  py::class_<driver::TupleTerm>(m, "TupleTerm").def(py::init<>())
    .def_readwrite("name", &driver::TupleTerm::name)
    .def_readwrite("expression", &driver::TupleTerm::expression)
    .def_readwrite("arity", &driver::TupleTerm::arity)
    .def_readwrite("particles", &driver::TupleTerm::particles)
    .def_readwrite("parameters", &driver::TupleTerm::parameters);
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
  auto initialstate = input<model::InitialState>(m, "InitialState");
  property(initialstate, "positions", &model::InitialState::positions);
  property(initialstate, "velocities", &model::InitialState::velocities);
  property(initialstate, "cell", &model::InitialState::cell);
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
      auto result = std::make_shared<Input<model::InitialState>>(); result->value = d.makeState(); return result;
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
  py::class_<Program>(m, "Program")
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
    Program result;
    result.compiled = unwrap(compiler::compile(prepared));
    result.track(system); result.track(state); result.track(integrator);
    result.track(ensemble); result.track(execution); result.track(schedule);
    return result;
  }, py::arg("system"), py::arg("state"), py::arg("integrator"),
     py::arg("ensemble"), py::arg("execution"), py::arg("schedule"));
}

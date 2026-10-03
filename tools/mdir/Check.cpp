// `mdir check`: reads the input of a run and prints what it describes,
// without compiling anything.

#include "Commands.h"

#include "mdir/Driver/Cell.h"
#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/Output.h"
#include "mdir/Driver/System.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace mdir::driver;

static void printJSON(llvm::json::Object report) {
  llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(report)))
               << "\n";
}

static int fail(llvm::Error error, bool json) {
  std::string message = llvm::toString(std::move(error));
  if (json)
    printJSON(llvm::json::Object{{"schema_version", 1}, {"ok", false},
                                {"errors", llvm::json::Array{message}},
                                {"warnings", llvm::json::Array{}},
                                {"notes", llvm::json::Array{}}});
  else
    llvm::errs() << "mdir: " << message << "\n";
  return 1;
}

static const char *getName(Integrator integrator) {
  return integrator == Integrator::Leapfrog   ? "LEAPFROG"
         : integrator == Integrator::Brownian ? "BROWNIAN"
                                              : "VELOCITY_VERLET";
}

static const char *getName(Precision precision) {
  switch (precision) {
  case Precision::Single:
    return "single";
  case Precision::Mixed:
    return "mixed";
  case Precision::Double:
    return "double";
  }
  return "";
}

/// What a topology describes.
static void describeTopology(const Control &control, const System &system) {
  const Topology &topology = *system.topology;
  size_t count = topology.getNumParticles();
  double totalMass = 0.0, totalCharge = 0.0;
  for (size_t i = 0; i != count; ++i) {
    totalMass += topology.masses[i];
    totalCharge += topology.charges[i];
  }
  size_t impropers = 0;
  for (const Topology::Dihedral &dihedral : topology.dihedrals)
    impropers += dihedral.improper;
  size_t hydrogenBonds = 0;
  for (const Topology::Bond &bond : topology.bonds)
    hydrogenBonds += bond.hydrogen;
  double volume = topology.box[0] * topology.box[1] * topology.box[2];

  const std::string &file = !control.prmtopFile.empty()
                               ? control.prmtopFile
                           : !control.charmmStructureFile.empty()
                               ? control.charmmStructureFile
                               : control.gromacsTopologyFile;
  std::printf("topology:           %s\n", file.c_str());
  std::printf("particles:          %zu\n", count);
  std::printf("residues:           %zu\n", topology.residueNames.size());
  std::printf("types:              %zu\n", topology.getNumTypes());
  std::printf("bonds:              %zu, %zu with hydrogen\n",
              topology.bonds.size(), hydrogenBonds);
  std::printf("angles:             %zu\n", topology.angles.size());
  if (!topology.ureyBradleys.empty())
    std::printf("Urey-Bradley terms: %zu\n", topology.ureyBradleys.size());
  std::printf("dihedrals:          %zu, %zu improper\n",
              topology.dihedrals.size(), impropers);
  if (!topology.harmonicImpropers.empty())
    std::printf("harmonic impropers: %zu\n",
                topology.harmonicImpropers.size());
  std::printf("pairs 1-4:          %zu\n", topology.pairs.size());
  std::printf("excluded pairs:     %zu\n", topology.exclusions.size());
  if (!topology.cmaps.empty())
    std::printf("CMAP terms:         %zu, %zu maps\n", topology.cmaps.size(),
                topology.cmapGrids.size());
  if (!topology.virtualSites.empty())
    std::printf("virtual sites:      %zu\n", topology.virtualSites.size());
  std::printf("total charge:       %.6f e\n", totalCharge);
  std::printf("total mass:         %g amu\n", totalMass);
  if (control.periodic) {
    Cell cell;
    for (int k = 0; k != 3; ++k) {
      cell.diagonal[k] = topology.box[k] / units::length;
      cell.tilt[k] = topology.tilt[k] / units::length;
    }
    if (cell.isOrthorhombic()) {
      std::printf("box:                %g %g %g Å\n", cell.diagonal[0],
                  cell.diagonal[1], cell.diagonal[2]);
    } else {
      std::array<double, 6> shape = cell.getLengthsAndAngles();
      std::printf("box:                %g %g %g Å, angles %g %g %g\n",
                  shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
      std::printf("cell vectors:       a (%g, 0, 0), b (%g, %g, 0), "
                  "c (%g, %g, %g) Å\n",
                  cell.diagonal[0], cell.tilt[0], cell.diagonal[1],
                  cell.tilt[1], cell.tilt[2], cell.diagonal[2]);
    }
    std::printf("density:            %g g/cm³\n",
                totalMass / volume * 1.66053906660e-3);
  } else {
    std::printf("boundary:           NONE (no periodic cell)\n");
  }
  std::printf("velocities:         %s\n",
              topology.velocities.empty() ? "no" : "yes");
  std::printf("degrees of freedom: %g\n", system.getDegreesOfFreedom());
  if (!system.restraintConstants.empty()) {
    size_t restrained = 0;
    for (double k : system.restraintConstants)
      restrained += k > 0.0;
    std::printf("restrained:         %zu particles\n", restrained);
  }
}

static void describeParticles(const Control &control, const System &system) {
  size_t count = system.getNumParticles();
  std::vector<size_t> perType(control.types.size(), 0);
  double totalMass = 0.0;
  for (size_t i = 0; i != count; ++i) {
    ++perType[system.types[i]];
    totalMass += system.masses[i];
  }
  // The box is held in nm; the control file is in Å.
  double volume = system.box[0] * system.box[1] * system.box[2];

  std::printf("particles:          %zu\n", count);
  for (size_t t = 0, e = control.types.size(); t != e; ++t)
    std::printf("  of type %-10s %zu\n", control.types[t].name.c_str(),
                perType[t]);
  std::printf("total mass:         %g amu\n", totalMass);
  if (control.periodic) {
    std::printf("box:                %g %g %g Å\n",
                system.box[0] / units::length, system.box[1] / units::length,
                system.box[2] / units::length);
    // amu/nm³ to g/cm³: 1 amu = 1.66053906660e-24 g, 1 nm³ = 1e-21 cm³.
    std::printf("density:            %g g/cm³\n",
                totalMass / volume * 1.66053906660e-3);
  } else {
    std::printf("boundary:           NONE (no periodic cell)\n");
  }
  std::printf("degrees of freedom: %g\n", system.getDegreesOfFreedom());
  std::printf("pair terms:         %zu\n", control.pairs.size());
}

namespace {
/// An output of the run (D149): the log goes to the standard output and,
/// with a path, to that file as well.
struct OutputFile {
  const char *kind;
  std::string path;
  const char *format;
  int64_t interval;
  bool enabled;
  bool atEnd = false;
  bool exists = false;
  /// How many rows, frames, or checkpoints the whole run writes, or -1
  /// where it depends on the run (a minimization ends where it
  /// converges).
  int64_t count = -1;
  const char *countOf = "";
  /// Where `mdir run` keeps the file that exists before it writes its own
  /// (D149), or empty.
  std::string backup = "";
};

struct Warning {
  std::string code;
  std::string message;
};

struct Preflight {
  std::vector<OutputFile> outputs;
  std::vector<Warning> warnings;
  /// What the run will do that needs no change: the backups of outputs
  /// that exist (D149).
  std::vector<Warning> notes;
};
} // namespace

static const char *getEnsemble(const Control &control) {
  return control.minimize ? "MINIMIZATION"
         : control.barostat ? "NPT"
         : control.thermostat ? "NVT" : "NVE";
}

static const char *getElectrostatics(const Control &control) {
  return control.pme ? "PME"
         : control.reactionField ? "REACTION_FIELD" : "CUTOFF";
}

/// Inspect configured outputs without opening or creating any of them:
/// the files of D149, in the order of its table.
static Preflight inspect(const Control &control) {
  Preflight report;
  // The rows of a run of dynamics begin at its first step; frames and
  // checkpoints come at the end of each interval.
  auto count = [&](int64_t period, bool first) -> int64_t {
    if (control.minimize || period <= 0)
      return -1;
    return control.numSteps / period + (first ? 1 : 0);
  };
  int64_t rows = count(control.energyPeriod, true);
  report.outputs = {
      {"log", control.logFile, "text", control.energyPeriod, true, false,
       false, rows, "rows"},
      {"energy", control.energyFile, "columns", control.energyPeriod,
       !control.energyFile.empty(), false, false, rows, "rows"},
      {"pull", control.pullFile, "columns", control.energyPeriod,
       !control.pullFile.empty(), false, false, rows, "rows"},
      {"trajectory", control.trajectoryFile,
       control.trajectoryFormat == TrajectoryFormat::XTC ? "XTC" : "DCD",
       control.framePeriod, control.framePeriod > 0, false, false,
       count(control.framePeriod, false), "frames"},
      {"checkpoint", control.restartOutput, "H5MD", control.checkpointPeriod,
       control.checkpointPeriod > 0, control.minimize, false,
       control.minimize ? 1 : count(control.checkpointPeriod, false),
       "checkpoints"}};
  // `mdir run` keeps an output of an earlier run as `#<name>.<n>#` before
  // it writes its own; under --continue the files are the run's own
  // (D149). The checkpoint before the last, `.prev`, is kept as well.
  auto backUp = [&](const char *kind, const std::string &path) {
    std::string backup = getBackupPath(path);
    if (backup.empty()) {
      report.warnings.push_back(
          {"backup_limit",
           std::string(kind) + " output '" + path + "' exists with " +
               std::to_string(MaxBackups) +
               " backups already; mdir run stops before it writes. Remove "
               "some of them, or continue the run with --continue"});
      return backup;
    }
    report.notes.push_back(
        {"output_backup",
         std::string(kind) + " output '" + path +
             "' exists; mdir run keeps it as '" + backup +
             "' before it writes its own, and mdir run --continue continues "
             "the run of its checkpoint instead"});
    return backup;
  };
  for (OutputFile &output : report.outputs) {
    output.exists = !output.path.empty() && llvm::sys::fs::exists(output.path);
    if (output.enabled && output.exists)
      output.backup = backUp(output.kind, output.path);
  }
  if (control.checkpointPeriod > 0) {
    std::string previous = getPreviousCheckpointPath(control.restartOutput);
    if (llvm::sys::fs::exists(previous))
      backUp("checkpoint", previous);
  }
  if (control.rebuildPeriod > 0)
    report.warnings.push_back(
        {"fixed_rebuild_interval",
         "'rebuild_interval = " + std::to_string(control.rebuildPeriod) +
             "' skips validity tests between rebuilds and may miss pairs "
             "within the cutoff; set it to 0 to test every step"});
  if (!control.minimize && control.numSteps > 0) {
    if (control.checkpointPeriod == 0)
      report.warnings.push_back(
          {"no_checkpoint", "the run writes no checkpoint and cannot be "
                            "continued with --continue; set [output].checkpoint "
                            "and checkpoint_interval to enable continuation"});
    if (control.energyPeriod == 0)
      report.warnings.push_back(
          {"no_energies", "'energy_interval = 0' disables energy reports; set "
                          "a positive interval to monitor the run"});
  }
  if (!control.restartInput.empty() &&
      !llvm::sys::fs::exists(control.restartInput))
    report.warnings.push_back(
        {"missing_input_checkpoint",
         "input checkpoint '" + control.restartInput +
             "' does not exist yet; complete the preceding stage or correct "
             "[input].checkpoint before running"});
  if ((!control.restartInput.empty() || control.checkpointPeriod > 0) &&
      !hasCheckpointSupport())
    report.warnings.push_back(
        {"hdf5_unavailable", "this build has no HDF5 support; use a build "
                              "with HDF5 to read or write checkpoints"});
  if (control.target == Target::GPU && !MDIR_HAS_CUDA)
    report.warnings.push_back(
        {"gpu_unavailable", "this build has no CUDA target; use a CUDA build "
                            "or select target = \"CPU\""});
  return report;
}

static void describeRun(const Control &control, const System &system,
                        const Preflight &report) {
  std::printf("ensemble:           %s\n", getEnsemble(control));
  if (control.minimize) {
    std::printf("minimizer:          STEEPEST_DESCENT, %lld steps\n",
                static_cast<long long>(control.numSteps));
  } else {
    std::printf("integrator:         %s, %lld steps of %g ps\n",
                getName(control.integrator),
                static_cast<long long>(control.numSteps), control.timestep);
    std::printf("run length:         %g ns\n",
                control.numSteps * control.timestep / 1000.0);
    std::printf("temperature:        %g K (%s)\n", control.temperature,
                control.thermostat ? "bath" : "initial velocities if drawn");
    if (control.thermostat)
      std::printf("thermostat:         %s\n",
                  control.isLangevin()     ? "LANGEVIN"
                  : control.isNoseHoover() ? "NOSE-HOOVER"
                                           : "V-RESCALE");
    if (control.barostat)
      std::printf("barostat:           C-RESCALE, %s, %g atm\n",
                  control.semiIsotropic ? "SEMI_ISOTROPIC" : "ISOTROPIC",
                  control.pressure);
  }
  std::printf("cutoff:             %g Å\n", control.cutoffDistance);
  std::printf("electrostatics:     %s\n", getElectrostatics(control));
  if (control.pme) {
    if (control.pmeGrid[0] > 0)
      std::printf("PME grid:           %lld %lld %lld, order %lld\n",
                  static_cast<long long>(control.pmeGrid[0]),
                  static_cast<long long>(control.pmeGrid[1]),
                  static_cast<long long>(control.pmeGrid[2]),
                  static_cast<long long>(control.pmeOrder));
    else
      std::printf("PME grid:           automatic, max spacing %g Å, order %lld\n",
                  control.pmeMaxSpacing,
                  static_cast<long long>(control.pmeOrder));
    if (control.pmeAlpha > 0)
      std::printf("PME beta:           %g Å^-1\n", control.pmeAlpha);
    else
      std::printf("PME beta:           automatic, tolerance %g\n",
                  control.pmeAlphaTolerance);
  }
  std::printf("constraints:        %zu distances; hydrogen bonds %s, "
              "rigid water %s\n",
              system.numConstraints, control.rigidBonds ? "yes" : "no",
              control.fastWater ? "yes" : "no");
  std::printf("target:             %s, %s precision\n",
              control.target == Target::GPU ? "gpu" : "cpu",
              getName(control.precision));
  if (control.target == Target::CPU)
    std::printf("threads:            %lld\n",
                static_cast<long long>(control.threads));
  if (!control.restartInput.empty())
    std::printf("input checkpoint:   %s (not loaded by check)\n",
                control.restartInput.c_str());
  std::printf("outputs:\n");
  for (const OutputFile &output : report.outputs) {
    bool log = llvm::StringRef(output.kind) == "log";
    std::string path = output.path.empty() ? "not configured" : output.path;
    if (log)
      path = output.path.empty() ? "stdout" : "stdout and " + output.path;
    std::printf("  %s: %s (%s)", output.kind, path.c_str(), output.format);
    if (!output.enabled)
      std::printf(", disabled");
    else if (output.atEnd)
      std::printf(", at the end");
    else if (output.interval == 0)
      std::printf(", no rows");
    else
      std::printf(", every %lld steps", static_cast<long long>(output.interval));
    if (output.enabled && output.count >= 0 && !output.atEnd)
      std::printf(", %lld %s", static_cast<long long>(output.count),
                  output.countOf);
    if (output.exists)
      std::printf(", exists");
    std::printf("\n");
  }
  for (const Warning &note : report.notes)
    llvm::errs() << "mdir: note: " << note.message << "\n";
  for (const Warning &warning : report.warnings)
    llvm::errs() << "mdir: warning: " << warning.message << "\n";
}

static llvm::json::Object makeJSON(const Control &control, const System &system,
                                  const Preflight &report) {
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;
  Object particles{
      {"particles", system.getNumParticles()},
      {"types", system.topology ? system.topology->getNumTypes()
                                : control.types.size()},
      {"degrees_of_freedom", system.getDegreesOfFreedom()},
      {"total_mass_amu", std::accumulate(system.masses.begin(),
                                         system.masses.end(), 0.0)},
      {"given_velocities", system.givenVelocities},
      {"restrained_particles", std::count_if(
          system.restraintConstants.begin(), system.restraintConstants.end(),
          [](double k) { return k > 0.0; })},
      {"periodic", control.periodic}, {"cell_angstrom", nullptr}};
  if (control.periodic) {
    Array diagonal, tilt;
    for (int k = 0; k != 3; ++k) {
      diagonal.push_back(system.box[k] / units::length);
      tilt.push_back(system.tilt[k] / units::length);
    }
    particles["cell_angstrom"] = Object{{"diagonal", std::move(diagonal)},
                                         {"tilt", std::move(tilt)}};
  }
  if (system.topology) {
    const Topology &topology = *system.topology;
    particles["topology"] = Object{
        {"path", !control.prmtopFile.empty() ? control.prmtopFile
                     : !control.charmmStructureFile.empty()
                           ? control.charmmStructureFile
                           : control.gromacsTopologyFile},
        {"residues", topology.residueNames.size()},
        {"bonds", topology.bonds.size()}, {"angles", topology.angles.size()},
        {"hydrogen_bonds", std::count_if(
            topology.bonds.begin(), topology.bonds.end(),
            [](const Topology::Bond &bond) { return bond.hydrogen; })},
        {"dihedrals", topology.dihedrals.size()},
        {"improper_dihedrals", std::count_if(
            topology.dihedrals.begin(), topology.dihedrals.end(),
            [](const Topology::Dihedral &dihedral) { return dihedral.improper; })},
        {"urey_bradley_terms", topology.ureyBradleys.size()},
        {"harmonic_impropers", topology.harmonicImpropers.size()},
        {"pairs_1_4", topology.pairs.size()},
        {"excluded_pairs", topology.exclusions.size()},
        {"cmap_terms", topology.cmaps.size()},
        {"cmap_maps", topology.cmapGrids.size()},
        {"total_charge_e", std::accumulate(topology.charges.begin(),
                                           topology.charges.end(), 0.0)},
        {"virtual_sites", topology.virtualSites.size()},
        {"rigid_waters", topology.settles.size()}};
  } else {
    particles["pair_terms"] = control.pairs.size();
  }
  Object run{
      {"kind", control.minimize ? "minimization" : "dynamics"},
      {"ensemble", getEnsemble(control)}, {"steps", control.numSteps},
      {"integrator", control.minimize ? "STEEPEST_DESCENT"
                                      : getName(control.integrator)},
      {"time_step_ps", control.minimize ? Value(nullptr) : Value(control.timestep)},
      {"duration_ns", control.minimize ? Value(nullptr)
                         : Value(control.numSteps * control.timestep / 1000.0)},
      {"temperature_kelvin", control.minimize ? Value(nullptr)
                                               : Value(control.temperature)},
      {"pressure_atm", control.barostat ? Value(control.pressure) : Value(nullptr)},
      {"thermostat", !control.thermostat ? "NONE"
                         : control.isLangevin() ? "LANGEVIN"
                         : control.isNoseHoover() ? "NOSE-HOOVER" : "V-RESCALE"},
      {"barostat_coupling", !control.barostat ? "NONE"
                          : control.semiIsotropic ? "SEMI_ISOTROPIC" : "ISOTROPIC"},
      {"cutoff_angstrom", control.cutoffDistance},
      {"electrostatics", getElectrostatics(control)},
      {"pme", nullptr},
      {"constraints", Object{{"distances", system.numConstraints},
                              {"hydrogen_bonds", control.rigidBonds},
                              {"rigid_water", control.fastWater},
                              {"analytic_bonds", control.analyticBonds}}},
      {"target", control.target == Target::GPU ? "gpu" : "cpu"},
      {"precision", getName(control.precision)},
      {"threads", control.threads},
      {"input_checkpoint", control.restartInput.empty() ? Value(nullptr)
                                                       : Value(control.restartInput)}};
  if (control.pme) {
    Array grid;
    for (int k = 0; k != 3; ++k)
      grid.push_back(control.pmeGrid[k] ? Value(control.pmeGrid[k]) : Value(nullptr));
    run["pme"] = Object{
        {"grid_points", std::move(grid)}, {"order", control.pmeOrder},
        {"max_spacing_angstrom", control.pmeMaxSpacing},
        {"beta_inverse_angstrom", control.pmeAlpha > 0 ? Value(control.pmeAlpha)
                                                       : Value(nullptr)},
        {"tolerance", control.pmeAlphaTolerance}};
  }
  Array outputs, warnings, notes;
  for (const OutputFile &output : report.outputs)
    outputs.push_back(Object{{"kind", output.kind},
                             {"path", output.path.empty() ? Value(nullptr)
                                                          : Value(output.path)},
                             {"format", output.format},
                             {"interval_steps", output.interval},
                             {"count", output.count >= 0 ? Value(output.count)
                                                         : Value(nullptr)},
                             {"count_of", output.countOf},
                             {"enabled", output.enabled},
                             {"at_end", output.atEnd},
                             {"exists", output.exists},
                             {"backup", output.backup.empty()
                                            ? Value(nullptr)
                                            : Value(output.backup)}});
  for (const Warning &warning : report.warnings)
    warnings.push_back(Object{{"code", warning.code}, {"message", warning.message}});
  for (const Warning &note : report.notes)
    notes.push_back(Object{{"code", note.code}, {"message", note.message}});
  return Object{{"schema_version", 1}, {"ok", true},
          {"system", std::move(particles)}, {"run", std::move(run)},
          {"outputs", std::move(outputs)}, {"warnings", std::move(warnings)},
          {"notes", std::move(notes)}, {"errors", Array{}}};
}

int mdir::tool::checkControl(llvm::StringRef controlFile, bool json) {
  auto control = readControl(controlFile);
  if (!control)
    return fail(control.takeError(), json);
  auto system = readSystem(*control);
  if (!system)
    return fail(system.takeError(), json);
  Preflight report = inspect(*control);
  for (const auto &[code, message] : system->warnings)
    report.warnings.push_back({code, message});
  if (json) {
    printJSON(makeJSON(*control, *system, report));
  } else {
    if (system->topology)
      describeTopology(*control, *system);
    else
      describeParticles(*control, *system);
    describeRun(*control, *system, report);
  }
  return 0;
}

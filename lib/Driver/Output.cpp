// What a run writes: the log, files of columns, the trajectory, and
// checkpoints (D149, docs/driver-m0.md, Section 2.8).

#include "mdir/Driver/Output.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <vector>

using namespace mdir::driver;

//===----------------------------------------------------------------------===//
// The log and files of columns
//===----------------------------------------------------------------------===//

Log::~Log() { close(); }

void Log::print(const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  va_list copy;
  va_copy(copy, arguments);
  int size = std::vsnprintf(nullptr, 0, format, arguments);
  va_end(arguments);
  std::string text(std::max(size, 0), '\0');
  if (size > 0)
    std::vsnprintf(text.data(), size + 1, format, copy);
  va_end(copy);
  std::fputs(text.c_str(), stdout);
  // Kept for the file, if one opens.
  if (file)
    std::fputs(text.c_str(), file);
  else
    pending += text;
}

llvm::Error Log::open(const std::string &path, bool appends) {
  close();
  file = std::fopen(path.c_str(), appends ? "a" : "w");
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write the log '%s'", path.c_str());
  std::fputs(pending.c_str(), file);
  pending.clear();
  std::fflush(file);
  return llvm::Error::success();
}

void Log::flush() {
  std::fflush(stdout);
  if (file)
    std::fflush(file);
}

void Log::close() {
  if (file)
    std::fclose(file);
  file = nullptr;
}

ColumnFile::~ColumnFile() { close(); }

std::string ColumnFile::getHeader(llvm::ArrayRef<Column> columns) {
  std::string names = "#", units = "#";
  for (const Column &column : columns) {
    names += " " + column.name;
    units += " " + column.unit;
  }
  return names + "\n" + units + "\n";
}

llvm::Error ColumnFile::open(const std::string &path,
                             std::vector<Column> given,
                             std::optional<int64_t> keepThrough) {
  close();
  columns = std::move(given);
  std::string header = getHeader(columns);
  // A continued run keeps the rows up to its checkpoint; those after it
  // the run computes again.
  std::string kept;
  if (keepThrough && llvm::sys::fs::exists(path)) {
    std::ifstream old(path);
    std::string first, second;
    std::getline(old, first);
    std::getline(old, second);
    if (first + "\n" + second + "\n" != header)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "'%s' does not begin with the columns that the run writes, '%s'; "
          "it is not the output of this run",
          path.c_str(), llvm::StringRef(header).rtrim().str().c_str());
    for (std::string line; std::getline(old, line);) {
      if (std::strtoll(line.c_str(), nullptr, 10) > *keepThrough)
        break;
      kept += line + "\n";
    }
  }
  file = std::fopen(path.c_str(), "w");
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write '%s'", path.c_str());
  std::fputs(header.c_str(), file);
  std::fputs(kept.c_str(), file);
  std::fflush(file);
  return llvm::Error::success();
}

void ColumnFile::write(int64_t step, llvm::ArrayRef<double> values) {
  if (!file)
    return;
  std::fprintf(file, "%lld", static_cast<long long>(step));
  for (size_t i = 0, e = values.size(); i != e; ++i) {
    if (i + 1 < columns.size() && columns[i + 1].integer)
      std::fprintf(file, " %lld", static_cast<long long>(values[i]));
    else
      std::fprintf(file, " %.6f", values[i]);
  }
  std::fprintf(file, "\n");
  std::fflush(file);
}

void ColumnFile::close() {
  if (file)
    std::fclose(file);
  file = nullptr;
}

std::string mdir::driver::getBackupPath(llvm::StringRef path) {
  llvm::StringRef directory = llvm::sys::path::parent_path(path);
  llvm::StringRef name = llvm::sys::path::filename(path);
  for (int n = 1; n <= MaxBackups; ++n) {
    llvm::SmallString<256> backup(directory);
    llvm::sys::path::append(backup, "#" + name + "." + llvm::Twine(n) + "#");
    if (!llvm::sys::fs::exists(backup))
      return std::string(backup);
  }
  return "";
}

llvm::Error mdir::driver::checkBackup(const std::string &path) {
  if (!llvm::sys::fs::exists(path) || !getBackupPath(path).empty())
    return llvm::Error::success();
  std::string name = llvm::sys::path::filename(path).str();
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "'%s' exists, and its directory holds %d backups of it already, "
      "'#%s.1#' to '#%s.%d#'; remove some of them, or continue the run "
      "with --continue",
      path.c_str(), MaxBackups, name.c_str(), name.c_str(), MaxBackups);
}

llvm::Expected<std::string> mdir::driver::backUpOutput(const std::string &path) {
  if (!llvm::sys::fs::exists(path))
    return "";
  if (llvm::Error error = checkBackup(path))
    return std::move(error);
  std::string backup = getBackupPath(path);
  if (std::error_code error = llvm::sys::fs::rename(path, backup))
    return llvm::createStringError(error, "cannot back up '%s' as '%s': %s",
                                   path.c_str(), backup.c_str(),
                                   error.message().c_str());
  return backup;
}

//===----------------------------------------------------------------------===//
// The log
//===----------------------------------------------------------------------===//

static Output *current = nullptr;

volatile std::sig_atomic_t mdir::driver::stopSignal = 0;

void mdir::driver::setOutput(Output *output) { current = output; }

void mdir::driver::writeLogHeader(Output &output) {
  if (output.minimizes) {
    output.log.print("INFO: %9s %14s %14s %14s %9s %14s\n", "STEP",
                 "POTENTIAL_ENE", "RMS_FORCE", "MAX_FORCE", "MAX_ATOM",
                 "STEP_SIZE");
    return;
  }
  if (output.overdamped)
    output.log.print("INFO: %9s %14s %14s %14s", "STEP", "TIME",
                     "POTENTIAL_ENE", "VIRIAL");
  else
    output.log.print("INFO: %9s %14s %14s %14s %14s %14s %14s",
                 "STEP", "TIME", "TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE",
                 "TEMPERATURE", "VIRIAL");
  // Without a periodic cell, the cell around the particles has no pressure
  // (D142).
  if (output.periodic)
    output.log.print(" %14s", "PRESSURE");
  if (output.couples)
    output.log.print(" %14s", "CONSERVED");
  if (output.changesCell)
    output.log.print(" %14s %14s", "VOLUME", "AREA_XY");
  output.log.print("\n");
}

std::vector<ColumnFile::Column>
mdir::driver::getEnergyColumns(const Output &output) {
  using Column = ColumnFile::Column;
  if (output.minimizes)
    return {{"step", "-", true},          {"potential", "kcal/mol"},
            {"rms_force", "kcal/mol/Å"}, {"max_force", "kcal/mol/Å"},
            {"max_atom", "-", true},      {"step_size", "Å"}};
  std::vector<Column> columns = {
      {"step", "-", true},       {"time", "ps"},
      {"total", "kcal/mol"},     {"potential", "kcal/mol"},
      {"kinetic", "kcal/mol"},   {"temperature", "K"},
      {"virial", "kcal/mol"}};
  if (output.overdamped)
    columns = {{"step", "-", true},
               {"time", "ps"},
               {"potential", "kcal/mol"},
               {"virial", "kcal/mol"}};
  if (output.periodic)
    columns.push_back({"pressure", "atm"});
  if (output.couples)
    columns.push_back({"conserved", "kcal/mol"});
  if (output.changesCell) {
    columns.push_back({"volume", "Å^3"});
    columns.push_back({"area_xy", "Å^2"});
  }
  return columns;
}

void _mlir_ciface_mdrtWriteVirial(double xx, double yy, double zz) {
  Output &output = *current;
  // The virials of the correction for the dispersion and of the background
  // of a net charge are isotropic: a third of each on each axis.
  double constant =
      (output.getDispersionVirial() + output.getCoulombConstantVirial()) / 3.0;
  output.log.print(
               "MDIR: the diagonal of the virial at the start, without the "
               "constraints, in kcal/mol:\nMDIR:   %16.6f %16.6f %16.6f\n",
               (xx + constant) / units::energy, (yy + constant) / units::energy,
               (zz + constant) / units::energy);
}

void _mlir_ciface_mdrtWriteTerms(void *terms) {
  Output &output = *current;
  auto *values = static_cast<StridedMemRefType<double, 1> *>(terms);
  static const char *names[] = {
      "Lennard-Jones", "Coulomb", "bonds", "angles", "dihedrals",
      "Lennard-Jones 1-4", "Coulomb 1-4", "CMAP", "Coulomb excluded",
      "Coulomb reciprocal", "Urey-Bradley", "harmonic impropers",
      "Lennard-Jones excluded", "Lennard-Jones reciprocal", "restraints"};
  // CMAP, Urey–Bradley, and harmonic impropers only where the topology has
  // them, and the terms of particle mesh Ewald only with it; with it
  // "Coulomb" is the direct sum, and with that of the dispersion (D162)
  // "Lennard-Jones" is.
  const Topology *topology =
      output.system ? output.system->topology.get() : nullptr;
  bool cmap = topology && !topology->cmaps.empty();
  bool ureyBradley = topology && !topology->ureyBradleys.empty();
  bool impropers = topology && !topology->harmonicImpropers.empty();
  output.log.print("MDIR: the terms at the start, in kcal/mol:\n");
  double total = output.getDispersionEnergy() +
                 output.getCoulombConstantEnergy() + output.ljpmeSelfEnergy;
  // The terms given by expressions follow those of the topology under
  // their names, those over tuples (D136) and then those over pairs (D137),
  // generalized Born (D144), and those of the positions (D148), and the
  // restraints come last.
  int custom = topology ? static_cast<int>(topology->tupleTerms.size()) : 0;
  int pairs = topology ? static_cast<int>(output.system->pairTermNames.size())
                       : 0;
  int born = topology ? static_cast<int>(output.system->bornTermNames.size())
                      : 0;
  int external =
      topology ? static_cast<int>(topology->externalTerms.size()) : 0;
  for (int i = 0, e = static_cast<int>(values->sizes[0]); i != e; ++i) {
    if ((i == 7 && !cmap) ||
        (i == 8 && !output.pme && !output.reactionField) ||
        (i == 9 && !output.pme) ||
        (i == 10 && !ureyBradley) || (i == 11 && !impropers) ||
        ((i == 12 || i == 13) && !output.ljpme))
      continue;
    std::string name =
        i < 14            ? names[i]
        : i < 14 + custom ? topology->tupleTerms[i - 14].name
        : i < 14 + custom + pairs
            ? output.system->pairTermNames[i - 14 - custom]
        : i < 14 + custom + pairs + born
            ? output.system->bornTermNames[i - 14 - custom - pairs]
        : i < 14 + custom + pairs + born + external
            ? topology->externalTerms[i - 14 - custom - pairs - born].name
            : names[14];
    double value = values->data[i * values->strides[0]];
    total += value;
    output.log.print("MDIR:   %-22s %16.6f\n", name.c_str(),
                 value / units::energy);
  }
  if (output.pme || output.reactionField)
    output.log.print("MDIR:   %-22s %16.6f\n", "Coulomb self",
                 output.getCoulombConstantEnergy() / units::energy);
  if (output.ljpme)
    output.log.print("MDIR:   %-22s %16.6f\n", "Lennard-Jones self",
                     output.ljpmeSelfEnergy / units::energy);
  else
    output.log.print("MDIR:   %-22s %16.6f\n", "dispersion",
                     output.getDispersionEnergy() / units::energy);
  output.log.print("MDIR:   %-22s %16.6f\n", "total",
               total / units::energy);
}

void _mlir_ciface_mdrtWritePull(int64_t step, void *coordinates,
                                void *terms) {
  Output &output = *current;
  if (!output.pull.isOpen())
    return;
  // For each term its coordinates, then its energy and the forces along
  // them, in the order of the header, in kcal/mol and per Å or radian.
  auto *q = static_cast<StridedMemRefType<double, 1> *>(coordinates);
  auto *e = static_cast<StridedMemRefType<double, 1> *>(terms);
  std::vector<double> row = {output.getTime(step)};
  int64_t column = 0, slot = 0;
  for (int64_t count : output.pullCounts) {
    for (int64_t k = 0; k != count; ++k)
      row.push_back(q->data[(column + k) * q->strides[0]]);
    for (int64_t k = 0; k != count + 1; ++k)
      row.push_back(e->data[(slot + k) * e->strides[0]]);
    column += count;
    slot += count + 1;
  }
  output.pull.write(step, row);
}

void _mlir_ciface_mdrtAddBath(double energy) { current->bath += energy; }

double Output::getChainEnergy() const {
  size_t m = chainMasses.size();
  double energy = 0.0;
  for (size_t j = 0; j != m; ++j) {
    double v = chain[m + j];
    energy += 0.5 * chainMasses[j] * v * v +
              (j == 0 ? chainFreedom : 1.0) * chainKT * chain[j];
  }
  return energy;
}

double mdrtNoseHooverFactor(double kinetic) {
  // The action of a Nose-Hoover chain over the time h of a period of
  // coupling, factorized as in Martyna, Tuckerman, Tobias, and Klein, Mol.
  // Phys. 87, 1117 (1996): by the Suzuki-Yoshida weights of order 6 (seven
  // parts w_k h, which sum to h), each a symmetric sequence of half-steps
  // of the velocities v_j of the thermostats from the end of the chain to
  // the first, a scaling of the particle velocities by exp(-v_1 w_k h), and
  // the reverse. The forces on the thermostats are G_1 = (2K - N_f k_B T)
  // / Q_1 and G_j = (Q_{j-1} v_{j-1}^2 - k_B T) / Q_j (Martyna, Klein, and
  // Tuckerman, J. Chem. Phys. 97, 2635 (1992), eq. 2.9).
  Output &output = *current;
  std::vector<double> &chain = output.chain;
  const std::vector<double> &q = output.chainMasses;
  size_t m = q.size();
  if (m == 0 || !(kinetic > 0.0))
    return 1.0;
  double *xi = chain.data(), *v = chain.data() + m;
  double kT = output.chainKT, freedom = output.chainFreedom;
  double before = output.getChainEnergy();
  static const double weights[7] = {
      0.784513610477560, 0.235573213359357, -1.17767998417887,
      1.31518632068391,  -1.17767998417887, 0.235573213359357,
      0.784513610477560};
  auto force = [&](size_t j, double k) {
    if (j == 0)
      return (2.0 * k - freedom * kT) / q[0];
    return (q[j - 1] * v[j - 1] * v[j - 1] - kT) / q[j];
  };
  double scale = 1.0, k = kinetic;
  int parts = output.chainSubsteps;
  for (int part = 0; part != parts; ++part) {
    for (double weight : weights) {
      double s = weight * output.chainTime / parts;
      // The velocities of the thermostats over s / 2 on each side of the
      // scaling, from the end of the chain to its start and back, each damped
      // by the one after it.
      v[m - 1] += 0.5 * s * force(m - 1, k);
      for (size_t j = m - 1; j-- > 0;) {
        double damp = std::exp(-0.25 * s * v[j + 1]);
        v[j] = (v[j] * damp + 0.5 * s * force(j, k)) * damp;
      }
      double factor = std::exp(-s * v[0]);
      scale *= factor;
      k *= factor * factor;
      for (size_t j = 0; j != m; ++j)
        xi[j] += s * v[j];
      for (size_t j = 0; j + 1 < m; ++j) {
        double damp = std::exp(-0.25 * s * v[j + 1]);
        v[j] = (v[j] * damp + 0.5 * s * force(j, k)) * damp;
      }
      v[m - 1] += 0.5 * s * force(m - 1, k);
    }
  }
  // The conserved energy is that of the system with the energy of the
  // chain; what the scaling takes from the particles is in the chain, so
  // the bath counts the change of the chain's energy.
  output.bath += output.getChainEnergy() - before;
  output.checkpoint.thermostatState = chain;
  return scale;
}

void _mlir_ciface_mdrtSetBarostatState(double w0, double w1, double w2,
                                       double g0, double g1, double g2,
                                       double k0, double k1, double k2) {
  current->checkpoint.barostatState = {w0, w1, w2, g0, g1, g2, k0, k1, k2};
}

void _mlir_ciface_mdrtSetBox(double lx, double ly, double lz) {
  Output &output = *current;
  output.box[0] = lx;
  output.box[1] = ly;
  output.box[2] = lz;
  output.volume = lx * ly * lz;
  double edges[3] = {lx / units::length, ly / units::length,
                     lz / units::length};
  if (output.trajectory)
    output.trajectory->setBox(edges);
  output.checkpoint.box[0] = lx;
  output.checkpoint.box[1] = ly;
  output.checkpoint.box[2] = lz;
  static const char axes[] = "xyz";
  for (int k = 0; k != 3; ++k)
    if (output.box[k] < output.leastEdge) {
      std::fprintf(stderr,
                   "mdir: the barostat has made the cell %.4f Å along %c, "
                   "less than twice the cutoff, %.4f Å; the run needs a "
                   "larger cell\n",
                   output.box[k] / units::length, axes[k],
                   0.5 * output.leastEdge / units::length);
      std::exit(1);
    }
}

void _mlir_ciface_mdrtSetTilt(double bx, double cx, double cy) {
  Output &output = *current;
  double tilts[3] = {bx / units::length, cx / units::length,
                     cy / units::length};
  if (output.trajectory)
    output.trajectory->setTilt(tilts);
  output.checkpoint.tilt[0] = bx;
  output.checkpoint.tilt[1] = cx;
  output.checkpoint.tilt[2] = cy;
}

void _mlir_ciface_mdrtWriteEnergies(int64_t step, double potential,
                                    double kinetic, double forceSquare,
                                    double virial) {
  Output &output = *current;
  if (output.energyPeriod > 0 &&
      (step - output.firstStep) % output.energyPeriod != 0)
    return;
  // `kinetic` is that of the velocities at the step. The total energy has
  // it, because that sum varies least.
  // The correction for the dispersion is a number of the volume.
  potential += output.getDispersionEnergy() +
               output.getCoulombConstantEnergy() + output.ljpmeSelfEnergy;
  virial += output.getDispersionVirial() + output.getCoulombConstantVirial();
  double total = potential + kinetic;

  // The mean of the kinetic energies half a step before and after exceeds
  // `kinetic` by (dt^2 / 8) sum F^2 / m. The pressure takes that mean, and
  // the temperature the mean of all three (D45). See Jung et al., J. Chem.
  // Phys. 148, 164109 (2018), and J. Chem. Theory Comput. 15, 84 (2019).
  double excess =
      0.125 * output.timestep * output.timestep * forceSquare;
  double half = kinetic + excess;
  double optimal = kinetic + 2.0 * excess / 3.0;
  double temperature =
      2.0 * optimal / (output.degreesOfFreedom * units::boltzmann);
  // Brownian dynamics has no momenta: the pressure takes those of the
  // bath (D163b).
  if (output.overdamped)
    half = output.bathKinetic;
  // `virial` is the trace of W, the sum of d (x) K over the pairs (B8).
  double pressure = (2.0 * half + virial) / (3.0 * output.volume);
  // The columns of the log, and of the file of the energies (D149).
  std::vector<double> row = {output.getTime(step),
                             total / units::energy,
                             potential / units::energy,
                             kinetic / units::energy,
                             temperature,
                             virial / units::energy};
  if (output.overdamped)
    row = {output.getTime(step), potential / units::energy,
           virial / units::energy};
  if (output.periodic)
    row.push_back(pressure * units::pressure);
  if (output.couples) {
    total += output.bath;
    row.push_back(total / units::energy);
  }
  if (output.changesCell) {
    row.push_back(output.volume /
                  (units::length * units::length * units::length));
    // The reduced cell has a = (Lx, 0, 0), b = (bx, Ly, 0), so
    // |a × b| = Lx Ly even with a tilt (D170).
    row.push_back(output.box[0] * output.box[1] /
                  (units::length * units::length));
  }
  output.log.print("INFO: %9lld", static_cast<long long>(step));
  for (double value : row)
    output.log.print(" %14.4f", value);
  output.log.print("\n");
  output.log.flush();
  output.energies.write(step, row);

  if (!output.hasEnergies)
    output.firstTotal = total;
  output.hasEnergies = true;
  output.lastTotal = total;
  output.energyTimes.emplace_back(
      step, std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
}

/// The values of a buffer with three numbers per particle, scaled, in the
/// order of the numbers of the particles, which `ids` holds. `element` is
/// the type that the buffer holds.
static std::vector<double> readVectors(void *descriptor, void *ids,
                                       Element element, double scale = 1.0) {
  auto *buffer = static_cast<StridedMemRefType<char, 2> *>(descriptor);
  auto *numbers = static_cast<StridedMemRefType<int32_t, 1> *>(ids);
  int64_t count = buffer->sizes[0];
  std::vector<double> values(3 * count);
  for (int64_t i = 0; i != count; ++i) {
    int64_t place =
        numbers->data[numbers->offset + i * numbers->strides[0]];
    for (int64_t c = 0; c != 3; ++c) {
      int64_t index =
          buffer->offset + i * buffer->strides[0] + c * buffer->strides[1];
      double value;
      if (element == Element::F32)
        value = reinterpret_cast<float *>(buffer->data)[index];
      else
        value = reinterpret_cast<double *>(buffer->data)[index];
      values[3 * place + c] = scale * value;
    }
  }
  return values;
}

void _mlir_ciface_mdrtWriteMinimization(int64_t step, double energy,
                                        double size, void *forces,
                                        void *ids) {
  Output &output = *current;
  output.lastMinimizationStep = step;
  std::vector<double> values = readVectors(forces, ids, output.force);
  const std::vector<double> &masses = output.system->masses;
  double square = 0.0, largest = 0.0;
  size_t counted = 0, where = 0;
  for (size_t i = 0, e = masses.size(); i != e; ++i) {
    if (masses[i] == 0.0)
      continue;
    double f2 = values[3 * i] * values[3 * i] +
                values[3 * i + 1] * values[3 * i + 1] +
                values[3 * i + 2] * values[3 * i + 2];
    square += f2;
    ++counted;
    if (f2 > largest) {
      largest = f2;
      where = i;
    }
  }
  double scale = units::energy / units::length;
  double rms = counted ? std::sqrt(square / counted) : 0.0;
  energy += output.getDispersionEnergy() +
            output.getCoulombConstantEnergy() + output.ljpmeSelfEnergy;
  output.log.print("INFO: %9lld %14.4f %14.4f %14.4f %9zu %14.6f\n",
               static_cast<long long>(step), energy / units::energy,
               rms / scale, std::sqrt(largest) / scale, where + 1,
               size / units::length);
  output.log.flush();
  output.energies.write(step, {energy / units::energy, rms / scale,
                               std::sqrt(largest) / scale,
                               static_cast<double>(where + 1),
                               size / units::length});
  if (!output.hasEnergies)
    output.firstTotal = energy;
  output.hasEnergies = true;
  output.lastTotal = energy;
}

/// Without a periodic cell (D142), stops the run if the particles have
/// spread so far along an axis that an image could come within the reach
/// of the neighbor structures: farther than the cell less that reach.
static void checkSpread(const Output &output, const std::vector<double> &x,
                        int64_t step) {
  if (output.periodic)
    return;
  static const char axes[] = "xyz";
  for (int k = 0; k != 3; ++k) {
    double least = x[k], most = x[k];
    for (size_t i = k; i < x.size(); i += 3) {
      least = std::min(least, x[i]);
      most = std::max(most, x[i]);
    }
    if (most - least > output.box[k] - output.listReach) {
      std::fprintf(stderr,
                   "mdir: at step %lld the particles spread %.4f Å along "
                   "%c, more than the cell around them less the reach of "
                   "the neighbor structures, %.4f Å; images of the "
                   "particles would interact. The run stops; begin it "
                   "again with a larger 'pairlist_distance', which places "
                   "a larger cell (D142)\n",
                   static_cast<long long>(step),
                   (most - least) / units::length, axes[k],
                   (output.box[k] - output.listReach) / units::length);
      std::exit(1);
    }
  }
}

void _mlir_ciface_mdrtCheckSpread(int64_t step, void *positions, void *ids) {
  Output &output = *current;
  checkSpread(output, readVectors(positions, ids, output.state), step);
}

void _mlir_ciface_mdrtWriteFrame(int64_t step, void *positions, void *ids) {
  Output &output = *current;
  if (!output.hasTrajectory)
    return;
  std::vector<double> values =
      readVectors(positions, ids, output.state, 1.0 / units::length);
  checkSpread(output, readVectors(positions, ids, output.state), step);
  std::vector<float> narrow(values.begin(), values.end());
  output.trajectory->writeFrame(narrow.data(), step, output.getTime(step));
}

void _mlir_ciface_mdrtFinish(void *positions, void *velocities,
                             void *ids) {
  Output &output = *current;
  if (!output.system)
    return;
  output.system->positions = readVectors(positions, ids, output.state);
  checkSpread(output, output.system->positions, output.endStep);
  output.system->velocities = readVectors(velocities, ids, output.state);
}

/// Writes the state as a checkpoint. `forces` is null if the state has
/// none.
static void writeState(int64_t step, void *positions, void *velocities,
                       void *forces, void *ids) {
  Output &output = *current;
  if (output.checkpointPath.empty())
    return;
  Checkpoint &checkpoint = output.checkpoint;
  checkpoint.step = step;
  checkpoint.time = output.getTime(step);
  checkpoint.positions = readVectors(positions, ids, output.state);
  checkSpread(output, checkpoint.positions, step);
  checkpoint.velocities = readVectors(velocities, ids, output.state);
  checkpoint.forces.clear();
  if (forces)
    checkpoint.forces = readVectors(forces, ids, output.force);

  checkpoint.frames = output.hasTrajectory ? output.trajectory->getNumFrames()
                                           : 0;
  checkpoint.bath = output.bath;

  if (llvm::Error error =
          writeCheckpoint(output.checkpointPath, checkpoint)) {
    std::fprintf(stderr, "mdir: %s\n",
                 llvm::toString(std::move(error)).c_str());
    std::exit(1);
  }
  ++output.numCheckpoints;

  // A stop that a signal or the wall time asks for is taken here, where
  // the run continues exactly from what it has just written (D131), and
  // not at the last step, where the run ends anyway. The wall time must
  // leave room for one more interval between checkpoints, as long as the
  // longest so far.
  if (step >= output.endStep)
    return;
  auto now = std::chrono::steady_clock::now();
  double segment =
      std::chrono::duration<double>(now - output.lastCheckpoint).count();
  output.lastCheckpoint = now;
  output.longestSegment = std::max(output.longestSegment, segment);
  double elapsed = std::chrono::duration<double>(now - output.began).count();
  const char *reason = nullptr;
  if (stopSignal == SIGTERM)
    reason = "SIGTERM";
  else if (stopSignal == SIGINT)
    reason = "SIGINT";
  else if (output.maxWalltime > 0.0 &&
           elapsed + output.longestSegment > output.maxWalltime)
    reason = "the wall time";
  if (!reason)
    return;
  if (output.trajectory)
    output.trajectory->close();
  output.log.print(
               "MDIR: stopped after step %lld on %s; '%s' holds the state, "
               "and `mdir run --continue` goes on from it\n",
               static_cast<long long>(step), reason,
               output.checkpointPath.c_str());
  output.log.flush();
  std::fprintf(stderr, "mdir: stopped after step %lld of %lld on %s\n",
               static_cast<long long>(step),
               static_cast<long long>(output.endStep), reason);
  if (output.recordStop)
    if (llvm::Error error = output.recordStop(step, reason)) {
      std::fprintf(stderr, "mdir: %s\n", llvm::toString(std::move(error)).c_str());
      std::exit(1);
    }
  std::exit(StoppedStatus);
}

void _mlir_ciface_mdrtWriteCheckpoint(int64_t step, void *positions,
                                      void *velocities, void *ids) {
  writeState(step, positions, velocities, nullptr, ids);
}

void _mlir_ciface_mdrtWriteCheckpointWithForces(int64_t step,
                                                void *positions,
                                                void *velocities,
                                                void *forces, void *ids) {
  writeState(step, positions, velocities, forces, ids);
}

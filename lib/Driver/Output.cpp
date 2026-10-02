// What a run writes: the log, the trajectory, and checkpoints.

#include "mdir/Driver/Output.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

using namespace mdir::driver;

//===----------------------------------------------------------------------===//
// DCD
//===----------------------------------------------------------------------===//

/// Writes a record: its length in bytes, the bytes, and the length again.
static void writeRecord(std::FILE *file, const void *data, int32_t size) {
  std::fwrite(&size, sizeof(size), 1, file);
  std::fwrite(data, 1, size, file);
  std::fwrite(&size, sizeof(size), 1, file);
}

DCDWriter::~DCDWriter() { close(); }

void DCDWriter::writeHeader() {
  // The first record: the tag, and 20 numbers that describe the file.
  struct {
    char tag[4];
    int32_t numbers[20];
  } head;
  std::memcpy(head.tag, "CORD", 4);
  std::memset(head.numbers, 0, sizeof(head.numbers));
  head.numbers[0] = numFrames;
  head.numbers[1] = static_cast<int32_t>(first);
  head.numbers[2] = static_cast<int32_t>(period);
  head.numbers[3] = numFrames * static_cast<int32_t>(period);
  // The time step, in units of 48.88821 fs.
  float delta = static_cast<float>(timestep / 0.04888821);
  std::memcpy(&head.numbers[9], &delta, sizeof(delta));
  // The frames hold the cell.
  head.numbers[10] = 1;
  // The version of the format.
  head.numbers[19] = 24;
  writeRecord(file, &head, sizeof(head));

  struct {
    int32_t count;
    char line[80];
  } title;
  title.count = 1;
  std::memset(title.line, ' ', sizeof(title.line));
  const char *text = "Written by MDIR";
  std::memcpy(title.line, text, std::strlen(text));
  writeRecord(file, &title, sizeof(title));

  int32_t count = static_cast<int32_t>(numParticles);
  writeRecord(file, &count, sizeof(count));
}

llvm::Error DCDWriter::open(const std::string &path, size_t numParticles,
                            int64_t first, int64_t period, double timestep,
                            const double box[3]) {
  file = std::fopen(path.c_str(), "wb");
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write '%s'", path.c_str());
  this->numParticles = numParticles;
  this->first = first;
  this->period = period;
  this->timestep = timestep;
  for (int i = 0; i != 3; ++i)
    this->box[i] = box[i];
  writeHeader();
  return llvm::Error::success();
}

llvm::Expected<int64_t> DCDWriter::append(const std::string &path,
                                          size_t numParticles,
                                          int64_t frames, int64_t period,
                                          double timestep,
                                          const double box[3]) {
  auto fail = [&](const llvm::Twine &message) -> llvm::Error {
    close();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "'" + path + "' " + message);
  };
  file = std::fopen(path.c_str(), "r+b");
  if (!file)
    return fail("cannot be opened to continue it");
  // The records that writeHeader writes: the description, the title, and
  // the number of particles.
  auto readInt = [&](int32_t &value) {
    return std::fread(&value, sizeof(value), 1, file) == 1;
  };
  int32_t size, end, count;
  struct {
    char tag[4];
    int32_t numbers[20];
  } head;
  bool read = readInt(size) && size == sizeof(head) &&
              std::fread(&head, sizeof(head), 1, file) == 1 && readInt(end) &&
              end == size && std::memcmp(head.tag, "CORD", 4) == 0;
  if (!read || head.numbers[10] != 1 || head.numbers[19] != 24)
    return fail("is not a trajectory that MDIR wrote");
  read = readInt(size) && size == 84 &&
         std::fseek(file, size, SEEK_CUR) == 0 && readInt(end) &&
         end == size && readInt(size) && size == 4 && readInt(count) &&
         readInt(end) && end == size;
  if (!read)
    return fail("is not a trajectory that MDIR wrote");
  if (count != static_cast<int32_t>(numParticles))
    return fail("holds " + llvm::Twine(count) + " particles, and the run " +
                llvm::Twine(numParticles));
  if (head.numbers[2] != static_cast<int32_t>(period))
    return fail("has a frame every " + llvm::Twine(head.numbers[2]) +
                " steps, and the run writes one every " +
                llvm::Twine(period));

  // The frames that the file holds in full: the cell, then x, y, and z.
  long header = std::ftell(file);
  long frameSize =
      (8 + 6 * sizeof(double)) + 3 * (8 + numParticles * sizeof(float));
  std::fseek(file, 0, SEEK_END);
  long length = std::ftell(file);
  int64_t held = (length - header) / frameSize;
  if (held < frames)
    return fail("holds " + llvm::Twine(held) + " frames, fewer than the " +
                llvm::Twine(frames) + " that the checkpoint counts");
  long kept = header + frames * frameSize;
  std::fflush(file);
  if (kept != length && ::ftruncate(::fileno(file), kept) != 0)
    return fail("cannot be cut to the frames that the checkpoint counts");

  this->numParticles = numParticles;
  this->first = head.numbers[1];
  this->period = period;
  this->timestep = timestep;
  for (int i = 0; i != 3; ++i)
    this->box[i] = box[i];
  numFrames = static_cast<int32_t>(frames);
  std::rewind(file);
  writeHeader();
  std::fseek(file, kept, SEEK_SET);
  std::fflush(file);
  return held - frames;
}

void DCDWriter::writeFrame(const float *positions) {
  // The cell: a, cos γ, b, cos β, cos α, c, as NAMD and OpenMM write it
  // (docs/triclinic-m2.md, Section 1); right angles for an orthorhombic
  // cell.
  double b = std::sqrt(tilt[0] * tilt[0] + box[1] * box[1]);
  double c = std::sqrt(tilt[1] * tilt[1] + tilt[2] * tilt[2] + box[2] * box[2]);
  double cell[6] = {box[0],
                    tilt[0] / b,
                    b,
                    tilt[1] / c,
                    0.0,
                    c};
  cell[4] = (tilt[0] * tilt[1] + box[1] * tilt[2]) / (b * c);
  writeRecord(file, cell, sizeof(cell));

  std::vector<float> component(numParticles);
  for (int c = 0; c != 3; ++c) {
    for (size_t i = 0; i != numParticles; ++i)
      component[i] = positions[3 * i + c];
    writeRecord(file, component.data(),
                static_cast<int32_t>(numParticles * sizeof(float)));
  }

  // Keep the number of frames in the header up to date, so that the file
  // can be read if the run ends early.
  ++numFrames;
  long end = std::ftell(file);
  std::rewind(file);
  writeHeader();
  std::fseek(file, end, SEEK_SET);
  std::fflush(file);
}

void DCDWriter::close() {
  if (file)
    std::fclose(file);
  file = nullptr;
}

//===----------------------------------------------------------------------===//
// The log
//===----------------------------------------------------------------------===//

static Output *current = nullptr;

volatile std::sig_atomic_t mdir::driver::stopSignal = 0;

void mdir::driver::setOutput(Output *output) { current = output; }

void mdir::driver::writeLogHeader(Output &output) {
  if (output.minimizes) {
    std::fprintf(output.log, "INFO: %9s %14s %14s %14s %9s %14s\n", "STEP",
                 "POTENTIAL_ENE", "RMS_FORCE", "MAX_FORCE", "MAX_ATOM",
                 "STEP_SIZE");
    return;
  }
  std::fprintf(output.log, "INFO: %9s %14s %14s %14s %14s %14s %14s %14s",
               "STEP", "TIME", "TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE",
               "TEMPERATURE", "VIRIAL", "PRESSURE");
  if (output.couples)
    std::fprintf(output.log, " %14s", "CONSERVED");
  if (output.changesCell)
    std::fprintf(output.log, " %14s", "VOLUME");
  std::fprintf(output.log, "\n");
}

void _mlir_ciface_mdrtWriteVirial(double xx, double yy, double zz) {
  Output &output = *current;
  // The virials of the correction for the dispersion and of the background
  // of a net charge are isotropic: a third of each on each axis.
  double constant =
      (output.getDispersionVirial() + output.getCoulombConstantVirial()) / 3.0;
  std::fprintf(output.log,
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
      "restraints"};
  // CMAP, Urey–Bradley, and harmonic impropers only where the topology has
  // them, and the terms of particle mesh Ewald only with it; with it
  // "Coulomb" is the direct sum.
  const Topology *topology =
      output.system ? output.system->topology.get() : nullptr;
  bool cmap = topology && !topology->cmaps.empty();
  bool ureyBradley = topology && !topology->ureyBradleys.empty();
  bool impropers = topology && !topology->harmonicImpropers.empty();
  std::fprintf(output.log, "MDIR: the terms at the start, in kcal/mol:\n");
  double total = output.getDispersionEnergy() + output.getCoulombConstantEnergy();
  // The terms given by expressions follow those of the topology under
  // their names, those over tuples (D136) and then those over pairs (D137),
  // and the restraints come last.
  int custom = topology ? static_cast<int>(topology->tupleTerms.size()) : 0;
  int pairs = topology ? static_cast<int>(output.system->pairTermNames.size())
                       : 0;
  for (int i = 0, e = static_cast<int>(values->sizes[0]); i != e; ++i) {
    if ((i == 7 && !cmap) ||
        (i == 8 && !output.pme && !output.reactionField) ||
        (i == 9 && !output.pme) ||
        (i == 10 && !ureyBradley) || (i == 11 && !impropers))
      continue;
    std::string name =
        i < 12            ? names[i]
        : i < 12 + custom ? topology->tupleTerms[i - 12].name
        : i < 12 + custom + pairs
            ? output.system->pairTermNames[i - 12 - custom]
            : names[12];
    double value = values->data[i * values->strides[0]];
    total += value;
    std::fprintf(output.log, "MDIR:   %-22s %16.6f\n", name.c_str(),
                 value / units::energy);
  }
  if (output.pme || output.reactionField)
    std::fprintf(output.log, "MDIR:   %-22s %16.6f\n", "Coulomb self",
                 output.getCoulombConstantEnergy() / units::energy);
  std::fprintf(output.log, "MDIR:   %-22s %16.6f\n", "dispersion",
               output.getDispersionEnergy() / units::energy);
  std::fprintf(output.log, "MDIR:   %-22s %16.6f\n", "total",
               total / units::energy);
}

void _mlir_ciface_mdrtAddBath(double energy) { current->bath += energy; }

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
  output.trajectory.setBox(edges);
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
  output.trajectory.setTilt(tilts);
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
  potential += output.getDispersionEnergy() + output.getCoulombConstantEnergy();
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
  // `virial` is the trace of W, the sum of d (x) K over the pairs (B8).
  double pressure = (2.0 * half + virial) / (3.0 * output.volume);
  std::fprintf(output.log,
               "INFO: %9lld %14.4f %14.4f %14.4f %14.4f %14.4f %14.4f "
               "%14.4f",
               static_cast<long long>(step), output.getTime(step),
               total / units::energy, potential / units::energy,
               kinetic / units::energy, temperature, virial / units::energy,
               pressure * units::pressure);
  if (output.couples) {
    total += output.bath;
    std::fprintf(output.log, " %14.4f", total / units::energy);
  }
  if (output.changesCell)
    std::fprintf(output.log, " %14.4f",
                 output.volume / (units::length * units::length *
                                  units::length));
  std::fprintf(output.log, "\n");
  std::fflush(output.log);

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
  energy += output.getDispersionEnergy() + output.getCoulombConstantEnergy();
  std::fprintf(output.log, "INFO: %9lld %14.4f %14.4f %14.4f %9zu %14.6f\n",
               static_cast<long long>(step), energy / units::energy,
               rms / scale, std::sqrt(largest) / scale, where + 1,
               size / units::length);
  std::fflush(output.log);
  if (!output.hasEnergies)
    output.firstTotal = energy;
  output.hasEnergies = true;
  output.lastTotal = energy;
}

void _mlir_ciface_mdrtWriteFrame(int64_t, void *positions, void *ids) {
  Output &output = *current;
  if (!output.hasTrajectory)
    return;
  std::vector<double> values =
      readVectors(positions, ids, output.state, 1.0 / units::length);
  std::vector<float> narrow(values.begin(), values.end());
  output.trajectory.writeFrame(narrow.data());
}

void _mlir_ciface_mdrtFinish(void *positions, void *velocities,
                             void *ids) {
  Output &output = *current;
  if (!output.system)
    return;
  output.system->positions = readVectors(positions, ids, output.state);
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
  checkpoint.velocities = readVectors(velocities, ids, output.state);
  checkpoint.forces.clear();
  if (forces)
    checkpoint.forces = readVectors(forces, ids, output.force);

  checkpoint.frames = output.hasTrajectory ? output.trajectory.getNumFrames()
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
  output.trajectory.close();
  std::fprintf(output.log,
               "MDIR: stopped after step %lld on %s; '%s' holds the state, "
               "and `mdir run --continue` goes on from it\n",
               static_cast<long long>(step), reason,
               output.checkpointPath.c_str());
  std::fflush(output.log);
  if (output.log != stdout)
    std::fflush(stdout);
  std::fprintf(stderr, "mdir: stopped after step %lld of %lld on %s\n",
               static_cast<long long>(step),
               static_cast<long long>(output.endStep), reason);
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

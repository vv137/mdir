// What a run writes: the log, the trajectory, and checkpoints.

#include "mdir/Driver/Output.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
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

void DCDWriter::writeFrame(const float *positions) {
  // The cell: the three edge lengths, and between them the cosines of the
  // angles, which are right angles.
  double cell[6] = {box[0], 0.0, box[1], 0.0, 0.0, box[2]};
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

void mdir::driver::setOutput(Output *output) { current = output; }

void mdir::driver::writeLogHeader(Output &output) {
  std::fprintf(output.log, "INFO: %9s %14s %14s %14s %14s %14s %14s %14s\n",
               "STEP", "TIME", "TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE",
               "TEMPERATURE", "VIRIAL", "PRESSURE");
}

void _mlir_ciface_mdrtWriteEnergies(int64_t step, double potential,
                                    double kinetic, double forceSquare,
                                    double virial) {
  Output &output = *current;
  // `kinetic` is that of the velocities at the step. The total energy has
  // it, because that sum varies least.
  double total = potential + kinetic;

  // The mean of the kinetic energies half a step before and after exceeds
  // `kinetic` by (dt^2 / 8) sum F^2 / m. The pressure takes that mean, and
  // the temperature the mean of all three (D45).
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
               "%14.4f\n",
               static_cast<long long>(step), output.getTime(step),
               total / units::energy, potential / units::energy,
               kinetic / units::energy, temperature, virial / units::energy,
               pressure * units::pressure);
  std::fflush(output.log);

  if (!output.hasEnergies)
    output.firstTotal = total;
  output.hasEnergies = true;
  output.lastTotal = total;
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

  if (llvm::Error error =
          writeCheckpoint(output.checkpointPath, checkpoint)) {
    std::fprintf(stderr, "mdir-run: %s\n",
                 llvm::toString(std::move(error)).c_str());
    std::exit(1);
  }
  ++output.numCheckpoints;
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

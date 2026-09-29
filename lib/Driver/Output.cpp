// What a run writes: the log and the trajectory.

#include "mdir/Driver/Output.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <cmath>
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
  head.numbers[1] = static_cast<int32_t>(period);
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
                            int64_t period, double timestep,
                            const double box[3]) {
  file = std::fopen(path.c_str(), "wb");
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write '%s'", path.c_str());
  this->numParticles = numParticles;
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
  std::fprintf(output.log, "INFO: %9s %14s %14s %14s %14s %14s\n", "STEP",
               "TIME", "TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE",
               "TEMPERATURE");
}

void _mlir_ciface_mdrtWriteEnergies(int64_t step, double potential,
                                    double kinetic) {
  Output &output = *current;
  double total = potential + kinetic;
  double temperature =
      2.0 * kinetic / (output.degreesOfFreedom * units::boltzmann);
  std::fprintf(output.log, "INFO: %9lld %14.4f %14.4f %14.4f %14.4f %14.4f\n",
               static_cast<long long>(step),
               static_cast<double>(step) * output.timestep,
               total / units::energy, potential / units::energy,
               kinetic / units::energy, temperature);
  std::fflush(output.log);

  if (!output.hasEnergies)
    output.firstTotal = total;
  output.hasEnergies = true;
  output.lastTotal = total;
}

/// The values of a buffer with three numbers per particle, converted to
/// the type `To` and scaled.
template <typename From, typename To>
static std::vector<To> readVectors(void *descriptor, double scale) {
  auto *buffer = static_cast<StridedMemRefType<From, 2> *>(descriptor);
  int64_t count = buffer->sizes[0];
  std::vector<To> values(3 * count);
  for (int64_t i = 0; i != count; ++i)
    for (int64_t c = 0; c != 3; ++c)
      values[3 * i + c] = static_cast<To>(
          scale * buffer->data[buffer->offset + i * buffer->strides[0] +
                               c * buffer->strides[1]]);
  return values;
}

template <typename From>
static void writeFrame(void *positions) {
  Output &output = *current;
  if (!output.hasTrajectory)
    return;
  std::vector<float> values =
      readVectors<From, float>(positions, 1.0 / units::length);
  output.trajectory.writeFrame(values.data());
}

template <typename From>
static void finish(void *positions, void *velocities) {
  Output &output = *current;
  if (!output.system)
    return;
  output.system->positions = readVectors<From, double>(positions, 1.0);
  output.system->velocities = readVectors<From, double>(velocities, 1.0);
}

void _mlir_ciface_mdrtWriteFrame_f32(int64_t, void *positions) {
  writeFrame<float>(positions);
}
void _mlir_ciface_mdrtWriteFrame_f64(int64_t, void *positions) {
  writeFrame<double>(positions);
}
void _mlir_ciface_mdrtFinish_f32(void *positions, void *velocities) {
  finish<float>(positions, velocities);
}
void _mlir_ciface_mdrtFinish_f64(void *positions, void *velocities) {
  finish<double>(positions, velocities);
}

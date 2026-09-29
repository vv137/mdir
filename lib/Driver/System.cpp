// The particles of a run.

#include "mdir/Driver/System.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cmath>
#include <cstdlib>

using namespace mdir::driver;
using llvm::StringRef;

llvm::Expected<System> mdir::driver::readSystem(const Control &control) {
  auto file = llvm::MemoryBuffer::getFile(control.pdbFile);
  if (!file)
    return llvm::createStringError(file.getError(), "cannot read '%s'",
                                   control.pdbFile.c_str());

  System system;
  for (int i = 0; i != 3; ++i)
    system.box[i] = control.box[i] * units::length;

  unsigned number = 0;
  StringRef rest = (*file)->getBuffer();
  while (!rest.empty()) {
    StringRef line;
    std::tie(line, rest) = rest.split('\n');
    ++number;
    if (!line.starts_with("ATOM") && !line.starts_with("HETATM"))
      continue;
    auto fail = [&](const llvm::Twine &message) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "%s:%u: %s", control.pdbFile.c_str(),
                                     number, message.str().c_str());
    };
    if (line.size() < 54)
      return fail("the line is too short to hold the coordinates");

    // Columns 13 to 16 hold the name, 31 to 54 the coordinates.
    StringRef name = line.substr(12, 4).trim();
    unsigned type = 0;
    while (type != control.types.size() && control.types[type].name != name)
      ++type;
    if (type == control.types.size())
      return fail("no [[energy.type]] is named '" + name + "'");

    for (int i = 0; i != 3; ++i) {
      StringRef field = line.substr(30 + 8 * i, 8).trim();
      double value;
      if (field.getAsDouble(value))
        return fail("expected a coordinate, got '" + field + "'");
      system.positions.push_back(value * units::length);
    }
    system.types.push_back(type);
    system.masses.push_back(control.types[type].mass);
  }

  if (system.types.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "%s: the file holds no atoms",
                                   control.pdbFile.c_str());
  system.velocities.assign(system.positions.size(), 0.0);
  return std::move(system);
}

namespace {

/// A generator of random numbers whose sequence is fixed by its definition,
/// not by a library: splitmix64 for the state, xoshiro256** for the numbers.
/// See Steele et al., OOPSLA '14, 453 (2014), and Blackman and Vigna, ACM
/// Trans. Math. Softw. 47(4), 1 (2021); docs/references.md.
class Generator {
public:
  explicit Generator(uint64_t seed) {
    for (uint64_t &word : state) {
      seed += 0x9e3779b97f4a7c15ull;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      word = z ^ (z >> 31);
    }
  }

  /// A number in (0, 1].
  double nextUniform() {
    auto rotate = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
    uint64_t result = rotate(state[1] * 5, 7) * 9;
    uint64_t t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotate(state[3], 45);
    return (static_cast<double>(result >> 11) + 1.0) * 0x1.0p-53;
  }

  /// A number from the normal distribution, by the method of Box and
  /// Muller, Ann. Math. Stat. 29, 610 (1958).
  double nextNormal() {
    double radius = std::sqrt(-2.0 * std::log(nextUniform()));
    double angle = 2.0 * M_PI * nextUniform();
    return radius * std::cos(angle);
  }

private:
  uint64_t state[4];
};

} // namespace

double mdir::driver::getKineticEnergy(const System &system) {
  double twice = 0.0;
  for (size_t i = 0, e = system.getNumParticles(); i != e; ++i)
    for (int c = 0; c != 3; ++c) {
      double v = system.velocities[3 * i + c];
      twice += system.masses[i] * v * v;
    }
  return 0.5 * twice;
}

void mdir::driver::assignVelocities(const Control &control, System &system) {
  size_t count = system.getNumParticles();
  Generator generator(control.seed);
  for (size_t i = 0; i != count; ++i) {
    double width =
        std::sqrt(units::boltzmann * control.temperature / system.masses[i]);
    for (int c = 0; c != 3; ++c)
      system.velocities[3 * i + c] = width * generator.nextNormal();
  }
  if (count < 2)
    return;

  // Bring the center of mass to rest.
  double total = 0.0, momentum[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0; i != count; ++i) {
    total += system.masses[i];
    for (int c = 0; c != 3; ++c)
      momentum[c] += system.masses[i] * system.velocities[3 * i + c];
  }
  for (size_t i = 0; i != count; ++i)
    for (int c = 0; c != 3; ++c)
      system.velocities[3 * i + c] -= momentum[c] / total;

  // Scale to the temperature.
  double kinetic = getKineticEnergy(system);
  double target =
      0.5 * system.getDegreesOfFreedom() * units::boltzmann * control.temperature;
  if (kinetic > 0.0) {
    double factor = std::sqrt(target / kinetic);
    for (double &v : system.velocities)
      v *= factor;
  }
}

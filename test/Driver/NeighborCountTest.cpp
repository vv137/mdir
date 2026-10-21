// The most neighbors of a particle (D316): the search by
// cells and threads gives the integer of the serial search over particles
// that the driver used before (kept here, as it was in Builder.cpp, as the
// reference), on inputs made to find a difference and on the systems of
// control files.
//
// Usage: mdir-neighbor-count-test
//          the made inputs;
//        mdir-neighbor-count-test [--time] [--no-reference] CONTROL...
//          the system of each control file, as `mdir run` reads it.
#include "mdir/Driver/Control.h"
#include "mdir/Driver/NeighborCount.h"
#include "mdir/Driver/System.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
using namespace mdir;
using namespace mdir::driver;

/// The search of the driver before D316, as it was.
static int64_t countMostNeighborsReference(const System &system,
                                           double reach) {
  size_t count = system.getNumParticles();
  int cells[3];
  for (int k = 0; k != 3; ++k)
    cells[k] = std::max(1, static_cast<int>(system.box[k] / reach));
  auto wrap = [&](double x, int k) {
    double length = system.box[k];
    return x - length * std::floor(x / length);
  };
  auto cellOf = [&](size_t i, int k) {
    int c = static_cast<int>(wrap(system.positions[3 * i + k], k) /
                             system.box[k] * cells[k]);
    return std::min(c, cells[k] - 1);
  };
  std::vector<std::vector<size_t>> members(cells[0] * cells[1] * cells[2]);
  for (size_t i = 0; i != count; ++i)
    members[(cellOf(i, 0) * cells[1] + cellOf(i, 1)) * cells[2] +
            cellOf(i, 2)]
        .push_back(i);

  double reach2 = reach * reach;
  int64_t most = 0;
  for (size_t i = 0; i != count; ++i) {
    int64_t found = 0;
    int home[3] = {cellOf(i, 0), cellOf(i, 1), cellOf(i, 2)};
    // Each cell once, where the cells on the two sides coincide.
    std::vector<int> seen;
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          int c[3] = {home[0] + dx, home[1] + dy, home[2] + dz};
          for (int k = 0; k != 3; ++k)
            c[k] = (c[k] % cells[k] + cells[k]) % cells[k];
          int cell = (c[0] * cells[1] + c[1]) * cells[2] + c[2];
          if (llvm::is_contained(seen, cell))
            continue;
          seen.push_back(cell);
          for (size_t j : members[cell]) {
            if (j == i)
              continue;
            double r2 = 0.0;
            for (int k = 0; k != 3; ++k) {
              double d = system.positions[3 * i + k] -
                         system.positions[3 * j + k];
              d -= system.box[k] * std::round(d / system.box[k]);
              r2 += d * d;
            }
            found += r2 < reach2;
          }
        }
    most = std::max(most, found);
  }
  return most;
}

static void require(bool ok, const llvm::Twine &what) {
  if (!ok) {
    llvm::errs() << what << "\n";
    std::exit(1);
  }
}

static double seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       since)
      .count();
}

/// The reference against the search with 1, 2, 3, 8 threads and the
/// default; returns the count.
static int64_t compare(const char *name, const System &system, double reach) {
  int64_t expected = countMostNeighborsReference(system, reach);
  for (unsigned threads : {1u, 2u, 3u, 8u, 0u}) {
    int64_t found = countMostNeighbors(
        llvm::ArrayRef(system.positions)
            .take_front(3 * system.getNumParticles()),
        system.box, reach, threads);
    require(found == expected, llvm::Twine(name) + ": " +
                                   llvm::Twine(found) + " with " +
                                   llvm::Twine(threads) +
                                   " threads, the reference " +
                                   llvm::Twine(expected));
  }
  return expected;
}

static System makeSystem(double lx, double ly, double lz) {
  System system;
  system.box[0] = lx;
  system.box[1] = ly;
  system.box[2] = lz;
  return system;
}

static void add(System &system, double x, double y, double z) {
  system.positions.push_back(x);
  system.positions.push_back(y);
  system.positions.push_back(z);
  system.types.push_back(0);
}

static int madeInputs() {
  std::mt19937_64 engine(471);
  // From the 53 high bits of the engine, whose sequence the standard fixes,
  // so that the pinned counts do not depend on the library's distributions.
  auto uniform = [&](double low, double high) {
    double unit = static_cast<double>(engine() >> 11) * 0x1.0p-53;
    return low + (high - low) * unit;
  };
  auto fill = [&](System &system, size_t n, double low, double high) {
    for (size_t i = 0; i != n; ++i)
      add(system, uniform(low, high) * system.box[0],
          uniform(low, high) * system.box[1],
          uniform(low, high) * system.box[2]);
  };
  int cases = 0;
  auto report = [&](const char *name, const System &system, double reach) {
    int64_t most = compare(name, system, reach);
    llvm::outs() << name << ": " << system.getNumParticles()
                 << " particles, most " << most << "\n";
    ++cases;
  };

  // Uniform in the box, 3 cells along each axis.
  {
    System system = makeSystem(30.0, 30.0, 30.0);
    fill(system, 3000, 0.0, 1.0);
    report("uniform, 3 cells", system, 10.0);
  }
  // 1, 2, and 4 cells along the axes, where offsets of the 27 coincide.
  {
    System system = makeSystem(12.0, 25.0, 40.0);
    fill(system, 2000, 0.0, 1.0);
    report("1, 2, 4 cells", system, 10.0);
  }
  {
    System system = makeSystem(19.0, 19.0, 19.0);
    fill(system, 1500, 0.0, 1.0);
    report("1 cell", system, 10.0);
  }
  {
    System system = makeSystem(20.0, 21.0, 39.9);
    fill(system, 1500, 0.0, 1.0);
    report("2 cells", system, 10.0);
  }
  // Positions far outside the box: the cells wrap them, the pairs take the
  // minimum image of the positions as given.
  {
    System system = makeSystem(31.0, 33.0, 35.0);
    fill(system, 3000, -2.0, 3.0);
    report("outside the box", system, 10.0);
  }
  // A lattice whose pairs are at the reach exactly, with particles on the
  // faces of the box and just inside and outside them.
  {
    System system = makeSystem(20.0, 20.0, 20.0);
    for (int i = 0; i <= 8; ++i)
      for (int j = 0; j <= 8; ++j)
        for (int k = 0; k <= 8; ++k)
          add(system, 2.5 * i, 2.5 * j, 2.5 * k);
    for (double edge : {-1e-12, 1e-12, 20.0 - 1e-12, 20.0 + 1e-12, 20.0})
      add(system, edge, 10.0, edge);
    report("lattice at the reach", system, 5.0);
    report("lattice, reach of 1 cell", system, 10.0);
    report("lattice, reach of 2 cells", system, 7.5);
  }
  // Pairs half a box apart along an axis, and an ulp either side, where
  // the minimum image rounds a half away from zero, both ways round; and
  // particles so far out that the quotient of a difference by the box is
  // beyond 2^52.
  {
    System system = makeSystem(16.0, 17.0, 18.0);
    fill(system, 300, 0.0, 1.0);
    for (double base : {0.0, 3.0, -5.0, 15.9})
      for (double half : {8.0, std::nextafter(8.0, 0.0),
                          std::nextafter(8.0, 9.0), -8.0})
        for (int k = 0; k != 3; ++k) {
          double a[3] = {base, base + 0.25, base + 0.5};
          double b[3] = {a[0], a[1], a[2]};
          b[k] += half;
          add(system, a[0], a[1], a[2]);
          add(system, b[0], b[1], b[2]);
        }
    add(system, 1.0e17, 2.0, 3.0);
    add(system, -3.0e17, 2.5, 3.5);
    add(system, 4.0, 5.0e16, 6.0);
    report("half a box", system, 9.0);
  }
  // Particles at the same position.
  {
    System system = makeSystem(30.0, 30.0, 30.0);
    fill(system, 500, 0.0, 1.0);
    for (int copy = 0; copy != 40; ++copy)
      add(system, 15.0, 15.0, 15.0);
    report("copies", system, 10.0);
  }
  // A solute in a corner of a box with room in it, as a solvated molecule
  // that tleap makes, and a slab with vacuum on its two sides.
  {
    System system = makeSystem(60.0, 60.0, 60.0);
    fill(system, 2000, 0.0, 0.2);
    fill(system, 2000, 0.0, 1.0);
    report("cluster and vacuum", system, 10.0);
  }
  {
    System system = makeSystem(30.0, 30.0, 90.0);
    for (int i = 0; i != 4000; ++i)
      add(system, uniform(0.0, 30.0), uniform(0.0, 30.0),
          uniform(30.0, 60.0));
    report("slab", system, 9.0);
  }
  // A tilt is not read: the count takes the diagonal of the cell.
  {
    System system = makeSystem(30.0, 28.0, 26.0);
    system.tilt[0] = 9.0;
    system.tilt[1] = -6.0;
    system.tilt[2] = 4.0;
    fill(system, 3000, 0.0, 1.0);
    report("triclinic", system, 9.0);
  }
  // Enough particles for threads, uniform and clustered.
  {
    System system = makeSystem(80.0, 70.0, 60.0);
    fill(system, 40000, 0.0, 1.0);
    fill(system, 2000, 0.4, 0.5);
    report("threads", system, 10.0);
  }
  {
    System system = makeSystem(50.0, 50.0, 50.0);
    report("no particles", system, 10.0);
    add(system, 1.0, 2.0, 3.0);
    report("one particle", system, 10.0);
  }
  llvm::outs() << cases << " made inputs agree with the reference\n";
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 1)
    return madeInputs();
  bool time = false, reference = true;
  for (int a = 1; a != argc; ++a) {
    llvm::StringRef arg = argv[a];
    if (arg == "--time") {
      time = true;
      continue;
    }
    if (arg == "--no-reference") {
      reference = false;
      continue;
    }
    llvm::Expected<Control> control = readControl(arg);
    require(static_cast<bool>(control),
            arg + ": " + llvm::toString(control.takeError()));
    auto start = std::chrono::steady_clock::now();
    llvm::Expected<System> system = readSystem(*control);
    require(static_cast<bool>(system),
            arg + ": " + llvm::toString(system.takeError()));
    double read = seconds(start);
    // As Builder::build takes it.
    double reach = control->pairlistDistance * units::length;
    llvm::outs() << arg << ": " << system->getNumParticles()
                 << " particles, box " << system->box[0] << " "
                 << system->box[1] << " " << system->box[2] << ", reach "
                 << reach;
    start = std::chrono::steady_clock::now();
    // As Builder::build passes them.
    int64_t most = countMostNeighbors(
        llvm::ArrayRef(system->positions)
            .take_front(3 * system->getNumParticles()),
        system->box, reach);
    double searched = seconds(start);
    llvm::outs() << ", most " << most;
    if (reference) {
      start = std::chrono::steady_clock::now();
      int64_t expected = countMostNeighborsReference(*system, reach);
      double referenced = seconds(start);
      require(most == expected, arg + ": " + llvm::Twine(most) +
                                    ", the reference " +
                                    llvm::Twine(expected));
      llvm::outs() << ", as the reference";
      if (time)
        llvm::outs() << " (reference " << llvm::format("%.3f", referenced)
                     << " s)";
    }
    if (time)
      llvm::outs() << ", read in " << llvm::format("%.3f", read)
                   << " s, counted in " << llvm::format("%.3f", searched)
                   << " s with "
                   << llvm::hardware_concurrency().compute_thread_count()
                   << " cores";
    llvm::outs() << "\n";
  }
  return 0;
}

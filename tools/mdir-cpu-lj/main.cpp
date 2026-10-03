// Fixed-layout reference transport with an MDIR-generated LJ kernel.
#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/TargetSelect.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mpi.h>
#include <set>
#include <sstream>
#include <vector>
using namespace mlir;
struct Atom {
  long long id;
  std::array<double, 3> x;
};
static void fail(const std::string &s) {
  std::cerr << "mdir-cpu-lj: " << s << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}
static int parsePositive(const std::string &s) {
  unsigned value = 0;
  if (s.empty())
    fail("empty numeric option");
  for (char c : s) {
    if (c < '0' || c > '9' || value > 100000)
      fail("invalid numeric option");
    value = value * 10 + (c - '0');
  }
  return value;
}
// Lennard-Jones energy and its radial derivative [AllenTildesley2017].
static std::string kernel(const double *h, bool mixed) {
  std::ostringstream o;
  o << std::scientific << std::setprecision(17);
  std::string t = mixed ? "f32" : "f64", v = "vector<3x" + t + ">";
  o << "module { md.particle_set @atoms\nfunc.func @evaluate(%counts: "
       "memref<?xi32>, %entries: memref<?x?xi32>, %x: memref<?x3xf64>, %f: "
       "memref<?x3xf64>, %tot: memref<10xf64>) attributes "
       "{llvm.emit_c_interface} {\n"
    << "%z = arith.constant 0 : index\n%n = memref.dim %x, %z : "
       "memref<?x3xf64>\n"
    << "%nl = md_exec.neighbor_view %counts, %entries local_size(%n) : "
       "memref<?xi32>, memref<?x?xi32> -> !mdrt.neighbors<@atoms>\n"
    << "%lx = arith.constant " << h[0] << " : f64\n%ly = arith.constant "
    << h[1] << " : f64\n%lz = arith.constant " << h[2]
    << " : f64\n%cell = md.orthorhombic_cell %lx, %ly, %lz\n"
    << "%zero = arith.constant 0.0 : f64\n%vz = arith.constant dense<0.0> : "
       "vector<9xf64>\n"
    << "%e, %vir = md_exec.pair_for %nl, %x, %cell outs(%f : memref<?x3xf64>) "
       "reduce(%zero, %vz : f64, vector<9xf64>) cutoff("
    << h[3]
    << ") weights [0.5, 0.5] overwrite [true] policy(directed, owner_only) {\n"
    << "^bb0(%r2: " << t << ", %d: " << v << "):\n";
  auto c = [&](const char *n, double value) {
    o << '%' << n << " = arith.constant " << value << " : " << t << '\n';
  };
  c("sig2", h[4] * h[4]);
  c("four", 4 * h[5]);
  c("twentyfour", 24 * h[5]);
  c("two", 2);
  o << "%s2 = arith.divf %sig2, %r2 : " << t
    << "\n%s4 = arith.mulf %s2, %s2 : " << t
    << "\n%s6 = arith.mulf %s4, %s2 : " << t
    << "\n%s12 = arith.mulf %s6, %s6 : " << t
    << "\n%u0 = arith.subf %s12, %s6 : " << t
    << "\n%u = arith.mulf %four, %u0 : " << t
    << "\n%a = arith.mulf %two, %s12 : " << t
    << "\n%b = arith.subf %a, %s6 : " << t
    << "\n%c = arith.mulf %twentyfour, %b : " << t
    << "\n%coef = arith.divf %c, %r2 : " << t
    << "\n%cv = vector.broadcast %coef : " << t << " to " << v
    << "\n%force = arith.mulf %cv, %d : " << v << "\n";
  for (int i = 0; i < 3; ++i)
    o << "%d" << i << " = vector.extract %d[" << i << "] : " << t << " from "
      << v << "\n%f" << i << " = vector.extract %force[" << i << "] : " << t
      << " from " << v << '\n';
  for (int i = 0; i < 9; ++i)
    o << "%w" << i << " = arith.mulf %d" << i / 3 << ", %f" << i % 3 << " : "
      << t << '\n';
  o << "%w = vector.from_elements ";
  for (int i = 0; i < 9; ++i)
    o << (i ? ", " : "") << "%w" << i;
  o << " : vector<9x" << t << ">\n";
  if (mixed)
    o << "%fd = arith.extf %force : " << v
      << " to vector<3xf64>\n%ud = arith.extf %u : f32 to f64\n%wd = "
         "arith.extf %w : vector<9xf32> to vector<9xf64>\n";
  o << "md_exec.yield " << (mixed ? "%fd, %ud, %wd" : "%force, %u, %w")
    << " : vector<3xf64>, f64, vector<9xf64>\n} : !mdrt.neighbors<@atoms>, "
       "memref<?x3xf64> -> f64, vector<9xf64>\nmemref.store %e, %tot[%z] : "
       "memref<10xf64>\n";
  for (int i = 0; i < 9; ++i)
    o << "%i" << i << " = arith.constant " << i + 1 << " : index\n%v" << i
      << " = vector.extract %vir[" << i
      << "] : f64 from vector<9xf64>\nmemref.store %v" << i << ", %tot[%i" << i
      << "] : memref<10xf64>\n";
  o << "return\n}}\n";
  return o.str();
}
int main(int argc, char **argv) {
  int provided, rank, ranks;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (provided < MPI_THREAD_FUNNELED)
    fail("MPI_THREAD_FUNNELED unavailable");
  std::string path, precision = "double", emit;
  int threads = 1, width = 4, repeats = 1;
  std::string gridOption = "auto", halo = "sync";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.find("--precision=") == 0)
      precision = a.substr(12);
    else if (a.find("--threads=") == 0)
      threads = parsePositive(a.substr(10));
    else if (a.find("--simd-width=") == 0)
      width = parsePositive(a.substr(13));
    else if (a.find("--repeat=") == 0)
      repeats = parsePositive(a.substr(9));
    else if (a.find("--grid=") == 0)
      gridOption = a.substr(7);
    else if (a.find("--halo=") == 0)
      halo = a.substr(7);
    else if (a.find("--emit=") == 0)
      emit = a.substr(7);
    else if (a[0] == '-' || !path.empty())
      fail("unknown argument: " + a);
    else
      path = a;
  }
  if (path.empty() || (precision != "mixed" && precision != "double") ||
      (halo != "sync" && halo != "async") || repeats < 1 || threads < 1 ||
      (width != 1 && width != 4 && width != 8) ||
      (!emit.empty() && emit != "source" && emit != "loops" && emit != "llvm" &&
       emit != "dist"))
    fail("usage: mdir-cpu-lj SNAPSHOT [--precision=mixed|double] [--threads=N] "
         "[--simd-width=1|4|8] [--repeat=N] [--grid=auto|Px,Py,Pz] "
         "[--halo=sync|async] "
         "[--emit=dist|source|loops|llvm]");
  setenv("OMP_NUM_THREADS", std::to_string(threads).c_str(), 1);
  double h[6];
  long long n = 0;
  std::vector<Atom> all;
  if (rank == 0) {
    std::ifstream in(path);
    if (!(in >> n >> h[0] >> h[1] >> h[2] >> h[3] >> h[4] >> h[5]) || n < 0 ||
        n > std::numeric_limits<int>::max() / 3)
      fail("invalid snapshot header");
    for (double x : h)
      if (!std::isfinite(x))
        fail("nonfinite header");
    if (h[3] <= 0 || h[4] <= 0 || h[5] <= 0 ||
        h[3] >= 0.5 * std::min({h[0], h[1], h[2]}))
      fail("require positive sigma/epsilon and cutoff < half each box edge");
    std::set<long long> ids;
    for (long long i = 0; i < n; ++i) {
      Atom a;
      if (!(in >> a.id >> a.x[0] >> a.x[1] >> a.x[2]) ||
          !ids.insert(a.id).second)
        fail("invalid or duplicate atom ID");
      for (int k = 0; k < 3; ++k)
        if (!std::isfinite(a.x[k]) || a.x[k] < 0 || a.x[k] >= h[k])
          fail("coordinates must be canonical and finite");
      all.push_back(a);
    }
    std::string extra;
    if (in >> extra)
      fail("extra snapshot data");
  }
  MPI_Bcast(h, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  // Conversion of geometry must not turn a finite box into infinity or zero.
  auto representable = [&](double value) {
    if (!std::isfinite(value) || value <= 0)
      return false;
    if (precision == "mixed") {
      float narrow = static_cast<float>(value);
      return std::isfinite(narrow) && narrow > 0;
    }
    return true;
  };
  for (int k = 0; k < 3; ++k)
    if (!representable(h[k]) || !representable(1.0 / h[k]))
      fail("cell is outside the selected precision's numerical range");
  if (!representable(h[3] * h[3]) || !representable(h[4] * h[4]) ||
      !representable(4 * h[5]) || !representable(24 * h[5]))
    fail("LJ constants are outside the selected precision's numerical range");

  // Enumerate ordered factor triples; axes are physical box directions.
  std::array<int, 3> grid{1, 1, 1};
  if (gridOption == "auto") {
    double best = std::numeric_limits<double>::infinity();
    for (int px = 1; px <= ranks; ++px) {
      if (ranks % px)
        continue;
      int rest = ranks / px;
      for (int py = 1; py <= rest; ++py) {
        if (rest % py)
          continue;
        std::array<int, 3> candidate{px, py, rest / py};
        double expanded = 1;
        for (int k = 0; k < 3; ++k) {
          expanded *=
              std::min(1.0, 1.0 / candidate[k] +
                                (candidate[k] > 1 ? 2 * (h[3] / h[k]) : 0));
        }
        // Favor x, then y, when estimates tie.
        if (expanded < best || (expanded == best && candidate > grid)) {
          best = expanded;
          grid = candidate;
        }
      }
    }
  } else {
    std::istringstream in(gridOption);
    std::string part;
    for (int k = 0; k < 3; ++k) {
      if (!std::getline(in, part, ','))
        fail("grid needs three dimensions");
      grid[k] = parsePositive(part);
      if (grid[k] < 1 || grid[k] > ranks)
        fail("invalid grid dimension");
    }
    if (std::getline(in, part, ',') || gridOption.back() == ',' ||
        int64_t(grid[0]) * grid[1] * grid[2] != ranks)
      fail("grid product must equal MPI rank count");
  }
  auto cellOf = [&](int r) {
    return std::array<int, 3>{r / (grid[1] * grid[2]), (r / grid[2]) % grid[1],
                              r % grid[2]};
  };
  auto ownerOf = [&](const Atom &a) {
    std::array<int, 3> c;
    for (int k = 0; k < 3; ++k)
      c[k] = std::min(grid[k] - 1, int(a.x[k] / h[k] * grid[k]));
    return (c[0] * grid[1] + c[1]) * grid[2] + c[2];
  };
  if (rank == 0)
    std::cerr << "grid " << grid[0] << ',' << grid[1] << ',' << grid[2]
              << " halo " << halo << '\n';
  std::vector<int> sizes(ranks), offsets(ranks);
  std::vector<long long> ids;
  std::vector<double> coords;
  if (rank == 0) {
    for (auto &a : all)
      ++sizes[ownerOf(a)];
    for (int r = 1; r < ranks; ++r)
      offsets[r] = offsets[r - 1] + sizes[r - 1];
    auto next = offsets;
    ids.resize(n);
    coords.resize(3 * n);
    for (auto &a : all) {
      int p = next[ownerOf(a)]++;
      ids[p] = a.id;
      std::copy(a.x.begin(), a.x.end(), coords.begin() + 3 * p);
    }
  }
  MPI_Bcast(sizes.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
  for (int r = 1; r < ranks; ++r)
    offsets[r] = offsets[r - 1] + sizes[r - 1];
  int owned = sizes[rank];
  std::vector<long long> localIDs(owned);
  std::vector<double> x(3 * owned);
  MPI_Scatterv(ids.data(), sizes.data(), offsets.data(), MPI_LONG_LONG,
               localIDs.data(), owned, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  auto size3 = sizes, off3 = offsets;
  for (int &v : size3)
    v *= 3;
  for (int &v : off3)
    v *= 3;
  MPI_Scatterv(coords.data(), size3.data(), off3.data(), MPI_DOUBLE, x.data(),
               3 * owned, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // A box-expanded support is conservative for a spherical cutoff. Each
  // destination receives a global ID once, even for periodic dimensions 1/2.
  std::vector<int> sendCounts(ranks), recvCounts(ranks), sendOff(ranks),
      recvOff(ranks), sendIndex;
  for (int r = 0; r < ranks; ++r) {
    sendOff[r] = sendIndex.size();
    if (r == rank)
      continue;
    auto cell = cellOf(r);
    for (int i = 0; i < owned; ++i) {
      bool needed = true;
      for (int k = 0; k < 3; ++k) {
        double lo = h[k] * (double(cell[k]) / grid[k]);
        double hi = h[k] * (double(cell[k] + 1) / grid[k]);
        double distance = h[k];
        for (int image = -1; image <= 1; ++image) {
          double point = x[3 * i + k] + image * h[k];
          distance =
              std::min(distance, std::max({lo - point, point - hi, 0.0}));
        }
        needed &= distance <=
                  h[3] + 32 * std::numeric_limits<float>::epsilon() * h[k];
      }
      if (needed) {
        if (sendIndex.size() >= size_t(std::numeric_limits<int>::max() / 3))
          fail("send capacity exceeds MPI int counts");
        sendIndex.push_back(i);
        ++sendCounts[r];
      }
    }
  }
  MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);
  long long ghosts = 0;
  for (int r = 0; r < ranks; ++r) {
    if (ghosts > std::numeric_limits<int>::max() / 3)
      fail("receive capacity exceeds MPI int counts");
    recvOff[r] = ghosts;
    ghosts += recvCounts[r];
  }
  if (ghosts + owned > std::numeric_limits<int>::max() / 3)
    fail("local capacity exceeds MPI int counts");
  std::vector<long long> sendIDs(sendIndex.size()), ghostIDs(ghosts);
  std::vector<double> packed(3 * sendIndex.size()), ghostX(3 * ghosts);
  for (size_t i = 0; i < sendIndex.size(); ++i) {
    sendIDs[i] = localIDs[sendIndex[i]];
    std::copy_n(x.data() + 3 * sendIndex[i], 3, packed.data() + 3 * i);
  }
  MPI_Alltoallv(sendIDs.data(), sendCounts.data(), sendOff.data(),
                MPI_LONG_LONG, ghostIDs.data(), recvCounts.data(),
                recvOff.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
  for (int &v : sendCounts)
    v *= 3;
  for (int &v : sendOff)
    v *= 3;
  for (int &v : recvCounts)
    v *= 3;
  for (int &v : recvOff)
    v *= 3;
  // Allocate once: descriptors remain stable until both stages finish.
  int local = owned + ghosts, row = 0;
  x.resize(3 * local, std::numeric_limits<double>::quiet_NaN());
  localIDs.insert(localIDs.end(), ghostIDs.begin(), ghostIDs.end());
  std::vector<bool> interior(owned, true);
  auto myCell = cellOf(rank);
  for (int i = 0; i < owned; ++i)
    for (int k = 0; k < 3; ++k) {
      if (grid[k] == 1)
        continue;
      double lo = h[k] * (double(myCell[k]) / grid[k]);
      double hi = h[k] * (double(myCell[k] + 1) / grid[k]);
      double margin = h[3] + 32 * std::numeric_limits<float>::epsilon() * h[k];
      interior[i] = interior[i] && x[3 * i + k] - lo > margin &&
                    hi - x[3 * i + k] > margin;
    }
  std::vector<int32_t> counts(owned), entries(size_t(owned) * row);
  std::vector<MPI_Request> requests;
  bool pending = false, ready = false;
  auto startHalo = [&] {
    if (pending || ready)
      fail("halo started more than once");
    pending = true;
    if (halo == "sync") {
      MPI_Alltoallv(packed.data(), sendCounts.data(), sendOff.data(),
                    MPI_DOUBLE, ghostX.data(), recvCounts.data(),
                    recvOff.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
      for (int r = 0; r < ranks; ++r)
        if (recvCounts[r]) {
          requests.push_back(MPI_REQUEST_NULL);
          MPI_Irecv(ghostX.data() + recvOff[r], recvCounts[r], MPI_DOUBLE, r,
                    17, MPI_COMM_WORLD, &requests.back());
        }
      for (int r = 0; r < ranks; ++r)
        if (sendCounts[r]) {
          requests.push_back(MPI_REQUEST_NULL);
          MPI_Isend(packed.data() + sendOff[r], sendCounts[r], MPI_DOUBLE, r,
                    17, MPI_COMM_WORLD, &requests.back());
        }
    }
  };
  auto waitHalo = [&] {
    if (!pending)
      fail("halo wait without start");
    if (!requests.empty())
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    std::copy(ghostX.begin(), ghostX.end(), x.begin() + 3 * owned);
    pending = false;
    ready = true;
  };
  // Cell lists use a conservative support radius. Sorted local indices
  // preserve the imported-neighbor contract and deterministic row order.
  auto prepareRows = [&](bool interiorStage) {
    if (!interiorStage && !ready)
      fail("boundary requires completed halo");
    int limit = interiorStage ? owned : local;
    std::array<int, 3> bins;
    double support = h[3] + 32 * std::numeric_limits<float>::epsilon() *
                                std::max({h[0], h[1], h[2]});
    for (int k = 0; k < 3; ++k)
      bins[k] = std::max(1, int(std::min(double(std::max(limit, 1)),
                                         std::floor(h[k] / support))));
    auto binOf = [&](int i) {
      std::array<int, 3> c;
      for (int k = 0; k < 3; ++k)
        c[k] = std::min(bins[k] - 1, int(x[3 * i + k] / h[k] * bins[k]));
      return c;
    };
    std::map<std::array<int, 3>, std::vector<int>> cells;
    for (int j = 0; j < limit; ++j)
      cells[binOf(j)].push_back(j);
    std::vector<std::vector<int32_t>> rows(owned);
    row = 0;
    for (int i = 0; i < owned; ++i) {
      counts[i] = 0;
      if (interior[i] != interiorStage)
        continue;
      auto center = binOf(i);
      std::set<std::array<int, 3>> visited;
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            std::array<int, 3> cell{(center[0] + dx + bins[0]) % bins[0],
                                    (center[1] + dy + bins[1]) % bins[1],
                                    (center[2] + dz + bins[2]) % bins[2]};
            if (!visited.insert(cell).second)
              continue;
            auto found = cells.find(cell);
            if (found == cells.end())
              continue;
            for (int j : found->second)
              if (j != i) {
                double r2 = 0;
                for (int k = 0; k < 3; ++k) {
                  double d = x[3 * i + k] - x[3 * j + k];
                  d -= h[k] * std::nearbyint(d / h[k]);
                  r2 += d * d;
                }
                if (r2 == 0)
                  fail("coincident particles");
                rows[i].push_back(j);
              }
          }
      std::sort(rows[i].begin(), rows[i].end());
      counts[i] = rows[i].size();
      row = std::max(row, int(counts[i]));
    }
    entries.resize(size_t(owned) * row);
    for (int i = 0; i < owned; ++i)
      std::copy(rows[i].begin(), rows[i].end(),
                entries.begin() + size_t(i) * row);
  };
  std::string source = kernel(h, precision == "mixed");
  std::ostringstream plan;
  plan << "md_dist.reference_plan [" << grid[0] << ", " << grid[1] << ", " << grid[2]
       << "] {\n"
       << "^bb0(%layout: !mdrt.layout<@atoms>, %map: "
          "!mdrt.transfer_map<@atoms>):\n"
       << "%event = md_dist.halo_start %layout via %map : "
          "!mdrt.layout<@atoms>, !mdrt.transfer_map<@atoms> -> !mdrt.event\n";
  if (halo == "sync")
    plan << "md_dist.halo_wait %event : !mdrt.event\n";
  plan << "md_dist.dispatch @evaluate \"interior\"\n";
  if (halo == "async")
    plan << "md_dist.halo_wait %event : !mdrt.event\n";
  plan << "md_dist.dispatch @evaluate \"boundary\"\n}\n";
  source.insert(source.rfind('}'), plan.str());
  if (emit == "source") {
    if (rank == 0)
      std::cout << source;
    MPI_Finalize();
    return 0;
  }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  registerAllPasses();
  mdir::registerMDIRConversionPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  registry.insert<mdir::md_dist::MDDistDialect, mdir::md::MDDialect,
                  mdir::md_exec::MDExecDialect, mdir::mdrt::MDRTDialect>();
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(source, &context);
  if (!module)
    fail("generated kernel did not parse");
  if (failed(verify(*module)))
    fail("invalid distribution plan");
  if (emit == "dist") {
    if (rank == 0)
      module->print(llvm::outs());
    MPI_Finalize();
    return 0;
  }
  enum class Action { Start, Wait, Interior, Boundary };
  SmallVector<Action> actions;
  for (auto execution : module->getOps<mdir::md_dist::PlanOp>()) {
    for (Operation &op : execution.getBody().front()) {
      if (isa<mdir::md_dist::HaloStartOp>(op))
        actions.push_back(Action::Start);
      else if (isa<mdir::md_dist::HaloWaitOp>(op))
        actions.push_back(Action::Wait);
      else {
        auto dispatch = cast<mdir::md_dist::DispatchOp>(op);
        actions.push_back(dispatch.getSubset() == "interior"
                              ? Action::Interior
                              : Action::Boundary);
      }
    }
    execution.erase();
    break;
  }
  PassManager pm(&context);
  std::string pipeline =
      "convert-md-exec-to-loops{simd-width=" + std::to_string(width) +
      "},fixed-order-reductions";
  if (emit != "loops")
    pipeline +=
        ",convert-scf-to-openmp,hoist-static-allocas,canonicalize,convert-scf-"
        "to-cf,convert-math-to-llvm,convert-math-to-libm,convert-vector-to-"
        "llvm,expand-strided-metadata,finalize-memref-to-llvm,convert-arith-to-"
        "llvm,convert-func-to-llvm,convert-cf-to-llvm,convert-ub-to-llvm,"
        "convert-openmp-to-llvm,reconcile-unrealized-casts";
  if (failed(parsePassPipeline(pipeline, pm)) || failed(pm.run(*module)))
    fail("kernel lowering failed");
  if (!emit.empty()) {
    if (rank == 0)
      module->print(llvm::outs());
    MPI_Finalize();
    return 0;
  }
  ExecutionEngineOptions options;
  std::string omp = std::string(MDIR_LLVM_LIBRARY_DIR) + "/libomp.so";
  llvm::SmallVector<llvm::StringRef> libs{omp};
  options.sharedLibPaths = libs;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  auto engine = ExecutionEngine::create(*module, options);
  if (!engine)
    fail(llvm::toString(engine.takeError()));
  std::vector<double> force(3 * owned);
  double totals[10] = {};
  StridedMemRefType<int32_t, 1> cd{
      counts.data(), counts.data(), 0, {owned}, {1}};
  StridedMemRefType<int32_t, 2> ed{
      entries.data(), entries.data(), 0, {owned, row}, {row, 1}};
  StridedMemRefType<double, 2> xd{x.data(), x.data(), 0, {local, 3}, {3, 1}},
      fd{force.data(), force.data(), 0, {owned, 3}, {3, 1}};
  StridedMemRefType<double, 1> td{totals, totals, 0, {10}, {1}};
  std::vector<double> interiorForce(force.size());
  double interiorTotals[10] = {};
  auto evaluateStage = [&](bool first) {
    prepareRows(first);
    ed = {entries.data(), entries.data(), 0, {owned, row}, {row, 1}};
    if (auto err = (*engine)->invoke("evaluate", &cd, &ed, &xd, &fd, &td))
      fail(llvm::toString(std::move(err)));
    if (first) {
      interiorForce = force;
      std::copy_n(totals, 10, interiorTotals);
    }
  };
  auto compiled = (*engine)->lookupPacked("_mlir_ciface_evaluate");
  if (!compiled)
    fail(llvm::toString(compiled.takeError()));
  // Warm the compiled OpenMP path with empty rows before measuring. No
  // scientific work or ghost reads occur in this invocation.
  std::fill(counts.begin(), counts.end(), 0);
  if (auto err = (*engine)->invoke("evaluate", &cd, &ed, &xd, &fd, &td))
    fail(llvm::toString(std::move(err)));
  MPI_Barrier(MPI_COMM_WORLD); // Exclude rank-local compilation skew.
  double stageTimes[4] = {}, elapsed = 0;
  for (int iteration = 0; iteration < repeats; ++iteration) {
    ready = false;
    requests.clear();
    std::fill(x.begin() + 3 * owned, x.end(),
              std::numeric_limits<double>::quiet_NaN());
    double evaluationStart = MPI_Wtime();
    for (Action action : actions) {
      double begin = MPI_Wtime();
      switch (action) {
      case Action::Start:
        startHalo();
        break;
      case Action::Wait:
        waitHalo();
        break;
      case Action::Interior:
        evaluateStage(true);
        break;
      case Action::Boundary:
        evaluateStage(false);
        break;
      }
      stageTimes[static_cast<int>(action)] += MPI_Wtime() - begin;
    }
    if (pending || !ready)
      fail("evaluation left an incomplete transfer");
    elapsed += MPI_Wtime() - evaluationStart;
  }
  elapsed /= repeats;
  for (double &time : stageTimes)
    time /= repeats;
  double maxTimes[4], maxElapsed;
  MPI_Reduce(stageTimes, maxTimes, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  long long localInterior = std::count(interior.begin(), interior.end(), true),
            globalInterior;
  MPI_Reduce(&localInterior, &globalInterior, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  if (rank == 0)
    std::cerr << "evaluation_seconds " << maxElapsed << " start " << maxTimes[0]
              << " wait " << maxTimes[1] << " interior " << maxTimes[2]
              << " boundary " << maxTimes[3] << " interior_centers "
              << globalInterior << '/' << n << '\n';
  for (size_t i = 0; i < force.size(); ++i)
    force[i] += interiorForce[i];
  for (int i = 0; i < 10; ++i)
    totals[i] += interiorTotals[i];
  for (double value : force)
    if (!std::isfinite(value))
      fail("nonfinite force; snapshot is outside the numerical range");
  for (double value : totals)
    if (!std::isfinite(value))
      fail("nonfinite total; snapshot is outside the numerical range");
  double global[10];
  MPI_Reduce(totals, global, 10, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  std::vector<double> forces(rank == 0 ? 3 * n : 0);
  MPI_Gatherv(force.data(), 3 * owned, MPI_DOUBLE, forces.data(), size3.data(),
              off3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (double value : global)
      if (!std::isfinite(value))
        fail("nonfinite global reduction");
    std::cout << std::setprecision(17) << "energy " << global[0] << "\nvirial";
    for (int i = 1; i < 10; ++i)
      std::cout << ' ' << global[i];
    std::cout << '\n';
    for (int i = 0; i < n; ++i)
      std::cout << "force " << ids[i] << ' ' << forces[3 * i] << ' '
                << forces[3 * i + 1] << ' ' << forces[3 * i + 2] << '\n';
  }
  MPI_Finalize();
  return 0;
}

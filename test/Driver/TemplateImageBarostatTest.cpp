// A native barostat-produced crossing followed by an exact checkpoint
// restart: a fresh list must select the image strategy for the new cell.
#include "TemplatePhysicalFrameCheck.h"
#include "mdir/Driver/Checkpoint.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mdir/Runtime/TemplateRefreshObserver.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <dlfcn.h>

namespace {
struct Capture {
  void *runtime;
  std::string directory;
  int phase = 0;
  llvm::json::Array observations;
  unsigned conditional = 0, nearest = 0;

  template <class F> F symbol(const char *name) const {
    auto pointer = dlsym(runtime, name);
    require(pointer != nullptr, name);
    return reinterpret_cast<F>(pointer);
  }
  using Buffer = StridedMemRefType<int32_t, 1>;
  using Frame = StridedMemRefType<double, 2>;
  using Copy = void (*)(void *, const void *, uint64_t);
  using Register = void (*)(MdrtTemplateRefreshObserver, void *);

  static llvm::json::Array integers(const std::vector<int32_t> &v) {
    llvm::json::Array out;
    for (int32_t x : v) out.push_back(int64_t(x));
    return out;
  }
  static void write(const std::string &name, llvm::json::Object value) {
    std::ofstream file(name);
    require(bool(file), "cannot create native refresh record");
    file << llvm::formatv("{0:2}", llvm::json::Value(std::move(value))).str() << '\n';
    require(bool(file), "cannot write native refresh record");
  }
  std::vector<int32_t> copy(Buffer b, int64_t count = -1) const {
    require(b.offset == 0 && b.strides[0] == 1, "noncontiguous refresh view");
    if (count < 0) count = b.sizes[0];
    require(count <= b.sizes[0] && count >= 0, "refresh view is too small");
    std::vector<int32_t> out(count);
    symbol<Copy>("mdrtDeviceCopyToHost")(out.data(), b.data, count * sizeof(int32_t));
    return out;
  }
  void observe(int64_t handle, int64_t path, int64_t active, const double *input) {
    require(path == 0 || path == 1, "barostat fixture reached general fallback");
    require(input[3] == 0 && input[4] == 0 && input[5] == 0,
            "barostat replay expects an orthorhombic cell");
    auto layoutValue = symbol<int64_t (*)(int64_t, int64_t)>("mdrtTemplatePairsLayoutValue");
    std::array<int64_t, 32> layout;
    llvm::json::Array layoutJSON;
    for (unsigned i = 0; i != layout.size(); ++i) {
      layout[i] = layoutValue(handle, i);
      layoutJSON.push_back(layout[i]);
    }
    require(layout[0] == 3 && layout[1] == 0, "expected three OPC waters, no atoms");
    auto arenaView = symbol<void (*)(Buffer *, int64_t, int64_t)>("_mlir_ciface_mdrtTemplatePairsArenaView");
    auto groupView = symbol<void (*)(Buffer *, int64_t, int64_t)>("_mlir_ciface_mdrtTemplatePairsGroupView");
    auto bufferView = symbol<void (*)(Buffer *, int64_t)>("_mlir_ciface_mdrtTemplatePairsBuffer");
    auto orderView = symbol<void (*)(Buffer *, int64_t)>("_mlir_ciface_mdrtTemplatePairsOrder");
    Buffer arena, order;
    bufferView(&arena, handle);
    orderView(&order, handle);
    // Copy only published data. Padding, tickets and dead workspace are
    // initialized on the CPU, never read as uninitialized device payload.
    std::vector<int32_t> words(arena.sizes[0], 0);
    auto section = [&](int64_t offset, int64_t size) {
      require(offset >= 0 && size >= 0 && offset + size <= int64_t(words.size()),
              "invalid published refresh span");
      symbol<Copy>("mdrtDeviceCopyToHost")(words.data() + offset,
          arena.data + offset, size * sizeof(int32_t));
    };
    section(0, MDRT_TEMPLATE_PAIR_HEADER_WORDS);
    section(layout[5], 4 * layout[2]);
    int64_t entries = 0, work = 0;
    for (unsigned tag = 0; tag != 3; ++tag) {
      entries = std::max(entries, int64_t(words[TEMPLATE_PAIR_OUTER_RANGES + 6 * tag + 3]));
      work = std::max(work, int64_t(words[TEMPLATE_PAIR_OUTER_RANGES + 6 * tag + 5]));
    }
    section(layout[9], 4 * entries);
    section(layout[7], 4 * work);
    llvm::json::Object snapshot;
    snapshot["waters"] = layout[0]; snapshot["atoms"] = layout[1];
    snapshot["columns_x"] = layout[29]; snapshot["columns_y"] = layout[30];
    snapshot["layout"] = std::move(layoutJSON);
    snapshot["arena"] = integers(words); snapshot["order"] = integers(copy(order));
    snapshot["water_active"] = bool(active & 1); snapshot["atom_active"] = bool(active & 2);
    snapshot["path"] = path; snapshot["phase"] = phase;
    llvm::json::Array cell, reaches;
    // Decimal strings of integer bit patterns avoid JSON's signed-integer
    // limit without a floating-point or text-format round trip.
    for (unsigned i = 0; i != 3; ++i) {
      cell.push_back(std::to_string(bits(input[i])));
      reaches.push_back(std::to_string(bits(input[6 + i])));
    }
    snapshot["cell_bits"] = std::move(cell);
    snapshot["reaches_bits"] = std::move(reaches);
    Frame refs;
    symbol<void (*)(Frame *, int64_t, int64_t)>("_mlir_ciface_mdrtTemplatePairsFrameState")(
        &refs, handle, TEMPLATE_FRAME_OUTER_REFERENCE);
    require(refs.offset == 0 && refs.strides[0] == 3 && refs.strides[1] == 1 &&
            refs.sizes[0] >= 3 && refs.sizes[1] == 3, "invalid native reference view");
    std::array<double, 9> references;
    symbol<Copy>("mdrtDeviceCopyToHost")(references.data(), refs.data, sizeof(references));
    llvm::json::Array referenceBits;
    for (unsigned i = 0; i != 3; ++i) {
      llvm::json::Array row;
      for (unsigned j = 0; j != 3; ++j)
        row.push_back(std::to_string(bits(references[3 * i + j])));
      referenceBits.push_back(std::move(row));
    }
    snapshot["outer_reference_bits"] = std::move(referenceBits);
    const char *groups[] = {"groupStarts", "groupLengths", "groupColumns", "atStart"};
    for (unsigned i = 0; i != 4; ++i) {
      Buffer view; groupView(&view, handle, i);
      auto values = i == 3 ? copy(view) : std::vector<int32_t>(view.sizes[0], 0);
      if (i != 3) {
        auto valid = copy(view, layout[27]);
        std::copy(valid.begin(), valid.end(), values.begin());
      }
      snapshot[groups[i]] = integers(values);
    }
    const char *maps[] = {"keys", "zKeys", "items", "places"};
    for (unsigned i = 0; i != 4; ++i) {
      Buffer view; arenaView(&view, handle, 18 + i);
      snapshot[maps[i]] = integers(copy(view));
    }
    std::string filename = "refresh-" + std::to_string(observations.size()) + ".json";
    write(directory + "/" + filename, std::move(snapshot));
    observations.push_back(llvm::json::Object{{"file", filename}, {"phase", phase}, {"path", path},
        {"half_diagonal_nm", .5 * std::min({input[0], input[1], input[2]})},
        {"max_reach_nm", std::max({input[6], input[7], input[8]})}});
    if (path == 0) ++nearest; else ++conditional;
    std::cout << "successful native refresh " << observations.size() << " phase " << phase
              << " path " << path << " entries " << entries << std::endl;
  }
  static void callback(int64_t handle, int64_t path, int64_t active,
                       const double *input, void *data) {
    static_cast<Capture *>(data)->observe(handle, path, active, input);
  }
  void enable(bool on) {
    symbol<Register>("mdrtTemplatePairsSetRefreshObserver")(on ? callback : nullptr,
                                                         on ? this : nullptr);
  }
};
}

int main(int argc, char **argv) {
  require(argc == 4, "expected control, output directory and CUDA runtime");
  llvm::InitializeNativeTarget(); llvm::InitializeNativeTargetAsmPrinter();
  std::string runtimePath = argv[3];
  std::string basePath = runtimePath.substr(0, runtimePath.find_last_of('/')) + "/libmdrt.so";
  require(dlopen(basePath.c_str(), RTLD_NOW | RTLD_GLOBAL) != nullptr,
          "cannot load base runtime for native refresh observation");
  void *runtime = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
  require(runtime != nullptr, "cannot load CUDA runtime for native refresh observation");
  Capture capture{runtime, argv[2]};
  model::PreparedModel p;
  p.control = take(driver::readControl(argv[1]));
  p.control.truncation = driver::Truncation::None;
  p.control.topologyDispersion = driver::DispersionCorrection::None;
  p.control.topologyDispersionGiven = true;
  p.system = take(driver::readSystem(p.control));
  driver::resolveNeighborStructure(p.control, p.system);
  p.execution.target = p.control.target; p.execution.precision = p.control.precision;
  require(p.control.timestep == .001 && p.control.numSteps == 100 && !p.control.pme,
          "expected the predeclared 100-step, 1 fs compression");
  auto physical = [&](const compiler::SimulationState &frame) {
    capture.enable(false);
    auto check = p;
    // The independent no-step references use the existing small-system
    // matrix route, including the deterministic double reference.
    check.control.neighborStructure = driver::NeighborStructure::Matrix;
    check.control.prunedDistance = 0;
    sameFrame(check, frame);
    capture.enable(true);
  };
  std::cout << std::setprecision(17);
  capture.enable(true);
  auto native = take(compiler::Simulation::create(p, false));
  require(take(native->run(5, {}, true)) == 5, "compression ended before step 5");
  auto first = take(native->getState()); physical(first);
  require(take(native->run(95, {}, true)) == 95, "compression ended before step 100");
  auto crossed = take(native->getState()); physical(crossed);
  require(capture.nearest > 0, "initial native refresh did not select nearest");
  std::string checkpoint = std::string(argv[2]) + "/crossed.h5md";
  succeed(native->saveCheckpoint(checkpoint, "template image refresh fixture"));
  auto saved = take(driver::readCheckpoint(checkpoint));
  require(saved.step == 100, "compression checkpoint has the wrong step");
  native.reset();
  // A template list is not checkpoint state. Continuing the exact native
  // checkpoint therefore gives a mandatory cold refresh in the new cell;
  // no validity threshold, seed or physical parameter is manipulated.
  capture.phase = 1;
  unsigned before = capture.conditional;
  auto continued = take(compiler::Simulation::create(p, false));
  take(continued->continueFrom(saved, false, false));
  require(take(continued->run(1, {}, true)) == 1, "restart did not reach step 101");
  auto after = take(continued->getState());
  require(after.step == 101 && capture.conditional > before,
          "fresh native list did not select conditional after the barostat crossing");
  physical(after);
  require(take(continued->run(4, {}, true)) == 4, "restart did not reach step 105");
  auto last = take(continued->getState()); physical(last);
  capture.enable(false);
  llvm::json::Array diagonals;
  for (const auto *frame : {&first, &crossed, &after, &last}) {
    llvm::json::Array diagonal;
    for (double x : frame->box) diagonal.push_back(x);
    diagonals.push_back(llvm::json::Object{{"step", frame->step}, {"diagonal_nm", std::move(diagonal)}});
  }
  Capture::write(std::string(argv[2]) + "/native.json", llvm::json::Object{
      {"observations", std::move(capture.observations)}, {"frames", std::move(diagonals)},
      {"physical_frames", 4}, {"nearest_refreshes", capture.nearest},
      {"conditional_refreshes", capture.conditional}, {"general_refreshes", 0}});
  std::cout << "native checkpoint refresh: nearest then conditional; general fallback 0" << std::endl;
  std::cout << "barostat physical checks: four frames passed unchanged bounds" << std::endl;
}

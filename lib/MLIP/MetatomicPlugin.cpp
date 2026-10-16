//===- MetatomicPlugin.cpp - A metatomic model behind MDIR's C boundary ---===//
//
// The plugin of D270 (docs/metatomic-ml1.md): a TorchScript
// artifact of metatomic [metatomic], evaluated on MDIR's device buffers and
// stream, with forces and virial from torch's autograd. C++20 with RTTI and
// exceptions, linking libtorch; MDIR's core reaches it through
// include/mdir/MLIP/Plugin.h alone, and no exception crosses that boundary.
//
// The model's inputs are those of metatomic's engine interface: a System of
// positions, cell, and atomic numbers, with the pair list that the model
// requests, whose vectors the engine registers with autograd
// (register_autograd_neighbors). Forces and virial are the derivatives of the
// energy in the positions and in a strain of positions and cell, the
// approach of metatomic's own ASE calculator. The plugin builds the pair list
// itself on the device, with every image within the cutoff and no skin
// (Section 6.4 of the design).
//
//===----------------------------------------------------------------------===//

#include "mdir/MLIP/Plugin.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/headeronly/version.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>
#include <metatomic/torch/version.h>

namespace mts = metatensor_torch;
namespace mta = metatomic_torch;

namespace {

thread_local std::string lastError;

#define MDIR_STRINGIFY2(x) #x
#define MDIR_STRINGIFY(x) MDIR_STRINGIFY2(x)
const char *const torchVersion = MDIR_STRINGIFY(TORCH_VERSION_MAJOR) "." MDIR_STRINGIFY(
    TORCH_VERSION_MINOR) "." MDIR_STRINGIFY(TORCH_VERSION_PATCH);
const std::string buildVersions = std::string("torch ") + torchVersion +
                                  ", metatomic-torch " +
                                  METATOMIC_TORCH_VERSION;

template <class F> int guarded(F &&f) {
  try {
    f();
    return MDIR_MLIP_OK;
  } catch (const std::exception &e) {
    lastError = e.what();
  } catch (...) {
    lastError = "an unknown exception";
  }
  return MDIR_MLIP_ERROR;
}

/// The memory that one block of the brute-force pair search may take.
constexpr int64_t searchBytes = int64_t(256) << 20;

} // namespace

struct mdir_mlip_model {
  torch::jit::Module module;
  torch::Device device{torch::kCPU};
  torch::Dtype dtype = torch::kFloat64;
  mta::NeighborListOptions request;
  double cutoff = 0.0;
  bool local = false;
  int64_t centersPerCall = 0;
  double interactionRange = -1.0;
  std::vector<int64_t> types;
  std::string lengthUnit, energyUnit, name;
  double lengthFactor = 1.0, energyFactor = 1.0;
  torch::Dict<std::string, mta::ModelOutput> outputs;
};

namespace {

/// The pairs within `cutoff` of `positions` (n x 3, f64, nm) in the cell
/// `cell` (3 x 3, rows the edges, f64 on the device; `host` the same on the
/// host), with every image: the first and the second particle and the shift
/// s of the second in cells, r = x_j - x_i + s h. The list is full and
/// strict, without the pair of a particle with itself.
void findPairs(const torch::Tensor &positions, const torch::Tensor &cell,
               const double host[9], double cutoff, torch::Tensor &pairs,
               torch::Tensor &shifts) {
  auto dev = positions.options();
  int64_t n = positions.size(0);
  // The particles wrapped into the cell, k the cells they were moved by:
  // x = x' + k h.
  torch::Tensor fraction = positions.matmul(torch::linalg_inv(cell));
  torch::Tensor k = torch::floor(fraction);
  torch::Tensor wrapped = positions - k.matmul(cell);

  // The images within the cutoff: along each edge, as many cells as the
  // cutoff spans of the distance between the planes of the other two.
  double a[3] = {host[0], host[1], host[2]}, b[3] = {host[3], host[4], host[5]},
         c[3] = {host[6], host[7], host[8]};
  auto cross = [](const double *u, const double *v, double *w) {
    w[0] = u[1] * v[2] - u[2] * v[1];
    w[1] = u[2] * v[0] - u[0] * v[2];
    w[2] = u[0] * v[1] - u[1] * v[0];
  };
  double bc[3], ca[3], ab[3];
  cross(b, c, bc);
  cross(c, a, ca);
  cross(a, b, ab);
  double volume = std::fabs(a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]);
  auto norm = [](const double *u) {
    return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  };
  int64_t reach[3] = {
      static_cast<int64_t>(std::ceil(cutoff * norm(bc) / volume)),
      static_cast<int64_t>(std::ceil(cutoff * norm(ca) / volume)),
      static_cast<int64_t>(std::ceil(cutoff * norm(ab) / volume))};
  std::vector<int32_t> list;
  int64_t zero = -1;
  for (int64_t i = -reach[0]; i <= reach[0]; ++i)
    for (int64_t j = -reach[1]; j <= reach[1]; ++j)
      for (int64_t l = -reach[2]; l <= reach[2]; ++l) {
        if (i == 0 && j == 0 && l == 0)
          zero = static_cast<int64_t>(list.size() / 3);
        list.insert(list.end(), {int32_t(i), int32_t(j), int32_t(l)});
      }
  int64_t count = static_cast<int64_t>(list.size() / 3);
  torch::Tensor images =
      torch::from_blob(list.data(), {count, 3}, torch::kInt32)
          .to(positions.device());
  torch::Tensor offsets = images.to(torch::kFloat64).matmul(cell);

  // Blocks of first particles, each against all second particles and all
  // images, within the memory of `searchBytes`.
  int64_t block =
      std::max<int64_t>(1, searchBytes / (48 * count * std::max<int64_t>(n, 1)));
  double cutoff2 = cutoff * cutoff;
  std::vector<torch::Tensor> firsts, seconds, cells;
  for (int64_t begin = 0; begin < n; begin += block) {
    int64_t end = std::min(n, begin + block);
    int64_t size = end - begin;
    torch::Tensor d = wrapped.unsqueeze(0).unsqueeze(0) -
                      wrapped.slice(0, begin, end).unsqueeze(1).unsqueeze(1) +
                      offsets.unsqueeze(0).unsqueeze(2);
    torch::Tensor within = (d * d).sum(-1) < cutoff2; // size x count x n
    torch::Tensor rows = torch::arange(size, dev.dtype(torch::kLong));
    within.index_put_({rows, torch::full_like(rows, zero), rows + begin},
                      torch::zeros({size}, dev.dtype(torch::kBool)));
    torch::Tensor found = within.nonzero(); // waits for the device
    torch::Tensor i = found.select(1, 0) + begin;
    torch::Tensor j = found.select(1, 2);
    torch::Tensor s = images.index_select(0, found.select(1, 1));
    // The shift between the particles as given: x_j - x_i + s' h of the
    // wrapped ones is x_j - x_i + (s' - k_j + k_i) h.
    torch::Tensor shift = s.to(torch::kLong) - k.index_select(0, j).to(torch::kLong) +
                          k.index_select(0, i).to(torch::kLong);
    firsts.push_back(i);
    seconds.push_back(j);
    cells.push_back(shift);
  }
  if (firsts.empty()) {
    pairs = torch::zeros({0, 2}, dev.dtype(torch::kInt32));
    shifts = torch::zeros({0, 3}, dev.dtype(torch::kInt32));
    return;
  }
  pairs = torch::stack({torch::cat(firsts), torch::cat(seconds)}, 1)
              .to(torch::kInt32)
              .contiguous();
  shifts = torch::cat(cells).to(torch::kInt32).contiguous();
}

} // namespace

extern "C" {

int32_t mdir_mlip_abi_version(void) { return MDIR_MLIP_ABI_VERSION; }

const char *mdir_mlip_build_versions(void) { return buildVersions.c_str(); }

const char *mdir_mlip_torch_version(void) { return torchVersion; }

const char *mdir_mlip_last_error(void) { return lastError.c_str(); }

int mdir_mlip_load(const mdir_mlip_load_options *options,
                   mdir_mlip_model **result) {
  return guarded([&] {
    *result = nullptr;
    if (std::string(options->backend) != "METATOMIC")
      throw std::runtime_error("this plugin loads the backend \"METATOMIC\", "
                               "not \"" + std::string(options->backend) + "\"");
    auto m = std::make_unique<mdir_mlip_model>();
    m->module = mta::load_atomistic_model(options->path);
    m->device = torch::Device(torch::kCUDA, options->device);
    m->module.to(m->device);
    m->module.eval();

    auto capabilities = m->module.run_method("capabilities")
                            .toCustomClass<mta::ModelCapabilitiesHolder>();
    std::string dtype = capabilities->dtype();
    if (dtype == "float32")
      m->dtype = torch::kFloat32;
    else if (dtype == "float64")
      m->dtype = torch::kFloat64;
    else
      throw std::runtime_error("the model takes inputs of type '" + dtype +
                               "'; float32 or float64 are taken");
    auto outputs = capabilities->outputs();
    if (!outputs.contains("energy"))
      throw std::runtime_error("the model has no 'energy' output");
    m->types = capabilities->atomic_types;
    m->lengthUnit = capabilities->length_unit();
    m->energyUnit = outputs.at("energy")->unit();
    m->lengthFactor = mta::unit_conversion_factor(m->lengthUnit, "nm");
    m->energyFactor =
        mta::unit_conversion_factor(m->energyUnit, "kJ/mol");
    m->interactionRange = capabilities->engine_interaction_range("nm");

    auto requests = m->module.run_method("requested_neighbor_lists").toList();
    if (requests.size() != 1)
      throw std::runtime_error(
          "the model asks for " + std::to_string(requests.size()) +
          " pair lists; the plugin gives one");
    m->request =
        requests.get(0).toCustomClass<mta::NeighborListOptionsHolder>();
    m->cutoff = m->request->engine_cutoff("nm");
    if (!(m->cutoff > 0.0) || !std::isfinite(m->cutoff))
      throw std::runtime_error("the model asks for a pair list without a "
                               "finite cutoff");
    if (!m->request->full_list())
      throw std::runtime_error("the model asks for a half pair list; the "
                               "plugin gives a full one");

    // A model is evaluated in chunks of centers only if it declares that
    // the energy of a selection takes the pairs of the selected centers
    // alone (docs/metatomic-ml1.md, Sections 4.2 and 5.3).
    auto metadata = m->module.run_method("metadata")
                        .toCustomClass<mta::ModelMetadataHolder>();
    m->name = metadata->name;
    m->local = metadata->extra.contains("mdir::local_centers") &&
               metadata->extra.at("mdir::local_centers") == "true" &&
               std::isfinite(m->interactionRange) &&
               m->interactionRange >= 0.0;
    m->centersPerCall = m->local ? options->centers_per_call : 0;

    m->outputs.insert("energy", torch::make_intrusive<mta::ModelOutputHolder>(
                                    "energy", "kJ/mol", "system",
                                    std::vector<std::string>{}, ""));
    *result = m.release();
  });
}

int mdir_mlip_describe(const mdir_mlip_model *m,
                       mdir_mlip_description *d) {
  return guarded([&] {
    d->cutoff = m->cutoff;
    d->full_list = m->request->full_list();
    d->strict = m->request->strict();
    d->dtype_bits = m->dtype == torch::kFloat32 ? 32 : 64;
    d->local_centers = m->local;
    d->centers_per_call = m->centersPerCall;
    d->interaction_range =
        std::isfinite(m->interactionRange) ? m->interactionRange : -1.0;
    d->num_types = static_cast<int64_t>(m->types.size());
    d->types = m->types.data();
    d->length_unit = m->lengthUnit.c_str();
    d->energy_unit = m->energyUnit.c_str();
    d->length_factor = m->lengthFactor;
    d->energy_factor = m->energyFactor;
    d->name = m->name.c_str();
    d->framework = buildVersions.c_str();
  });
}

int mdir_mlip_evaluate(mdir_mlip_model *m, const mdir_mlip_inputs *in,
                       mdir_mlip_outputs *out) {
  return guarded([&] {
    auto stream = c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(in->stream), m->device.index());
    c10::cuda::CUDAStreamGuard guard(stream);
    auto dev = torch::TensorOptions().device(m->device);
    int64_t n = in->count;

    // MDIR's buffers, borrowed: from_blob takes no copy and frees nothing.
    torch::Tensor positions =
        torch::from_blob(const_cast<void *>(in->positions), {n, 3},
                         dev.dtype(in->position_bits == 64 ? torch::kFloat64
                                                           : torch::kFloat32))
            .to(torch::kFloat64);
    torch::Tensor numbers = torch::from_blob(
        const_cast<int32_t *>(in->numbers), {n}, dev.dtype(torch::kInt32));
    torch::Tensor forces = torch::from_blob(
        out->forces, {n, 3},
        dev.dtype(out->force_bits == 64 ? torch::kFloat64 : torch::kFloat32));
    torch::Tensor energy =
        torch::from_blob(out->energy, {1}, dev.dtype(torch::kFloat64));
    torch::Tensor virial =
        torch::from_blob(out->virial, {9}, dev.dtype(torch::kFloat64));
    torch::Tensor cell =
        torch::from_blob(const_cast<double *>(in->cell), {3, 3},
                         torch::TensorOptions().dtype(torch::kFloat64))
            .to(m->device);
    torch::Tensor periodic =
        torch::tensor({bool(in->periodic[0]), bool(in->periodic[1]),
                       bool(in->periodic[2])},
                      torch::kBool)
            .to(m->device);

    torch::Tensor pairs, shifts;
    {
      torch::NoGradGuard noGrad;
      findPairs(positions, cell, in->cell, m->cutoff, pairs, shifts);
    }
    int64_t p = pairs.size(0);

    // The positions and a strain, both of 64 bits, which the derivatives
    // are taken in; the model takes them rounded to its type.
    torch::Tensor x = positions.detach().requires_grad_(true);
    torch::Tensor strain =
        torch::eye(3, dev.dtype(torch::kFloat64)).requires_grad_(true);
    auto samples = torch::make_intrusive<mts::LabelsHolder>(
        std::vector<std::string>{"first_atom", "second_atom", "cell_shift_a",
                                 "cell_shift_b", "cell_shift_c"},
        torch::cat({pairs, shifts}, 1).contiguous(),
        metatensor::assume_unique{});
    auto xyz = torch::make_intrusive<mts::LabelsHolder>(
        "xyz", torch::arange(3, dev.dtype(torch::kInt32)).reshape({3, 1}));
    auto distance = torch::make_intrusive<mts::LabelsHolder>(
        "distance", torch::zeros({1, 1}, dev.dtype(torch::kInt32)));
    torch::Tensor first = pairs.select(1, 0).to(torch::kLong);
    torch::Tensor second = pairs.select(1, 1).to(torch::kLong);
    torch::Tensor shiftValues = shifts.to(torch::kFloat64);

    // One call for the whole system, or one for each chunk of centers. A
    // chunk takes the whole list; a local model drops the pairs of the
    // other centers, and the backward of each chunk frees its graph.
    int64_t chunk = m->centersPerCall > 0 && m->centersPerCall < n
                        ? m->centersPerCall
                        : n;
    torch::Tensor total = torch::zeros({1}, dev.dtype(torch::kFloat64));
    int64_t calls = 0;
    for (int64_t begin = 0; begin < n || (n == 0 && calls == 0);
         begin += chunk) {
      int64_t end = std::min(n, begin + chunk);
      torch::Tensor xs = x.matmul(strain);
      torch::Tensor cs = cell.matmul(strain);
      auto system = torch::make_intrusive<mta::SystemHolder>(
          numbers, xs.to(m->dtype), cs.to(m->dtype), periodic);
      torch::Tensor vectors;
      {
        // The vectors are computed in 64 bits and rounded to the model's
        // type; register_autograd_neighbors connects them to the positions
        // and the cell of the system.
        torch::NoGradGuard noGrad;
        vectors = (x.index_select(0, second) - x.index_select(0, first) +
                   shiftValues.matmul(cell))
                      .to(m->dtype);
      }
      auto block = torch::make_intrusive<mts::TensorBlockHolder>(
          vectors.reshape({p, 3, 1}), samples,
          std::vector<mts::Labels>{xyz}, distance);
      mta::register_autograd_neighbors(system, block, false);
      system->add_neighbor_list(m->request, block);

      torch::optional<mts::Labels> selected = torch::nullopt;
      if (chunk < n) {
        torch::Tensor atoms = torch::arange(begin, end, dev.dtype(torch::kInt32));
        selected = torch::make_intrusive<mts::LabelsHolder>(
            std::vector<std::string>{"system", "atom"},
            torch::stack({torch::zeros_like(atoms), atoms}, 1).contiguous());
      }
      auto options = torch::make_intrusive<mta::ModelEvaluationOptionsHolder>(
          "nm", m->outputs, selected);
      auto systems = c10::List<mta::System>({system});
      auto result =
          m->module.forward({systems, options, false}).toGenericDict();
      auto map = result.at("energy").toCustomClass<mts::TensorMapHolder>();
      torch::Tensor e = mts::TensorMapHolder::block_by_id(map, 0)->values().sum();
      e.backward();
      total += e.detach().to(torch::kFloat64);
      ++calls;
      if (n == 0)
        break;
    }

    // Added to MDIR's forces, in the buffer of its other terms and in its
    // type, and written to the energy and the virial, on its stream:
    // nothing waits.
    if (n > 0 && x.grad().defined())
      forces.sub_(x.grad());
    energy.copy_(total);
    if (strain.grad().defined())
      virial.copy_(-strain.grad().reshape({9}));
    else
      virial.zero_();
    out->pairs = p;
    out->calls = calls;
  });
}

int mdir_mlip_memory_stats(mdir_mlip_model *m, int32_t reset,
                           mdir_mlip_memory *memory) {
  return guarded([&] {
    auto index = m->device.index();
    c10::CachingDeviceAllocator::DeviceStats stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(index);
    auto aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    memory->allocated = stats.allocated_bytes[aggregate].current;
    memory->peak_allocated = stats.allocated_bytes[aggregate].peak;
    memory->reserved = stats.reserved_bytes[aggregate].current;
    memory->peak_reserved = stats.reserved_bytes[aggregate].peak;
    if (reset)
      c10::cuda::CUDACachingAllocator::resetPeakStats(index);
  });
}

void mdir_mlip_free(mdir_mlip_model *model) { delete model; }

} // extern "C"

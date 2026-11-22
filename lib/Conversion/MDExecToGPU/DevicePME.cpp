#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

// Bricks use particle weights. Tiles additionally need one binning pass,
// shared by both windows. Emit the bin functions with the spline stages.
template <Window window> void DevicePME::emitTiles(int64_t tiles, bool listed) {
  auto emit = [&](auto edgeTag) {
    constexpr int edge = decltype(edgeTag)::value;
    std::string suffix = std::to_string(edge);
    if constexpr (window == Window::Spline) {
      std::string bin = "mdrt_gpu_pme_bin" + suffix;
      emitBin<false, edge>(bin);
      if (listed)
        emitBin<true, edge>(bin + "_listed");
      emitSpreadTile<window, edge>("mdrt_gpu_pme_spread_tiles" + suffix);
    } else {
      emitSpreadTile<window, edge>("mdrt_gpu_pme_prolate_spread_tiles" +
                                   suffix);
    }
  };
  switch (tiles) {
  case 4:
    emit(std::integral_constant<int, 4>{});
    break;
  case 8:
    emit(std::integral_constant<int, 8>{});
    break;
  case 16:
    emit(std::integral_constant<int, 16>{});
    break;
  }
}

template <bool tilted>
void DevicePME::emitSplineStages(bool potential, bool listed, int64_t tiles, bool ordered) {
  auto name = [](StringRef stage) {
    return ("mdrt_gpu_pme_" + stage + (tilted ? "_triclinic" : "")).str();
  };
  emitBlocks(name("blocks"));
  emitSpread<tilted, false, true>(name("spread"));
  emitSpread<tilted, false, false>(name("spread_float"));
  emitWeights<tilted>(name("weights"));
  if constexpr (!tilted) {
    emitSpreadBricks<false>(name("spread_bricks"));
    emitReal(name("real"));
  }
  emitTables<Window::Spline, Splitting::Gaussian, tilted>(name("tables"));
  emitConvolve<Splitting::Gaussian, tilted>(name("convolve"));
  emitScale<Window::Spline, Splitting::Gaussian, tilted>(name("scale"));
  emitTables<Window::Spline, Splitting::Dispersion, tilted>(
      name("tables_dispersion"));
  emitConvolve<Splitting::Dispersion, tilted>(name("convolve_dispersion"));
  emitScale<Window::Spline, Splitting::Dispersion, tilted>(
      name("scale_dispersion"));
  emitGather<tilted, false, false, false>(name("gather"));
  emitGather<tilted, false, false, true>(name("gather_weights"));
  if (ordered) {
    emitTables<Window::Spline, Splitting::Gaussian, tilted>(
        name("tables_ordered"), true);
    emitScale<Window::Spline, Splitting::Gaussian, tilted>(name("influence"), true);
    emitCellOrder<tilted>(name("cell_order"));
    emitGather<tilted, false, false, true>(name("gather_ordered"), true);
    emitGather<tilted, false, false, true>(name("gather_ordered_two"), true, 2);
    if constexpr (!tilted) {
      emitPack(name("pack"));
      emitIntegerPatch(name("spread_ordered"));
      emitPatchReal(name("patch_real"));
    }
  }
  if (potential)
    emitGather<tilted, false, true, false>(name("potential"));
  if (listed) {
    emitSpread<tilted, true, true>(name("spread_listed"));
    emitSpread<tilted, true, false>(name("spread_float_listed"));
    if constexpr (!tilted)
      emitSpreadBricks<true>(name("spread_bricks_listed"));
    emitGather<tilted, true, false, false>(name("gather_listed"));
    emitGather<tilted, true, false, true>(name("gather_weights_listed"));
  }
  if constexpr (!tilted) {
    emitTiles<Window::Spline>(tiles, listed);
    emitTables<Window::Spline, Splitting::P3M, false>(name("tables_p3m"));
    emitConvolve<Splitting::P3M, false>(name("convolve_p3m"));
  }
}

template void DevicePME::emitSplineStages<false>(bool, bool, int64_t, bool);
template void DevicePME::emitSplineStages<true>(bool, bool, int64_t, bool);

void DevicePME::emitProlate(bool listed, int64_t tiles) {
  emitBlocks("mdrt_gpu_pme_prolate_blocks");
  emitCoefficients("mdrt_gpu_pme_prolate_coefficients");
  emitPlaces<false>("mdrt_gpu_pme_prolate_places");
  emitPlaces<true>("mdrt_gpu_pme_prolate_places_triclinic");
  emitWarpSpread<false, false>("mdrt_gpu_pme_prolate_spread_bricks");
  emitWarpSpread<false, true>("mdrt_gpu_pme_prolate_spread_fixed");
  emitWarpGather<false, false, false>("mdrt_gpu_pme_prolate_gather");
  emitWarpGather<false, false, true>("mdrt_gpu_pme_prolate_potential");
  emitWarpGather<true, false, false>("mdrt_gpu_pme_prolate_gather_triclinic");
  emitWarpGather<true, false, true>("mdrt_gpu_pme_prolate_potential_triclinic");
  emitTables<Window::Prolate, Splitting::Radial, false>(
      "mdrt_gpu_pme_prolate_tables");
  emitTables<Window::Prolate, Splitting::Radial, true>(
      "mdrt_gpu_pme_prolate_tables_triclinic");
  emitConvolve<Splitting::Radial, false>("mdrt_gpu_pme_prolate_convolve");
  emitScale<Window::Prolate, Splitting::Radial, false>(
      "mdrt_gpu_pme_prolate_scale");
  emitConvolve<Splitting::Radial, true>(
      "mdrt_gpu_pme_prolate_convolve_triclinic");
  emitScale<Window::Prolate, Splitting::Radial, true>(
      "mdrt_gpu_pme_prolate_scale_triclinic");
  if (listed) {
    emitWarpSpread<true, false>("mdrt_gpu_pme_prolate_spread_bricks_listed");
    emitWarpSpread<true, true>("mdrt_gpu_pme_prolate_spread_fixed_listed");
    emitWarpGather<false, true, false>("mdrt_gpu_pme_prolate_gather_listed");
    emitWarpGather<true, true, false>(
        "mdrt_gpu_pme_prolate_gather_listed_triclinic");
  }
  emitTiles<Window::Prolate>(tiles, listed);
}

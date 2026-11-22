#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

void DevicePME::emitOrthorhombic(bool potential, bool listed, int64_t tiles) {
  emitBlocks("mdrt_gpu_pme_blocks");
  emitSpread<false, false, true>("mdrt_gpu_pme_spread");
  emitSpread<false, false, false>("mdrt_gpu_pme_spread_float");
  emitWeights<false>("mdrt_gpu_pme_weights");
  emitSpreadBricks<false>("mdrt_gpu_pme_spread_bricks");
  emitReal("mdrt_gpu_pme_real");
  emitTables<Splitting::Gaussian, false>("mdrt_gpu_pme_tables");
  emitConvolve<Splitting::Gaussian, false>("mdrt_gpu_pme_convolve");
  emitScale<Splitting::Gaussian, false>("mdrt_gpu_pme_scale");
  emitTables<Splitting::Dispersion, false>("mdrt_gpu_pme_tables_dispersion");
  emitConvolve<Splitting::Dispersion, false>(
      "mdrt_gpu_pme_convolve_dispersion");
  emitScale<Splitting::Dispersion, false>("mdrt_gpu_pme_scale_dispersion");
  emitGather<false, false, false, false>("mdrt_gpu_pme_gather");
  emitGather<false, false, false, true>("mdrt_gpu_pme_gather_weights");
  if (potential)
    emitGather<false, false, true, false>("mdrt_gpu_pme_potential");
  if (listed)
    emitSpread<false, true, true>("mdrt_gpu_pme_spread_listed");
  if (listed)
    emitSpread<false, true, false>("mdrt_gpu_pme_spread_float_listed");
  if (listed)
    emitSpreadBricks<true>("mdrt_gpu_pme_spread_bricks_listed");
  if (listed)
    emitGather<false, true, false, false>("mdrt_gpu_pme_gather_listed");
  if (listed)
    emitGather<false, true, false, true>("mdrt_gpu_pme_gather_weights_listed");
  if (tiles == 4)
    emitBin<false, 4>("mdrt_gpu_pme_bin4");
  if (listed && tiles == 4)
    emitBin<true, 4>("mdrt_gpu_pme_bin4_listed");
  if (tiles == 4)
    emitSpreadTile<Window::Spline, 4>("mdrt_gpu_pme_spread_tiles4");
  if (tiles == 8)
    emitBin<false, 8>("mdrt_gpu_pme_bin8");
  if (listed && tiles == 8)
    emitBin<true, 8>("mdrt_gpu_pme_bin8_listed");
  if (tiles == 8)
    emitSpreadTile<Window::Spline, 8>("mdrt_gpu_pme_spread_tiles8");
  if (tiles == 16)
    emitBin<false, 16>("mdrt_gpu_pme_bin16");
  if (listed && tiles == 16)
    emitBin<true, 16>("mdrt_gpu_pme_bin16_listed");
  if (tiles == 16)
    emitSpreadTile<Window::Spline, 16>("mdrt_gpu_pme_spread_tiles16");
  emitTables<Splitting::P3M, false>("mdrt_gpu_pme_tables_p3m");
  emitConvolve<Splitting::P3M, false>("mdrt_gpu_pme_convolve_p3m");
}

void DevicePME::emitTriclinic(bool potential, bool listed, int64_t tiles) {
  emitBlocks("mdrt_gpu_pme_blocks_triclinic");
  emitSpread<true, false, true>("mdrt_gpu_pme_spread_triclinic");
  emitSpread<true, false, false>("mdrt_gpu_pme_spread_float_triclinic");
  emitWeights<true>("mdrt_gpu_pme_weights_triclinic");
  emitTables<Splitting::Gaussian, true>("mdrt_gpu_pme_tables_triclinic");
  emitConvolve<Splitting::Gaussian, true>("mdrt_gpu_pme_convolve_triclinic");
  emitScale<Splitting::Gaussian, true>("mdrt_gpu_pme_scale_triclinic");
  emitTables<Splitting::Dispersion, true>(
      "mdrt_gpu_pme_tables_dispersion_triclinic");
  emitConvolve<Splitting::Dispersion, true>(
      "mdrt_gpu_pme_convolve_dispersion_triclinic");
  emitScale<Splitting::Dispersion, true>(
      "mdrt_gpu_pme_scale_dispersion_triclinic");
  emitGather<true, false, false, false>("mdrt_gpu_pme_gather_triclinic");
  emitGather<true, false, false, true>("mdrt_gpu_pme_gather_weights_triclinic");
  if (potential)
    emitGather<true, false, true, false>("mdrt_gpu_pme_potential_triclinic");
  if (listed)
    emitSpread<true, true, true>("mdrt_gpu_pme_spread_listed_triclinic");
  if (listed)
    emitSpread<true, true, false>("mdrt_gpu_pme_spread_float_listed_triclinic");
  if (listed)
    emitGather<true, true, false, false>(
        "mdrt_gpu_pme_gather_listed_triclinic");
  if (listed)
    emitGather<true, true, false, true>(
        "mdrt_gpu_pme_gather_weights_listed_triclinic");
}

void DevicePME::emitProlate(bool potential, bool listed, int64_t tiles) {
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
  emitTables<Splitting::Radial, false>("mdrt_gpu_pme_prolate_tables");
  emitTables<Splitting::Radial, true>("mdrt_gpu_pme_prolate_tables_triclinic");
  emitConvolve<Splitting::Radial, false>("mdrt_gpu_pme_prolate_convolve");
  emitScale<Splitting::Radial, false>("mdrt_gpu_pme_prolate_scale");
  emitConvolve<Splitting::Radial, true>(
      "mdrt_gpu_pme_prolate_convolve_triclinic");
  emitScale<Splitting::Radial, true>("mdrt_gpu_pme_prolate_scale_triclinic");
  if (listed)
    emitWarpSpread<true, false>("mdrt_gpu_pme_prolate_spread_bricks_listed");
  if (listed)
    emitWarpSpread<true, true>("mdrt_gpu_pme_prolate_spread_fixed_listed");
  if (listed)
    emitWarpGather<false, true, false>("mdrt_gpu_pme_prolate_gather_listed");
  if (listed)
    emitWarpGather<true, true, false>(
        "mdrt_gpu_pme_prolate_gather_listed_triclinic");
  if (tiles == 4)
    emitSpreadTile<Window::Prolate, 4>("mdrt_gpu_pme_prolate_spread_tiles4");
  if (tiles == 8)
    emitSpreadTile<Window::Prolate, 8>("mdrt_gpu_pme_prolate_spread_tiles8");
  if (tiles == 16)
    emitSpreadTile<Window::Prolate, 16>("mdrt_gpu_pme_prolate_spread_tiles16");
}

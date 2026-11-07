// Immutable parameter metadata; values of theta never pass through this plan.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHPARAMETERPLAN_H
#define MDIR_LIB_DRIVER_FRAMEBATCHPARAMETERPLAN_H
#include "mdir/Driver/Model.h"
#include "mdir/Driver/Builder.h"
#include "FrameBatchLJIR.h"
#include "FrameBatchPMEIR.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>
#include <array>
#include <optional>
namespace mdir::driver::builder {
struct FrameBatchMappedResource {
  enum Kind { Field, Table, TupleField } kind;
  unsigned resource = 0, field = 0;
  std::vector<int64_t> map;
  enum Update { Direct, MixSigma, MixEpsilon, SeedSigma, SeedEpsilon } update = Direct;
  unsigned seed = 0;
};
struct FrameBatchParameterPlan {
  // Offsets of the named arrays in the contiguous f64 theta buffer.
  std::vector<int64_t> offsets;
  int64_t parameters = 0;
  std::vector<FrameBatchMappedResource> mapped;
  struct Scalar { unsigned start; int64_t parameter; };
  std::vector<Scalar> scalars;
  unsigned types = 0;
  bool mixesSigma = false, mixesEpsilon = false, geometricSigma = false;
  std::vector<int64_t> sigmaMap, epsilonMap, sigmaPairs, epsilonPairs;
  std::vector<double> typeSigma, typeEpsilon;
  std::vector<bool> fixedPairs;
  // Raw derivatives in input order: scalars, each particle field, charge
  // corrections, then each table seed's type sums with their corrections.
  int64_t rawValues = 0, chargeValues = -1;
  std::vector<int64_t> fieldValues, typeSums;
  std::vector<int64_t> rowOffsets, sourceIndices;
  std::vector<double> weights;
  std::vector<int64_t> rawKinds;
  std::vector<double> rawVolumeConstants, rawFixedConstants;
  // Stable grouping preserves the host's particle order within each type.
  std::vector<int64_t> typeOffsets, particlesByType;
  double referenceVolume = 1.0;
  std::array<double, 5> corrections{}, correctionOffsets{};
  std::optional<FrameBatchLJConstants> lj;
  std::vector<double> initialSigma, initialEpsilon, typeCounts;
  std::vector<int64_t> excludedTypes;
  struct Seed {
    bool sigma = false, perType = false, geometric = false;
    int64_t rawBase = -1;
    std::vector<int64_t> mask;
    std::vector<double> weights;
  };
  std::vector<Seed> seeds;
  std::vector<double> orderedTypePairs;
  double ljSlopeFactor = 0.0, ljFixedFactor = 0.0;
  std::optional<FrameBatchPMEConstants> pme;
  std::vector<int64_t> chargeMap;
  std::vector<double> initialCharges;
  std::vector<double> initialPositions;
  std::vector<double> initialParameters;
  std::vector<int64_t> parameterKinds;
};
llvm::Expected<FrameBatchParameterPlan> makeFrameBatchParameterPlan(
    const Program &, const model::TunableSet &, const Control &, const System &,
    double coulomb);
}
#endif

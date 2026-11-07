// Immutable parameter metadata; values of theta never pass through this plan.
#ifndef MDIR_LIB_DRIVER_FRAMEBATCHPARAMETERPLAN_H
#define MDIR_LIB_DRIVER_FRAMEBATCHPARAMETERPLAN_H
#include "mdir/Driver/Model.h"
#include "mdir/Driver/Builder.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>
namespace mdir::driver::builder {
struct FrameBatchMappedResource {
  enum Kind { Field, Table, TupleField } kind;
  unsigned resource = 0, field = 0;
  std::vector<int64_t> map;
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
  // Stable grouping preserves the host's particle order within each type.
  std::vector<int64_t> typeOffsets, particlesByType;
};
llvm::Expected<FrameBatchParameterPlan> makeFrameBatchParameterPlan(
    const Program &, const model::TunableSet &, const Control &, const System &);
}
#endif

// Explicit factories retain the term translation units in static links.
#include "BuilderImpl.h"

using namespace mdir::driver::builder;

llvm::ArrayRef<TermBuilderFactory>
mdir::driver::builder::registerTermBuilders() {
  static const TermBuilderFactory factories[] = {
      &createExternalTermBuilder,    &createPullTermBuilder,
      &createRestraintTermBuilder,   &createBornTermBuilder,
      &createTupleTermBuilder,       &createPairTermBuilder,
      &createTripletTermBuilder,     &createModelTermBuilder,
      &createBondTermBuilder,        &createAngleTermBuilder,
      &createUreyBradleyTermBuilder, &createDihedralTermBuilder,
      &createImproperTermBuilder,    &createPairs14TermBuilder,
      &createCMapTermBuilder,
  };
  return factories;
}

std::vector<const TermBuilder *>
mdir::driver::builder::termBuildersByParameterRank(
    llvm::ArrayRef<const TermBuilder *> builders) {
  std::vector<const TermBuilder *> ordered(builders.begin(), builders.end());
  llvm::stable_sort(ordered, [](const TermBuilder *a, const TermBuilder *b) {
    return a->parameterRank() < b->parameterRank();
  });
  return ordered;
}

std::vector<const TermBuilder *>
mdir::driver::builder::termBuildersByPotentialRank(
    llvm::ArrayRef<const TermBuilder *> builders) {
  std::vector<const TermBuilder *> ordered(builders.begin(), builders.end());
  llvm::stable_sort(ordered, [](const TermBuilder *a, const TermBuilder *b) {
    return a->potentialRank() < b->potentialRank();
  });
  return ordered;
}

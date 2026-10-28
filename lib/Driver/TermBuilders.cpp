// The explicit order of the migrated term builders (D359).
// These references keep their translation units in the static library link.

#include "BuilderImpl.h"

using namespace mdir::driver::builder;

llvm::ArrayRef<const TermBuilder *>
mdir::driver::builder::registerTermBuilders() {
  static const TermBuilder *const builders[] = {&getExternalTermBuilder(),
                                                &getPullTermBuilder(),
                                                &getRestraintTermBuilder(),
                                                &getBornTermBuilder(),
                                                &getTupleTermBuilder(),
                                                &getPairTermBuilder(),
                                                &getTripletTermBuilder()};
  return builders;
}

std::vector<const TermBuilder *>
mdir::driver::builder::termBuildersByParameterRank() {
  auto builders = registerTermBuilders();
  std::vector<const TermBuilder *> ordered(builders.begin(), builders.end());
  llvm::stable_sort(ordered, [](const TermBuilder *a, const TermBuilder *b) {
    return a->parameterRank() < b->parameterRank();
  });
  return ordered;
}

std::vector<const TermBuilder *>
mdir::driver::builder::termBuildersByPotentialRank() {
  auto builders = registerTermBuilders();
  std::vector<const TermBuilder *> ordered(builders.begin(), builders.end());
  llvm::stable_sort(ordered, [](const TermBuilder *a, const TermBuilder *b) {
    return a->potentialRank() < b->potentialRank();
  });
  return ordered;
}

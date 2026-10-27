// The explicit order of the migrated term builders (D[term-registry]).
// These references keep their translation units in the static library link.

#include "BuilderImpl.h"

using namespace mdir::driver::builder;

llvm::ArrayRef<const TermBuilder *>
mdir::driver::builder::registerTermBuilders() {
  static const TermBuilder *const builders[] = {&getExternalTermBuilder()};
  return builders;
}

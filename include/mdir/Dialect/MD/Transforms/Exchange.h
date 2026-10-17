// The proof of exchange contracts of pair kernels.

#ifndef MDIR_DIALECT_MD_TRANSFORMS_EXCHANGE_H
#define MDIR_DIALECT_MD_TRANSFORMS_EXCHANGE_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

namespace mdir {
namespace md {

/// Compares `value`, which the pair kernel `kernel` computes, with its
/// value after the two particles are swapped: +1 if it is the same
/// (symmetric), −1 if it is the negative (antisymmetric), 0 if neither
/// could be shown. The kernel takes the distance, the displacement, and
/// two values of each gathered field, of the first particle and of the
/// second, as those of `md.sum_relation` do. `md-check-exchange` and the
/// verifiers of `md_dist` share this proof (D275).
int proveExchange(mlir::Block &kernel, mlir::Value value);

} // namespace md
} // namespace mdir

#endif // MDIR_DIALECT_MD_TRANSFORMS_EXCHANGE_H

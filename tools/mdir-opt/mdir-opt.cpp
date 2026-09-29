// mdir-opt: parses, verifies, transforms, and prints MDIR modules.

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

namespace mdir {
namespace test {
void registerTestOutlineKernels();
} // namespace test
} // namespace mdir

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  mdir::md::registerMDPasses();
  mdir::test::registerTestOutlineKernels();

  mlir::DialectRegistry registry;
  registry.insert<mdir::md::MDDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MDIR optimizer driver\n", registry));
}

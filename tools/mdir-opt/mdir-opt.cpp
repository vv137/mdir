// mdir-opt: parses, verifies, transforms, and prints MDIR modules.

#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
  mdir::registerMDIRConversionPasses();
  mdir::test::registerTestOutlineKernels();

  mlir::DialectRegistry registry;
  registry.insert<mdir::dyn::DynDialect, mdir::md::MDDialect,
                  mdir::md_exec::MDExecDialect, mdir::mdrt::MDRTDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MDIR optimizer driver\n", registry));
}

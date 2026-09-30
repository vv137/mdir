// mdir-opt: parses, verifies, transforms, and prints MDIR modules. It
// knows every pass and dialect of MLIR as well, so that it replays the
// pipeline of `mdir run` from a reproducer (docs/debugging.md).

#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/Transforms/Passes.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mdir {
namespace test {
void registerTestOutlineKernels();
} // namespace test
} // namespace mdir

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mdir::md::registerMDPasses();
  mdir::registerMDIRConversionPasses();
  mdir::md_exec::registerMDExecPasses();
  mdir::test::registerTestOutlineKernels();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mdir::dyn::DynDialect, mdir::md::MDDialect,
                  mdir::md_exec::MDExecDialect, mdir::mdrt::MDRTDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MDIR optimizer driver\n", registry));
}

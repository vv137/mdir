// SETTLE, SHAKE and RATTLE, the virtual sites, and the tests and stops
// of the constraints (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"
#include "ConstraintIR.h"
#include "DynamicsIR.h"
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

bool Builder::spreadsSitesDerived() const {
  const char *value = std::getenv("MDIR_SITE_SPREAD");
  return value && StringRef(value) == "derived";
}

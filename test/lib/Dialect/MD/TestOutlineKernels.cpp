// A test pass that turns kernels into functions, so that a kernel can be
// lowered and run on its own.

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir::md;

namespace {

/// For every op with a kernel, creates a `func.func` with the body of the
/// kernel.
///
/// The function is named `<enclosing function>.kernel<n>`, where `n` counts
/// the kernels of the enclosing function from 0. Its arguments are, in this
/// order:
///
///  1. the arguments of the kernel;
///  2. the arguments of the enclosing function that do not have a type of
///     the md dialect;
///  3. any other value that the kernel uses from outside, in order of first
///     use.
///
/// If the module declares a function of that name, the declaration must have
/// the expected type and receives the body.
struct TestOutlineKernels
    : public PassWrapper<TestOutlineKernels, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOutlineKernels)

  TestOutlineKernels() = default;
  TestOutlineKernels(const TestOutlineKernels &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "test-md-outline-kernels"; }
  StringRef getDescription() const final {
    return "Turns the kernels of md ops into functions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override;

  Option<bool> eraseMD{
      *this, "erase-md",
      llvm::cl::desc("Erase the ops of the md dialect from the module "
                     "after the kernels have been turned into functions"),
      llvm::cl::init(false)};

private:
  LogicalResult outline(Operation *op, FunctionOpInterface parent,
                        StringRef name);
};

} // namespace

/// Returns true if `op` has a kernel.
static bool hasKernel(Operation *op) {
  return isa<SumRelationOp, GatherRelationOp, SumParticlesOp, MapParticlesOp>(
      op);
}

/// Returns true if `type` belongs to the md dialect.
static bool isMDType(Type type) { return isa<MDDialect>(type.getDialect()); }

LogicalResult TestOutlineKernels::outline(Operation *op,
                                          FunctionOpInterface parent,
                                          StringRef name) {
  ModuleOp module = getOperation();
  Location loc = op->getLoc();
  Block &kernel = op->getRegion(0).front();
  Value yielded = cast<YieldOp>(kernel.getTerminator()).getOperand(0);

  // The values that become arguments, after those of the kernel.
  llvm::SetVector<Value> outer;
  for (BlockArgument argument : parent.getArguments())
    if (!isMDType(argument.getType()))
      outer.insert(argument);
  for (Operation &nested : kernel)
    for (Value operand : nested.getOperands()) {
      Block *block = operand.getParentBlock();
      if (block != &kernel)
        outer.insert(operand);
    }

  SmallVector<Type> inputs;
  for (BlockArgument argument : kernel.getArguments())
    inputs.push_back(argument.getType());
  for (Value value : outer)
    inputs.push_back(value.getType());
  FunctionType type =
      FunctionType::get(module.getContext(), inputs, {yielded.getType()});

  func::FuncOp function;
  if (Operation *existing = SymbolTable::lookupSymbolIn(module, name)) {
    function = dyn_cast<func::FuncOp>(existing);
    if (!function || !function.isExternal())
      return op->emitError()
             << "the name '" << name << "' is needed for a kernel, but is "
                "taken";
    if (function.getFunctionType() != type)
      return function.emitError()
             << "expected the declaration to have type " << type;
  } else {
    function = func::FuncOp::create(loc, name, type);
    module.push_back(function);
  }

  Block *entry = function.addEntryBlock();
  IRMapping mapping;
  unsigned index = 0;
  for (BlockArgument argument : kernel.getArguments())
    mapping.map(argument, entry->getArgument(index++));
  for (Value value : outer)
    mapping.map(value, entry->getArgument(index++));

  OpBuilder builder = OpBuilder::atBlockEnd(entry);
  for (Operation &nested : kernel.without_terminator())
    builder.clone(nested, mapping);
  func::ReturnOp::create(builder, loc,
                         ValueRange{mapping.lookupOrDefault(yielded)});
  return success();
}

void TestOutlineKernels::runOnOperation() {
  ModuleOp module = getOperation();

  SmallVector<FunctionOpInterface> parents;
  for (Operation &op : module)
    if (auto function = dyn_cast<FunctionOpInterface>(&op))
      if (isa<MDDialect>(op.getDialect()) && !function.isExternal())
        parents.push_back(function);

  for (FunctionOpInterface parent : parents) {
    SmallVector<Operation *> ops;
    parent->walk([&](Operation *op) {
      if (hasKernel(op))
        ops.push_back(op);
    });
    for (unsigned i = 0, e = ops.size(); i != e; ++i) {
      std::string name =
          (parent.getName() + ".kernel" + std::to_string(i)).str();
      if (failed(outline(ops[i], parent, name)))
        return signalPassFailure();
    }
  }

  if (!eraseMD)
    return;
  for (Operation &op : llvm::make_early_inc_range(module))
    if (isa<MDDialect>(op.getDialect()))
      op.erase();
}

namespace mdir {
namespace test {
void registerTestOutlineKernels() { PassRegistration<TestOutlineKernels>(); }
} // namespace test
} // namespace mdir

// A test pass that turns kernels into functions, so that a kernel can be
// lowered and run on its own.

#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
///  1. the arguments of the kernel; for an op over tuples, in place of the
///     coordinates, the positions of the members of the tuple, from which
///     the function computes the coordinates without a cell;
///  2. the arguments of the enclosing function that do not have a type of
///     the md dialect;
///  3. any other value that the kernel uses from outside, in order of first
///     use.
///
/// The function returns what the kernel yields.
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
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    math::MathDialect, vector::VectorDialect>();
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
  return isa<SumRelationOp, GatherRelationOp, SumTuplesOp, GatherTuplesOp,
             SumParticlesOp, MapParticlesOp>(op);
}

/// Returns true if `type` belongs to the md dialect.
static bool isMDType(Type type) { return isa<MDDialect>(type.getDialect()); }

LogicalResult TestOutlineKernels::outline(Operation *op,
                                          FunctionOpInterface parent,
                                          StringRef name) {
  ModuleOp module = getOperation();
  Location loc = op->getLoc();
  Block &kernel = op->getRegion(0).front();
  auto yield = cast<YieldOp>(kernel.getTerminator());

  // An op over tuples takes positions in place of coordinates.
  SmallVector<Coordinate, 2> coordinates;
  unsigned arity = 0;
  if (isa<SumTuplesOp, GatherTuplesOp>(op)) {
    coordinates = getCoordinates(
        op->getAttrOfType<DenseI32ArrayAttr>("coordinate_kinds").asArrayRef(),
        op->getAttrOfType<DenseI64ArrayAttr>("coordinate_members")
            .asArrayRef());
    arity = cast<RelationType>(op->getOperand(0).getType()).getArity();
  }
  ArrayRef<BlockArgument> kernelArguments =
      kernel.getArguments().drop_front(coordinates.size());

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

  Type positionType =
      VectorType::get({3}, Builder(module.getContext()).getF64Type());
  SmallVector<Type> inputs(arity, positionType);
  for (BlockArgument argument : kernelArguments)
    inputs.push_back(argument.getType());
  for (Value value : outer)
    inputs.push_back(value.getType());
  SmallVector<Type> results;
  for (Value value : yield.getOperands())
    results.push_back(value.getType());
  FunctionType type =
      FunctionType::get(module.getContext(), inputs, results);

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
  OpBuilder builder = OpBuilder::atBlockEnd(entry);
  IRMapping mapping;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    SmallVector<Value, 3> displacements;
    for (const Coordinate &displacement : getDisplacements(coordinate))
      displacements.push_back(arith::SubFOp::create(
          builder, loc, entry->getArgument(displacement.members[0]),
          entry->getArgument(displacement.members[1])));
    mapping.map(kernel.getArgument(index),
                emitCoordinate(builder, loc, coordinate.kind, displacements));
  }
  unsigned index = arity;
  for (BlockArgument argument : kernelArguments)
    mapping.map(argument, entry->getArgument(index++));
  for (Value value : outer)
    mapping.map(value, entry->getArgument(index++));

  for (Operation &nested : kernel.without_terminator())
    builder.clone(nested, mapping);
  SmallVector<Value> yielded;
  for (Value value : yield.getOperands())
    yielded.push_back(mapping.lookupOrDefault(value));
  func::ReturnOp::create(builder, loc, yielded);
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

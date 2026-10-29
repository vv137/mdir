// User-written staged model declarations and bodies, cloned as native IR.
#include "StagedModelIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mdir::driver::builder;

namespace {
llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

ModuleOp enclosingModule(Block *block) {
  if (!block)
    return {};
  Operation *owner = block->getParentOp();
  if (auto module = dyn_cast<ModuleOp>(owner))
    return module;
  return owner ? owner->getParentOfType<ModuleOp>() : ModuleOp{};
}

void remapOperands(Operation *operation, IRMapping &mapping) {
  operation->walk([&](Operation *nested) {
    for (OpOperand &operand : nested->getOpOperands())
      if (Value replacement = mapping.lookupOrNull(operand.get()))
        operand.set(replacement);
  });
}
} // namespace

struct StagedModelIR::Impl {
  MLIRContext *context;
  // Resource blobs may keep references to the input buffer after parsing.
  std::shared_ptr<llvm::SourceMgr> source;
  OwningOpRef<ModuleOp> module;
  mdir::md::PotentialOp potential;
};

StagedModelIR::StagedModelIR(std::unique_ptr<Impl> value)
    : impl(std::move(value)) {}
StagedModelIR::~StagedModelIR() = default;

llvm::Expected<std::shared_ptr<StagedModelIR>> StagedModelIR::parse(
    MLIRContext &context, llvm::StringRef text, llvm::StringRef sourceName,
    llvm::StringRef termName) {
  auto data = std::make_unique<Impl>();
  data->context = &context;
  data->source = std::make_shared<llvm::SourceMgr>();
  data->source->AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(text, sourceName), llvm::SMLoc());
  std::string diagnostics;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(diagnostics);
    diagnostic.print(stream);
    stream << '\n';
    return success();
  });
  // The program supplies symbols such as @atoms. Their ordinary combined
  // verification boundary remains after this input has been integrated.
  data->module = parseSourceFile<ModuleOp>(
      data->source, ParserConfig(&context, /*verifyAfterParse=*/false));
  if (!data->module)
    return invalid("cannot parse the model in stages '" + termName +
                   "' from '" + sourceName + "': " + diagnostics);
  auto potentials = data->module->getOps<mdir::md::PotentialOp>();
  if (!llvm::hasSingleElement(potentials))
    return invalid("the model in stages '" + termName +
                   "' must hold one 'md.potential'");
  data->potential = *potentials.begin();
  auto type = data->potential.getFunctionType();
  auto atoms = FlatSymbolRefAttr::get(&context, "atoms");
  SmallVector<Type> inputs = {
      mdir::md::FieldType::get(&context, atoms, 3, Float64Type::get(&context)),
      mdir::md::CellType::get(&context),
      mdir::md::FieldType::get(&context, atoms, 1,
                              IntegerType::get(&context, 32))};
  if (type.getInputs() != ArrayRef<Type>(inputs) ||
      !llvm::hasSingleElement(data->potential.getBody()))
    return invalid("the potential of the model in stages '" + termName +
                   "' must take the positions, the cell, and the atomic "
                   "numbers");
  Block &body = data->potential.getBody().front();
  if (body.empty() || !isa<mdir::md::ReturnOp>(body.back()))
    return invalid("the potential of the model in stages '" + termName +
                   "' has no 'md.return'");
  auto returned = cast<mdir::md::ReturnOp>(body.back());
  if (type.getNumResults() != 1 || !type.getResult(0).isF64() ||
      returned->getNumOperands() != 1 ||
      !returned->getOperand(0).getType().isF64())
    return invalid("the potential of the model in stages '" + termName +
                   "' must return one f64 energy");

  // Cloning must never leave a use of an SSA value in the source module.
  // Check SSA structure independently of the externally supplied symbols.
  bool malformed = false;
  data->module->walk([&](Operation *operation) {
    if (operation->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
        failed(OpTrait::impl::verifyIsIsolatedFromAbove(operation)))
      malformed = true;
  });
  DominanceInfo dominance(*data->module);
  data->module->walk([&](Operation *operation) {
    for (Value operand : operation->getOperands())
      if (!dominance.properlyDominates(operand, operation))
        malformed = true;
  });
  if (malformed)
    return invalid("the model in stages '" + termName +
                   "' has invalid SSA dependencies");
  return std::shared_ptr<StagedModelIR>(
      new StagedModelIR(std::move(data)));
}

llvm::Error StagedModelIR::cloneDeclarations(OpBuilder &builder) const {
  if (builder.getContext() != impl->context)
    return invalid("staged model and destination must share their build context");
  Block *target = builder.getInsertionBlock();
  if (!target || !isa<ModuleOp>(target->getParentOp()))
    return invalid("staged model declarations require a module insertion point");
  auto source = *impl->module;
  auto potential = impl->potential;
  IRMapping mapping;
  SmallVector<Operation *> cloned;
  for (Operation &operation : source.getBody()->getOperations())
    if (&operation != potential.getOperation())
      cloned.push_back(builder.clone(operation, mapping));
  for (Operation *operation : cloned)
    remapOperands(operation, mapping);
  return llvm::Error::success();
}

llvm::Expected<Value>
StagedModelIR::cloneBody(OpBuilder &builder, ValueRange inputs) const {
  if (builder.getContext() != impl->context)
    return invalid("staged model and destination must share their build context");
  auto potential = impl->potential;
  Block &body = potential.getBody().front();
  if (inputs.size() != body.getNumArguments())
    return invalid("staged model argument count differs from its input boundary");
  ModuleOp destination = enclosingModule(builder.getInsertionBlock());
  if (!destination)
    return invalid("staged model body requires a program insertion point");
  IRMapping mapping;
  for (auto [argument, input] : llvm::zip(body.getArguments(), inputs)) {
    if (argument.getType() != input.getType())
      return invalid("staged model argument type differs from its input boundary");
    if (enclosingModule(input.getParentBlock()) != destination)
      return invalid("staged model operands must belong to the destination program");
    mapping.map(argument, input);
  }
  SmallVector<Operation *> cloned;
  for (Operation &operation : body.without_terminator())
    cloned.push_back(builder.clone(operation, mapping));
  for (Operation *operation : cloned)
    remapOperands(operation, mapping);
  Value result = mapping.lookupOrNull(body.back().getOperand(0));
  if (!result)
    return invalid("staged model return has no cloned value");
  return result;
}

// A loop's signature and checked yield derive from the same physical state.
#include "EntryLoopIR.h"
#include "mlir/IR/Builders.h"
#include <optional>
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
using namespace mdir::driver::builder;
llvm::Expected<scf::ForOp> mdir::driver::builder::buildEntryLoop(
    OpBuilder &builder, llvm::StringRef scope, Value lower, Value upper,
    Value step, ValueRange initial,
    llvm::function_ref<llvm::Expected<llvm::SmallVector<Value>>(
        OpBuilder &, Value, ValueRange)> body) {
  std::optional<llvm::Error> failure;
  auto invalid = [&](const llvm::Twine &reason) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native " + scope + " loop: " + reason);
  };
  auto loop = scf::ForOp::create(builder, builder.getUnknownLoc(), lower,
      upper, step, initial,
      [&](OpBuilder &at, Location loc, Value index, ValueRange arguments) {
        Block *block = at.getInsertionBlock();
        auto result = body(at, index, arguments);
        if (!result) {
          failure.emplace(result.takeError());
          return;
        }
        if (at.getInsertionBlock() != block) {
          failure.emplace(invalid("body left its insertion scope"));
          return;
        }
        if (result->size() != initial.size()) {
          failure.emplace(invalid("yield has " + llvm::Twine(result->size()) +
              " values, expected " + llvm::Twine(initial.size())));
          return;
        }
        for (auto [slot, value] : llvm::enumerate(*result)) {
          if (!value || value.getType() != initial[slot].getType()) {
            failure.emplace(invalid("yield type differs at physical slot " + llvm::Twine(slot)));
            return;
          }
        }
        scf::YieldOp::create(at, loc, *result);
      });
  if (failure) {
    loop.erase();
    return std::move(*failure);
  }
  return loop;
}

llvm::Error mdir::driver::builder::checkFrameInputs(Operation &frame, ValueRange forbidden) {
  llvm::DenseSet<Value> excluded, visited;
  for (Value value : forbidden) if (value) excluded.insert(value);
  llvm::DenseSet<Operation *> inspected;
  bool stale = false;
  frame.walk([&](Operation *operation) {
    if (stale) return;
    SmallVector<Value> pending(operation->getOperands());
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second) continue;
      if (excluded.contains(value)) {
        stale = true;
        return;
      }
      // Trace header expressions, aliases and captures in producer regions. Internal definitions
      // are visited by the operation walk; frame block arguments are local.
      if (auto *definition = value.getDefiningOp();
          definition && !frame.isAncestor(definition) && inspected.insert(definition).second)
        definition->walk([&](Operation *producer) {
          llvm::append_range(pending, producer->getOperands());
        });
    }
  });
  if (stale)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
        "native frame iteration read entry positions or cell values");
  return llvm::Error::success();
}

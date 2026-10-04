// Activity is a proof of dependence, not a numerical test of a tangent.
#ifndef MDIR_DIALECT_MD_TRANSFORMS_ACTIVITY_H
#define MDIR_DIALECT_MD_TRANSFORMS_ACTIVITY_H
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <string>
namespace mdir { namespace md {
enum class Activity { Inactive, Active, Unknown };
struct ActivityResult {
  Activity dependence = Activity::Inactive;
  std::string reason;
  mlir::Operation *unknownOperation = nullptr;
};
class ActivityAnalysis {
public:
  explicit ActivityAnalysis(mlir::Value variable);
  void addKnownBlock(mlir::Block *block) { knownBlocks.insert(block); }
  ActivityResult classify(mlir::Value value);
private:
  mlir::Value variable;
  llvm::DenseSet<mlir::Block *> knownBlocks;
  llvm::DenseMap<mlir::Value, ActivityResult> verdicts;
};
} }
#endif

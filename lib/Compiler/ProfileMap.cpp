// The profile map of a lowered program (D398, #840); see
// include/mdir/Compiler/ProfileMap.h and docs/profile-map.md.

#include "mdir/Compiler/ProfileMap.h"

#include "mdir/Dialect/MDExec/ProfileMap.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <vector>

using namespace mlir;
using namespace mdir;

/// The function that marks a region in the copy that is translated.
static constexpr llvm::StringLiteral kMarkName = "mdir.profile.region";

namespace {
/// A region of the module, in the order of a walk of its functions.
struct MappedRegion {
  std::string parent;
  DictionaryAttr record;
  std::string precision;
  std::string function;
};
} // namespace

static llvm::json::Value getJSON(Attribute value) {
  if (auto text = dyn_cast<StringAttr>(value))
    return text.getValue().str();
  if (auto list = dyn_cast<ArrayAttr>(value)) {
    llvm::json::Array array;
    for (Attribute element : list)
      array.push_back(getJSON(element));
    return array;
  }
  return nullptr;
}

llvm::Expected<std::string>
compiler::getProfileMap(ModuleOp module, const driver::Control &control) {
  OwningOpRef<ModuleOp> copy(cast<ModuleOp>(module->clone()));
  MLIRContext *context = copy->getContext();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(copy->getBody());
  Type i64 = builder.getI64Type();
  auto mark = LLVM::LLVMFuncOp::create(
      builder, copy->getLoc(), kMarkName,
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), {i64}));

  std::vector<MappedRegion> regions;
  for (auto function : copy->getOps<LLVM::LLVMFuncOp>()) {
    if (function.isExternal())
      continue;
    function.walk<WalkOrder::PreOrder>([&](omp::ParallelOp parallel) {
      MappedRegion region;
      region.parent = function.getSymName().str();
      region.record = profile::find(parallel.getLoc());
      region.precision = profile::getPrecision(parallel.getRegion());
      Block &entry = parallel.getRegion().front();
      OpBuilder at = OpBuilder::atBlockBegin(&entry);
      Value index = LLVM::ConstantOp::create(
          at, parallel.getLoc(), i64,
          at.getI64IntegerAttr(static_cast<int64_t>(regions.size())));
      LLVM::CallOp::create(at, parallel.getLoc(), mark, ValueRange{index});
      regions.push_back(std::move(region));
    });
  }

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> translated =
      translateModuleToLLVMIR(*copy, llvmContext);
  if (!translated)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the profile map: cannot translate the "
                                   "lowered module to LLVM IR");
  if (llvm::Function *marker = translated->getFunction(kMarkName))
    for (llvm::User *user : marker->users())
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(user))
        if (auto *index =
                llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0)))
          if (index->getZExtValue() < regions.size())
            regions[index->getZExtValue()].function =
                call->getFunction()->getName().str();

  llvm::json::Array entries;
  for (const MappedRegion &region : regions) {
    llvm::json::Object entry;
    // A region that no translation outlined (none is known) has no
    // function: a reader takes that as a defect of the map.
    entry["function"] = region.function.empty()
                            ? llvm::json::Value(nullptr)
                            : llvm::json::Value(region.function);
    entry["parent"] = region.parent;
    entry["source"] = nullptr;
    if (region.record)
      for (NamedAttribute named : region.record)
        entry[named.getName().getValue()] = getJSON(named.getValue());
    if (!entry.getArray("potentials"))
      entry["potentials"] = llvm::json::Array();
    if (!entry.getArray("terms"))
      entry["terms"] = llvm::json::Array();
    if (!entry.get("class"))
      entry["class"] = nullptr;
    entry["precision"] = region.precision;
    entries.push_back(std::move(entry));
  }
  llvm::json::Object map;
  map["format"] = 1;
  map["target"] = control.target == driver::Target::GPU ? "GPU" : "CPU";
  map["precision"] = control.precision == driver::Precision::Single ? "SINGLE"
                     : control.precision == driver::Precision::Mixed
                         ? "MIXED"
                         : "DOUBLE";
  map["threads"] = static_cast<int64_t>(control.threads);
  map["regions"] = std::move(entries);
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(map))) << "\n";
  return text;
}

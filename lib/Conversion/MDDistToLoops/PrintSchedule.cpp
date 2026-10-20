// Inspection of the md_dist halo schedule proof (D[ml2-schedule-proof]).

#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/MDDist/MDDistOps.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mdir::md_dist;

namespace mdir {
#define GEN_PASS_DEF_MDDISTPRINTSCHEDULE
#include "mdir/Conversion/Passes.h.inc"

namespace {

class PrintSchedule
    : public impl::MDDistPrintScheduleBase<PrintSchedule> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    AsmState names(module);
    bool failedProof = false;
    module.walk([&](Block *block) {
      if (failedProof)
        return;
      SmallVector<HaloScheduleRecord, 0> records;
      if (failed(proveHaloSchedule(*block, records))) {
        failedProof = true;
        return;
      }
      for (const HaloScheduleRecord &record : records) {
        llvm::raw_ostream &out = llvm::errs();
        StringRef symbol = "anonymous";
        for (Operation *parent = record.transfer->getParentOp(); parent;
             parent = parent->getParentOp()) {
          if (auto name = parent->getAttrOfType<StringAttr>("sym_name")) {
            symbol = name.getValue();
            break;
          }
        }
        out << "md_dist schedule @" << symbol << " "
            << (isa<HaloCopyOp>(record.transfer) ? "G" : "G^T")
            << " source=";
        for (auto [i, source] : llvm::enumerate(record.sources)) {
          if (i)
            out << ",";
          source.printAsOperand(out, names);
        }
        out << " halo=";
        record.halo.printAsOperand(out, names);
        out << " reach=";
        cast<HaloType>(record.halo.getType()).getReach().print(out);
        out << " layout=";
        record.layout.printAsOperand(out, names);
        out << " result=";
        record.result.printAsOperand(out, names);
        auto printOps = [&](StringRef label, ArrayRef<Operation *> ops) {
          out << " " << label << "=[";
          for (auto [i, op] : llvm::enumerate(ops)) {
            if (i)
              out << ",";
            if (op->getNumResults())
              op->getResult(0).printAsOperand(out, names);
            else
              out << op->getName().getStringRef();
          }
          out << "]";
        };
        printOps("start-after", record.startAfter);
        out << " ready-at-entry=[";
        for (auto [i, value] : llvm::enumerate(record.readyAtEntry)) {
          if (i)
            out << ",";
          value.printAsOperand(out, names);
        }
        out << "]";
        printOps("complete-before", record.completeBefore);
        out << " first-completion=";
        if (record.firstCompletion && record.firstCompletion->getNumResults())
          record.firstCompletion->getResult(0).printAsOperand(out, names);
        else if (record.firstCompletion)
          out << record.firstCompletion->getName().getStringRef();
        else
          out << "none";
        printOps("independent-owned", record.independentOwnedWork);
        out << "\n";
      }
    });
    if (failedProof)
      signalPassFailure();
  }
};

} // namespace
} // namespace mdir

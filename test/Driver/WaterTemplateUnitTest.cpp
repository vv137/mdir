// Verify all class arities and the rotating state of the GPU scheduler.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mdir::kernels;

int runWaterTemplateUnitIRTest() {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                      scf::SCFDialect, vector::VectorDialect>();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());
  auto vec = VectorType::get({3}, builder.getF32Type());
  auto tensor = VectorType::get({9}, builder.getF64Type());
  const char *names[] = {"ww", "sw", "ss", "energy", "three", "select"};
  for (unsigned k = 0; k != 6; ++k) {
    auto kind = static_cast<TemplatePairClass>(k < 3 ? k : 0);
    unsigned ni = kind == TemplatePairClass::WaterWater ? 3 : 1;
    unsigned nj = kind == TemplatePairClass::SoluteSolute ? 1 : 3;
    std::string prefix = std::string("unit_") + names[k];
    SmallVector<Type> firstTypes(ni, vec), secondTypes{builder.getI32Type()};
    secondTypes.append(nj, vec);
    auto writeI = func::FuncOp::create(builder, loc, prefix + "_first",
        builder.getFunctionType(firstTypes, {}));
    writeI.setPrivate();
    auto writeJ = func::FuncOp::create(builder, loc, prefix + "_second",
        builder.getFunctionType(secondTypes, {}));
    writeJ.setPrivate();
    SmallVector<Type> arguments{builder.getI32Type(), builder.getI32Type(),
        builder.getI32Type(), builder.getI1Type()};
    arguments.append(ni, vec);
    arguments.append({builder.getI32Type(), builder.getF32Type()});
    arguments.append(nj, vec);
    arguments.append({builder.getI32Type(), builder.getF32Type(), vec, builder.getI32Type()});
    auto function = func::FuncOp::create(builder, loc, prefix,
        builder.getFunctionType(arguments, {builder.getF64Type(), tensor}));
    Block *entry = function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    auto args = entry->getArguments();
    TemplateUnit unit{kind, args[0], args[1], args[2], args[3]};
    unit.first.assign(args.begin() + 4, args.begin() + 4 + ni);
    unit.parameters.assign(args.begin() + 4 + ni, args.begin() + 6 + ni);
    unit.forces = k != 3;
    unit.sumTypes = {builder.getF64Type(), tensor};
    unit.hasSite = k != 2 && k != 4;
    unit.alpha = 0.75;
    // A speculatable body runs on every lane; unpaired lanes select +0.
    unit.speculate = k == 5;
    auto result = emitTemplateClassUnit(builder, loc, unit,
        [&](OpBuilder &, Value) {
          TemplatePartner partner;
          partner.frame.assign(args.begin() + 6 + ni, args.begin() + 6 + ni + nj);
          partner.parameters = {args[6 + ni + nj], args[7 + ni + nj]};
          partner.shift = args[8 + ni + nj]; partner.mask = args[9 + ni + nj];
          return partner;
        }, [&](OpBuilder &body, ValueRange coordinates, ValueRange, ValueRange,
               ValueRange sites) {
          assert(sites.size() == (k == 2 || k == 4 ? 0u : k == 1 ? 1u : 2u));
          // A synthetic typed contribution tests scheduling, not a second
          // physical expression. Production supplies the differentiated body.
          Value d = coordinates.front();
          Value zero = arith::ConstantOp::create(body, loc, DenseElementsAttr::get(vec, 0.0f));
          TemplateContribution contribution;
          if (unit.forces) {
            contribution.firstForces.assign(ni, zero);
            contribution.secondForces.assign(nj, zero);
            contribution.firstForces[0] = arith::NegFOp::create(body, loc, d);
            contribution.secondForces[0] = d;
          }
          Value square = arith::MulFOp::create(body, loc, d, d);
          Value sum = vector::ReductionOp::create(body, loc, vector::CombiningKind::ADD, square);
          Value wide = arith::ExtFOp::create(body, loc, body.getF64Type(), sum);
          contribution.sums = {wide, vector::BroadcastOp::create(body, loc, tensor, wide)};
          return contribution;
        }, [&](OpBuilder &body, ValueRange forces) {
          func::CallOp::create(body, loc, writeI.getSymName(), TypeRange{}, forces);
        }, [&](OpBuilder &body, Value at, ValueRange forces) {
          SmallVector<Value> values{at};
          llvm::append_range(values, forces);
          func::CallOp::create(body, loc, writeJ.getSymName(), TypeRange{}, values);
        });
    func::ReturnOp::create(builder, loc, result);
    unsigned firstSites = 0, partnerSites = 0;
    bool repeatedSite = false;
    function.walk([&](arith::MulFOp product) {
      bool site = false;
      for (Value operand : product->getOperands()) {
        DenseFPElementsAttr literal;
        site |= matchPattern(operand, m_Constant(&literal)) && literal.isSplat() &&
                literal.getSplatValue<APFloat>().convertToDouble() == 0.75;
      }
      if (!site) return;
      // The partner's site is formed from its two rotated offsets once per
      // turn (step 1, inside the batch loop of step 32), not rotated itself.
      if (auto parent = product->getParentOfType<scf::ForOp>()) {
        APInt step, outer;
        auto batch = parent->getParentOfType<scf::ForOp>();
        repeatedSite |= !matchPattern(parent.getStep(), m_ConstantInt(&step)) ||
                        step.getSExtValue() != 1 || !batch ||
                        !matchPattern(batch.getStep(), m_ConstantInt(&outer)) ||
                        outer.getSExtValue() != 32;
        ++partnerSites;
      } else {
        ++firstSites;
      }
    });
    unsigned expectedFirst = unit.hasSite && ni == 3 ? 1 : 0;
    unsigned expectedPartner = unit.hasSite && nj == 3 ? 1 : 0;
    if (repeatedSite || firstSites != expectedFirst || partnerSites != expectedPartner) {
      llvm::errs() << "formed-site geometry repeated in the rotation or missing\n";
      module.erase();
      return 1;
    }
    builder.setInsertionPointAfter(function);
  }
  if (failed(verify(module))) { module.erase(); return 1; }
  module.print(llvm::outs());
  module.erase();
  return 0;
}

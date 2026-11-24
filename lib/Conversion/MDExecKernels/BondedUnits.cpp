// Atom-local evaluation of the application's differentiated tuple region.
#include "mdir/Conversion/BondedUnits.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Support/ForceWords.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;
using namespace mdir::kernels::detail;
namespace {
struct UnitKernel : NeighborBuildCode {
  using NeighborBuildCode::NeighborBuildCode;
  void failIf(Value bad) {
    ifThen(bad, [&] {
      LLVM::InlineAsmOp::create(b, loc, TypeRange{}, ValueRange{}, "trap;", "",
                                true, false, LLVM::TailCallKind::None,
                                LLVM::AsmDialectAttr(), ArrayAttr());
    });
  }
  Value quantize(Value value, double scale) {
    Type type = value.getType();
    Value scaled = mulf(value, createReal(b, loc, type, scale));
    Value fits = andi(
        cmpf(CmpFPredicate::OGE, scaled, createReal(b, loc, type, -0x1p63)),
        cmpf(CmpFPredicate::OLT, scaled, createReal(b, loc, type, 0x1p63)));
    failIf(notNarrow(fits));
    return fptosi(i64,
                  roundeven(select(fits, scaled, createReal(b, loc, type, 0))));
  }
  void addWord(Value dest, ValueRange indices, Value value) {
    Value old = memref::AtomicRMWOp::create(b, loc, arith::AtomicRMWKind::addi,
                                            value, dest, indices);
    Value sum = addi(old, value);
    failIf(cmpi(CmpIPredicate::slt, andi(xori(sum, old), xori(sum, value)),
                cl(0)));
  }
  void emit(ArrayRef<md_exec::TupleForOp> ops, Value block, Value lane,
            Value stream, Value place, Value box, Value inverse, Value global,
            ValueRange outputs, ValueRange shared, int mode) {
    Value positions = shared[0], liveIds = shared[1], owned = shared[2],
          forces = shared[3];
    auto word = [&](Value p) { return load(stream, {p}); };
    auto offset = [&](Value p) { return indexCast(index, word(p)); };
    Value base = offset(block), atoms = offset(base);
    Value imports = addi(base, cx(1 + ops.size())),
          instructions = addi(imports, atoms);
    auto firstOp = ops.front();
    auto lanes = [&](Value n, function_ref<void(Value)> body) {
      return scf::ForOp::create(
          b, loc, lane, n, cx(128), ValueRange{},
          [&](OpBuilder &inner, Location l, Value k, ValueRange) {
            body(k);
            scf::YieldOp::create(inner, l);
          });
    };
    lanes(atoms, [&](Value i) {
      Value rep = word(addi(imports, i)),
            here = cmpi(CmpIPredicate::sge, rep, ci(0));
      Value slot = indexCast(index, select(here, rep, xori(rep, ci(-1))));
      Value atom = place ? indexCast(index, load(place, {slot})) : slot;
      store(atom, liveIds, {i});
      store(select(here, ci(1), ci(0)), owned, {i});
      storeElement(b, loc, loadElement(b, loc, firstOp.getPositions(), atom),
                   positions, i);
      if (mode != 3)
        for (int c = 0; c < 3; ++c)
          store(cl(0), forces, {i, cx(c)});
    });
    gpu::BarrierOp::create(b, loc);
    auto isOwned = [&](Value i) {
      return cmpi(CmpIPredicate::ne, load(owned, {i}), ci(0));
    };
    for (unsigned set = 0; set < ops.size(); ++set) {
      auto op = ops[set];
      Value terms = offset(addi(base, cx(set + 1)));
      int arity = op.getArity();
      int stride = mode == 1 ? (arity > 3 ? 3 : 2) : 1 + arity;
      auto termLoop = lanes(terms, [&](Value k) {
        Value first = addi(instructions, muli(k, cx(stride))),
              t = offset(first);
        Value packed;
        if (mode == 1) {
          auto extend = [&](Value x) {
            return arith::ExtUIOp::create(b, loc, i64, x).getResult();
          };
          packed = extend(word(addi(first, cx(1))));
          if (arity > 3)
            packed =
                ori(packed, shli(extend(word(addi(first, cx(2)))), cl(32)));
        }
        Values slots;
        for (int j = 0; j < arity; ++j)
          slots.push_back(
              mode == 1
                  ? indexCast(index, andi(shrui(packed, cl(10 * j)), cl(1023)))
                  : offset(addi(first, cx(j + 1))));
        auto yields = emitWorkUnitTuple(b, op, t, slots, positions, liveIds,
                                        box, inverse);
        for (int j = 0; j < arity; ++j)
          ifThen(isOwned(slots[j]), [&] {
            Value dest = mode == 3 ? global : forces;
            Value row = mode == 3 ? load(liveIds, {slots[j]}) : slots[j];
            for (int c = 0; c < 3; ++c)
              addWord(
                  dest, {row, cx(c)},
                  quantize(convertReal(
                               b, loc,
                               vector::ExtractOp::create(b, loc, yields[j], c),
                               getElementTypeOrSelf(op.getOuts()[0].getType())),
                           forceWordScale));
            if (j == 0)
              for (auto [i, buffer] : llvm::enumerate(outputs)) {
                Value v = yields[arity + i];
                if (isa<FloatType>(v.getType()))
                  addWord(buffer, {cx(0)}, quantize(v, bondedEnergyScale));
                else
                  storeElement(b, loc, v, buffer, t);
              }
          });
      });
      if (op->hasAttr("mdir.bonded_cmap"))
        termLoop->setAttr("mdir.fixed_slot_rounding", b.getUnitAttr());
      instructions = addi(instructions, muli(terms, cx(stride)));
    }
    if (mode != 3) {
      gpu::BarrierOp::create(b, loc);
      lanes(atoms, [&](Value i) {
        ifThen(isOwned(i), [&] {
          Value atom = load(liveIds, {i});
          for (int c = 0; c < 3; ++c)
            addWord(global, {atom, cx(c)}, load(forces, {i, cx(c)}));
        });
      });
    }
  }
};
} // namespace
gpu::LaunchOp kernels::emitBondedUnits(OpBuilder &b,
                                       ArrayRef<md_exec::TupleForOp> ops,
                                       Value blocks, Value instructions,
                                       Value place, Value box, Value inverse,
                                       Value forceWords,
                                       ValueRange observerTerms, int mode) {
  auto first = ops.front();
  Location loc = first.getLoc();
  auto space =
      gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup);
  auto mem = [&](ArrayRef<int64_t> shape, Type type) {
    return MemRefType::get(shape, type, MemRefLayoutAttrInterface(), space);
  };
  Type real = cast<MemRefType>(first.getPositions().getType()).getElementType();
  SmallVector<Type> shared{mem({256, 3}, real), mem({256}, b.getIndexType()),
                           mem({256}, b.getI32Type()),
                           mem({256, 3}, b.getI64Type())};
  Value one = createIndex(b, loc, 1);
  auto launch =
      gpu::LaunchOp::create(b, loc, blocks, one, one, createIndex(b, loc, 128),
                            one, one, nullptr, nullptr, ValueRange{}, shared);
  OpBuilder at = OpBuilder::atBlockEnd(&launch.getBody().front());
  UnitKernel(at, loc).emit(ops, launch.getBlockIds().x, launch.getThreadIds().x,
                           instructions, place, box, inverse, forceWords,
                           observerTerms,
                           launch.getWorkgroupAttributionBBArgs(), mode);
  gpu::TerminatorOp::create(at, loc);
  return launch;
}

#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::kernels;

std::array<Value, 3> mdir::kernels::emitPackTemplateFrame(
    OpBuilder &builder, Location loc, ArrayRef<Value> whole,
    Value shift, Type real) {
  assert(whole.size() == 3);
  std::array<Value, 3> packed;
  for (unsigned site = 0; site != 3; ++site) {
    Value framed = arith::AddFOp::create(builder, loc, whole[site], shift);
    packed[site] = convertReal(builder, loc, framed, real);
  }
  return packed;
}

SmallVector<Value, 5> mdir::kernels::emitTemplateWindowCoordinates(
    OpBuilder &builder, Location loc, TemplatePairClass kind,
    ArrayRef<Value> first, ArrayRef<Value> second, Value shift) {
  assert(first.size() == (kind == TemplatePairClass::WaterWater ? 3u : 1u));
  assert(second.size() == (kind == TemplatePairClass::SoluteSolute ? 1u : 3u));
  auto sub = [&](Value a, Value b) -> Value {
    return arith::SubFOp::create(builder, loc, a, b);
  };
  Value anchor = arith::AddFOp::create(builder, loc, second[0], shift);
  SmallVector<Value, 5> coordinates{sub(first[0], anchor)};
  if (kind == TemplatePairClass::WaterWater) {
    coordinates.push_back(sub(first[1], first[0]));
    coordinates.push_back(sub(first[2], first[0]));
  }
  if (kind != TemplatePairClass::SoluteSolute) {
    coordinates.push_back(sub(second[1], second[0]));
    coordinates.push_back(sub(second[2], second[0]));
  }
  return coordinates;
}

namespace {
static Value materializeWindowValue(OpBuilder &builder, Value value,
                                    Block &source, IRMapping &mapping) {
  if (Value known = mapping.lookupOrNull(value)) return known;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != &source) return value;
  // A structured scalar expression can capture another value of the
  // window in a nested region without listing it as a direct operand.
  definition->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      materializeWindowValue(builder, operand, source, mapping);
  });
  builder.clone(*definition, mapping);
  return mapping.lookup(value);
}
} // namespace

TemplateContribution mdir::kernels::emitTemplateWindowContributions(
    OpBuilder &builder, Location loc, mdir::md_exec::TemplateForOp loop,
    unsigned tag, IRMapping &mapping) {
  assert(tag < 3 && loop.getOuts().size() <= 1);
  constexpr unsigned first[] = {3, 1, 1};
  constexpr unsigned second[] = {3, 3, 1};
  unsigned arity = first[tag] + second[tag];
  unsigned outs = loop.getOuts().size();
  unsigned columns = outs + loop.getReduce().size();
  Block &source = loop->getRegion(tag).front();
  Operation *yield = source.getTerminator();
  auto zero = [&](Type type) -> Value {
    return arith::ConstantOp::create(builder, loc, builder.getZeroAttr(type));
  };
  TemplateContribution result;
  if (outs) {
    Type force = mdir::md_exec::getKernelValueType(loop.getOuts()[0].getType());
    result.firstForces.assign(first[tag], zero(force));
    result.secondForces.assign(second[tag], zero(force));
  }
  for (Value initial : loop.getReduce()) result.sums.push_back(zero(initial.getType()));
  unsigned offset = 0;
  for (unsigned column = 0; column != columns; ++column) {
    bool scatter = column < outs;
    unsigned stride = scatter ? arity : 1;
    unsigned pairs = loop.getPairs()[tag * columns + column];
    for (unsigned pair = 0; pair != pairs; ++pair) {
      Value displacement = materializeWindowValue(builder,
          yield->getOperand(offset + stride * pairs + pair), source, mapping);
      Value square = arith::MulFOp::create(builder, loc, displacement, displacement);
      Value r2 = vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, square);
      double cutoff = loop.getCutoff().convertToDouble();
      Value cutoff2 = arith::ConstantOp::create(builder, loc,
          FloatAttr::get(r2.getType(), cutoff * cutoff));
      Value inside = arith::CmpFOp::create(builder, loc,
          arith::CmpFPredicate::OLT, r2, cutoff2);
      auto guarded = scf::IfOp::create(builder, loc, inside,
          [&](OpBuilder &body, Location) {
            IRMapping local = mapping;
            SmallVector<Value> values;
            for (unsigned member = 0; member != stride; ++member)
              values.push_back(materializeWindowValue(body,
                  yield->getOperand(offset + pair * stride + member), source, local));
            scf::YieldOp::create(body, loc, values);
          }, [&](OpBuilder &body, Location) {
            SmallVector<Value> values;
            for (unsigned member = 0; member != stride; ++member) {
              Type type = yield->getOperand(offset + pair * stride + member).getType();
              values.push_back(arith::ConstantOp::create(body, loc, body.getZeroAttr(type)));
            }
            scf::YieldOp::create(body, loc, values);
          });
      for (unsigned member = 0; member != stride; ++member) {
        Value *destination = !scatter ? &result.sums[column - outs]
            : member < first[tag] ? &result.firstForces[member]
                                 : &result.secondForces[member - first[tag]];
        *destination = arith::AddFOp::create(builder, loc, *destination,
                                             guarded.getResult(member));
      }
    }
    offset += pairs * (stride + 1);
  }
  return result;
}

namespace {
static Value word(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 32);
}
static Value shuffle(OpBuilder &builder, Location loc, Value value,
                     Value source, gpu::ShuffleMode mode) {
  auto scalar = [&](Value x) -> Value {
    assert(x.getType().isF32() || x.getType().isInteger(32));
    return gpu::ShuffleOp::create(builder, loc, x, source,
        word(builder, loc, 32), mode).getResult(0);
  };
  if (auto type = dyn_cast<VectorType>(value.getType())) {
    SmallVector<Value> components;
    for (int64_t i = 0; i != type.getNumElements(); ++i)
      components.push_back(scalar(vector::ExtractOp::create(builder, loc, value, i)));
    return vector::FromElementsOp::create(builder, loc, type, components);
  }
  return scalar(value);
}
} // namespace

SmallVector<Value, 2> mdir::kernels::emitTemplateClassUnit(
    OpBuilder &builder, Location loc, const TemplateUnit &unit,
    llvm::function_ref<TemplatePartner(OpBuilder &, Value)> loadPartner,
    llvm::function_ref<TemplateContribution(OpBuilder &, ValueRange,
                                            ValueRange, ValueRange)> evaluate,
    llvm::function_ref<void(OpBuilder &, ValueRange)> writeFirst,
    llvm::function_ref<void(OpBuilder &, Value, ValueRange)> writeSecond) {
  unsigned ni = unit.kind == TemplatePairClass::WaterWater ? 3 : 1;
  unsigned nj = unit.kind == TemplatePairClass::SoluteSolute ? 1 : 3;
  assert(unit.first.size() == ni);
  Type forceType = unit.first.front().getType();
  assert(cast<VectorType>(forceType).getElementType().isF32());
  unsigned fi = unit.forces ? ni : 0, fj = unit.forces ? nj : 0;
  SmallVector<Type> resultTypes(fi + fj, forceType);
  llvm::append_range(resultTypes, unit.sumTypes);
  Value u = arith::AndIOp::create(builder, loc, unit.lane, word(builder, loc, 15));
  Value half = arith::AndIOp::create(builder, loc, unit.lane, word(builder, loc, 16));
  Value previous = arith::AndIOp::create(builder, loc,
      arith::AddIOp::create(builder, loc, u, word(builder, loc, 15)), word(builder, loc, 15));
  Value source = arith::OrIOp::create(builder, loc, half, previous);
  SmallVector<Value> initial;
  for (unsigned i = 0; i != fi; ++i) initial.push_back(createZero(builder, loc, forceType));
  for (Type type : unit.sumTypes) initial.push_back(createZero(builder, loc, type));
  auto batches = scf::ForOp::create(builder, loc, unit.begin, unit.end,
      word(builder, loc, 32), initial,
      [&](OpBuilder &batch, Location, Value begin, ValueRange own) {
        Value entry = arith::AddIOp::create(batch, loc, begin, unit.lane);
        Value has = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::slt, entry, unit.end);
        // A partial final batch still loads a valid entry. Its cleared mask
        // participates in every shuffle, and its J destination is not written.
        Value safe = arith::SelectOp::create(batch, loc, has, entry, unit.begin);
        TemplatePartner partner = loadPartner(batch, safe);
        assert(partner.frame.size() == nj);
        Value mask = arith::SelectOp::create(batch, loc, has, partner.mask, word(batch, loc, 0));
        unsigned shiftAt = nj + partner.parameters.size();
        unsigned maskAt = shiftAt + 1, fjAt = maskAt + 1, fiAt = fjAt + fj;
        unsigned sumsAt = fiAt + fi;
        SmallVector<Value> carried(partner.frame.begin(), partner.frame.end());
        llvm::append_range(carried, partner.parameters);
        carried.append({partner.shift, mask});
        for (unsigned i = 0; i != fj; ++i) carried.push_back(createZero(batch, loc, forceType));
        llvm::append_range(carried, own);
        auto turns = scf::ForOp::create(batch, loc, word(batch, loc, 0),
            word(batch, loc, 16), word(batch, loc, 1), carried,
            [&](OpBuilder &turn, Location, Value, ValueRange state) {
              Value bit = arith::AndIOp::create(turn, loc,
                  arith::ShRUIOp::create(turn, loc, state[maskAt], u), word(turn, loc, 1));
              Value paired = arith::CmpIOp::create(turn, loc, arith::CmpIPredicate::ne,
                                                   bit, word(turn, loc, 0));
              auto contribution = scf::IfOp::create(turn, loc, paired,
                  [&](OpBuilder &body, Location) {
                    SmallVector<Value> second(state.take_front(nj));
                    auto coordinates = emitTemplateWindowCoordinates(body, loc, unit.kind,
                        unit.first, second, state[shiftAt]);
                    auto result = evaluate(body, coordinates, unit.parameters,
                        state.slice(nj, partner.parameters.size()));
                    assert(result.firstForces.size() == fi && result.secondForces.size() == fj &&
                           result.sums.size() == unit.sumTypes.size());
                    SmallVector<Value> values(result.firstForces.begin(), result.firstForces.end());
                    llvm::append_range(values, result.secondForces);
                    llvm::append_range(values, result.sums);
                    scf::YieldOp::create(body, loc, values);
                  }, [&](OpBuilder &body, Location) {
                    SmallVector<Value> zeros;
                    for (Type type : resultTypes) zeros.push_back(createZero(body, loc, type));
                    scf::YieldOp::create(body, loc, zeros);
                  });
              ValueRange values(contribution.getResults());
              SmallVector<Value> next;
              for (Value value : state.take_front(maskAt + 1))
                next.push_back(shuffle(turn, loc, value, source, gpu::ShuffleMode::IDX));
              for (unsigned i = 0; i != fj; ++i) {
                Value accumulated = arith::AddFOp::create(turn, loc, state[fjAt + i], values[fi + i]);
                next.push_back(shuffle(turn, loc, accumulated, source, gpu::ShuffleMode::IDX));
              }
              for (unsigned i = 0; i != fi; ++i)
                next.push_back(arith::AddFOp::create(turn, loc, state[fiAt + i], values[i]));
              for (unsigned i = 0; i != unit.sumTypes.size(); ++i)
                next.push_back(arith::AddFOp::create(turn, loc, state[sumsAt + i], values[fi + fj + i]));
              scf::YieldOp::create(turn, loc, next);
            });
        ValueRange final(turns.getResults());
        if (unit.forces)
          scf::IfOp::create(batch, loc, has, [&](OpBuilder &body, Location) {
            // Sixteen rotations restore the original partner and its force.
            writeSecond(body, entry, final.slice(fjAt, fj));
            scf::YieldOp::create(body, loc);
          });
        scf::YieldOp::create(batch, loc, final.drop_front(fiAt));
      });
  ValueRange final(batches.getResults());
  if (unit.forces) {
    SmallVector<Value> combined;
    for (Value force : final.take_front(fi))
      combined.push_back(arith::AddFOp::create(builder, loc, force,
          shuffle(builder, loc, force, word(builder, loc, 16), gpu::ShuffleMode::XOR)));
    Value lowerHalf = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                            unit.lane, word(builder, loc, 16));
    Value writes = arith::AndIOp::create(builder, loc, lowerHalf, unit.firstValid);
    scf::IfOp::create(builder, loc, writes, [&](OpBuilder &body, Location) {
      writeFirst(body, combined);
      scf::YieldOp::create(body, loc);
    });
  }
  return SmallVector<Value, 2>(final.drop_front(fi));
}

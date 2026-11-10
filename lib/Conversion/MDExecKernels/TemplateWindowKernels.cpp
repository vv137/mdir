#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include <optional>

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

PreparedTemplateFrame mdir::kernels::emitPrepareTemplateFrame(
    OpBuilder &builder, Location loc, ArrayRef<Value> positions,
    Value cell, Value inverse, Type packedReal) {
  assert(positions.size() == 3);
  PreparedTemplateFrame result;
  result.whole[0] = positions[0];
  for (unsigned site = 1; site != 3; ++site) {
    Value raw = arith::SubFOp::create(builder, loc, positions[site], positions[0]);
    Value local = emitMinimumImage(builder, loc, raw, cell, inverse);
    result.whole[site] = arith::AddFOp::create(builder, loc, positions[0], local);
  }
  Value anchor = emitMinimumImage(builder, loc, positions[0], cell, inverse);
  result.latticeShift = arith::SubFOp::create(builder, loc, anchor, positions[0]);
  result.packed = emitPackTemplateFrame(builder, loc, result.whole,
                                        result.latticeShift, packedReal);
  return result;
}

Value mdir::kernels::emitTemplateFractional(OpBuilder &b, Location loc,
    Value position, Value cell, Value inverse) {
  if (!isTriclinic(cell))
    return arith::MulFOp::create(b, loc, position, inverse);
  auto component = [&](Value v, int64_t axis) -> Value {
    return vector::ExtractOp::create(b, loc, v, axis);
  };
  auto mul = [&](Value a, Value c) -> Value { return arith::MulFOp::create(b, loc, a, c); };
  auto sub = [&](Value a, Value c) -> Value { return arith::SubFOp::create(b, loc, a, c); };
  Value z = mul(component(position, 2), component(inverse, 2));
  Value y = mul(sub(component(position, 1), mul(component(cell, 5), z)), component(inverse, 1));
  Value x = mul(sub(sub(component(position, 0), mul(component(cell, 4), z)),
                    mul(component(cell, 3), y)), component(inverse, 0));
  return vector::FromElementsOp::create(b, loc, position.getType(), ValueRange{x, y, z});
}

Value mdir::kernels::emitTemplatePackingCells(OpBuilder &b, Location loc,
    Value anchor, Value cell, Value inverse) {
  if (!isTriclinic(cell))
    return arith::NegFOp::create(b, loc, math::RoundEvenOp::create(b, loc,
        arith::MulFOp::create(b, loc, anchor, inverse)));
  auto component = [&](Value v, int64_t axis) -> Value {
    return vector::ExtractOp::create(b, loc, v, axis);
  };
  Value raw[] = {component(anchor, 0), component(anchor, 1), component(anchor, 2)};
  Value shift[3];
  for (int axis = 2; axis >= 0; --axis) {
    Value count = math::RoundEvenOp::create(b, loc,
        arith::MulFOp::create(b, loc, raw[axis], component(inverse, axis)));
    shift[axis] = arith::NegFOp::create(b, loc, count);
    // Only the not-yet-reduced Cartesian components are needed to choose
    // the remaining lattice coordinates. The layout is ax,by,cz,bx,cx,cy.
    for (int row = 0; row < axis; ++row) {
      int column = axis == 2 ? 4 + row : 3;
      raw[row] = arith::SubFOp::create(b, loc, raw[row],
          arith::MulFOp::create(b, loc, count, component(cell, column)));
    }
  }
  return vector::FromElementsOp::create(b, loc, anchor.getType(),
                                        ValueRange{shift[0], shift[1], shift[2]});
}

PreparedTemplateFrame mdir::kernels::emitRetainedTemplateFrame(
    OpBuilder &b, Location loc, ArrayRef<Value> positions, Value cell,
    Value inverse, Value packingCells, Type packedReal) {
  assert(positions.size() == 3);
  PreparedTemplateFrame result;
  result.whole[0] = positions[0];
  for (unsigned site = 1; site != 3; ++site) {
    Value displacement = arith::SubFOp::create(b, loc, positions[site], positions[0]);
    result.whole[site] = arith::AddFOp::create(b, loc, positions[0],
        emitMinimumImage(b, loc, displacement, cell, inverse));
  }
  result.latticeShift = emitLatticeShift(b, loc, packingCells, cell);
  result.packed = emitPackTemplateFrame(b, loc, result.whole, result.latticeShift, packedReal);
  return result;
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

unsigned mdir::kernels::mapTemplateSiteOffsets(
    mdir::md_exec::TemplateForOp loop, unsigned tag, ValueRange sites,
    IRMapping &mapping) {
  assert(tag < 3 && sites.size() <= (tag == 0 ? 2u : tag == 1 ? 1u : 0u));
  Block &source = loop->getRegion(tag).front();
  unsigned bound = 0;
  for (auto [site, cached] : llvm::enumerate(sites)) {
    unsigned first = 1 + 2 * site;
    Value h1 = source.getArgument(first), h2 = source.getArgument(first + 1);
    auto type = cast<VectorType>(h1.getType());
    APFloat alpha = FloatAttr::get(type.getElementType(),
        loop.getAlpha().convertToDouble()).getValue();
    auto isWeight = [&](Value value) {
      std::optional<APFloat> literal;
      if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>()) {
        FloatAttr scalar;
        if (matchPattern(broadcast.getSource(), m_Constant(&scalar)))
          literal = scalar.getValue();
      } else {
        DenseFPElementsAttr dense;
        if (matchPattern(value, m_Constant(&dense)) && dense.isSplat())
          literal = dense.getSplatValue<APFloat>();
      }
      return literal && literal->bitwiseIsEqual(alpha);
    };
    for (Operation &op : source) {
      auto product = dyn_cast<arith::MulFOp>(&op);
      if (!product || product.getType() != cached.getType()) continue;
      for (unsigned side = 0; side != 2; ++side) {
        auto sum = product->getOperand(side).getDefiningOp<arith::AddFOp>();
        if (!sum || !isWeight(product->getOperand(1 - side))) continue;
        if ((sum.getLhs() == h1 && sum.getRhs() == h2) ||
            (sum.getLhs() == h2 && sum.getRhs() == h1)) {
          mapping.map(product.getResult(), cached);
          ++bound;
          break;
        }
      }
    }
  }
  return bound;
}

TemplateContribution mdir::kernels::emitTemplateWindowContributions(
    OpBuilder &builder, Location loc, mdir::md_exec::TemplateForOp loop,
    unsigned tag, IRMapping &mapping) {
  assert(tag < 3 && loop.getOuts().size() <= 1);
  constexpr unsigned first[] = {3, 1, 1};
  constexpr unsigned second[] = {3, 3, 1};
  unsigned arity = first[tag] + second[tag];
  unsigned outs = loop.getOuts().size();
  bool siteForces = outs && mdir::md::getTemplateForceArity(loop, arity, 0, outs) != arity;
  unsigned firstSlots = first[tag] + (siteForces && first[tag] == 3);
  unsigned secondSlots = second[tag] + (siteForces && second[tag] == 3);
  unsigned columns = outs + loop.getReduce().size();
  Block &source = loop->getRegion(tag).front();
  Operation *yield = source.getTerminator();
  auto zero = [&](Type type) -> Value {
    return arith::ConstantOp::create(builder, loc, builder.getZeroAttr(type));
  };
  TemplateContribution result;
  if (outs) {
    Type force = mdir::md_exec::getKernelValueType(loop.getOuts()[0].getType());
    result.firstForces.assign(firstSlots, zero(force));
    result.secondForces.assign(secondSlots, zero(force));
  }
  for (Value initial : loop.getReduce()) result.sums.push_back(zero(initial.getType()));
  unsigned offset = 0;
  for (unsigned column = 0; column != columns; ++column) {
    bool scatter = column < outs;
    unsigned stride = scatter ? mdir::md::getTemplateForceArity(loop, arity, column, outs) : 1;
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
            : member < firstSlots ? &result.firstForces[member]
                                 : &result.secondForces[member - firstSlots];
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
// Keep one anchor and reusable local geometry per unit. This has exactly
// the source window's subtraction/addition/multiplication order. The image
// shift affects the anchor displacement only, never these local offsets.
static SmallVector<Value, 4> prepareTemplateFrame(
    OpBuilder &builder, Location loc, ValueRange frame, bool hasSite, double alpha) {
  if (frame.size() == 1) return {frame.front()};
  assert(frame.size() == 3);
  SmallVector<Value, 4> prepared{frame[0],
      arith::SubFOp::create(builder, loc, frame[1], frame[0]),
      arith::SubFOp::create(builder, loc, frame[2], frame[0])};
  if (hasSite) {
    auto type = cast<VectorType>(frame[0].getType());
    Value weight = arith::ConstantOp::create(builder, loc,
        DenseElementsAttr::get(type, FloatAttr::get(type.getElementType(), alpha)));
    prepared.push_back(arith::MulFOp::create(builder, loc, weight,
        arith::AddFOp::create(builder, loc, prepared[1], prepared[2])));
  }
  return prepared;
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
                                            ValueRange, ValueRange, ValueRange)> evaluate,
    llvm::function_ref<void(OpBuilder &, ValueRange)> writeFirst,
    llvm::function_ref<void(OpBuilder &, Value, ValueRange)> writeSecond) {
  unsigned ni = unit.kind == TemplatePairClass::WaterWater ? 3 : 1;
  unsigned nj = unit.kind == TemplatePairClass::SoluteSolute ? 1 : 3;
  assert(unit.first.size() == ni);
  Type forceType = unit.first.front().getType();
  assert(cast<VectorType>(forceType).getElementType().isF32());
  auto firstFrame = prepareTemplateFrame(builder, loc, unit.first,
                                         unit.hasSite, unit.alpha);
  unsigned fi = unit.forces ? ni + (unit.siteForces && ni == 3) : 0;
  unsigned fj = unit.forces ? nj + (unit.siteForces && nj == 3) : 0;
  auto realForces = [&](OpBuilder &at, ValueRange values) {
    return mdir::md::emitTemplateForceTranspose(at, loc, values, unit.alpha);
  };
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
        auto secondFrame = prepareTemplateFrame(batch, loc, partner.frame,
                                                unit.hasSite, unit.alpha);
        unsigned frameSize = secondFrame.size();
        Value mask = arith::SelectOp::create(batch, loc, has, partner.mask, word(batch, loc, 0));
        unsigned shiftAt = frameSize + partner.parameters.size();
        unsigned maskAt = shiftAt + 1, fjAt = maskAt + 1, fiAt = fjAt + fj;
        unsigned sumsAt = fiAt + fi;
        SmallVector<Value> carried(secondFrame.begin(), secondFrame.end());
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
              paired = arith::AndIOp::create(turn, loc, paired, unit.firstValid);
              auto contribution = scf::IfOp::create(turn, loc, paired,
                  [&](OpBuilder &body, Location) {
                    Value shiftedAnchor = arith::AddFOp::create(body, loc,
                        state[0], state[shiftAt]);
                    SmallVector<Value> coordinates{arith::SubFOp::create(
                        body, loc, firstFrame[0], shiftedAnchor)};
                    SmallVector<Value> sites;
                    if (ni == 3) {
                      coordinates.append({firstFrame[1], firstFrame[2]});
                      if (unit.hasSite) sites.push_back(firstFrame[3]);
                    }
                    if (nj == 3) {
                      coordinates.append({state[1], state[2]});
                      if (unit.hasSite) sites.push_back(state[3]);
                    }
                    auto result = evaluate(body, coordinates, unit.parameters,
                        state.slice(frameSize, partner.parameters.size()), sites);
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
            writeSecond(body, entry, realForces(body, final.slice(fjAt, fj)));
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
      writeFirst(body, realForces(body, combined));
      scf::YieldOp::create(body, loc);
    });
  }
  return SmallVector<Value, 2>(final.drop_front(fi));
}

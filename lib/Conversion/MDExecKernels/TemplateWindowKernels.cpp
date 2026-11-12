#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
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
  // The nearest lattice point in fractional coordinates, in either cell:
  // the framed point then lies in [-1/2, 1/2] on the cell's own lattice
  // frame. In a triclinic cell the sequential Cartesian central image
  // (c, then b, then a) leaves the fractions of a and b spread over
  // about (-1/4, 5/4), so one group of units mixes lattice frames and
  // the search splits its entries by frame.
  return arith::NegFOp::create(b, loc, math::RoundEvenOp::create(b, loc,
      emitTemplateFractional(b, loc, anchor, cell, inverse)));
}

std::array<Value, 2> mdir::kernels::emitTemplateLocalGeometry(
    OpBuilder &b, Location loc, ArrayRef<Value> positions, Value cell,
    Value inverse, Value narrowCell, Value narrowInverse) {
  assert(positions.size() == 3);
  auto narrowType = [&](Value v) {
    return VectorType::get(cast<VectorType>(v.getType()).getShape(), b.getF32Type());
  };
  if (!narrowCell) {
    narrowCell = arith::TruncFOp::create(b, loc, narrowType(cell), cell);
    narrowInverse = arith::TruncFOp::create(b, loc, narrowType(inverse), inverse);
  }
  std::array<Value, 2> local;
  for (unsigned site = 1; site != 3; ++site) {
    // The difference is taken in the state's precision, then narrowed:
    // an O-H bond is a few tenths of a nanometer, so the f32 value has a
    // relative error of 2^-24 whatever the magnitude of the positions.
    Value wide = arith::SubFOp::create(b, loc, positions[site], positions[0]);
    Value raw = arith::TruncFOp::create(b, loc, narrowType(wide), wide);
    // A water split by a face of the cell is the rare case: its
    // displacement is a lattice vector away, and its minimum image is
    // found in the state's precision before narrowing.
    Value imaged = emitMinimumImage(b, loc, raw, narrowCell, narrowInverse);
    Value same = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND,
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, imaged, raw));
    auto chosen = scf::IfOp::create(b, loc, same, [&](OpBuilder &in, Location) {
      scf::YieldOp::create(in, loc, ValueRange{raw});
    }, [&](OpBuilder &in, Location) {
      Value image = emitMinimumImage(in, loc, wide, cell, inverse);
      scf::YieldOp::create(in, loc, ValueRange{
          arith::TruncFOp::create(in, loc, narrowType(image), image)});
    });
    local[site - 1] = chosen.getResult(0);
  }
  return local;
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
  if (!packedReal.isF32()) {
    result.packed = emitPackTemplateFrame(b, loc, result.whole, result.latticeShift, packedReal);
    return result;
  }
  // The anchor is shifted in the state's precision and narrowed; the
  // hydrogens are its narrowed image plus their f32 displacements, as the
  // per-step writer packs them (emitTemplateLocalGeometry).
  auto local = emitTemplateLocalGeometry(b, loc, positions, cell, inverse);
  Value anchor = arith::AddFOp::create(b, loc, positions[0], result.latticeShift);
  result.packed[0] = convertReal(b, loc, anchor, packedReal);
  for (unsigned site = 1; site != 3; ++site)
    result.packed[site] = arith::AddFOp::create(b, loc, result.packed[0], local[site - 1]);
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
// The unsigned bounds [lo, hi] of minui(maxui(x, lo), hi), constants.
static std::optional<std::pair<uint64_t, uint64_t>> unsignedClamp(Value value) {
  auto upper = value.getDefiningOp<arith::MinUIOp>();
  if (!upper) return std::nullopt;
  for (unsigned side = 0; side != 2; ++side) {
    APInt hi, lo;
    auto lower = upper->getOperand(1 - side).getDefiningOp<arith::MaxUIOp>();
    if (!lower || !matchPattern(upper->getOperand(side), m_ConstantInt(&hi))) continue;
    for (unsigned inner = 0; inner != 2; ++inner)
      if (matchPattern(lower->getOperand(inner), m_ConstantInt(&lo)))
        return std::make_pair(std::min(lo.getZExtValue(), hi.getZExtValue()),
                              hi.getZExtValue());
  }
  return std::nullopt;
}
// An index of a read that a clamp keeps in [0, extent - width]: a
// constant, min(max(x, lo), hi) with 0 <= lo and hi <= extent - width, or
// (minui(maxui(x, lo), hi) >> s) - b, the form of a radial table's index,
// with b <= lo >> s and (hi >> s) - b <= extent - width.
static bool clampedInto(Value index, int64_t extent, int64_t width) {
  if (auto cast = index.getDefiningOp<arith::IndexCastOp>()) index = cast.getIn();
  APInt value;
  if (matchPattern(index, m_ConstantInt(&value)))
    return value.getSExtValue() >= 0 && value.getSExtValue() + width <= extent;
  if (auto offset = index.getDefiningOp<arith::SubIOp>()) {
    APInt base, shift;
    auto key = offset.getLhs().getDefiningOp<arith::ShRUIOp>();
    if (!key || !matchPattern(offset.getRhs(), m_ConstantInt(&base)) ||
        !matchPattern(key.getRhs(), m_ConstantInt(&shift)) ||
        shift.getZExtValue() >= key.getType().getIntOrFloatBitWidth())
      return false;
    auto bounds = unsignedClamp(key.getLhs());
    if (!bounds || base.isNegative()) return false;
    uint64_t lo = bounds->first >> shift.getZExtValue();
    uint64_t hi = bounds->second >> shift.getZExtValue();
    uint64_t b = base.getZExtValue();
    return b <= lo && int64_t(hi - b) + width <= extent;
  }
  auto upper = index.getDefiningOp<arith::MinSIOp>();
  if (!upper) return false;
  for (unsigned side = 0; side != 2; ++side) {
    APInt hi, lo;
    auto lower = upper->getOperand(1 - side).getDefiningOp<arith::MaxSIOp>();
    if (!lower || !matchPattern(upper->getOperand(side), m_ConstantInt(&hi))) continue;
    for (unsigned inner = 0; inner != 2; ++inner)
      if (matchPattern(lower->getOperand(inner), m_ConstantInt(&lo)))
        return lo.getSExtValue() >= 0 && hi.getSExtValue() >= 0 &&
               hi.getSExtValue() + width <= extent;
  }
  return false;
}
// A read of a constant global table whose every index is clamped into its
// static shape cannot fault and sees no store.
static bool isBoundedTableRead(Operation *op) {
  Value base;
  ValueRange indices;
  ArrayRef<int64_t> widths;
  SmallVector<int64_t> ones;
  if (auto read = dyn_cast<vector::LoadOp>(op)) {
    base = read.getBase();
    indices = read.getIndices();
    auto type = read.getVectorType();
    ones.assign(indices.size() - type.getRank(), 1);
    llvm::append_range(ones, type.getShape());
  } else if (auto read = dyn_cast<memref::LoadOp>(op)) {
    base = read.getMemRef();
    indices = read.getIndices();
    ones.assign(indices.size(), 1);
  } else {
    return false;
  }
  widths = ones;
  auto global = base.getDefiningOp<memref::GetGlobalOp>();
  auto type = dyn_cast<MemRefType>(base.getType());
  if (!global || !type || !type.hasStaticShape() || type.getRank() != (int64_t)indices.size())
    return false;
  auto definition = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      global, global.getNameAttr());
  if (!definition || !definition.getConstant()) return false;
  for (auto [index, extent, width] : llvm::zip(indices, type.getShape(), widths))
    if (!clampedInto(index, extent, width)) return false;
  return true;
}
// Whether `value` of the window body is computed by pure operations and
// bounded table reads alone, so that it may be computed before a guard.
static bool isSpeculatable(Value value, Block &source, llvm::DenseSet<Value> &known) {
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != &source || known.contains(value)) return true;
  if (definition->getNumRegions() ||
      !(isPure(definition) || isBoundedTableRead(definition)))
    return false;
  for (Value operand : definition->getOperands())
    if (!isSpeculatable(operand, source, known)) return false;
  known.insert(value);
  return true;
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
    unsigned tag, IRMapping &mapping, ValueRange forces) {
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
  if (outs && !forces.empty()) {
    assert(forces.size() == firstSlots + secondSlots);
    result.firstForces.assign(forces.begin(), forces.begin() + firstSlots);
    result.secondForces.assign(forces.begin() + firstSlots, forces.end());
  } else if (outs) {
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
    auto liveMembers = [&](unsigned pair) {
      SmallVector<unsigned> members;
      for (unsigned member = 0; member != stride; ++member)
        if (!scatter || !matchPattern(
                yield->getOperand(offset + pair * stride + member), m_AnyZeroFloat()))
          members.push_back(member);
      return members;
    };
    // Force pairs go in an order where consecutive pairs add to different
    // sites where one exists, so that neighboring guards stay independent
    // and their table loads can overlap. Sums keep the source order.
    SmallVector<unsigned> order;
    {
      SmallVector<bool> done(pairs, false);
      SmallVector<unsigned> last;
      for (unsigned step = 0; step != pairs; ++step) {
        unsigned next = pairs;
        for (unsigned pair = 0; pair != pairs && scatter; ++pair) {
          if (done[pair]) continue;
          if (next == pairs) next = pair;
          auto members = liveMembers(pair);
          if (llvm::none_of(members, [&](unsigned m) { return llvm::is_contained(last, m); })) {
            next = pair;
            break;
          }
        }
        if (!scatter) next = step;
        done[next] = true;
        order.push_back(next);
        last = liveMembers(next);
      }
    }
    // The guarded bodies are predicated and issue anyway. The bounded
    // table reads of the next pair go before the guard of this one, so that
    // a read is in flight while the previous pair is evaluated instead of
    // waiting behind its guard; their values are those of the bodies.
    auto hoistReads = [&](unsigned pair) {
      if (!scatter) return;
      SmallVector<Operation *> reads;
      llvm::DenseSet<Operation *> seen;
      std::function<void(Value)> collect = [&](Value value) {
        Operation *definition = value.getDefiningOp();
        if (!definition || definition->getBlock() != &source ||
            !seen.insert(definition).second)
          return;
        for (Value operand : definition->getOperands()) collect(operand);
        if (isBoundedTableRead(definition)) reads.push_back(definition);
      };
      for (unsigned member : liveMembers(pair))
        collect(yield->getOperand(offset + pair * stride + member));
      llvm::DenseSet<Value> known;
      for (Operation *read : reads)
        if (isSpeculatable(read->getResult(0), source, known))
          materializeWindowValue(builder, read->getResult(0), source, mapping);
    };
    if (!order.empty()) hoistReads(order.front());
    for (auto [position, pair] : llvm::enumerate(order)) {
      if (position + 1 < order.size()) hoistReads(order[position + 1]);
      Value displacement = materializeWindowValue(builder,
          yield->getOperand(offset + stride * pairs + pair), source, mapping);
      Value square = arith::MulFOp::create(builder, loc, displacement, displacement);
      Value r2 = vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, square);
      double cutoff = loop.getCutoff().convertToDouble();
      Value cutoff2 = arith::ConstantOp::create(builder, loc,
          FloatAttr::get(r2.getType(), cutoff * cutoff));
      Value inside = arith::CmpFOp::create(builder, loc,
          arith::CmpFPredicate::OLT, r2, cutoff2);
      auto destination = [&](unsigned member) -> Value & {
        return !scatter ? result.sums[column - outs]
            : member < firstSlots ? result.firstForces[member]
                                 : result.secondForces[member - firstSlots];
      };
      // A force member accumulates into its destination inside the guard:
      // the taken branch adds the pair's term, the other keeps the sum. A
      // member that is constant zero in the source adds nothing and is not
      // carried. Reductions keep their zero-started per-pair terms.
      SmallVector<unsigned> members = liveMembers(pair);
      if (members.empty()) continue;
      // The next-to-last force pair of a water pair, when it reads a table
      // and is otherwise pure, is computed in every lane and kept by a
      // select instead of a guard. ptxas branched around this pair alone of
      // those it if-converted and moved its hoisted table read below the
      // branch, where the read waited behind the comparison (D[rigid-water-
      // template], U9). Other pairs keep their guards: making them selects
      // took the kernel above 96 registers, and making the first pair one
      // (the oxygen pair with Lennard-Jones) left it at 80 and slower.
      bool predicated = false;
      if (scatter && tag == 0 && order.size() >= 3 && position + 2 == order.size()) {
        llvm::DenseSet<Value> known;
        bool reads = false;
        llvm::DenseSet<Operation *> seen;
        std::function<void(Value)> find = [&](Value value) {
          Operation *definition = value.getDefiningOp();
          if (!definition || definition->getBlock() != &source || !seen.insert(definition).second)
            return;
          reads |= isBoundedTableRead(definition);
          for (Value operand : definition->getOperands()) find(operand);
        };
        predicated = llvm::all_of(members, [&](unsigned member) {
          Value term = yield->getOperand(offset + pair * stride + member);
          find(term);
          return isSpeculatable(term, source, known);
        }) && reads;
      }
      if (predicated) {
        for (unsigned member : members) {
          Value term = materializeWindowValue(builder,
              yield->getOperand(offset + pair * stride + member), source, mapping);
          Value sum = arith::AddFOp::create(builder, loc, destination(member), term);
          destination(member) = arith::SelectOp::create(builder, loc, inside, sum,
                                                        destination(member));
        }
        continue;
      }
      auto guarded = scf::IfOp::create(builder, loc, inside,
          [&](OpBuilder &body, Location) {
            IRMapping local = mapping;
            SmallVector<Value> values;
            for (unsigned member : members) {
              Value term = materializeWindowValue(body,
                  yield->getOperand(offset + pair * stride + member), source, local);
              values.push_back(scatter
                  ? arith::AddFOp::create(body, loc, destination(member), term) : term);
            }
            scf::YieldOp::create(body, loc, values);
          }, [&](OpBuilder &body, Location) {
            SmallVector<Value> values;
            for (unsigned member : members) {
              Type type = yield->getOperand(offset + pair * stride + member).getType();
              values.push_back(scatter ? destination(member)
                  : arith::ConstantOp::create(body, loc, body.getZeroAttr(type)));
            }
            scf::YieldOp::create(body, loc, values);
          });
      for (auto [value, member] : llvm::zip(guarded.getResults(), members))
        destination(member) = scatter ? Value(value)
            : arith::AddFOp::create(builder, loc, destination(member), value);
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
static Value formSiteOffset(OpBuilder &builder, Location loc, Value first,
                            Value second, double alpha) {
  auto type = cast<VectorType>(first.getType());
  Value weight = arith::ConstantOp::create(builder, loc,
      DenseElementsAttr::get(type, FloatAttr::get(type.getElementType(), alpha)));
  return arith::MulFOp::create(builder, loc, weight,
      arith::AddFOp::create(builder, loc, first, second));
}
static SmallVector<Value, 4> prepareTemplateFrame(
    OpBuilder &builder, Location loc, ValueRange frame, bool hasSite, double alpha) {
  if (frame.size() == 1) return {frame.front()};
  assert(frame.size() == 3);
  SmallVector<Value, 4> prepared{frame[0],
      arith::SubFOp::create(builder, loc, frame[1], frame[0]),
      arith::SubFOp::create(builder, loc, frame[2], frame[0])};
  if (hasSite)
    prepared.push_back(formSiteOffset(builder, loc, prepared[1], prepared[2], alpha));
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
        ValueRange, ValueRange, ValueRange, ValueRange)> evaluate,
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
  // A row group of `width` units takes `width` lanes; the warp holds
  // 32 / width such lane groups, each rotating its own entries through
  // `width` turns. WW rows use MDRT_TEMPLATE_WATER_GROUP (16 or 8), the
  // solute rows 16. The caller's first unit is lane & (width - 1).
  const int64_t width = unit.kind == TemplatePairClass::WaterWater
      ? MDRT_TEMPLATE_WATER_GROUP : 16;
  Value u = arith::AndIOp::create(builder, loc, unit.lane, word(builder, loc, width - 1));
  Value half = arith::AndIOp::create(builder, loc, unit.lane, word(builder, loc, 32 - width));
  Value previous = arith::AndIOp::create(builder, loc,
      arith::AddIOp::create(builder, loc, u, word(builder, loc, width - 1)),
      word(builder, loc, width - 1));
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
        // The rotated state is the smallest set that generates the partner
        // frame: the image shift is the same for all sixteen turns, so it is
        // added to the anchor once (the same addition the turn made), and the
        // formed site is recomputed from the two offsets each turn rather
        // than rotated. Both keep the source window's rounding exactly.
        auto secondFrame = prepareTemplateFrame(batch, loc, partner.frame,
                                                /*hasSite=*/false, unit.alpha);
        secondFrame[0] = arith::AddFOp::create(batch, loc, secondFrame[0], partner.shift);
        unsigned frameSize = secondFrame.size();
        Value mask = arith::SelectOp::create(batch, loc, has, partner.mask, word(batch, loc, 0));
        unsigned maskAt = frameSize + partner.parameters.size();
        unsigned fjAt = maskAt + 1, fiAt = fjAt + fj;
        unsigned sumsAt = fiAt + fi;
        SmallVector<Value> carried(secondFrame.begin(), secondFrame.end());
        llvm::append_range(carried, partner.parameters);
        carried.push_back(mask);
        for (unsigned i = 0; i != fj; ++i) carried.push_back(createZero(batch, loc, forceType));
        llvm::append_range(carried, own);
        auto turns = scf::ForOp::create(batch, loc, word(batch, loc, 0),
            word(batch, loc, width), word(batch, loc, 1), carried,
            [&](OpBuilder &turn, Location, Value, ValueRange state) {
              Value bit = arith::AndIOp::create(turn, loc,
                  arith::ShRUIOp::create(turn, loc, state[maskAt], u), word(turn, loc, 1));
              Value paired = arith::CmpIOp::create(turn, loc, arith::CmpIPredicate::ne,
                                                   bit, word(turn, loc, 0));
              paired = arith::AndIOp::create(turn, loc, paired, unit.firstValid);
              auto contribution = scf::IfOp::create(turn, loc, paired,
                  [&](OpBuilder &body, Location) {
                    SmallVector<Value> coordinates{arith::SubFOp::create(
                        body, loc, firstFrame[0], state[0])};
                    SmallVector<Value> sites;
                    if (ni == 3) {
                      coordinates.append({firstFrame[1], firstFrame[2]});
                      if (unit.hasSite) sites.push_back(firstFrame[3]);
                    }
                    if (nj == 3) {
                      coordinates.append({state[1], state[2]});
                      if (unit.hasSite)
                        sites.push_back(formSiteOffset(body, loc, state[1], state[2], unit.alpha));
                    }
                    SmallVector<Value> forces(state.slice(fiAt, fi));
                    llvm::append_range(forces, state.slice(fjAt, fj));
                    auto result = evaluate(body, coordinates, unit.parameters,
                        state.slice(frameSize, partner.parameters.size()), sites, forces);
                    assert(result.firstForces.size() == fi && result.secondForces.size() == fj &&
                           result.sums.size() == unit.sumTypes.size());
                    SmallVector<Value> values(result.firstForces.begin(), result.firstForces.end());
                    llvm::append_range(values, result.secondForces);
                    llvm::append_range(values, result.sums);
                    scf::YieldOp::create(body, loc, values);
                  }, [&](OpBuilder &body, Location) {
                    // An unpaired turn keeps the carried forces as they are.
                    SmallVector<Value> values(state.slice(fiAt, fi));
                    llvm::append_range(values, state.slice(fjAt, fj));
                    for (Type type : unit.sumTypes) values.push_back(createZero(body, loc, type));
                    scf::YieldOp::create(body, loc, values);
                  });
              ValueRange values(contribution.getResults());
              SmallVector<Value> next;
              for (Value value : state.take_front(maskAt + 1))
                next.push_back(shuffle(turn, loc, value, source, gpu::ShuffleMode::IDX));
              for (unsigned i = 0; i != fj; ++i)
                next.push_back(shuffle(turn, loc, values[fi + i], source, gpu::ShuffleMode::IDX));
              for (unsigned i = 0; i != fi; ++i) next.push_back(values[i]);
              for (unsigned i = 0; i != unit.sumTypes.size(); ++i)
                next.push_back(arith::AddFOp::create(turn, loc, state[sumsAt + i], values[fi + fj + i]));
              scf::YieldOp::create(turn, loc, next);
            });
        ValueRange final(turns.getResults());
        if (unit.forces)
          scf::IfOp::create(batch, loc, has, [&](OpBuilder &body, Location) {
            // `width` rotations restore the original partner and its force.
            writeSecond(body, partner.id ? partner.id : entry,
                        realForces(body, final.slice(fjAt, fj)));
            scf::YieldOp::create(body, loc);
          });
        scf::YieldOp::create(batch, loc, final.drop_front(fiAt));
      });
  ValueRange final(batches.getResults());
  if (unit.forces) {
    SmallVector<Value> combined;
    // The lane groups of the warp hold the same units: sum them by
    // butterfly, 16 then 8 (for width 8), and write from the first.
    for (Value force : final.take_front(fi)) {
      for (int64_t step = 16; step >= width; step /= 2)
        force = arith::AddFOp::create(builder, loc, force,
            shuffle(builder, loc, force, word(builder, loc, step), gpu::ShuffleMode::XOR));
      combined.push_back(force);
    }
    Value lowerHalf = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                            unit.lane, word(builder, loc, width));
    Value writes = arith::AndIOp::create(builder, loc, lowerHalf, unit.firstValid);
    scf::IfOp::create(builder, loc, writes, [&](OpBuilder &body, Location) {
      writeFirst(body, realForces(body, combined));
      scf::YieldOp::create(body, loc);
    });
  }
  return SmallVector<Value, 2>(final.drop_front(fi));
}

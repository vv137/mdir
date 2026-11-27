// The serial definition of md_alg.traverse_pairs over the segments of a
// list of units (D417, Sections 4.3 and 6): the oracle of the
// SIMT schedule of the segments. It is written in the order of the
// definition:
//
//   for each segment s(f, g) that the op visits,
//     for each unit of work of its rows (each active one of an inner list),
//       for each entry of its block, and each of the 16 places of its group
//       whose bit is set: the pair (i, j) of a unit of f and one of g;
//         the anchor displacement d_0, and the offsets c of the places;
//         for each site pair (p, q) of the table: d = (d_0 + c_p) - c_q,
//         r^2, its cutoff, and its slice of the region;
//         the destinations of the places, then W^T to the members.
//
// The places of a unit are formed from its members by the weights W of its
// kind (rule R2 of D403): x_p = x_0 + sum_k W_pk (x_k - x_0). The precision
// changes where the meaning says: a member's position is moved into the
// frame of its group in the type it is stored in, and converted to the type
// of the region; the offsets x_k - x_0 are formed in the stored type and
// converted; everything after, in the type of the region. The buffer form
// takes the members of a unit stored whole about its anchor.

#include "mdir/Conversion/TraversePairs.h"

#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MDRT/SegmentRegions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

namespace {

/// One unit of a pair: its kind, its place, and the values the definition
/// forms for it.
struct Unit {
  unsigned kind = 0;
  Value place;
  /// The position of the anchor in the frame of its group, in the type of
  /// the region (for the partner, with the entry's shift).
  Value anchor;
  /// The offsets h_k = x_k - x_0 of the members, k >= 1, in that type.
  SmallVector<Value> offsets;
  /// The particle of each member, and whether the place holds a unit.
  SmallVector<Value> members;
  Value holds;
  /// The values of each field of `ins` at each member.
  SmallVector<SmallVector<Value>> fields;
};

struct Segments {
  OpBuilder &b;
  md_alg::TraversePairsOp op;
  const SerialLists &lists;
  SerialOptions options;
  Location loc;
  mdrt::SegmentedPairsType type;
  unsigned numOuts, numValues;
  Type computed, real;
  Value box;

  Segments(OpBuilder &b, md_alg::TraversePairsOp op, const SerialLists &lists,
           SerialOptions options)
      : b(b), op(op), lists(lists), options(options), loc(op.getLoc()) {
    type = cast<mdrt::SegmentedPairsType>(op.getListType());
    numOuts = op.getOuts().size();
    numValues = numOuts + op.getReduce().size();
    computed = op.getSegmentKernels()[0].front().getArgument(0).getType();
    real = cast<MemRefType>(lists.positions.getType()).getElementType();
    box = lists.cell;
  }

  Value i32(OpBuilder &at, int64_t v) {
    return arith::ConstantOp::create(at, loc, at.getI32Type(),
                                     at.getI32IntegerAttr(v));
  }
  Value index(OpBuilder &at, int64_t v) { return createIndex(at, loc, v); }
  Value toIndex(OpBuilder &at, Value v) {
    return arith::IndexCastOp::create(at, loc, at.getIndexType(), v);
  }
  Value load(OpBuilder &at, Value buffer, Value i) {
    return memref::LoadOp::create(at, loc, buffer, ValueRange{i});
  }
  scf::IfOp branch(OpBuilder &at, TypeRange types, Value condition) {
    auto choice = scf::IfOp::create(at, loc, types, condition,
                                    /*withElseRegion=*/true);
    for (Block *block : {choice.thenBlock(), choice.elseBlock()})
      if (!block->empty())
        block->getTerminator()->erase();
    return choice;
  }
  scf::ForOp loop(OpBuilder &at, Value first, Value last, ValueRange initial,
                  function_ref<SmallVector<Value>(OpBuilder &, Value,
                                                  ValueRange)>
                      body) {
    Value one = index(at, 1);
    return scf::ForOp::create(
        at, loc, first, last, one, initial,
        [&](OpBuilder &in, Location, Value k, ValueRange carried) {
          Value step = k;
          if (options.reverse)
            step = arith::SubIOp::create(
                in, loc, arith::AddIOp::create(in, loc, first, last),
                arith::AddIOp::create(in, loc, k, one));
          scf::YieldOp::create(in, loc, body(in, step, carried));
        });
  }

  /// The shift of a place, ten bits an axis from -512, in the stored type.
  Value placeShift(OpBuilder &at, Value place) {
    Value packed = load(at, lists.groups.shifts, place);
    SmallVector<Value> cells;
    for (int64_t k = 0; k != 3; ++k) {
      Value bits = arith::AndIOp::create(
          at, loc, arith::ShRUIOp::create(at, loc, packed, i32(at, 10 * k)),
          i32(at, 1023));
      cells.push_back(arith::SIToFPOp::create(
          at, loc, real, arith::SubIOp::create(at, loc, bits, i32(at, 512))));
    }
    return emitLatticeShift(
        at, loc,
        vector::FromElementsOp::create(at, loc, VectorType::get({3}, real),
                                       cells),
        convertReal(at, loc, box, real));
  }

  /// The unit of kind `kind` at `place`: its members, the anchor in the
  /// frame of its group, the offsets, and the fields. An empty place takes
  /// the particle 0, as the groups' definition does.
  Unit unitAt(OpBuilder &at, unsigned kind, Value place) {
    Unit unit;
    unit.kind = kind;
    unit.place = place;
    Value held = load(at, lists.groups.order, place);
    unit.holds = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::sge,
                                       held, i32(at, 0));
    Value slot = arith::SelectOp::create(at, loc, unit.holds, held, i32(at, 0));
    unsigned m = type.getNumMembers(kind);
    if (!type.getKind(kind)) {
      unit.members.push_back(toIndex(at, slot));
    } else {
      // The place holds the particle of the copy's first place, and the
      // places of a copy are consecutive particles (D446).
      for (int64_t p : type.getKind(kind).getFrame().asArrayRef())
        unit.members.push_back(
            toIndex(at, arith::AddIOp::create(at, loc, slot, i32(at, p))));
    }
    Value x0 = loadElement(at, loc, lists.positions, unit.members[0]);
    unit.anchor = convertReal(
        at, loc, arith::AddFOp::create(at, loc, x0, placeShift(at, place)),
        computed);
    for (unsigned k = 1; k != m; ++k) {
      Value xk = loadElement(at, loc, lists.positions, unit.members[k]);
      unit.offsets.push_back(
          convertReal(at, loc, arith::SubFOp::create(at, loc, xk, x0),
                      computed));
    }
    for (Value buffer : lists.ins) {
      SmallVector<Value> values;
      for (Value member : unit.members)
        values.push_back(loadElement(at, loc, buffer, member));
      unit.fields.push_back(values);
    }
    return unit;
  }

  /// The offset c_p of place `p` of `unit` from its anchor, or null for
  /// the anchor itself: the frame atoms' offsets of equal weight are summed
  /// before they are scaled, then the terms are added in increasing k.
  Value placeOffset(OpBuilder &at, const Unit &unit, unsigned p) {
    SmallVector<double> weights = type.getPlaceWeights(unit.kind, p);
    // A frame atom's place is its member.
    for (unsigned k = 0, e = weights.size(); k != e; ++k)
      if (weights[k] == 1.0 && llvm::count(weights, 0.0) == e - 1)
        return k ? unit.offsets[k - 1] : Value();
    Value total;
    SmallVector<bool> done(weights.size(), false);
    for (unsigned k = 1, e = weights.size(); k != e; ++k) {
      if (done[k] || weights[k] == 0.0)
        continue;
      Value sum = unit.offsets[k - 1];
      for (unsigned l = k + 1; l != e; ++l)
        if (!done[l] && weights[l] == weights[k]) {
          sum = arith::AddFOp::create(at, loc, sum, unit.offsets[l - 1]);
          done[l] = true;
        }
      Type element = getElementTypeOrSelf(computed);
      Value weight = vector::BroadcastOp::create(
          at, loc, sum.getType(),
          arith::ConstantOp::create(at, loc, element,
                                    at.getFloatAttr(element, weights[k])));
      Value term = arith::MulFOp::create(at, loc, weight, sum);
      total = total ? Value(arith::AddFOp::create(at, loc, total, term)) : term;
    }
    return total;
  }

  /// Adds `value` to destination `out` at `particle` where `holds`.
  void addAt(OpBuilder &at, unsigned out, Value particle, Value holds,
             Value value) {
    scf::IfOp::create(at, loc, holds, [&](OpBuilder &then, Location) {
      Value destination = lists.outs[out];
      Type stored = getElementTypeOrSelf(destination.getType());
      Value old = loadElement(then, loc, destination, particle);
      Value sum = arith::AddFOp::create(then, loc, old,
                                        convertReal(then, loc, value, stored));
      storeElement(then, loc, sum, destination, particle);
      scf::YieldOp::create(then, loc);
    });
  }

  /// The pair (i, j) of segment `index`: its site pairs, and the
  /// destinations of its places, which reach the members by W^T. Returns
  /// the sums.
  SmallVector<Value> pair(OpBuilder &at, unsigned index, const Unit &i,
                          const Unit &j, ValueRange sums) {
    Block &kernel = op.getSegmentKernels()[index].front();
    Operation *yield = kernel.getTerminator();
    SmallVector<mdrt::SitePair> table =
        mdrt::getSitePairs(op.getSitesAttr(), index);
    unsigned M = table.size();
    unsigned ni = type.getNumPlaces(i.kind), nj = type.getNumPlaces(j.kind);
    Value d0 = arith::SubFOp::create(at, loc, i.anchor, j.anchor);
    llvm::DenseMap<unsigned, Value> ci, cj;
    for (const mdrt::SitePair &sites : table) {
      if (!ci.count(sites.p))
        ci[sites.p] = placeOffset(at, i, sites.p);
      if (!cj.count(sites.q))
        cj[sites.q] = placeOffset(at, j, sites.q);
    }
    // The accumulators of the places, for each destination: those of i,
    // then those of j. They start from zero and take each contribution.
    SmallVector<Type> outTypes;
    for (unsigned a = 0; a != numOuts; ++a) {
      auto buffer = cast<MemRefType>(lists.outs[a].getType());
      outTypes.push_back(buffer.getRank() == 1
                             ? buffer.getElementType()
                             : VectorType::get({buffer.getDimSize(1)},
                                               buffer.getElementType()));
    }
    // Kept in the type of the yields: the region's values for a destination.
    SmallVector<Type> accumulatorTypes;
    SmallVector<Value> carried;
    unsigned operand = 0;
    SmallVector<Type> valueTypes(numValues);
    for (const mdrt::SitePair &sites : table)
      for (unsigned value : sites.values)
        valueTypes[value] = yield->getOperand(operand++).getType();
    for (unsigned a = 0; a != numOuts; ++a) {
      Type t = valueTypes[a] ? valueTypes[a] : outTypes[a];
      for (unsigned p = 0; p != ni + nj; ++p)
        carried.push_back(createZero(at, loc, t));
    }
    llvm::append_range(carried, sums);
    auto accumulator = [&](unsigned a, bool second, unsigned place) {
      return a * (ni + nj) + (second ? ni : 0) + place;
    };
    unsigned numAccumulators = numOuts * (ni + nj);

    unsigned begin = 0;
    for (unsigned m = 0; m != M; ++m) {
      const mdrt::SitePair &sites = table[m];
      unsigned end = begin + sites.values.size();
      Value d = d0;
      if (Value c = ci.lookup(sites.p))
        d = arith::AddFOp::create(at, loc, d, c);
      if (Value c = cj.lookup(sites.q))
        d = arith::SubFOp::create(at, loc, d, c);
      Value squares = arith::MulFOp::create(at, loc, d, d);
      Value r2 = vector::ReductionOp::create(
          at, loc, vector::CombiningKind::ADD, squares);
      Value within = arith::CmpFOp::create(
          at, loc, arith::CmpFPredicate::OLT, r2,
          createCutoff2(at, loc, computed, op.getCutoff().convertToDouble()));
      auto test = branch(at, ValueRange(carried).getTypes(), within);
      OpBuilder then = OpBuilder::atBlockEnd(test.thenBlock());
      IRMapping inside;
      for (auto [from, to] : lists.captures)
        inside.map(from, to);
      inside.map(kernel.getArgument(2 * m), r2);
      inside.map(kernel.getArgument(2 * m + 1), d);
      unsigned members = i.members.size() + j.members.size();
      for (unsigned field = 0, e = lists.ins.size(); field != e; ++field) {
        unsigned first = 2 * M + field * members;
        for (unsigned k = 0; k != i.members.size(); ++k)
          inside.map(kernel.getArgument(first + k), i.fields[field][k]);
        for (unsigned k = 0; k != j.members.size(); ++k)
          inside.map(kernel.getArgument(first + i.members.size() + k),
                     j.fields[field][k]);
      }
      for (Operation *nested : mdrt::getSiteSlice(kernel, m, begin, end))
        then.clone(*nested, inside);
      SmallVector<Value> next(carried);
      for (unsigned n = begin; n != end; ++n) {
        unsigned value = sites.values[n - begin];
        Value k = inside.lookupOrDefault(yield->getOperand(n));
        if (options.magnitudes)
          k = math::AbsFOp::create(then, loc, k);
        if (options.omitted)
          k = createZero(then, loc, k.getType());
        if (value < numOuts) {
          unsigned mine = accumulator(value, false, sites.p);
          unsigned theirs = accumulator(value, true, sites.q);
          next[mine] = arith::AddFOp::create(then, loc, next[mine], k);
          bool antisymmetric =
              op.getExchangeOf(value) == "antisymmetric" && !options.magnitudes;
          next[theirs] = arith::AddFOp::create(
              then, loc, next[theirs],
              antisymmetric ? Value(arith::NegFOp::create(then, loc, k)) : k);
          continue;
        }
        unsigned sum = numAccumulators + value - numOuts;
        next[sum] = arith::AddFOp::create(then, loc, next[sum], k);
      }
      scf::YieldOp::create(then, loc, next);
      OpBuilder otherwise = OpBuilder::atBlockEnd(test.elseBlock());
      scf::YieldOp::create(otherwise, loc, carried);
      carried.assign(test.getResults().begin(), test.getResults().end());
      begin = end;
    }

    // W^T: the force of each place reaches the members of its unit with
    // the weights of its kind, in increasing place (rule A).
    for (unsigned a = 0; a != numOuts; ++a)
      for (bool second : {false, true}) {
        const Unit &unit = second ? j : i;
        unsigned n = second ? nj : ni;
        for (unsigned k = 0, e = unit.members.size(); k != e; ++k) {
          Value total;
          for (unsigned p = 0; p != n; ++p) {
            double weight = type.getPlaceWeights(unit.kind, p)[k];
            if (weight == 0.0)
              continue;
            Value force = carried[accumulator(a, second, p)];
            if (weight != 1.0) {
              if (options.magnitudes)
                weight = std::fabs(weight);
              Type element = getElementTypeOrSelf(force.getType());
              Value w = arith::ConstantOp::create(
                  at, loc, element, at.getFloatAttr(element, weight));
              if (isa<VectorType>(force.getType()))
                w = vector::BroadcastOp::create(at, loc, force.getType(), w);
              force = arith::MulFOp::create(at, loc, w, force);
            }
            total = total ? Value(arith::AddFOp::create(at, loc, total, force))
                          : force;
          }
          if (total)
            addAt(at, a, unit.members[k], unit.holds, total);
        }
      }
    return SmallVector<Value>(ValueRange(carried).drop_front(numAccumulators));
  }

  SmallVector<Value> run() {
    // A destination that the op overwrites starts from zero.
    for (auto [i, destination] : llvm::enumerate(lists.outs)) {
      if (!op.overwrites(i))
        continue;
      auto buffer = cast<MemRefType>(destination.getType());
      Type element = buffer.getElementType();
      Type value = buffer.getRank() == 1
                       ? element
                       : VectorType::get({buffer.getDimSize(1)}, element);
      Value zero = createZero(b, loc, value);
      scf::ForOp::create(
          b, loc, index(b, 0),
          memref::DimOp::create(b, loc, destination, index(b, 0)),
          index(b, 1), ValueRange(),
          [&](OpBuilder &in, Location, Value particle, ValueRange) {
            storeElement(in, loc, zero, destination, particle);
            scf::YieldOp::create(in, loc);
          });
    }
    // The sums start from zero, in the types of the yields.
    SmallVector<Value> sums;
    for (unsigned s = numOuts; s != numValues; ++s)
      sums.push_back(createZero(b, loc, op.getReduce()[s - numOuts].getType()));
    Value zero = index(b, 0);
    int64_t width = getEntryShiftBits(isTriclinic(box));
    int64_t offset = getEntryShiftOffset(isTriclinic(box));
    Value entries = lists.dual ? lists.inner.entries : lists.groups.entries;
    Value masks = lists.dual ? lists.inner.masks : lists.groups.masks;
    for (unsigned regionIndex = 0, e = op.getSegmentKernels().size();
         regionIndex != e; ++regionIndex) {
      auto [f, g] = mdrt::getSegmentKinds(op.getSegmentsAttr(), regionIndex);
      unsigned s = mdrt::SegmentedPairsType::getSegmentIndex(f, g);
      Value first = toIndex(b, load(b, lists.segments, index(b, s)));
      Value last = toIndex(b, load(b, lists.segments, index(b, s + 1)));
      // The units of work of the segment; of an inner list, its active
      // units that lie in the segment's range.
      Value count = lists.dual ? toIndex(b, load(b, lists.activeCount, zero))
                               : last;
      Value start = lists.dual ? zero : first;
      auto overUnits = loop(
          b, start, count, sums,
          [&](OpBuilder &u, Value step, ValueRange sums) {
            Value unit = step;
            Value inSegment = arith::ConstantOp::create(
                u, loc, u.getI1Type(), u.getBoolAttr(true));
            if (lists.dual) {
              unit = toIndex(u, load(u, lists.active, step));
              inSegment = arith::AndIOp::create(
                  u, loc,
                  arith::CmpIOp::create(u, loc, arith::CmpIPredicate::uge,
                                        unit, first),
                  arith::CmpIOp::create(u, loc, arith::CmpIPredicate::ult,
                                        unit, last));
            }
            auto inRange = branch(u, ValueRange(sums).getTypes(), inSegment);
            OpBuilder r = OpBuilder::atBlockEnd(inRange.thenBlock());
            Value group = load(r, lists.groups.units, unit);
            Value begin, end;
            if (lists.dual) {
              begin = index(r, 0);
              end = toIndex(r, load(r, lists.inner.counts, unit));
            } else {
              // Over segments a unit of work holds its own count: a group
              // is a row of several segments.
              begin = index(r, 0);
              end = toIndex(r, load(r, lists.groups.counts, unit));
            }
            Value base = arith::MulIOp::create(r, loc, unit, index(r, 64));
            auto overEntries = loop(
                r, begin, end, sums,
                [&](OpBuilder &en, Value entry, ValueRange sums) {
                  Value at = arith::AddIOp::create(
                      en, loc, arith::SubIOp::create(en, loc, entry, begin),
                      base);
                  Value other = toIndex(en, load(en, entries, at));
                  Value mask = load(en, masks, at);
                  Unit partner = unitAt(en, g, other);
                  // The entry's shift, by the cells of its mask (D95,
                  // D115), in the type of the region.
                  Type element = getElementTypeOrSelf(computed);
                  SmallVector<Value> cells;
                  for (int64_t axis = 0; axis != 3; ++axis) {
                    Value bits = arith::AndIOp::create(
                        en, loc,
                        arith::ShRUIOp::create(en, loc, mask,
                                               i32(en, 16 + width * axis)),
                        i32(en, (1 << width) - 1));
                    cells.push_back(arith::SIToFPOp::create(
                        en, loc, element,
                        arith::SubIOp::create(en, loc, bits, i32(en, offset))));
                  }
                  partner.anchor = arith::AddFOp::create(
                      en, loc, partner.anchor,
                      emitLatticeShift(
                          en, loc,
                          vector::FromElementsOp::create(
                              en, loc, partner.anchor.getType(), cells),
                          convertReal(en, loc, box, element)));
                  auto overPlaces = loop(
                      en, index(en, 0), index(en, 16), sums,
                      [&](OpBuilder &p, Value u16, ValueRange sums) {
                        Value u32 = arith::IndexCastOp::create(
                            p, loc, p.getI32Type(), u16);
                        Value bit = arith::AndIOp::create(
                            p, loc, arith::ShRUIOp::create(p, loc, mask, u32),
                            i32(p, 1));
                        Value set = arith::CmpIOp::create(
                            p, loc, arith::CmpIPredicate::ne, bit, i32(p, 0));
                        // A bit of an empty place stands for no pair.
                        Value held = load(
                            p, lists.groups.order,
                            toIndex(p, arith::AddIOp::create(
                                           p, loc,
                                           arith::MulIOp::create(
                                               p, loc, group, i32(p, 16)),
                                           u32)));
                        set = arith::AndIOp::create(
                            p, loc, set,
                            arith::CmpIOp::create(p, loc,
                                                  arith::CmpIPredicate::sge,
                                                  held, i32(p, 0)));
                        auto paired =
                            branch(p, ValueRange(sums).getTypes(), set);
                        OpBuilder then =
                            OpBuilder::atBlockEnd(paired.thenBlock());
                        Value mine = toIndex(
                            then,
                            arith::AddIOp::create(
                                then, loc,
                                arith::MulIOp::create(then, loc, group,
                                                      i32(then, 16)),
                                u32));
                        Unit unit = unitAt(then, f, mine);
                        scf::YieldOp::create(
                            then, loc,
                            pair(then, regionIndex, unit, partner, sums));
                        OpBuilder otherwise =
                            OpBuilder::atBlockEnd(paired.elseBlock());
                        scf::YieldOp::create(otherwise, loc, sums);
                        return SmallVector<Value>(paired.getResults());
                      });
                  return SmallVector<Value>(overPlaces.getResults());
                });
            scf::YieldOp::create(r, loc, overEntries.getResults());
            OpBuilder skip = OpBuilder::atBlockEnd(inRange.elseBlock());
            scf::YieldOp::create(skip, loc, sums);
            return SmallVector<Value>(inRange.getResults());
          });
      sums.assign(overUnits.getResults().begin(), overUnits.getResults().end());
    }

    // A weight is that of the sum over both orders of each pair: each pair
    // once counts twice.
    SmallVector<Value> results;
    for (unsigned s = numOuts; s != numValues; ++s) {
      Value total = sums[s - numOuts];
      Type outside = op.getReduce()[s - numOuts].getType();
      double weight = 2.0;
      if (auto weights = op.getWeights())
        weight *= (*weights)[s - numOuts];
      Type type = total.getType();
      FloatAttr scalar = b.getFloatAttr(getElementTypeOrSelf(type), weight);
      Value factor;
      if (auto vector = dyn_cast<VectorType>(type))
        factor = arith::ConstantOp::create(
            b, loc, type, DenseElementsAttr::get(vector, scalar.getValue()));
      else
        factor = arith::ConstantOp::create(b, loc, type, scalar);
      total = arith::MulFOp::create(b, loc, factor, total);
      (void)outside;
      results.push_back(
          arith::AddFOp::create(b, loc, op.getReduce()[s - numOuts], total));
    }
    return results;
  }
};
} // namespace

SmallVector<Value> kernels::emitTraverseSegmentsSerial(
    OpBuilder &builder, md_alg::TraversePairsOp op, const SerialLists &lists,
    SerialOptions options) {
  return Segments(builder, op, lists, options).run();
}

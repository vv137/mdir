// The SIMT schedule of md_alg.traverse_pairs over the segments of a list of
// units (D[md-alg-segments-simt], D417, Section 6); see
// include/mdir/Conversion/TraversePairs.h. It is a second schedule, chosen
// at compile time by the type of the list: the schedule of the groups (D415,
// TraversePairs.cpp) is not touched, so a program of groups keeps its bytes.
//
// Its mechanisms are those of the warp-cooperative schedule of #795, made
// general over the kinds of the type (rules R1 to R5 of D417):
//
//   - a warp takes one record of one segment s(f, g); lanes u and u + 16 hold
//     the unit at place 16 g + u of the record's group, of kind f;
//   - the entries are taken 32 at a time, one a lane; each partner, of kind
//     g, is loaded once, and its anchor (with the entry's shift), the offsets
//     of its members, the fields of its members that a slice reads, its mask,
//     and the accumulators of its places turn through the half-warp in
//     `rotation` shuffle steps (the rotation of Laine et al. 2013's
//     warp-wide work distribution, as #795 measured it);
//   - the offsets c_p of the places are formed from the kind's weights W
//     (R2): once for the lane's unit, and at each step from the partner's
//     rotating offsets, so that a site is formed in the kernel and never
//     loaded;
//   - each site pair (p, q) of the segment's table has a guard r^2 < r_c^2
//     and a clone of its slice of the region alone; a pair that the table
//     does not hold emits nothing;
//   - a partner's place accumulators reach its members by W^T once per
//     entry, after its rotation, one atomic addition a member; the lane's
//     unit is written once per record, its two half-warp copies added first
//     (rule A).
//
// The frames are packed before the loop, a thread a place: the anchor moved
// by the place's shift in the stored type and narrowed, and the offsets of
// the members from the anchor formed in the stored type and narrowed (D95;
// D417, Section 4.3), with the fields of the members that the slices read.
// S4a's writer is to pack them in the loop that writes the positions.

#include "mdir/Conversion/TraversePairs.h"

#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mdir/Dialect/MDRT/SegmentRegions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

namespace {

/// What one region of the op reads and yields: its kinds, its table, and the
/// arguments of the fields that its slices read.
struct Region {
  unsigned index = 0, f = 0, g = 0, s = 0;
  unsigned mf = 1, mg = 1, nf = 1, ng = 1;
  SmallVector<mdrt::SitePair> table;
  Block *kernel = nullptr;
  /// For each field of `ins` and each member (those of f, then those of g),
  /// whether a slice reads it.
  SmallVector<SmallVector<bool>> reads;
  /// The places of f and of g that the table names.
  SmallVector<unsigned> placesF, placesG;
  /// The type of each value (destinations, then sums) that the region
  /// yields, or null for a value that it does not yield.
  SmallVector<Type> valueTypes;
  /// The yields that are taken before their widening.
  llvm::SmallDenseSet<unsigned> narrowed;
};

struct Schedule {
  md_alg::TraversePairsOp op;
  TraverseHost &host;
  const SegmentLists &lists;
  int64_t blockSize, rotation;
  Location loc;
  mdrt::SegmentedPairsType type;
  unsigned K = 0, S = 0, numOuts = 0, numValues = 0, maxMembers = 1;
  Type f32, i32;
  std::vector<Region> regions;
  /// The fields of `ins` that some slice reads.
  SmallVector<bool> fieldRead;
  /// The packed frames and fields, a row a place.
  Value anchors, offsets;
  SmallVector<Value> fields;

  Schedule(md_alg::TraversePairsOp op, TraverseHost &host,
           const SegmentLists &lists, int64_t blockSize, int64_t rotation)
      : op(op), host(host), lists(lists), blockSize(blockSize),
        rotation(rotation), loc(op.getLoc()) {}

  Value ci(OpBuilder &b, int64_t v) {
    return arith::ConstantOp::create(b, loc, b.getI32Type(),
                                     b.getI32IntegerAttr(v));
  }
  Value ix(OpBuilder &b, int64_t v) { return createIndex(b, loc, v); }
  /// An scf.if with results of `types` and both branches empty: the
  /// callers yield in each.
  scf::IfOp branch(OpBuilder &b, TypeRange types, Value condition) {
    auto choice = scf::IfOp::create(b, loc, types, condition,
                                    /*withElseRegion=*/true);
    for (Block *block : {choice.thenBlock(), choice.elseBlock()})
      if (!block->empty())
        block->getTerminator()->erase();
    return choice;
  }
  Value toIndex(OpBuilder &b, Value v) {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
  }
  Value load(OpBuilder &b, Value buffer, Value i) {
    if (!i.getType().isIndex())
      i = toIndex(b, i);
    return memref::LoadOp::create(b, loc, buffer, ValueRange{i});
  }

  /// A shuffle of every component of `value` from lane `source`.
  Value shuffle(OpBuilder &b, Value value, Value source, gpu::ShuffleMode mode) {
    auto one = [&](Value x) -> Value {
      return gpu::ShuffleOp::create(b, loc, x, source, ci(b, 32), mode)
          .getShuffleResult();
    };
    if (auto vector = dyn_cast<VectorType>(value.getType())) {
      SmallVector<Value> parts;
      for (int64_t c = 0, e = vector.getNumElements(); c != e; ++c)
        parts.push_back(one(vector::ExtractOp::create(b, loc, value, c)));
      return vector::FromElementsOp::create(b, loc, vector, parts);
    }
    return one(value);
  }

  /// The shift of a place, ten bits an axis from -512, in `real`.
  Value placeShift(OpBuilder &b, Value packed, Value box, Type real) {
    SmallVector<Value> cells;
    for (int64_t k = 0; k != 3; ++k) {
      Value bits = arith::AndIOp::create(
          b, loc, arith::ShRUIOp::create(b, loc, packed, ci(b, 10 * k)),
          ci(b, 1023));
      cells.push_back(arith::SIToFPOp::create(
          b, loc, real, arith::SubIOp::create(b, loc, bits, ci(b, 512))));
    }
    return emitLatticeShift(
        b, loc,
        vector::FromElementsOp::create(b, loc, VectorType::get({3}, real),
                                       cells),
        convertReal(b, loc, box, real));
  }

  /// The offset c_p of place `p` of a unit of kind `kind` from its anchor,
  /// from the offsets h_k of its members (k >= 1), or null for the anchor:
  /// the offsets of equal weight are summed before they are scaled, then
  /// the terms are added in increasing k (D417, Section 4.3, step 3).
  Value placeOffset(OpBuilder &b, unsigned kind, unsigned p,
                    ArrayRef<Value> h) {
    SmallVector<double> weights = type.getPlaceWeights(kind, p);
    for (unsigned k = 0, e = weights.size(); k != e; ++k)
      if (weights[k] == 1.0 && llvm::count(weights, 0.0) == e - 1)
        return k ? h[k - 1] : Value();
    Value total;
    SmallVector<bool> done(weights.size(), false);
    for (unsigned k = 1, e = weights.size(); k != e; ++k) {
      if (done[k] || weights[k] == 0.0)
        continue;
      Value sum = h[k - 1];
      for (unsigned l = k + 1; l != e; ++l)
        if (!done[l] && weights[l] == weights[k]) {
          sum = arith::AddFOp::create(b, loc, sum, h[l - 1]);
          done[l] = true;
        }
      Value weight = vector::BroadcastOp::create(
          b, loc, sum.getType(),
          arith::ConstantOp::create(b, loc, f32,
                                    b.getFloatAttr(f32, weights[k])));
      Value term = arith::MulFOp::create(b, loc, weight, sum);
      total = total ? Value(arith::AddFOp::create(b, loc, total, term)) : term;
    }
    return total;
  }

  /// W^T: the accumulators `forces` of the places of a unit of kind `kind`
  /// (by place, null where none) to each of its members, in increasing
  /// place (rule A); null for a member that no place reaches.
  SmallVector<Value> transpose(OpBuilder &b, unsigned kind,
                               ArrayRef<Value> forces) {
    unsigned m = type.getNumMembers(kind), n = type.getNumPlaces(kind);
    SmallVector<Value> members;
    for (unsigned k = 0; k != m; ++k) {
      Value total;
      for (unsigned p = 0; p != n; ++p) {
        if (!forces[p])
          continue;
        double weight = type.getPlaceWeights(kind, p)[k];
        if (weight == 0.0)
          continue;
        Value force = forces[p];
        if (weight != 1.0) {
          Type element = getElementTypeOrSelf(force.getType());
          Value w = arith::ConstantOp::create(b, loc, element,
                                              b.getFloatAttr(element, weight));
          if (isa<VectorType>(force.getType()))
            w = vector::BroadcastOp::create(b, loc, force.getType(), w);
          force = arith::MulFOp::create(b, loc, w, force);
        }
        total = total ? Value(arith::AddFOp::create(b, loc, total, force))
                      : force;
      }
      members.push_back(total);
    }
    return members;
  }

  /// The particle of member `k` of the unit that a place holds (`held`, the
  /// slot of its first member or the particle of the rest).
  Value memberParticle(OpBuilder &b, unsigned kind, Value held, unsigned k) {
    if (kind == K)
      return toIndex(b, held);
    return toIndex(b, load(b, lists.members,
                           arith::AddIOp::create(b, loc, held, ci(b, k))));
  }

  void addToMembers(OpBuilder &b, unsigned kind, Value held, unsigned out,
                    ArrayRef<Value> values) {
    // Every member's particle is read before the first atomic, so that the
    // loads overlap instead of each waiting behind the atomics before it.
    SmallVector<Value> particles;
    for (unsigned k = 0, e = values.size(); k != e; ++k)
      particles.push_back(values[k] ? memberParticle(b, kind, held, k)
                                    : Value());
    Value destination = op.getOuts()[out];
    Type stored = getElementTypeOrSelf(destination.getType());
    for (auto [value, particle] : llvm::zip(values, particles))
      if (value)
        emitAtomicAdd(b, loc, convertReal(b, loc, value, stored), destination,
                      particle);
  }

  LogicalResult analyze();
  void emitPack(OpBuilder &b, Value box, Value room);
  SmallVector<Value> emitRecord(OpBuilder &b, const Region &region,
                                Value record, Value lane, Value box32);
  LogicalResult run();
};

LogicalResult Schedule::analyze() {
  type = cast<mdrt::SegmentedPairsType>(op.getListType());
  K = type.getNumKinds();
  S = type.getNumSegments();
  numOuts = op.getOuts().size();
  numValues = numOuts + op.getReduce().size();
  MLIRContext *context = op.getContext();
  f32 = Float32Type::get(context);
  i32 = IntegerType::get(context, 32);
  for (unsigned f = 0; f <= K; ++f)
    maxMembers = std::max(maxMembers, type.getNumMembers(f));
  fieldRead.assign(op.getIns().size(), false);
  for (unsigned r = 0, e = op.getSegmentKernels().size(); r != e; ++r) {
    Region region;
    region.index = r;
    std::tie(region.f, region.g) =
        mdrt::getSegmentKinds(op.getSegmentsAttr(), r);
    region.s = mdrt::SegmentedPairsType::getSegmentIndex(region.f, region.g);
    region.mf = type.getNumMembers(region.f);
    region.mg = type.getNumMembers(region.g);
    region.nf = type.getNumPlaces(region.f);
    region.ng = type.getNumPlaces(region.g);
    region.table = mdrt::getSitePairs(op.getSitesAttr(), r);
    region.kernel = &op.getSegmentKernels()[r].front();
    unsigned M = region.table.size();
    Type real = region.kernel->getArgument(0).getType();
    if (!real.isF32())
      return op.emitOpError()
             << "computes its site pairs of segment (" << region.f << ", "
             << region.g << ") in " << real
             << "; the SIMT schedule of a list of units takes f32 (mixed "
                "precision), and the serial definition any type";
    unsigned members = region.mf + region.mg;
    for (unsigned j = 0, ej = op.getIns().size(); j != ej; ++j) {
      SmallVector<bool> reads;
      Type element = getElementTypeOrSelf(op.getIns()[j].getType());
      for (unsigned k = 0; k != members; ++k) {
        bool used = !region.kernel->getArgument(2 * M + j * members + k)
                         .use_empty();
        if (used && element.getIntOrFloatBitWidth() != 32)
          return op.emitOpError()
                 << "reads a field of " << element
                 << " in segment (" << region.f << ", " << region.g
                 << "); the SIMT schedule of a list of units turns fields of "
                    "32 bits";
        reads.push_back(used);
        fieldRead[j] = fieldRead[j] || used;
      }
      region.reads.push_back(reads);
    }
    llvm::SetVector<unsigned> placesF, placesG;
    for (const mdrt::SitePair &pair : region.table) {
      placesF.insert(pair.p);
      placesG.insert(pair.q);
    }
    region.placesF.assign(placesF.begin(), placesF.end());
    region.placesG.assign(placesG.begin(), placesG.end());
    region.valueTypes.assign(numValues, Type());
    Operation *yield = region.kernel->getTerminator();
    // A sum that every slice computes in a narrower type and widens to
    // yield it is summed in that type, and widened once (D415's rule for
    // the groups).
    SmallVector<bool> narrow(numValues, true);
    unsigned operand = 0;
    for (const mdrt::SitePair &pair : region.table)
      for (unsigned value : pair.values) {
        Value yielded = yield->getOperand(operand++);
        auto widen = yielded.getDefiningOp<arith::ExtFOp>();
        Type type = widen && widen->getBlock() == region.kernel
                        ? widen.getIn().getType()
                        : Type();
        if (value < numOuts || !type ||
            (region.valueTypes[value] && region.valueTypes[value] != type))
          narrow[value] = false;
        region.valueTypes[value] = type ? type : yielded.getType();
      }
    operand = 0;
    for (const mdrt::SitePair &pair : region.table)
      for (unsigned value : pair.values) {
        Value yielded = yield->getOperand(operand++);
        if (!narrow[value]) {
          region.valueTypes[value] = yielded.getType();
          continue;
        }
        region.narrowed.insert(operand - 1);
      }
    regions.push_back(std::move(region));
  }
  return success();
}

void Schedule::emitPack(OpBuilder &b, Value box, Value room) {
  Value positions = op.getPositions();
  auto stored = cast<MemRefType>(positions.getType());
  Attribute space = stored.getMemorySpace();
  auto rows = [&](Type element, int64_t width) {
    SmallVector<int64_t> shape = {ShapedType::kDynamic};
    if (width)
      shape.push_back(width);
    return MemRefType::get(shape, element, MemRefLayoutAttrInterface(), space);
  };
  anchors = host.createDeviceBuffer(b, loc, rows(f32, 3), ValueRange{room});
  if (maxMembers > 1)
    offsets = host.createDeviceBuffer(
        b, loc, rows(f32, 3),
        ValueRange{arith::MulIOp::create(b, loc, room,
                                         ix(b, maxMembers - 1))});
  for (auto [j, in] : llvm::enumerate(op.getIns())) {
    if (!fieldRead[j]) {
      fields.push_back(Value());
      continue;
    }
    fields.push_back(host.createDeviceBuffer(
        b, loc, rows(getElementTypeOrSelf(in.getType()), 0),
        ValueRange{arith::MulIOp::create(b, loc, room, ix(b, maxMembers))}));
  }
  Type real = stored.getElementType();
  Value places = arith::IndexCastOp::create(
      b, loc, b.getIndexType(),
      memref::LoadOp::create(b, loc, host.getSizes(), ValueRange{ix(b, 0)}));
  host.launchOver(b, loc, places, [&](OpBuilder &body, Value place) {
    Value held = load(body, lists.order, place);
    Value holds = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::sge,
                                        held, ci(body, 0));
    scf::IfOp::create(body, loc, holds, [&](OpBuilder &in, Location) {
      // The rank of the place's group, from the places where the ranks
      // begin: the rest rank 0, kind f rank K - f.
      Value place32 = arith::IndexCastOp::create(in, loc, i32, place);
      Value rank = ci(in, 0);
      for (unsigned r = 1; r <= K; ++r) {
        Value past = arith::CmpIOp::create(
            in, loc, arith::CmpIPredicate::sge, place32,
            load(in, lists.rankPlaces, ix(in, r)));
        rank = arith::SelectOp::create(in, loc, past, ci(in, r), rank);
      }
      Value shift = placeShift(in, load(in, lists.shifts, place), box, real);
      for (unsigned kind = 0; kind <= K; ++kind) {
        Value is = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq,
                                         rank, ci(in, K - kind));
        scf::IfOp::create(in, loc, is, [&](OpBuilder &at, Location) {
          unsigned m = type.getNumMembers(kind);
          SmallVector<Value> particles;
          for (unsigned k = 0; k != m; ++k)
            particles.push_back(memberParticle(at, kind, held, k));
          Value x0 = loadElement(at, loc, positions, particles[0]);
          storeElement(at, loc,
                       convertReal(at, loc,
                                   arith::AddFOp::create(at, loc, x0, shift),
                                   f32),
                       anchors, place);
          for (unsigned k = 1; k < m; ++k) {
            Value xk = loadElement(at, loc, positions, particles[k]);
            Value row = arith::AddIOp::create(
                at, loc,
                arith::MulIOp::create(at, loc, place,
                                      ix(at, maxMembers - 1)),
                ix(at, k - 1));
            storeElement(at, loc,
                         convertReal(at, loc,
                                     arith::SubFOp::create(at, loc, xk, x0),
                                     f32),
                         offsets, row);
          }
          for (auto [j, in] : llvm::enumerate(op.getIns())) {
            if (!fields[j])
              continue;
            for (unsigned k = 0; k != m; ++k) {
              Value row = arith::AddIOp::create(
                  at, loc,
                  arith::MulIOp::create(at, loc, place, ix(at, maxMembers)),
                  ix(at, k));
              storeElement(at, loc, loadElement(at, loc, in, particles[k]),
                           fields[j], row);
            }
          }
          scf::YieldOp::create(at, loc);
        });
      }
      scf::YieldOp::create(in, loc);
    });
  });
}

SmallVector<Value> Schedule::emitRecord(OpBuilder &b, const Region &region,
                                        Value record, Value lane, Value box32) {
  bool triclinic = isTriclinic(box32);
  int64_t width = getEntryShiftBits(triclinic);
  int64_t shiftOffset = getEntryShiftOffset(triclinic);
  unsigned M = region.table.size();
  unsigned members = region.mf + region.mg;
  unsigned numFields = op.getIns().size();
  Operation *yield = region.kernel->getTerminator();
  Value cutoff2 = createCutoff2(b, loc, f32, op.getCutoff().convertToDouble());

  Value group = load(b, lists.groups, record);
  Value count = load(b, lists.dual ? lists.inner.counts : lists.counts, record);
  Value entries = lists.dual ? lists.inner.entries : lists.entries;
  Value masks = lists.dual ? lists.inner.masks : lists.masks;

  // The unit of this lane: place 16 g + u, of kind f.
  Value u = arith::AndIOp::create(b, loc, lane, ci(b, 15));
  Value half = arith::AndIOp::create(b, loc, lane, ci(b, 16));
  Value mine = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, group, ci(b, 16)), u);
  Value heldI = load(b, lists.order, mine);
  Value validI = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                       heldI, ci(b, 0));
  auto loadOffsets = [&](OpBuilder &at, Value place, unsigned m) {
    SmallVector<Value> h;
    for (unsigned k = 1; k < m; ++k)
      h.push_back(loadElement(
          at, loc, offsets,
          toIndex(at, arith::AddIOp::create(
                          at, loc,
                          arith::MulIOp::create(at, loc, place,
                                                ci(at, maxMembers - 1)),
                          ci(at, k - 1)))));
    return h;
  };
  // The fields that a slice reads, of the members of a unit: `first` for
  // those of f.
  auto loadFields = [&](OpBuilder &at, Value place, bool first) {
    SmallVector<Value> values;
    unsigned m = first ? region.mf : region.mg;
    for (unsigned j = 0; j != numFields; ++j)
      for (unsigned k = 0; k != m; ++k)
        if (region.reads[j][first ? k : region.mf + k])
          values.push_back(loadElement(
              at, loc, fields[j],
              toIndex(at, arith::AddIOp::create(
                              at, loc,
                              arith::MulIOp::create(at, loc, place,
                                                    ci(at, maxMembers)),
                              ci(at, k)))));
    return values;
  };
  Value anchorI = loadElement(b, loc, anchors, toIndex(b, mine));
  SmallVector<Value> hI = loadOffsets(b, mine, region.mf);
  SmallVector<Value> fieldsI = loadFields(b, mine, true);
  llvm::DenseMap<unsigned, Value> cI;
  for (unsigned p : region.placesF)
    cI[p] = placeOffset(b, region.f, p, hI);

  // The accumulators: for each destination the region yields, one for each
  // place of the table, of the lane's unit and of the partner.
  struct Slot {
    unsigned out, place;
  };
  SmallVector<Slot> slotsF, slotsG;
  for (unsigned a = 0; a != numOuts; ++a) {
    if (!region.valueTypes[a])
      continue;
    for (unsigned p : region.placesF)
      slotsF.push_back({a, p});
    for (unsigned q : region.placesG)
      slotsG.push_back({a, q});
  }
  auto slotOf = [](ArrayRef<Slot> slots, unsigned a, unsigned place) {
    for (auto [i, slot] : llvm::enumerate(slots))
      if (slot.out == a && slot.place == place)
        return unsigned(i);
    llvm_unreachable("a slot of the table");
  };
  SmallVector<unsigned> sumsYielded;
  for (unsigned v = numOuts; v != numValues; ++v)
    if (region.valueTypes[v])
      sumsYielded.push_back(v);

  SmallVector<Value> initial;
  for (const Slot &slot : slotsF)
    initial.push_back(createZero(b, loc, region.valueTypes[slot.out]));
  for (unsigned v : sumsYielded)
    initial.push_back(createZero(b, loc, region.valueTypes[v]));
  unsigned numF = slotsF.size(), numG = slotsG.size();

  Value previous = arith::AndIOp::create(
      b, loc, arith::AddIOp::create(b, loc, u, ci(b, rotation - 1)),
      ci(b, rotation - 1));
  Value source = arith::OrIOp::create(b, loc, half, previous);
  Value base = arith::MulIOp::create(b, loc, toIndex(b, record), ix(b, 64));

  auto batches = scf::ForOp::create(
      b, loc, ci(b, 0), count, ci(b, 32), initial,
      [&](OpBuilder &batch, Location, Value begin, ValueRange carried) {
        Value e = arith::AddIOp::create(batch, loc, begin, lane);
        Value has = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::slt,
                                          e, count);
        // A partial last batch loads a valid entry; its cleared mask takes
        // part in every shuffle, and its partner is not written.
        Value safe = arith::SelectOp::create(batch, loc, has, e, begin);
        Value at = arith::AddIOp::create(batch, loc, base, toIndex(batch, safe));
        Value partner = load(batch, entries, at);
        Value mask = arith::SelectOp::create(batch, loc, has,
                                             load(batch, masks, at),
                                             ci(batch, 0));
        Value heldJ = load(batch, lists.order, partner);
        SmallVector<Value> cells;
        for (int64_t axis = 0; axis != 3; ++axis) {
          Value bits = arith::AndIOp::create(
              batch, loc,
              arith::ShRUIOp::create(batch, loc, mask,
                                     ci(batch, 16 + width * axis)),
              ci(batch, (1 << width) - 1));
          cells.push_back(arith::SIToFPOp::create(
              batch, loc, f32,
              arith::SubIOp::create(batch, loc, bits,
                                    ci(batch, shiftOffset))));
        }
        Value anchorJ = arith::AddFOp::create(
            batch, loc, loadElement(batch, loc, anchors, toIndex(batch, partner)),
            emitLatticeShift(batch, loc,
                             vector::FromElementsOp::create(
                                 batch, loc, VectorType::get({3}, f32), cells),
                             box32));
        // The turning state: the anchor, the offsets, the fields, the mask,
        // and the partner's accumulators.
        SmallVector<Value> state = {anchorJ};
        llvm::append_range(state, loadOffsets(batch, partner, region.mg));
        llvm::append_range(state, loadFields(batch, partner, false));
        unsigned maskAt = state.size();
        state.push_back(mask);
        unsigned accAt = state.size();
        for (const Slot &slot : slotsG)
          state.push_back(createZero(batch, loc, region.valueTypes[slot.out]));
        unsigned turning = state.size();
        llvm::append_range(state, carried);
        unsigned numH = region.mg - 1;
        unsigned numFieldsJ = maskAt - 1 - numH;

        auto turns = scf::ForOp::create(
            batch, loc, ci(batch, 0), ci(batch, rotation), ci(batch, 1), state,
            [&](OpBuilder &turn, Location, Value, ValueRange now) {
              Value bit = arith::AndIOp::create(
                  turn, loc, arith::ShRUIOp::create(turn, loc, now[maskAt], u),
                  ci(turn, 1));
              Value paired = arith::AndIOp::create(
                  turn, loc,
                  arith::CmpIOp::create(turn, loc, arith::CmpIPredicate::ne,
                                        bit, ci(turn, 0)),
                  validI);
              // What the pair changes: the partner's accumulators, then the
              // lane's, then the sums.
              SmallVector<Value> changed(now.slice(accAt, numG));
              llvm::append_range(changed, now.drop_front(turning));
              auto pairing =
                  branch(turn, ValueRange(changed).getTypes(), paired);
              {
                OpBuilder then = OpBuilder::atBlockEnd(pairing.thenBlock());
                SmallVector<Value> values(changed);
                Value d0 = arith::SubFOp::create(then, loc, anchorI, now[0]);
                SmallVector<Value> hJ(now.slice(1, numH));
                llvm::DenseMap<unsigned, Value> cJ;
                for (unsigned q : region.placesG)
                  cJ[q] = placeOffset(then, region.g, q, hJ);
                ValueRange fieldsJ = now.slice(1 + numH, numFieldsJ);
                unsigned begin = 0;
                for (unsigned m = 0; m != M; ++m) {
                  const mdrt::SitePair &sites = region.table[m];
                  unsigned end = begin + sites.values.size();
                  Value d = d0;
                  if (Value c = cI.lookup(sites.p))
                    d = arith::AddFOp::create(then, loc, d, c);
                  if (Value c = cJ.lookup(sites.q))
                    d = arith::SubFOp::create(then, loc, d, c);
                  Value r2 = vector::ReductionOp::create(
                      then, loc, vector::CombiningKind::ADD,
                      arith::MulFOp::create(then, loc, d, d));
                  Value within = arith::CmpFOp::create(
                      then, loc, arith::CmpFPredicate::OLT, r2, cutoff2);
                  // The accumulators that the slice adds to.
                  SmallVector<unsigned> touched;
                  for (unsigned value : sites.values) {
                    if (value < numOuts) {
                      touched.push_back(slotOf(slotsG, value, sites.q));
                      touched.push_back(numG + slotOf(slotsF, value, sites.p));
                    } else {
                      touched.push_back(
                          numG + numF +
                          unsigned(llvm::find(sumsYielded, value) -
                                   sumsYielded.begin()));
                    }
                  }
                  SmallVector<Value> before;
                  for (unsigned t : touched)
                    before.push_back(values[t]);
                  auto guard =
                      branch(then, ValueRange(before).getTypes(), within);
                  {
                    OpBuilder in = OpBuilder::atBlockEnd(guard.thenBlock());
                    IRMapping mapping;
                    mapping.map(region.kernel->getArgument(2 * m), r2);
                    mapping.map(region.kernel->getArgument(2 * m + 1), d);
                    unsigned iField = 0, jField = 0;
                    for (unsigned j = 0; j != numFields; ++j)
                      for (unsigned k = 0; k != members; ++k) {
                        if (!region.reads[j][k])
                          continue;
                        Value value = k < region.mf ? fieldsI[iField++]
                                                    : fieldsJ[jField++];
                        mapping.map(
                            region.kernel->getArgument(2 * M + j * members + k),
                            value);
                      }
                    for (Operation *nested :
                         mdrt::getSiteSlice(*region.kernel, m, begin, end))
                      in.clone(*nested, mapping);
                    SmallVector<Value> after(before);
                    unsigned t = 0;
                    for (unsigned n = begin; n != end; ++n) {
                      unsigned value = sites.values[n - begin];
                      Value yielded = yield->getOperand(n);
                      if (region.narrowed.contains(n))
                        yielded =
                            yielded.getDefiningOp<arith::ExtFOp>().getIn();
                      Value k = mapping.lookupOrDefault(yielded);
                      if (value < numOuts) {
                        bool antisymmetric =
                            op.getExchangeOf(value) == "antisymmetric";
                        after[t] =
                            antisymmetric
                                ? arith::SubFOp::create(in, loc, after[t], k)
                                      .getResult()
                                : arith::AddFOp::create(in, loc, after[t], k)
                                      .getResult();
                        after[t + 1] =
                            arith::AddFOp::create(in, loc, after[t + 1], k);
                        t += 2;
                        continue;
                      }
                      after[t] = arith::AddFOp::create(in, loc, after[t], k);
                      ++t;
                    }
                    scf::YieldOp::create(in, loc, after);
                    OpBuilder out = OpBuilder::atBlockEnd(guard.elseBlock());
                    scf::YieldOp::create(out, loc, before);
                  }
                  for (auto [t, result] :
                       llvm::zip(touched, guard.getResults()))
                    values[t] = result;
                  begin = end;
                }
                scf::YieldOp::create(then, loc, values);
                OpBuilder otherwise = OpBuilder::atBlockEnd(pairing.elseBlock());
                scf::YieldOp::create(otherwise, loc, changed);
              }
              ValueRange results = pairing.getResults();
              SmallVector<Value> next;
              for (unsigned i = 0; i != accAt; ++i)
                next.push_back(
                    shuffle(turn, now[i], source, gpu::ShuffleMode::IDX));
              for (unsigned i = 0; i != numG; ++i)
                next.push_back(
                    shuffle(turn, results[i], source, gpu::ShuffleMode::IDX));
              llvm::append_range(next, results.drop_front(numG));
              scf::YieldOp::create(turn, loc, next);
            });
        ValueRange done = turns.getResults();
        // After the rotation each partner is back at its lane: its places'
        // values reach its members by W^T.
        if (numG)
          scf::IfOp::create(batch, loc, has, [&](OpBuilder &then, Location) {
            for (unsigned a = 0; a != numOuts; ++a) {
              if (!region.valueTypes[a])
                continue;
              SmallVector<Value> forces(region.ng, Value());
              for (unsigned q : region.placesG)
                forces[q] = done[accAt + slotOf(slotsG, a, q)];
              addToMembers(then, region.g, heldJ, a,
                           transpose(then, region.g, forces));
            }
            scf::YieldOp::create(then, loc);
          });
        scf::YieldOp::create(batch, loc,
                             ValueRange(done.drop_front(turning)));
      });

  // The two half-warps hold the same unit: they add up, and the first
  // writes it.
  ValueRange totals = batches.getResults();
  if (numF) {
    SmallVector<Value> combined;
    for (unsigned i = 0; i != numF; ++i)
      combined.push_back(arith::AddFOp::create(
          b, loc, totals[i],
          shuffle(b, totals[i], ci(b, 16), gpu::ShuffleMode::XOR)));
    Value writes = arith::AndIOp::create(
        b, loc,
        arith::AndIOp::create(
            b, loc,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, half,
                                  ci(b, 0)),
            validI),
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, count,
                              ci(b, 0)));
    scf::IfOp::create(b, loc, writes, [&](OpBuilder &then, Location) {
      for (unsigned a = 0; a != numOuts; ++a) {
        if (!region.valueTypes[a])
          continue;
        SmallVector<Value> forces(region.nf, Value());
        for (unsigned p : region.placesF)
          forces[p] = combined[slotOf(slotsF, a, p)];
        addToMembers(then, region.f, heldI, a,
                     transpose(then, region.f, forces));
      }
      scf::YieldOp::create(then, loc);
    });
  }

  // The sums: a weight is that of the sum over both orders of each pair, so
  // each pair once counts twice; a sum the region does not yield is zero.
  SmallVector<Value> sums;
  for (unsigned v = numOuts; v != numValues; ++v) {
    Type outside = op.getReduce()[v - numOuts].getType();
    auto found = llvm::find(sumsYielded, v);
    if (found == sumsYielded.end()) {
      sums.push_back(createZero(b, loc, outside));
      continue;
    }
    Value total = totals[numF + unsigned(found - sumsYielded.begin())];
    if (total.getType() != outside)
      total = arith::ExtFOp::create(b, loc, outside, total);
    double weight = 2.0;
    if (auto weights = op.getWeights())
      weight *= (*weights)[v - numOuts];
    FloatAttr scalar = b.getFloatAttr(getElementTypeOrSelf(outside), weight);
    Value factor;
    if (auto vector = dyn_cast<VectorType>(outside))
      factor = arith::ConstantOp::create(
          b, loc, outside, DenseElementsAttr::get(vector, scalar.getValue()));
    else
      factor = arith::ConstantOp::create(b, loc, outside, scalar);
    sums.push_back(arith::MulFOp::create(b, loc, factor, total));
  }
  return sums;
}

LogicalResult Schedule::run() {
  if (failed(analyze()))
    return failure();
  OpBuilder builder(op);
  MLIRContext *context = op.getContext();
  Value positions = op.getPositions();
  Value size = memref::DimOp::create(builder, loc, positions, ix(builder, 0));
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCell(), real);
  // In f32 for the shifts of the entries, which are added in the type of
  // the site arithmetic (D417, Section 4.3, step 2).
  Value box32 = convertReal(builder, loc, op.getCell(), f32);

  // The loop adds to its destinations: one that it would overwrite is
  // cleared first.
  for (auto [i, out] : llvm::enumerate(op.getOuts())) {
    if (!op.overwrites(i))
      continue;
    Value destination = out;
    host.launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
      Type element = getElementTypeOrSelf(destination.getType());
      auto memref = cast<MemRefType>(destination.getType());
      Type value = memref.getRank() == 1
                       ? element
                       : VectorType::get({memref.getDimSize(1)}, element);
      storeElement(body, loc, createZero(body, loc, value), destination,
                   particle);
    });
  }

  // The frames and fields of the places, in the room of the structure's
  // places (D312).
  Value room = memref::DimOp::create(builder, loc, lists.order, ix(builder, 0));
  emitPack(builder, box, room);

  // A warp for each record, as many warps as particles at most (the
  // scratch of the sums has a row a particle); a warp takes the records
  // w, w + warps, ... of the device's count, so that none is read back.
  Value one = ix(builder, 1);
  Value warp32 = ix(builder, 32);
  Value block = ix(builder, blockSize);
  Value capacity =
      memref::DimOp::create(builder, loc, lists.groups, ix(builder, 0));
  Value warps = arith::MaxUIOp::create(
      builder, loc, arith::MinUIOp::create(builder, loc, capacity, size), one);
  Value threads = arith::MulIOp::create(builder, loc, warps, warp32);
  Value grid = arith::DivUIOp::create(
      builder, loc,
      arith::AddIOp::create(builder, loc, threads, ix(builder, blockSize - 1)),
      block);
  auto launch = gpu::LaunchOp::create(builder, loc, grid, one, one, block, one,
                                      one);
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value thread = arith::AddIOp::create(
      kernel, loc,
      arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, block),
      launch.getThreadIds().x);
  Value warp = arith::DivUIOp::create(kernel, loc, thread, warp32);
  Value lane = arith::IndexCastOp::create(
      kernel, loc, i32, arith::RemUIOp::create(kernel, loc, thread, warp32));
  Value valid = arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::ult,
                                      warp, warps);
  Value end = lists.dual ? load(kernel, lists.activeCount, ix(kernel, 0))
                         : load(kernel, lists.offsets, ix(kernel, S));
  end = toIndex(kernel, end);
  Value start = arith::SelectOp::create(kernel, loc, valid, warp, end);
  SmallVector<Value> zeros;
  for (Value initial : op.getReduce())
    zeros.push_back(createZero(kernel, loc, initial.getType()));
  auto loop = scf::ForOp::create(
      kernel, loc, start, end, warps, zeros,
      [&](OpBuilder &body, Location, Value w, ValueRange sums) {
        Value record = lists.dual
                           ? Value(load(body, lists.active, w))
                           : Value(arith::IndexCastOp::create(body, loc, i32, w));
        // The record's segment, from the first record of each: uniform in
        // the warp.
        Value segment = ci(body, 0);
        for (unsigned t = 1; t < S; ++t) {
          Value past = arith::CmpIOp::create(
              body, loc, arith::CmpIPredicate::sge, record,
              load(body, lists.offsets, ix(body, t)));
          segment = arith::SelectOp::create(body, loc, past, ci(body, t),
                                            segment);
        }
        // A segment that the op names is a branch of its own; another
        // contributes nothing and has no code.
        std::function<SmallVector<Value>(OpBuilder &, unsigned)> chain =
            [&](OpBuilder &at, unsigned r) -> SmallVector<Value> {
          SmallVector<Value> none;
          for (Value initial : op.getReduce())
            none.push_back(createZero(at, loc, initial.getType()));
          if (r == regions.size())
            return none;
          Value here = arith::CmpIOp::create(
              at, loc, arith::CmpIPredicate::eq, segment,
              ci(at, regions[r].s));
          auto choice = branch(at, ValueRange(none).getTypes(), here);
          OpBuilder then = OpBuilder::atBlockEnd(choice.thenBlock());
          scf::YieldOp::create(then, loc,
                               emitRecord(then, regions[r], record, lane,
                                          box32));
          OpBuilder otherwise = OpBuilder::atBlockEnd(choice.elseBlock());
          scf::YieldOp::create(otherwise, loc, chain(otherwise, r + 1));
          return SmallVector<Value>(choice.getResults());
        };
        SmallVector<Value> contributions = chain(body, 0);
        SmallVector<Value> next;
        for (auto [sum, value] : llvm::zip(sums, contributions))
          next.push_back(arith::AddFOp::create(body, loc, sum, value));
        scf::YieldOp::create(body, loc, next);
      });
  SmallVector<Value> totals(loop.getResults());
  for (Value &total : totals)
    total = emitWarpReduce(kernel, loc, total, Combiner::Add);
  if (!totals.empty()) {
    Value leader = arith::AndIOp::create(
        kernel, loc, valid,
        arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::eq, lane,
                              ci(kernel, 0)));
    scf::IfOp::create(kernel, loc, leader, [&](OpBuilder &then, Location) {
      for (auto [index, total] : llvm::enumerate(totals))
        storeElement(then, loc, total, op.getScratch()[2 * index], warp);
      scf::YieldOp::create(then, loc);
    });
  }
  gpu::TerminatorOp::create(kernel, loc);
  host.bringIn(launch);
  (void)context;

  // The packed frames are freed once the loop has read them.
  SmallVector<Value> packed = {anchors};
  if (offsets)
    packed.push_back(offsets);
  for (Value field : fields)
    if (field)
      packed.push_back(field);
  for (Value buffer : packed) {
    auto memref = cast<MemRefType>(buffer.getType());
    Value plain = memref::MemorySpaceCastOp::create(
        builder, loc,
        MemRefType::get(memref.getShape(), memref.getElementType()), buffer);
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }
  return host.finishSums(op, builder, op.getReduce(), op.getScratch(), warps);
}
} // namespace

LogicalResult kernels::lowerTraversePairsSegmentsSIMT(
    md_alg::TraversePairsOp op, TraverseHost &host, const SegmentLists &lists,
    int64_t blockSize, int64_t rotation) {
  if (op.isBufferForm())
    return op.emitOpError()
           << "takes its list of units as buffers, which only its serial "
              "lowering takes: the SIMT lowering takes the structure form";
  // A stable visit needs a list whose segments state `stable(group,
  // search)`, which no build of a list of units states (D417, Section 2),
  // so the verifier refuses it (V9); the pruning in the loop is set by
  // convert-md-exec-to-gpu for lists of groups alone (findSharedFrames).
  assert(op.getVisit() != md_alg::Visit::Stable &&
         "a stable visit over a list of units, which no build states");
  assert(!op.getPruneGroups() &&
         "the pruning in the loop over a list of units, which the pass does "
         "not set");
  auto type = cast<mdrt::SegmentedPairsType>(op.getListType());
  if (type.getNumKinds() > 2)
    return op.emitOpError()
           << "takes " << type.getNumKinds()
           << " kinds; the pair segments take at most 2 kinds of rigid "
              "groups (D405), and the schedule is measured with those";
  if (rotation != 16)
    return op.emitOpError()
           << "turns its partners through " << rotation
           << " lanes; the schedule of a list of units holds 16 units a "
              "half-warp and turns them through 16";
  if (blockSize % 32 != 0)
    return op.emitOpError() << "runs a warp for each record, which needs "
                               "blocks of whole warps, not "
                            << blockSize << " threads";
  return Schedule(op, host, lists, blockSize, rotation).run();
}

// The lowerings of md_alg.traverse_pairs (D415); see
// include/mdir/Conversion/TraversePairs.h. The SIMT schedule is that of
// the loops over pairs over groups before the op existed, moved here from
// convert-md-exec-to-gpu with its code unchanged, so that every program
// keeps its bytes (D406's method). The serial definition is written apart
// from it, in the order of the definition.

#include "mdir/Conversion/TraversePairs.h"

#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mdir/Conversion/ScatterEmitter.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDExec/Scatter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

/// The algebras that `op` states, one for each destination and then each
/// sum.
static SmallVector<md_exec::Algebra> getAlgebras(md_alg::TraversePairsOp op) {
  SmallVector<md_exec::Algebra> algebras;
  for (Attribute entry : op.getAlgebra())
    algebras.push_back(
        *md_exec::symbolizeAlgebra(cast<StringAttr>(entry).getValue()));
  return algebras;
}

/// Those of its destinations.
static std::optional<SmallVector<md_exec::Algebra>>
getDestinationAlgebras(md_alg::TraversePairsOp op) {
  SmallVector<md_exec::Algebra> algebras = getAlgebras(op);
  algebras.truncate(op.getOuts().size());
  return algebras;
}

static Value createSize(OpBuilder &builder, Location loc, Value buffer) {
  return memref::DimOp::create(builder, loc, buffer,
                               createIndex(builder, loc, 0));
}

/// The number of groups of `group` items that hold `count` items.
static Value createGroups(OpBuilder &builder, Location loc, Value count,
                          int64_t group) {
  Value padded = arith::AddIOp::create(builder, loc, count,
                                       createIndex(builder, loc, group - 1));
  return arith::DivUIOp::create(builder, loc, padded,
                                createIndex(builder, loc, group));
}

//===----------------------------------------------------------------------===//
// The SIMT schedule of the groups
//===----------------------------------------------------------------------===//

static SmallVector<Value> getComponents(OpBuilder &b, Location loc,
                                        Value value) {
  auto vector = dyn_cast<VectorType>(value.getType());
  if (!vector)
    return {value};
  SmallVector<Value> parts;
  for (int64_t c = 0, e = vector.getNumElements(); c != e; ++c)
    parts.push_back(vector::ExtractOp::create(b, loc, value, c));
  return parts;
}

/// A value of `type` (a number or a vector) from its components.
static Value fromComponents(OpBuilder &b, Location loc, Type type,
                            ArrayRef<Value> parts) {
  if (auto vector = dyn_cast<VectorType>(type))
    return vector::FromElementsOp::create(b, loc, vector, parts);
  return parts.front();
}

static SmallVector<Value> emitGroupPairKernel(
    OpBuilder &builder, md_alg::TraversePairsOp op, const GroupLists &lists,
    const PairLayout &layout, Value box, Value inverse, Value unit,
    Value lane, IRMapping &local, const GroupStream *stream,
    AddToMember add) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  unsigned numIns = op.getIns().size();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();
  Type computed = kernel.getArgument(0).getType();
  double cutoff = op.getCutoff().convertToDouble();
  Value cutoff2 = createCutoff2(builder, loc, computed, cutoff);
  (void)inverse;
  Type i32 = builder.getI32Type();
  auto constant32 = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v));
  };
  auto toIndex = [&](OpBuilder &b, Value v) -> Value {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
  };

  // The unit: its group, and its entries.
  Value group, k;
  if (stream) {
    group = arith::IndexCastOp::create(builder, loc, i32, unit);
  } else {
    group = memref::LoadOp::create(builder, loc, lists.units, ValueRange{unit});
    k = memref::LoadOp::create(builder, loc, lists.ordinals, ValueRange{unit});
  }
  Value groupIndex = toIndex(builder, group);
  Value begin, end;
  if (stream) {
    // The stream of the group, from its front (D284).
    begin = constant32(builder, 0);
    end = stream->count;
  } else if (lists.blockCounts) {
    // The inner list keeps its entries at the front of each block.
    begin = constant32(builder, 0);
    end = memref::LoadOp::create(builder, loc, lists.blockCounts,
                                 ValueRange{unit});
  } else {
    Value count = memref::LoadOp::create(builder, loc, lists.counts,
                                         ValueRange{groupIndex});
    begin = arith::MulIOp::create(builder, loc, k, constant32(builder, 64));
    end = arith::MinSIOp::create(
        builder, loc, count,
        arith::AddIOp::create(builder, loc, begin, constant32(builder, 64)));
  }

  // The particle of this lane: place 16 g + u.
  Value lane32 = arith::IndexCastOp::create(builder, loc, i32, lane);
  Value u = arith::AndIOp::create(builder, loc, lane32, constant32(builder, 15));
  Value half = arith::AndIOp::create(builder, loc, lane32,
                                     constant32(builder, 16));
  Value mine32 = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, group, constant32(builder, 16)), u);
  Value mine = toIndex(builder, mine32);
  Value myPosition = loadElement(builder, loc, layout.positions, mine);
  SmallVector<Value> myValues;
  for (Value buffer : layout.ins)
    myValues.push_back(loadElement(builder, loc, buffer, mine));
  // The lane that the values come from at each step: the one before in the
  // half-warp.
  Value source = arith::OrIOp::create(
      builder, loc, half,
      arith::AndIOp::create(
          builder, loc,
          arith::AddIOp::create(builder, loc, u, constant32(builder, 15)),
          constant32(builder, 15)));
  Value width = constant32(builder, 32);
  auto shuffle = [&](OpBuilder &b, Value value) -> Value {
    SmallVector<Value> parts = getComponents(b, loc, value);
    for (Value &part : parts)
      part = gpu::ShuffleOp::create(b, loc, part, source, width,
                                    gpu::ShuffleMode::IDX)
                 .getShuffleResult();
    return fromComponents(b, loc, value.getType(), parts);
  };

  // The values of the kernel go to the particle of the group in the type
  // the kernel computes them in; the sums likewise, widened at the end.
  auto getNarrow = [&](unsigned i) -> Value {
    if (i < numOuts)
      return Value();
    auto widen = yield->getOperand(i).getDefiningOp<arith::ExtFOp>();
    if (!widen || widen->getBlock() != &kernel)
      return Value();
    return widen.getIn();
  };
  auto yieldType = [&](unsigned i) {
    Value narrow = getNarrow(i);
    return narrow ? narrow.getType() : yield->getOperand(i).getType();
  };
  SmallVector<Value> initial;
  for (unsigned i = 0; i != numYields; ++i)
    initial.push_back(createZero(builder, loc, yieldType(i)));

  // The entries, 32 at a time: at most two rounds for a unit of 64. The
  // unit is block `unit` of the entries.
  Value blockBase = arith::MulIOp::create(builder, loc, unit,
                                          createIndex(builder, loc, 64));
  // Of a stream: the row of the group's blocks.
  Value streamRow;
  if (stream)
    streamRow = arith::MulIOp::create(builder, loc, groupIndex, stream->stride);
  Value first = toIndex(builder, begin);
  Value last = toIndex(builder, end);
  Value step32 = createIndex(builder, loc, 32);
  auto rounds = scf::ForOp::create(
      builder, loc, first, last, step32, initial,
      [&](OpBuilder &b, Location, Value e0, ValueRange acc) {
        Value e = arith::AddIOp::create(b, loc, e0, lane);
        Value has = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                          e, last);
        Value safe = arith::SelectOp::create(b, loc, has, e, first);
        // Entry e of the list is entry e - 64 k of the block; of a stream,
        // entry e mod 64 of its block e / 64.
        Value at;
        if (stream) {
          Value c64 = createIndex(b, loc, 64);
          Value block = toIndex(
              b, memref::LoadOp::create(
                     b, loc, stream->blockOf,
                     ValueRange{arith::AddIOp::create(
                         b, loc, streamRow,
                         arith::DivUIOp::create(b, loc, safe, c64))}));
          at = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, block, c64),
              arith::RemUIOp::create(b, loc, safe, c64));
        } else {
          at = arith::AddIOp::create(
              b, loc, arith::SubIOp::create(b, loc, safe, first), blockBase);
        }
        Value place32 =
            memref::LoadOp::create(b, loc, lists.entries, ValueRange{at});
        Value mask = arith::SelectOp::create(
            b, loc, has,
            memref::LoadOp::create(b, loc, lists.masks, ValueRange{at}),
            constant32(b, 0));
        Value place = toIndex(b, place32);
        // The entry in the frame of the group: its position, in the frame
        // of its own group, moved by the whole cells in bits 16 to 27 of
        // its mask, (e + 4) in four bits an axis, or by the lattice vectors
        // in bits 16 to 30, (e + 16) in five bits each, for a triclinic
        // cell (D95, D115).
        Value position = loadElement(b, loc, layout.positions, place);
        {
          auto positionType = cast<VectorType>(position.getType());
          Type element = positionType.getElementType();
          int64_t width = getEntryShiftBits(isTriclinic(box));
          int64_t offset = getEntryShiftOffset(isTriclinic(box));
          SmallVector<Value> cells;
          for (int64_t axis = 0; axis != 3; ++axis) {
            Value bits = arith::AndIOp::create(
                b, loc,
                arith::ShRUIOp::create(b, loc, mask,
                                       constant32(b, 16 + width * axis)),
                constant32(b, (1 << width) - 1));
            cells.push_back(arith::SIToFPOp::create(
                b, loc, element,
                arith::SubIOp::create(b, loc, bits, constant32(b, offset))));
          }
          Value shift = emitLatticeShift(
              b, loc, vector::FromElementsOp::create(b, loc, positionType, cells),
              convertReal(b, loc, box, element));
          position = arith::AddFOp::create(b, loc, position, shift);
        }
        SmallVector<Value> values;
        for (Value buffer : layout.ins)
          values.push_back(loadElement(b, loc, buffer, place));

        // The bits of the pairs of this lane's particle with the 16 entries
        // of its half-warp, as they come by: a ballot for each particle of
        // the group gives the bits of every entry, and each lane keeps
        // those of its own, rotated so that step k tests bit 15 - k and
        // each step shifts by one. The mask then does not turn with the
        // entry: a shuffle a pair less, on the pipe of loads and shuffles
        // that bounds the loop (D97).
        Value own16 = constant32(b, 0);
        for (int64_t p = 0; p != 16; ++p) {
          Value set = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::ne,
              arith::AndIOp::create(b, loc, mask, constant32(b, 1 << p)),
              constant32(b, 0));
          Value word = gpu::BallotOp::create(b, loc, b.getI32Type(), set);
          own16 = arith::SelectOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, u,
                                    constant32(b, p)),
              word, own16);
        }
        Value low16 = constant32(b, 0xFFFF);
        Value halfBits = arith::AndIOp::create(
            b, loc, arith::ShRUIOp::create(b, loc, own16, half), low16);
        Value turn = arith::SubIOp::create(b, loc, constant32(b, 15), u);
        Value pairBits = arith::AndIOp::create(
            b, loc,
            arith::OrIOp::create(
                b, loc, arith::ShLIOp::create(b, loc, halfBits, turn),
                arith::ShRUIOp::create(
                    b, loc, halfBits,
                    arith::SubIOp::create(b, loc, constant32(b, 16), turn))),
            low16);

        // 16 steps, the entry's values and what it receives turning.
        SmallVector<Value> carried = {position, pairBits};
        carried.append(values.begin(), values.end());
        for (unsigned i = 0; i != numOuts; ++i)
          carried.push_back(createZero(b, loc, yieldType(i)));
        carried.append(acc.begin(), acc.end());
        // The counter in i32: an index is 64 bits on the device, two
        // additions and two comparisons a step.
        auto steps = scf::ForOp::create(
            b, loc, constant32(b, 0), constant32(b, 16), constant32(b, 1),
            carried,
            [&](OpBuilder &s, Location, Value, ValueRange state) {
              Value otherPosition = state[0];
              Value bits = state[1];
              ValueRange otherValues = state.slice(2, numIns);
              ValueRange received = state.slice(2 + numIns, numOuts);
              ValueRange own = state.slice(2 + numIns + numOuts, numYields);

              Value bit = arith::AndIOp::create(s, loc, bits,
                                                constant32(s, 0x8000));
              Value paired = arith::CmpIOp::create(
                  s, loc, arith::CmpIPredicate::ne, bit, constant32(s, 0));
              // Both are in the frame of the group: the displacement is the
              // minimum image without a rounding (D95).
              Value d = convertReal(
                  s, loc,
                  arith::SubFOp::create(s, loc, myPosition, otherPosition),
                  computed);
              Value squares = arith::MulFOp::create(s, loc, d, d);
              Value distance2 = vector::ReductionOp::create(
                  s, loc, vector::CombiningKind::ADD, squares);
              // A slot whose bit is clear takes a squared distance beyond
              // the cutoff, so that one comparison masks it: the kernel
              // sees a finite distance and its value is not taken.
              Value r2 = arith::SelectOp::create(
                  s, loc, paired, distance2,
                  createReal(s, loc, computed, 4.0 * cutoff * cutoff));
              IRMapping inside = local;
              inside.map(kernel.getArgument(0), r2);
              inside.map(kernel.getArgument(1), d);
              for (unsigned i = 0; i != numIns; ++i) {
                inside.map(kernel.getArgument(2 + 2 * i), myValues[i]);
                inside.map(kernel.getArgument(3 + 2 * i), otherValues[i]);
              }
              for (Operation &nested : kernel.without_terminator())
                s.clone(nested, inside);
              Value within = arith::CmpFOp::create(
                  s, loc, arith::CmpFPredicate::OLT, r2, cutoff2);

              SmallVector<Value> nextOwn, nextReceived;
              for (unsigned i = 0; i != numYields; ++i) {
                Value narrow = getNarrow(i);
                Value value =
                    inside.lookupOrDefault(narrow ? narrow : yield->getOperand(i));
                Value nothing = createZero(s, loc, value.getType());
                Value masked =
                    arith::SelectOp::create(s, loc, within, value, nothing);
                nextOwn.push_back(arith::AddFOp::create(s, loc, own[i], masked));
                if (i < numOuts) {
                  // k(j, i) = s k(i, j), s the sign of the contract.
                  bool antisymmetric =
                      op.getExchangeOf(i) == "antisymmetric";
                  nextReceived.push_back(
                      antisymmetric
                          ? arith::SubFOp::create(s, loc, received[i], masked)
                                .getResult()
                          : arith::AddFOp::create(s, loc, received[i], masked)
                                .getResult());
                }
              }
              SmallVector<Value> next = {
                  shuffle(s, otherPosition),
                  arith::ShLIOp::create(s, loc, bits, constant32(s, 1))};
              for (Value value : otherValues)
                next.push_back(shuffle(s, value));
              for (Value value : nextReceived)
                next.push_back(shuffle(s, value));
              next.append(nextOwn.begin(), nextOwn.end());
              scf::YieldOp::create(s, loc, next);
            });
        // After 16 steps each entry is back at its lane: its values go to
        // its particle.
        ValueRange results = steps.getResults();
        Value particle32 = memref::LoadOp::create(b, loc, lists.order,
                                                  ValueRange{place});
        Value real = arith::AndIOp::create(
            b, loc, has,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                  particle32, constant32(b, 0)));
        scf::IfOp::create(b, loc, real, [&](OpBuilder &then, Location) {
          Value particle = toIndex(then, particle32);
          for (unsigned i = 0; i != numOuts; ++i) {
            Value destination = op.getOuts()[i];
            Type stored = getElementTypeOrSelf(destination.getType());
            Value value = convertReal(then, loc, results[2 + numIns + i],
                                      stored);
            if (add)
              add(then, i, particle, value);
            else
              emitAtomicAdd(then, loc, value, destination, particle);
          }
          scf::YieldOp::create(then, loc);
        });
        scf::YieldOp::create(
            b, loc,
            ValueRange(results.slice(2 + numIns + numOuts, numYields)));
      });

  // The values of the particles of the group: the two half-warps add up,
  // and the first adds them to the destinations.
  SmallVector<Value> totals(rounds.getResults());
  Value sixteen = constant32(builder, 16);
  for (unsigned i = 0; i != numOuts; ++i) {
    SmallVector<Value> parts = getComponents(builder, loc, totals[i]);
    for (Value &part : parts)
      part = arith::AddFOp::create(
          builder, loc, part,
          gpu::ShuffleOp::create(builder, loc, part, sixteen, width,
                                 gpu::ShuffleMode::XOR)
              .getShuffleResult());
    totals[i] = fromComponents(builder, loc, totals[i].getType(), parts);
  }
  Value particle32 = memref::LoadOp::create(builder, loc, lists.order,
                                            ValueRange{mine});
  // A unit with no entries, a block that the inner list of a dual list
  // leaves empty (D114), writes nothing.
  Value writes = arith::AndIOp::create(
      builder, loc,
      arith::AndIOp::create(
          builder, loc,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, half,
                                constant32(builder, 0)),
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                particle32, constant32(builder, 0))),
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, begin,
                            end));
  if (numOuts != 0)
    scf::IfOp::create(builder, loc, writes, [&](OpBuilder &then, Location) {
      Value particle = toIndex(then, particle32);
      for (unsigned i = 0; i != numOuts; ++i) {
        Value destination = op.getOuts()[i];
        Type stored = getElementTypeOrSelf(destination.getType());
        Value value = convertReal(then, loc, totals[i], stored);
        if (add)
          add(then, i, particle, value);
        else
          emitAtomicAdd(then, loc, value, destination, particle);
      }
      scf::YieldOp::create(then, loc);
    });

  // A weight is that of the sum over both orders of each pair; the
  // kernel of a sum is symmetric, so each pair once counts twice.
  SmallVector<Value> contributions;
  for (unsigned i = numOuts; i != numYields; ++i) {
    Value total = totals[i];
    if (getNarrow(i))
      total = arith::ExtFOp::create(builder, loc,
                                    yield->getOperand(i).getType(), total);
    double weight = 2.0;
    if (auto weights = op.getWeights())
      weight *= (*weights)[i - numOuts];
    if (weight != 1.0) {
      Type type = total.getType();
      FloatAttr scalar =
          builder.getFloatAttr(getElementTypeOrSelf(type), weight);
      Value factor;
      if (auto vector = dyn_cast<VectorType>(type))
        factor = arith::ConstantOp::create(
            builder, loc, type,
            DenseElementsAttr::get(vector, scalar.getValue()));
      else
        factor = arith::ConstantOp::create(builder, loc, type, scalar);
      total = arith::MulFOp::create(builder, loc, factor, total);
    }
    contributions.push_back(total);
  }
  return contributions;
}

LogicalResult kernels::lowerTraversePairsSIMT(md_alg::TraversePairsOp op,
                                              TraverseHost &host) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  MLIRContext *context = op.getContext();
  int64_t blockSize = host.getBlockSize();
  if (op.isBufferForm())
    return op.emitOpError()
           << "takes its list as buffers, which only its serial lowering "
              "takes: the SIMT lowering takes the structure form";
  if (op.getVisit() == md_alg::Visit::Stable)
    return op.emitOpError()
           << "visits its entries stably, which the schedule of the groups "
              "does not: its rotation and its atomics fix no tree";
  // The verifier has refused what the schedule could not keep: `owner`
  // takes `directed`, which convert-md-exec-to-gpu refuses before it calls
  // the schedule; and a `fixed_order` value with `unique` or `atomic`
  // (D290, D301): the atomic additions make the sums depend on the order
  // of the threads.
  assert(op.getCombine() == md_alg::Combine::Atomic &&
         op.getMultiplicity() == md_alg::Multiplicity::Unique &&
         "the verifier and the pass refuse another combination");
  assert(!llvm::is_contained(getAlgebras(op), md_exec::Algebra::FixedOrder) &&
         "the verifier refuses fixed_order with unique and atomic");
  if (blockSize % 32 != 0)
    return op.emitOpError() << "runs a warp for each unit of work, which "
                               "needs blocks of whole warps, not "
                            << blockSize << " threads";

  Value positions = op.getPositions();
  Value size = createSize(builder, loc, positions);
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Value box = convertReal(builder, loc, op.getCell(), real);
  Value inverse = createInverse(builder, loc, box);

  // The loop adds to its destinations: one that it would overwrite is
  // cleared first.
  for (auto [i, out] : llvm::enumerate(op.getOuts())) {
    if (!op.overwrites(i))
      continue;
    Value destination = out;
    host.launchOver(builder, loc, size, [&](OpBuilder &body, Value particle) {
      Type element = getElementTypeOrSelf(destination.getType());
      auto type = cast<MemRefType>(destination.getType());
      Type value = type.getRank() == 1
                       ? element
                       : VectorType::get({type.getDimSize(1)}, element);
      storeElement(body, loc, createZero(body, loc, value), destination,
                   particle);
    });
  }

  // The buffers of the structure as the runtime holds them now.
  GroupBuffers buffers = host.getGroupsBuffers(builder, loc);

  // The positions and the fields that the kernel reads, in the order of
  // the places of the last build; an empty place takes those of the
  // particle 0, which no pair reads. A position is moved by whole cells to
  // the frame of its group (the shift of its place, ten bits an axis from
  // -512), so that the kernel takes no minimum image (D95). The positions
  // come as they are stored, unwrapped and in f64 in mixed precision: the
  // shift is added there, and the sum converted to the type of the kernel,
  // so that a position in the frame is as exact however far the particle
  // has gone.
  Type computed = op.getKernel().front().getArgument(0).getType();
  Value places = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      memref::LoadOp::create(builder, loc, host.getSizes(),
                             ValueRange{createIndex(builder, loc, 0)}));
  // Where the loop follows a refresh of a dual list at the same positions
  // and in the same cell (sharesPrunedFrames), and computes in f32, a
  // pruning of that refresh has left the positions in the frames as the
  // loop takes them, to the bit: the loop takes them as they are then
  // (D268).
  Value sharedFrames, pruned;
  {
    auto stored = cast<MemRefType>(positions.getType());
    MemRefType framed =
        MemRefType::get(stored.getShape(), computed, stored.getLayout(),
                        stored.getMemorySpace());
    if (host.isDual() && host.getPrunedNow() && computed.isF32() &&
        op.getFrames()) {
      Value frames = host.getGroupsPlaced(builder, loc);
      if (frames.getType() == framed) {
        sharedFrames = frames;
        pruned = memref::LoadOp::create(builder, loc, host.getPrunedNow(),
                                        ValueRange{});
      }
    }
  }
  // The kernel below takes it as a number.
  Value pruned32 =
      pruned ? Value(arith::ExtUIOp::create(builder, loc, builder.getI32Type(),
                                            pruned))
             : Value();
  SmallVector<Value> sources = {positions};
  llvm::append_range(sources, op.getIns());
  // The buffers take the room of the structure's places, not the number
  // of places that this build occupied (D312): that number
  // changes at nearly every build, and the runtime's pool reuses a block
  // only for a request of the same size, so a buffer sized by it left a
  // block in the pool for each new number. The room changes only when the
  // structure grows.
  Value room = memref::DimOp::create(builder, loc, buffers.order,
                                     createIndex(builder, loc, 0));
  SmallVector<Value> targets;
  for (auto [index, source] : llvm::enumerate(sources)) {
    auto type = cast<MemRefType>(source.getType());
    if (index == 0)
      type = MemRefType::get(type.getShape(), computed, type.getLayout(),
                             type.getMemorySpace());
    targets.push_back(
        host.createDeviceBuffer(builder, loc, type, ValueRange{room}));
  }
  host.launchOver(builder, loc, places, [&](OpBuilder &body, Value place) {
    Value at = memref::LoadOp::create(body, loc, buffers.order,
                                      ValueRange{place});
    Value empty = arith::CmpIOp::create(
        body, loc, arith::CmpIPredicate::slt, at,
        arith::ConstantOp::create(body, loc, body.getI32Type(),
                                  body.getI32IntegerAttr(0)));
    Value particle = arith::SelectOp::create(
        body, loc, empty, createIndex(body, loc, 0),
        arith::IndexCastOp::create(body, loc, body.getIndexType(), at));
    Value packed = memref::LoadOp::create(body, loc, buffers.shifts,
                                          ValueRange{place});
    auto positionType = cast<MemRefType>(sources.front().getType());
    Type element = positionType.getElementType();
    SmallVector<Value> cells;
    for (int64_t k = 0; k != 3; ++k) {
      Value bits = arith::AndIOp::create(
          body, loc,
          arith::ShRUIOp::create(
              body, loc, packed,
              arith::ConstantOp::create(body, loc, body.getI32Type(),
                                        body.getI32IntegerAttr(10 * k))),
          arith::ConstantOp::create(body, loc, body.getI32Type(),
                                    body.getI32IntegerAttr(1023)));
      Value count = arith::SubIOp::create(
          body, loc, bits,
          arith::ConstantOp::create(body, loc, body.getI32Type(),
                                    body.getI32IntegerAttr(512)));
      cells.push_back(arith::SIToFPOp::create(body, loc, element, count));
    }
    Value shift = emitLatticeShift(
        body, loc,
        vector::FromElementsOp::create(body, loc,
                                       VectorType::get({3}, element), cells),
        convertReal(body, loc, box, element));
    for (auto [index, pair] : llvm::enumerate(llvm::zip(sources, targets))) {
      auto [source, target] = pair;
      auto store = [&, source = source, target = target](OpBuilder &at) {
        Value value = loadElement(at, loc, source, particle);
        if (index == 0)
          value = convertReal(
              at, loc, arith::AddFOp::create(at, loc, value, shift),
              computed);
        storeElement(at, loc, value, target, place);
      };
      if (index != 0 || !pruned32) {
        store(body);
        continue;
      }
      scf::IfOp::create(
          body, loc,
          arith::CmpIOp::create(
              body, loc, arith::CmpIPredicate::eq, pruned32,
              arith::ConstantOp::create(body, loc, body.getI32Type(),
                                        body.getI32IntegerAttr(0))),
          [&](OpBuilder &then, Location) {
            store(then);
            scf::YieldOp::create(then, loc);
          });
    }
  });
  PairLayout layout;
  layout.order = buffers.order;
  layout.positions = targets.front();
  if (sharedFrames) {
    Type type = targets.front().getType();
    auto choice = scf::IfOp::create(builder, loc, TypeRange{type}, pruned,
                                    /*withElseRegion=*/true);
    OpBuilder then = OpBuilder::atBlockEnd(choice.thenBlock());
    scf::YieldOp::create(then, loc, sharedFrames);
    OpBuilder otherwise = OpBuilder::atBlockEnd(choice.elseBlock());
    scf::YieldOp::create(otherwise, loc, targets.front());
    layout.positions = choice.getResult(0);
  }
  layout.ins.assign(targets.begin() + 1, targets.end());
  kernels::GroupLists lists{buffers.entries, buffers.masks, buffers.counts,
                            buffers.units, buffers.ordinals, buffers.order};
  // A dual list: the inner list, in the blocks of the outer one (D114),
  // through the units that hold entries of it (D268).
  Value activeUnits, activeCount;
  if (host.isDual()) {
    InnerLists inner = host.getInnerLists(builder, loc);
    lists.entries = inner.entries;
    lists.masks = inner.masks;
    lists.blockCounts = inner.counts;
    PruneBuffers prune = host.getPruneBuffers(builder, loc);
    activeUnits = prune.activeUnits;
    activeCount = prune.activeCount;
  }

  // A warp for each unit of work, as many warps as particles at most: the
  // sums of the warps go to the scratch of the particles. A warp takes the
  // units w, w + warps, ...
  Operation *yield = op.getKernel().front().getTerminator();
  unsigned numOuts = op.getOuts().size();
  Value one = createIndex(builder, loc, 1);
  Value warp32 = createIndex(builder, loc, 32);
  Value block = createIndex(builder, loc, blockSize);
  // The table of a block for the values at the particles
  // (`scatter(hash(S, P, L))`, D291), its precondition checked again
  // with the chooser's function (Scatter.h).
  auto scatter =
      dyn_cast_if_present<md_exec::ScatterAttr>(op.getScatterAttr());
  bool hashed = scatter && scatter.isHash();
  SmallVector<Value> outs(op.getOuts());
  SmallVector<md_exec::scatter::Destination> destinations;
  if (hashed)
    if (md_exec::scatter::Refusal refused = md_exec::scatter::checkHash(
            scatter.getSlots(), op.getOuts().getTypes(),
            getDestinationAlgebras(op), &destinations))
      return op.emitOpError()
             << "takes the strategy 'hash', whose precondition does not "
                "hold: "
             << *refused;
  using WarpBody =
      function_ref<SmallVector<Value>(OpBuilder &, Value, Value, AddToMember)>;
  // A loop of `count` units of work, `body` emitting each: returns the
  // number of warps whose sums are in the scratch. With `tabled`, what the
  // warps of a block add to the particles goes to the table of the block,
  // flushed once after the loop.
  auto emitWarps = [&](OpBuilder &at, Value count,
                       function_ref<Value(OpBuilder &, Value)> first,
                       WarpBody body, bool tabled) -> Value {
    Value warps = arith::MaxUIOp::create(
        at, loc, arith::MinUIOp::create(at, loc, count, size), one);
    Value threads = arith::MulIOp::create(at, loc, warps, warp32);
    Value grid = createGroups(at, loc, threads, blockSize);
    SmallVector<Type> attributions;
    if (tabled)
      attributions = kernels::getHashAttributions(context, scatter.getSlots(),
                                                  destinations);
    auto launch = gpu::LaunchOp::create(
        at, loc, grid, one, one, block, one, one,
        /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
        /*asyncDependencies=*/{}, /*workgroupAttributions=*/attributions);
    OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
    std::optional<kernels::BlockHashTable> table;
    if (tabled) {
      table.emplace(kernel, loc, scatter.getSlots(), scatter.getProbes(),
                    launch.getWorkgroupAttributionBBArgs(), outs, destinations,
                    launch.getThreadIds().x, block);
      table->emitClear(kernel);
    }
    Value thread = arith::AddIOp::create(
        kernel, loc,
        arith::MulIOp::create(kernel, loc, launch.getBlockIds().x, block),
        launch.getThreadIds().x);
    Value warp = arith::DivUIOp::create(kernel, loc, thread, warp32);
    Value lane = arith::RemUIOp::create(kernel, loc, thread, warp32);
    Value valid = arith::CmpIOp::create(kernel, loc,
                                        arith::CmpIPredicate::ult, warp, warps);
    Value end = first(kernel, count);
    Value start = arith::SelectOp::create(kernel, loc, valid, warp, end);
    SmallVector<Value> initial;
    for (unsigned i = numOuts, e = yield->getNumOperands(); i != e; ++i)
      initial.push_back(
          createZero(kernel, loc, yield->getOperand(i).getType()));
    auto add = [&](OpBuilder &b, unsigned out, Value member, Value value) {
      table->add(b, outs[out], member, value);
    };
    auto loop = scf::ForOp::create(
        kernel, loc, start, end, warps, initial,
        [&](OpBuilder &b, Location, Value unit, ValueRange sums) {
          SmallVector<Value> contributions =
              body(b, unit, lane, tabled ? AddToMember(add) : AddToMember());
          SmallVector<Value> next;
          for (auto [sum, contribution] : llvm::zip(sums, contributions))
            next.push_back(arith::AddFOp::create(b, loc, sum, contribution));
          scf::YieldOp::create(b, loc, next);
        });
    SmallVector<Value> totals(loop.getResults());
    for (Value &total : totals)
      total = kernels::emitWarpReduce(kernel, loc, total,
                                      kernels::Combiner::Add);
    if (!totals.empty()) {
      Value leader = arith::AndIOp::create(
          kernel, loc, valid,
          arith::CmpIOp::create(kernel, loc, arith::CmpIPredicate::eq, lane,
                                createIndex(kernel, loc, 0)));
      scf::IfOp::create(kernel, loc, leader, [&](OpBuilder &then, Location) {
        for (auto [index, total] : llvm::enumerate(totals))
          storeElement(then, loc, total, op.getScratch()[2 * index], warp);
        scf::YieldOp::create(then, loc);
      });
    }
    if (table)
      table->emitFlush(kernel);
    gpu::TerminatorOp::create(kernel, loc);
    host.bringIn(launch);
    return warps;
  };
  // The loop with the table where the loop takes it: from `L` units of
  // work on (the count on the host), both kernels emitted and the host
  // choosing, as for the tuples (D283).
  auto emitWarpsChosen = [&](OpBuilder &at, Value count,
                             function_ref<Value(OpBuilder &, Value)> first,
                             WarpBody body) -> Value {
    if (!hashed)
      return emitWarps(at, count, first, body, false);
    if (scatter.getLeast() == 0)
      return emitWarps(at, count, first, body, true);
    Value many = arith::CmpIOp::create(
        at, loc, arith::CmpIPredicate::uge, count,
        createIndex(at, loc, scatter.getLeast()));
    auto choice = scf::IfOp::create(at, loc, TypeRange{at.getIndexType()},
                                    many, /*withElseRegion=*/true);
    OpBuilder then = OpBuilder::atBlockEnd(choice.thenBlock());
    scf::YieldOp::create(then, loc, emitWarps(then, count, first, body, true));
    OpBuilder otherwise = OpBuilder::atBlockEnd(choice.elseBlock());
    scf::YieldOp::create(otherwise, loc,
                         emitWarps(otherwise, count, first, body, false));
    return choice.getResult(0);
  };

  // The loop over the units of work: of a dual list, the warps take the
  // active units, as many as the last pruning left on the device; the grid
  // is that of all units.
  Value units = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      memref::LoadOp::create(builder, loc, host.getSizes(),
                             ValueRange{createIndex(builder, loc, 2)}));
  auto emitLoop = [&](OpBuilder &at) -> Value {
    return emitWarpsChosen(
        at, units,
        [&](OpBuilder &kernel, Value count) -> Value {
          if (!activeCount)
            return count;
          return arith::IndexCastOp::create(
              kernel, loc, kernel.getIndexType(),
              memref::LoadOp::create(kernel, loc, activeCount,
                                     ValueRange{createIndex(kernel, loc, 0)}));
        },
        [&](OpBuilder &body, Value unit, Value lane, AddToMember add) {
          if (activeUnits)
            unit = arith::IndexCastOp::create(
                body, loc, body.getIndexType(),
                memref::LoadOp::create(body, loc, activeUnits,
                                       ValueRange{unit}));
          IRMapping local;
          return emitGroupPairKernel(body, op, lists, layout, box,
                                              inverse, unit, lane, local,
                                              /*stream=*/nullptr, add);
        });
  };

  // The loop that prunes for its refresh (D284): where the
  // refresh pruned, and from `fusePruningGroups` groups, a warp for each
  // group prunes the inner list of the group from the outer one as the
  // refresh would (emitPruneGroup) and then computes the pairs of the
  // entries it kept, read back from the inner list after a barrier of the
  // warp, as a stream of the group; the active units of the steps that
  // follow are listed after.
  Value warps;
  if (std::optional<int64_t> fusedGroups = op.getPruneGroups()) {
    Value cell = op.getCell();
    double reach = op.getPruneReach()->convertToDouble();
    Value now = arith::AndIOp::create(
        builder, loc,
        memref::LoadOp::create(builder, loc, host.getPrunedNow(),
                               ValueRange{}),
        host.takesFusedPruning(builder, loc, *fusedGroups));
    auto choice = scf::IfOp::create(builder, loc,
                                    TypeRange{builder.getIndexType()}, now,
                                    /*withElseRegion=*/true);
    {
      OpBuilder then = OpBuilder::atBlockEnd(choice.thenBlock());
      host.countFusedPruning(then, loc);
      PruneContext ctx = host.getPruneContext(then, loc, cell, reach);
      // The warp stores the inner list and loads it back through the same
      // buffers, one argument of the kernel each.
      ctx.inner = {lists.entries, lists.masks, lists.blockCounts};
      Value groupWarps = emitWarpsChosen(
          then, ctx.groups,
          [&](OpBuilder &, Value count) -> Value { return count; },
          [&](OpBuilder &body, Value group, Value lane, AddToMember add) {
            Value kept = host.emitPruneGroup(
                body, loc, ctx, group,
                arith::IndexCastOp::create(body, loc, body.getI32Type(),
                                           lane));
            // The entries that other lanes of the warp wrote.
            NVVM::SyncWarpOp::create(
                body, loc,
                arith::ConstantOp::create(body, loc, body.getI32Type(),
                                          body.getI32IntegerAttr(-1)));
            kernels::GroupLists inner = lists;
            kernels::GroupStream stream{ctx.prune.blockOf, ctx.stride, kept};
            IRMapping local;
            return emitGroupPairKernel(body, op, inner, layout, box,
                                                inverse, group, lane, local,
                                                &stream, add);
          });
      host.emitActiveUnits(then, loc, ctx);
      scf::YieldOp::create(then, loc, groupWarps);
      OpBuilder otherwise = OpBuilder::atBlockEnd(choice.elseBlock());
      scf::YieldOp::create(otherwise, loc, emitLoop(otherwise));
    }
    warps = choice.getResult(0);
    // The inner list is whole only now: the check of the pruning follows
    // the loop (D268).
    host.emitFusedPruningCheck(builder, loc, positions, cell);
  } else {
    warps = emitLoop(builder);
  }

  // The buffers are freed once the loop has read them.
  for (Value target : targets) {
    auto type = cast<MemRefType>(target.getType());
    Value plain = memref::MemorySpaceCastOp::create(
        builder, loc, MemRefType::get(type.getShape(), type.getElementType()),
        target);
    gpu::DeallocOp::create(builder, loc, /*asyncToken=*/Type(),
                           /*asyncDependencies=*/ValueRange(), plain);
  }
  return host.finishSums(op, builder, op.getReduce(), op.getScratch(),
                         warps);
}

//===----------------------------------------------------------------------===//
// The serial definition
//===----------------------------------------------------------------------===//

namespace {
/// The serial definition of one op: the plain loop over the units of work,
/// the entries of each, and the 16 places of its group, in that order or
/// in reverse.
struct Serial {
  OpBuilder &b;
  md_alg::TraversePairsOp op;
  const SerialLists &lists;
  SerialOptions options;
  Location loc;
  Block &kernel;
  Operation *yield;
  unsigned numIns, numOuts, numYields;
  Type computed, real;
  Value box;

  Serial(OpBuilder &b, md_alg::TraversePairsOp op, const SerialLists &lists,
         SerialOptions options)
      : b(b), op(op), lists(lists), options(options), loc(op.getLoc()),
        kernel(op.getKernel().front()), yield(kernel.getTerminator()) {
    numIns = op.getIns().size();
    numOuts = op.getOuts().size();
    numYields = yield->getNumOperands();
    computed = kernel.getArgument(0).getType();
    real = cast<MemRefType>(lists.positions.getType()).getElementType();
    box = lists.cell;
  }

  Value i32(OpBuilder &at, int64_t v) {
    return arith::ConstantOp::create(at, loc, at.getI32Type(),
                                     at.getI32IntegerAttr(v));
  }
  Value toIndex(OpBuilder &at, Value v) {
    return arith::IndexCastOp::create(at, loc, at.getIndexType(), v);
  }
  /// An scf.if with results of `types` and both branches empty: the
  /// callers yield in each.
  scf::IfOp branch(OpBuilder &at, TypeRange types, Value condition) {
    auto choice = scf::IfOp::create(at, loc, types, condition,
                                    /*withElseRegion=*/true);
    for (Block *block : {choice.thenBlock(), choice.elseBlock()})
      if (!block->empty())
        block->getTerminator()->erase();
    return choice;
  }

  Value load(OpBuilder &at, Value buffer, Value index) {
    return memref::LoadOp::create(at, loc, buffer, ValueRange{index});
  }

  /// A loop of `count` steps from `first`, forward or in reverse, whose
  /// body takes the step's value.
  scf::ForOp loop(OpBuilder &at, Value first, Value last, ValueRange initial,
                  function_ref<SmallVector<Value>(OpBuilder &, Value,
                                                  ValueRange)>
                      body) {
    Value one = createIndex(at, loc, 1);
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

  /// The particle at a place, and whether the place holds one.
  std::pair<Value, Value> particleAt(OpBuilder &at, Value place) {
    Value particle32 = load(at, lists.groups.order, place);
    Value real = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::sge,
                                       particle32, i32(at, 0));
    return {particle32, real};
  }

  /// The position of the place in the frame of its group: the stored
  /// position plus the place's shift (whole cells, ten bits an axis from
  /// -512) in the stored type, converted to the type of the region (D95).
  /// An empty place takes the particle 0, as the device does.
  Value framePosition(OpBuilder &at, Value place) {
    Value particle32 = load(at, lists.groups.order, place);
    Value empty = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::slt,
                                        particle32, i32(at, 0));
    Value particle = arith::SelectOp::create(at, loc, empty,
                                             createIndex(at, loc, 0),
                                             toIndex(at, particle32));
    Value packed = load(at, lists.groups.shifts, place);
    SmallVector<Value> cells;
    for (int64_t k = 0; k != 3; ++k) {
      Value bits = arith::AndIOp::create(
          at, loc, arith::ShRUIOp::create(at, loc, packed, i32(at, 10 * k)),
          i32(at, 1023));
      cells.push_back(arith::SIToFPOp::create(
          at, loc, real, arith::SubIOp::create(at, loc, bits, i32(at, 512))));
    }
    Value shift = emitLatticeShift(
        at, loc,
        vector::FromElementsOp::create(at, loc, VectorType::get({3}, real),
                                       cells),
        convertReal(at, loc, box, real));
    Value position = loadElement(at, loc, lists.positions, particle);
    return convertReal(at, loc, arith::AddFOp::create(at, loc, position, shift),
                       computed);
  }

  /// The values of `ins` at a place.
  SmallVector<Value> fieldsAt(OpBuilder &at, Value place) {
    Value particle32 = load(at, lists.groups.order, place);
    Value empty = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::slt,
                                        particle32, i32(at, 0));
    Value particle = arith::SelectOp::create(at, loc, empty,
                                             createIndex(at, loc, 0),
                                             toIndex(at, particle32));
    SmallVector<Value> values;
    for (Value buffer : lists.ins)
      values.push_back(loadElement(at, loc, buffer, particle));
    return values;
  }

  /// Adds `value` to destination `out` at the particle of `place`, if the
  /// place holds one.
  void addAt(OpBuilder &at, unsigned out, Value place, Value value) {
    auto [particle32, holds] = particleAt(at, place);
    scf::IfOp::create(at, loc, holds, [&](OpBuilder &then, Location) {
      Value destination = lists.outs[out];
      Value particle = toIndex(then, particle32);
      Type stored = getElementTypeOrSelf(destination.getType());
      Value old = loadElement(then, loc, destination, particle);
      Value sum = arith::AddFOp::create(
          then, loc, old, convertReal(then, loc, value, stored));
      storeElement(then, loc, sum, destination, particle);
      scf::YieldOp::create(then, loc);
    });
  }

  /// The pair of places `mine` and `other`, whose entry has the shift of
  /// `mask`: the region where it is within the cutoff, its values at both
  /// ends, and the sums. Returns the sums.
  SmallVector<Value> pair(OpBuilder &at, Value mine, Value otherPosition,
                          ValueRange otherValues, Value other,
                          ValueRange sums) {
    Value myPosition = framePosition(at, mine);
    SmallVector<Value> myValues = fieldsAt(at, mine);
    Value d = convertReal(
        at, loc, arith::SubFOp::create(at, loc, myPosition, otherPosition),
        computed);
    Value squares = arith::MulFOp::create(at, loc, d, d);
    Value r2 = vector::ReductionOp::create(at, loc,
                                           vector::CombiningKind::ADD, squares);
    Value within = arith::CmpFOp::create(
        at, loc, arith::CmpFPredicate::OLT, r2,
        createCutoff2(at, loc, computed, op.getCutoff().convertToDouble()));
    auto test = branch(at, ValueRange(sums).getTypes(), within);
    {
      OpBuilder then = OpBuilder::atBlockEnd(test.thenBlock());
      IRMapping inside;
      for (auto [from, to] : lists.captures)
        inside.map(from, to);
      inside.map(kernel.getArgument(0), r2);
      inside.map(kernel.getArgument(1), d);
      for (unsigned i = 0; i != numIns; ++i) {
        inside.map(kernel.getArgument(2 + 2 * i), myValues[i]);
        inside.map(kernel.getArgument(3 + 2 * i), otherValues[i]);
      }
      for (Operation &nested : kernel.without_terminator())
        then.clone(nested, inside);
      SmallVector<Value> next;
      for (unsigned i = 0; i != numYields; ++i) {
        Value value = inside.lookupOrDefault(yield->getOperand(i));
        if (options.magnitudes)
          value = math::AbsFOp::create(then, loc, value);
        if (options.omitted)
          value = createZero(then, loc, value.getType());
        if (i < numOuts) {
          addAt(then, i, mine, value);
          bool antisymmetric =
              op.getExchangeOf(i) == "antisymmetric" && !options.magnitudes;
          addAt(then, i, other,
                antisymmetric ? Value(arith::NegFOp::create(then, loc, value))
                              : value);
          continue;
        }
        next.push_back(
            arith::AddFOp::create(then, loc, sums[i - numOuts], value));
      }
      scf::YieldOp::create(then, loc, next);
      OpBuilder otherwise = OpBuilder::atBlockEnd(test.elseBlock());
      scf::YieldOp::create(otherwise, loc, sums);
    }
    return test.getResults();
  }

  SmallVector<Value> run() {
    // A destination that the op overwrites starts from zero.
    for (auto [index, destination] : llvm::enumerate(lists.outs)) {
      if (!op.overwrites(index))
        continue;
      auto type = cast<MemRefType>(destination.getType());
      Type element = type.getElementType();
      Type value = type.getRank() == 1
                       ? element
                       : VectorType::get({type.getDimSize(1)}, element);
      Value zero = createZero(b, loc, value);
      scf::ForOp::create(
          b, loc, createIndex(b, loc, 0),
          memref::DimOp::create(b, loc, destination, createIndex(b, loc, 0)),
          createIndex(b, loc, 1), ValueRange(),
          [&](OpBuilder &in, Location, Value particle, ValueRange) {
            storeElement(in, loc, zero, destination, particle);
            scf::YieldOp::create(in, loc);
          });
    }
    SmallVector<Value> initial;
    for (unsigned i = numOuts; i != numYields; ++i)
      initial.push_back(
          createZero(b, loc, yield->getOperand(i).getType()));

    Value zero = createIndex(b, loc, 0);
    Value units = lists.units;
    if (lists.dual)
      units = toIndex(b, load(b, lists.activeCount, zero));
    Value entries = lists.dual ? lists.inner.entries : lists.groups.entries;
    Value masks = lists.dual ? lists.inner.masks : lists.groups.masks;
    int64_t width = getEntryShiftBits(isTriclinic(box));
    int64_t offset = getEntryShiftOffset(isTriclinic(box));
    auto overUnits = loop(
        b, zero, units, initial,
        [&](OpBuilder &u, Value index, ValueRange sums) {
          Value unit = index;
          if (lists.dual)
            unit = toIndex(u, load(u, lists.active, index));
          Value group = load(u, lists.groups.units, unit);
          Value first, last;
          if (lists.dual) {
            first = createIndex(u, loc, 0);
            last = toIndex(u, load(u, lists.inner.counts, unit));
          } else {
            Value ordinal = load(u, lists.groups.ordinals, unit);
            Value count = load(u, lists.groups.counts, toIndex(u, group));
            Value begin = arith::MulIOp::create(u, loc, ordinal, i32(u, 64));
            Value end = arith::MinSIOp::create(
                u, loc, count, arith::AddIOp::create(u, loc, begin, i32(u, 64)));
            first = toIndex(u, begin);
            last = toIndex(u, end);
          }
          Value base = arith::MulIOp::create(u, loc, unit,
                                             createIndex(u, loc, 64));
          auto overEntries = loop(
              u, first, last, sums,
              [&](OpBuilder &e, Value entry, ValueRange sums) {
                // Entry e of the list is entry e - first of its block.
                Value at = arith::AddIOp::create(
                    e, loc, arith::SubIOp::create(e, loc, entry, first), base);
                Value other = toIndex(e, load(e, entries, at));
                Value mask = load(e, masks, at);
                // The entry in the frame of the group, by the cells of its
                // mask (D95, D115), in the type of the region.
                Value position = framePosition(e, other);
                Type element = getElementTypeOrSelf(position.getType());
                SmallVector<Value> cells;
                for (int64_t axis = 0; axis != 3; ++axis) {
                  Value bits = arith::AndIOp::create(
                      e, loc,
                      arith::ShRUIOp::create(e, loc, mask,
                                             i32(e, 16 + width * axis)),
                      i32(e, (1 << width) - 1));
                  cells.push_back(arith::SIToFPOp::create(
                      e, loc, element,
                      arith::SubIOp::create(e, loc, bits, i32(e, offset))));
                }
                position = arith::AddFOp::create(
                    e, loc, position,
                    emitLatticeShift(
                        e, loc,
                        vector::FromElementsOp::create(
                            e, loc, position.getType(), cells),
                        convertReal(e, loc, box, element)));
                SmallVector<Value> values = fieldsAt(e, other);
                // The 16 places of the group whose bit is set.
                auto overPlaces = loop(
                    e, createIndex(e, loc, 0), createIndex(e, loc, 16), sums,
                    [&](OpBuilder &p, Value u16, ValueRange sums) {
                      Value u32 = arith::IndexCastOp::create(
                          p, loc, p.getI32Type(), u16);
                      Value bit = arith::AndIOp::create(
                          p, loc, arith::ShRUIOp::create(p, loc, mask, u32),
                          i32(p, 1));
                      Value set = arith::CmpIOp::create(
                          p, loc, arith::CmpIPredicate::ne, bit, i32(p, 0));
                      auto paired =
                          branch(p, ValueRange(sums).getTypes(), set);
                      OpBuilder then = OpBuilder::atBlockEnd(paired.thenBlock());
                      Value mine = toIndex(
                          then, arith::AddIOp::create(
                                    then, loc,
                                    arith::MulIOp::create(then, loc, group,
                                                          i32(then, 16)),
                                    u32));
                      scf::YieldOp::create(
                          then, loc,
                          pair(then, mine, position, values, other, sums));
                      OpBuilder otherwise =
                          OpBuilder::atBlockEnd(paired.elseBlock());
                      scf::YieldOp::create(otherwise, loc, sums);
                      return SmallVector<Value>(paired.getResults());
                    });
                return SmallVector<Value>(overPlaces.getResults());
              });
          return SmallVector<Value>(overEntries.getResults());
        });

    // A weight is that of the sum over both orders of each pair: each pair
    // once counts twice.
    SmallVector<Value> results;
    for (unsigned i = numOuts; i != numYields; ++i) {
      Value total = overUnits.getResult(i - numOuts);
      double weight = 2.0;
      if (auto weights = op.getWeights())
        weight *= (*weights)[i - numOuts];
      Type type = total.getType();
      FloatAttr scalar = b.getFloatAttr(getElementTypeOrSelf(type), weight);
      Value factor;
      if (auto vector = dyn_cast<VectorType>(type))
        factor = arith::ConstantOp::create(
            b, loc, type, DenseElementsAttr::get(vector, scalar.getValue()));
      else
        factor = arith::ConstantOp::create(b, loc, type, scalar);
      total = arith::MulFOp::create(b, loc, factor, total);
      results.push_back(arith::AddFOp::create(
          b, loc, op.getReduce()[i - numOuts], total));
    }
    return results;
  }
};
} // namespace

SmallVector<Value>
kernels::emitTraversePairsSerial(OpBuilder &builder,
                                 md_alg::TraversePairsOp op,
                                 const SerialLists &lists,
                                 SerialOptions options) {
  if (op.isSegmented())
    return emitTraverseSegmentsSerial(builder, op, lists, options);
  return Serial(builder, op, lists, options).run();
}

// Fusion of loops: of loops over the pairs of one neighbor structure, of
// loops over the tuples of one set, and of loops over the particles of one
// set.

#include "mdir/Dialect/MDExec/Transforms/Passes.h"

#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MD/MDTypes.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;

/// Returns true if `a` and `b` run over the same pairs in the same way.
static bool haveSameDomain(PairForOp a, PairForOp b) {
  return a.getNeighbors() == b.getNeighbors() &&
         a.getPositions() == b.getPositions() && a.getCell() == b.getCell() &&
         a.getCutoffAttr() == b.getCutoffAttr() &&
         a.getTraversal() == b.getTraversal() &&
         a.getConflict() == b.getConflict();
}

/// Returns true if `op` or an op nested in it uses a result of `producer`.
static bool uses(Operation *op, Operation *producer) {
  bool found = false;
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      found |= operand.getDefiningOp() == producer;
  });
  return found;
}

/// Returns true if every op that uses a result of `op` comes after `point`,
/// which is in the block of `op`.
static bool usersComeAfter(Operation *op, Operation *point) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      // The op of the block of `op` that holds the user.
      while (user->getBlock() != op->getBlock())
        user = user->getParentOp();
      if (!point->isBeforeInBlock(user))
        return false;
    }
  }
  return true;
}

/// The weights of the global sums of `op`, with 1 where none is given.
static void appendWeights(PairForOp op, SmallVectorImpl<double> &weights) {
  if (auto given = op.getWeights()) {
    weights.append(given->begin(), given->end());
    return;
  }
  weights.append(op.getReduce().size(), 1.0);
}

/// Replaces `first` and `second` with one loop, placed where `second` is.
static void fuse(PairForOp first, PairForOp second) {
  OpBuilder builder(second);
  Location loc = second.getLoc();

  // The fields that the kernels read. A field that both read is read once.
  SmallVector<Value> ins;
  llvm::DenseMap<Value, unsigned> position;
  auto addFields = [&](PairForOp op) {
    for (Value field : op.getIns())
      if (position.try_emplace(field, ins.size()).second)
        ins.push_back(field);
  };
  addFields(first);
  addFields(second);

  SmallVector<Value> outs(first.getOuts().begin(), first.getOuts().end());
  outs.append(second.getOuts().begin(), second.getOuts().end());
  SmallVector<Value> reduce(first.getReduce().begin(),
                            first.getReduce().end());
  reduce.append(second.getReduce().begin(), second.getReduce().end());

  SmallVector<double> weights;
  appendWeights(first, weights);
  appendWeights(second, weights);
  DenseF64ArrayAttr weightsAttr;
  if (llvm::any_of(weights, [](double weight) { return weight != 1.0; }))
    weightsAttr = builder.getDenseF64ArrayAttr(weights);

  SmallVector<Type> resultTypes;
  for (Value value : outs)
    resultTypes.push_back(value.getType());
  for (Value value : reduce)
    resultTypes.push_back(value.getType());

  // The exchange contracts, in the order of the values: the destinations
  // of the first and the second, then their sums.
  ArrayAttr exchangeAttr;
  if (first.getExchange() || second.getExchange()) {
    SmallVector<Attribute> kinds;
    auto contract = [&](PairForOp op, unsigned index) {
      kinds.push_back(md::ExchangeAttr::get(builder.getContext(),
                                            op.getExchange(index)));
    };
    for (unsigned i = 0, e = first.getOuts().size(); i != e; ++i)
      contract(first, i);
    for (unsigned i = 0, e = second.getOuts().size(); i != e; ++i)
      contract(second, i);
    for (unsigned i = 0, e = first.getReduce().size(); i != e; ++i)
      contract(first, first.getOuts().size() + i);
    for (unsigned i = 0, e = second.getReduce().size(); i != e; ++i)
      contract(second, second.getOuts().size() + i);
    exchangeAttr = builder.getArrayAttr(kinds);
  }

  auto fused = PairForOp::create(
      builder, loc, resultTypes, first.getNeighbors(), first.getPositions(),
      first.getCell(), ins, outs, reduce, /*scratch=*/ValueRange(),
      first.getCutoffAttr(), weightsAttr,
      /*overwrite=*/DenseBoolArrayAttr(), first.getTraversalAttr(),
      first.getConflictAttr(), exchangeAttr);

  // The kernel: the first kernel, then the second.
  Block *block = new Block();
  fused.getKernel().push_back(block);
  Block &firstKernel = first.getKernel().front();
  for (unsigned i = 0; i != 2; ++i)
    block->addArgument(firstKernel.getArgument(i).getType(), loc);
  for (Value field : ins) {
    Type type = cast<md::FieldType>(field.getType()).getKernelValueType();
    block->addArgument(type, loc);
    block->addArgument(type, loc);
  }

  OpBuilder kernel(builder.getContext());
  kernel.setInsertionPointToEnd(block);
  SmallVector<Value> yieldedOuts, yieldedSums;
  auto addKernel = [&](PairForOp op) {
    Block &source = op.getKernel().front();
    IRMapping mapping;
    mapping.map(source.getArgument(0), block->getArgument(0));
    mapping.map(source.getArgument(1), block->getArgument(1));
    for (auto [index, field] : llvm::enumerate(op.getIns())) {
      unsigned target = 2 + 2 * position.lookup(field);
      mapping.map(source.getArgument(2 + 2 * index),
                  block->getArgument(target));
      mapping.map(source.getArgument(3 + 2 * index),
                  block->getArgument(target + 1));
    }
    for (Operation &nested : source.without_terminator())
      kernel.clone(nested, mapping);

    Operation *yield = source.getTerminator();
    unsigned numOuts = op.getOuts().size();
    for (unsigned i = 0, e = yield->getNumOperands(); i != e; ++i) {
      Value value = mapping.lookupOrDefault(yield->getOperand(i));
      (i < numOuts ? yieldedOuts : yieldedSums).push_back(value);
    }
  };
  addKernel(first);
  addKernel(second);

  SmallVector<Value> yielded(yieldedOuts);
  yielded.append(yieldedSums);
  YieldOp::create(kernel, loc, yielded);

  // Results: the destinations of the first, those of the second, the sums
  // of the first, those of the second.
  unsigned firstOuts = first.getOuts().size();
  unsigned secondOuts = second.getOuts().size();
  unsigned firstSums = first.getReduce().size();
  unsigned allOuts = firstOuts + secondOuts;
  for (unsigned i = 0, e = first.getNumResults(); i != e; ++i) {
    unsigned target = i < firstOuts ? i : allOuts + (i - firstOuts);
    first.getResult(i).replaceAllUsesWith(fused.getResult(target));
  }
  for (unsigned i = 0, e = second.getNumResults(); i != e; ++i) {
    unsigned target = i < secondOuts
                          ? firstOuts + i
                          : allOuts + firstSums + (i - secondOuts);
    second.getResult(i).replaceAllUsesWith(fused.getResult(target));
  }
  first.erase();
  second.erase();
}

//===----------------------------------------------------------------------===//
// Loops over particles
//===----------------------------------------------------------------------===//

/// The particle set of the fields of `loop`.
static Attribute getParticleSet(ParticleForOp loop) {
  Value field = loop.getIns().empty() ? loop.getOuts().front()
                                      : loop.getIns().front();
  return cast<md::FieldType>(field.getType()).getParticleSet();
}

/// Returns true if `value` is the result of `loop` for one of its
/// destinations.
static bool isFieldOf(Value value, ParticleForOp loop) {
  auto result = dyn_cast<OpResult>(value);
  return result && result.getOwner() == loop.getOperation() &&
         result.getResultNumber() < loop.getOuts().size();
}

/// Returns true if `second` can read what `first` writes within one loop:
/// it uses the results of `first` as fields that it reads, and in no other
/// way.
static bool canFollow(ParticleForOp first, ParticleForOp second) {
  if (first.isStorageForm() || second.isStorageForm() ||
      getParticleSet(first) != getParticleSet(second))
    return false;

  bool follows = true;
  auto isOf = [&](Value value) {
    return value.getDefiningOp() == first.getOperation();
  };
  // A destination or the start of a global sum.
  for (Value value :
       llvm::concat<Value>(second.getOuts(), second.getReduce()))
    follows &= !isOf(value);
  // A global sum of the first loop is complete only when the loop is.
  for (Value value : second.getIns())
    follows &= !isOf(value) || isFieldOf(value, first);
  second.getKernel().walk([&](Operation *nested) {
    for (Value operand : nested->getOperands())
      follows &= !isOf(operand);
  });
  return follows;
}

/// Returns true if every user of a result of `first` other than `second`
/// comes after `second`. The two are in one block.
static bool canMoveDown(ParticleForOp first, ParticleForOp second) {
  for (Value result : first->getResults()) {
    for (Operation *user : result.getUsers()) {
      while (user->getBlock() != first->getBlock())
        user = user->getParentOp();
      if (user != second.getOperation() && !second->isBeforeInBlock(user))
        return false;
    }
  }
  return true;
}

/// Returns true if everything that `second` uses, apart from the results of
/// `first`, is there where `first` is. The two are in one block.
static bool canMoveUp(ParticleForOp first, ParticleForOp second) {
  bool available = true;
  second->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition || definition == first.getOperation() ||
          definition->getBlock() != first->getBlock())
        continue;
      available &= definition->isBeforeInBlock(first);
    }
  });
  return available;
}

/// Replaces `first` and `second` with one loop, placed at `point`, which
/// is one of the two. The kernel of a particle runs the first kernel and
/// then the second, which reads what the first has yielded for that
/// particle.
static void fuse(ParticleForOp first, ParticleForOp second,
                 Operation *point) {
  OpBuilder builder(point);
  Location loc = second.getLoc();
  unsigned firstOuts = first.getOuts().size();

  // A field that only the second loop reads is not stored.
  SmallVector<bool> isKept;
  for (unsigned i = 0; i != firstOuts; ++i)
    isKept.push_back(llvm::any_of(
        first.getResult(i).getUsers(),
        [&](Operation *user) { return user != second.getOperation(); }));

  // The fields that the kernels read. A field that both read is read once.
  SmallVector<Value> ins;
  llvm::DenseMap<Value, unsigned> position;
  auto addFields = [&](ParticleForOp op) {
    for (Value field : op.getIns())
      if (!isFieldOf(field, first) &&
          position.try_emplace(field, ins.size()).second)
        ins.push_back(field);
  };
  addFields(first);
  addFields(second);

  SmallVector<Value> outs;
  for (unsigned i = 0; i != firstOuts; ++i)
    if (isKept[i])
      outs.push_back(first.getOuts()[i]);
  unsigned keptOuts = outs.size();
  outs.append(second.getOuts().begin(), second.getOuts().end());
  SmallVector<Value> reduce(first.getReduce().begin(),
                            first.getReduce().end());
  reduce.append(second.getReduce().begin(), second.getReduce().end());

  SmallVector<Type> resultTypes;
  for (Value value : llvm::concat<Value>(outs, reduce))
    resultTypes.push_back(value.getType());
  auto fused = ParticleForOp::create(builder, loc, resultTypes, ins, outs,
                                     reduce, /*scratch=*/ValueRange());

  Block *block = new Block();
  fused.getKernel().push_back(block);
  for (Value field : ins)
    block->addArgument(
        cast<md::FieldType>(field.getType()).getKernelValueType(), loc);

  OpBuilder kernel(builder.getContext());
  kernel.setInsertionPointToEnd(block);

  // What the first kernel yields for its destinations, in the fused kernel.
  SmallVector<Value> firstYielded;
  SmallVector<Value> yieldedOuts, yieldedSums;
  auto addKernel = [&](ParticleForOp op, bool isFirst) {
    Block &source = op.getKernel().front();
    IRMapping mapping;
    for (auto [index, field] : llvm::enumerate(op.getIns())) {
      Value value;
      if (isFieldOf(field, first))
        value = firstYielded[cast<OpResult>(field).getResultNumber()];
      else
        value = block->getArgument(position.lookup(field));
      mapping.map(source.getArgument(index), value);
    }
    for (Operation &nested : source.without_terminator())
      kernel.clone(nested, mapping);

    Operation *yield = source.getTerminator();
    unsigned numOuts = op.getOuts().size();
    for (unsigned i = 0, e = yield->getNumOperands(); i != e; ++i) {
      Value value = mapping.lookupOrDefault(yield->getOperand(i));
      if (i >= numOuts) {
        yieldedSums.push_back(value);
        continue;
      }
      if (isFirst)
        firstYielded.push_back(value);
      if (!isFirst || isKept[i])
        yieldedOuts.push_back(value);
    }
  };
  addKernel(first, /*isFirst=*/true);
  addKernel(second, /*isFirst=*/false);

  SmallVector<Value> yielded(yieldedOuts);
  yielded.append(yieldedSums);
  YieldOp::create(kernel, loc, yielded);

  // Results: the destinations of the first that are kept, those of the
  // second, the sums of the first, those of the second.
  unsigned secondOuts = second.getOuts().size();
  unsigned allOuts = keptOuts + secondOuts;
  unsigned firstSums = first.getReduce().size();
  unsigned kept = 0;
  for (unsigned i = 0, e = first.getNumResults(); i != e; ++i) {
    if (i >= firstOuts) {
      first.getResult(i).replaceAllUsesWith(
          fused.getResult(allOuts + (i - firstOuts)));
      continue;
    }
    if (isKept[i])
      first.getResult(i).replaceAllUsesWith(fused.getResult(kept++));
  }
  for (unsigned i = 0, e = second.getNumResults(); i != e; ++i) {
    unsigned target = i < secondOuts
                          ? keptOuts + i
                          : allOuts + firstSums + (i - secondOuts);
    second.getResult(i).replaceAllUsesWith(fused.getResult(target));
  }
  // The second loop uses the first, so it goes first.
  second.erase();
  first.erase();
}

/// Fuses two loops over particles of `block`, if two can be fused. Returns
/// true if it did.
static bool fuseParticleLoopsOnce(Block &block) {
  SmallVector<ParticleForOp> loops;
  for (Operation &op : block)
    if (auto loop = dyn_cast<ParticleForOp>(&op))
      loops.push_back(loop);

  for (unsigned i = 0, e = loops.size(); i != e; ++i) {
    for (unsigned j = i + 1; j != e; ++j) {
      ParticleForOp first = loops[i];
      ParticleForOp second = loops[j];
      if (!canFollow(first, second))
        continue;
      if (canMoveDown(first, second)) {
        fuse(first, second, second);
        return true;
      }
      if (canMoveUp(first, second)) {
        fuse(first, second, first);
        return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Returns true if `second` can move to follow `first`, in one block: every
/// value that it or its kernel uses is defined before `first`, inside
/// `second`, or by a pure op without operands (a constant, a field of
/// zeros), which `moved` receives to move along.
static bool canMoveAfter(Operation *second, Operation *first,
                         SmallVectorImpl<Operation *> &moved) {
  bool movable = true;
  second->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition) {
        // A block argument: of a region of `second`, or of an enclosing
        // one, which dominates both.
        continue;
      }
      if (second->isAncestor(definition) ||
          definition->getBlock() != first->getBlock() ||
          definition->isBeforeInBlock(first) || definition == first)
        continue;
      if (definition->getNumOperands() == 0 && isPure(definition) &&
          definition->getNumRegions() == 0) {
        if (!llvm::is_contained(moved, definition))
          moved.push_back(definition);
        continue;
      }
      movable = false;
    }
  });
  return movable;
}

//===----------------------------------------------------------------------===//
// Loops over tuples
//===----------------------------------------------------------------------===//

/// Returns true if `a` and `b` run over the same tuples in the same way.
static bool haveSameDomain(TupleForOp a, TupleForOp b) {
  return a.getIncidence() == b.getIncidence() &&
         a.getPositions() == b.getPositions() && a.getCell() == b.getCell() &&
         a.getArity() == b.getArity() && a.getDisjoint() == b.getDisjoint() &&
         !a.getOverwrite() && !b.getOverwrite();
}

/// Replaces `first` and `second` with one loop, placed where `second` is.
/// The kernel takes the displacements of both, each once, then the values
/// of the fields of the first and of the second, then the parameters of the
/// tuples of the first and of the second; it yields the destinations of
/// the first and of the second, then the sums of the first and of the
/// second.
static void fuse(TupleForOp first, TupleForOp second) {
  OpBuilder builder(second);
  Location loc = second.getLoc();
  int64_t arity = first.getArity();

  SmallVector<md::Coordinate, 4> coordinates(first.getCoordinates());
  auto place = [&](const md::Coordinate &wanted) -> unsigned {
    for (auto [index, known] : llvm::enumerate(coordinates))
      if (known.kind == wanted.kind && known.members == wanted.members)
        return index;
    coordinates.push_back(wanted);
    return coordinates.size() - 1;
  };
  SmallVector<unsigned, 4> secondPlaces;
  for (const md::Coordinate &coordinate : second.getCoordinates())
    secondPlaces.push_back(place(coordinate));
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  md::getCoordinateAttrs(builder, coordinates, kinds, members);

  auto join = [](auto a, auto b) {
    SmallVector<Value> all(a.begin(), a.end());
    all.append(b.begin(), b.end());
    return all;
  };
  SmallVector<Value> ins = join(first.getIns(), second.getIns());
  SmallVector<Value> parameters =
      join(first.getParameters(), second.getParameters());
  SmallVector<Value> outs = join(first.getOuts(), second.getOuts());
  SmallVector<Value> reduce = join(first.getReduce(), second.getReduce());
  SmallVector<Type> resultTypes;
  for (Value value : llvm::concat<Value>(outs, reduce))
    resultTypes.push_back(value.getType());

  auto fused = TupleForOp::create(
      builder, loc, resultTypes, first.getIncidence(), first.getPositions(),
      first.getCell(), ins, parameters, outs, reduce,
      /*scratch=*/ValueRange(), kinds, members, first.getArityAttr(),
      /*overwrite=*/DenseBoolArrayAttr(), first.getDisjointAttr());

  // The arguments of a kernel: its displacements, its values of fields,
  // and its parameters.
  struct Layout {
    unsigned coordinates, values, parameters;
  };
  auto layout = [](TupleForOp op) {
    Block &kernel = op.getKernel().front();
    unsigned coordinates = op.getCoordinates().size();
    unsigned parameters = op.getParameters().size();
    return Layout{coordinates,
                  kernel.getNumArguments() - coordinates - parameters,
                  parameters};
  };
  Layout a = layout(first), b = layout(second);
  Block *block = new Block();
  fused.getKernel().push_back(block);
  Block &firstKernel = first.getKernel().front();
  Block &secondKernel = second.getKernel().front();
  for (unsigned i = 0, e = coordinates.size(); i != e; ++i)
    block->addArgument(VectorType::get({3}, builder.getF64Type()), loc);
  auto addArguments = [&](Block &source, unsigned from, unsigned count) {
    for (unsigned i = 0; i != count; ++i)
      block->addArgument(source.getArgument(from + i).getType(), loc);
  };
  unsigned firstValues = block->getNumArguments();
  addArguments(firstKernel, a.coordinates, a.values);
  unsigned secondValues = block->getNumArguments();
  addArguments(secondKernel, b.coordinates, b.values);
  unsigned firstParameters = block->getNumArguments();
  addArguments(firstKernel, a.coordinates + a.values, a.parameters);
  unsigned secondParameters = block->getNumArguments();
  addArguments(secondKernel, b.coordinates + b.values, b.parameters);
  // The displacements arrive in the type of the kernel.
  for (unsigned i = 0, e = coordinates.size(); i != e; ++i) {
    Type type = i < a.coordinates
                    ? firstKernel.getArgument(i).getType()
                    : secondKernel
                          .getArgument(llvm::find(secondPlaces, i) -
                                       secondPlaces.begin())
                          .getType();
    block->getArgument(i).setType(type);
  }

  OpBuilder kernel(builder.getContext());
  kernel.setInsertionPointToEnd(block);
  SmallVector<Value> yieldedOuts, yieldedSums;
  auto addKernel = [&](TupleForOp op, Layout shape, ArrayRef<unsigned> places,
                       unsigned values, unsigned parameters) {
    Block &source = op.getKernel().front();
    IRMapping mapping;
    for (unsigned i = 0; i != shape.coordinates; ++i)
      mapping.map(source.getArgument(i),
                  block->getArgument(places.empty() ? i : places[i]));
    for (unsigned i = 0; i != shape.values; ++i)
      mapping.map(source.getArgument(shape.coordinates + i),
                  block->getArgument(values + i));
    for (unsigned i = 0; i != shape.parameters; ++i)
      mapping.map(source.getArgument(shape.coordinates + shape.values + i),
                  block->getArgument(parameters + i));
    for (Operation &nested : source.without_terminator())
      kernel.clone(nested, mapping);
    Operation *yield = source.getTerminator();
    unsigned numOuts = op.getOuts().size() * arity;
    for (unsigned i = 0, e = yield->getNumOperands(); i != e; ++i) {
      Value value = mapping.lookupOrDefault(yield->getOperand(i));
      (i < numOuts ? yieldedOuts : yieldedSums).push_back(value);
    }
  };
  addKernel(first, a, {}, firstValues, firstParameters);
  addKernel(second, b, secondPlaces, secondValues, secondParameters);
  SmallVector<Value> yielded(yieldedOuts);
  yielded.append(yieldedSums);
  YieldOp::create(kernel, loc, yielded);

  unsigned firstOuts = first.getOuts().size();
  unsigned secondOuts = second.getOuts().size();
  unsigned firstSums = first.getReduce().size();
  unsigned allOuts = firstOuts + secondOuts;
  for (unsigned i = 0, e = first.getNumResults(); i != e; ++i) {
    unsigned target = i < firstOuts ? i : allOuts + (i - firstOuts);
    first.getResult(i).replaceAllUsesWith(fused.getResult(target));
  }
  for (unsigned i = 0, e = second.getNumResults(); i != e; ++i) {
    unsigned target = i < secondOuts
                          ? firstOuts + i
                          : allOuts + firstSums + (i - secondOuts);
    second.getResult(i).replaceAllUsesWith(fused.getResult(target));
  }
  first.erase();
  second.erase();
}

/// Fuses two loops over tuples of `block`, if two can be fused. Returns
/// true if it did.
static bool fuseTupleLoopsOnce(Block &block) {
  SmallVector<TupleForOp> loops;
  for (Operation &op : block)
    if (auto loop = dyn_cast<TupleForOp>(&op))
      if (!loop.isStorageForm())
        loops.push_back(loop);

  for (unsigned i = 0, e = loops.size(); i != e; ++i) {
    for (unsigned j = i + 1; j != e; ++j) {
      TupleForOp first = loops[i];
      TupleForOp second = loops[j];
      if (!haveSameDomain(first, second) || uses(second, first))
        continue;
      if (!usersComeAfter(first, second)) {
        SmallVector<Operation *> moved;
        if (!canMoveAfter(second, first, moved))
          continue;
        second->moveAfter(first);
        for (Operation *op : moved)
          op->moveBefore(second);
      }
      fuse(first, second);
      return true;
    }
  }
  return false;
}

/// Fuses two loops of `block`, if two can be fused. Returns true if it did.
static bool fuseOnce(Block &block) {
  SmallVector<PairForOp> loops;
  // Fusion relies on the value form: in the storage form, what a loop
  // depends on is not visible in its operands.
  for (Operation &op : block)
    if (auto loop = dyn_cast<PairForOp>(&op))
      if (!loop.isStorageForm())
        loops.push_back(loop);

  for (unsigned i = 0, e = loops.size(); i != e; ++i) {
    for (unsigned j = i + 1; j != e; ++j) {
      PairForOp first = loops[i];
      PairForOp second = loops[j];
      if (!haveSameDomain(first, second) || uses(second, first))
        continue;
      if (!usersComeAfter(first, second)) {
        // Something between the two uses a result of the first, as the sum
        // of the energies of the terms uses the energy of the loop over
        // pairs: the second moves up to follow the first, with the pure
        // ops without operands that it needs, if it can.
        SmallVector<Operation *> moved;
        if (!canMoveAfter(second, first, moved))
          continue;
        second->moveAfter(first);
        for (Operation *op : moved)
          op->moveBefore(second);
      }
      fuse(first, second);
      return true;
    }
  }
  return false;
}

namespace mdir {
namespace md_exec {

#define GEN_PASS_DEF_FUSELOOPS
#include "mdir/Dialect/MDExec/Transforms/Passes.h.inc"

namespace {
class FuseLoops : public impl::FuseLoopsBase<FuseLoops> {
public:
  using impl::FuseLoopsBase<FuseLoops>::FuseLoopsBase;

  void runOnOperation() final {
    SmallVector<Block *> blocks;
    getOperation()->walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          blocks.push_back(&block);
    });
    for (Block *block : blocks) {
      while (fuseOnce(*block))
        ;
      while (fuseTupleLoopsOnce(*block))
        ;
      while (fuseParticleLoopsOnce(*block))
        ;
    }
  }
};
} // namespace

} // namespace md_exec
} // namespace mdir

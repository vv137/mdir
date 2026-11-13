// One generator of the site windows of every segment (D405):
// the site pairs of a segment are derived from its kinds' per-site data,
// and the site geometry stays straight-line arithmetic, visible to
// differentiation and lowering.
#include "KindPairsIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDSegmentWindow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::driver;
using namespace mdir::driver::builder;

llvm::Expected<KindPairsPlan>
mdir::driver::planKindPairs(const Topology &topology,
                            llvm::ArrayRef<const RigidKind *> kinds,
                            llvm::ArrayRef<double> extents) {
  auto refuse = [](const llvm::Twine &reason) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  };
  if (kinds.size() > MaxPairKinds)
    return refuse("the pair segments take at most " + llvm::Twine(MaxPairKinds) +
                  " kinds of rigid groups (D405), not " +
                  llvm::Twine(kinds.size()));
  if (extents.size() != kinds.size())
    return refuse("expected one covering radius for each kind");
  KindPairsPlan plan;
  plan.typeCount = topology.typeNames.size();
  unsigned count = plan.typeCount;
  plan.lennardJonesPairs.assign(count * count, false);
  std::vector<bool> rows(count, false);
  for (unsigned a = 0; a != count; ++a)
    for (unsigned b = 0; b != count; ++b)
      if (topology.epsilon[a * count + b] != 0.0)
        plan.lennardJonesPairs[a * count + b] = rows[a] = true;
  for (auto [index, kind] : llvm::enumerate(kinds)) {
    // Each segment of a taken kind holds whole molecules: no pair of its
    // own particles may be left to the pair term, and no exclusion may
    // remove a pair of the segment (D403's facts).
    if (kind->incompleteExclusions)
      return refuse("kind " + llvm::Twine(index) + " has " +
                    llvm::Twine(kind->incompleteExclusions) +
                    " molecules with a pair of their own that is not excluded");
    if (kind->outsideExclusions)
      return refuse("kind " + llvm::Twine(index) + " has " +
                    llvm::Twine(kind->outsideExclusions) +
                    " molecules with an excluded pair outside the molecule");
    if (!(extents[index] > 0.0))
      return refuse("expected a positive covering radius of kind " + llvm::Twine(index));
    PairKind taken;
    taken.frame = kind->frame;
    taken.sites = kind->sites;
    taken.charges = kind->charges;
    taken.types = kind->types;
    for (unsigned type : kind->types) taken.lennardJones.push_back(rows[type]);
    taken.extent = extents[index];
    plan.kinds.push_back(std::move(taken));
  }
  return plan;
}

std::string mdir::driver::builder::getKindTupleSetName(unsigned f) {
  return "kind_" + std::to_string(f);
}

std::string mdir::driver::builder::getSegmentTupleSetName(unsigned f, unsigned g) {
  return "segment_" + std::to_string(f) + "_" + std::to_string(g);
}

namespace {
/// The frame size of kind f, 1 for the rest.
static unsigned getFrameSize(const KindPairsPlan &plan, unsigned f) {
  return f < plan.kinds.size() ? plan.kinds[f].frame.size() : 1;
}

static mdir::md::Orientation getSegmentOrientation(const KindPairsPlan &plan,
                                                   unsigned f, unsigned g) {
  return f == g && getFrameSize(plan, f) == 1 ? mdir::md::Orientation::Unordered
                                              : mdir::md::Orientation::Ordered;
}

struct WindowBuilder {
  OpBuilder &builder;
  Location loc;
  VectorType vector;
  Type real;

  WindowBuilder(OpBuilder &builder, Value displacement)
      : builder(builder), loc(builder.getUnknownLoc()),
        vector(cast<VectorType>(displacement.getType())),
        real(vector.getElementType()) {}
  Value constant(double value) {
    return arith::ConstantOp::create(builder, loc, FloatAttr::get(real, value));
  }
  Value type(unsigned value) {
    return arith::ConstantIntOp::create(builder, loc, value, 32);
  }
  Value zero() {
    return arith::ConstantOp::create(
        builder, loc, DenseElementsAttr::get(vector, builder.getZeroAttr(real)));
  }
  Value add(Value a, Value b) { return arith::AddFOp::create(builder, loc, a, b); }
  Value sub(Value a, Value b) { return arith::SubFOp::create(builder, loc, a, b); }
  Value scale(double weight, Value value) {
    Value splat = vector::BroadcastOp::create(builder, loc, vector, constant(weight));
    return arith::MulFOp::create(builder, loc, splat, value);
  }

  /// The offset of each place of a molecule from its anchor, x_p - x_0,
  /// from the frame offsets h_k = x_k - x_0, k >= 1: zero for the anchor,
  /// h_k for a frame atom, and for a site with the weights w of R2,
  /// sum_{k >= 1} w_k h_k (the weights sum to 1). Frame offsets of equal
  /// weight are summed before they are scaled, w (h_1 + h_2) for two.
  llvm::SmallVector<Value> offsets(const PairKind &kind, ValueRange frame) {
    llvm::SmallVector<Value> values(kind.getSize());
    for (auto [index, place] : llvm::enumerate(kind.frame))
      values[place] = index ? frame[index - 1] : zero();
    for (const RigidKind::Site &site : kind.sites) {
      Value total;
      std::vector<bool> done(kind.frame.size(), false);
      for (unsigned k = 1; k < kind.frame.size(); ++k) {
        double weight = site.weights[k];
        if (done[k] || weight == 0.0) continue;
        Value sum;
        for (unsigned j = k; j < kind.frame.size(); ++j)
          if (!done[j] && site.weights[j] == weight) {
            done[j] = true;
            sum = sum ? add(sum, frame[j - 1]) : frame[j - 1];
          }
        Value term = scale(weight, sum);
        total = total ? add(total, term) : term;
      }
      values[site.place] = total ? total : zero();
    }
    return values;
  }
};
} // namespace

void mdir::driver::builder::declareKindPairsTupleSets(
    OpBuilder &builder, const KindPairsPlan &plan, StringRef atoms) {
  auto loc = builder.getUnknownLoc();
  unsigned taken = plan.kinds.size();
  for (unsigned f = 0; f != taken; ++f)
    mdir::md::TupleSetOp::create(builder, loc, getKindTupleSetName(f), atoms,
                                 plan.kinds[f].getSize(),
                                 mdir::md::Orientation::Ordered, true);
  for (unsigned f = 0; f <= taken; ++f)
    for (unsigned g = 0; g <= f; ++g)
      mdir::md::TupleSetOp::create(builder, loc, getSegmentTupleSetName(f, g),
                                   atoms, getFrameSize(plan, f) + getFrameSize(plan, g),
                                   getSegmentOrientation(plan, f, g), false);
}

/// The site pairs of segment (f, g) and the energy of the pair term over
/// them. Side a is the molecule of f, side b that of g; the rest, kind K,
/// is the largest number, so it is always side a when it takes part.
static Value buildSegment(OpBuilder &builder, const KindPairsPlan &plan,
                          Value relation, unsigned f, unsigned g,
                          const KindPairsInputs &input, PairEnergy pairEnergy) {
  OpBuilder::InsertionGuard guard(builder);
  auto loc = builder.getUnknownLoc();
  unsigned taken = plan.kinds.size();
  const PairKind *kindA = f < taken ? &plan.kinds[f] : nullptr;
  const PairKind *kindB = g < taken ? &plan.kinds[g] : nullptr;
  unsigned ma = getFrameSize(plan, f), mb = getFrameSize(plan, g);
  unsigned arity = ma + mb, coordinates = arity - 1;
  // Per-particle values are gathered only where the rest takes part; a
  // taken kind's charges and types are constants of the body.
  bool rest = !kindA;
  SmallVector<Value> gathered;
  if (rest) gathered = {input.types, input.charges};
  if (rest && input.alchemical) gathered.push_back(input.alchemical);
  auto sum = mdir::md::SumSegmentWindowOp::create(
      builder, loc, builder.getF64Type(), relation, input.positions, input.cell,
      gathered, builder.getI64IntegerAttr(f), builder.getI64IntegerAttr(g),
      builder.getF64FloatAttr(input.cutoff), UnitAttr());
  Type vector = VectorType::get({3}, builder.getF64Type());
  SmallVector<Type> arguments(coordinates, vector);
  if (rest) {
    arguments.append(arity, builder.getI32Type());
    arguments.append(arity, builder.getF64Type());
    if (input.alchemical) arguments.append(arity, builder.getF64Type());
  }
  SmallVector<Location> locations(arguments.size(), loc);
  Block *block = builder.createBlock(&sum.getKernel(), {}, arguments, locations);
  Value base = block->getArgument(0);
  WindowBuilder window(builder, base);
  auto frameA = block->getArguments().slice(1, ma - 1);
  auto frameB = block->getArguments().slice(ma, mb - 1);
  llvm::SmallVector<Value> offsetsA, offsetsB;
  if (kindA) offsetsA = window.offsets(*kindA, frameA);
  if (kindB) offsetsB = window.offsets(*kindB, frameB);
  // The values of a place: constants for a taken kind, the gathered values
  // of the molecule's member for the rest.
  auto typeOf = [&](const PairKind *kind, unsigned place, unsigned member) {
    return kind ? window.type(kind->types[place])
                : block->getArgument(coordinates + member);
  };
  auto chargeOf = [&](const PairKind *kind, unsigned place, unsigned member) {
    return kind ? window.constant(kind->charges[place])
                : block->getArgument(coordinates + arity + member);
  };
  llvm::SmallVector<KindSitePair> pairs;
  unsigned na = kindA ? kindA->getSize() : 1, nb = kindB ? kindB->getSize() : 1;
  for (unsigned p = 0; p != na; ++p)
    for (unsigned q = 0; q != nb; ++q) {
      // The rule of D405: a site pair is taken when the product
      // of its charges is nonzero or both sites have Lennard-Jones. Where
      // both types are known their entry decides; a value of the rest is
      // known only at run time.
      bool lj, coulomb;
      if (kindA) {
        assert(kindB && "the rest is the last kind, so side b is taken");
        lj = plan.hasLennardJones(kindA->types[p], kindB->types[q]);
        coulomb = kindA->charges[p] != 0.0 && kindB->charges[q] != 0.0;
      } else if (kindB) {
        lj = kindB->lennardJones[q];
        coulomb = kindB->charges[q] != 0.0;
      } else {
        lj = coulomb = true;
      }
      if (!lj && !coulomb)
        continue;
      // Do not reassociate this as base + (offsetA - offsetB): the window
      // arithmetic and its mixed-precision rounding are part of the contract.
      Value displacement = base;
      if (kindA) displacement = window.add(displacement, offsetsA[p]);
      if (kindB) displacement = window.sub(displacement, offsetsB[q]);
      KindSitePair pair;
      pair.displacement = displacement;
      pair.typeI = typeOf(kindA, p, 0);
      pair.typeJ = typeOf(kindB, q, ma);
      pair.chargeI = chargeOf(kindA, p, 0);
      pair.chargeJ = chargeOf(kindB, q, ma);
      pair.lennardJones = lj;
      pair.coulomb = coulomb;
      pairs.push_back(pair);
    }
  SmallVector<Value> energies, displacements;
  Value unperturbed;
  if (input.alchemical)
    unperturbed = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
  for (KindSitePair pair : pairs) {
    pair.lennardJones &= input.lennardJones;
    pair.coulomb &= input.coulomb;
    if (!pair.lennardJones && !pair.coulomb) continue;
    if (input.alchemical) {
      // A taken kind is unperturbed; the rest's values are gathered.
      unsigned first = coordinates + 2 * arity;
      pair.alchemicalI = kindA ? unperturbed : block->getArgument(first);
      pair.alchemicalJ = kindB ? unperturbed : block->getArgument(first + ma);
    }
    energies.push_back(pairEnergy(builder, pair));
    displacements.push_back(pair.displacement);
  }
  if (energies.empty()) {
    builder.setInsertionPoint(sum);
    Value zero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
    sum.erase();
    return zero;
  }
  llvm::append_range(energies, displacements);
  mdir::md::YieldOp::create(builder, loc, energies);
  return sum.getResult();
}

Value mdir::driver::builder::buildKindPairsTerm(OpBuilder &builder,
                                                const KindPairsPlan &plan,
                                                const KindPairsInputs &input,
                                                PairEnergy pairEnergy) {
  auto loc = builder.getUnknownLoc();
  auto *context = builder.getContext();
  auto atoms = cast<mdir::md::FieldType>(input.positions.getType()).getParticleSet();
  unsigned taken = plan.kinds.size();
  assert(taken <= MaxPairKinds && input.kinds.size() == taken);
  SmallVector<Type> segments;
  for (unsigned f = 0; f <= taken; ++f)
    for (unsigned g = 0; g <= f; ++g)
      segments.push_back(mdir::md::RelationType::get(
          context, atoms, getFrameSize(plan, f) + getFrameSize(plan, g),
          getSegmentOrientation(plan, f, g),
          FlatSymbolRefAttr::get(context, getSegmentTupleSetName(f, g))));
  SmallVector<Attribute> descriptors;
  for (const PairKind &kind : plan.kinds) {
    SmallVector<int64_t> frame(kind.frame.begin(), kind.frame.end()), sites;
    SmallVector<double> weights;
    for (const RigidKind::Site &site : kind.sites) {
      sites.push_back(site.place);
      llvm::append_range(weights, site.weights);
    }
    descriptors.push_back(mdir::md::RigidKindAttr::get(
        context, DenseI64ArrayAttr::get(context, frame),
        DenseI64ArrayAttr::get(context, sites),
        DenseF64ArrayAttr::get(context, weights),
        builder.getF64FloatAttr(kind.extent)));
  }
  auto pairs = mdir::md::KindPairsOp::create(
      builder, loc, segments, input.types, input.kinds,
      builder.getArrayAttr(descriptors), input.excluded, input.anchored,
      input.anchored ? builder.getF64FloatAttr(input.radius) : FloatAttr());
  Value total;
  for (unsigned f = 0; f <= taken; ++f)
    for (unsigned g = 0; g <= f; ++g) {
      Value relation = pairs.getResult(mdir::md::getSegmentNumber(f, g));
      Value term = buildSegment(builder, plan, relation, f, g, input, pairEnergy);
      total = total ? arith::AddFOp::create(builder, loc, total, term).getResult() : term;
    }
  return total;
}

Value mdir::driver::builder::cloneKindPairEnergy(OpBuilder &builder, Block &source,
                                                 const KindSitePair &pair) {
  auto loc = builder.getUnknownLoc();
  assert(source.getNumArguments() == (pair.alchemicalI ? 8u : 6u));
  Value squares = arith::MulFOp::create(builder, loc, pair.displacement, pair.displacement);
  Value r2 = vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, squares);
  Value r = math::SqrtOp::create(builder, loc, r2);
  SmallVector<Value> arguments{r, pair.displacement, pair.typeI, pair.typeJ,
                               pair.chargeI, pair.chargeJ};
  if (pair.alchemicalI) arguments.append({pair.alchemicalI, pair.alchemicalJ});
  IRMapping mapping;
  for (auto [old, value] : llvm::zip_equal(source.getArguments(), arguments))
    mapping.map(old, value);
  for (Operation &operation : source.without_terminator()) builder.clone(operation, mapping);
  return mapping.lookupOrDefault(source.getTerminator()->getOperand(0));
}

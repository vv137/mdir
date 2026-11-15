// The regions of the segments of a list of units (D[md-alg-segments]).

#include "mdir/Dialect/MDRT/SegmentRegions.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir::mdrt;

Attribute mdir::mdrt::getSegmentAttr(MLIRContext *context, unsigned f,
                                     unsigned g) {
  return DenseI64ArrayAttr::get(context, {static_cast<int64_t>(f),
                                          static_cast<int64_t>(g)});
}

Attribute mdir::mdrt::getSitesAttr(MLIRContext *context,
                                   ArrayRef<SitePair> pairs) {
  SmallVector<Attribute> entries;
  for (const SitePair &pair : pairs) {
    SmallVector<int64_t> row = {pair.p, pair.q};
    for (unsigned value : pair.values)
      row.push_back(value);
    entries.push_back(DenseI64ArrayAttr::get(context, row));
  }
  return ArrayAttr::get(context, entries);
}

std::pair<unsigned, unsigned> mdir::mdrt::getSegmentKinds(ArrayAttr segments,
                                                          unsigned index) {
  ArrayRef<int64_t> kinds = cast<DenseI64ArrayAttr>(segments[index]).asArrayRef();
  return {static_cast<unsigned>(kinds[0]), static_cast<unsigned>(kinds[1])};
}

SmallVector<SitePair> mdir::mdrt::getSitePairs(ArrayAttr sites,
                                               unsigned index) {
  SmallVector<SitePair> pairs;
  for (Attribute entry : cast<ArrayAttr>(sites[index])) {
    ArrayRef<int64_t> row = cast<DenseI64ArrayAttr>(entry).asArrayRef();
    SitePair pair;
    pair.p = row[0];
    pair.q = row[1];
    for (int64_t value : row.drop_front(2))
      pair.values.push_back(value);
    pairs.push_back(pair);
  }
  return pairs;
}

ParseResult mdir::mdrt::parseSegmentRegions(
    OpAsmParser &parser, ArrayAttr &segments, ArrayAttr &sites,
    SmallVectorImpl<std::unique_ptr<Region>> &regions) {
  MLIRContext *context = parser.getContext();
  SmallVector<Attribute> kinds, tables;
  while (succeeded(parser.parseOptionalKeyword("segment"))) {
    int64_t f, g;
    if (parser.parseLParen() || parser.parseInteger(f) || parser.parseComma() ||
        parser.parseInteger(g) || parser.parseRParen() ||
        parser.parseKeyword("sites"))
      return failure();
    SmallVector<Attribute> rows;
    auto parseRow = [&]() -> ParseResult {
      int64_t p, q;
      if (parser.parseLParen() || parser.parseInteger(p) ||
          parser.parseComma() || parser.parseInteger(q) ||
          parser.parseRParen())
        return failure();
      SmallVector<int64_t> row = {p, q};
      if (parser.parseCommaSeparatedList(
              OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
                int64_t value;
                if (parser.parseInteger(value))
                  return failure();
                row.push_back(value);
                return success();
              }))
        return failure();
      rows.push_back(DenseI64ArrayAttr::get(context, row));
      return success();
    };
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                       parseRow))
      return failure();
    auto region = std::make_unique<Region>();
    if (parser.parseRegion(*region, /*arguments=*/{}))
      return failure();
    kinds.push_back(DenseI64ArrayAttr::get(context, {f, g}));
    tables.push_back(ArrayAttr::get(context, rows));
    regions.push_back(std::move(region));
  }
  segments = ArrayAttr::get(context, kinds);
  sites = ArrayAttr::get(context, tables);
  return success();
}

void mdir::mdrt::printSegmentRegions(OpAsmPrinter &printer, Operation *,
                                     ArrayAttr segments, ArrayAttr sites,
                                     MutableArrayRef<Region> regions) {
  for (unsigned index = 0, e = regions.size(); index != e; ++index) {
    auto [f, g] = getSegmentKinds(segments, index);
    if (index)
      printer << " ";
    printer << "segment(" << f << ", " << g << ") sites [";
    llvm::interleaveComma(getSitePairs(sites, index), printer,
                          [&](const SitePair &pair) {
                            printer << "(" << pair.p << ", " << pair.q
                                    << ") [";
                            llvm::interleaveComma(pair.values, printer);
                            printer << "]";
                          });
    printer << "] ";
    printer.printRegion(regions[index], /*printEntryBlockArgs=*/true,
                        /*printBlockTerminators=*/true);
  }
}

SmallVector<Type> mdir::mdrt::getSegmentArgumentTypes(
    Type real, unsigned pairs, unsigned members, unsigned partners,
    const SegmentSignature &signature) {
  SmallVector<Type> types;
  Type vector = VectorType::get({3}, real);
  for (unsigned m = 0; m != pairs; ++m) {
    types.push_back(real);
    types.push_back(vector);
  }
  for (Type field : signature.fields)
    types.append(members + partners, field);
  return types;
}

SmallVector<Operation *> mdir::mdrt::getSiteSlice(Block &block, unsigned,
                                                  unsigned yieldBegin,
                                                  unsigned yieldEnd) {
  llvm::DenseSet<Operation *> inSlice;
  SmallVector<Value> work;
  Operation *terminator = block.getTerminator();
  for (unsigned i = yieldBegin; i != yieldEnd; ++i)
    work.push_back(terminator->getOperand(i));
  while (!work.empty()) {
    Value value = work.pop_back_val();
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != &block || !inSlice.insert(op).second)
      continue;
    for (Value operand : op->getOperands())
      work.push_back(operand);
    // Values that nested regions use from the block.
    op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        work.push_back(operand);
    });
  }
  SmallVector<Operation *> slice;
  for (Operation &op : block.without_terminator())
    if (inSlice.contains(&op))
      slice.push_back(&op);
  return slice;
}

LogicalResult mdir::mdrt::verifySegmentRegions(
    Operation *op, SegmentedPairsType type, ArrayAttr segments,
    ArrayAttr sites, MutableArrayRef<Region> regions,
    const SegmentSignature &signature) {
  if (segments.size() != regions.size() || sites.size() != regions.size())
    return op->emitOpError() << "expected one segment and one table of site "
                                "pairs for each region";
  if (regions.empty())
    return op->emitOpError() << "expected a region for at least one segment";
  unsigned k = type.getNumKinds();
  unsigned numValues = signature.values.size();
  int64_t previous = -1;
  // V5: one float type T for the site arithmetic of every segment.
  Type common;
  std::pair<int64_t, int64_t> first;
  for (unsigned index = 0, e = regions.size(); index != e; ++index) {
    auto segment = dyn_cast<DenseI64ArrayAttr>(segments[index]);
    if (!segment || segment.size() != 2)
      return op->emitOpError() << "expected the kinds (f, g) of each segment";
    int64_t f = segment[0], g = segment[1];
    // V2: the segments of the list, each once, in increasing s(f, g).
    if (f < 0 || g < 0 || f > k)
      return op->emitOpError()
             << "has segment(" << f << ", " << g
             << "), but the list's kinds are 0 to K = " << k;
    if (g > f)
      return op->emitOpError() << "has segment(" << f << ", " << g
                               << "): its kinds must have g <= f";
    int64_t number = SegmentedPairsType::getSegmentIndex(f, g);
    if (number <= previous)
      return op->emitOpError()
             << "has segment(" << f << ", " << g << ") after segment "
             << previous << ": each segment once, in increasing s(f, g)";
    previous = number;

    // V3 and V4: the table and its masks.
    unsigned nf = type.getNumPlaces(f), ng = type.getNumPlaces(g);
    auto table = dyn_cast<ArrayAttr>(sites[index]);
    if (!table || table.empty())
      return op->emitOpError() << "expected at least one site pair in the "
                                  "table of segment("
                               << f << ", " << g << ")";
    for (Attribute entry : table) {
      auto row = dyn_cast<DenseI64ArrayAttr>(entry);
      if (!row || row.size() < 2)
        return op->emitOpError() << "expected each site pair as (p, q) and "
                                    "its values";
    }
    SmallVector<SitePair> pairs = getSitePairs(sites, index);
    llvm::DenseSet<std::pair<unsigned, unsigned>> seen;
    for (const SitePair &pair : pairs) {
      if (pair.p >= nf || pair.q >= ng)
        return op->emitOpError()
               << "has the site pair (" << pair.p << ", " << pair.q
               << ") in segment(" << f << ", " << g << "), whose units have "
               << nf << " and " << ng << " places";
      if (!seen.insert({pair.p, pair.q}).second)
        return op->emitOpError() << "has the site pair (" << pair.p << ", "
                                 << pair.q << ") twice in segment(" << f
                                 << ", " << g << ")";
      if (pair.values.empty())
        return op->emitOpError()
               << "has the site pair (" << pair.p << ", " << pair.q
               << ") with no value: a pair that yields nothing is not "
                  "in the table";
      for (unsigned i = 0, n = pair.values.size(); i != n; ++i) {
        if (pair.values[i] >= numValues)
          return op->emitOpError()
                 << "has value " << pair.values[i] << " in the mask of ("
                 << pair.p << ", " << pair.q << "), but the op has "
                 << numValues << " values";
        if (i && pair.values[i] <= pair.values[i - 1])
          return op->emitOpError()
                 << "expected the mask of (" << pair.p << ", " << pair.q
                 << ") in increasing order";
      }
    }

    // V5: the arguments.
    Region &region = regions[index];
    if (!region.hasOneBlock())
      return op->emitOpError() << "expected one block in the region of "
                                  "segment("
                               << f << ", " << g << ")";
    Block &block = region.front();
    unsigned M = pairs.size();
    unsigned mf = type.getNumMembers(f), mg = type.getNumMembers(g);
    Type real = block.getNumArguments() ? block.getArgument(0).getType()
                                        : Type();
    if (!real || !isa<FloatType>(real))
      return op->emitOpError() << "expected the region of segment(" << f
                               << ", " << g
                               << ") to take r^2 first, of a float type";
    if (!common) {
      common = real;
      first = {f, g};
    } else if (real != common) {
      return op->emitOpError()
             << "takes r^2 of segment(" << f << ", " << g << ") in " << real
             << " and of segment(" << first.first << ", " << first.second
             << ") in " << common
             << ": the site arithmetic of every segment is in one type";
    }
    SmallVector<Type> expected =
        getSegmentArgumentTypes(real, M, mf, mg, signature);
    if (block.getArgumentTypes() != ArrayRef<Type>(expected)) {
      InFlightDiagnostic diagnostic =
          op->emitOpError()
          << "expected the region of segment(" << f << ", " << g
          << ") to take r^2 and the displacement for each of its " << M
          << " site pairs, in one type, then " << mf << " + " << mg
          << " values for each field of 'ins': (";
      llvm::interleaveComma(expected, diagnostic);
      return diagnostic << ")";
    }

    // V6: the yields.
    Operation *terminator = block.getTerminator();
    if (!terminator || terminator->getName().getStringRef() !=
                           signature.terminator)
      return op->emitOpError() << "ends the region of segment(" << f << ", "
                               << g << ") in '" << signature.terminator << "'";
    SmallVector<Type> yielded;
    for (const SitePair &pair : pairs)
      for (unsigned value : pair.values)
        yielded.push_back(signature.values[value]);
    if (terminator->getOperandTypes() != ArrayRef<Type>(yielded)) {
      InFlightDiagnostic diagnostic =
          op->emitOpError() << "expected the region of segment(" << f << ", "
                            << g
                            << ") to yield the values of each mask, site "
                               "pair by site pair: (";
      llvm::interleaveComma(yielded, diagnostic);
      return diagnostic << ")";
    }

    // V7: the slice of a site pair reads no argument of another.
    unsigned begin = 0;
    for (unsigned m = 0; m != M; ++m) {
      unsigned end = begin + pairs[m].values.size();
      auto foreign = [&](Value value) {
        auto argument = dyn_cast<BlockArgument>(value);
        return argument && argument.getOwner() == &block &&
               argument.getArgNumber() < 2 * M &&
               argument.getArgNumber() / 2 != m;
      };
      for (unsigned i = begin; i != end; ++i)
        if (foreign(terminator->getOperand(i)))
          return op->emitOpError()
                 << "yields for the site pair (" << pairs[m].p << ", "
                 << pairs[m].q << ") of segment(" << f << ", " << g
                 << ") a value of another site pair";
      for (Operation *sliced : getSiteSlice(block, m, begin, end)) {
        bool reads = false;
        sliced->walk([&](Operation *nested) {
          for (Value operand : nested->getOperands())
            reads |= foreign(operand);
        });
        if (reads)
          return sliced->emitOpError()
                 << "reads, in the slice of the site pair (" << pairs[m].p
                 << ", " << pairs[m].q << ") of segment(" << f << ", " << g
                 << "), the distance of another site pair: the slices of "
                    "the site pairs must be separable";
      }
      begin = end;
    }
  }
  return success();
}

// The strategies of a scatter-accumulate and their preconditions
// (D[privatize-scatter]); see include/mdir/Dialect/MDExec/Scatter.h.

#include "mdir/Dialect/MDExec/Scatter.h"

#include "mdir/Dialect/MD/MDTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;
using namespace mdir::md_exec::scatter;

static std::string describe(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

Refusal scatter::checkReorder(Type element, bool deterministic) {
  if (!element.isF32() && !element.isF64())
    return "the values are of " + describe(element) +
           ", not of f32 or f64, which are the types privatized";
  if (deterministic)
    return std::string("the deterministic mode adds floating-point values "
                       "in a fixed order, and a sum of a block first would "
                       "round otherwise");
  return std::nullopt;
}

bool scatter::isTileEdge(int64_t edge) {
  return edge == 4 || edge == 8 || edge == 16;
}

int64_t scatter::getTileThreads(int64_t edge) { return edge == 16 ? 512 : 128; }

int64_t scatter::getTileSide(int64_t edge, int64_t order) {
  return edge + order - 1;
}

int64_t scatter::getTileBytes(int64_t edge, int64_t order, Type element) {
  int64_t threads = getTileThreads(edge);
  int64_t side = getTileSide(edge, order);
  int64_t real = element.isF32() ? 4 : 8;
  return (side * side * side + 4 * threads) * real + 3 * threads * 4;
}

Refusal scatter::checkTile(int64_t edge, int64_t order, ArrayRef<int64_t> grid,
                           Type element, bool deterministic) {
  if (Refusal refused = checkReorder(element, deterministic))
    return refused;
  if (!isTileEdge(edge))
    return "the template has no kernels of bins of " + std::to_string(edge) +
           " points, only of 4, 8, and 16";
  int64_t bytes = getTileBytes(edge, order, element);
  if (bytes > kBlockMemoryBytes)
    return "a tile with the particles of a round takes " +
           std::to_string(bytes) + " bytes of block memory, more than " +
           std::to_string(kBlockMemoryBytes);
  int64_t side = getTileSide(edge, order);
  for (int64_t length : grid)
    if (length < side)
      return "an edge of the grid of " + std::to_string(length) +
             " points is shorter than a tile of " + std::to_string(side) +
             ", whose points would wrap around it more than once";
  return std::nullopt;
}

Refusal scatter::checkTile(ReciprocalOp op, int64_t edge, bool deterministic) {
  // The grid is in the type of the forces (D271): the field of the value
  // form, or its buffer in the storage form.
  Type element;
  if (op.isStorageForm()) {
    if (op.getOut())
      element = cast<MemRefType>(op.getOut().getType()).getElementType();
  } else if (op.getForces()) {
    if (auto forces = dyn_cast<md::FieldType>(op.getForces().getType()))
      element = forces.getElementType();
  }
  if (!element)
    return std::string("the sum yields no forces, whose type the grid takes");
  return checkTile(edge, op.getOrder(), op.getGrid(), element, deterministic);
}

int64_t scatter::getHashSlots(int64_t blockSize, int64_t arity) {
  return static_cast<int64_t>(llvm::PowerOf2Ceil(blockSize * arity));
}

Refusal scatter::getHashDestination(Type fieldOrBuffer, Destination &out) {
  if (auto field = dyn_cast<md::FieldType>(fieldOrBuffer)) {
    out.element = field.getElementType();
    out.components = field.getNumComponents();
  } else if (auto buffer = dyn_cast<MemRefType>(fieldOrBuffer)) {
    out.element = buffer.getElementType();
    out.components = buffer.getRank() == 1 ? 1 : buffer.getDimSize(1);
    if (buffer.getRank() > 2 || ShapedType::isDynamic(out.components))
      return "a destination of the type " + describe(fieldOrBuffer) +
             " has no atomic of the device to fall back on";
  } else {
    return "a destination of the type " + describe(fieldOrBuffer) +
           " is neither a field nor a buffer";
  }
  if (!out.element.isF32() && !out.element.isF64())
    return "a destination of " + describe(out.element) +
           " has no atomic of the device to fall back on";
  return std::nullopt;
}

int64_t scatter::getHashBytes(int64_t slots, ArrayRef<Destination> outs) {
  int64_t bytes = 4 * slots;
  for (const Destination &out : outs)
    bytes += slots * out.components * out.element.getIntOrFloatBitWidth() / 8;
  return bytes;
}

Refusal scatter::checkHash(int64_t slots, TypeRange outs, bool deterministic,
                           SmallVectorImpl<Destination> *destinations) {
  if (outs.empty())
    return std::string("the loop adds to no field");
  SmallVector<Destination> found;
  for (Type type : outs) {
    Destination destination;
    if (Refusal refused = getHashDestination(type, destination))
      return refused;
    if (Refusal refused = checkReorder(destination.element, deterministic))
      return refused;
    found.push_back(destination);
  }
  int64_t bytes = getHashBytes(slots, found);
  if (bytes > kBlockMemoryBytes)
    return "a table of " + std::to_string(slots) + " slots takes " +
           std::to_string(bytes) + " bytes of block memory, more than " +
           std::to_string(kBlockMemoryBytes);
  if (destinations)
    destinations->append(found.begin(), found.end());
  return std::nullopt;
}

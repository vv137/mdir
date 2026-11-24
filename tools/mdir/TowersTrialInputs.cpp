#include "TowersTrialInputs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mdir::driver;
namespace {
template <typename T> void append(std::vector<char> &buffer, T value) {
  auto p = reinterpret_cast<const char *>(&value);
  buffer.insert(buffer.end(), p, p + sizeof(T));
}

Value hostBuffer(Value device) {
  if (auto cast = device.getDefiningOp<memref::CastOp>())
    device = cast.getSource();
  for (auto copy : device.getUsers())
    if (auto transfer = dyn_cast<gpu::MemcpyOp>(copy))
      if (transfer.getDst() == device)
        return transfer.getSrc();
  return {};
}

// The storage pass builds folded type tables with straight-line pointwise
// expressions inside two loops. Evaluate their scalar SSA at one table cell
// with the dialect folders so the trial takes the program's type-pair values.
struct TableCell {
  const Program &program;
  unsigned firstTable;
  int64_t row, col;
  Attribute read(Value mem) const {
    if (auto arg = dyn_cast<BlockArgument>(mem)) {
      unsigned i = arg.getArgNumber() - firstTable;
      if (i >= program.tables.size())
        return {};
      const auto &table = program.tables[i];
      if (row >= table.count || col >= table.getColumns())
        return {};
      return FloatAttr::get(Float64Type::get(mem.getContext()),
                            table.values[row * table.getColumns() + col]);
    }
    for (auto *user : mem.getUsers())
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        auto loop = store->getParentOfType<scf::ForOp>();
        auto outer = loop ? loop->getParentOfType<scf::ForOp>() : scf::ForOp();
        if (!outer || store.getMemRef() != mem ||
            !llvm::equal(
                store.getIndices(),
                ValueRange{outer.getInductionVar(), loop.getInductionVar()}))
          return {};
        return evaluate(store.getValue(), store.getIndices());
      }
    return {};
  }
  Attribute evaluate(Value value, ValueRange indices) const {
    Attribute c;
    if (matchPattern(value, m_Constant(&c)))
      return c;
    if (auto load = value.getDefiningOp<memref::LoadOp>())
      return llvm::equal(load.getIndices(), indices) ? read(load.getMemRef())
                                                     : Attribute();
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || op->getNumRegions())
      return {};
    SmallVector<Attribute> operands;
    for (Value operand : op->getOperands()) {
      auto a = evaluate(operand, indices);
      if (!a)
        return {};
      operands.push_back(a);
    }
    if (auto trunc = dyn_cast<arith::TruncFOp>(op);
        trunc && !trunc.getRoundingmode()) {
      // The dialect folder may decline inexact truncation without a rounding
      // mode. Match the host table initializer's nearest-even conversion.
      if (auto type = dyn_cast<FloatType>(value.getType())) {
        if (auto scalar = dyn_cast<FloatAttr>(operands.front())) {
          APFloat rounded = scalar.getValue();
          bool losesInfo;
          rounded.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                          &losesInfo);
          return FloatAttr::get(type, rounded);
        }
      }
      auto type = dyn_cast<VectorType>(value.getType());
      auto vector = dyn_cast<DenseFPElementsAttr>(operands.front());
      if (type && type.getElementType().isF32() && vector) {
        SmallVector<float> elements;
        for (auto x : vector.getValues<APFloat>())
          elements.push_back(static_cast<float>(x.convertToDouble()));
        return DenseFPElementsAttr::get(type, ArrayRef<float>(elements));
      }
    }
    SmallVector<OpFoldResult> folded;
    if (failed(op->fold(operands, folded)) || folded.size() != 1)
      return {};
    if (auto a = dyn_cast<Attribute>(folded.front()))
      return a;
    auto it = llvm::find(op->getOperands(), cast<Value>(folded.front()));
    return it == op->operand_end() ? Attribute()
                                   : operands[it - op->operand_begin()];
  }
};
} // namespace

MemRefType mdir::tool::getTowersTableType(const Program &program, Value table) {
  auto type = dyn_cast<MemRefType>(table.getType());
  if (!type || type.getRank() != 2)
    return {};
  if (type.hasStaticShape())
    return type;
  auto alloc = table.getDefiningOp<gpu::AllocOp>();
  if (!alloc)
    return {};
  SmallVector<int64_t, 2> shape(type.getShape());
  unsigned firstTable = 3 + program.takesForces + program.fields.size();
  for (auto [axis, extent] : llvm::enumerate(shape)) {
    if (!ShapedType::isDynamic(extent))
      continue;
    Value size = alloc.getDynamicSizes()[type.getDynamicDimIndex(axis)];
    if (auto constant = getConstantIntValue(size)) {
      extent = *constant;
      continue;
    }
    // The canonicalized initializer (including a folded or narrowed copy)
    // takes this extent from the original host table argument.
    auto dim = size.getDefiningOp<memref::DimOp>();
    if (!dim)
      return {};
    auto arg = dyn_cast<BlockArgument>(dim.getSource());
    auto index = dim.getConstantIndex();
    if (!arg || !arg.getOwner()->isEntryBlock() ||
        !isa<func::FuncOp>(arg.getOwner()->getParentOp()) || !index ||
        (*index != 0 && *index != 1) ||
        arg.getArgNumber() - firstTable >= program.tables.size())
      return {};
    const auto &backing = program.tables[arg.getArgNumber() - firstTable];
    extent = *index == 0 ? backing.count : backing.getColumns();
  }
  return MemRefType::get(shape, type.getElementType(), type.getLayout(),
                         type.getMemorySpace());
}

llvm::Expected<mdir::tool::TowersBuffers>
mdir::tool::bindTowersTrialInputs(const Program &program, ValueRange fields,
                                  llvm::ArrayRef<Value> captures) {
  TowersBuffers buffers;
  auto fail = [](const char *what) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "towers_shape: could not bind trial %s",
                                   what);
  };
  unsigned firstField = 3 + program.takesForces;
  buffers.fields = fields.size();
  for (Value field : fields) {
    auto arg = dyn_cast_if_present<BlockArgument>(hostBuffer(field));
    if (!arg || arg.getArgNumber() - firstField >= program.fields.size())
      return fail("particle field");
    auto &array = buffers.values.emplace_back();
    const auto &input = program.fields[arg.getArgNumber() - firstField];
    for (double x : input.values)
      if (input.isInteger)
        append(array, int32_t(x));
      else if (cast<MemRefType>(field.getType()).getElementType().isF32())
        append(array, float(x));
      else
        append(array, x);
  }
  for (Value v : captures) {
    if (matchPattern(v, m_Constant()))
      continue;
    auto &array = buffers.values.emplace_back();
    if (auto global = v.getDefiningOp<memref::GetGlobalOp>()) {
      auto def = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          global, global.getNameAttr());
      auto attr = def ? def.getInitialValue() : std::nullopt;
      auto values =
          attr ? dyn_cast<DenseFPElementsAttr>(*attr) : DenseFPElementsAttr();
      if (!values || !values.getElementType().isF32())
        return fail("constant table");
      for (float x : values.getValues<float>())
        append(array, x);
      continue;
    }
    auto type = dyn_cast<MemRefType>(v.getType());
    Value host = hostBuffer(v);
    if (!host || !type || !type.hasStaticShape() || type.getRank() != 2)
      return fail("type table");
    for (int64_t i = 0; i < type.getDimSize(0); ++i)
      for (int64_t j = 0; j < type.getDimSize(1); ++j) {
        auto cell =
            TableCell{program, unsigned(firstField + program.fields.size()), i,
                      j}
                .read(host);
        if (auto scalar = dyn_cast_if_present<FloatAttr>(cell))
          append(array, float(scalar.getValueAsDouble()));
        else if (auto vec = dyn_cast_if_present<DenseFPElementsAttr>(cell))
          for (auto x : vec.getValues<APFloat>())
            append(array, float(x.convertToDouble()));
        else
          return fail("folded type-table element");
      }
  }
  return buffers;
}

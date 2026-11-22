// Typed builders of the reciprocal sum on a device [Essmann1995, Liang2026].
#pragma once
#include "../MDExecKernels/PMECode.h"
#include "../MDExecKernels/ShuffleLane.h"
#include "mdir/Dialect/MDAlg/MDAlgOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mdir::kernels {
using namespace mlir;
enum class Window { Spline, Prolate };
enum class Splitting { Gaussian, Dispersion, P3M, Radial };
class DevicePME : public detail::PMECode {
public:
  DevicePME(OpBuilder &builder, ModuleOp module, Type position, Type charge,
            Type force, int64_t order, int64_t lanes)
      : PMECode(builder, position, charge, force), order(order), lanes(lanes) {
    b.setInsertionPointToEnd(module.getBody());
  }
  template <bool tilted>
  void emitSplineStages(bool potential, bool listed, int64_t tiles);
  void emitProlate(bool listed, int64_t tiles);

private:
  template <Window window> void emitTiles(int64_t tiles, bool listed);
  int64_t order, lanes;
  int64_t realBytes = realType.getIntOrFloatBitWidth() / 8;
  Type tokenType = gpu::AsyncTokenType::get(b.getContext());
  Type globalPointer = LLVM::LLVMPointerType::get(b.getContext(), 1);
  MemRefType globalMem(ArrayRef<int64_t> shape, Type element) {
    return MemRefType::get(shape, element, MemRefLayoutAttrInterface(),
                           b.getI64IntegerAttr(1));
  }
  MemRefType shared(int64_t count, Type element) {
    return MemRefType::get({count}, element, MemRefLayoutAttrInterface(),
                           gpu::AddressSpaceAttr::get(
                               b.getContext(), gpu::AddressSpace::Workgroup));
  }
  int64_t tileSide(int edge) { return edge + order - 1; }
  int64_t tilePoints(int edge) {
    int64_t n = tileSide(edge);
    return n * n * n;
  }
  MemRefType tileType(int edge) { return shared(tilePoints(edge), realType); }
  Value laneValue(gpu::ShuffleMode mode, Value v, Value offset, Value width) {
    return detail::shuffleLane(b, loc, mode, v, offset, width).first;
  }
  Value sumBlock(Value value) {
    auto add = gpu::AllReduceOperationAttr::get(b.getContext(),
                                                gpu::AllReduceOperation::ADD);
    return gpu::AllReduceOp::create(b, loc, value, add, true);
  }
  // The f32 reduction maps to one PTX instruction. LLVM's atomicrmw
  // for this type can expand to a compare-and-swap loop (D84).
  void addGlobal(Value pointer, Value value) {
    if (realType.isF64()) {
      LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::fadd, pointer, value,
                                LLVM::AtomicOrdering::monotonic, "device");
    } else {
      LLVM::InlineAsmOp::create(b, loc, TypeRange{}, ValueRange{pointer, value},
                                "red.relaxed.gpu.global.add.f32 [$0], $1;",
                                "l,f", true, false, LLVM::TailCallKind::None,
                                LLVM::AsmDialectAttr{}, ArrayAttr{});
    }
  }
  // The scope of the tile is one block (D282).
  void addShared(Value tile, Value at, Value value) {
    Value base = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, tile);
    Value size = arith::ConstantIndexOp::create(b, loc, realBytes);
    Value offset = muli(at, size);
    Value address = addi(base, offset);
    Value integer = indexCast(i64Type, address);
    Value pointer = LLVM::IntToPtrOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext(), 3), integer);
    LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::fadd, pointer, value,
                              LLVM::AtomicOrdering::monotonic, "block");
  }
  struct LaneWindow {
    Value weight, slope, axis, zero, one;
  };
  LaneWindow laneWindow(Value lane, Value i, Value places, Value tables,
                        Value n_all, Value pe, Value c0w, Value two_i,
                        Value three_i, bool slopes);
  Value placementStart(Value places, Value n, Value i, int axis);
  void splineWeights(Value fraction, Value pointCount, Value weights,
                     Value slopes = {});
  std::pair<Value, Value> placeAxis(Value coordinate, Value inverseLength,
                                    Value size, Value pointCount);
  Value pickSpline(Value array, Value atIndex, Value pointCount);
  Value clenshaw(Value base, Value fraction, Value one, Value terms,
                 Value points, Value j, Value tables);
  Value waveNumber64(Value k, Value count, Value inverseLength, Value cc2);
  struct Influence {
    Value factor;
    // Gaussian uses the inverse squared wave number. The radial and
    // dispersion formulas supply the full derivative factor.
    Value virialTerm;
    Value atOrigin;
  };
  Influence radialInfluence(Value tables, Value pe, Value cc1, Value msqr,
                            Value rone, Value t1, Value t2, Value t3);
  Influence dispersionInfluence(Value msqr, Value pibr, Value sqrtpir,
                                Value rone, Value t1, Value t2, Value t3,
                                Value sixgaussr);
  Influence gaussianInfluence(Value msqr, Value rzero, Value rone, Value t1,
                              Value t2, Value t3);
  Influence gaussianInfluenceTriclinic(Value msqr, Value rzero, Value rone,
                                       Value t1, Value t2, Value t3,
                                       Value gaussr);
  void emitBlocks(StringRef symbol);
  template <bool tilted, bool useList, bool fixedGrid>
  void emitSpread(StringRef symbol);
  template <bool tilted> void emitWeights(StringRef symbol);
  template <bool useList> void emitSpreadBricks(StringRef symbol);
  void emitReal(StringRef symbol);
  template <Splitting splitting, bool tilted> void emitTables(StringRef symbol);
  template <Splitting splitting, bool tilted>
  void emitConvolve(StringRef symbol);
  template <Splitting splitting, bool tilted> void emitScale(StringRef symbol);
  template <bool tilted, bool useList, bool potentialOnly, bool cachedWeights>
  void emitGather(StringRef symbol);
  template <bool useList, int tileEdge> void emitBin(StringRef symbol);
  template <Window window, int tileEdge> void emitSpreadTile(StringRef symbol);
  void emitCoefficients(StringRef symbol);
  template <bool tilted> void emitPlaces(StringRef symbol);
  template <bool useList, bool fixedGrid> void emitWarpSpread(StringRef symbol);
  template <bool tilted, bool useList, bool potentialOnly>
  void emitWarpGather(StringRef symbol);
};
} // namespace mdir::kernels

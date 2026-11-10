#include "mdir/Conversion/TemplateRefreshKernels.h"
#include "mdir/Conversion/TemplateSearchKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir::kernels;

std::array<Value, 2> mdir::kernels::emitRefreshTemplatePairArena(
    OpBuilder &b, Location loc, ModuleOp module, const TemplateRefreshInputs &v,
    llvm::function_ref<void(OpBuilder &, Value, Value)> prepare) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto wide = [&](OpBuilder &at, int64_t n) -> Value {
    return arith::ConstantIntOp::create(at, loc, n, 64);
  };
  auto boolean = [&](OpBuilder &at, bool yes) -> Value {
    return arith::ConstantOp::create(at, loc, at.getBoolAttr(yes));
  };
  auto device = MemRefType::get({ShapedType::kDynamic}, b.getI32Type(),
      MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
  auto controls = MemRefType::get({4}, b.getI64Type(),
      MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
  auto call = [&](OpBuilder &at, StringRef name, ValueRange args, Type result,
                  bool ciface) -> Value {
    auto type = at.getFunctionType(args.getTypes(), TypeRange{result});
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (!callee) {
      OpBuilder symbols(module.getBodyRegion());
      callee = func::FuncOp::create(symbols, loc, name, type);
      callee.setPrivate();
    }
    assert(callee.getFunctionType() == type && "template runtime declaration mismatch");
    if (ciface) callee->setAttr("llvm.emit_c_interface", at.getUnitAttr());
    return func::CallOp::create(at, loc, callee, args).getResult(0);
  };
  auto layout = [&](OpBuilder &at, int64_t slot) -> Value {
    Value n = call(at, "mdrtTemplatePairsLayoutValue",
        ValueRange{v.handle, wide(at, slot)}, at.getI64Type(), false);
    return arith::IndexCastOp::create(at, loc, at.getIndexType(), n);
  };
  auto view = [&](OpBuilder &at, int64_t slot) -> Value {
    return call(at, "mdrtTemplatePairsArenaView",
        ValueRange{v.handle, wide(at, slot)}, device, true);
  };
  Value host = memref::AllocaOp::create(b, loc, MemRefType::get({4}, b.getI64Type()));
  // A row's sparse append publishes the exact required total even when
  // staging is too small. E entries take at most ceil(E/64)+rows chunks,
  // so growth from that total must fit the next complete build.
  auto attempts = scf::ForOp::create(b, loc, ix(b, 0), ix(b, 2), ix(b, 1),
      ValueRange{boolean(b, false), v.waterActive, v.atomActive},
      [&](OpBuilder &at, Location, Value attempt, ValueRange state) {
    Value pending = arith::XOrIOp::create(at, loc, state[0], boolean(at, true));
    auto run = scf::IfOp::create(at, loc, pending,
        [&](OpBuilder &in, Location) {
      prepare(in, state[1], state[2]);
      TemplateSearchBuffers s;
      s.cell = v.cell; s.baseReach = v.baseReach; s.extent = v.extent;
      s.reaches = v.reaches;
      s.classCells = v.classCells; s.classReaches = v.classReaches;
      auto boxes = MemRefType::get({ShapedType::kDynamic, 3}, in.getF64Type(),
          MemRefLayoutAttrInterface(), in.getI64IntegerAttr(1));
      auto searchBoxes = MemRefType::get({ShapedType::kDynamic, 3}, in.getF32Type(),
          MemRefLayoutAttrInterface(), in.getI64IntegerAttr(1));
      s.groupBoxes = call(in, "mdrtTemplatePairsGroupBoxesF32",
          ValueRange{v.handle}, searchBoxes, true);
      s.fractionalBoxes = call(in, "mdrtTemplatePairsFrameState",
          ValueRange{v.handle, wide(in, TEMPLATE_FRAME_FRACTIONAL_BOXES)}, boxes, true);
      auto &p = s.spatial;
      p.points = v.points;
      p.columnsX = v.columnsX; p.columnsY = v.columnsY;
      p.waterActive = state[1]; p.atomActive = state[2];
      p.waters = layout(in, TEMPLATE_LAYOUT_WATERS);
      p.atoms = layout(in, TEMPLATE_LAYOUT_ATOMS);
      p.scanBlocks = layout(in, TEMPLATE_LAYOUT_SCAN_BLOCKS);
      p.keys = view(in, TEMPLATE_LAYOUT_SPATIAL_KEYS);
      p.zKeys = view(in, TEMPLATE_LAYOUT_SPATIAL_Z_KEYS);
      p.items = view(in, TEMPLATE_LAYOUT_SPATIAL_ITEMS);
      p.places = view(in, TEMPLATE_LAYOUT_SPATIAL_PLACES);
      p.counts = view(in, TEMPLATE_LAYOUT_SPATIAL_COUNTS);
      p.starts = view(in, TEMPLATE_LAYOUT_SPATIAL_STARTS);
      p.cursors = view(in, TEMPLATE_LAYOUT_SPATIAL_CURSORS);
      p.scanScratch = view(in, TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH);
      p.order = call(in, "mdrtTemplatePairsOrder", ValueRange{v.handle}, device, true);
      auto host3 = MemRefType::get({3}, in.getF64Type());
      Value diagnosticCell = memref::AllocaOp::create(in, loc, host3);
      Value diagnosticReaches = memref::AllocaOp::create(in, loc, host3);
      for (int64_t axis = 0; axis != 3; ++axis) {
        memref::StoreOp::create(in, loc, vector::ExtractOp::create(in, loc, v.cell, axis),
            diagnosticCell, ValueRange{ix(in, axis)});
        memref::StoreOp::create(in, loc, vector::ExtractOp::create(in, loc, v.reaches, axis),
            diagnosticReaches, ValueRange{ix(in, axis)});
      }
      SmallVector<Value> diagnosticArgs{v.handle, diagnosticCell, diagnosticReaches,
          arith::ExtUIOp::create(in, loc, in.getI64Type(), state[1]),
          arith::ExtUIOp::create(in, loc, in.getI64Type(), state[2])};
      Value diagnosticRows = call(in, "mdrtPrototypePrepare", diagnosticArgs, in.getI64Type(), true);
      TemplateGroupBuffers groups;
      groups.rows = arith::IndexCastOp::create(in, loc, in.getIndexType(), diagnosticRows);
      groups.waterGroups = layout(in, TEMPLATE_LAYOUT_WATER_GROUPS);
      groups.atomGroups = layout(in, TEMPLATE_LAYOUT_ATOM_GROUPS);
      groups.starts = call(in,"mdrtTemplatePairsGroupView",ValueRange{v.handle,wide(in,TEMPLATE_GROUP_STARTS)},device,true);
      groups.lengths = call(in,"mdrtTemplatePairsGroupView",ValueRange{v.handle,wide(in,TEMPLATE_GROUP_LENGTHS)},device,true);
      groups.columns = call(in,"mdrtTemplatePairsGroupView",ValueRange{v.handle,wide(in,TEMPLATE_GROUP_COLUMNS)},device,true);
      groups.atStart = call(in,"mdrtTemplatePairsGroupView",ValueRange{v.handle,wide(in,TEMPLATE_GROUP_AT_START)},device,true);
      p.scanScratch = view(in,TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH);
      p.scanBlocks = layout(in,TEMPLATE_LAYOUT_SCAN_BLOCKS);
      Value bins = arith::MulIOp::create(in, loc, ix(in, 2),
          arith::MulIOp::create(in, loc, p.columnsX, p.columnsY));
      Value fitsBins = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ule,
          bins, layout(in, TEMPLATE_LAYOUT_SPATIAL_BINS));
      cf::AssertOp::create(in, loc, fitsBins, "template spatial grid exceeds its arena budget");
      auto &a = s.build;
      a.groups = groups;
      a.arena = call(in, "mdrtTemplatePairsBuffer", ValueRange{v.handle}, device, true);
      a.stageUnits = view(in, TEMPLATE_LAYOUT_STAGE_UNITS);
      a.stageEntries = view(in, TEMPLATE_LAYOUT_STAGE_ENTRIES);
      a.counts = view(in, TEMPLATE_LAYOUT_BUILD_COUNTS);
      a.entryOffsets = view(in, TEMPLATE_LAYOUT_BUILD_ENTRY_OFFSETS);
      a.workCounts = view(in, TEMPLATE_LAYOUT_BUILD_WORK_COUNTS);
      a.workOffsets = view(in, TEMPLATE_LAYOUT_BUILD_WORK_OFFSETS);
      a.scanScratch = p.scanScratch;
      a.outerRows = view(in, TEMPLATE_LAYOUT_OUTER_ROWS);
      a.innerRows = view(in, TEMPLATE_LAYOUT_INNER_ROWS);
      a.outerUnits = view(in, TEMPLATE_LAYOUT_OUTER_UNITS);
      a.innerUnits = view(in, TEMPLATE_LAYOUT_INNER_UNITS);
      a.outerEntries = view(in, TEMPLATE_LAYOUT_OUTER_ENTRIES);
      a.innerEntries = view(in, TEMPLATE_LAYOUT_INNER_ENTRIES);
      a.waters = p.waters; a.atoms = p.atoms;
      a.rows = groups.rows;
      a.unitCapacity = layout(in, TEMPLATE_LAYOUT_UNITS);
      a.scanBlocks = p.scanBlocks;
      s.counters = call(in, "mdrtTemplatePairsCounters", ValueRange{v.handle}, controls, true);
      s.excludedRows = call(in, "mdrtTemplatePairsMetadata",
          ValueRange{v.handle, wide(in, TEMPLATE_SS_ROWS)}, device, true);
      s.excludedNeighbors = call(in, "mdrtTemplatePairsMetadata",
          ValueRange{v.handle, wide(in, TEMPLATE_SS_NEIGHBORS)}, device, true);
      // Never-merged diagnostic: retain native grouping/epochs and replace
      // candidate generation with the unchanged archived CUDA kernels.
      SmallVector<Value> searchArgs{v.handle};
      auto diagnosticType = in.getFunctionType(ValueRange(searchArgs).getTypes(), {});
      auto diagnostic = module.lookupSymbol<func::FuncOp>("mdrtPrototypeSearch");
      if (!diagnostic) {
        OpBuilder symbols(module.getBodyRegion());
        diagnostic = func::FuncOp::create(symbols, loc, "mdrtPrototypeSearch", diagnosticType);
        diagnostic.setPrivate(); diagnostic->setAttr("llvm.emit_c_interface", in.getUnitAttr());
      }
      func::CallOp::create(in, loc, diagnostic, searchArgs);
      Type token = gpu::AsyncTokenType::get(in.getContext());
      Value ready = gpu::WaitOp::create(in, loc, token, ValueRange()).getAsyncToken();
      Value copied = gpu::MemcpyOp::create(in, loc, token, ValueRange{ready},
          host, s.counters).getAsyncToken();
      gpu::WaitOp::create(in, loc, Type(), ValueRange{copied});
      Value flags = memref::LoadOp::create(in, loc, host, ValueRange{ix(in, TEMPLATE_COUNTER_FLAGS)});
      Value entries = memref::LoadOp::create(in, loc, host, ValueRange{ix(in, TEMPLATE_COUNTER_ENTRIES)});
      Value capacity = call(in, "mdrtTemplatePairsCapacity", ValueRange{v.handle}, in.getI64Type(), false);
      Value noIndexError = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq,
          arith::AndIOp::create(in, loc, flags, wide(in, ~int64_t(TEMPLATE_BUILD_CAPACITY_OVERFLOW))),
          wide(in, 0));
      cf::AssertOp::create(in, loc, noIndexError, "template search exceeds its index or image domain");
      Value withinIndex = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ule,
          entries, wide(in, INT32_MAX));
      cf::AssertOp::create(in, loc, withinIndex, "template search exceeds the i32 entry domain");
      Value overflow = arith::OrIOp::create(in, loc,
          arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ugt, entries, capacity),
          arith::CmpIOp::create(in, loc, arith::CmpIPredicate::ne, flags, wide(in, 0)));
      auto result = scf::IfOp::create(in, loc, overflow,
          [&](OpBuilder &grow, Location) {
        Value first = arith::CmpIOp::create(grow, loc, arith::CmpIPredicate::eq,
            attempt, ix(grow, 0));
        cf::AssertOp::create(grow, loc, first,
            "template pair arena overflowed again after exact sizing");
        call(grow, "mdrtTemplatePairsGrow", ValueRange{v.handle, entries}, grow.getI64Type(), false);
        scf::YieldOp::create(grow, loc,
            ValueRange{boolean(grow, false), boolean(grow, true), boolean(grow, true)});
      }, [&](OpBuilder &compact, Location) {
        emitCompactTemplatePairs(compact, loc, a);
        scf::YieldOp::create(compact, loc, ValueRange{boolean(compact, true), state[1], state[2]});
      });
      scf::YieldOp::create(in, loc, result.getResults());
    }, [&](OpBuilder &skip, Location) {
      scf::YieldOp::create(skip, loc, state);
    });
    scf::YieldOp::create(at, loc, run.getResults());
  });
  cf::AssertOp::create(b, loc, attempts.getResult(0),
      "template pair arena overflowed again after exact sizing");
  return {attempts.getResult(1), attempts.getResult(2)};
}

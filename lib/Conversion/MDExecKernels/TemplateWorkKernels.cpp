// The physical pair-arena adapter for the source-derived template body.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mdir::kernels;

SmallVector<Value, 2> mdir::kernels::emitTemplateWorkUnit(
    OpBuilder &b, Location loc, mdir::md_exec::TemplateForOp loop,
    const TemplateWorkBuffers &buffers, Value work, Value lane,
    const IRMapping &captures) {
  auto word = [&](OpBuilder &at, int64_t value) -> Value {
    return arith::ConstantIntOp::create(at, loc, value, 32);
  };
  auto index = [&](OpBuilder &at, Value value) -> Value {
    if (value.getType().isIndex()) return value;
    // Padding between the two classes can put a place just above 2^31,
    // even though every original unit ID fits a nonnegative signed i32.
    // Places are unsigned payload indices; -1 membership sentinels have
    // already been replaced with a safe unit before they reach this cast.
    return arith::IndexCastUIOp::create(at, loc, at.getIndexType(), value);
  };
  auto record = [&](OpBuilder &at, Value offset, Value row, unsigned column) -> Value {
    Value address = arith::AddIOp::create(at, loc, offset,
        arith::AddIOp::create(at, loc,
            arith::MulIOp::create(at, loc, index(at, row), createIndex(at, loc, 4)),
            createIndex(at, loc, column)));
    return memref::LoadOp::create(at, loc, buffers.pairs, ValueRange{address});
  };
  Value tag = record(b, buffers.workOffset, work, 0);
  Value firstBegin = record(b, buffers.workOffset, work, 1);
  Value firstLength = memref::LoadOp::create(b, loc, buffers.groupLengths,
      ValueRange{index(b, firstBegin)});
  Value begin = record(b, buffers.workOffset, work, 2);
  Value end = record(b, buffers.workOffset, work, 3);
  SmallVector<Type> sumTypes;
  for (Value value : loop.getReduce()) sumTypes.push_back(value.getType());

  // Each work record belongs to one class; all branches are warp uniform.
  auto emitClass = [&](OpBuilder &at, unsigned kind) -> SmallVector<Value, 2> {
    constexpr unsigned firstSizes[] = {3, 1, 1}, secondSizes[] = {3, 3, 1};
    unsigned firstSize = firstSizes[kind], secondSize = secondSizes[kind];
    unsigned arity = firstSize + secondSize;
    Block &source = loop->getRegion(kind).front();
    Value local = arith::AndIOp::create(at, loc, lane, word(at, 15));
    Value inSpan = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, local, firstLength);
    Value limit = arith::SubIOp::create(at, loc,
        arith::MaxUIOp::create(at, loc, firstLength, word(at, 1)), word(at, 1));
    Value place = arith::AddIOp::create(at, loc, firstBegin,
        arith::MinUIOp::create(at, loc, local, limit));
    Value first = memref::LoadOp::create(at, loc, buffers.order,
                                         ValueRange{index(at, place)});
    Value valid = arith::AndIOp::create(at, loc, inSpan,
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::sge, first, word(at, 0)));
    Value safeFirst = arith::SelectOp::create(at, loc, valid, first, word(at, 0));
    auto rowOf = [&](OpBuilder &body, Value unit) -> Value {
      Value water = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ult,
                                           unit, buffers.waters);
      Value waterRow = arith::MulIOp::create(body, loc, unit, word(body, 3));
      Value atomRow = arith::AddIOp::create(body, loc, unit,
          arith::MulIOp::create(body, loc, buffers.waters, word(body, 2)));
      return arith::SelectOp::create(body, loc, water, waterRow, atomRow);
    };
    auto memberOf = [&](OpBuilder &body, Value unit, unsigned site) -> Value {
      Value row = arith::AddIOp::create(body, loc, rowOf(body, unit), word(body, site));
      return index(body, memref::LoadOp::create(body, loc, buffers.members,
                                                ValueRange{index(body, row)}));
    };
    auto loadFrame = [&](OpBuilder &body, Value unit, unsigned count) {
      SmallVector<Value, 3> result;
      Value firstRow = rowOf(body, unit);
      for (unsigned site = 0; site != count; ++site) {
        Value row = arith::AddIOp::create(body, loc, firstRow, word(body, site));
        result.push_back(loadElement(body, loc, buffers.frames, index(body, row)));
      }
      return result;
    };
    auto loadParameters = [&](OpBuilder &body, Value unit, unsigned count) {
      SmallVector<Value, 3> result;
      for (Value input : loop.getIns())
        for (unsigned site = 0; site != count; ++site)
          result.push_back(loadElement(body, loc, captures.lookupOrDefault(input),
                                        memberOf(body, unit, site)));
      return result;
    };
    // A table lookup at constant types (a water-water LJ coefficient, for
    // example) and the pure arithmetic that only it, constants, and captures
    // feed are the same for every pair of the class: take them once per
    // unit, before the rotation, instead of under every turn's guard. A
    // lookup indexed by loaded parameters or by the geometry stays in the
    // pair body, and so does anything with a region or a memory effect.
    IRMapping invariant = captures;
    {
      llvm::DenseSet<Value> hoisted;
      Region *body = source.getParent();
      auto available = [&](Value value) {
        return hoisted.contains(value) || !body->isAncestor(value.getParentRegion());
      };
      for (Operation &op : source.without_terminator()) {
        if (op.getNumRegions() || !llvm::all_of(op.getOperands(), available)) continue;
        if (op.getNumOperands() == 0 && matchPattern(&op, m_Constant())) {
          at.clone(op, invariant);
          hoisted.insert(op.result_begin(), op.result_end());
          continue;
        }
        bool lookup = false;
        if (auto table = dyn_cast<mdir::md::LookupOp>(op))
          lookup = llvm::all_of(table.getIndices(),
              [](Value index) { return matchPattern(index, m_Constant()); });
        bool arithmetic = isa<arith::ArithDialect, math::MathDialect>(op.getDialect()) &&
                          isPure(&op) && !matchPattern(&op, m_Constant());
        bool fed = llvm::any_of(op.getOperands(),
            [&](Value value) { return hoisted.contains(value); });
        if (!lookup && !(arithmetic && fed)) continue;
        at.clone(op, invariant);
        hoisted.insert(op.result_begin(), op.result_end());
      }
    }
    TemplateUnit unit{static_cast<TemplatePairClass>(kind), lane, begin, end, valid};
    unit.first = loadFrame(at, safeFirst, firstSize);
    unit.parameters = loadParameters(at, safeFirst, firstSize);
    unit.forces = !loop.getOuts().empty();
    unit.hasSite = buffers.hasSite;
    unit.alpha = loop.getAlpha().convertToDouble();
    unit.siteForces = unit.forces && mdir::md::getTemplateForceArity(
        loop, arity, 0, loop.getOuts().size()) != arity;
    llvm::append_range(unit.sumTypes, sumTypes);
    auto write = [&](OpBuilder &body, Value unit, ValueRange values) {
      if (values.empty()) return;
      Value destination = captures.lookupOrDefault(loop.getOuts()[0]);
      Type stored = cast<MemRefType>(destination.getType()).getElementType();
      for (auto [site, value] : llvm::enumerate(values))
        emitAtomicAdd(body, loc, convertReal(body, loc, value, stored),
                       destination, memberOf(body, unit, site));
    };
    return emitTemplateClassUnit(at, loc, unit,
        [&](OpBuilder &body, Value entry) {
          TemplatePartner partner;
          Value other = record(body, buffers.entryOffset, entry, 0);
          partner.frame = loadFrame(body, other, secondSize);
          partner.parameters = loadParameters(body, other, secondSize);
          partner.mask = record(body, buffers.entryOffset, entry, 2);
          Value image = record(body, buffers.entryOffset, entry, 1);
          SmallVector<Value, 3> cells;
          for (unsigned axis = 0; axis != 3; ++axis) {
            Value component = arith::AndIOp::create(body, loc,
                arith::ShRUIOp::create(body, loc, image,
                    word(body, MDRT_TEMPLATE_IMAGE_BITS * axis)),
                word(body, MDRT_TEMPLATE_IMAGE_MASK));
            component = arith::SubIOp::create(body, loc, component,
                word(body, MDRT_TEMPLATE_IMAGE_OFFSET));
            cells.push_back(arith::SIToFPOp::create(body, loc, body.getF32Type(), component));
          }
          Value lattice = vector::FromElementsOp::create(body, loc,
              VectorType::get({3}, body.getF32Type()), cells);
          partner.shift = emitLatticeShift(body, loc, lattice,
              convertReal(body, loc, buffers.cell, body.getF32Type()));
          return partner;
        }, [&](OpBuilder &body, ValueRange coordinates, ValueRange firstParameters,
                ValueRange secondParameters, ValueRange sites) {
          IRMapping mapping = invariant;
          unsigned coordinateCount = kind == 0 ? 5 : kind == 1 ? 3 : 1;
          mapping.map(source.getArguments().take_front(coordinateCount), coordinates);
          for (unsigned input = 0; input != loop.getIns().size(); ++input) {
            unsigned offset = coordinateCount + input * arity;
            mapping.map(source.getArguments().slice(offset, firstSize),
                        firstParameters.slice(input * firstSize, firstSize));
            mapping.map(source.getArguments().slice(offset + firstSize, secondSize),
                        secondParameters.slice(input * secondSize, secondSize));
          }
          mapTemplateSiteOffsets(loop, kind, sites, mapping);
          return emitTemplateWindowContributions(body, loc, loop, kind, mapping);
        }, [&](OpBuilder &body, ValueRange values) { write(body, safeFirst, values); },
        [&](OpBuilder &body, Value entry, ValueRange values) {
          write(body, record(body, buffers.entryOffset, entry, 0), values);
        });
  };
  Value water = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, tag, word(b, 0));
  auto chosen = scf::IfOp::create(b, loc, water,
      [&](OpBuilder &body, Location) { scf::YieldOp::create(body, loc, emitClass(body, 0)); },
      [&](OpBuilder &body, Location) {
        Value mixed = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq, tag, word(body, 1));
        auto other = scf::IfOp::create(body, loc, mixed,
            [&](OpBuilder &inner, Location) { scf::YieldOp::create(inner, loc, emitClass(inner, 1)); },
            [&](OpBuilder &inner, Location) { scf::YieldOp::create(inner, loc, emitClass(inner, 2)); });
        scf::YieldOp::create(body, loc, other.getResults());
      });
  return SmallVector<Value, 2>(chosen.getResults());
}

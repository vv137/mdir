// Warp-cooperative spreading of the shifted prolate window [Liang2026].
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool useList, bool fixedGrid>
void DevicePME::emitWarpSpread(StringRef symbol) {
  StringRef blockFunction = "mdrt_gpu_pme_prolate_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  inputs.append(
      2, globalMem({ShapedType::kDynamic}, realType)); // places and tables
  if constexpr (fixedGrid)
    inputs.push_back(globalMem({ShapedType::kDynamic}, i64Type));
  else
    inputs.append(
        2, globalMem({ShapedType::kDynamic}, realType)); // bricks and mesh
  inputs.append(3, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value q = entry->getArgument(arg++);
  Value list;
  if constexpr (useList) {
    list = entry->getArgument(arg++);
  }
  Value places = entry->getArgument(arg++);
  Value tables = entry->getArgument(arg++);
  Value bricks;
  Value realBuffer;
  if constexpr (!fixedGrid) {
    bricks = entry->getArgument(arg++);
    realBuffer = entry->getArgument(arg++);
  }
  Value grid;
  if constexpr (fixedGrid) {
    grid = entry->getArgument(arg++);
  }
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value c16 = arith::ConstantIndexOp::create(b, loc, 16);
  Value c32 = arith::ConstantIndexOp::create(b, loc, 32);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value places_length = memref::DimOp::create(b, loc, places, c0);
  Value c6 = arith::ConstantIndexOp::create(b, loc, 6);
  Value n_all = divui(places_length, c6);
  Value k12 = addi(k1, k2);
  Value k123 = addi(k12, k3);
  Value pe = muli(k123, c3);
  Value warps = addi(n_all, c3);
  Value blocks = divui(warps, c4);
  Value b2;
  Value b12;
  if constexpr (!fixedGrid) {
    Value k1u = addi(k1, c3);
    Value b1 = divui(k1u, c4);
    Value k2u = addi(k2, c3);
    b2 = divui(k2u, c4);
    b12 = muli(b1, b2);
  }
  Value points_grid;
  Value blocks_grid;
  if constexpr (fixedGrid) {
    points_grid = memref::DimOp::create(b, loc, grid, c0);
    blocks_grid =
        func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                             ValueRange{points_grid})
            .getResult(0);
  }
  auto spreadParticle = [&](Value lane, Value i, Value c0w, Value c32w,
                            Value two_i, Value three_i) {
    auto [wvalue, wslope, axis, zero_w, one] = laneWindow(
        lane, i, places, tables, n_all, pe, c0w, two_i, three_i, false);
    Value qs = load(q, {i});
    Value qi = convert(realType, qs);
    Value first_axis = cmpi(arith::CmpIPredicate::eq, axis, c0w);
    Value qv = mulf(wvalue, qi);
    Value weight = select(first_axis, qv, wvalue);
    Value xstart = placementStart(places, n_all, i, 0);
    Value ystart = placementStart(places, n_all, i, 1);
    Value zstart = placementStart(places, n_all, i, 2);
    Value k1t = indexCast(i32Type, k1);
    Value k2t = indexCast(i32Type, k2);
    Value k3t = indexCast(i32Type, k3);
    Value pt =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
    Value ppt = muli(pt, pt);
    Value pppt = muli(ppt, pt);
    Value thirty_two =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value zero32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
    Value width =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value passes_n =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
    Value passes_n2 = muli(passes_n, passes_n);
    Value passes_n3 = muli(passes_n2, passes_n);
    Value c31p = arith::ConstantIndexOp::create(b, loc, 31);
    Value passes_u = addi(passes_n3, c31p);
    Value passes = divui(passes_u, c32w);
    Value b2t, four32, sixteen, bytes, bbase;
    Value scale, eight, gridbase;
    if constexpr (!fixedGrid) {
      b2t = indexCast(i32Type, b2);
      four32 = arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      sixteen =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 16));
      bytes = arith::ConstantOp::create(b, loc,
                                        b.getIntegerAttr(i64Type, realBytes));
      Value bindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bricks);
      bbase = indexCast(i64Type, bindex);
    } else {
      scale = arith::ConstantOp::create(
          b, loc, b.getFloatAttr(realType, 1099511627776.0));
      eight = arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 8));
      Value gridindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, grid);
      gridbase = indexCast(i64Type, gridindex);
    }
    scf::ForOp::create(
        b, loc, c0w, passes, one, ValueRange{},
        [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
          Value pass32 = indexCast(i32Type, pass);
          Value pass_base = muli(pass32, thirty_two);
          Value kk = addi(pass_base, lane);
          Value in = cmpi(arith::CmpIPredicate::slt, kk, pppt);
          Value kc = select(in, kk, zero32);
          Value pc = divui(kc, ppt);
          Value pab = remui(kc, ppt);
          Value pb = divui(pab, pt);
          Value pa = remui(pab, pt);
          Value wx = laneValue(gpu::ShuffleMode::IDX, weight, pa, width);

          Value pbs = addi(pb, pt);
          Value wy = laneValue(gpu::ShuffleMode::IDX, weight, pbs, width);

          Value pt2 = addi(pt, pt);
          Value pcs = addi(pc, pt2);
          Value wz = laneValue(gpu::ShuffleMode::IDX, weight, pcs, width);

          auto inBranch =
              scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(inBranch.thenBlock());
            Value g1s = addi(xstart, pa);
            Value g1w = subi(g1s, k1t);
            Value g1over = cmpi(arith::CmpIPredicate::sge, g1s, k1t);
            Value g1 = select(g1over, g1w, g1s);
            Value g2s = addi(ystart, pb);
            Value g2w = subi(g2s, k2t);
            Value g2over = cmpi(arith::CmpIPredicate::sge, g2s, k2t);
            Value g2 = select(g2over, g2w, g2s);
            Value g3s = addi(zstart, pc);
            Value g3w = subi(g3s, k3t);
            Value g3over = cmpi(arith::CmpIPredicate::sge, g3s, k3t);
            Value g3 = select(g3over, g3w, g3s);
            Value wxy = mulf(wx, wy);
            Value contribution = mulf(wxy, wz);
            if constexpr (!fixedGrid) {
              Value bx1 = divui(g1, four32);
              Value by1 = divui(g2, four32);
              Value brow = muli(bx1, b2t);
              Value bxy = addi(brow, by1);
              Value bxyz = muli(bxy, k3t);
              Value slab = addi(bxyz, g3);
              Value slab16 = muli(slab, sixteen);
              Value inx = remui(g1, four32);
              Value iny = remui(g2, four32);
              Value iny4 = muli(iny, four32);
              Value inxy = addi(iny4, inx);
              Value atb = addi(slab16, inxy);
              Value ati = arith::ExtUIOp::create(b, loc, i64Type, atb);
              Value offset = muli(ati, bytes);
              Value address = addi(bbase, offset);
              Value pointer =
                  LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
              Value value = addf(contribution, zero_w);
              addGlobal(pointer, value);
            } else {
              Value scaled = mulf(contribution, scale);
              Value rounded = math::RoundEvenOp::create(b, loc, scaled);
              Value fixedv = fptosi(i64Type, rounded);
              Value row1 = muli(g1, k2t);
              Value row = addi(row1, g2);
              Value row64 = arith::ExtUIOp::create(b, loc, i64Type, row);
              Value k3w = arith::ExtUIOp::create(b, loc, i64Type, k3t);
              Value gbase = muli(row64, k3w);
              Value g364 = arith::ExtUIOp::create(b, loc, i64Type, g3);
              Value at = addi(gbase, g364);
              Value offset = muli(at, eight);
              Value address = addi(gridbase, offset);
              Value pointer =
                  LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
              LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::add, pointer,
                                        fixedv, LLVM::AtomicOrdering::monotonic,
                                        "device");
            }
            scf::YieldOp::create(b, loc);
          }
          scf::YieldOp::create(b, loc);
        });
  };
  {
    gpu::LaunchOp launch;
    if constexpr (!fixedGrid) {
      launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    } else {
      launch = gpu::LaunchOp::create(b, loc, blocks_grid, c1, c1, c128, c1, c1);
    }
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;

    Value tx = launch.getThreadIds().x;

    Value c0w;
    Value c32w;
    Value two_i;
    Value three_i;
    Value warp;
    Value lane;
    Value wb;
    if constexpr (!fixedGrid) {
      c0w = arith::ConstantIndexOp::create(b, loc, 0);
      c32w = arith::ConstantIndexOp::create(b, loc, 32);
      two_i = arith::ConstantIndexOp::create(b, loc, 2);
      three_i = arith::ConstantIndexOp::create(b, loc, 3);
      Value four_w = arith::ConstantIndexOp::create(b, loc, 4);
      warp = divui(tx, c32w);
      Value lane_x = remui(tx, c32w);
      lane = indexCast(i32Type, lane_x);
      wb = muli(bx, four_w);
    }
    Value i;
    Value inside;
    if constexpr (!useList && !fixedGrid) {
      i = addi(wb, warp);
      inside = cmpi(arith::CmpIPredicate::ult, i, n_all);
    }
    Value item;
    if constexpr (fixedGrid) {
      Value base = muli(bx, c128);
      item = addi(base, tx);
      inside = cmpi(arith::CmpIPredicate::ult, item, points_grid);
    }
    Value slot;
    if constexpr (useList && !fixedGrid) {
      slot = addi(wb, warp);
      inside = cmpi(arith::CmpIPredicate::ult, slot, n_all);
    }
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      scf::IfOp is_listedBranch;
      if constexpr (useList && !fixedGrid) {
        Value listed = load(list, {slot});
        Value listed_none =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value is_listed = cmpi(arith::CmpIPredicate::sge, listed, listed_none);
        is_listedBranch =
            scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, false);
        b.setInsertionPointToStart(is_listedBranch.thenBlock());
        i = indexCast(idxType, listed);
      }
      if constexpr (!fixedGrid) {
        spreadParticle(lane, i, c0w, c32w, two_i, three_i);
      } else {
        Value none =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 0));
        store(none, grid, {item});
      }
      scf::YieldOp::create(b, loc);
      if constexpr (useList && !fixedGrid) {
        b.setInsertionPointAfter(is_listedBranch);
        scf::YieldOp::create(b, loc);
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  Value c17;
  Value zchunks;
  Value copy_blocks;
  if constexpr (!fixedGrid) {
    arith::ConstantIndexOp::create(b, loc, 8);
    c17 = arith::ConstantIndexOp::create(b, loc, 17);
    Value c31 = arith::ConstantIndexOp::create(b, loc, 31);
    Value k3_31 = addi(k3, c31);
    zchunks = divui(k3_31, c32);
    copy_blocks = muli(b12, zchunks);
  }
  {
    gpu::LaunchOp launch;
    if constexpr (!fixedGrid) {
      launch = gpu::LaunchOp::create(b, loc, copy_blocks, c1, c1, c128, c1, c1,
                                     Value{}, Type{}, ValueRange{},
                                     TypeRange{shared(544, realType)});
    } else {
      launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    }
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    if constexpr (!fixedGrid) {
      Value tile = launch.getWorkgroupAttributionBBArgs()[0];
      Value bxy = divui(bx, zchunks);
      Value zc = remui(bx, zchunks);
      Value z0 = muli(zc, c32);
      Value bx1 = divui(bxy, b2);
      Value by1 = remui(bxy, b2);
      Value slab0 = muli(bxy, k3);
      Value zero_r =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value bytes = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(i64Type, realBytes));
      Value bindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bricks);
      Value bbase = indexCast(i64Type, bindex);
      scf::ForOp::create(
          b, loc, c0, c4, c1, ValueRange{},
          [&](OpBuilder &b, Location, Value j, ValueRange carried) {
            Value j128 = muli(j, c128);
            Value e = addi(j128, tx);
            Value zz = divui(e, c16);
            Value inxy = remui(e, c16);
            Value z = addi(z0, zz);
            Value in = cmpi(arith::CmpIPredicate::ult, z, k3);
            auto inBranch2 =
                scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(inBranch2.thenBlock());
              Value slab = addi(slab0, z);
              Value slab16 = muli(slab, c16);
              Value at = addi(slab16, inxy);
              Value ati = indexCast(i64Type, at);
              Value offset = muli(ati, bytes);
              Value address = addi(bbase, offset);
              Value pointer =
                  LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
              Value value =
                  LLVM::LoadOp::create(b, loc, realType, pointer, realBytes);
              Value t17 = muli(zz, c17);
              Value tat = addi(t17, inxy);
              store(value, tile, {tat});
              scf::YieldOp::create(b, loc);
            }
            scf::YieldOp::create(b, loc);
          });
      gpu::BarrierOp::create(b, loc);
      scf::ForOp::create(
          b, loc, c0, c4, c1, ValueRange{},
          [&](OpBuilder &b, Location, Value j, ValueRange carried) {
            Value j128 = muli(j, c128);
            Value e = addi(j128, tx);
            Value zz = divui(e, c16);
            Value inxy = remui(e, c16);
            Value z = addi(z0, zz);
            Value in = cmpi(arith::CmpIPredicate::ult, z, k3);
            auto inBranch3 =
                scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(inBranch3.thenBlock());
              Value slab = addi(slab0, z);
              Value slab16 = muli(slab, c16);
              Value at = addi(slab16, inxy);
              Value ati = indexCast(i64Type, at);
              Value offset = muli(ati, bytes);
              Value address = addi(bbase, offset);
              Value pointer =
                  LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
              LLVM::StoreOp::create(b, loc, zero_r, pointer, realBytes);
              scf::YieldOp::create(b, loc);
            }
            scf::YieldOp::create(b, loc);
          });
      Value warp = divui(tx, c32);
      Value lane = remui(tx, c32);
      Value z = addi(z0, lane);
      Value in = cmpi(arith::CmpIPredicate::ult, z, k3);
      Value t17 = muli(lane, c17);
      scf::ForOp::create(
          b, loc, c0, c4, c1, ValueRange{},
          [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
            Value pass4 = muli(pass, c4);
            Value r = addi(pass4, warp);
            Value inx = remui(r, c4);
            Value iny = divui(r, c4);
            Value g1a = muli(bx1, c4);
            Value g1 = addi(g1a, inx);
            Value g2a = muli(by1, c4);
            Value g2 = addi(g2a, iny);
            Value in1 = cmpi(arith::CmpIPredicate::ult, g1, k1);
            Value in2 = cmpi(arith::CmpIPredicate::ult, g2, k2);
            Value in12 = andi(in1, in2);
            Value writes = andi(in12, in);
            auto writesBranch =
                scf::IfOp::create(b, loc, TypeRange{}, writes, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(writesBranch.thenBlock());
              Value row1 = muli(g1, k2);
              Value row = addi(row1, g2);
              Value gbase = muli(row, k3);
              Value tat = addi(t17, r);
              Value value = load(tile, {tat});
              Value gat = addi(gbase, z);
              store(value, realBuffer, {gat});
              scf::YieldOp::create(b, loc);
            }
            scf::YieldOp::create(b, loc);
          });
    }
    Value c0w;
    Value c32w;
    Value two_i;
    Value three_i;
    Value warp;
    Value lane;
    Value wb;
    if constexpr (fixedGrid) {
      c0w = arith::ConstantIndexOp::create(b, loc, 0);
      c32w = arith::ConstantIndexOp::create(b, loc, 32);
      two_i = arith::ConstantIndexOp::create(b, loc, 2);
      three_i = arith::ConstantIndexOp::create(b, loc, 3);
      Value four_w = arith::ConstantIndexOp::create(b, loc, 4);
      warp = divui(tx, c32w);
      Value lane_x = remui(tx, c32w);
      lane = indexCast(i32Type, lane_x);
      wb = muli(bx, four_w);
    }
    Value i;
    Value inside;
    if constexpr (!useList && fixedGrid) {
      i = addi(wb, warp);
      inside = cmpi(arith::CmpIPredicate::ult, i, n_all);
    }
    Value slot;
    if constexpr (useList && fixedGrid) {
      slot = addi(wb, warp);
      inside = cmpi(arith::CmpIPredicate::ult, slot, n_all);
    }
    if constexpr (fixedGrid) {
      auto insideBranch2 =
          scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(insideBranch2.thenBlock());
        scf::IfOp is_listedBranch;
        if constexpr (useList) {
          Value listed = load(list, {slot});
          Value listed_none =
              arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
          Value is_listed =
              cmpi(arith::CmpIPredicate::sge, listed, listed_none);
          is_listedBranch =
              scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, false);
          b.setInsertionPointToStart(is_listedBranch.thenBlock());
          i = indexCast(idxType, listed);
        }
        spreadParticle(lane, i, c0w, c32w, two_i, three_i);
        scf::YieldOp::create(b, loc);
        if constexpr (useList) {
          b.setInsertionPointAfter(is_listedBranch);
          scf::YieldOp::create(b, loc);
        }
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitWarpSpread<false, false>(StringRef);
template void DevicePME::emitWarpSpread<false, true>(StringRef);
template void DevicePME::emitWarpSpread<true, false>(StringRef);
template void DevicePME::emitWarpSpread<true, true>(StringRef);

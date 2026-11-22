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
  if constexpr (!useList || !fixedGrid) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  if constexpr (useList || !fixedGrid) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  if constexpr (fixedGrid) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i64Type));
  }
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
  Value n_all = arith::DivUIOp::create(b, loc, places_length, c6);
  Value k12 = arith::AddIOp::create(b, loc, k1, k2);
  Value k123 = arith::AddIOp::create(b, loc, k12, k3);
  Value pe = arith::MulIOp::create(b, loc, k123, c3);
  Value warps = arith::AddIOp::create(b, loc, n_all, c3);
  Value blocks = arith::DivUIOp::create(b, loc, warps, c4);
  Value b2;
  Value b12;
  if constexpr (!fixedGrid) {
    Value k1u = arith::AddIOp::create(b, loc, k1, c3);
    Value b1 = arith::DivUIOp::create(b, loc, k1u, c4);
    Value k2u = arith::AddIOp::create(b, loc, k2, c3);
    b2 = arith::DivUIOp::create(b, loc, k2u, c4);
    b12 = arith::MulIOp::create(b, loc, b1, b2);
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
      warp = arith::DivUIOp::create(b, loc, tx, c32w);
      Value lane_x = arith::RemUIOp::create(b, loc, tx, c32w);
      lane = arith::IndexCastOp::create(b, loc, i32Type, lane_x);
      wb = arith::MulIOp::create(b, loc, bx, four_w);
    }
    Value i;
    Value inside;
    if constexpr (!useList && !fixedGrid) {
      i = arith::AddIOp::create(b, loc, wb, warp);
      inside =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, i, n_all);
    }
    Value item;
    if constexpr (fixedGrid) {
      Value base = arith::MulIOp::create(b, loc, bx, c128);
      item = arith::AddIOp::create(b, loc, base, tx);
      inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item,
                                     points_grid);
    }
    Value slot;
    if constexpr (useList && !fixedGrid) {
      slot = arith::AddIOp::create(b, loc, wb, warp);
      inside =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, slot, n_all);
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
        Value is_listed = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::sge, listed, listed_none);
        is_listedBranch =
            scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, false);
        b.setInsertionPointToStart(is_listedBranch.thenBlock());
        i = arith::IndexCastOp::create(b, loc, idxType, listed);
      }
      if constexpr (!fixedGrid) {
        Value points =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
        Value one = arith::ConstantIndexOp::create(b, loc, 1);
        Value lane_i = arith::IndexCastOp::create(b, loc, idxType, lane);
        Value three_p = arith::ConstantIndexOp::create(b, loc, 3);
        Value tp = arith::MulIOp::create(b, loc, three_p, points);
        Value active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                             lane_i, tp);
        Value lane_a = arith::SelectOp::create(b, loc, active, lane_i, c0w);
        Value axis = arith::DivUIOp::create(b, loc, lane_a, points);
        Value j = arith::RemUIOp::create(b, loc, lane_a, points);
        Value fat0 = arith::AddIOp::create(b, loc, axis, three_i);
        Value fat1 = arith::MulIOp::create(b, loc, fat0, n_all);
        Value fat = arith::AddIOp::create(b, loc, fat1, i);
        Value frac = load(places, {fat});
        Value twor =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
        Value oner =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
        Value frac2 = arith::MulFOp::create(b, loc, frac, twor);
        Value t = arith::SubFOp::create(b, loc, frac2, oner);
        Value meta_d = arith::AddIOp::create(b, loc, pe, two_i);
        Value dr = load(tables, {meta_d});
        Value meta_n = load(tables, {pe});
        Value terms_i = arith::FPToSIOp::create(b, loc, i32Type, dr);
        Value terms = arith::IndexCastOp::create(b, loc, idxType, terms_i);
        Value n_i = arith::FPToSIOp::create(b, loc, i32Type, meta_n);
        Value n_s = arith::IndexCastOp::create(b, loc, idxType, n_i);
        Value n_2 = arith::AddIOp::create(b, loc, n_s, n_s);
        Value wb0 = arith::AddIOp::create(b, loc, pe, three_i);
        Value wbase = arith::AddIOp::create(b, loc, wb0, n_2);
        Value wvalue0 = clenshaw(wbase, t, one, terms, points, j, tables);
        Value zero_w =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
        Value wvalue = arith::SelectOp::create(b, loc, active, wvalue0, zero_w);
        Value qs = load(q, {i});
        Value qi = convert(realType, qs);
        Value first_axis =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, axis, c0w);
        Value qv = arith::MulFOp::create(b, loc, wvalue, qi);
        Value weight = arith::SelectOp::create(b, loc, first_axis, qv, wvalue);
        Value xsat = arith::ConstantIndexOp::create(b, loc, 0);
        Value xsn = arith::MulIOp::create(b, loc, n_all, xsat);
        Value xst = arith::AddIOp::create(b, loc, xsn, i);
        Value xstartr = load(places, {xst});
        Value xstart = arith::FPToSIOp::create(b, loc, i32Type, xstartr);
        Value ysat = arith::ConstantIndexOp::create(b, loc, 1);
        Value ysn = arith::MulIOp::create(b, loc, n_all, ysat);
        Value yst = arith::AddIOp::create(b, loc, ysn, i);
        Value ystartr = load(places, {yst});
        Value ystart = arith::FPToSIOp::create(b, loc, i32Type, ystartr);
        Value zsat = arith::ConstantIndexOp::create(b, loc, 2);
        Value zsn = arith::MulIOp::create(b, loc, n_all, zsat);
        Value zst = arith::AddIOp::create(b, loc, zsn, i);
        Value zstartr = load(places, {zst});
        Value zstart = arith::FPToSIOp::create(b, loc, i32Type, zstartr);
        Value k1t = arith::IndexCastOp::create(b, loc, i32Type, k1);
        Value k2t = arith::IndexCastOp::create(b, loc, i32Type, k2);
        Value k3t = arith::IndexCastOp::create(b, loc, i32Type, k3);
        Value pt =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
        Value ppt = arith::MulIOp::create(b, loc, pt, pt);
        Value pppt = arith::MulIOp::create(b, loc, ppt, pt);
        Value thirty_two =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
        Value zero32 =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value width =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
        Value passes_n =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
        Value passes_n2 = arith::MulIOp::create(b, loc, passes_n, passes_n);
        Value passes_n3 = arith::MulIOp::create(b, loc, passes_n2, passes_n);
        Value c31p = arith::ConstantIndexOp::create(b, loc, 31);
        Value passes_u = arith::AddIOp::create(b, loc, passes_n3, c31p);
        Value passes = arith::DivUIOp::create(b, loc, passes_u, c32w);
        Value b2t = arith::IndexCastOp::create(b, loc, i32Type, b2);
        Value four32 =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
        Value sixteen =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 16));
        Value bytes = arith::ConstantOp::create(
            b, loc, b.getIntegerAttr(i64Type, realBytes));
        Value bindex =
            memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bricks);
        Value bbase = arith::IndexCastOp::create(b, loc, i64Type, bindex);
        scf::ForOp::create(
            b, loc, c0w, passes, one, ValueRange{},
            [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
              Value pass32 = arith::IndexCastOp::create(b, loc, i32Type, pass);
              Value pass_base =
                  arith::MulIOp::create(b, loc, pass32, thirty_two);
              Value kk = arith::AddIOp::create(b, loc, pass_base, lane);
              Value in = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::slt, kk, pppt);
              Value kc = arith::SelectOp::create(b, loc, in, kk, zero32);
              Value pc = arith::DivUIOp::create(b, loc, kc, ppt);
              Value pab = arith::RemUIOp::create(b, loc, kc, ppt);
              Value pb = arith::DivUIOp::create(b, loc, pab, pt);
              Value pa = arith::RemUIOp::create(b, loc, pab, pt);
              auto wxShuffle = gpu::ShuffleOp::create(b, loc, weight, pa, width,
                                                      gpu::ShuffleMode::IDX);
              Value wx = wxShuffle.getShuffleResult();

              Value pbs = arith::AddIOp::create(b, loc, pb, pt);
              auto wyShuffle = gpu::ShuffleOp::create(
                  b, loc, weight, pbs, width, gpu::ShuffleMode::IDX);
              Value wy = wyShuffle.getShuffleResult();

              Value pt2 = arith::AddIOp::create(b, loc, pt, pt);
              Value pcs = arith::AddIOp::create(b, loc, pc, pt2);
              auto wzShuffle = gpu::ShuffleOp::create(
                  b, loc, weight, pcs, width, gpu::ShuffleMode::IDX);
              Value wz = wzShuffle.getShuffleResult();

              auto inBranch =
                  scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(inBranch.thenBlock());
                Value g1s = arith::AddIOp::create(b, loc, xstart, pa);
                Value g1w = arith::SubIOp::create(b, loc, g1s, k1t);
                Value g1over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g1s, k1t);
                Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
                Value g2s = arith::AddIOp::create(b, loc, ystart, pb);
                Value g2w = arith::SubIOp::create(b, loc, g2s, k2t);
                Value g2over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g2s, k2t);
                Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
                Value g3s = arith::AddIOp::create(b, loc, zstart, pc);
                Value g3w = arith::SubIOp::create(b, loc, g3s, k3t);
                Value g3over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g3s, k3t);
                Value g3 = arith::SelectOp::create(b, loc, g3over, g3w, g3s);
                Value wxy = arith::MulFOp::create(b, loc, wx, wy);
                Value contribution = arith::MulFOp::create(b, loc, wxy, wz);
                Value bx1 = arith::DivUIOp::create(b, loc, g1, four32);
                Value by1 = arith::DivUIOp::create(b, loc, g2, four32);
                Value brow = arith::MulIOp::create(b, loc, bx1, b2t);
                Value bxy = arith::AddIOp::create(b, loc, brow, by1);
                Value bxyz = arith::MulIOp::create(b, loc, bxy, k3t);
                Value slab = arith::AddIOp::create(b, loc, bxyz, g3);
                Value slab16 = arith::MulIOp::create(b, loc, slab, sixteen);
                Value inx = arith::RemUIOp::create(b, loc, g1, four32);
                Value iny = arith::RemUIOp::create(b, loc, g2, four32);
                Value iny4 = arith::MulIOp::create(b, loc, iny, four32);
                Value inxy = arith::AddIOp::create(b, loc, iny4, inx);
                Value atb = arith::AddIOp::create(b, loc, slab16, inxy);
                Value ati = arith::ExtUIOp::create(b, loc, i64Type, atb);
                Value offset = arith::MulIOp::create(b, loc, ati, bytes);
                Value address = arith::AddIOp::create(b, loc, bbase, offset);
                Value pointer =
                    LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
                Value value =
                    arith::AddFOp::create(b, loc, contribution, zero_w);
                addGlobal(pointer, value);
                scf::YieldOp::create(b, loc);
              }
              scf::YieldOp::create(b, loc);
            });
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
    Value k3_31 = arith::AddIOp::create(b, loc, k3, c31);
    zchunks = arith::DivUIOp::create(b, loc, k3_31, c32);
    copy_blocks = arith::MulIOp::create(b, loc, b12, zchunks);
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
      Value bxy = arith::DivUIOp::create(b, loc, bx, zchunks);
      Value zc = arith::RemUIOp::create(b, loc, bx, zchunks);
      Value z0 = arith::MulIOp::create(b, loc, zc, c32);
      Value bx1 = arith::DivUIOp::create(b, loc, bxy, b2);
      Value by1 = arith::RemUIOp::create(b, loc, bxy, b2);
      Value slab0 = arith::MulIOp::create(b, loc, bxy, k3);
      Value zero_r =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value bytes = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(i64Type, realBytes));
      Value bindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bricks);
      Value bbase = arith::IndexCastOp::create(b, loc, i64Type, bindex);
      scf::ForOp::create(
          b, loc, c0, c4, c1, ValueRange{},
          [&](OpBuilder &b, Location, Value j, ValueRange carried) {
            Value j128 = arith::MulIOp::create(b, loc, j, c128);
            Value e = arith::AddIOp::create(b, loc, j128, tx);
            Value zz = arith::DivUIOp::create(b, loc, e, c16);
            Value inxy = arith::RemUIOp::create(b, loc, e, c16);
            Value z = arith::AddIOp::create(b, loc, z0, zz);
            Value in =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, z, k3);
            auto inBranch2 =
                scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(inBranch2.thenBlock());
              Value slab = arith::AddIOp::create(b, loc, slab0, z);
              Value slab16 = arith::MulIOp::create(b, loc, slab, c16);
              Value at = arith::AddIOp::create(b, loc, slab16, inxy);
              Value ati = arith::IndexCastOp::create(b, loc, i64Type, at);
              Value offset = arith::MulIOp::create(b, loc, ati, bytes);
              Value address = arith::AddIOp::create(b, loc, bbase, offset);
              Value pointer =
                  LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
              Value value =
                  LLVM::LoadOp::create(b, loc, realType, pointer, realBytes);
              Value t17 = arith::MulIOp::create(b, loc, zz, c17);
              Value tat = arith::AddIOp::create(b, loc, t17, inxy);
              store(value, tile, {tat});
              scf::YieldOp::create(b, loc);
            }
            scf::YieldOp::create(b, loc);
          });
      gpu::BarrierOp::create(b, loc);
      scf::ForOp::create(
          b, loc, c0, c4, c1, ValueRange{},
          [&](OpBuilder &b, Location, Value j, ValueRange carried) {
            Value j128 = arith::MulIOp::create(b, loc, j, c128);
            Value e = arith::AddIOp::create(b, loc, j128, tx);
            Value zz = arith::DivUIOp::create(b, loc, e, c16);
            Value inxy = arith::RemUIOp::create(b, loc, e, c16);
            Value z = arith::AddIOp::create(b, loc, z0, zz);
            Value in =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, z, k3);
            auto inBranch3 =
                scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(inBranch3.thenBlock());
              Value slab = arith::AddIOp::create(b, loc, slab0, z);
              Value slab16 = arith::MulIOp::create(b, loc, slab, c16);
              Value at = arith::AddIOp::create(b, loc, slab16, inxy);
              Value ati = arith::IndexCastOp::create(b, loc, i64Type, at);
              Value offset = arith::MulIOp::create(b, loc, ati, bytes);
              Value address = arith::AddIOp::create(b, loc, bbase, offset);
              Value pointer =
                  LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
              LLVM::StoreOp::create(b, loc, zero_r, pointer, realBytes);
              scf::YieldOp::create(b, loc);
            }
            scf::YieldOp::create(b, loc);
          });
      Value warp = arith::DivUIOp::create(b, loc, tx, c32);
      Value lane = arith::RemUIOp::create(b, loc, tx, c32);
      Value z = arith::AddIOp::create(b, loc, z0, lane);
      Value in =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, z, k3);
      Value t17 = arith::MulIOp::create(b, loc, lane, c17);
      scf::ForOp::create(
          b, loc, c0, c4, c1, ValueRange{},
          [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
            Value pass4 = arith::MulIOp::create(b, loc, pass, c4);
            Value r = arith::AddIOp::create(b, loc, pass4, warp);
            Value inx = arith::RemUIOp::create(b, loc, r, c4);
            Value iny = arith::DivUIOp::create(b, loc, r, c4);
            Value g1a = arith::MulIOp::create(b, loc, bx1, c4);
            Value g1 = arith::AddIOp::create(b, loc, g1a, inx);
            Value g2a = arith::MulIOp::create(b, loc, by1, c4);
            Value g2 = arith::AddIOp::create(b, loc, g2a, iny);
            Value in1 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                              g1, k1);
            Value in2 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                              g2, k2);
            Value in12 = arith::AndIOp::create(b, loc, in1, in2);
            Value writes = arith::AndIOp::create(b, loc, in12, in);
            auto writesBranch =
                scf::IfOp::create(b, loc, TypeRange{}, writes, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(writesBranch.thenBlock());
              Value row1 = arith::MulIOp::create(b, loc, g1, k2);
              Value row = arith::AddIOp::create(b, loc, row1, g2);
              Value gbase = arith::MulIOp::create(b, loc, row, k3);
              Value tat = arith::AddIOp::create(b, loc, t17, r);
              Value value = load(tile, {tat});
              Value gat = arith::AddIOp::create(b, loc, gbase, z);
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
      warp = arith::DivUIOp::create(b, loc, tx, c32w);
      Value lane_x = arith::RemUIOp::create(b, loc, tx, c32w);
      lane = arith::IndexCastOp::create(b, loc, i32Type, lane_x);
      wb = arith::MulIOp::create(b, loc, bx, four_w);
    }
    Value i;
    Value inside;
    if constexpr (!useList && fixedGrid) {
      i = arith::AddIOp::create(b, loc, wb, warp);
      inside =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, i, n_all);
    }
    Value slot;
    if constexpr (useList && fixedGrid) {
      slot = arith::AddIOp::create(b, loc, wb, warp);
      inside =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, slot, n_all);
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
          Value is_listed = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::sge, listed, listed_none);
          is_listedBranch =
              scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, false);
          b.setInsertionPointToStart(is_listedBranch.thenBlock());
          i = arith::IndexCastOp::create(b, loc, idxType, listed);
        }
        Value points =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
        Value one = arith::ConstantIndexOp::create(b, loc, 1);
        Value lane_i = arith::IndexCastOp::create(b, loc, idxType, lane);
        Value three_p = arith::ConstantIndexOp::create(b, loc, 3);
        Value tp = arith::MulIOp::create(b, loc, three_p, points);
        Value active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                             lane_i, tp);
        Value lane_a = arith::SelectOp::create(b, loc, active, lane_i, c0w);
        Value axis = arith::DivUIOp::create(b, loc, lane_a, points);
        Value j = arith::RemUIOp::create(b, loc, lane_a, points);
        Value fat0 = arith::AddIOp::create(b, loc, axis, three_i);
        Value fat1 = arith::MulIOp::create(b, loc, fat0, n_all);
        Value fat = arith::AddIOp::create(b, loc, fat1, i);
        Value frac = load(places, {fat});
        Value twor =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 2.0));
        Value oner =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
        Value frac2 = arith::MulFOp::create(b, loc, frac, twor);
        Value t = arith::SubFOp::create(b, loc, frac2, oner);
        Value meta_d = arith::AddIOp::create(b, loc, pe, two_i);
        Value dr = load(tables, {meta_d});
        Value meta_n = load(tables, {pe});
        Value terms_i = arith::FPToSIOp::create(b, loc, i32Type, dr);
        Value terms = arith::IndexCastOp::create(b, loc, idxType, terms_i);
        Value n_i = arith::FPToSIOp::create(b, loc, i32Type, meta_n);
        Value n_s = arith::IndexCastOp::create(b, loc, idxType, n_i);
        Value n_2 = arith::AddIOp::create(b, loc, n_s, n_s);
        Value wb0 = arith::AddIOp::create(b, loc, pe, three_i);
        Value wbase = arith::AddIOp::create(b, loc, wb0, n_2);
        Value wvalue0 = clenshaw(wbase, t, one, terms, points, j, tables);
        Value zero_w =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
        Value wvalue = arith::SelectOp::create(b, loc, active, wvalue0, zero_w);
        Value qs = load(q, {i});
        Value qi = convert(realType, qs);
        Value first_axis =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, axis, c0w);
        Value qv = arith::MulFOp::create(b, loc, wvalue, qi);
        Value weight = arith::SelectOp::create(b, loc, first_axis, qv, wvalue);
        Value xsat = arith::ConstantIndexOp::create(b, loc, 0);
        Value xsn = arith::MulIOp::create(b, loc, n_all, xsat);
        Value xst = arith::AddIOp::create(b, loc, xsn, i);
        Value xstartr = load(places, {xst});
        Value xstart = arith::FPToSIOp::create(b, loc, i32Type, xstartr);
        Value ysat = arith::ConstantIndexOp::create(b, loc, 1);
        Value ysn = arith::MulIOp::create(b, loc, n_all, ysat);
        Value yst = arith::AddIOp::create(b, loc, ysn, i);
        Value ystartr = load(places, {yst});
        Value ystart = arith::FPToSIOp::create(b, loc, i32Type, ystartr);
        Value zsat = arith::ConstantIndexOp::create(b, loc, 2);
        Value zsn = arith::MulIOp::create(b, loc, n_all, zsat);
        Value zst = arith::AddIOp::create(b, loc, zsn, i);
        Value zstartr = load(places, {zst});
        Value zstart = arith::FPToSIOp::create(b, loc, i32Type, zstartr);
        Value k1t = arith::IndexCastOp::create(b, loc, i32Type, k1);
        Value k2t = arith::IndexCastOp::create(b, loc, i32Type, k2);
        Value k3t = arith::IndexCastOp::create(b, loc, i32Type, k3);
        Value pt =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
        Value ppt = arith::MulIOp::create(b, loc, pt, pt);
        Value pppt = arith::MulIOp::create(b, loc, ppt, pt);
        Value thirty_two =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
        Value zero32 =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value width =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
        Value passes_n =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
        Value passes_n2 = arith::MulIOp::create(b, loc, passes_n, passes_n);
        Value passes_n3 = arith::MulIOp::create(b, loc, passes_n2, passes_n);
        Value c31p = arith::ConstantIndexOp::create(b, loc, 31);
        Value passes_u = arith::AddIOp::create(b, loc, passes_n3, c31p);
        Value passes = arith::DivUIOp::create(b, loc, passes_u, c32w);
        Value scale = arith::ConstantOp::create(
            b, loc, b.getFloatAttr(realType, 1099511627776.0));
        Value eight =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 8));
        Value gridindex =
            memref::ExtractAlignedPointerAsIndexOp::create(b, loc, grid);
        Value gridbase = arith::IndexCastOp::create(b, loc, i64Type, gridindex);
        scf::ForOp::create(
            b, loc, c0w, passes, one, ValueRange{},
            [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
              Value pass32 = arith::IndexCastOp::create(b, loc, i32Type, pass);
              Value pass_base =
                  arith::MulIOp::create(b, loc, pass32, thirty_two);
              Value kk = arith::AddIOp::create(b, loc, pass_base, lane);
              Value in = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::slt, kk, pppt);
              Value kc = arith::SelectOp::create(b, loc, in, kk, zero32);
              Value pc = arith::DivUIOp::create(b, loc, kc, ppt);
              Value pab = arith::RemUIOp::create(b, loc, kc, ppt);
              Value pb = arith::DivUIOp::create(b, loc, pab, pt);
              Value pa = arith::RemUIOp::create(b, loc, pab, pt);
              auto wxShuffle = gpu::ShuffleOp::create(b, loc, weight, pa, width,
                                                      gpu::ShuffleMode::IDX);
              Value wx = wxShuffle.getShuffleResult();

              Value pbs = arith::AddIOp::create(b, loc, pb, pt);
              auto wyShuffle = gpu::ShuffleOp::create(
                  b, loc, weight, pbs, width, gpu::ShuffleMode::IDX);
              Value wy = wyShuffle.getShuffleResult();

              Value pt2 = arith::AddIOp::create(b, loc, pt, pt);
              Value pcs = arith::AddIOp::create(b, loc, pc, pt2);
              auto wzShuffle = gpu::ShuffleOp::create(
                  b, loc, weight, pcs, width, gpu::ShuffleMode::IDX);
              Value wz = wzShuffle.getShuffleResult();

              auto inBranch =
                  scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(inBranch.thenBlock());
                Value g1s = arith::AddIOp::create(b, loc, xstart, pa);
                Value g1w = arith::SubIOp::create(b, loc, g1s, k1t);
                Value g1over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g1s, k1t);
                Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
                Value g2s = arith::AddIOp::create(b, loc, ystart, pb);
                Value g2w = arith::SubIOp::create(b, loc, g2s, k2t);
                Value g2over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g2s, k2t);
                Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
                Value g3s = arith::AddIOp::create(b, loc, zstart, pc);
                Value g3w = arith::SubIOp::create(b, loc, g3s, k3t);
                Value g3over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g3s, k3t);
                Value g3 = arith::SelectOp::create(b, loc, g3over, g3w, g3s);
                Value wxy = arith::MulFOp::create(b, loc, wx, wy);
                Value contribution = arith::MulFOp::create(b, loc, wxy, wz);
                Value scaled =
                    arith::MulFOp::create(b, loc, contribution, scale);
                Value rounded = math::RoundEvenOp::create(b, loc, scaled);
                Value fixedv =
                    arith::FPToSIOp::create(b, loc, i64Type, rounded);
                Value row1 = arith::MulIOp::create(b, loc, g1, k2t);
                Value row = arith::AddIOp::create(b, loc, row1, g2);
                Value row64 = arith::ExtUIOp::create(b, loc, i64Type, row);
                Value k3w = arith::ExtUIOp::create(b, loc, i64Type, k3t);
                Value gbase = arith::MulIOp::create(b, loc, row64, k3w);
                Value g364 = arith::ExtUIOp::create(b, loc, i64Type, g3);
                Value at = arith::AddIOp::create(b, loc, gbase, g364);
                Value offset = arith::MulIOp::create(b, loc, at, eight);
                Value address = arith::AddIOp::create(b, loc, gridbase, offset);
                Value pointer =
                    LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
                LLVM::AtomicRMWOp::create(
                    b, loc, LLVM::AtomicBinOp::add, pointer, fixedv,
                    LLVM::AtomicOrdering::monotonic, "device");
                scf::YieldOp::create(b, loc);
              }
              scf::YieldOp::create(b, loc);
            });
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

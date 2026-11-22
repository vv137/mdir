// Order-four brick spreading and transfer to the global grid (D108).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool useList> void DevicePME::emitSpreadBricks(StringRef symbol) {
  StringRef blockFunction = "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic, 3}, positionType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.append(3, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value x = entry->getArgument(arg++);
  Value q = entry->getArgument(arg++);
  Value list;
  if constexpr (useList) {
    list = entry->getArgument(arg++);
  }
  Value weights = entry->getArgument(arg++);
  Value bricks = entry->getArgument(arg++);
  Value realBuffer = entry->getArgument(arg++);
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
  Value n_all = memref::DimOp::create(b, loc, x, c0);
  Value k1u = addi(k1, c3);
  Value b1 = divui(k1u, c4);
  Value k2u = addi(k2, c3);
  Value b2 = divui(k2u, c4);
  Value b12 = muli(b1, b2);
  Value b123 = muli(b12, k3);
  muli(b123, c16);
  Value laneCount = muli(n_all, c32);
  Value blocks_lanes =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{laneCount})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_lanes, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;

    Value tx = launch.getThreadIds().x;

    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, laneCount);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value c32i = arith::ConstantIndexOp::create(b, loc, 32);
      Value i;
      if constexpr (!useList) {
        i = divui(item, c32i);
      }
      scf::IfOp is_listedBranch;
      if constexpr (useList) {
        Value slot = divui(item, c32i);
        Value listed = load(list, {slot});
        Value listed_none =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value is_listed = cmpi(arith::CmpIPredicate::sge, listed, listed_none);
        is_listedBranch =
            scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, false);
        b.setInsertionPointToStart(is_listedBranch.thenBlock());
        i = indexCast(idxType, listed);
      }
      Value item32 = indexCast(i32Type, item);
      Value n32 = indexCast(i32Type, n_all);
      Value i32 = indexCast(i32Type, i);
      Value t32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value t2 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 3));
      Value t4 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 7));
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 11));
      Value t16 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 16));
      Value lane = remui(item32, t32);
      Value xa = remui(lane, t4);
      Value lane4 = divui(lane, t4);
      Value yb = remui(lane4, t4);
      Value zc = divui(lane, t16);
      Value zc2 = addi(zc, t2);
      Value k1t = indexCast(i32Type, k1);
      Value k2t = indexCast(i32Type, k2);
      Value k3t = indexCast(i32Type, k3);
      Value b2t = indexCast(i32Type, b2);
      Value qs = load(q, {i});
      Value qi = convert(realType, qs);
      Value t0c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value stx_r_n = muli(n32, t0c);
      Value stx_r_at32 = addi(stx_r_n, i32);
      Value stx_r_at = indexCast(idxType, stx_r_at32);
      Value stx_r = load(weights, {stx_r_at});
      Value stx = fptosi(i32Type, stx_r);
      Value t1c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      Value sty_r_n = muli(n32, t1c);
      Value sty_r_at32 = addi(sty_r_n, i32);
      Value sty_r_at = indexCast(idxType, sty_r_at32);
      Value sty_r = load(weights, {sty_r_at});
      Value sty = fptosi(i32Type, sty_r);
      Value t2c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      Value stz_r_n = muli(n32, t2c);
      Value stz_r_at32 = addi(stz_r_n, i32);
      Value stz_r_at = indexCast(idxType, stz_r_at32);
      Value stz_r = load(weights, {stz_r_at});
      Value stz = fptosi(i32Type, stz_r);
      Value f0c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 3));
      Value frx_n = muli(n32, f0c);
      Value frx_at32 = addi(frx_n, i32);
      Value frx_at = indexCast(idxType, frx_at32);
      Value frx = load(weights, {frx_at});
      Value f1c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      Value fry_n = muli(n32, f1c);
      Value fry_at32 = addi(fry_n, i32);
      Value fry_at = indexCast(idxType, fry_at32);
      Value fry = load(weights, {fry_at});
      Value f2c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 5));
      Value frz_n = muli(n32, f2c);
      Value frz_at32 = addi(frz_n, i32);
      Value frz_at = indexCast(idxType, frz_at32);
      Value frz = load(weights, {frz_at});
      Value w_sixth = arith::ConstantOp::create(
          b, loc, b.getFloatAttr(realType, 0.16666666666666666));
      Value w_one =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
      Value w_three =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 3.0));
      Value w_four =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 4.0));
      Value w_six =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 6.0));
      Value w_zero32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value w_one32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      Value w_two32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      Value wxa_w2 = mulf(frx, frx);
      Value wxa_w3 = mulf(wxa_w2, frx);
      Value wxa_omw = subf(w_one, frx);
      Value wxa_omw2 = mulf(wxa_omw, wxa_omw);
      Value wxa_m0 = mulf(wxa_omw2, wxa_omw);
      Value wxa_a1 = mulf(w_three, wxa_w3);
      Value wxa_b1 = mulf(w_six, wxa_w2);
      Value wxa_c1 = subf(wxa_a1, wxa_b1);
      Value wxa_m1 = addf(wxa_c1, w_four);
      Value wxa_a2 = mulf(w_three, wxa_w2);
      Value wxa_b2 = mulf(w_three, frx);
      Value wxa_c2 = subf(wxa_a2, wxa_a1);
      Value wxa_d2 = addf(wxa_c2, wxa_b2);
      Value wxa_m2 = addf(wxa_d2, w_one);
      Value wxa_is0 = cmpi(arith::CmpIPredicate::eq, xa, w_zero32);
      Value wxa_is1 = cmpi(arith::CmpIPredicate::eq, xa, w_one32);
      Value wxa_is2 = cmpi(arith::CmpIPredicate::eq, xa, w_two32);
      Value wxa_s2 = select(wxa_is2, wxa_m2, wxa_w3);
      Value wxa_s1 = select(wxa_is1, wxa_m1, wxa_s2);
      Value wxa_s0 = select(wxa_is0, wxa_m0, wxa_s1);
      Value wxa = mulf(wxa_s0, w_sixth);
      Value wyb_w2 = mulf(fry, fry);
      Value wyb_w3 = mulf(wyb_w2, fry);
      Value wyb_omw = subf(w_one, fry);
      Value wyb_omw2 = mulf(wyb_omw, wyb_omw);
      Value wyb_m0 = mulf(wyb_omw2, wyb_omw);
      Value wyb_a1 = mulf(w_three, wyb_w3);
      Value wyb_b1 = mulf(w_six, wyb_w2);
      Value wyb_c1 = subf(wyb_a1, wyb_b1);
      Value wyb_m1 = addf(wyb_c1, w_four);
      Value wyb_a2 = mulf(w_three, wyb_w2);
      Value wyb_b2 = mulf(w_three, fry);
      Value wyb_c2 = subf(wyb_a2, wyb_a1);
      Value wyb_d2 = addf(wyb_c2, wyb_b2);
      Value wyb_m2 = addf(wyb_d2, w_one);
      Value wyb_is0 = cmpi(arith::CmpIPredicate::eq, yb, w_zero32);
      Value wyb_is1 = cmpi(arith::CmpIPredicate::eq, yb, w_one32);
      Value wyb_is2 = cmpi(arith::CmpIPredicate::eq, yb, w_two32);
      Value wyb_s2 = select(wyb_is2, wyb_m2, wyb_w3);
      Value wyb_s1 = select(wyb_is1, wyb_m1, wyb_s2);
      Value wyb_s0 = select(wyb_is0, wyb_m0, wyb_s1);
      Value wyb = mulf(wyb_s0, w_sixth);
      Value wz0_w2 = mulf(frz, frz);
      Value wz0_w3 = mulf(wz0_w2, frz);
      Value wz0_omw = subf(w_one, frz);
      Value wz0_omw2 = mulf(wz0_omw, wz0_omw);
      Value wz0_m0 = mulf(wz0_omw2, wz0_omw);
      Value wz0_a1 = mulf(w_three, wz0_w3);
      Value wz0_b1 = mulf(w_six, wz0_w2);
      Value wz0_c1 = subf(wz0_a1, wz0_b1);
      Value wz0_m1 = addf(wz0_c1, w_four);
      Value wz0_a2 = mulf(w_three, wz0_w2);
      Value wz0_b2 = mulf(w_three, frz);
      Value wz0_c2 = subf(wz0_a2, wz0_a1);
      Value wz0_d2 = addf(wz0_c2, wz0_b2);
      Value wz0_m2 = addf(wz0_d2, w_one);
      Value wz0_is0 = cmpi(arith::CmpIPredicate::eq, zc, w_zero32);
      Value wz0_is1 = cmpi(arith::CmpIPredicate::eq, zc, w_one32);
      Value wz0_is2 = cmpi(arith::CmpIPredicate::eq, zc, w_two32);
      Value wz0_s2 = select(wz0_is2, wz0_m2, wz0_w3);
      Value wz0_s1 = select(wz0_is1, wz0_m1, wz0_s2);
      Value wz0_s0 = select(wz0_is0, wz0_m0, wz0_s1);
      Value wz0 = mulf(wz0_s0, w_sixth);
      Value wz1_w2 = mulf(frz, frz);
      Value wz1_w3 = mulf(wz1_w2, frz);
      Value wz1_omw = subf(w_one, frz);
      Value wz1_omw2 = mulf(wz1_omw, wz1_omw);
      Value wz1_m0 = mulf(wz1_omw2, wz1_omw);
      Value wz1_a1 = mulf(w_three, wz1_w3);
      Value wz1_b1 = mulf(w_six, wz1_w2);
      Value wz1_c1 = subf(wz1_a1, wz1_b1);
      Value wz1_m1 = addf(wz1_c1, w_four);
      Value wz1_a2 = mulf(w_three, wz1_w2);
      Value wz1_b2 = mulf(w_three, frz);
      Value wz1_c2 = subf(wz1_a2, wz1_a1);
      Value wz1_d2 = addf(wz1_c2, wz1_b2);
      Value wz1_m2 = addf(wz1_d2, w_one);
      Value wz1_is0 = cmpi(arith::CmpIPredicate::eq, zc2, w_zero32);
      Value wz1_is1 = cmpi(arith::CmpIPredicate::eq, zc2, w_one32);
      Value wz1_is2 = cmpi(arith::CmpIPredicate::eq, zc2, w_two32);
      Value wz1_s2 = select(wz1_is2, wz1_m2, wz1_w3);
      Value wz1_s1 = select(wz1_is1, wz1_m1, wz1_s2);
      Value wz1_s0 = select(wz1_is0, wz1_m0, wz1_s1);
      Value wz1 = mulf(wz1_s0, w_sixth);
      Value qx = mulf(qi, wxa);
      Value qxy = mulf(qx, wyb);
      Value g1s = addi(stx, xa);
      Value g1w = subi(g1s, k1t);
      Value g1over = cmpi(arith::CmpIPredicate::uge, g1s, k1t);
      Value g1 = select(g1over, g1w, g1s);
      Value g2s = addi(sty, yb);
      Value g2w = subi(g2s, k2t);
      Value g2over = cmpi(arith::CmpIPredicate::uge, g2s, k2t);
      Value g2 = select(g2over, g2w, g2s);
      Value bx1 = divui(g1, t4);
      Value by1 = divui(g2, t4);
      Value brow = muli(bx1, b2t);
      Value bxy = addi(brow, by1);
      Value bxyz = muli(bxy, k3t);
      Value inx = remui(g1, t4);
      Value iny = remui(g2, t4);
      Value iny4 = muli(iny, t4);
      Value inxy = addi(iny4, inx);
      Value bytes = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(i64Type, realBytes));
      Value bindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bricks);
      Value bbase = indexCast(i64Type, bindex);
      Value h0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value h1 = arith::ConstantIndexOp::create(b, loc, 1);
      Value h2 = arith::ConstantIndexOp::create(b, loc, 2);
      scf::ForOp::create(
          b, loc, h0, h2, h1, ValueRange{},
          [&](OpBuilder &b, Location, Value h, ValueRange carried) {
            Value second = cmpi(arith::CmpIPredicate::ne, h, h0);
            Value c = select(second, zc2, zc);
            Value wz = select(second, wz1, wz0);
            Value g3s = addi(stz, c);
            Value g3w = subi(g3s, k3t);
            Value g3over = cmpi(arith::CmpIPredicate::uge, g3s, k3t);
            Value g3 = select(g3over, g3w, g3s);
            Value slab = addi(bxyz, g3);
            Value slab16 = muli(slab, t16);
            Value at = addi(slab16, inxy);
            Value ati = arith::ExtUIOp::create(b, loc, i64Type, at);
            Value offset = muli(ati, bytes);
            Value address = addi(bbase, offset);
            Value pointer =
                LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
            Value value = mulf(qxy, wz);
            addGlobal(pointer, value);
            scf::YieldOp::create(b, loc);
          });
      scf::YieldOp::create(b, loc);
      if constexpr (useList) {
        b.setInsertionPointAfter(is_listedBranch);
        scf::YieldOp::create(b, loc);
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  copyBricks(bricks, realBuffer, k1, k2, k3, b2, b12, c0, c1, c4, c16, c32,
             c128);
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitSpreadBricks<false>(StringRef);
template void DevicePME::emitSpreadBricks<true>(StringRef);

void DevicePME::copyBricks(Value bricks, Value realBuffer, Value k1, Value k2,
                           Value k3, Value b2, Value b12, Value c0, Value c1,
                           Value c4, Value c16, Value c32, Value c128) {
  arith::ConstantIndexOp::create(b, loc, 8);
  Value c17 = arith::ConstantIndexOp::create(b, loc, 17);
  Value c31 = arith::ConstantIndexOp::create(b, loc, 31);
  Value k3_31 = addi(k3, c31);
  Value zchunks = divui(k3_31, c32);
  Value copy_blocks = muli(b12, zchunks);
  {
    auto launch = gpu::LaunchOp::create(b, loc, copy_blocks, c1, c1, c128, c1,
                                        c1, Value{}, Type{}, ValueRange{},
                                        TypeRange{shared(544, realType)});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value tile = launch.getWorkgroupAttributionBBArgs()[0];
    Value bxy = divui(bx, zchunks);
    Value zc = remui(bx, zchunks);
    Value z0 = muli(zc, c32);
    Value bx1 = divui(bxy, b2);
    Value by1 = remui(bxy, b2);
    Value slab0 = muli(bxy, k3);
    Value zero_r =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value bytes =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, realBytes));
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
          auto inBranch =
              scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(inBranch.thenBlock());
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
    gpu::TerminatorOp::create(b, loc);
  }
}

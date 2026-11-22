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
  Value k1u = arith::AddIOp::create(b, loc, k1, c3);
  Value b1 = arith::DivUIOp::create(b, loc, k1u, c4);
  Value k2u = arith::AddIOp::create(b, loc, k2, c3);
  Value b2 = arith::DivUIOp::create(b, loc, k2u, c4);
  Value b12 = arith::MulIOp::create(b, loc, b1, b2);
  Value b123 = arith::MulIOp::create(b, loc, b12, k3);
  arith::MulIOp::create(b, loc, b123, c16);
  Value laneCount = arith::MulIOp::create(b, loc, n_all, c32);
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

    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value item = arith::AddIOp::create(b, loc, base, tx);
    Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                         item, laneCount);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value c32i = arith::ConstantIndexOp::create(b, loc, 32);
      Value i;
      if constexpr (!useList) {
        i = arith::DivUIOp::create(b, loc, item, c32i);
      }
      scf::IfOp is_listedBranch;
      if constexpr (useList) {
        Value slot = arith::DivUIOp::create(b, loc, item, c32i);
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
      Value item32 = arith::IndexCastOp::create(b, loc, i32Type, item);
      Value n32 = arith::IndexCastOp::create(b, loc, i32Type, n_all);
      Value i32 = arith::IndexCastOp::create(b, loc, i32Type, i);
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
      Value lane = arith::RemUIOp::create(b, loc, item32, t32);
      Value xa = arith::RemUIOp::create(b, loc, lane, t4);
      Value lane4 = arith::DivUIOp::create(b, loc, lane, t4);
      Value yb = arith::RemUIOp::create(b, loc, lane4, t4);
      Value zc = arith::DivUIOp::create(b, loc, lane, t16);
      Value zc2 = arith::AddIOp::create(b, loc, zc, t2);
      Value k1t = arith::IndexCastOp::create(b, loc, i32Type, k1);
      Value k2t = arith::IndexCastOp::create(b, loc, i32Type, k2);
      Value k3t = arith::IndexCastOp::create(b, loc, i32Type, k3);
      Value b2t = arith::IndexCastOp::create(b, loc, i32Type, b2);
      Value qs = load(q, {i});
      Value qi = convert(realType, qs);
      Value t0c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value stx_r_n = arith::MulIOp::create(b, loc, n32, t0c);
      Value stx_r_at32 = arith::AddIOp::create(b, loc, stx_r_n, i32);
      Value stx_r_at = arith::IndexCastOp::create(b, loc, idxType, stx_r_at32);
      Value stx_r = load(weights, {stx_r_at});
      Value stx = arith::FPToSIOp::create(b, loc, i32Type, stx_r);
      Value t1c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      Value sty_r_n = arith::MulIOp::create(b, loc, n32, t1c);
      Value sty_r_at32 = arith::AddIOp::create(b, loc, sty_r_n, i32);
      Value sty_r_at = arith::IndexCastOp::create(b, loc, idxType, sty_r_at32);
      Value sty_r = load(weights, {sty_r_at});
      Value sty = arith::FPToSIOp::create(b, loc, i32Type, sty_r);
      Value t2c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      Value stz_r_n = arith::MulIOp::create(b, loc, n32, t2c);
      Value stz_r_at32 = arith::AddIOp::create(b, loc, stz_r_n, i32);
      Value stz_r_at = arith::IndexCastOp::create(b, loc, idxType, stz_r_at32);
      Value stz_r = load(weights, {stz_r_at});
      Value stz = arith::FPToSIOp::create(b, loc, i32Type, stz_r);
      Value f0c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 3));
      Value frx_n = arith::MulIOp::create(b, loc, n32, f0c);
      Value frx_at32 = arith::AddIOp::create(b, loc, frx_n, i32);
      Value frx_at = arith::IndexCastOp::create(b, loc, idxType, frx_at32);
      Value frx = load(weights, {frx_at});
      Value f1c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
      Value fry_n = arith::MulIOp::create(b, loc, n32, f1c);
      Value fry_at32 = arith::AddIOp::create(b, loc, fry_n, i32);
      Value fry_at = arith::IndexCastOp::create(b, loc, idxType, fry_at32);
      Value fry = load(weights, {fry_at});
      Value f2c =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 5));
      Value frz_n = arith::MulIOp::create(b, loc, n32, f2c);
      Value frz_at32 = arith::AddIOp::create(b, loc, frz_n, i32);
      Value frz_at = arith::IndexCastOp::create(b, loc, idxType, frz_at32);
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
      Value wxa_w2 = arith::MulFOp::create(b, loc, frx, frx);
      Value wxa_w3 = arith::MulFOp::create(b, loc, wxa_w2, frx);
      Value wxa_omw = arith::SubFOp::create(b, loc, w_one, frx);
      Value wxa_omw2 = arith::MulFOp::create(b, loc, wxa_omw, wxa_omw);
      Value wxa_m0 = arith::MulFOp::create(b, loc, wxa_omw2, wxa_omw);
      Value wxa_a1 = arith::MulFOp::create(b, loc, w_three, wxa_w3);
      Value wxa_b1 = arith::MulFOp::create(b, loc, w_six, wxa_w2);
      Value wxa_c1 = arith::SubFOp::create(b, loc, wxa_a1, wxa_b1);
      Value wxa_m1 = arith::AddFOp::create(b, loc, wxa_c1, w_four);
      Value wxa_a2 = arith::MulFOp::create(b, loc, w_three, wxa_w2);
      Value wxa_b2 = arith::MulFOp::create(b, loc, w_three, frx);
      Value wxa_c2 = arith::SubFOp::create(b, loc, wxa_a2, wxa_a1);
      Value wxa_d2 = arith::AddFOp::create(b, loc, wxa_c2, wxa_b2);
      Value wxa_m2 = arith::AddFOp::create(b, loc, wxa_d2, w_one);
      Value wxa_is0 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, xa, w_zero32);
      Value wxa_is1 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, xa, w_one32);
      Value wxa_is2 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, xa, w_two32);
      Value wxa_s2 = arith::SelectOp::create(b, loc, wxa_is2, wxa_m2, wxa_w3);
      Value wxa_s1 = arith::SelectOp::create(b, loc, wxa_is1, wxa_m1, wxa_s2);
      Value wxa_s0 = arith::SelectOp::create(b, loc, wxa_is0, wxa_m0, wxa_s1);
      Value wxa = arith::MulFOp::create(b, loc, wxa_s0, w_sixth);
      Value wyb_w2 = arith::MulFOp::create(b, loc, fry, fry);
      Value wyb_w3 = arith::MulFOp::create(b, loc, wyb_w2, fry);
      Value wyb_omw = arith::SubFOp::create(b, loc, w_one, fry);
      Value wyb_omw2 = arith::MulFOp::create(b, loc, wyb_omw, wyb_omw);
      Value wyb_m0 = arith::MulFOp::create(b, loc, wyb_omw2, wyb_omw);
      Value wyb_a1 = arith::MulFOp::create(b, loc, w_three, wyb_w3);
      Value wyb_b1 = arith::MulFOp::create(b, loc, w_six, wyb_w2);
      Value wyb_c1 = arith::SubFOp::create(b, loc, wyb_a1, wyb_b1);
      Value wyb_m1 = arith::AddFOp::create(b, loc, wyb_c1, w_four);
      Value wyb_a2 = arith::MulFOp::create(b, loc, w_three, wyb_w2);
      Value wyb_b2 = arith::MulFOp::create(b, loc, w_three, fry);
      Value wyb_c2 = arith::SubFOp::create(b, loc, wyb_a2, wyb_a1);
      Value wyb_d2 = arith::AddFOp::create(b, loc, wyb_c2, wyb_b2);
      Value wyb_m2 = arith::AddFOp::create(b, loc, wyb_d2, w_one);
      Value wyb_is0 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, yb, w_zero32);
      Value wyb_is1 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, yb, w_one32);
      Value wyb_is2 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, yb, w_two32);
      Value wyb_s2 = arith::SelectOp::create(b, loc, wyb_is2, wyb_m2, wyb_w3);
      Value wyb_s1 = arith::SelectOp::create(b, loc, wyb_is1, wyb_m1, wyb_s2);
      Value wyb_s0 = arith::SelectOp::create(b, loc, wyb_is0, wyb_m0, wyb_s1);
      Value wyb = arith::MulFOp::create(b, loc, wyb_s0, w_sixth);
      Value wz0_w2 = arith::MulFOp::create(b, loc, frz, frz);
      Value wz0_w3 = arith::MulFOp::create(b, loc, wz0_w2, frz);
      Value wz0_omw = arith::SubFOp::create(b, loc, w_one, frz);
      Value wz0_omw2 = arith::MulFOp::create(b, loc, wz0_omw, wz0_omw);
      Value wz0_m0 = arith::MulFOp::create(b, loc, wz0_omw2, wz0_omw);
      Value wz0_a1 = arith::MulFOp::create(b, loc, w_three, wz0_w3);
      Value wz0_b1 = arith::MulFOp::create(b, loc, w_six, wz0_w2);
      Value wz0_c1 = arith::SubFOp::create(b, loc, wz0_a1, wz0_b1);
      Value wz0_m1 = arith::AddFOp::create(b, loc, wz0_c1, w_four);
      Value wz0_a2 = arith::MulFOp::create(b, loc, w_three, wz0_w2);
      Value wz0_b2 = arith::MulFOp::create(b, loc, w_three, frz);
      Value wz0_c2 = arith::SubFOp::create(b, loc, wz0_a2, wz0_a1);
      Value wz0_d2 = arith::AddFOp::create(b, loc, wz0_c2, wz0_b2);
      Value wz0_m2 = arith::AddFOp::create(b, loc, wz0_d2, w_one);
      Value wz0_is0 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, zc, w_zero32);
      Value wz0_is1 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, zc, w_one32);
      Value wz0_is2 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, zc, w_two32);
      Value wz0_s2 = arith::SelectOp::create(b, loc, wz0_is2, wz0_m2, wz0_w3);
      Value wz0_s1 = arith::SelectOp::create(b, loc, wz0_is1, wz0_m1, wz0_s2);
      Value wz0_s0 = arith::SelectOp::create(b, loc, wz0_is0, wz0_m0, wz0_s1);
      Value wz0 = arith::MulFOp::create(b, loc, wz0_s0, w_sixth);
      Value wz1_w2 = arith::MulFOp::create(b, loc, frz, frz);
      Value wz1_w3 = arith::MulFOp::create(b, loc, wz1_w2, frz);
      Value wz1_omw = arith::SubFOp::create(b, loc, w_one, frz);
      Value wz1_omw2 = arith::MulFOp::create(b, loc, wz1_omw, wz1_omw);
      Value wz1_m0 = arith::MulFOp::create(b, loc, wz1_omw2, wz1_omw);
      Value wz1_a1 = arith::MulFOp::create(b, loc, w_three, wz1_w3);
      Value wz1_b1 = arith::MulFOp::create(b, loc, w_six, wz1_w2);
      Value wz1_c1 = arith::SubFOp::create(b, loc, wz1_a1, wz1_b1);
      Value wz1_m1 = arith::AddFOp::create(b, loc, wz1_c1, w_four);
      Value wz1_a2 = arith::MulFOp::create(b, loc, w_three, wz1_w2);
      Value wz1_b2 = arith::MulFOp::create(b, loc, w_three, frz);
      Value wz1_c2 = arith::SubFOp::create(b, loc, wz1_a2, wz1_a1);
      Value wz1_d2 = arith::AddFOp::create(b, loc, wz1_c2, wz1_b2);
      Value wz1_m2 = arith::AddFOp::create(b, loc, wz1_d2, w_one);
      Value wz1_is0 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                            zc2, w_zero32);
      Value wz1_is1 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, zc2, w_one32);
      Value wz1_is2 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, zc2, w_two32);
      Value wz1_s2 = arith::SelectOp::create(b, loc, wz1_is2, wz1_m2, wz1_w3);
      Value wz1_s1 = arith::SelectOp::create(b, loc, wz1_is1, wz1_m1, wz1_s2);
      Value wz1_s0 = arith::SelectOp::create(b, loc, wz1_is0, wz1_m0, wz1_s1);
      Value wz1 = arith::MulFOp::create(b, loc, wz1_s0, w_sixth);
      Value qx = arith::MulFOp::create(b, loc, qi, wxa);
      Value qxy = arith::MulFOp::create(b, loc, qx, wyb);
      Value g1s = arith::AddIOp::create(b, loc, stx, xa);
      Value g1w = arith::SubIOp::create(b, loc, g1s, k1t);
      Value g1over =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, g1s, k1t);
      Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
      Value g2s = arith::AddIOp::create(b, loc, sty, yb);
      Value g2w = arith::SubIOp::create(b, loc, g2s, k2t);
      Value g2over =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, g2s, k2t);
      Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
      Value bx1 = arith::DivUIOp::create(b, loc, g1, t4);
      Value by1 = arith::DivUIOp::create(b, loc, g2, t4);
      Value brow = arith::MulIOp::create(b, loc, bx1, b2t);
      Value bxy = arith::AddIOp::create(b, loc, brow, by1);
      Value bxyz = arith::MulIOp::create(b, loc, bxy, k3t);
      Value inx = arith::RemUIOp::create(b, loc, g1, t4);
      Value iny = arith::RemUIOp::create(b, loc, g2, t4);
      Value iny4 = arith::MulIOp::create(b, loc, iny, t4);
      Value inxy = arith::AddIOp::create(b, loc, iny4, inx);
      Value bytes = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(i64Type, realBytes));
      Value bindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bricks);
      Value bbase = arith::IndexCastOp::create(b, loc, i64Type, bindex);
      Value h0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value h1 = arith::ConstantIndexOp::create(b, loc, 1);
      Value h2 = arith::ConstantIndexOp::create(b, loc, 2);
      scf::ForOp::create(
          b, loc, h0, h2, h1, ValueRange{},
          [&](OpBuilder &b, Location, Value h, ValueRange carried) {
            Value second =
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, h, h0);
            Value c = arith::SelectOp::create(b, loc, second, zc2, zc);
            Value wz = arith::SelectOp::create(b, loc, second, wz1, wz0);
            Value g3s = arith::AddIOp::create(b, loc, stz, c);
            Value g3w = arith::SubIOp::create(b, loc, g3s, k3t);
            Value g3over = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::uge, g3s, k3t);
            Value g3 = arith::SelectOp::create(b, loc, g3over, g3w, g3s);
            Value slab = arith::AddIOp::create(b, loc, bxyz, g3);
            Value slab16 = arith::MulIOp::create(b, loc, slab, t16);
            Value at = arith::AddIOp::create(b, loc, slab16, inxy);
            Value ati = arith::ExtUIOp::create(b, loc, i64Type, at);
            Value offset = arith::MulIOp::create(b, loc, ati, bytes);
            Value address = arith::AddIOp::create(b, loc, bbase, offset);
            Value pointer =
                LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
            Value value = arith::MulFOp::create(b, loc, qxy, wz);
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
  arith::ConstantIndexOp::create(b, loc, 8);
  Value c17 = arith::ConstantIndexOp::create(b, loc, 17);
  Value c31 = arith::ConstantIndexOp::create(b, loc, 31);
  Value k3_31 = arith::AddIOp::create(b, loc, k3, c31);
  Value zchunks = arith::DivUIOp::create(b, loc, k3_31, c32);
  Value copy_blocks = arith::MulIOp::create(b, loc, b12, zchunks);
  {
    auto launch = gpu::LaunchOp::create(b, loc, copy_blocks, c1, c1, c128, c1,
                                        c1, Value{}, Type{}, ValueRange{},
                                        TypeRange{shared(544, realType)});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value tile = launch.getWorkgroupAttributionBBArgs()[0];
    Value bxy = arith::DivUIOp::create(b, loc, bx, zchunks);
    Value zc = arith::RemUIOp::create(b, loc, bx, zchunks);
    Value z0 = arith::MulIOp::create(b, loc, zc, c32);
    Value bx1 = arith::DivUIOp::create(b, loc, bxy, b2);
    Value by1 = arith::RemUIOp::create(b, loc, bxy, b2);
    Value slab0 = arith::MulIOp::create(b, loc, bxy, k3);
    Value zero_r =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value bytes =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, realBytes));
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
          auto inBranch =
              scf::IfOp::create(b, loc, TypeRange{}, in, true, false);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(inBranch.thenBlock());
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
            LLVM::StoreOp::create(b, loc, zero_r, pointer, realBytes);
            scf::YieldOp::create(b, loc);
          }
          scf::YieldOp::create(b, loc);
        });
    Value warp = arith::DivUIOp::create(b, loc, tx, c32);
    Value lane = arith::RemUIOp::create(b, loc, tx, c32);
    Value z = arith::AddIOp::create(b, loc, z0, lane);
    Value in = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, z, k3);
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
          Value in1 =
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, g1, k1);
          Value in2 =
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, g2, k2);
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
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitSpreadBricks<false>(StringRef);
template void DevicePME::emitSpreadBricks<true>(StringRef);

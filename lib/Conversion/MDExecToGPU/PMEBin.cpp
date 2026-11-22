// Count and compact particles into tile bins and the overflow list (D282).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool useList, int tileEdge>
void DevicePME::emitBin(StringRef symbol) {
  StringRef blockFunction = "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
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
  Value bins = entry->getArgument(arg++);
  Value realBuffer = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value n_all = memref::DimOp::create(b, loc, q, c0);
  Value edge = arith::ConstantIndexOp::create(b, loc, tileEdge);
  Value edge32 =
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, tileEdge));
  Value edge_m1 = arith::ConstantIndexOp::create(b, loc, tileEdge - 1);
  Value k1e = addi(k1, edge_m1);
  Value nb1 = divui(k1e, edge);
  Value k2e = addi(k2, edge_m1);
  Value nb2 = divui(k2e, edge);
  Value k3e = addi(k3, edge_m1);
  Value nb3 = divui(k3e, edge);
  Value nb12 = muli(nb1, nb2);
  Value nbins = muli(nb12, nb3);
  Value nbins_m1 = subi(nbins, c1);
  Value mean_u = addi(n_all, nbins_m1);
  Value mean = divui(mean_u, nbins);
  Value mean2 = addi(mean, mean);
  Value slack = arith::ConstantIndexOp::create(b, loc, 8);
  Value capacity = addi(mean2, slack);
  Value of_lists = addi(nbins, c1);
  Value lists = muli(nbins, capacity);
  Value of_overflow = addi(of_lists, lists);
  Value points = memref::DimOp::create(b, loc, realBuffer, c0);
  Value blocks_points =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{points})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_points, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;

    Value tx = launch.getThreadIds().x;

    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, points);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value none =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      store(none, realBuffer, {item});
      Value counts_end = addi(nbins, c1);
      Value a_count = cmpi(arith::CmpIPredicate::ult, item, counts_end);
      auto a_countBranch =
          scf::IfOp::create(b, loc, TypeRange{}, a_count, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(a_countBranch.thenBlock());
        Value none32 =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        store(none32, bins, {item});
        scf::YieldOp::create(b, loc);
      }
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  Value blocks_entries =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{n_all})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_entries, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, n_all);
    Value no = arith::ConstantOp::create(b, loc, b.getBoolAttr(false));
    Value n_last = subi(n_all, c1);
    Value at_item = minui(item, n_last);
    Value item32 = indexCast(i32Type, item);
    Value t32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value valid;
    Value i;
    if constexpr (!useList) {
      valid = ori(inside, no);
      i = addi(at_item, c0);
    } else {
      Value listed_raw = load(list, {at_item});
      Value listed_none =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value is_listed =
          cmpi(arith::CmpIPredicate::sge, listed_raw, listed_none);
      valid = andi(inside, is_listed);
      Value listed_0 = select(is_listed, listed_raw, listed_none);
      i = indexCast(idxType, listed_0);
    }
    Value xc = arith::ConstantIndexOp::create(b, loc, 0);
    Value xn = muli(n_all, xc);
    Value xat = addi(xn, i);
    Value xstart_r = load(places, {xat});
    Value xstart = fptosi(i32Type, xstart_r);
    Value xbin = divui(xstart, edge32);
    Value yc = arith::ConstantIndexOp::create(b, loc, 1);
    Value yn = muli(n_all, yc);
    Value yat = addi(yn, i);
    Value ystart_r = load(places, {yat});
    Value ystart = fptosi(i32Type, ystart_r);
    Value ybin = divui(ystart, edge32);
    Value zc = arith::ConstantIndexOp::create(b, loc, 2);
    Value zn = muli(n_all, zc);
    Value zat = addi(zn, i);
    Value zstart_r = load(places, {zat});
    Value zstart = fptosi(i32Type, zstart_r);
    Value zbin = divui(zstart, edge32);
    Value nb2t = indexCast(i32Type, nb2);
    Value nb3t = indexCast(i32Type, nb3);
    Value bxy = muli(xbin, nb2t);
    Value bxy2 = addi(bxy, ybin);
    Value bxyz = muli(bxy2, nb3t);
    Value bin_of = addi(bxyz, zbin);
    Value no_bin =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, -1));
    Value bin = select(valid, bin_of, no_bin);
    Value same =
        LLVM::InlineAsmOp::create(b, loc, TypeRange{i32Type}, ValueRange{bin},
                                  "match.any.sync.b32 $0, $1, -1;", "=r,r",
                                  true, false, LLVM::TailCallKind::None,
                                  LLVM::AsmDialectAttr{}, ArrayAttr{})
            .getResult(0);
    Value leader = math::CountTrailingZerosOp::create(b, loc, same);
    Value lane = remui(item32, t32);
    Value one32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
    Value below_bit = shli(one32, lane);
    Value below = subi(below_bit, one32);
    Value same_below = andi(same, below);
    Value offset = math::CtPopOp::create(b, loc, same_below);
    Value together = math::CtPopOp::create(b, loc, same);
    Value leads = cmpi(arith::CmpIPredicate::eq, lane, leader);
    Value counts = andi(leads, valid);
    Value bins_index =
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bins);
    Value bins_base = indexCast(i64Type, bins_index);
    Value four64 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 4));
    Value zero32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
    auto first_rankOp =
        scf::IfOp::create(b, loc, TypeRange{i32Type}, counts, true, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(first_rankOp.thenBlock());
      Value bin64 = arith::ExtUIOp::create(b, loc, i64Type, bin);
      Value bin_offset = muli(bin64, four64);
      Value bin_address = addi(bins_base, bin_offset);
      Value bin_pointer =
          LLVM::IntToPtrOp::create(b, loc, globalPointer, bin_address);
      Value old = LLVM::AtomicRMWOp::create(
          b, loc, LLVM::AtomicBinOp::add, bin_pointer, together,
          LLVM::AtomicOrdering::monotonic, "device");
      scf::YieldOp::create(b, loc, ValueRange{old});
    }
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(first_rankOp.elseBlock());
      scf::YieldOp::create(b, loc, ValueRange{zero32});
    }
    Value first_rank = first_rankOp.getResult(0);
    Value width =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value base_rank =
        laneValue(gpu::ShuffleMode::IDX, first_rank, leader, width);

    Value rank = addi(base_rank, offset);
    Value capacity32 = indexCast(i32Type, capacity);
    Value fits = cmpi(arith::CmpIPredicate::slt, rank, capacity32);
    Value particle = indexCast(i32Type, i);
    auto validBranch =
        scf::IfOp::create(b, loc, TypeRange{}, valid, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(validBranch.thenBlock());
      auto fitsBranch =
          scf::IfOp::create(b, loc, TypeRange{}, fits, true, true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(fitsBranch.thenBlock());
        Value bin_i = indexCast(idxType, bin);
        Value run = muli(bin_i, capacity);
        Value run_at = addi(of_lists, run);
        Value rank_i = indexCast(idxType, rank);
        Value place = addi(run_at, rank_i);
        store(particle, bins, {place});
        scf::YieldOp::create(b, loc);
      }
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(fitsBranch.elseBlock());
        Value over_offset = indexCast(i64Type, nbins);
        Value over_bytes = muli(over_offset, four64);
        Value over_address = addi(bins_base, over_bytes);
        Value over_pointer =
            LLVM::IntToPtrOp::create(b, loc, globalPointer, over_address);
        Value over = LLVM::AtomicRMWOp::create(
            b, loc, LLVM::AtomicBinOp::add, over_pointer, one32,
            LLVM::AtomicOrdering::monotonic, "device");
        Value over_i = indexCast(idxType, over);
        Value over_at = addi(of_overflow, over_i);
        store(particle, bins, {over_at});
        scf::YieldOp::create(b, loc);
      }
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitBin<false, 4>(StringRef);
template void DevicePME::emitBin<true, 4>(StringRef);
template void DevicePME::emitBin<false, 8>(StringRef);
template void DevicePME::emitBin<true, 8>(StringRef);
template void DevicePME::emitBin<false, 16>(StringRef);
template void DevicePME::emitBin<true, 16>(StringRef);

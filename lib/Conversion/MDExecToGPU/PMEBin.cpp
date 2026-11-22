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
  Value k1e = arith::AddIOp::create(b, loc, k1, edge_m1);
  Value nb1 = arith::DivUIOp::create(b, loc, k1e, edge);
  Value k2e = arith::AddIOp::create(b, loc, k2, edge_m1);
  Value nb2 = arith::DivUIOp::create(b, loc, k2e, edge);
  Value k3e = arith::AddIOp::create(b, loc, k3, edge_m1);
  Value nb3 = arith::DivUIOp::create(b, loc, k3e, edge);
  Value nb12 = arith::MulIOp::create(b, loc, nb1, nb2);
  Value nbins = arith::MulIOp::create(b, loc, nb12, nb3);
  Value nbins_m1 = arith::SubIOp::create(b, loc, nbins, c1);
  Value mean_u = arith::AddIOp::create(b, loc, n_all, nbins_m1);
  Value mean = arith::DivUIOp::create(b, loc, mean_u, nbins);
  Value mean2 = arith::AddIOp::create(b, loc, mean, mean);
  Value slack = arith::ConstantIndexOp::create(b, loc, 8);
  Value capacity = arith::AddIOp::create(b, loc, mean2, slack);
  Value of_lists = arith::AddIOp::create(b, loc, nbins, c1);
  Value lists = arith::MulIOp::create(b, loc, nbins, capacity);
  Value of_overflow = arith::AddIOp::create(b, loc, of_lists, lists);
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

    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value item = arith::AddIOp::create(b, loc, base, tx);
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item, points);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value none =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      store(none, realBuffer, {item});
      Value counts_end = arith::AddIOp::create(b, loc, nbins, c1);
      Value a_count = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                            item, counts_end);
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
    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value item = arith::AddIOp::create(b, loc, base, tx);
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item, n_all);
    Value no = arith::ConstantOp::create(b, loc, b.getBoolAttr(false));
    Value n_last = arith::SubIOp::create(b, loc, n_all, c1);
    Value at_item = arith::MinUIOp::create(b, loc, item, n_last);
    Value item32 = arith::IndexCastOp::create(b, loc, i32Type, item);
    Value t32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value valid;
    Value i;
    if constexpr (!useList) {
      valid = arith::OrIOp::create(b, loc, inside, no);
      i = arith::AddIOp::create(b, loc, at_item, c0);
    } else {
      Value listed_raw = load(list, {at_item});
      Value listed_none =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value is_listed = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                              listed_raw, listed_none);
      valid = arith::AndIOp::create(b, loc, inside, is_listed);
      Value listed_0 =
          arith::SelectOp::create(b, loc, is_listed, listed_raw, listed_none);
      i = arith::IndexCastOp::create(b, loc, idxType, listed_0);
    }
    Value xc = arith::ConstantIndexOp::create(b, loc, 0);
    Value xn = arith::MulIOp::create(b, loc, n_all, xc);
    Value xat = arith::AddIOp::create(b, loc, xn, i);
    Value xstart_r = load(places, {xat});
    Value xstart = arith::FPToSIOp::create(b, loc, i32Type, xstart_r);
    Value xbin = arith::DivUIOp::create(b, loc, xstart, edge32);
    Value yc = arith::ConstantIndexOp::create(b, loc, 1);
    Value yn = arith::MulIOp::create(b, loc, n_all, yc);
    Value yat = arith::AddIOp::create(b, loc, yn, i);
    Value ystart_r = load(places, {yat});
    Value ystart = arith::FPToSIOp::create(b, loc, i32Type, ystart_r);
    Value ybin = arith::DivUIOp::create(b, loc, ystart, edge32);
    Value zc = arith::ConstantIndexOp::create(b, loc, 2);
    Value zn = arith::MulIOp::create(b, loc, n_all, zc);
    Value zat = arith::AddIOp::create(b, loc, zn, i);
    Value zstart_r = load(places, {zat});
    Value zstart = arith::FPToSIOp::create(b, loc, i32Type, zstart_r);
    Value zbin = arith::DivUIOp::create(b, loc, zstart, edge32);
    Value nb2t = arith::IndexCastOp::create(b, loc, i32Type, nb2);
    Value nb3t = arith::IndexCastOp::create(b, loc, i32Type, nb3);
    Value bxy = arith::MulIOp::create(b, loc, xbin, nb2t);
    Value bxy2 = arith::AddIOp::create(b, loc, bxy, ybin);
    Value bxyz = arith::MulIOp::create(b, loc, bxy2, nb3t);
    Value bin_of = arith::AddIOp::create(b, loc, bxyz, zbin);
    Value no_bin =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, -1));
    Value bin = arith::SelectOp::create(b, loc, valid, bin_of, no_bin);
    Value same =
        LLVM::InlineAsmOp::create(b, loc, TypeRange{i32Type}, ValueRange{bin},
                                  "match.any.sync.b32 $0, $1, -1;", "=r,r",
                                  true, false, LLVM::TailCallKind::None,
                                  LLVM::AsmDialectAttr{}, ArrayAttr{})
            .getResult(0);
    Value leader = math::CountTrailingZerosOp::create(b, loc, same);
    Value lane = arith::RemUIOp::create(b, loc, item32, t32);
    Value one32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
    Value below_bit = arith::ShLIOp::create(b, loc, one32, lane);
    Value below = arith::SubIOp::create(b, loc, below_bit, one32);
    Value same_below = arith::AndIOp::create(b, loc, same, below);
    Value offset = math::CtPopOp::create(b, loc, same_below);
    Value together = math::CtPopOp::create(b, loc, same);
    Value leads =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, leader);
    Value counts = arith::AndIOp::create(b, loc, leads, valid);
    Value bins_index =
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, bins);
    Value bins_base = arith::IndexCastOp::create(b, loc, i64Type, bins_index);
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
      Value bin_offset = arith::MulIOp::create(b, loc, bin64, four64);
      Value bin_address = arith::AddIOp::create(b, loc, bins_base, bin_offset);
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
    auto base_rankShuffle = gpu::ShuffleOp::create(
        b, loc, first_rank, leader, width, gpu::ShuffleMode::IDX);
    Value base_rank = base_rankShuffle.getShuffleResult();

    Value rank = arith::AddIOp::create(b, loc, base_rank, offset);
    Value capacity32 = arith::IndexCastOp::create(b, loc, i32Type, capacity);
    Value fits = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, rank,
                                       capacity32);
    Value particle = arith::IndexCastOp::create(b, loc, i32Type, i);
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
        Value bin_i = arith::IndexCastOp::create(b, loc, idxType, bin);
        Value run = arith::MulIOp::create(b, loc, bin_i, capacity);
        Value run_at = arith::AddIOp::create(b, loc, of_lists, run);
        Value rank_i = arith::IndexCastOp::create(b, loc, idxType, rank);
        Value place = arith::AddIOp::create(b, loc, run_at, rank_i);
        store(particle, bins, {place});
        scf::YieldOp::create(b, loc);
      }
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(fitsBranch.elseBlock());
        Value over_offset = arith::IndexCastOp::create(b, loc, i64Type, nbins);
        Value over_bytes = arith::MulIOp::create(b, loc, over_offset, four64);
        Value over_address =
            arith::AddIOp::create(b, loc, bins_base, over_bytes);
        Value over_pointer =
            LLVM::IntToPtrOp::create(b, loc, globalPointer, over_address);
        Value over = LLVM::AtomicRMWOp::create(
            b, loc, LLVM::AtomicBinOp::add, over_pointer, one32,
            LLVM::AtomicOrdering::monotonic, "device");
        Value over_i = arith::IndexCastOp::create(b, loc, idxType, over);
        Value over_at = arith::AddIOp::create(b, loc, of_overflow, over_i);
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

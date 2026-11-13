// The device build of the groups of 16 particles that share a list of
// neighbors (D89, docs/neighbor-groups.md); see
// include/mdir/Conversion/GroupsBuild.h.
//
// The build follows the cluster pair lists of [Pall2013]: particles close
// in space share one list. The keys in brackets are those of
// docs/references.md.
//
// The particles are put in a compact order: the columns of a grid in x-y,
// each of the width of about 64 particles, sorted by z (a counting sort by
// column and bin of z, the particles of a bin by number), cut into chunks
// of 64, each sorted by x and halved, each half sorted by y and halved: 4
// groups of up to 16 particles that fill a near-cube. The places of a chunk
// are 64 consecutive places; those of a short group are empty (`order` is
// -1 there). Every sort breaks ties by the number of the particle, so the
// order does not depend on the threads.
//
// The list of group g holds, for every particle j at a place q within
// `reach` of one of its particles (in the minimum image) and not before
// the group (q >= 16 g), the entry (q, m): bit u of m is set when the pair
// of the particle at place 16 g + u and j is within the reach, is not an
// excluded pair of `excluded` (an incidence structure of pairs, as for the
// matrix), and, within the group itself, u is before q. Each pair within
// the reach is in one list once, for each of its images within the reach.
// An entry takes one image of its candidate, the one nearest to the center
// of the box of the group; where the box and the reach are more than half
// the cell wide along an axis, a pair can need another, and a second
// kernel adds entries for the images a cell away along each axis to the
// lists of such groups, continuing each where the first left it (D115).
// These are every image within the reach while the reach is less than the
// least edge of the cell (D242). The candidates come from a grid of cells
// of half the reach over the places, each cell in the order of its places,
// visited in a fixed order, so the order of the entries of a group does not
// depend on the threads either. The layout is that of [SalomonFerrer2013];
// every pair is tested against the reach.
//
// The lists are held in blocks of 64 entries, which a group takes from
// `entries` and `masks` as it fills them, with an atomic addition: block b
// holds entries 64 b to 64 b + 63, and is block `ordinals[b]` of the list
// of group `units[b]`, which holds `counts[g]` entries. A block is the
// unit of work of the loops. The lists take the room they need, not a
// fixed room for each group.
//
// On return, `sizes` (on the host) holds the number of places, the largest
// number of entries of a group, the number of blocks, the number of groups
// whose partners of excluded pairs did not fit the memory of a warp (D106),
// and the number of positions that are not numbers, which get no place
// (D107). A number of places or blocks beyond the capacity of its buffer
// means that the structure is not complete; the caller makes more room and
// builds again.
//
// A kernel runs in blocks of 128 threads, a thread or a warp an item. The
// buffers are on the device; positions are in the cell `box`, periodic.
// The search computes in f32, on positions wrapped into the cell.
//
// In a triclinic cell (docs/triclinic.md, P3) the cell is the
// lower-triangular a_x, b_y, c_z, b_x, c_x, c_y. The positions are wrapped
// into the brick [0, a_x) x [0, b_y) x [0, c_z), along c, b, and a in turn,
// and put in the compact order and the grid of the candidates of the
// brick. Across a face of z the row of the grid t_z cells beyond it is that
// of c t_z away, x and y with it, and across a face of y that of b t_y
// away: the window of y of each row of z, and of x of each row of y, moves
// by those tilts. A range of z wider than the grid makes those of y and x
// the whole grid, and one of y that of x, so that each cell is visited once
// and the image of a candidate is the one nearest to the center of the box.
// The minimum image is taken in one pass along c, b, and a, and the shifts
// of the frames and of the entries are lattice vectors n, applied as n H;
// the shift of an entry takes five bits a lattice vector, (e + 16) in bits
// 16 to 30 of its mask. The images kernel of D115 takes the images a
// lattice vector away, by their codes (emitOffsetTriclinic).
//
// The code is emitted op by op in the order in which it runs. The scans,
// the scatters, the sorts within bins, and the reads of the ragged rows are
// md_alg ops (D406), which convert-md-exec-to-gpu lowers by their
// emitters after the build is emitted; the warp's votes and scans are the
// emitters' code, emitted where they stand (CompactionEmitter.h).

#include "mdir/Conversion/GroupsBuild.h"

#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/MDAlg/MDAlgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mdir;

namespace {

using Values = SmallVector<Value>;
using arith::CmpFPredicate;
using arith::CmpIPredicate;

/// The ops of the build, one a call, created where the builder stands.
/// Every op is created by a statement of its own: the order of the
/// arguments of a call in C++ is not fixed, and the order of the ops is
/// that of the code.
class Code {
public:
  Code(OpBuilder &b, Location loc)
      : b(b), loc(loc), context(b.getContext()), i1(b.getI1Type()),
        i32(b.getI32Type()), i64(b.getI64Type()), index(b.getIndexType()),
        f32(b.getF32Type()), f64(b.getF64Type()) {
    Attribute global = b.getI64IntegerAttr(1);
    Attribute workgroup = gpu::AddressSpaceAttr::get(
        context, gpu::GPUDialect::getWorkgroupAddressSpace());
    buffer = MemRefType::get({ShapedType::kDynamic}, i32,
                             MemRefLayoutAttrInterface(), global);
    wideBuffer = MemRefType::get({ShapedType::kDynamic}, i64,
                                 MemRefLayoutAttrInterface(), global);
    sizesType = MemRefType::get({5}, i32, MemRefLayoutAttrInterface(), global);
    positions = MemRefType::get({ShapedType::kDynamic, 3}, f64,
                                MemRefLayoutAttrInterface(), global);
    wrappedType = MemRefType::get({ShapedType::kDynamic, 3}, f32,
                                  MemRefLayoutAttrInterface(), global);
    placesType = MemRefType::get({ShapedType::kDynamic, 4}, f32,
                                 MemRefLayoutAttrInterface(), global);
    boxesType = MemRefType::get({ShapedType::kDynamic, 8}, f32,
                                MemRefLayoutAttrInterface(), global);
    ids = MemRefType::get({256}, i32, MemRefLayoutAttrInterface(), workgroup);
    relative = MemRefType::get({64, 4}, f32, MemRefLayoutAttrInterface(),
                               workgroup);
    partnersType = MemRefType::get({1024}, i32, MemRefLayoutAttrInterface(),
                                   workgroup);
    pointer = LLVM::LLVMPointerType::get(context, 1);
  }

  OpBuilder &b;
  Location loc;
  MLIRContext *context;
  Type i1, i32, i64, index, f32, f64;
  MemRefType buffer, wideBuffer, sizesType, positions, wrappedType,
      placesType, boxesType, ids, relative, partnersType;
  Type pointer;

  // Constants.
  Value ci(int64_t v) {
    return arith::ConstantOp::create(b, loc, i32, b.getIntegerAttr(i32, v));
  }
  Value cl(int64_t v) {
    return arith::ConstantOp::create(b, loc, i64, b.getIntegerAttr(i64, v));
  }
  Value cx(int64_t v) { return arith::ConstantIndexOp::create(b, loc, v); }
  Value cb(bool v) {
    return arith::ConstantOp::create(b, loc, i1, b.getIntegerAttr(i1, v));
  }
  Value cf(double v) {
    return arith::ConstantOp::create(b, loc, f32, b.getFloatAttr(f32, v));
  }
  Value cd(double v) {
    return arith::ConstantOp::create(b, loc, f64, b.getFloatAttr(f64, v));
  }

  // Arithmetic.
#define MDIR_BINARY(name, Op)                                                  \
  Value name(Value x, Value y) { return arith::Op::create(b, loc, x, y); }
  MDIR_BINARY(addi, AddIOp)
  MDIR_BINARY(subi, SubIOp)
  MDIR_BINARY(muli, MulIOp)
  MDIR_BINARY(divui, DivUIOp)
  MDIR_BINARY(divsi, DivSIOp)
  MDIR_BINARY(remui, RemUIOp)
  MDIR_BINARY(remsi, RemSIOp)
  MDIR_BINARY(andi, AndIOp)
  MDIR_BINARY(ori, OrIOp)
  MDIR_BINARY(xori, XOrIOp)
  MDIR_BINARY(shli, ShLIOp)
  MDIR_BINARY(shrui, ShRUIOp)
  MDIR_BINARY(shrsi, ShRSIOp)
  MDIR_BINARY(minsi, MinSIOp)
  MDIR_BINARY(maxsi, MaxSIOp)
  MDIR_BINARY(minui, MinUIOp)
  MDIR_BINARY(maxui, MaxUIOp)
  MDIR_BINARY(addf, AddFOp)
  MDIR_BINARY(subf, SubFOp)
  MDIR_BINARY(mulf, MulFOp)
  MDIR_BINARY(divf, DivFOp)
  MDIR_BINARY(maximumf, MaximumFOp)
  MDIR_BINARY(minimumf, MinimumFOp)
  MDIR_BINARY(maxnumf, MaxNumFOp)
  MDIR_BINARY(minnumf, MinNumFOp)
#undef MDIR_BINARY
  Value cmpi(CmpIPredicate p, Value x, Value y) {
    return arith::CmpIOp::create(b, loc, p, x, y);
  }
  Value cmpf(CmpFPredicate p, Value x, Value y) {
    return arith::CmpFOp::create(b, loc, p, x, y);
  }
  Value select(Value c, Value x, Value y) {
    return arith::SelectOp::create(b, loc, c, x, y);
  }
  Value indexCast(Type type, Value v) {
    return arith::IndexCastOp::create(b, loc, type, v);
  }
  Value sitofp(Type type, Value v) {
    return arith::SIToFPOp::create(b, loc, type, v);
  }
  Value fptosi(Type type, Value v) {
    return arith::FPToSIOp::create(b, loc, type, v);
  }
  Value truncf(Value v) { return arith::TruncFOp::create(b, loc, f32, v); }
  Value trunci(Value v) { return arith::TruncIOp::create(b, loc, i32, v); }
  Value extui(Value v) { return arith::ExtUIOp::create(b, loc, i32, v); }
  Value bitcast(Type type, Value v) {
    return arith::BitcastOp::create(b, loc, type, v);
  }
  Value floor(Value v) { return math::FloorOp::create(b, loc, v); }
  Value roundeven(Value v) { return math::RoundEvenOp::create(b, loc, v); }
  Value absf(Value v) { return math::AbsFOp::create(b, loc, v); }
  Value powf(Value x, Value y) { return math::PowFOp::create(b, loc, x, y); }

  // Memory, with no attributes (the parser of a template sets none).
  Value load(Value memref, ValueRange indices) {
    return memref::LoadOp::create(b, loc, memref, indices,
                                  /*nontemporal=*/BoolAttr(),
                                  /*alignment=*/IntegerAttr(),
                                  /*invariant=*/BoolAttr());
  }
  void store(Value v, Value memref, ValueRange indices) {
    memref::StoreOp::create(b, loc, v, memref, indices,
                            /*nontemporal=*/BoolAttr(),
                            /*alignment=*/IntegerAttr());
  }
  Value dim(Value memref, Value i) {
    return memref::DimOp::create(b, loc, memref, i);
  }
  Value alloc(MemRefType type, ValueRange sizes) {
    return gpu::AllocOp::create(b, loc, type, /*asyncToken=*/Type(),
                                ValueRange{}, sizes, ValueRange{})
        .getMemref();
  }
  /// A buffer of the device, freed: its memory space taken away, as
  /// gpu.dealloc takes it.
  void dealloc(Value memref) {
    auto type = cast<MemRefType>(memref.getType());
    auto plain = MemRefType::get(type.getShape(), type.getElementType());
    Value cast = memref::MemorySpaceCastOp::create(b, loc, plain, memref);
    gpu::DeallocOp::create(b, loc, /*asyncToken=*/Type(), ValueRange{}, cast);
  }
  Value extract(Value vector, int64_t at) {
    return vector::ExtractOp::create(b, loc, vector, ArrayRef<int64_t>{at});
  }

  // The device.
  std::pair<Value, Value> shuffle(gpu::ShuffleMode mode, Value v,
                                  Value offset, Value width) {
    auto op = gpu::ShuffleOp::create(b, loc, v, offset, width, mode);
    return {op.getShuffleResult(), op.getValid()};
  }
  void warpSync(Value mask) { NVVM::SyncWarpOp::create(b, loc, mask); }
  /// The address of `memref` plus `offset` bytes (an i64 already made),
  /// as a pointer of the device: the aligned pointer, cast to i64.
  Value pointerTo(Value base64, Value offset) {
    Value address = addi(base64, offset);
    return LLVM::IntToPtrOp::create(b, loc, pointer, address);
  }
  Value baseAddress(Value memref) {
    Value base = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, memref);
    return indexCast(i64, base);
  }
  /// A relaxed atomic at the scope of the device, whose old value it gives.
  Value atomic(LLVM::AtomicBinOp op, Value ptr, Value v) {
    return LLVM::AtomicRMWOp::create(b, loc, op, ptr, v,
                                     LLVM::AtomicOrdering::monotonic,
                                     "device");
  }

  // Calls.
  ResultRange call(StringRef callee, TypeRange results, ValueRange operands) {
    return func::CallOp::create(b, loc, callee, results, operands)
        .getResults();
  }
  Value call1(StringRef callee, Type result, ValueRange operands) {
    return call(callee, result, operands)[0];
  }

  // Structure. The bodies run with the builder in the region.
  void ifThen(Value cond, function_ref<void()> then) {
    scf::IfOp::create(b, loc, cond, [&](OpBuilder &, Location) {
      then();
      scf::YieldOp::create(b, loc);
    });
  }
  void ifThenElse(Value cond, function_ref<void()> then,
                  function_ref<void()> otherwise) {
    scf::IfOp::create(
        b, loc, cond,
        [&](OpBuilder &, Location) {
          then();
          scf::YieldOp::create(b, loc);
        },
        [&](OpBuilder &, Location) {
          otherwise();
          scf::YieldOp::create(b, loc);
        });
  }
  ResultRange ifElse(TypeRange types, Value cond, function_ref<Values()> then,
                     function_ref<Values()> otherwise) {
    auto op = scf::IfOp::create(b, loc, types, cond, /*addThenBlock=*/true,
                                /*addElseBlock=*/true);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.thenBlock());
    Values v = then();
    scf::YieldOp::create(b, loc, v);
    b.setInsertionPointToStart(op.elseBlock());
    Values w = otherwise();
    scf::YieldOp::create(b, loc, w);
    return op.getResults();
  }
  ResultRange forLoop(Value lower, Value upper, Value step, ValueRange inits,
                      function_ref<Values(Value, ValueRange)> body) {
    return scf::ForOp::create(b, loc, lower, upper, step, inits,
                              [&](OpBuilder &, Location, Value iv,
                                  ValueRange args) {
                                Values v = body(iv, args);
                                scf::YieldOp::create(b, loc, v);
                              })
        .getResults();
  }
  ResultRange whileLoop(TypeRange types, ValueRange inits,
                        function_ref<std::pair<Value, Values>(ValueRange)>
                            before,
                        function_ref<Values(ValueRange)> after) {
    return scf::WhileOp::create(
               b, loc, types, inits,
               [&](OpBuilder &, Location, ValueRange args) {
                 auto [cond, v] = before(args);
                 scf::ConditionOp::create(b, loc, cond, v);
               },
               [&](OpBuilder &, Location, ValueRange args) {
                 Values v = after(args);
                 scf::YieldOp::create(b, loc, v);
               })
        .getResults();
  }

  /// A private function of `type` at the builder, whose body `body` writes
  /// from its arguments; with no body, a declaration.
  func::FuncOp function(StringRef name, FunctionType type,
                        function_ref<void(ValueRange)> body = nullptr) {
    auto op = func::FuncOp::create(b, loc, name, type);
    op.setPrivate();
    if (body) {
      OpBuilder::InsertionGuard guard(b);
      Block *entry = op.addEntryBlock();
      b.setInsertionPointToStart(entry);
      body(entry->getArguments());
    }
    return op;
  }
};

/// The names of the functions of the build.
constexpr StringLiteral kGrid = "mdrt_gpu_groups_grid";
constexpr StringLiteral kPartner = "mdrt_gpu_groups_partner";
constexpr StringLiteral kImage = "mdrt_gpu_groups_image";
constexpr StringLiteral kImageTriclinic = "mdrt_gpu_groups_image_triclinic";
constexpr StringLiteral kOffsetTriclinic = "mdrt_gpu_groups_offset_triclinic";

/// The build: its code, and the values of its function that its kernels
/// take, by the names they have in the code.
class GroupsBuild : public Code {
public:
  GroupsBuild(OpBuilder &b, Location loc, bool tilted, bool singlePass)
      : Code(b, loc), tilted(tilted), singlePass(singlePass) {}

  void emit();

private:
  bool tilted, singlePass;

  // The arguments.
  Value x, box, reach, excluded, order, placeOf, entries, masks, counts,
      units, ordinals, shift, sizes;
  // The sizes and the cell.
  Value c0, c1, block, n, capacity, groupCapacity;
  Value lx, ly, lz, ilx, ily, ilz, flx, fly, flz, filx, fily, filz;
  Value tbx, tcx, tcy, ftbx, ftcx, ftcy, freach;
  // The columns and the grid of the candidates.
  Value ncx64, ncy64, ncx, ncy, columns, nzb, gnx32, gny32, gnz32;
  // The buffers.
  Value wrapped, key, keyCount, keyStart, keyCursor, sorted, chunkBase,
      columnChunks, sizesDevice, xp, gkey, gcount, gstart, gcursor, gplace,
      boxes;

  void emitGrid();
  void emitPartner();
  void emitImage();
  void emitImageTriclinic();
  void emitOffsetTriclinic();
  void emitBuild();

  // The helpers of the kernels.
  void launch(Value grid, Value threads, TypeRange workgroup,
              function_ref<void(gpu::LaunchOp)> body);
  void perItem(gpu::LaunchOp op, Value count, function_ref<void(Value)> body);
  void clear(Value grid, Value target, Value count, int64_t value);
  void atomicMax(Value target, Value v);
  std::pair<Value, SmallVector<Value>> scanExecution(Value count);
  void scan(Value countsOf, Value start, Value cursor, Value count);
  void sortByKey(Value count, Value bins, Value keys, Value counts,
                 Value start, Value cursor, Value out, ValueRange writes,
                 ValueRange adds, Value grid, Value binGrid,
                 function_ref<void(Value)> key);
  /// The image of the displacements (dx, dy, dz) nearest to the center:
  /// three calls of the image of an axis, or in a triclinic cell one of
  /// the image in the cell, whose last three results are the lattice
  /// shift.
  Values image(Value dx, Value dy, Value dz);
  Values imageTriclinic(Value dx, Value dy, Value dz);

  // The kernels.
  void emitKeys(Value item);
  void emitChunks(gpu::LaunchOp op);
  std::pair<Values, Values> bitonic(int64_t values, Values keys, Values ids,
                                    Value lane);
  void emitBoxes(gpu::LaunchOp op);
  void reduceBox(Values &values);
  void emitLists(gpu::LaunchOp op, bool images);
};

//===----------------------------------------------------------------------===//
// The helpers
//===----------------------------------------------------------------------===//

// The number of blocks of `block` threads that hold `count` threads.
void GroupsBuild::emitGrid() {
  function(kGrid, b.getFunctionType({index, index}, {index}),
           [&](ValueRange a) {
             Value count = a[0], threads = a[1];
             Value one = cx(1);
             Value last = subi(threads, one);
             Value padded = addi(count, last);
             Value result = divui(padded, threads);
             func::ReturnOp::create(b, loc, result);
           });
}

// The other member of the excluded pair `k` of the row that begins at
// `first` in the incidence structure `excluded`, ragged rows (see the
// build of the matrix).
void GroupsBuild::emitPartner() {
  function(kPartner, b.getFunctionType({buffer, index, index}, {i32}),
           [&](ValueRange a) {
             Value rows = a[0], first = a[1], k = a[2];
             Value i1c = cx(1);
             Value i2c = cx(2);
             Value i3c = cx(3);
             Value i4c = cx(4);
             Value zero = ci(0);
             Value place = md_alg::RaggedEntryOp::create(b, loc, rows, first,
                                                         k, i1c, i4c);
             Value atFirst = cmpi(CmpIPredicate::eq, place, zero);
             Value column = select(atFirst, i3c, i2c);
             Value partner = md_alg::RaggedEntryOp::create(
                 b, loc, rows, first, k, column, i4c);
             func::ReturnOp::create(b, loc, partner);
           });
}

// The image of the difference `d` in a cell of edge `l` (`il` = 1 / l).
void GroupsBuild::emitImage() {
  function(kImage, b.getFunctionType({f32, f32, f32}, {f32}),
           [&](ValueRange a) {
             Value d = a[0], l = a[1], il = a[2];
             Value s = mulf(d, il);
             Value r = roundeven(s);
             Value m = mulf(r, l);
             Value result = subf(d, m);
             func::ReturnOp::create(b, loc, result);
           });
}

// The minimum image of the displacement d in the triclinic cell of the
// diagonal a_x, b_y, c_z and the tilts b_x, c_x, c_y, in one pass along c,
// b, and a, and the lattice shift n that it takes away, r = d - n H, in
// lattice vectors (n_a, n_b, n_c): exact where the nearest image is within
// half of the least of a_x, b_y, c_z (docs/triclinic.md). A displacement
// that is a lattice vector gives its n.
void GroupsBuild::emitImageTriclinic() {
  SmallVector<Type> inputs(12, f32), results(6, f32);
  function(kImageTriclinic, b.getFunctionType(inputs, results),
           [&](ValueRange a) {
             Value dx = a[0], dy = a[1], dz = a[2], ax = a[3], by = a[4],
                   cz = a[5], bx = a[6], cxt = a[7], cy = a[8], iax = a[9],
                   iby = a[10], icz = a[11];
             Value nc0 = mulf(dz, icz);
             Value nc = roundeven(nc0);
             Value sx1 = mulf(nc, cxt);
             Value sy1 = mulf(nc, cy);
             Value sz1 = mulf(nc, cz);
             Value x1 = subf(dx, sx1);
             Value y1 = subf(dy, sy1);
             Value z1 = subf(dz, sz1);
             Value nb0 = mulf(y1, iby);
             Value nb = roundeven(nb0);
             Value sx2 = mulf(nb, bx);
             Value sy2 = mulf(nb, by);
             Value x2 = subf(x1, sx2);
             Value y2 = subf(y1, sy2);
             Value na0 = mulf(x2, iax);
             Value na = roundeven(na0);
             Value sx3 = mulf(na, ax);
             Value x3 = subf(x2, sx3);
             func::ReturnOp::create(b, loc,
                                    ValueRange{x3, y2, z1, na, nb, nc});
           });
}

// The images of a triclinic cell. The list kernel takes for each candidate
// its image nearest to the center of the box of the group, by
// @mdrt_gpu_groups_image_triclinic; the images kernel takes the others
// within the reach of a group whose box and the reach are more than half
// the cell wide along a_x, b_y, or c_z (D115): those a lattice vector
// away, Δn_c from -1 to 1, then y rounded again and Δn_b from -1 to 1, then
// x rounded again and Δn_a from -1 to 1, 27 codes of which 13 is the
// nearest image itself. These are every image with |z| <= 3 c_z / 2, |y|
// <= 3 b_y / 2, and |x| <= 3 a_x / 2 from the center of the box, since the
// image of the pass is within half of c_z, b_y, and a_x of it and each
// rounding takes the next coordinate back within its half. An image within
// the reach R of a box of half-widths h is within h + R of the center along
// each axis, and a box spans at most the brick, h <= (a_x, b_y, c_z) / 2,
// its particles being taken in the image of the pass of its first: the 27
// hold every image within the reach of every box while the reach is less
// than the least of a_x, b_y, c_z, which the builder and the runtime ask
// (D242).
//
// The image of the code `code`, 0 to 26, of a displacement r that
// @mdrt_gpu_groups_image_triclinic reduced: Δn_c = code / 9 - 1 lattice
// vectors c away, then y rounded again and (code / 3) mod 3 - 1 more along
// b, then x rounded again and code mod 3 - 1 more along a; and the lattice
// shift that it takes away besides that of r. Code 13 is r itself (D115).
void GroupsBuild::emitOffsetTriclinic() {
  SmallVector<Type> inputs = {f32, f32, f32, i32, f32, f32,
                              f32, f32, f32, f32, f32, f32};
  SmallVector<Type> results(6, f32);
  function(kOffsetTriclinic, b.getFunctionType(inputs, results),
           [&](ValueRange a) {
             Value rx = a[0], ry = a[1], rz = a[2], code = a[3], ax = a[4],
                   by = a[5], cz = a[6], bx = a[7], cxt = a[8], cy = a[9],
                   iax = a[10], iby = a[11];
             Value three = ci(3);
             Value nine = ci(9);
             Value one = ci(1);
             Value qc = divui(code, nine);
             Value q3 = divui(code, three);
             Value qb = remui(q3, three);
             Value qa = remui(code, three);
             Value dcI = subi(qc, one);
             Value dbI = subi(qb, one);
             Value daI = subi(qa, one);
             Value dc = sitofp(f32, dcI);
             Value db = sitofp(f32, dbI);
             Value da = sitofp(f32, daI);
             Value sx1 = mulf(dc, cxt);
             Value sy1 = mulf(dc, cy);
             Value sz1 = mulf(dc, cz);
             Value x1 = subf(rx, sx1);
             Value y1 = subf(ry, sy1);
             Value z1 = subf(rz, sz1);
             Value nb0 = mulf(y1, iby);
             Value nb1 = roundeven(nb0);
             Value nb = addf(nb1, db);
             Value sx2 = mulf(nb, bx);
             Value sy2 = mulf(nb, by);
             Value x2 = subf(x1, sx2);
             Value y2 = subf(y1, sy2);
             Value na0 = mulf(x2, iax);
             Value na1 = roundeven(na0);
             Value na = addf(na1, da);
             Value sx3 = mulf(na, ax);
             Value x3 = subf(x2, sx3);
             func::ReturnOp::create(b, loc,
                                    ValueRange{x3, y2, z1, na, nb, dc});
           });
}

/// A launch of `grid` blocks of `threads` threads running `body`, with the
/// memory of the block of `workgroup`.
void GroupsBuild::launch(Value grid, Value threads, TypeRange workgroup,
                         function_ref<void(gpu::LaunchOp)> body) {
  auto op = gpu::LaunchOp::create(
      b, loc, grid, c1, c1, threads, c1, c1,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/{}, /*workgroupAttributions=*/workgroup);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(&op.getBody().front());
  body(op);
  gpu::TerminatorOp::create(b, loc);
}

/// A thread an item below `count`: `body(item)`.
void GroupsBuild::perItem(gpu::LaunchOp op, Value count,
                          function_ref<void(Value)> body) {
  Value base = muli(op.getBlockIds().x, block);
  Value item = addi(base, op.getThreadIds().x);
  Value inside = cmpi(CmpIPredicate::ult, item, count);
  ifThen(inside, [&] { body(item); });
}

void GroupsBuild::clear(Value grid, Value target, Value count,
                        int64_t value) {
  launch(grid, block, {}, [&](gpu::LaunchOp op) {
    perItem(op, count, [&](Value item) {
      Value v = ci(value);
      store(v, target, item);
    });
  });
}

/// A relaxed atomic maximum of the i32 `v` into `target[1]`.
void GroupsBuild::atomicMax(Value target, Value v) {
  Value bi = baseAddress(target);
  Value four = cl(4);
  Value ptr = pointerTo(bi, four);
  atomic(LLVM::AtomicBinOp::max, ptr, v);
}

/// The execution of a scan of `count` i32 (`md_alg.scan`): by the
/// three-launch strategy, or with `singlePass` the decoupled look-back
/// (D321; CompactionEmitter.h), in blocks of 256 threads with 4 counts a
/// thread, whose sums of the blocks (or the states of the tiles) this
/// allocates; the scratch and the launch values.
std::pair<Value, SmallVector<Value>> GroupsBuild::scanExecution(Value count) {
  Value perBlock = cx(1024);
  Value threads = cx(256);
  Value blocks0 = call1(kGrid, index, {count, perBlock});
  Value blocks = maxui(blocks0, c1);
  Value size = blocks;
  if (singlePass)
    size = addi(blocks, c1);
  Value scratch = alloc(singlePass ? wideBuffer : buffer, size);
  return {scratch, {blocks, threads, c1}};
}

static md_alg::ScanStrategyAttr scanStrategy(MLIRContext *context,
                                             bool singlePass) {
  return md_alg::ScanStrategyAttr::get(
      context, singlePass ? md_alg::ScanStrategy::SinglePass
                          : md_alg::ScanStrategy::ReduceThenScan);
}

/// The exclusive sums of the `count` i32 of `countsOf` into `start`
/// (count + 1) and, unless it is null, `cursor` (count).
void GroupsBuild::scan(Value countsOf, Value start, Value cursor,
                       Value count) {
  auto [scratch, launch] = scanExecution(count);
  md_alg::ScanOp::create(b, loc, countsOf, count, start, cursor,
                         scanStrategy(context, singlePass), scratch, launch,
                         /*determinism=*/nullptr);
  dealloc(scratch);
}

/// A counting sort of the `count` items into `bins` bins
/// (`md_alg.sort_by_key`), whose counts the caller has cleared: the key of
/// each item from `key(item)`, which writes the slots of its item in
/// `writes` and adds to the counters of `adds`; the scan as above, a
/// thread an item for the key and the scatter, and the particles of a bin
/// sorted by number by insertion, a thread a bin: the bins hold few items,
/// whose order after the atomics is not fixed.
void GroupsBuild::sortByKey(Value count, Value bins, Value keys, Value counts,
                            Value start, Value cursor, Value out,
                            ValueRange writes, ValueRange adds, Value grid,
                            Value binGrid, function_ref<void(Value)> key) {
  auto [scratch, launch] = scanExecution(bins);
  auto op = md_alg::SortByKeyOp::create(
      b, loc, count, bins,
      md_alg::OrderAttr::get(context, md_alg::Order::Stable), keys, counts, start, cursor, out, /*by=*/Value(),
      writes, adds, b.getUnitAttr(), /*every_key=*/UnitAttr(),
      ValueRange{grid, block, c1}, scanStrategy(context, singlePass), scratch,
      launch,
      md_alg::SortStrategyAttr::get(context, md_alg::SortStrategy::Insertion),
      /*sort_buffer=*/Value(), ValueRange{binGrid, block, c1},
      /*determinism=*/nullptr);
  {
    OpBuilder::InsertionGuard guard(b);
    Block *body = b.createBlock(&op.getKey(), {}, {index}, {loc});
    key(body->getArgument(0));
  }
  dealloc(scratch);
}

Values GroupsBuild::image(Value dx, Value dy, Value dz) {
  if (tilted)
    return imageTriclinic(dx, dy, dz);
  Value rx = call1(kImage, f32, {dx, flx, filx});
  Value ry = call1(kImage, f32, {dy, fly, fily});
  Value rz = call1(kImage, f32, {dz, flz, filz});
  return {rx, ry, rz};
}

Values GroupsBuild::imageTriclinic(Value dx, Value dy, Value dz) {
  SmallVector<Type> results(6, f32);
  ResultRange r = call(kImageTriclinic, results,
                       {dx, dy, dz, flx, fly, flz, ftbx, ftcx, ftcy, filx,
                        fily, filz});
  return Values(r.begin(), r.end());
}

//===----------------------------------------------------------------------===//
// The compact order
//===----------------------------------------------------------------------===//

/// The bin of each particle (its column and bin of z), its position
/// wrapped into the cell, and the count of the positions that have no bin:
/// the key region of the counting sort.
void GroupsBuild::emitKeys(Value item) {
  Value i0 = cx(0);
  Value i1c = cx(1);
  Value i2c = cx(2);
  Value one = ci(1);
  Value xi = load(x, {item, i0});
  Value yi = load(x, {item, i1c});
  Value zi = load(x, {item, i2c});
  Value wx, wy, wz;
  if (!tilted) {
    Value x0 = mulf(xi, ilx);
    Value x1 = floor(x0);
    Value x2 = mulf(x1, lx);
    wx = subf(xi, x2);
    Value y0 = mulf(yi, ily);
    Value y1 = floor(y0);
    Value y2 = mulf(y1, ly);
    wy = subf(yi, y2);
    Value z0 = mulf(zi, ilz);
    Value z1 = floor(z0);
    Value z2 = mulf(z1, lz);
    wz = subf(zi, z2);
  } else {
    // Wrapped into the brick: along c, then b, then a.
    Value z0 = mulf(zi, ilz);
    Value z1 = floor(z0);
    Value z2 = mulf(z1, lz);
    wz = subf(zi, z2);
    Value z1y = mulf(z1, tcy);
    Value z1x = mulf(z1, tcx);
    Value yc = subf(yi, z1y);
    Value xc = subf(xi, z1x);
    Value y0 = mulf(yc, ily);
    Value y1 = floor(y0);
    Value y2 = mulf(y1, ly);
    wy = subf(yc, y2);
    Value y1x = mulf(y1, tbx);
    Value xb = subf(xc, y1x);
    Value x0 = mulf(xb, ilx);
    Value x1 = floor(x0);
    Value x2 = mulf(x1, lx);
    wx = subf(xb, x2);
  }
  Value fx = truncf(wx);
  Value fy = truncf(wy);
  Value fz = truncf(wz);
  store(fx, wrapped, {item, i0});
  store(fy, wrapped, {item, i1c});
  store(fz, wrapped, {item, i2c});
  Value ux = mulf(wx, ilx);
  Value uy = mulf(wy, ily);
  Value uz = mulf(wz, ilz);
  Value ncxF = sitofp(f64, ncx64);
  Value ncyF = sitofp(f64, ncy64);
  Value nzb64 = indexCast(i64, nzb);
  Value nzbF = sitofp(f64, nzb64);
  Value vx = mulf(ux, ncxF);
  Value vy = mulf(uy, ncyF);
  Value vz = mulf(uz, nzbF);
  Value kx64 = fptosi(i64, vx);
  Value ky64 = fptosi(i64, vy);
  Value kz64 = fptosi(i64, vz);
  Value kx0 = indexCast(index, kx64);
  Value ky0 = indexCast(index, ky64);
  Value kz0 = indexCast(index, kz64);
  Value ncx1 = subi(ncx, i1c);
  Value ncy1 = subi(ncy, i1c);
  Value nzb1 = subi(nzb, i1c);
  // A position far beyond the cell wraps with a large error, of either
  // sign: its cell is clamped on both sides, so that no key falls outside
  // the table of counts, whatever the position.
  Value kx1 = minsi(kx0, ncx1);
  Value ky1 = minsi(ky0, ncy1);
  Value kz1 = minsi(kz0, nzb1);
  Value kx = maxsi(kx1, i0);
  Value ky = maxsi(ky1, i0);
  Value kz = maxsi(kz1, i0);
  Value colY = muli(ky, ncx);
  Value column = addi(colY, kx);
  Value colZ = muli(column, nzb);
  Value k = addi(colZ, kz);
  // A position that is not a number, or is beyond any cell, has no key and
  // no place, so that the build completes, and is counted in the fifth
  // size: the run stops after the build (D107). The largest coordinate,
  // not their sum, which opposite coordinates could cancel; a NaN in any
  // of them propagates through the maximum.
  Value absX = absf(xi);
  Value absY = absf(yi);
  Value absZ = absf(zi);
  Value absXY = maximumf(absX, absY);
  Value magnitude = maximumf(absXY, absZ);
  Value notNumber = cmpf(CmpFPredicate::UNO, magnitude, magnitude);
  Value farOff = cd(1.0e100);
  Value huge = cmpf(CmpFPredicate::OGT, magnitude, farOff);
  // A position whose wrap does not land in the cell, which happens once
  // its distance from the cell is beyond what the wrap computes exactly
  // (about 10^16 cells), is beyond any cell as well: its wrapped position
  // would send the later stages of the build outside their tables.
  Value uLow = cd(-1.0e-6);
  Value uHigh = cd(1.000001);
  Value uxLow = cmpf(CmpFPredicate::OLT, ux, uLow);
  Value uxHigh = cmpf(CmpFPredicate::OGT, ux, uHigh);
  Value uyLow = cmpf(CmpFPredicate::OLT, uy, uLow);
  Value uyHigh = cmpf(CmpFPredicate::OGT, uy, uHigh);
  Value uzLow = cmpf(CmpFPredicate::OLT, uz, uLow);
  Value uzHigh = cmpf(CmpFPredicate::OGT, uz, uHigh);
  Value outX = ori(uxLow, uxHigh);
  Value outY = ori(uyLow, uyHigh);
  Value outZ = ori(uzLow, uzHigh);
  Value outXY = ori(outX, outY);
  Value outside = ori(outXY, outZ);
  Value hugeOrOutside = ori(huge, outside);
  Value bad = ori(notNumber, hugeOrOutside);
  ifThenElse(
      bad,
      [&] {
        md_alg::NoBinOp::create(b, loc);
        Value bi = baseAddress(sizesDevice);
        Value sixteen = cl(16);
        Value ptr = pointerTo(bi, sixteen);
        atomic(LLVM::AtomicBinOp::add, ptr, one);
      },
      [&] { md_alg::BinOp::create(b, loc, k); });
}

/// A bitonic network sorting `values` (32 or 64) pairs (key f32, id i32)
/// ascending by key, then id, held one or two a lane (position lane + 32 v
/// for value v). Gives the sorted values of the lane.
std::pair<Values, Values> GroupsBuild::bitonic(int64_t values, Values keys,
                                               Values idsOf, Value lane) {
  int64_t perLane = values / 32;
  for (int64_t size = 2; size <= values; size *= 2) {
    for (int64_t stride = size / 2; stride > 0; stride /= 2) {
      if (stride == 32) {
        // The two values of a lane; the whole sequence goes up.
        Value ka = keys[0], kb = keys[1], ia = idsOf[0], ib = idsOf[1];
        Value klt = cmpf(CmpFPredicate::OLT, kb, ka);
        Value keq = cmpf(CmpFPredicate::OEQ, kb, ka);
        Value ilt = cmpi(CmpIPredicate::slt, ib, ia);
        Value tie = andi(keq, ilt);
        Value swap = ori(klt, tie);
        Value na = select(swap, kb, ka);
        Value nb = select(swap, ka, kb);
        Value nia = select(swap, ib, ia);
        Value nib = select(swap, ia, ib);
        keys = {na, nb};
        idsOf = {nia, nib};
        continue;
      }
      Values newKeys, newIds;
      for (int64_t v = 0; v < perLane; ++v) {
        Value k = keys[v], i = idsOf[v];
        Value s = ci(stride);
        Value w = ci(32);
        Value pk = shuffle(gpu::ShuffleMode::XOR, k, s, w).first;
        Value pi = shuffle(gpu::ShuffleMode::XOR, i, s, w).first;
        Value off = ci(32 * v);
        Value pos = addi(lane, off);
        Value sz = ci(size);
        Value zero = ci(0);
        Value upb = andi(pos, sz);
        Value up = cmpi(CmpIPredicate::eq, upb, zero);
        Value lob = andi(pos, s);
        Value lower = cmpi(CmpIPredicate::eq, lob, zero);
        Value plt0 = cmpf(CmpFPredicate::OLT, pk, k);
        Value peq = cmpf(CmpFPredicate::OEQ, pk, k);
        Value pilt = cmpi(CmpIPredicate::slt, pi, i);
        Value pitie = andi(peq, pilt);
        Value plt = ori(plt0, pitie);
        Value pgt0 = cmpf(CmpFPredicate::OGT, pk, k);
        Value pigt = cmpi(CmpIPredicate::sgt, pi, i);
        Value pgtie = andi(peq, pigt);
        Value pgt = ori(pgt0, pgtie);
        // The lower of a pair keeps the smaller when the run goes up.
        Value min = cmpi(CmpIPredicate::eq, up, lower);
        Value take = select(min, plt, pgt);
        Value nk = select(take, pk, k);
        Value ni = select(take, pi, i);
        newKeys.push_back(nk);
        newIds.push_back(ni);
      }
      keys = std::move(newKeys);
      idsOf = std::move(newIds);
    }
  }
  return {keys, idsOf};
}

/// A warp a chunk: the order of its 64 particles.
void GroupsBuild::emitChunks(gpu::LaunchOp op) {
  Value tx = op.getThreadIds().x;
  Value base = muli(op.getBlockIds().x, block);
  Value item = addi(base, tx);
  Value c0w = cx(0);
  Value c1w = cx(1);
  Value c2w = cx(2);
  Value c32w = cx(32);
  Value c64w = cx(64);
  Value c2i = ci(2);
  Value c16i = ci(16);
  Value c32i = ci(32);
  Value c64i = ci(64);
  Value zeroI = ci(0);
  Value oneI = ci(1);
  Value t = divui(item, c32w);
  Value lane = remui(tx, c32w);
  Value lane32 = indexCast(i32, lane);
  Value warp = divui(tx, c32w);
  Value places32 = load(sizesDevice, c0w);
  Value places = indexCast(index, places32);
  Value totalChunks = divui(places, c64w);
  Value realChunk = cmpi(CmpIPredicate::ult, t, totalChunks);
  Value idsWg = op.getWorkgroupAttributionBBArgs()[0];
  ifThen(realChunk, [&] {
    Value t32 = indexCast(i32, t);
    // The column: the last whose first chunk is not after t.
    ResultRange found = whileLoop(
        {index, index}, {c0w, columns},
        [&](ValueRange a) -> std::pair<Value, Values> {
          Value span = subi(a[1], a[0]);
          Value go = cmpi(CmpIPredicate::ugt, span, c1w);
          return {go, {a[0], a[1]}};
        },
        [&](ValueRange a) -> Values {
          Value lo = a[0], hi = a[1];
          Value sum = addi(lo, hi);
          Value mid = divui(sum, c2w);
          Value firstMid = load(chunkBase, mid);
          Value after = cmpi(CmpIPredicate::sgt, firstMid, t32);
          Value nlo = select(after, lo, mid);
          Value nhi = select(after, mid, hi);
          return {nlo, nhi};
        });
    Value loC = found[0];
    Value first32 = load(chunkBase, loC);
    Value j32 = subi(t32, first32);
    Value keyBegin = muli(loC, nzb);
    Value column1 = addi(loC, c1w);
    Value keyEnd = muli(column1, nzb);
    auto column = md_alg::BinSpanOp::create(b, loc, keyStart, keyBegin, keyEnd);
    Value colBegin32 = column.getBegin();
    Value colEnd32 = column.getEnd();
    Value j64 = muli(j32, c64i);
    Value begin32 = addi(colBegin32, j64);
    Value left32 = subi(colEnd32, begin32);
    Value len32 = minsi(left32, c64i);
    Value begin = indexCast(index, begin32);
    // Two particles a lane: positions lane and lane + 32 of the chunk.
    Value l1 = addi(lane32, c32i);
    Value hasA = cmpi(CmpIPredicate::slt, lane32, len32);
    Value hasB = cmpi(CmpIPredicate::slt, l1, len32);
    Value l1I = indexCast(index, l1);
    Value sa = addi(begin, lane);
    Value sb = addi(begin, l1I);
    Value big = ci(2139095040);
    Value inf = bitcast(f32, big);
    Value none = ci(2147483647);
    Value ia0 = ifElse(
        i32, hasA,
        [&] { return Values{md_alg::BinItemOp::create(b, loc, sorted, sa)}; },
        [&] { return Values{none}; })[0];
    Value ib0 = ifElse(
        i32, hasB,
        [&] { return Values{md_alg::BinItemOp::create(b, loc, sorted, sb)}; },
        [&] { return Values{none}; })[0];
    auto xOf = [&](Value has, Value id) {
      return ifElse(
          f32, has,
          [&] {
            Value ii = indexCast(index, id);
            return Values{load(wrapped, {ii, c0w})};
          },
          [&] { return Values{inf}; })[0];
    };
    Value xa0 = xOf(hasA, ia0);
    Value xb0 = xOf(hasB, ib0);
    auto [k64, i64s] = bitonic(64, {xa0, xb0}, {ia0, ib0}, lane32);
    // The chunk by x, into the memory of the block; then each half by y.
    Value warpSlot0 = muli(warp, c64w);
    Value saAt = addi(warpSlot0, lane);
    Value sbAt = addi(warpSlot0, l1I);
    store(i64s[0], idsWg, saAt);
    store(i64s[1], idsWg, sbAt);
    Value all = ci(-1);
    warpSync(all);
    Value len1 = addi(len32, oneI);
    Value h32 = divsi(len1, c2i);
    Value chunkPlace0 = muli(t, c64w);
    for (int64_t half = 0; half < 2; ++half) {
      Value from = half ? h32 : zeroI;
      Value count = half ? subi(len32, h32) : addi(h32, zeroI);
      Value has = cmpi(CmpIPredicate::slt, lane32, count);
      Value at32 = addi(from, lane32);
      Value at = indexCast(index, at32);
      Value slot = addi(warpSlot0, at);
      Value i0 = ifElse(
          i32, has, [&] { return Values{load(idsWg, slot)}; },
          [&] { return Values{none}; })[0];
      Value y0 = ifElse(
          f32, has,
          [&] {
            Value ii = indexCast(index, i0);
            return Values{load(wrapped, {ii, c1w})};
          },
          [&] { return Values{inf}; })[0];
      auto [k2, i2] = bitonic(32, {y0}, {i0}, lane32);
      // Quarter 2 half or 2 half + 1: the first ceil(count / 2) by y, then
      // the rest.
      Value c1h = addi(count, oneI);
      Value q = divsi(c1h, c2i);
      Value second = cmpi(CmpIPredicate::sge, lane32, q);
      Value rankB = subi(lane32, q);
      Value rank = select(second, rankB, lane32);
      Value quarterB = ci(2 * half + 1);
      Value quarterA = ci(2 * half);
      Value quarter = select(second, quarterB, quarterA);
      Value q16 = muli(quarter, c16i);
      Value inChunk = addi(q16, rank);
      Value inChunkI = indexCast(index, inChunk);
      Value place = addi(chunkPlace0, inChunkI);
      Value id = i2[0];
      ifThen(has, [&] {
        store(id, order, place);
        Value ii = indexCast(index, id);
        Value place32 = indexCast(i32, place);
        store(place32, placeOf, ii);
      });
    }
  });
}

//===----------------------------------------------------------------------===//
// The boxes of the groups
//===----------------------------------------------------------------------===//

/// The least and greatest relative coordinates of the 16 lanes of a
/// half-warp, by shuffles: `values` are ax, ay, az (the least) and bx, by,
/// bz (the greatest).
void GroupsBuild::reduceBox(Values &values) {
  int64_t offsets[] = {8, 4, 2, 1};
  for (int64_t offset : offsets) {
    for (unsigned v = 0; v < 6; ++v) {
      Value s = ci(offset);
      Value w = ci(32);
      Value o = shuffle(gpu::ShuffleMode::XOR, values[v], s, w).first;
      values[v] = v < 3 ? minimumf(values[v], o) : maximumf(values[v], o);
    }
  }
}

/// A half-warp a group, relative to its first particle in the minimum
/// image.
void GroupsBuild::emitBoxes(gpu::LaunchOp op) {
  Value tx = op.getThreadIds().x;
  Value base = muli(op.getBlockIds().x, block);
  Value item = addi(base, tx);
  Value c0w = cx(0);
  Value c1w = cx(1);
  Value c2w = cx(2);
  Value c16w = cx(16);
  Value zero = ci(0);
  Value places32 = load(sizesDevice, c0w);
  Value places = indexCast(index, places32);
  Value groups = divui(places, c16w);
  Value g = divui(item, c16w);
  Value u = remui(item, c16w);
  Value inside = cmpi(CmpIPredicate::ult, g, groups);
  Value gs = select(inside, g, c0w);
  Value first = muli(gs, c16w);
  Value p = addi(first, u);
  Value pi = load(order, p);
  Value real0 = cmpi(CmpIPredicate::sge, pi, zero);
  Value real = andi(inside, real0);
  Value ox = load(xp, {first, c0w});
  Value oy = load(xp, {first, c1w});
  Value oz = load(xp, {first, c2w});
  Value vx = load(xp, {p, c0w});
  Value vy = load(xp, {p, c1w});
  Value vz = load(xp, {p, c2w});
  Value dx = subf(vx, ox);
  Value dy = subf(vy, oy);
  Value dz = subf(vz, oz);
  Values r = image(dx, dy, dz);
  Value rx = r[0], ry = r[1], rz = r[2];
  Value big = cf(1.0e30);
  Value small = cf(-1.0e30);
  Value ax0 = select(real, rx, big);
  Value ay0 = select(real, ry, big);
  Value az0 = select(real, rz, big);
  Value bx0 = select(real, rx, small);
  Value by0 = select(real, ry, small);
  Value bz0 = select(real, rz, small);
  Values extent = {ax0, ay0, az0, bx0, by0, bz0};
  reduceBox(extent);
  // The frame of a group: its first particle where it is wrapped, the
  // others at their places relative to it, as the loops over the groups
  // take it. The shift of the place to that frame, in cells, from its
  // particle as it is kept, unwrapped: ten bits for each axis, from -512.
  // The cells between the frame and the wrapped position, packed the same
  // way, go in the fourth lane of the wrapped position, which the lists
  // read with it; they are -1 to 1, so that the shifts of the entries stay
  // small however far the particles have gone. In a triclinic cell the
  // shifts are lattice vectors.
  ifThenElse(
      real,
      [&] {
        Value qp = indexCast(index, pi);
        Value pxd = load(x, {qp, c0w});
        Value pyd = load(x, {qp, c1w});
        Value pzd = load(x, {qp, c2w});
        Value pxf = truncf(pxd);
        Value pyf = truncf(pyd);
        Value pzf = truncf(pzd);
        Value tx0 = addf(ox, rx);
        Value ty0 = addf(oy, ry);
        Value tz0 = addf(oz, rz);
        Value sx0 = subf(tx0, pxf);
        Value sy0 = subf(ty0, pyf);
        Value sz0 = subf(tz0, pzf);
        // The cells of a displacement, rounded.
        auto cells = [&](Value ex, Value ey, Value ez) -> Values {
          if (tilted) {
            Values s = imageTriclinic(ex, ey, ez);
            return {s[3], s[4], s[5]};
          }
          Value ex1 = mulf(ex, filx);
          Value ey1 = mulf(ey, fily);
          Value ez1 = mulf(ez, filz);
          Value ex2 = roundeven(ex1);
          Value ey2 = roundeven(ey1);
          Value ez2 = roundeven(ez1);
          return {ex2, ey2, ez2};
        };
        Values s2 = cells(sx0, sy0, sz0);
        Value six = fptosi(i32, s2[0]);
        Value siy = fptosi(i32, s2[1]);
        Value siz = fptosi(i32, s2[2]);
        Value wx0 = subf(tx0, vx);
        Value wy0 = subf(ty0, vy);
        Value wz0 = subf(tz0, vz);
        Values w2 = cells(wx0, wy0, wz0);
        Value wix = fptosi(i32, w2[0]);
        Value wiy = fptosi(i32, w2[1]);
        Value wiz = fptosi(i32, w2[2]);
        Value off = ci(512);
        Value ten = ci(10);
        Value twenty = ci(20);
        Value six1 = addi(six, off);
        Value siy1 = addi(siy, off);
        Value siz1 = addi(siz, off);
        Value siy2 = shli(siy1, ten);
        Value siz2 = shli(siz1, twenty);
        Value sxy = ori(six1, siy2);
        Value packed = ori(sxy, siz2);
        store(packed, shift, p);
        Value wix1 = addi(wix, off);
        Value wiy1 = addi(wiy, off);
        Value wiz1 = addi(wiz, off);
        Value wiy2 = shli(wiy1, ten);
        Value wiz2 = shli(wiz1, twenty);
        Value wxy = ori(wix1, wiy2);
        Value wpacked = ori(wxy, wiz2);
        Value wpackedF = bitcast(f32, wpacked);
        Value c3s = cx(3);
        store(wpackedF, xp, {p, c3s});
      },
      [&] {
        // An empty place: no shift, and no cells between the frame and the
        // wrapped position. The lists read the fourth lane at the first
        // place of every group, also of a group that a short chunk leaves
        // without a particle; it is written here, so that no read takes
        // memory that nothing wrote (#413). No entry of such a group uses
        // it.
        Value noneShift = ci(537395712);
        ifThen(inside, [&] {
          store(noneShift, shift, p);
          Value noneF = bitcast(f32, noneShift);
          Value c3e = cx(3);
          store(noneF, xp, {p, c3e});
        });
      });
  Value lead = cmpi(CmpIPredicate::eq, u, c0w);
  Value storeBox = andi(lead, inside);
  ifThen(storeBox, [&] {
    Value loX = addf(ox, extent[0]);
    Value loY = addf(oy, extent[1]);
    Value loZ = addf(oz, extent[2]);
    Value hiX = addf(ox, extent[3]);
    Value hiY = addf(oy, extent[4]);
    Value hiZ = addf(oz, extent[5]);
    Value c3w = cx(3);
    Value c4w = cx(4);
    Value c5w = cx(5);
    store(loX, boxes, {g, c0w});
    store(loY, boxes, {g, c1w});
    store(loZ, boxes, {g, c2w});
    store(hiX, boxes, {g, c3w});
    store(hiY, boxes, {g, c4w});
    store(hiZ, boxes, {g, c5w});
  });
}

//===----------------------------------------------------------------------===//
// The lists
//===----------------------------------------------------------------------===//

// The images of an orthorhombic cell. The list kernel takes for each
// candidate its image nearest to the center of the box of the group, along
// each axis; the images kernel takes the others within the reach of a group
// whose box and the reach are more than half the cell wide along an axis
// (D115): those a cell away or none along each axis, 27 codes of which 13
// is the nearest image itself, code = 9 (o_z + 1) + 3 (o_y + 1) + (o_x + 1)
// for the image r - o L. The nearest image has |r_a| <= L_a / 2, so an
// image two cells away or more is at least 3 L_a / 2 from the center along
// that axis, and an image within the reach R of a box of half-widths h is
// within h_a + R of it: the 27 hold every image within the reach of the box
// while h_a + R < 3 L_a / 2. A box spans at most a cell along an axis, its
// particles being taken in the minimum image of its first, so h_a <= L_a /
// 2, and the condition holds for every group while the reach is less than
// the least edge of the cell, which the builder and the runtime ask (D242,
// #263). The eight images of before, the nearest or the one on the other
// side of the boundary along each axis, held them only while h_a + R <=
// L_a.
//
// A warp a group: its list, or with `images` the entries of the images of
// its candidates other than the nearest to the center of its box, added to
// its list, for a group whose box and the reach are more than half the
// cell wide along an axis (D115).
void GroupsBuild::emitLists(gpu::LaunchOp op, bool images) {
  using P = CmpIPredicate;
  using F = CmpFPredicate;
  Value tx = op.getThreadIds().x;
  Value relWg = op.getWorkgroupAttributionBBArgs()[0];
  Value partners = op.getWorkgroupAttributionBBArgs()[1];
  Value queue = op.getWorkgroupAttributionBBArgs()[2];
  Value base = muli(op.getBlockIds().x, block);
  Value item = addi(base, tx);
  Value c0w = cx(0);
  Value c1w = cx(1);
  Value c2w = cx(2);
  Value c3w = cx(3);
  cx(4);
  Value c16w = cx(16);
  Value c32w = cx(32);
  Value zeroI = ci(0);
  Value oneI = ci(1);
  Value c2i = ci(2);
  Value c16i = ci(16);
  Value c32i = ci(32);
  Value all = ci(-1);
  Value g = divui(item, c32w);
  Value lane = remui(tx, c32w);
  Value lane32 = indexCast(i32, lane);
  Value warp = divui(tx, c32w);
  Value places32 = load(sizesDevice, c0w);
  Value places = indexCast(index, places32);
  Value groups = divui(places, c16w);
  Value realGroup;
  if (!images) {
    realGroup = cmpi(P::ult, g, groups);
  } else {
    // Only those groups: within the reach of the box, an image on the
    // other side is possible only there.
    Value realGroup0 = cmpi(P::ult, g, groups);
    Value imgC0 = cx(0);
    Value imgC1 = cx(1);
    Value imgC2 = cx(2);
    Value imgC3 = cx(3);
    Value imgC4 = cx(4);
    Value imgC5 = cx(5);
    // The warps past the last group read the box of the first, which they
    // do not use: the buffer of the boxes ends with the groups it can
    // hold.
    Value imgG = select(realGroup0, g, imgC0);
    Value lox = load(boxes, {imgG, imgC0});
    Value loy = load(boxes, {imgG, imgC1});
    Value loz = load(boxes, {imgG, imgC2});
    Value hix = load(boxes, {imgG, imgC3});
    Value hiy = load(boxes, {imgG, imgC4});
    Value hiz = load(boxes, {imgG, imgC5});
    Value wx = subf(hix, lox);
    Value wy = subf(hiy, loy);
    Value wz = subf(hiz, loz);
    Value rx = addf(freach, freach);
    Value ex = addf(wx, rx);
    Value ey = addf(wy, rx);
    Value ez = addf(wz, rx);
    Value nx = cmpf(F::OGE, ex, flx);
    Value ny = cmpf(F::OGE, ey, fly);
    Value nz = cmpf(F::OGE, ez, flz);
    Value nxy = ori(nx, ny);
    Value nxyz = ori(nxy, nz);
    realGroup = andi(realGroup0, nxyz);
  }
  ifThen(realGroup, [&] {
    Value first = muli(g, c16w);
    Value first32 = indexCast(i32, first);
    // The particles of the group relative to the center of its box, in the
    // memory of the block: x, y, z, and 1 or 0 for a real one.
    Value boxLoX = load(boxes, {g, c0w});
    Value boxLoY = load(boxes, {g, c1w});
    Value boxLoZ = load(boxes, {g, c2w});
    Value boxHiX = load(boxes, {g, c3w});
    Value c4b = cx(4);
    Value c5b = cx(5);
    Value boxHiY = load(boxes, {g, c4b});
    Value boxHiZ = load(boxes, {g, c5b});
    Value halfF = cf(0.5);
    Value sumX = addf(boxLoX, boxHiX);
    Value sumY = addf(boxLoY, boxHiY);
    Value sumZ = addf(boxLoZ, boxHiZ);
    Value cxb = mulf(sumX, halfF);
    Value cyb = mulf(sumY, halfF);
    Value czb = mulf(sumZ, halfF);
    Value difX = subf(boxHiX, boxLoX);
    Value difY = subf(boxHiY, boxLoY);
    Value difZ = subf(boxHiZ, boxLoZ);
    Value hx = mulf(difX, halfF);
    Value hy = mulf(difY, halfF);
    Value hz = mulf(difZ, halfF);
    // The frame of the group less the wrapped positions, in cells, for its
    // first particle (whose place shift is zero).
    Value wfirstF = load(xp, {first, c3w});
    Value wfirst = bitcast(i32, wfirstF);
    Value rel0 = muli(warp, c16w);
    Value mineLane = cmpi(P::ult, lane, c16w);
    ifThen(mineLane, [&] {
      Value p = addi(first, lane);
      Value pi = load(order, p);
      Value real = cmpi(P::sge, pi, zeroI);
      Value xi = load(xp, {p, c0w});
      Value yi = load(xp, {p, c1w});
      Value zi = load(xp, {p, c2w});
      Value ex0 = subf(xi, cxb);
      Value ey0 = subf(yi, cyb);
      Value ez0 = subf(zi, czb);
      Values r = image(ex0, ey0, ez0);
      Value oneFF = cf(1.0);
      Value zeroFF = cf(0.0);
      Value flag = select(real, oneFF, zeroFF);
      Value at = addi(rel0, lane);
      store(r[0], relWg, {at, c0w});
      store(r[1], relWg, {at, c1w});
      store(r[2], relWg, {at, c2w});
      store(flag, relWg, {at, c3w});
    });
    // The partners of the excluded pairs of the group: (place << 4) | u,
    // sorted, 256 at most; a group with more takes its excluded pairs from
    // their rows (D106).
    Value c256 = cx(256);
    Value pbase = muli(warp, c256);
    Value c64q = cx(64);
    Value qbase = muli(warp, c64q);
    Value largestI = ci(2147483647);
    forLoop(lane, c256, c32w, {}, [&](Value s, ValueRange) -> Values {
      Value at = addi(pbase, s);
      store(largestI, partners, at);
      return {};
    });
    warpSync(all);
    // Each particle of the group in a lane of each half-warp: its partners
    // at even k from the first half, at odd k from the second, from where
    // the counts of the particles before it end (D105).
    Value u16 = remui(lane, c16w);
    Value hk = divui(lane, c16w);
    Value u16x32 = indexCast(i32, u16);
    Value pe = addi(first, u16);
    Value pie = load(order, pe);
    Value piie = indexCast(index, pie);
    Value reale = cmpi(P::sge, pie, zeroI);
    Value entryE = cx(4);
    auto row = md_alg::RaggedRowOp::create(b, loc, excluded, piie, entryE,
                                           /*checked=*/true);
    Value firste = row.getFirst();
    Value counte = select(reale, row.getCount(), c0w);
    Value count32e = indexCast(i32, counte);
    // Shuffles over the whole warp: the halves hold the same counts, so a
    // lane reads one of its own half (shuffles up by segments of 16 lost
    // excluded pairs).
    Value wexcl = ci(32);
    Value pre0 = addi(count32e, zeroI);
    Value pre4 = kernels::emitWarpScan(b, loc, pre0, u16x32, zeroI, wexcl,
                                       /*lanes=*/16, /*unsignedLane=*/true);
    Value start32e = subi(pre4, count32e);
    Value starte = indexCast(index, start32e);
    Value c15i = ci(15);
    Value total32 = shuffle(gpu::ShuffleMode::IDX, pre4, c15i, wexcl).first;
    forLoop(hk, counte, c2w, {}, [&](Value k, ValueRange) -> Values {
      Value partner = call1(kPartner, i32, {excluded, firste, k});
      Value pj = indexCast(index, partner);
      Value q = load(placeOf, pj);
      Value four = ci(4);
      Value shifted = shli(q, four);
      Value partnerKey = ori(shifted, u16x32);
      Value slot0 = addi(starte, k);
      Value fits = cmpi(P::ult, slot0, c256);
      ifThen(fits, [&] {
        Value at = addi(pbase, slot0);
        store(partnerKey, partners, at);
      });
      return {};
    });
    Value npAll = indexCast(index, total32);
    Value np = minui(npAll, c256);
    // Whether the partners did not all fit: then the lists take the
    // excluded pairs from their rows (D106). The count came from one lane,
    // so every lane sees the same.
    Value overflow = cmpi(P::ugt, npAll, c256);
    Value ovLane0 = cmpi(P::eq, lane, c0w);
    Value ovCount = andi(overflow, ovLane0);
    ifThen(ovCount, [&] {
      Value bi = baseAddress(sizesDevice);
      Value twelve = cl(12);
      Value ptr = pointerTo(bi, twelve);
      atomic(LLVM::AtomicBinOp::add, ptr, oneI);
    });
    warpSync(all);
    // The network of the fewest values that hold the partners (D105): a
    // bitonic network over the memory of the warp, 256 values at most
    // (keys (place << 4) | u; empty slots hold the largest i32), 4 pairs a
    // lane a stage.
    auto sortPartners = [&](int64_t span) {
      for (int64_t size = 2; size <= span; size *= 2) {
        for (int64_t stride = size / 2; stride > 0; stride /= 2) {
          for (int64_t r = 0; r < span / 2 / 32; ++r) {
            Value pair = cx(32 * r);
            Value tp = addi(lane, pair);
            Value s = cx(stride);
            Value sm1 = cx(stride - 1);
            Value low = andi(tp, sm1);
            Value twice = addi(tp, tp);
            Value pa = subi(twice, low);
            Value pb = addi(pa, s);
            Value sz = cx(size);
            Value upb = andi(pa, sz);
            Value z = cx(0);
            Value up = cmpi(P::eq, upb, z);
            Value aa = addi(pbase, pa);
            Value bb = addi(pbase, pb);
            Value va = load(partners, aa);
            Value vb = load(partners, bb);
            Value gt = cmpi(P::sgt, va, vb);
            Value swap = cmpi(P::eq, gt, up);
            Value na = select(swap, vb, va);
            Value nb = select(swap, va, vb);
            store(na, partners, aa);
            store(nb, partners, bb);
          }
          warpSync(all);
        }
      }
    };
    Value c64s = cx(64);
    Value c128s = cx(128);
    Value fits64 = cmpi(P::ule, np, c64s);
    Value fits128 = cmpi(P::ule, np, c128s);
    ifThenElse(
        fits64, [&] { sortPartners(64); },
        [&] {
          ifThenElse(
              fits128, [&] { sortPartners(128); },
              [&] { sortPartners(256); });
        });
    // The range of the places of the partners: a candidate outside it has
    // no excluded pair with the group, and is not searched for.
    Value hasPartners = cmpi(P::ne, np, c0w);
    Value npLast0 = subi(np, c1w);
    Value npLast = select(hasPartners, npLast0, c0w);
    Value pminAt = addi(pbase, c0w);
    Value pmaxAt = addi(pbase, npLast);
    Value pminKey = load(partners, pminAt);
    Value pmaxKey = load(partners, pmaxAt);
    Value fourP = ci(4);
    Value pmin = shrsi(pminKey, fourP);
    Value pmax0 = shrsi(pmaxKey, fourP);
    Value noneMax = ci(-1);
    Value pmax = select(hasPartners, pmax0, noneMax);
    // The cells of the grid within the reach of the box.
    Value ecx = addf(hx, freach);
    Value ecy = addf(hy, freach);
    Value ecz = addf(hz, freach);
    Value loxF = subf(cxb, ecx);
    Value loyF = subf(cyb, ecy);
    Value lozF = subf(czb, ecz);
    Value hixF = addf(cxb, ecx);
    Value hiyF = addf(cyb, ecy);
    Value hizF = addf(czb, ecz);
    Value gnxf = sitofp(f32, gnx32);
    Value gnyf = sitofp(f32, gny32);
    Value gnzf = sitofp(f32, gnz32);
    Value rsx0 = mulf(loxF, filx);
    Value rsy0 = mulf(loyF, fily);
    Value rsz0 = mulf(lozF, filz);
    Value sx1 = mulf(rsx0, gnxf);
    Value sy1 = mulf(rsy0, gnyf);
    Value sz1 = mulf(rsz0, gnzf);
    Value rtx0 = mulf(hixF, filx);
    Value rty0 = mulf(hiyF, fily);
    Value rtz0 = mulf(hizF, filz);
    Value tx1 = mulf(rtx0, gnxf);
    Value ty1 = mulf(rty0, gnyf);
    Value tz1 = mulf(rtz0, gnzf);
    Value sxf = floor(sx1);
    Value syf = floor(sy1);
    Value szf = floor(sz1);
    Value txf = floor(tx1);
    Value tyf = floor(ty1);
    Value tzf = floor(tz1);
    Value c0x = fptosi(i32, sxf);
    Value c0y = fptosi(i32, syf);
    Value c0z = fptosi(i32, szf);
    Value c1x0 = fptosi(i32, txf);
    Value c1y0 = fptosi(i32, tyf);
    Value c1z0 = fptosi(i32, tzf);
    // A range wider than the grid is the whole grid.
    Value wx = subi(c1x0, c0x);
    Value wy = subi(c1y0, c0y);
    Value wz = subi(c1z0, c0z);
    Value wideX = cmpi(P::sge, wx, gnx32);
    Value wideY = cmpi(P::sge, wy, gny32);
    Value wideZ = cmpi(P::sge, wz, gnz32);
    Value gnx1 = subi(gnx32, oneI);
    Value gny1 = subi(gny32, oneI);
    Value gnz1 = subi(gnz32, oneI);
    Value fx0 = select(wideX, zeroI, c0x);
    Value fy0 = select(wideY, zeroI, c0y);
    Value fz0 = select(wideZ, zeroI, c0z);
    Value fx1 = select(wideX, gnx1, c1x0);
    Value fy1 = select(wideY, gny1, c1y0);
    Value fz1 = select(wideZ, gnz1, c1z0);
    Value reach2 = mulf(freach, freach);
    Value below = kernels::emitLanesBelow(b, loc, lane32, oneI);
    Value first16 = addi(first32, c16i);
    Value unitCapacityW = dim(units, c0w);
    Value noneBlock = ci(-1);
    Value g32W = indexCast(i32, g);
    Value fz1p = addi(fz1, oneI);
    Value fy1p = addi(fy1, oneI);

    // A candidate of the queue, `qv`: the bits of its entry in the list,
    // 0 for none. With `images` it carries in the bits from 26 the code of
    // the image that the entry takes.
    auto candidate = [&](Value qv) -> Value {
      Value q, img;
      if (!images) {
        q = addi(qv, zeroI);
      } else {
        // The place, and in the bits from 26 the code of the image of the
        // candidate, 9 (o_z + 1) + 3 (o_y + 1) + (o_x + 1) (D115, D242),
        // or in a triclinic cell that of emitOffsetTriclinic. Places are
        // fewer than 2^26.
        Value imMask = ci(67108863);
        q = andi(qv, imMask);
        Value imC26 = ci(26);
        img = shrui(qv, imC26);
      }
      Value qi = indexCast(index, q);
      Value xq = load(xp, {qi, c0w});
      Value yq = load(xp, {qi, c1w});
      Value zq = load(xp, {qi, c2w});
      Value dx0 = subf(xq, cxb);
      Value dy0 = subf(yq, cyb);
      Value dz0 = subf(zq, czb);
      // The image of the candidate, and in a triclinic cell its lattice
      // shift (kx1, ky1, kz1 below, of the image that the entry takes).
      Value rqx, rqy, rqz, kna, knb, knc, imOx, imOy, imOz;
      if (!images) {
        Values r = image(dx0, dy0, dz0);
        rqx = r[0], rqy = r[1], rqz = r[2];
        if (tilted)
          kna = r[3], knb = r[4], knc = r[5];
      } else if (!tilted) {
        Values r = image(dx0, dy0, dz0);
        // The image the queue asked for: o cells from the image nearest
        // to the center of the box, r - o L.
        Value imOneI = ci(1);
        Value imThreeI = ci(3);
        Value imNineI = ci(9);
        Value imQx = remui(img, imThreeI);
        Value imQ3 = divui(img, imThreeI);
        Value imQy = remui(imQ3, imThreeI);
        Value imQz = divui(img, imNineI);
        Value imOxI = subi(imQx, imOneI);
        Value imOyI = subi(imQy, imOneI);
        Value imOzI = subi(imQz, imOneI);
        imOx = sitofp(f32, imOxI);
        imOy = sitofp(f32, imOyI);
        imOz = sitofp(f32, imOzI);
        Value imLx = mulf(imOx, flx);
        Value imLy = mulf(imOy, fly);
        Value imLz = mulf(imOz, flz);
        rqx = subf(r[0], imLx);
        rqy = subf(r[1], imLy);
        rqz = subf(r[2], imLz);
      } else {
        Values r = imageTriclinic(dx0, dy0, dz0);
        // The image the queue asked for, and its lattice shift.
        SmallVector<Type> results(6, f32);
        ResultRange o = call(kOffsetTriclinic, results,
                             {r[0], r[1], r[2], img, flx, fly, flz, ftbx,
                              ftcx, ftcy, filx, fily});
        rqx = o[0], rqy = o[1], rqz = o[2];
        kna = addf(r[3], o[3]);
        knb = addf(r[4], o[4]);
        knc = addf(r[5], o[5]);
      }
      Value ax = absf(rqx);
      Value ay = absf(rqy);
      Value az = absf(rqz);
      Value zeroF = cf(0.0);
      Value bx0 = subf(ax, hx);
      Value by0 = subf(ay, hy);
      Value bz0 = subf(az, hz);
      Value bxd = maximumf(bx0, zeroF);
      Value byd = maximumf(by0, zeroF);
      Value bzd = maximumf(bz0, zeroF);
      Value bx2 = mulf(bxd, bxd);
      Value by2 = mulf(byd, byd);
      Value bz2 = mulf(bzd, bzd);
      Value bxy = addf(bx2, by2);
      Value bd2 = addf(bxy, bz2);
      Value near = cmpf(F::OLE, bd2, reach2);
      Value b3 = ifElse(
          i32, near,
          [&]() -> Values {
            Value own = cmpi(P::slt, q, first16);
            Value m = forLoop(
                c0w, c16w, c1w, {zeroI},
                [&](Value u, ValueRange a) -> Values {
                  Value at = addi(rel0, u);
                  Value gx = load(relWg, {at, c0w});
                  Value gy = load(relWg, {at, c1w});
                  Value gz = load(relWg, {at, c2w});
                  Value gflag = load(relWg, {at, c3w});
                  Value ex = subf(rqx, gx);
                  Value ey = subf(rqy, gy);
                  Value ez = subf(rqz, gz);
                  Value ex2 = mulf(ex, ex);
                  Value ey2 = mulf(ey, ey);
                  Value ez2 = mulf(ez, ez);
                  Value exy = addf(ex2, ey2);
                  Value e2 = addf(exy, ez2);
                  Value within = cmpf(F::OLE, e2, reach2);
                  Value u32 = indexCast(i32, u);
                  Value pu = addi(first32, u32);
                  Value notAfter = cmpi(P::sle, q, pu);
                  Value same = andi(own, notAfter);
                  Value yes = cb(true);
                  Value distinct = xori(same, yes);
                  Value halfFF = cf(0.5);
                  Value isReal = cmpf(F::OGT, gflag, halfFF);
                  Value ok0 = andi(within, distinct);
                  Value ok = andi(ok0, isReal);
                  Value bit0 = shli(oneI, u32);
                  Value bit = select(ok, bit0, zeroI);
                  Value nm = ori(a[0], bit);
                  return {nm};
                })[0];
            // The excluded pairs. A group whose partners did not fit the
            // memory of the warp (D106) reads the rows of the excluded
            // pairs of its particles that the candidate pairs with; the
            // others search the sorted partners for the first at the place
            // q.
            Value mex = ifElse(
                i32, overflow,
                [&]() -> Values {
                  Value sj = load(order, qi);
                  Value sm = forLoop(
                      c0w, c16w, c1w, {m},
                      [&](Value su, ValueRange a) -> Values {
                        Value smm = a[0];
                        Value su32 = indexCast(i32, su);
                        Value sbit = shli(oneI, su32);
                        Value sset0 = andi(smm, sbit);
                        Value sset = cmpi(P::ne, sset0, zeroI);
                        Value snm = ifElse(
                            i32, sset,
                            [&]() -> Values {
                              Value sp = addi(first, su);
                              Value si = load(order, sp);
                              Value sii = indexCast(index, si);
                              Value sentry = cx(4);
                              auto srow = md_alg::RaggedRowOp::create(
                                  b, loc, excluded, sii, sentry,
                                  /*checked=*/true);
                              Value sfirst = srow.getFirst(),
                                    sc = srow.getCount();
                              Value sin = cmpi(P::ne, sc, c0w);
                              Value sfound = ifElse(
                                  i1, sin,
                                  [&]() -> Values {
                                    Value sno = cb(false);
                                    Value sf = forLoop(
                                        c0w, sc, c1w, {sno},
                                        [&](Value sk,
                                            ValueRange acc) -> Values {
                                          Value spt = call1(
                                              kPartner, i32,
                                              {excluded, sfirst, sk});
                                          Value seq =
                                              cmpi(P::eq, spt, sj);
                                          Value sor = ori(acc[0], seq);
                                          return {sor};
                                        })[0];
                                    return {sf};
                                  },
                                  [&]() -> Values {
                                    Value sno2 = cb(false);
                                    return {sno2};
                                  })[0];
                              Value sall = ci(-1);
                              Value skeep = xori(sbit, sall);
                              Value scleared = andi(smm, skeep);
                              Value snext = select(sfound, scleared, smm);
                              return {snext};
                            },
                            [&]() -> Values { return {smm}; })[0];
                        return {snm};
                      })[0];
                  return {sm};
                },
                [&]() -> Values {
                  Value anyBits = cmpi(P::ne, m, zeroI);
                  Value fromMin = cmpi(P::sge, q, pmin);
                  Value toMax = cmpi(P::sle, q, pmax);
                  Value inRange = andi(fromMin, toMax);
                  Value any = andi(anyBits, inRange);
                  Value mex0 = ifElse(
                      i32, any,
                      [&]() -> Values {
                        Value four = ci(4);
                        Value loR = whileLoop(
                            {index, index}, {c0w, np},
                            [&](ValueRange a) -> std::pair<Value, Values> {
                              Value go = cmpi(P::ult, a[0], a[1]);
                              return {go, {a[0], a[1]}};
                            },
                            [&](ValueRange a) -> Values {
                              Value lo = a[0], hi = a[1];
                              Value sum = addi(lo, hi);
                              Value mid = divui(sum, c2w);
                              Value at = addi(pbase, mid);
                              Value v = load(partners, at);
                              Value vq = shrsi(v, four);
                              Value less = cmpi(P::slt, vq, q);
                              Value mid1 = addi(mid, c1w);
                              Value nlo = select(less, mid1, lo);
                              Value nhi = select(less, hi, mid);
                              return {nlo, nhi};
                            })[0];
                        Value cleared = whileLoop(
                            {i32, index}, {m, loR},
                            [&](ValueRange a) -> std::pair<Value, Values> {
                              Value e = a[1];
                              Value in = cmpi(P::ult, e, np);
                              Value at = addi(pbase, e);
                              Value safe = select(in, at, pbase);
                              Value v = load(partners, safe);
                              Value vq = shrsi(v, four);
                              Value match = cmpi(P::eq, vq, q);
                              Value go = andi(in, match);
                              return {go, {a[0], a[1]}};
                            },
                            [&](ValueRange a) -> Values {
                              Value mm = a[0], e = a[1];
                              Value at = addi(pbase, e);
                              Value v = load(partners, at);
                              Value fifteen = ci(15);
                              Value uu = andi(v, fifteen);
                              Value bit = shli(oneI, uu);
                              Value allBits = ci(-1);
                              Value keep = xori(bit, allBits);
                              Value nm = andi(mm, keep);
                              Value e1 = addi(e, c1w);
                              return {nm, e1};
                            })[0];
                        return {cleared};
                      },
                      [&]() -> Values { return {m}; })[0];
                  return {mex0};
                })[0];
            return {mex};
          },
          [&]() -> Values { return {zeroI}; })[0];
      // The shift of the entry to the frame of the group, from that of its
      // own place, in cells: e = W_g − W_q − k, with W the frame less the
      // wrapped position of a place (the first of the group for W_g) and k
      // the cells between the candidate and the center of the box, of the
      // image that the entry takes; -4 to 4 along each axis, (e + 4) in
      // four bits each, in bits 16 to 27 of its mask (D115); in a
      // triclinic cell lattice vectors, (e + 16) in five bits each.
      Value pairs = cmpi(P::ne, b3, zeroI);
      return ifElse(
          i32, pairs,
          [&]() -> Values {
            Value c3q = cx(3);
            Value wqF = load(xp, {qi, c3q});
            Value wq = bitcast(i32, wqF);
            // In the packed form: e + 4 (e + 16) is 0 to 8 (0 to 31) along
            // each axis, so the sum over the fields comes out with no
            // carry.
            Value kx1, ky1, kz1;
            if (!tilted) {
              Value kx0 = mulf(dx0, filx);
              Value ky0 = mulf(dy0, fily);
              Value kz0 = mulf(dz0, filz);
              if (!images) {
                kx1 = roundeven(kx0);
                ky1 = roundeven(ky0);
                kz1 = roundeven(kz0);
              } else {
                Value kx1r = roundeven(kx0);
                kx1 = addf(kx1r, imOx);
                Value ky1r = roundeven(ky0);
                ky1 = addf(ky1r, imOy);
                Value kz1r = roundeven(kz0);
                kz1 = addf(kz1r, imOz);
              }
            } else if (!images) {
              // The image of an entry of the list kernel is that of its
              // minimum image.
              Value kzero = cf(0.0);
              kx1 = addf(kna, kzero);
              ky1 = addf(knb, kzero);
              kz1 = addf(knc, kzero);
            } else {
              kx1 = kna, ky1 = knb, kz1 = knc;
            }
            Value f2p10 = cf(1024.0);
            Value f2p20 = cf(1048576.0);
            Value kyz = mulf(kz1, f2p20);
            Value kyy = mulf(ky1, f2p10);
            Value kxy = addf(kx1, kyy);
            Value kpF = addf(kxy, kyz);
            Value kp = fptosi(i32, kpF);
            Value twos = ci(tilted ? 16793616 : 4198404);
            Value wd = subi(wfirst, wq);
            Value wd2 = addi(wd, twos);
            Value ep = subi(wd2, kp);
            Value fifteenE = ci(tilted ? 31 : 15);
            Value sixE = ci(tilted ? 5 : 6);
            Value c240 = ci(tilted ? 992 : 240);
            Value twelveE = ci(tilted ? 10 : 12);
            Value c3840 = ci(tilted ? 31744 : 3840);
            Value sixteenS = ci(16);
            Value eqx = andi(ep, fifteenE);
            Value eqy0 = shrui(ep, sixE);
            Value eqy = andi(eqy0, c240);
            Value eqz0 = shrui(ep, twelveE);
            Value eqz = andi(eqz0, c3840);
            Value eqXY = ori(eqx, eqy);
            Value eq = ori(eqXY, eqz);
            Value eq16 = shli(eq, sixteenS);
            Value b3x = ori(b3, eq16);
            return {b3x};
          },
          [&]() -> Values { return {zeroI}; })[0];
    };

    // The entries of the lanes that kept a candidate, in the order of the
    // lanes, into the list of the group, of `cs` entries whose last block
    // is `csb`: the count and the last block after them.
    auto addEntries = [&](Value cs, Value csb, Value qv,
                          Value bits) -> std::pair<Value, Value> {
      Value keep = cmpi(P::ne, bits, zeroI);
      Value place = qv;
      if (images) {
        Value qplaceMask = ci(67108863);
        place = andi(qv, qplaceMask);
      }
      // Into blocks of 64 entries from the count of blocks,
      // `sizes_device[2]`, named by the group and their ordinal: the first
      // lane takes the new ones; the warp shares them (md_alg.append).
      Value at = cx(2);
      auto append = md_alg::AppendOp::create(
          b, loc, keep, ValueRange{place, bits}, ValueRange{entries, masks},
          cs, csb, g32W, units, ordinals, sizesDevice, at, unitCapacityW,
          b.getI64IntegerAttr(64), lane32, below, /*determinism=*/nullptr);
      return {append.getNextCount(), append.getNextLast()};
    };

    // A round of the queue of the warp: the candidate `nearq` of the lane
    // (-1 for none) to the queue, and once it holds 32, the entries of 32,
    // the rest moved to its front. Gives the count, the last block, and
    // the length of the queue after it.
    Value noneQ;
    auto round = [&](Value nearq, Value cs, Value csb, Value qn) -> Values {
      Value isNear = cmpi(P::sge, nearq, zeroI);
      Value nearBits = kernels::emitWarpBallot(b, loc, isNear);
      Value nearRank = kernels::emitWarpRank(b, loc, nearBits, below);
      Value nearN = kernels::emitWarpCount(b, loc, nearBits);
      ifThen(isNear, [&] {
        Value qslot0 = addi(qn, nearRank);
        Value qslot1 = indexCast(index, qslot0);
        Value qslot = addi(qbase, qslot1);
        store(nearq, queue, qslot);
      });
      warpSync(all);
      Value qn1 = addi(qn, nearN);
      Value full = cmpi(P::sge, qn1, c32i);
      ResultRange r = ifElse(
          {i32, i32, i32}, full,
          [&]() -> Values {
            Value qslot = addi(qbase, lane);
            Value qv = load(queue, qslot);
            Value valid = cb(true);
            Value bits = ifElse(
                i32, valid, [&]() -> Values { return {candidate(qv)}; },
                [&]() -> Values { return {zeroI}; })[0];
            auto [next, nextBlock] = addEntries(cs, csb, qv, bits);
            // The rest of the queue to its front.
            Value rest = subi(qn1, c32i);
            Value moves = cmpi(P::slt, lane32, rest);
            Value fromSlot = addi(qslot, c32w);
            Value moved = ifElse(
                i32, moves,
                [&]() -> Values { return {load(queue, fromSlot)}; },
                [&]() -> Values { return {noneQ}; })[0];
            warpSync(all);
            ifThen(moves, [&] { store(moved, queue, qslot); });
            warpSync(all);
            return {next, nextBlock, rest};
          },
          [&]() -> Values { return {cs, csb, qn1}; });
      return Values(r.begin(), r.end());
    };

    // The candidates near the box go to the queue of the warp; the warp
    // takes them 32 at a time, so that its lanes test them together
    // against the particles of the group (D99).
    auto run = [&](Value s0, ValueRange a, Value end) -> Values {
      Value cs = a[0], csb = a[1], qn = a[2];
      Value s = addi(s0, lane32);
      Value inRun = cmpi(P::slt, s, end);
      noneQ = ci(-1);
      if (!images) {
        Value nearq = ifElse(
            i32, inRun,
            [&]() -> Values {
              Value si = indexCast(index, s);
              Value q = md_alg::BinItemOp::create(b, loc, gplace, si);
              Value later = cmpi(P::sge, q, first32);
              Value qa = ifElse(
                  i32, later,
                  [&]() -> Values {
                    Value qi = indexCast(index, q);
                    Value xq = load(xp, {qi, c0w});
                    Value yq = load(xp, {qi, c1w});
                    Value zq = load(xp, {qi, c2w});
                    Value dx0 = subf(xq, cxb);
                    Value dy0 = subf(yq, cyb);
                    Value dz0 = subf(zq, czb);
                    Values r = image(dx0, dy0, dz0);
                    Value ax = absf(r[0]);
                    Value ay = absf(r[1]);
                    Value az = absf(r[2]);
                    Value zeroF = cf(0.0);
                    Value bx0 = subf(ax, hx);
                    Value by0 = subf(ay, hy);
                    Value bz0 = subf(az, hz);
                    Value bxd = maximumf(bx0, zeroF);
                    Value byd = maximumf(by0, zeroF);
                    Value bzd = maximumf(bz0, zeroF);
                    Value bx2 = mulf(bxd, bxd);
                    Value by2 = mulf(byd, byd);
                    Value bz2 = mulf(bzd, bzd);
                    Value bxy = addf(bx2, by2);
                    Value bd2 = addf(bxy, bz2);
                    Value near = cmpf(F::OLE, bd2, reach2);
                    Value qnSel = select(near, q, noneQ);
                    return {qnSel};
                  },
                  [&]() -> Values { return {noneQ}; })[0];
              return {qa};
            },
            [&]() -> Values { return {noneQ}; })[0];
        return round(nearq, cs, csb, qn);
      }
      // The images of the candidate other than the one nearest to the
      // center of the box: a cell away or none along each axis, or a
      // lattice vector in a triclinic cell, by their codes (D115, D242); a
      // warp in a wide cell finds none within the reach.
      Value imFar = cf(3.0e+38);
      SmallVector<Type> candidateTypes = {i32, f32, f32, f32};
      ResultRange c = ifElse(
          candidateTypes, inRun,
          [&]() -> Values {
            Value si = indexCast(index, s);
            Value q = md_alg::BinItemOp::create(b, loc, gplace, si);
            Value later = cmpi(P::sge, q, first32);
            ResultRange qa = ifElse(
                candidateTypes, later,
                [&]() -> Values {
                  Value qi = indexCast(index, q);
                  Value xq = load(xp, {qi, c0w});
                  Value yq = load(xp, {qi, c1w});
                  Value zq = load(xp, {qi, c2w});
                  Value dx0 = subf(xq, cxb);
                  Value dy0 = subf(yq, cyb);
                  Value dz0 = subf(zq, czb);
                  Values r = image(dx0, dy0, dz0);
                  return {q, r[0], r[1], r[2]};
                },
                [&]() -> Values { return {noneQ, imFar, imFar, imFar}; });
            return Values(qa.begin(), qa.end());
          },
          [&]() -> Values { return {noneQ, imFar, imFar, imFar}; });
      Value cand = c[0], crx = c[1], cry = c[2], crz = c[3];
      Value hasCand = cmpi(P::sge, cand, zeroI);
      Value imN27 = ci(27);
      Value imMid = ci(13);
      Value imC26q = ci(26);
      Value imThree, imNine;
      if (!tilted) {
        imThree = ci(3);
        imNine = ci(9);
      }
      ResultRange r = forLoop(
          zeroI, imN27, oneI, {cs, csb, qn},
          [&](Value imgS, ValueRange ia) -> Values {
            Value irx, iry, irz;
            if (!tilted) {
              Value iqx = remui(imgS, imThree);
              Value iq3 = divui(imgS, imThree);
              Value iqy = remui(iq3, imThree);
              Value iqz = divui(imgS, imNine);
              Value ioxI = subi(iqx, oneI);
              Value ioyI = subi(iqy, oneI);
              Value iozI = subi(iqz, oneI);
              Value iox = sitofp(f32, ioxI);
              Value ioy = sitofp(f32, ioyI);
              Value ioz = sitofp(f32, iozI);
              Value imlx = mulf(iox, flx);
              Value imly = mulf(ioy, fly);
              Value imlz = mulf(ioz, flz);
              irx = subf(crx, imlx);
              iry = subf(cry, imly);
              irz = subf(crz, imlz);
            } else {
              SmallVector<Type> results(6, f32);
              ResultRange o = call(kOffsetTriclinic, results,
                                   {crx, cry, crz, imgS, flx, fly, flz,
                                    ftbx, ftcx, ftcy, filx, fily});
              irx = o[0], iry = o[1], irz = o[2];
            }
            Value iax = absf(irx);
            Value iay = absf(iry);
            Value iaz = absf(irz);
            Value izeroF = cf(0.0);
            Value ibx0 = subf(iax, hx);
            Value iby0 = subf(iay, hy);
            Value ibz0 = subf(iaz, hz);
            Value ibxd = maximumf(ibx0, izeroF);
            Value ibyd = maximumf(iby0, izeroF);
            Value ibzd = maximumf(ibz0, izeroF);
            Value ibx2 = mulf(ibxd, ibxd);
            Value iby2 = mulf(ibyd, ibyd);
            Value ibz2 = mulf(ibzd, ibzd);
            Value ibxy = addf(ibx2, iby2);
            Value ids2 = addf(ibxy, ibz2);
            Value nearImg = cmpf(F::OLE, ids2, reach2);
            Value otherImg = cmpi(P::ne, imgS, imMid);
            Value takeImg0 = andi(nearImg, otherImg);
            Value takeImg = andi(takeImg0, hasCand);
            Value imgBits = shli(imgS, imC26q);
            Value qimg = ori(cand, imgBits);
            Value nearq = select(takeImg, qimg, noneQ);
            return round(nearq, ia[0], ia[1], ia[2]);
          });
      return Values(r.begin(), r.end());
    };

    Value cz0 = zeroI, czb0 = noneBlock;
    if (images) {
      // Each list continues where the list kernel left it.
      cz0 = load(counts, g);
      Value imgC6 = cx(6);
      Value imgBlock0F = load(boxes, {g, imgC6});
      czb0 = bitcast(i32, imgBlock0F);
    }
    ResultRange main = forLoop(
        fz0, fz1p, oneI, {cz0, czb0, zeroI},
        [&](Value zz, ValueRange za) -> Values {
          Value zm = remsi(zz, gnz32);
          Value zneg = cmpi(P::slt, zm, zeroI);
          Value zw = addi(zm, gnz32);
          Value iz = select(zneg, zw, zm);
          Value yFrom = fy0, yTo = fy1p, czx, twideY;
          if (tilted) {
            // The row t_z cells of the grid beyond the faces of z is that
            // of c t_z away: the window of y, and of x, moves by -t_z
            // (c_x, c_y).
            Value tzD = subi(zz, iz);
            Value tzI = divsi(tzD, gnz32);
            Value tzF = sitofp(f32, tzI);
            Value tzX = mulf(tzF, ftcx);
            Value tzY = mulf(tzF, ftcy);
            czx = subf(cxb, tzX);
            Value czy = subf(cyb, tzY);
            Value tloy = subf(czy, ecy);
            Value thiy = addf(czy, ecy);
            Value tsy0 = mulf(tloy, fily);
            Value tsy1 = mulf(tsy0, gnyf);
            Value tty0 = mulf(thiy, fily);
            Value tty1 = mulf(tty0, gnyf);
            Value tsyf = floor(tsy1);
            Value ttyf = floor(tty1);
            Value tc0y = fptosi(i32, tsyf);
            Value tc1y = fptosi(i32, ttyf);
            Value twy = subi(tc1y, tc0y);
            Value twideY0 = cmpi(P::sge, twy, gny32);
            twideY = ori(twideY0, wideZ);
            yFrom = select(twideY, zeroI, tc0y);
            Value tfy1 = select(twideY, gny1, tc1y);
            yTo = addi(tfy1, oneI);
          }
          ResultRange ry = forLoop(
              yFrom, yTo, oneI, {za[0], za[1], za[2]},
              [&](Value yy, ValueRange ya) -> Values {
                Value ym = remsi(yy, gny32);
                Value yneg = cmpi(P::slt, ym, zeroI);
                Value yw = addi(ym, gny32);
                Value iy = select(yneg, yw, ym);
                Value xFrom = fx0, xTo = fx1;
                if (tilted) {
                  // Across the faces of y, the row is that of b t_y away:
                  // the window of x moves by -t_y b_x.
                  Value tyD = subi(yy, iy);
                  Value tyI = divsi(tyD, gny32);
                  Value tyF = sitofp(f32, tyI);
                  Value tyX = mulf(tyF, ftbx);
                  Value czyx = subf(czx, tyX);
                  Value tlox = subf(czyx, ecx);
                  Value thix = addf(czyx, ecx);
                  Value tsx0 = mulf(tlox, filx);
                  Value tsx1 = mulf(tsx0, gnxf);
                  Value ttx0 = mulf(thix, filx);
                  Value ttx1 = mulf(ttx0, gnxf);
                  Value tsxf = floor(tsx1);
                  Value ttxf = floor(ttx1);
                  Value tc0x = fptosi(i32, tsxf);
                  Value tc1x = fptosi(i32, ttxf);
                  Value twx = subi(tc1x, tc0x);
                  Value twideX0 = cmpi(P::sge, twx, gnx32);
                  Value twideX = ori(twideX0, twideY);
                  // The window moved by whole periods of a, which moves
                  // nothing else, to begin within the grid: the run along
                  // x then goes around the edge once at most, however far
                  // the tilts took it.
                  Value tmx0 = remsi(tc0x, gnx32);
                  Value tmxNeg = cmpi(P::slt, tmx0, zeroI);
                  Value tmxUp = addi(tmx0, gnx32);
                  Value tmx = select(tmxNeg, tmxUp, tmx0);
                  Value tperiods = subi(tc0x, tmx);
                  Value tc1xm = subi(tc1x, tperiods);
                  xFrom = select(twideX, zeroI, tmx);
                  xTo = select(twideX, gnx1, tc1xm);
                }
                Value rowz = muli(iz, gny32);
                Value rowzy = addi(rowz, iy);
                Value rowStart = muli(rowzy, gnx32);
                // The run along x, in one or two pieces around the edge.
                Value under = cmpi(P::slt, xFrom, zeroI);
                Value over = cmpi(P::sge, xTo, gnx32);
                Value aFrom = addi(xFrom, gnx32);
                Value from1 = select(under, aFrom, xFrom);
                Value to1 = select(under, gnx1, xTo);
                Value to1b = select(over, gnx1, to1);
                Value twoPieces = ori(under, over);
                Value to2a = subi(xTo, gnx32);
                Value to2 = select(under, xTo, to2a);
                Value pieces = select(twoPieces, c2i, oneI);
                ResultRange rp = forLoop(
                    zeroI, pieces, oneI, {ya[0], ya[1], ya[2]},
                    [&](Value piece, ValueRange pa) -> Values {
                      Value isSecond = cmpi(P::ne, piece, zeroI);
                      Value from = select(isSecond, zeroI, from1);
                      Value to = select(isSecond, to2, to1b);
                      Value cfrom = addi(rowStart, from);
                      Value toNext = addi(to, oneI);
                      Value cto = addi(rowStart, toNext);
                      Value cfromI = indexCast(index, cfrom);
                      Value ctoI = indexCast(index, cto);
                      auto span = md_alg::BinSpanOp::create(b, loc, gstart,
                                                            cfromI, ctoI);
                      Value begin = span.getBegin();
                      Value end = span.getEnd();
                      ResultRange rs = forLoop(
                          begin, end, c32i, {pa[0], pa[1], pa[2]},
                          [&](Value s0, ValueRange sa) -> Values {
                            return run(s0, sa, end);
                          });
                      return Values(rs.begin(), rs.end());
                    });
                return Values(rp.begin(), rp.end());
              });
          return Values(ry.begin(), ry.end());
        });
    Value countMain = main[0], blockMain = main[1], queuedAll = main[2];
    // What the queue holds at the end.
    Value left = cmpi(P::sgt, queuedAll, zeroI);
    ResultRange done = ifElse(
        {i32, i32}, left,
        [&]() -> Values {
          Value cs = addi(countMain, zeroI);
          Value csb = addi(blockMain, zeroI);
          Value valid = cmpi(P::slt, lane32, queuedAll);
          Value qslot = addi(qbase, lane);
          Value qv = load(queue, qslot);
          Value bits = ifElse(
              i32, valid, [&]() -> Values { return {candidate(qv)}; },
              [&]() -> Values { return {zeroI}; })[0];
          auto [next, nextBlock] = addEntries(cs, csb, qv, bits);
          return {next, nextBlock};
        },
        [&]() -> Values { return {countMain, blockMain}; });
    Value countAll = done[0], blockAll = done[1];
    Value lane0 = cmpi(P::eq, lane, c0w);
    ifThen(lane0, [&] {
      store(countAll, counts, g);
      // The block that the list ends in, where the kernel of the images
      // continues it (D115), in the column of the box that is free.
      Value lastBlockF = bitcast(f32, blockAll);
      Value c6Last = cx(6);
      store(lastBlockF, boxes, {g, c6Last});
      atomicMax(sizesDevice, countAll);
    });
  });
}

//===----------------------------------------------------------------------===//
// The build
//===----------------------------------------------------------------------===//

void GroupsBuild::emitBuild() {
  Type cell = VectorType::get({tilted ? 6 : 3}, f64);
  auto hostSizes = MemRefType::get({5}, i32);
  SmallVector<Type> inputs = {positions, cell,   f64,    buffer, buffer,
                              buffer,    buffer, buffer, buffer, buffer,
                              buffer,    buffer, hostSizes};
  function(
      kernels::getGroupsBuildName(tilted), b.getFunctionType(inputs, {}),
      [&](ValueRange a) {
        x = a[0], box = a[1], reach = a[2], excluded = a[3], order = a[4],
        placeOf = a[5], entries = a[6], masks = a[7], counts = a[8],
        units = a[9], ordinals = a[10], shift = a[11], sizes = a[12];
        c0 = cx(0);
        c1 = cx(1);
        block = cx(128);
        cx(256);
        n = dim(x, c0);
        capacity = dim(order, c0);
        groupCapacity = dim(counts, c0);
        dim(units, c0);
        lx = extract(box, 0);
        ly = extract(box, 1);
        lz = extract(box, 2);
        Value unit = cd(1.0);
        ilx = divf(unit, lx);
        ily = divf(unit, ly);
        ilz = divf(unit, lz);
        flx = truncf(lx);
        fly = truncf(ly);
        flz = truncf(lz);
        Value funit = cf(1.0);
        filx = divf(funit, flx);
        fily = divf(funit, fly);
        filz = divf(funit, flz);
        if (tilted) {
          // The tilts b_x, c_x, c_y.
          tbx = extract(box, 3);
          tcx = extract(box, 4);
          tcy = extract(box, 5);
          ftbx = truncf(tbx);
          ftcx = truncf(tcx);
          ftcy = truncf(tcy);
        }
        // The reach, wider than asked by more than the rounding of the
        // positions in f32 can move a distance: a few ulp of the edges of
        // the cell (with the tilts in a triclinic one), as for the matrix,
        // so that every pair within the reach is in the lists (D80).
        Value edgesXY = addf(lx, ly);
        Value edges;
        if (!tilted) {
          edges = addf(edgesXY, lz);
        } else {
          Value edgesXYZ = addf(edgesXY, lz);
          Value absBx = absf(tbx);
          Value absCx = absf(tcx);
          Value absCy = absf(tcy);
          Value edgesB = addf(edgesXYZ, absBx);
          Value edgesC = addf(edgesB, absCx);
          edges = addf(edgesC, absCy);
        }
        Value tiny = cd(3.0e-6);
        Value margin = mulf(edges, tiny);
        Value reachWide = addf(reach, margin);
        freach = truncf(reachWide);

        // The columns: about 64 particles in a cube of their width.
        // At least one column and at most 256 along an edge
        // (md_alg.grid_plan), with maxnumf and minnumf, which take the
        // other operand for one that is not a number: a cell that is not a
        // number, or one that a run that has failed has blown up, gives
        // counts that are defined and fit memory (#168).
        Value lxy = mulf(lx, ly);
        Value volume = mulf(lxy, lz);
        auto columnPlan = md_alg::GridPlanOp::create(
            b, loc, TypeRange{i64, i64}, ValueRange{lx, ly}, /*width=*/Value(),
            volume, b.getF64FloatAttr(64.0), n, b.getF64FloatAttr(1.0),
            b.getF64FloatAttr(256.0), /*determinism=*/nullptr);
        ncx64 = columnPlan.getCounts()[0];
        ncy64 = columnPlan.getCounts()[1];
        ncx = indexCast(index, ncx64);
        ncy = indexCast(index, ncy64);
        columns = muli(ncx, ncy);
        // Bins along z: two for each particle of a column, on average.
        Value n2 = addi(n, n);
        Value nzb0 = divui(n2, columns);
        nzb = maxui(nzb0, c1);
        Value keys = muli(columns, nzb);
        Value keys1 = addi(keys, c1);

        // The grid of the candidates: cells of half the reach, 3 a side or
        // more.
        // At most 256 a side, as the columns: wider cells only give more
        // candidates, whose range is clamped to the whole grid.
        Value half = cd(0.5);
        Value cellSize = mulf(reach, half);
        auto cellPlan = md_alg::GridPlanOp::create(
            b, loc, TypeRange{i64, i64, i64}, ValueRange{lx, ly, lz}, cellSize,
            /*volume=*/Value(), /*occupancy=*/FloatAttr(),
            /*population=*/Value(), b.getF64FloatAttr(3.0),
            b.getF64FloatAttr(256.0), /*determinism=*/nullptr);
        Value gx64 = cellPlan.getCounts()[0];
        Value gy64 = cellPlan.getCounts()[1];
        Value gz64 = cellPlan.getCounts()[2];
        Value gnx = indexCast(index, gx64);
        Value gny = indexCast(index, gy64);
        Value gnz = indexCast(index, gz64);
        gnx32 = trunci(gx64);
        gny32 = trunci(gy64);
        gnz32 = trunci(gz64);
        Value gxy = muli(gnx, gny);
        Value gcells = muli(gxy, gnz);
        Value gcells1 = addi(gcells, c1);

        Value gridOfN = call1(kGrid, index, {n, block});
        Value gridOfKeys = call1(kGrid, index, {keys, block});
        Value gridOfGcells = call1(kGrid, index, {gcells, block});
        Value gridOfCapacity = call1(kGrid, index, {capacity, block});
        call1(kGrid, index, {groupCapacity, block});

        wrapped = alloc(wrappedType, n);
        key = alloc(buffer, n);
        keyCount = alloc(buffer, keys);
        keyStart = alloc(buffer, keys1);
        keyCursor = alloc(buffer, keys);
        sorted = alloc(buffer, n);
        Value columns1 = addi(columns, c1);
        chunkBase = alloc(buffer, columns1);
        columnChunks = alloc(buffer, columns);
        Value gridOfColumns = call1(kGrid, index, {columns, block});
        sizesDevice = alloc(sizesType, {});
        xp = alloc(placesType, capacity);
        gkey = alloc(buffer, capacity);
        gcount = alloc(buffer, gcells);
        gstart = alloc(buffer, gcells1);
        gcursor = alloc(buffer, gcells);
        gplace = alloc(buffer, n);
        boxes = alloc(boxesType, groupCapacity);

        //===------------------------------------------------------------===//
        // The compact order
        //===------------------------------------------------------------===//

        clear(gridOfKeys, keyCount, keys, 0);
        clear(gridOfCapacity, order, capacity, -1);
        // The counts of the positions that are not numbers (D107) and of
        // the groups with too many excluded partners (D106), from zero.
        launch(c1, c1, {}, [&](gpu::LaunchOp) {
          Value zNone = ci(0);
          Value zI3 = cx(3);
          Value zI4 = cx(4);
          store(zNone, sizesDevice, zI3);
          store(zNone, sizesDevice, zI4);
        });
        // The key of each particle, and its position wrapped into the
        // cell; each particle at the next slot of its key, and the
        // particles of a key sorted by number.
        sortByKey(n, keys, key, keyCount, keyStart, keyCursor, sorted,
                  {wrapped}, {sizesDevice}, gridOfN, gridOfKeys,
                  [&](Value item) { emitKeys(item); });
        // The chunks of each column, the first chunk of each, and the
        // places.
        launch(gridOfColumns, block, {}, [&](gpu::LaunchOp op) {
          perItem(op, columns, [&](Value item) {
            Value i1c = cx(1);
            Value sixtyThree = ci(63);
            Value six = ci(6);
            Value kb = muli(item, nzb);
            Value next = addi(item, i1c);
            Value ke = muli(next, nzb);
            auto run = md_alg::BinSpanOp::create(b, loc, keyStart, kb, ke);
            Value b32 = run.getBegin();
            Value e32 = run.getEnd();
            Value len = subi(e32, b32);
            Value len63 = addi(len, sixtyThree);
            Value nch = shrsi(len63, six);
            store(nch, columnChunks, item);
          });
        });
        scan(columnChunks, chunkBase, Value(), columns);
        launch(c1, c1, {}, [&](gpu::LaunchOp) {
          Value i0 = cx(0);
          Value i1c = cx(1);
          Value i2c = cx(2);
          Value none = ci(0);
          Value chunksAll = load(chunkBase, columns);
          Value sixtyFourI = ci(64);
          Value placesAll = muli(chunksAll, sixtyFourI);
          store(placesAll, sizesDevice, i0);
          store(none, sizesDevice, i1c);
          store(none, sizesDevice, i2c);
        });
        // The order: a warp a chunk of 64 particles. There are at most n /
        // 32 + (one for each column) chunks; the warps beyond the last do
        // nothing.
        Value c32l = cx(32);
        Value c16b = cx(16);
        Value chunkBound0 = divui(n, c32l);
        Value chunkBound1 = addi(chunkBound0, columns);
        Value chunkThreads = muli(chunkBound1, c32l);
        Value gridOfChunks = call1(kGrid, index, {chunkThreads, block});
        launch(gridOfChunks, block, {ids},
               [&](gpu::LaunchOp op) { emitChunks(op); });
        dealloc(key);
        dealloc(keyCount);
        dealloc(keyStart);
        dealloc(keyCursor);
        dealloc(sorted);
        dealloc(chunkBase);
        dealloc(columnChunks);

        //===------------------------------------------------------------===//
        // The positions of the places, and the grid of the candidates
        //===------------------------------------------------------------===//

        clear(gridOfGcells, gcount, gcells, 0);
        // The cell of each place in the grid of the candidates, and the
        // places of each cell in order.
        sortByKey(capacity, gcells, gkey, gcount, gstart, gcursor, gplace,
                  {xp}, {}, gridOfCapacity, gridOfGcells, [&](Value item) {
            Value i0 = cx(0);
            Value i1c = cx(1);
            Value i2c = cx(2);
            cx(3);
            Value zero = ci(0);
            Value one = ci(1);
            Value used32 = load(sizesDevice, i0);
            Value used = indexCast(index, used32);
            Value inUse = cmpi(CmpIPredicate::ult, item, used);
            Value pi = load(order, item);
            Value real0 = cmpi(CmpIPredicate::sge, pi, zero);
            Value real = andi(inUse, real0);
            ifThenElse(
                real,
                [&] {
                  Value ii = indexCast(index, pi);
                  Value xw = load(wrapped, {ii, i0});
                  Value yw = load(wrapped, {ii, i1c});
                  Value zw = load(wrapped, {ii, i2c});
                  store(xw, xp, {item, i0});
                  store(yw, xp, {item, i1c});
                  store(zw, xp, {item, i2c});
                  Value gxF = sitofp(f32, gnx32);
                  Value gyF = sitofp(f32, gny32);
                  Value gzF = sitofp(f32, gnz32);
                  Value ux = mulf(xw, filx);
                  Value uy = mulf(yw, fily);
                  Value uz = mulf(zw, filz);
                  Value tx = mulf(ux, gxF);
                  Value ty = mulf(uy, gyF);
                  Value tz = mulf(uz, gzF);
                  Value kx0 = fptosi(i32, tx);
                  Value ky0 = fptosi(i32, ty);
                  Value kz0 = fptosi(i32, tz);
                  Value gx1 = subi(gnx32, one);
                  Value gy1 = subi(gny32, one);
                  Value gz1 = subi(gnz32, one);
                  Value kx1 = minsi(kx0, gx1);
                  Value ky1 = minsi(ky0, gy1);
                  Value kz1 = minsi(kz0, gz1);
                  Value kx = maxsi(kx1, zero);
                  Value ky = maxsi(ky1, zero);
                  Value kz = maxsi(kz1, zero);
                  Value rz = muli(kz, gny32);
                  Value rzy = addi(rz, ky);
                  Value r = muli(rzy, gnx32);
                  Value k = addi(r, kx);
                  md_alg::BinOp::create(b, loc, k);
                },
                [&] {
                  md_alg::NoBinOp::create(b, loc);
                  Value far = cf(1.0e30);
                  store(far, xp, {item, i0});
                  store(far, xp, {item, i1c});
                  store(far, xp, {item, i2c});
                });
        });


        //===------------------------------------------------------------===//
        // The boxes of the groups: a half-warp a group, relative to its
        // first particle in the minimum image
        //===------------------------------------------------------------===//

        Value boxThreads = muli(groupCapacity, c16b);
        Value gridOfBoxes = call1(kGrid, index, {boxThreads, block});
        launch(gridOfBoxes, block, {},
               [&](gpu::LaunchOp op) { emitBoxes(op); });

        //===------------------------------------------------------------===//
        // The lists: a warp a group, a lane a candidate
        //===------------------------------------------------------------===//

        Value c32b = cx(32);
        Value listThreads = muli(groupCapacity, c32b);
        Value gridOfLists = call1(kGrid, index, {listThreads, block});
        SmallVector<Type> shared = {relative, partnersType, ids};
        launch(gridOfLists, block, shared,
               [&](gpu::LaunchOp op) { emitLists(op, /*images=*/false); });
        launch(gridOfLists, block, shared,
               [&](gpu::LaunchOp op) { emitLists(op, /*images=*/true); });

        // The sizes, to the host.
        Type token = gpu::AsyncTokenType::get(context);
        Value token0 = gpu::WaitOp::create(b, loc, token, ValueRange{})
                           .getAsyncToken();
        Value copied = gpu::MemcpyOp::create(b, loc, token, token0, sizes,
                                             sizesDevice)
                           .getAsyncToken();
        gpu::WaitOp::create(b, loc, Type(), copied);

        dealloc(wrapped);
        dealloc(sizesDevice);
        dealloc(xp);
        dealloc(gkey);
        dealloc(gcount);
        dealloc(gstart);
        dealloc(gcursor);
        dealloc(gplace);
        dealloc(boxes);
        func::ReturnOp::create(b, loc);
      });
}

void GroupsBuild::emit() {
  emitGrid();
  emitPartner();
  emitImage();
  if (tilted) {
    emitImageTriclinic();
    emitOffsetTriclinic();
  }
  emitBuild();
}

} // namespace

std::string kernels::getGroupsBuildName(bool triclinic) {
  return triclinic ? "mdrt_gpu_build_neighbors_groups_triclinic"
                   : "mdrt_gpu_build_neighbors_groups";
}

void kernels::emitGroupsBuild(ModuleOp module, bool triclinic,
                              bool singlePassScan) {
  OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
  GroupsBuild(b, module.getLoc(), triclinic, singlePassScan).emit();
}

namespace mdir {
#define GEN_PASS_DEF_ADDGROUPSBUILD
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

namespace {
class AddGroupsBuild
    : public mdir::impl::AddGroupsBuildBase<AddGroupsBuild> {
public:
  using AddGroupsBuildBase::AddGroupsBuildBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    std::string name = kernels::getGroupsBuildName(triclinic);
    if (auto declared = module.lookupSymbol<func::FuncOp>(name)) {
      if (!declared.isDeclaration()) {
        declared.emitOpError() << "is the build of the groups already";
        return signalPassFailure();
      }
      declared.erase();
    }
    kernels::emitGroupsBuild(module, triclinic, singlePassScan);
  }
};
} // namespace

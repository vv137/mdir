#include "mdir/Conversion/OrderedRowWorkEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cassert>
using namespace mlir;
using namespace mdir::kernels;
namespace {
Value ix(OpBuilder &b, Location l, int64_t n) { return createIndex(b,l,n); }
Value word(OpBuilder &b, Location l, int64_t n) { return arith::ConstantIntOp::create(b,l,n,32); }
Value add(OpBuilder &b, Location l, Value a, Value c) { return arith::AddIOp::create(b,l,a,c); }
Value mul(OpBuilder &b, Location l, Value a, int64_t n) { return arith::MulIOp::create(b,l,a,ix(b,l,n)); }
Value less(OpBuilder &b, Location l, Value a, Value c) { return arith::CmpIOp::create(b,l,arith::CmpIPredicate::ult,a,c); }
}
OrderedRowWorkState mdir::kernels::emitFlushOrderedRowWork(OpBuilder &b, Location l,
    const OrderedRowWorkBuffers &v, OrderedRowWorkState state,
    OrderedRowWorker worker, OrderedRowAppend append) {
  Value nonempty=arith::CmpIOp::create(b,l,arith::CmpIPredicate::ne,state.queued,ix(b,l,0));
  auto flushed=scf::IfOp::create(b,l,nonempty,[&](OpBuilder &at, Location) {
    // Producer control and queued are block-uniform. Publish all raw jobs.
    gpu::BarrierOp::create(at,l);
    Value warp=arith::DivUIOp::create(at,l,v.thread,ix(at,l,32));
    Value lane=arith::RemUIOp::create(at,l,v.thread,ix(at,l,32));
    Value lower=less(at,l,lane,ix(at,l,ORDERED_ROW_MEMBERS));
    scf::ForOp::create(at,l,ix(at,l,0),ix(at,l,32),ix(at,l,1),ValueRange{},
        [&](OpBuilder &job, Location, Value j, ValueRange) {
      Value slot=add(job,l,mul(job,l,warp,32),j);
      Value valid=less(job,l,slot,state.queued);
      auto result=scf::IfOp::create(job,l,valid,[&](OpBuilder &run, Location) {
        Value group=memref::LoadOp::create(run,l,v.groups,ValueRange{slot});
        SmallVector<Value,3> images;
        for(int64_t a=0;a<3;++a) images.push_back(memref::LoadOp::create(run,l,v.images,ValueRange{slot,ix(run,l,a)}));
        auto record=worker(run,group,images);
        assert(record.fields.size()==4);
        SmallVector<Value> values{record.keep};values.append(record.fields.begin(),record.fields.end());
        scf::YieldOp::create(run,l,values);
      },[&](OpBuilder &empty, Location) {
        Value no=arith::ConstantIntOp::create(empty,l,0,1);
        Value z=word(empty,l,0);
        scf::YieldOp::create(empty,l,ValueRange{no,z,z,z,z});
      });
      // Initialize every lower-half output, including unused tail jobs.
      scf::IfOp::create(job,l,lower,[&](OpBuilder &store, Location) {
        Value keep=arith::ExtUIOp::create(store,l,store.getI32Type(),result.getResult(0));
        memref::StoreOp::create(store,l,keep,v.keeps,ValueRange{slot,lane});
        for(int64_t a=0;a<4;++a)memref::StoreOp::create(store,l,result.getResult(a+1),v.records,ValueRange{slot,lane,ix(store,l,a)});
        scf::YieldOp::create(store,l);
      });
      scf::YieldOp::create(job,l);
    });
    // Each worker executes a finite bounded loop. No barrier is inside it.
    gpu::BarrierOp::create(at,l);
    Value owner=less(at,l,v.thread,ix(at,l,32));
    scf::IfOp::create(at,l,owner,[&](OpBuilder &consume, Location) {
      auto loop=scf::ForOp::create(consume,l,ix(consume,l,0),state.queued,ix(consume,l,2),
          ValueRange{state.stage.count,state.stage.chunk},
          [&](OpBuilder &take, Location, Value j, ValueRange carried) {
        Value half=arith::DivUIOp::create(take,l,lane,ix(take,l,16));
        Value member=arith::RemUIOp::create(take,l,lane,ix(take,l,16));
        Value slot=add(take,l,j,half);
        Value valid=less(take,l,slot,state.queued);
        // Slot128 is never loaded for an odd tail ending at128: such a tail
        // ends at127. Clamp anyway so every speculative address is initialized.
        Value safe=arith::SelectOp::create(take,l,valid,slot,ix(take,l,0));
        Value keep=memref::LoadOp::create(take,l,v.keeps,ValueRange{safe,member});
        keep=arith::AndIOp::create(take,l,valid,arith::CmpIOp::create(take,l,arith::CmpIPredicate::ne,keep,word(take,l,0)));
        SmallVector<Value,4> fields;
        for(int64_t a=0;a<4;++a)fields.push_back(memref::LoadOp::create(take,l,v.records,ValueRange{safe,member,ix(take,l,a)}));
        auto next=append(take,{carried[0],carried[1]},keep,fields);
        scf::YieldOp::create(take,l,ValueRange{next.count,next.chunk});
      });
      Value leader=arith::CmpIOp::create(consume,l,arith::CmpIPredicate::eq,lane,ix(consume,l,0));
      scf::IfOp::create(consume,l,leader,[&](OpBuilder &publish, Location) {
        memref::StoreOp::create(publish,l,loop.getResult(0),v.carry,ValueRange{ix(publish,l,0)});
        memref::StoreOp::create(publish,l,loop.getResult(1),v.carry,ValueRange{ix(publish,l,1)});
        scf::YieldOp::create(publish,l);
      });
      scf::YieldOp::create(consume,l);
    });
    gpu::BarrierOp::create(at,l);
    Value count=memref::LoadOp::create(at,l,v.carry,ValueRange{ix(at,l,0)});
    Value chunk=memref::LoadOp::create(at,l,v.carry,ValueRange{ix(at,l,1)});
    scf::YieldOp::create(at,l,ValueRange{count,chunk});
  },[&](OpBuilder &empty, Location) {
    scf::YieldOp::create(empty,l,ValueRange{state.stage.count,state.stage.chunk});
  });
  return {{flushed.getResult(0),flushed.getResult(1)},ix(b,l,0)};
}
OrderedRowWorkState mdir::kernels::emitEnqueueOrderedRowWork(OpBuilder &b, Location l,
    const OrderedRowWorkBuffers &v, OrderedRowWorkState state, Value group,
    ValueRange images, OrderedRowWorker worker, OrderedRowAppend append) {
  assert(images.size()==3);
  Value leader=arith::CmpIOp::create(b,l,arith::CmpIPredicate::eq,v.thread,ix(b,l,0));
  scf::IfOp::create(b,l,leader,[&](OpBuilder &store, Location) {
    memref::StoreOp::create(store,l,group,v.groups,ValueRange{state.queued});
    for(int64_t a=0;a<3;++a)memref::StoreOp::create(store,l,images[a],v.images,ValueRange{state.queued,ix(store,l,a)});
    scf::YieldOp::create(store,l);
  });
  Value queued=add(b,l,state.queued,ix(b,l,1));
  Value full=arith::CmpIOp::create(b,l,arith::CmpIPredicate::eq,queued,ix(b,l,ORDERED_ROW_JOBS));
  auto result=scf::IfOp::create(b,l,full,[&](OpBuilder &flush, Location) {
    auto next=emitFlushOrderedRowWork(flush,l,v,{state.stage,queued},worker,append);
    scf::YieldOp::create(flush,l,ValueRange{next.stage.count,next.stage.chunk,next.queued});
  },[&](OpBuilder &skip, Location) {
    scf::YieldOp::create(skip,l,ValueRange{state.stage.count,state.stage.chunk,queued});
  });
  return {{result.getResult(0),result.getResult(1)},result.getResult(2)};
}

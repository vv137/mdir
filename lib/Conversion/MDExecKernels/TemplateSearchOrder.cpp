#include "mdir/Conversion/TemplateSearchOrder.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
using namespace mlir;
using namespace mdir::kernels;

void mdir::kernels::emitOrderTemplateSearch(OpBuilder &b, Location loc,
    const TemplateBuildBuffers &v, const TemplateSpatialBuffers &p) {
  auto ix = [&](OpBuilder &a, int64_t n) { return createIndex(a, loc, n); };
  auto add = [&](OpBuilder &a, Value x, Value y) -> Value { return arith::AddIOp::create(a, loc, x, y); };
  auto sub = [&](OpBuilder &a, Value x, Value y) -> Value { return arith::SubIOp::create(a, loc, x, y); };
  auto mul = [&](OpBuilder &a, Value x, int64_t n) -> Value { return arith::MulIOp::create(a, loc, x, ix(a,n)); };
  auto index = [&](OpBuilder &a, Value x) -> Value { return arith::IndexCastUIOp::create(a, loc, a.getIndexType(), x); };
  auto word = [&](OpBuilder &a, Value x) -> Value { return arith::IndexCastOp::create(a, loc, a.getI32Type(), x); };
  auto load = [&](OpBuilder &a, Value x, Value i) -> Value { return memref::LoadOp::create(a, loc, x, ValueRange{i}); };
  auto store = [&](OpBuilder &a, Value x, Value i, Value z) { memref::StoreOp::create(a, loc, z, x, ValueRange{i}); };
  auto at = [&](OpBuilder &a, Value i, int64_t c) -> Value { return add(a,mul(a,i,4),ix(a,c)); };
  auto less = [&](OpBuilder &a, Value x, Value y) -> Value { return arith::CmpIOp::create(a,loc,arith::CmpIPredicate::ult,x,y); };
  auto active = [&](OpBuilder &a, Value row) -> Value {
    Value ww=less(a,row,v.groups.waterGroups);
    Value sw=less(a,row,add(a,v.groups.waterGroups,v.groups.atomGroups));
    return arith::SelectOp::create(a,loc,ww,p.waterActive,
        arith::SelectOp::create(a,loc,sw,arith::OrIOp::create(a,loc,p.waterActive,p.atomActive),p.atomActive));
  };
  auto keyLess = [&](OpBuilder &a, Value x, Value y) -> Value {
    Value tx=load(a,v.stageEntries,at(a,x,0)),ty=load(a,v.stageEntries,at(a,y,0));
    auto byte = [&](Value tag,unsigned shift)->Value {
      Value s=arith::ConstantIntOp::create(a,loc,shift,32);
      return arith::AndIOp::create(a,loc,arith::ShRUIOp::create(a,loc,tag,s),arith::ConstantIntOp::create(a,loc,255,32));
    };
    SmallVector<Value> xs{byte(tx,24),byte(tx,0),load(a,v.stageEntries,at(a,x,1)),byte(tx,8),load(a,v.stageEntries,at(a,x,2)),byte(tx,16),load(a,v.stageEntries,at(a,x,3))};
    SmallVector<Value> ys{byte(ty,24),byte(ty,0),load(a,v.stageEntries,at(a,y,1)),byte(ty,8),load(a,v.stageEntries,at(a,y,2)),byte(ty,16),load(a,v.stageEntries,at(a,y,3))};
    Value result=arith::ConstantIntOp::create(a,loc,0,1);
    for(int i=6;i>=0;--i) {
      Value eq=arith::CmpIOp::create(a,loc,arith::CmpIPredicate::eq,xs[i],ys[i]);
      result=arith::OrIOp::create(a,loc,less(a,xs[i],ys[i]),arith::AndIOp::create(a,loc,eq,result));
    }
    return result;
  };
  Value one=ix(b,1),block=ix(b,128);
  auto launch=gpu::LaunchOp::create(b,loc,arith::MaxUIOp::create(b,loc,one,v.rows),one,one,block,one,one);
  OpBuilder k=OpBuilder::atBlockEnd(&launch.getBody().front());
  Value row=launch.getBlockIds().x,thread=launch.getThreadIds().x;
  Value exists=less(k,row,v.rows);
  scf::IfOp::create(k,loc,exists,[&](OpBuilder &a,Location){
    scf::IfOp::create(a,loc,active(a,row),[&](OpBuilder &in,Location){
      Value begin=index(in,load(in,v.entryOffsets,row));
      Value end=index(in,load(in,v.entryOffsets,add(in,row,ix(in,1))));
      Value count=sub(in,end,begin);
      auto each=[&](OpBuilder &z,auto body){
        scf::ForOp::create(z,loc,thread,count,block,ValueRange(),[&](OpBuilder &q,Location,Value i,ValueRange){body(q,i,add(q,begin,i));scf::YieldOp::create(q,loc);});
      };
      each(in,[&](OpBuilder &q,Value,Value entry){
        Value partner=index(q,load(q,v.outerEntries,at(q,entry,0)));
        Value tag=load(q,v.outerEntries,at(q,entry,3));
        Value columns=arith::MulIOp::create(q,loc,p.columnsX,p.columnsY);
        Value column=arith::RemUIOp::create(q,loc,index(q,load(q,p.keys,partner)),columns);
        Value cx=arith::DivUIOp::create(q,loc,column,p.columnsY);
        Value serp=arith::RemUIOp::create(q,loc,column,p.columnsY);
        Value odd=arith::CmpIOp::create(q,loc,arith::CmpIPredicate::ne,arith::AndIOp::create(q,loc,cx,ix(q,1)),ix(q,0));
        Value cy=arith::SelectOp::create(q,loc,odd,sub(q,sub(q,p.columnsY,ix(q,1)),serp),serp);
        Value fields[]={tag,word(q,cx),word(q,cy),load(q,p.places,partner)};
        for(unsigned j=0;j<4;++j)store(q,v.stageEntries,at(q,entry,j),fields[j]);
        store(q,v.outerEntries,at(q,entry,3),word(q,entry));
      });
      gpu::BarrierOp::create(in,loc);
      auto widths=scf::WhileOp::create(in,loc,TypeRange{in.getIndexType()},ValueRange{ix(in,1)},
        [&](OpBuilder &test,Location,ValueRange state){scf::ConditionOp::create(test,loc,less(test,state[0],count),state);},
        [&](OpBuilder &pass,Location,ValueRange state){
          Value width=state[0],twice=mul(pass,width,2);
          each(pass,[&](OpBuilder &q,Value i,Value entry){
            Value pair=mul(q,arith::DivUIOp::create(q,loc,i,twice),1);
            pair=arith::MulIOp::create(q,loc,pair,twice);
            Value middle=arith::MinUIOp::create(q,loc,add(q,pair,width),count);
            Value stop=arith::MinUIOp::create(q,loc,add(q,pair,twice),count);
            Value left=less(q,i,middle);
            Value lo=arith::SelectOp::create(q,loc,left,middle,pair);
            Value hi=arith::SelectOp::create(q,loc,left,stop,middle);
            Value self=index(q,load(q,v.outerEntries,at(q,entry,3)));
            auto search=scf::WhileOp::create(q,loc,TypeRange{q.getIndexType(),q.getIndexType()},ValueRange{lo,hi},
              [&](OpBuilder &t,Location,ValueRange r){scf::ConditionOp::create(t,loc,less(t,r[0],r[1]),r);},
              [&](OpBuilder &t,Location,ValueRange r){
                Value mid=add(t,r[0],arith::DivUIOp::create(t,loc,sub(t,r[1],r[0]),ix(t,2)));
                Value other=index(t,load(t,v.outerEntries,at(t,add(t,begin,mid),3)));
                Value lt=keyLess(t,other,self),reverse=keyLess(t,self,other);
                Value le=arith::XOrIOp::create(t,loc,reverse,arith::ConstantIntOp::create(t,loc,1,1));
                Value before=arith::SelectOp::create(t,loc,left,lt,le);
                scf::YieldOp::create(t,loc,ValueRange{arith::SelectOp::create(t,loc,before,add(t,mid,ix(t,1)),r[0]),arith::SelectOp::create(t,loc,before,r[1],mid)});
              });
            Value ownStart=arith::SelectOp::create(q,loc,left,pair,middle);
            Value destination=add(q,pair,add(q,sub(q,i,ownStart),sub(q,search.getResult(0),lo)));
            store(q,v.innerEntries,add(q,begin,destination),word(q,self));
          });
          gpu::BarrierOp::create(pass,loc);
          each(pass,[&](OpBuilder &q,Value,Value entry){store(q,v.outerEntries,at(q,entry,3),load(q,v.innerEntries,entry));});
          gpu::BarrierOp::create(pass,loc);
          scf::YieldOp::create(pass,loc,ValueRange{twice});
        });
      (void)widths;
      scf::YieldOp::create(in,loc);
    });
    scf::YieldOp::create(a,loc);
  });
  gpu::TerminatorOp::create(k,loc);
  // Kernel boundaries protect contiguous permutation scratch from the
  // four-word record writes of other rows. Never fuse these phases.
  for(unsigned phase=0;phase<2;++phase){
    auto kernel=gpu::LaunchOp::create(b,loc,arith::MaxUIOp::create(b,loc,one,v.rows),one,one,block,one,one);
    OpBuilder a=OpBuilder::atBlockEnd(&kernel.getBody().front());
    Value r=kernel.getBlockIds().x,t=kernel.getThreadIds().x;
    scf::IfOp::create(a,loc,less(a,r,v.rows),[&](OpBuilder &q,Location){
      // Permutation scratch can alias an inactive row's inner payload too.
      // Restore every inner row in gather; only active outer rows need copyback.
      Value selected = phase == 0 ? Value(arith::ConstantIntOp::create(q,loc,1,1)) : active(q,r);
      scf::IfOp::create(q,loc,selected,[&](OpBuilder &z,Location){
        Value begin=index(z,load(z,v.entryOffsets,r)),end=index(z,load(z,v.entryOffsets,add(z,r,ix(z,1))));
        scf::ForOp::create(z,loc,add(z,begin,t),end,block,ValueRange(),[&](OpBuilder &copy,Location,Value entry,ValueRange){
          Value source=entry;
          if (phase == 0) {
            // Do not interpret an inactive row's public reserved zero as a permutation.
            Value permutation=index(copy,load(copy,v.outerEntries,at(copy,entry,3)));
            source=arith::SelectOp::create(copy,loc,active(copy,r),permutation,entry);
          }
          for(unsigned c=0;c<4;++c){
            Value value=(phase==0&&c==3)?word(copy,ix(copy,0)):load(copy,phase==0?v.outerEntries:v.innerEntries,at(copy,source,c));
            store(copy,phase==0?v.innerEntries:v.outerEntries,at(copy,entry,c),value);
          }
          scf::YieldOp::create(copy,loc);
        });
        scf::YieldOp::create(z,loc);
      });
      scf::YieldOp::create(q,loc);
    });
    gpu::TerminatorOp::create(a,loc);
  }
}

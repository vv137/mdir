#include "NativeStage.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <climits>

namespace prototype_io {
namespace {
void checked(cudaError_t error) {
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
__device__ bool sameShift(int4 a,int4 b){return a.x==b.x&&a.y==b.y&&a.z==b.z;}
struct ConvertedPiece { int partner; unsigned image, mask; };
__device__ unsigned encodeImage(unsigned code,int4 first,int4 partner,bool &valid) {
  int a[3]={first.x,first.y,first.z},b[3]={partner.x,partner.y,partner.z};
  unsigned result=0;
  for(int k=0;k<3;++k){long long n=static_cast<long long>(int((code>>(16+4*k))&15)-4)-a[k]+b[k];
    valid &= n>=-512&&n<512;
    result |= unsigned((n+512)&1023)<<(10*k);}
  return result;
}
// One record's bits are partitioned only by equal retained I shifts.
// Partner identity and image encoding are converted exactly; no pair is
// admitted, rejected, or reclassified using coordinates.
template<class Consumer>
__device__ int pieces(SessionDeviceView view,int waters,int kind,int group,
    int2 entry,Consumer consume,unsigned long long *flags) {
  const int *orderI=kind==0?view.waterOrder:view.atomOrder;
  const int *orderJ=kind==2?view.atomOrder:view.waterOrder;
  const int4 *shiftI=static_cast<const int4 *>(kind==0?view.waterShifts:view.atomShifts);
  const int4 *shiftJ=static_cast<const int4 *>(kind==2?view.atomShifts:view.waterShifts);
  int paddedJ=kind==2?view.atomPadded:view.waterPadded;
  if(entry.x<0||entry.x>=paddedJ){atomicOr(flags,1ull);return 0;}
  int partner=orderJ[entry.x];
  if(partner<0){atomicOr(flags,1ull);return 0;}
  int4 partnerShift=shiftJ[partner];
  unsigned remaining=unsigned(entry.y)&65535u;
  if(!remaining){atomicOr(flags,1ull);return 0;}
  int count=0;
  while(remaining){int first=__ffs(int(remaining))-1;int id=orderI[16*group+first];
    if(id<0){atomicOr(flags,1ull);return count;}
    int4 shift=shiftI[id];unsigned mask=0;
    for(int bit=0;bit<16;++bit)if(remaining&(1u<<bit)){
      int member=orderI[16*group+bit];
      if(member<0){atomicOr(flags,1ull);return count;}
      if(sameShift(shift,shiftI[member]))mask|=1u<<bit;
    }
    bool valid=true;unsigned image=encodeImage(unsigned(entry.y),shift,partnerShift,valid);
    if(!valid){atomicOr(flags,1ull);return count;}
    consume(ConvertedPiece{partner+(kind==2?waters:0),image,mask},count++);
    remaining &= ~mask;
  }
  return count;
}
__device__ int prefix(int n,int lane) {
  for(int offset=1;offset<32;offset<<=1){int value=__shfl_up_sync(0xffffffffu,n,offset);if(lane>=offset)n+=value;}
  return n;
}
__global__ void convertRows(SessionDeviceView view,int waters,int kind,bool inner,
    int *units,int *entries,int *rowCounts,unsigned long long *controls,
    unsigned long long capacity) {
  int lane=threadIdx.x&31,group=int(blockIdx.x)*4+(int(threadIdx.x)>>5);
  const auto list=view.classes[kind];if(group>=list.groups)return;
  int row=group+(kind?view.waterGroups:0)+(kind==2?view.atomGroups:0);
  int count=(inner?list.innerCount:list.outerCount)[group];
  size_t base=size_t(list.blockStart[group])*64;
  const int2 *source=static_cast<const int2 *>(inner?list.innerEntries:list.outerEntries);
  int total=0;
  for(int e=0;e<count;e+=32){int n=0;
    if(e+lane<count)n=pieces(view,waters,kind,group,source[base+e+lane],
        [] __device__(ConvertedPiece,int){},controls+1);
    n=prefix(n,lane);total+=__shfl_sync(0xffffffffu,n,31);
  }
  unsigned long long chunkBase=0;
  int chunks=(total+63)/64;
  if(lane==0){rowCounts[row]=total;
    chunkBase=atomicAdd(controls,static_cast<unsigned long long>(chunks));
    auto prior=atomicAdd(controls+2,static_cast<unsigned long long>(total));
    if(prior+total>INT_MAX)atomicOr(controls+1,1ull);
    if(chunkBase+chunks>capacity)atomicOr(controls+1,2ull);
    for(int ord=0;ord<chunks;++ord)if(chunkBase+ord<capacity){
      auto at=4*(chunkBase+ord);units[at]=row;
      units[at+1]=min(64,total-64*ord);units[at+2]=ord;units[at+3]=0;}
  }
  unsigned lo=__shfl_sync(0xffffffffu,unsigned(chunkBase),0);
  unsigned hi=__shfl_sync(0xffffffffu,unsigned(chunkBase>>32),0);
  chunkBase=(static_cast<unsigned long long>(hi)<<32)|lo;
  int carried=0;
  for(int e=0;e<count;e+=32){int n=0;
    if(e+lane<count)n=pieces(view,waters,kind,group,source[base+e+lane],
        [] __device__(ConvertedPiece,int){},controls+1);
    int inclusive=prefix(n,lane),before=inclusive-n;
    if(e+lane<count)pieces(view,waters,kind,group,source[base+e+lane],
        [=] __device__(ConvertedPiece piece,int ordinal){
          int local=carried+before+ordinal;
          auto chunk=chunkBase+unsigned(local/64);
          if(chunk<capacity){auto at=4*(64*chunk+unsigned(local&63));
            entries[at]=piece.partner;entries[at+1]=int(piece.image);
            entries[at+2]=int(piece.mask);entries[at+3]=0;}
        },controls+1);
    carried+=__shfl_sync(0xffffffffu,inclusive,31);
  }
}
}
void stageNative(Session *session,bool inner,int32_t *units,int32_t *entries,
    int32_t *counts,uint64_t *controls,uint64_t capacity) {
  auto view=sessionDeviceView(session);auto info=sessionInfo(session);
  checked(cudaMemsetAsync(controls,0,3*sizeof(uint64_t)));
  checked(cudaMemsetAsync(counts,0,sizeof(int32_t)*(view.waterGroups+2*view.atomGroups)));
  for(int kind=0;kind<3;++kind){int groups=view.classes[kind].groups;
    if(groups)convertRows<<<(groups+3)/4,128>>>(view,info.waters,kind,inner,units,entries,counts,
        reinterpret_cast<unsigned long long *>(controls),capacity);}
  checked(cudaGetLastError());
}
}

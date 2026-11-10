// Experimental I/O adapter only; the archived prototype device bodies are immutable.
#include "PrototypeSession.h"
#include "NativeStage.h"
#include "PhysicalCoverage.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
using namespace prototype_io;
namespace {
struct Trace { explicit Trace(const char *name){nvtxRangePushA(name);} ~Trace(){nvtxRangePop();} };
template<class T,int R> struct Mem {T *allocated,*aligned;int64_t offset,size[R],stride[R];};
template<class F> F symbol(const char *name) {auto p=dlsym(RTLD_DEFAULT,name);if(!p)throw std::runtime_error(std::string("missing native runtime symbol ")+name);return reinterpret_cast<F>(p);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int64_t layout(int64_t h,int64_t slot){return symbol<int64_t(*)(int64_t,int64_t)>("mdrtTemplatePairsLayoutValue")(h,slot);}
Mem<int32_t,1> ints(int64_t h,const char *name,int64_t slot=-1){Mem<int32_t,1> b{};if(slot<0)symbol<void(*)(void*,int64_t)>(name)(&b,h);else symbol<void(*)(void*,int64_t,int64_t)>(name)(&b,h,slot);return b;}
Mem<double,2> frame(int64_t h,int64_t slot){Mem<double,2>b{};symbol<void(*)(void*,int64_t,int64_t)>("_mlir_ciface_mdrtTemplatePairsFrameState")(&b,h,slot);return b;}
template<class T,int R> T* ptr(const Mem<T,R>&m){return m.aligned+m.offset;}
template<class T> std::vector<T> copy(const T *p,size_t n){std::vector<T>v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T> void put(T*p,const std::vector<T>&v){if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
std::vector<double> refs(int64_t h,int slot,size_t n){auto v=frame(h,slot);if(v.size[0]<int64_t(n)||v.stride[0]!=3||v.stride[1]!=1)throw std::runtime_error("native frame view mismatch");return copy(ptr(v),3*n);}
struct State {Session*session=nullptr;SessionInfo info{};bool physicalChecked=false;NativeBasis basis;std::vector<int32_t>members,fileIds;double *coordinates=nullptr,*deviceCell=nullptr;std::array<double,3>cell{};~State(){if(deviceCell)cudaFree(deviceCell);if(coordinates)cudaFree(coordinates);if(session)destroySession(session);}};
std::map<int64_t,std::unique_ptr<State>> states;
State& state(int64_t h){auto it=states.find(h);if(it==states.end())throw std::runtime_error("prototype owner was not registered");return *it->second;}
NativeBasis basis(int64_t h,const std::vector<double>&fraction){
 NativeBasis b;int64_t w=layout(h,TEMPLATE_LAYOUT_WATERS),a=layout(h,TEMPLATE_LAYOUT_ATOMS),wg=layout(h,TEMPLATE_LAYOUT_WATER_GROUPS),ag=layout(h,TEMPLATE_LAYOUT_ATOM_GROUPS);b.waters=w;
 auto order=copy(ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsOrder")),w+a);b.place.resize(w+a);for(size_t p=0;p<order.size();++p)b.place.at(order[p])=p;
 auto starts=ints(h,"_mlir_ciface_mdrtTemplatePairsGroupView",TEMPLATE_GROUP_STARTS),lengths=ints(h,"_mlir_ciface_mdrtTemplatePairsGroupView",TEMPLATE_GROUP_LENGTHS);
 auto add=[&](int64_t begin,int64_t n,std::vector<NativeGroup>&out){auto s=copy(ptr(starts)+begin,n),l=copy(ptr(lengths)+begin,n);for(int64_t i=0;i<n;++i)out.push_back({s[i],std::vector<int32_t>(order.begin()+s[i],order.begin()+s[i]+l[i])});};
 add(0,wg,b.waterGroups);add(w,ag,b.atomGroups);b.frames.resize(w+a);for(size_t i=0;i<b.frames.size();++i)for(int k=0;k<3;++k)b.frames[i][k]=int64_t(std::floor(fraction[3*i+k]));b.validate();return b;
}
// Input coefficients are permuted, never recomputed. The original file has
// solutes first, then uniform OHH[EP] waters. Native particle IDs prove it.
std::string mappedInput(const std::string&input,const std::vector<int32_t>&fileIds,const std::vector<int32_t>&members,int w,int a,int sites,int64_t owner){
 std::ifstream f(input,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(f)),{});if(bytes.size()<200)throw std::runtime_error("invalid sidecar");
 int64_t header[8];std::memcpy(header,bytes.data(),64);if(header[1]!=w||header[2]!=sites||header[3]!=a)throw std::runtime_error("sidecar model mismatch");
 std::vector<int> permutation(a);for(int k=0;k<a;++k){int p=members[3*w+k];permutation[k]=fileIds.at(p);if(permutation[k]<0||permutation[k]>=a)throw std::runtime_error("solute file-ID mismatch");}
 for(int k=0;k<w;++k){int o=fileIds.at(members[3*k]);if(o<a||(o-a)%sites||fileIds.at(members[3*k+1])!=o+1||fileIds.at(members[3*k+2])!=o+2)throw std::runtime_error("OHH file-ID provenance mismatch");}
 size_t offset=64+17*8;for(size_t width:{size_t(8),size_t(4),size_t(8),size_t(8)}){auto old=bytes;for(int k=0;k<a;++k)std::memcpy(bytes.data()+offset+k*width,old.data()+offset+permutation[k]*width,width);offset+=a*width;}
 const char*tmp=std::getenv("TMPDIR");std::string path=std::string(tmp?tmp:"/tmp")+"/prototype-mapped-"+std::to_string(owner)+".s2in";std::ofstream out(path,std::ios::binary);out.write(bytes.data(),bytes.size());if(!out)throw std::runtime_error("cannot write mapped sidecar");return path;
}
int64_t prepareGroups(int64_t h,const std::array<double,3>&cell,const std::array<double,3>&r,bool water,bool atom){
 Trace trace("prototype refresh and native group I/O");auto&s=state(h);check(cudaDeviceSynchronize());auto points=refs(h,TEMPLATE_FRAME_OUTER_REFERENCE,s.info.waters+s.info.atoms);
 auto er=ints(h,"_mlir_ciface_mdrtTemplatePairsMetadata",TEMPLATE_SS_ROWS),en=ints(h,"_mlir_ciface_mdrtTemplatePairsMetadata",TEMPLATE_SS_NEIGHBORS);
 buildSession(s.session,water,atom,points.data(),cell.data(),r.data(),copy(ptr(er),s.info.atoms+1),copy(ptr(en),en.size[0]));s.cell=cell;check(cudaMemcpy(s.deviceCell,cell.data(),3*sizeof(double),cudaMemcpyHostToDevice));
 auto p=sessionDeviceView(s.session);int w=s.info.waters,a=s.info.atoms;std::vector<int32_t>order,starts(w+a,0),lengths(w+a,0),atStart(w+a,0),columns(w+a,0);
 auto adopt=[&](const int32_t*device,int padded,int groups,int classOffset,int unitOffset){auto raw=copy(device,padded);for(int g=0;g<groups;++g){int begin=order.size();for(int bit=0;bit<16;++bit){int m=raw[16*g+bit];if(m>=0)order.push_back(unitOffset+m);}int length=int(order.size())-begin;if(length<1||length>16)throw std::runtime_error("invalid prototype group padding");starts[classOffset+g]=begin;lengths[classOffset+g]=length;atStart[begin]=length;}};
 adopt(p.waterOrder,p.waterPadded,p.waterGroups,0,0);adopt(p.atomOrder,p.atomPadded,p.atomGroups,w,w);
 if(order.size()!=size_t(w+a))throw std::runtime_error("prototype order did not cover native units");
 int64_t rows=symbol<int64_t(*)(int64_t,int64_t,int64_t,int64_t,int64_t)>("mdrtTemplatePairsCommitGroups")(h,p.waterGroups,p.atomGroups,water,atom);
 put(ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsOrder")),order);
 put(ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsGroupView",TEMPLATE_GROUP_STARTS)),starts);put(ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsGroupView",TEMPLATE_GROUP_LENGTHS)),lengths);put(ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsGroupView",TEMPLATE_GROUP_AT_START)),atStart);put(ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsGroupView",TEMPLATE_GROUP_COLUMNS)),columns);
 s.basis=basis(h,points);return rows;
}
void stage(int64_t h,bool inner){
 Trace trace(inner?"prototype inner native I/O":"prototype outer native I/O");auto&s=state(h);Mem<int64_t,1>controls{};symbol<void(*)(void*,int64_t)>("_mlir_ciface_mdrtTemplatePairsCounters")(&controls,h);
 stageNative(s.session,inner,ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsArenaView",TEMPLATE_LAYOUT_STAGE_UNITS)),ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsArenaView",TEMPLATE_LAYOUT_STAGE_ENTRIES)),ptr(ints(h,"_mlir_ciface_mdrtTemplatePairsArenaView",TEMPLATE_LAYOUT_BUILD_COUNTS)),reinterpret_cast<uint64_t*>(ptr(controls)),layout(h,TEMPLATE_LAYOUT_UNITS));check(cudaDeviceSynchronize());if(inner){auto flags=copy(ptr(controls)+1,1);if(flags[0])throw std::runtime_error("prototype inner I/O conversion or capacity failure");}
}
void verifyPhysical(int64_t h) {
 auto &s=state(h);if(s.physicalChecked||!std::getenv("MDIR_PROTOTYPE_PHYSICAL_CHECK"))return;
 Trace trace("diagnostic independent f64 actual-cutoff coverage");
 auto result=verifySessionPhysical(s.session,s.coordinates);uint64_t missing=0;
 for(int k=0;k<3;++k){std::printf("prototype physical coverage class %d required %llu missing %llu extra %llu\n",k,(unsigned long long)result.required[k],(unsigned long long)result.missing[k],(unsigned long long)result.extra[k]);missing+=result.missing[k];}
 std::fflush(stdout);
 if(missing){
  std::string prefix="prototype-physical-stop-"+std::to_string(h);
  auto write=[&](const char *suffix,const auto &v){std::ofstream out(prefix+suffix,std::ios::binary);out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(v[0]));if(!out)throw std::runtime_error("could not preserve physical coverage stop");};
  write(".coordinates-A.f64",copy(s.coordinates,3*s.info.particles));write(".cell-nm.f64",s.cell);write(".members.i32",s.members);write(".file-ids.i32",s.fileIds);
  PrototypeBasis b;std::vector<PrototypeEntry>entries;exportSession(s.session,true,b,entries);write(".inner-entries.bin",entries);
  std::fprintf(stderr,"prototype physical missing first class %d I %d J %d image %d %d %d\n",result.firstKind,result.firstI,result.firstJ,result.firstImage[0],result.firstImage[1],result.firstImage[2]);
  throw std::runtime_error("missing physical pair within actual cutoff; no retry");
 }
 s.physicalChecked=true;
}
template<class P> __global__ void gatherCoordinates(const P*x,int64_t stride0,int64_t stride1,const int32_t*members,int w,int a,int sites,const double*packing,const double*cell,double*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x,n=a+sites*w;if(i>=3*n)return;int atom=i/3,k=i%3,unit,p=-1,site=0;
 if(atom<a){unit=w+atom;p=members[3*w+atom];}else{unit=(atom-a)/sites;site=(atom-a)%sites;if(site<3)p=members[3*unit+site];}
 double value=0.;if(p>=0){value=double(x[p*stride0+k*stride1]);if(atom>=a&&site){double o=double(x[members[3*unit]*stride0+k*stride1]);double d=__dsub_rn(value,o);d=__dsub_rn(d,__dmul_rn(nearbyint(__ddiv_rn(d,cell[k])),cell[k]));value=__dadd_rn(o,d);}value=__dadd_rn(value,__dmul_rn(packing[3*unit+k],cell[k]));}
 out[i]=__dmul_rn(value,10.);
}
template<class F> __global__ void addForces(F*f,int64_t stride0,int64_t stride1,const int32_t*members,int w,int a,int sites,const float*in){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*(3*w+a))return;int atom=i/3,k=i%3,p=members[atom],source=atom<3*w?a+sites*(atom/3)+atom%3:atom-3*w;f[p*stride0+k*stride1]+=F(double(in[3*source+k])*41.84);}
void prepareForce(int64_t h,void*positions,const std::array<double,3>&cell,const std::array<double,3>&r,int pw,bool water,bool atom){
 Trace trace("prototype coordinate I/O pack and prune");auto&s=state(h);check(cudaDeviceSynchronize());if(cell!=s.cell)throw std::runtime_error("changing cell outside fixed-cell diagnostic");
 auto members=ints(h,"_mlir_ciface_mdrtTemplatePairsMetadata",TEMPLATE_MEMBERS);auto pos=*static_cast<Mem<double,2>*>(positions);auto packing=frame(h,TEMPLATE_FRAME_PACKING_CELLS);int count=3*s.info.particles;
 double*deviceCell=s.deviceCell;
 if(pw==8)gatherCoordinates<<<(count+127)/128,128>>>(static_cast<double*>(pos.aligned)+pos.offset,pos.stride[0],pos.stride[1],ptr(members),s.info.waters,s.info.atoms,s.info.sites,ptr(packing),deviceCell,s.coordinates);
 else gatherCoordinates<<<(count+127)/128,128>>>(reinterpret_cast<float*>(pos.aligned)+pos.offset,pos.stride[0],pos.stride[1],ptr(members),s.info.waters,s.info.atoms,s.info.sites,ptr(packing),deviceCell,s.coordinates);
 check(cudaGetLastError());prepareSession(s.session,s.coordinates);if(water||atom){prunePreparedSession(s.session,water,atom,r.data());stage(h,true); }verifyPhysical(h);
}
void publishForce(int64_t h,void*forces,int fw){
 Trace trace("prototype evaluation unpack and force I/O");auto&s=state(h);auto out=*static_cast<Mem<double,2>*>(forces);auto members=ints(h,"_mlir_ciface_mdrtTemplatePairsMetadata",TEMPLATE_MEMBERS);const float*f=forcePreparedSession(s.session);int real=3*(3*s.info.waters+s.info.atoms);
 if(fw==8)addForces<<<(real+127)/128,128>>>(static_cast<double*>(out.aligned)+out.offset,out.stride[0],out.stride[1],ptr(members),s.info.waters,s.info.atoms,s.info.sites,f);
 else addForces<<<(real+127)/128,128>>>(reinterpret_cast<float*>(out.aligned)+out.offset,out.stride[0],out.stride[1],ptr(members),s.info.waters,s.info.atoms,s.info.sites,f);
 check(cudaGetLastError());check(cudaDeviceSynchronize());
}
template<class Fn> void guarded(Fn fn){try{fn();}catch(const std::exception&e){std::fprintf(stderr,"prototype adapter STOP: %s\n",e.what());std::abort();}}
std::array<double,3> host3(void*p){auto&m=*static_cast<Mem<double,1>*>(p);if(m.size[0]!=3)throw std::runtime_error("nonorthogonal diagnostic cell/reach");return {m.aligned[m.offset],m.aligned[m.offset+m.stride[0]],m.aligned[m.offset+2*m.stride[0]]};}
}
extern "C" void _mlir_ciface_mdrtPrototypeRegister(int64_t h,void*idsArg,int64_t sites){guarded([&]{check(cudaDeviceSynchronize());auto ids=*static_cast<Mem<int32_t,1>*>(idsArg);auto file=copy(ptr(ids),ids.size[0]);auto s=std::make_unique<State>();s->fileIds=file;int w=layout(h,TEMPLATE_LAYOUT_WATERS),a=layout(h,TEMPLATE_LAYOUT_ATOMS);auto m=ints(h,"_mlir_ciface_mdrtTemplatePairsMetadata",TEMPLATE_MEMBERS);s->members=copy(ptr(m),3*w+a);const char*input=std::getenv("MDIR_PROTOTYPE_SIDECAR");if(!input)throw std::runtime_error("MDIR_PROTOTYPE_SIDECAR required");auto mapped=mappedInput(input,file,s->members,w,a,sites,h);s->session=createSession(mapped);s->info=sessionInfo(s->session);check(cudaMalloc(&s->coordinates,3*s->info.particles*sizeof(double)));check(cudaMalloc(&s->deviceCell,3*sizeof(double)));states[h]=std::move(s);});}
extern "C" int64_t _mlir_ciface_mdrtPrototypePrepare(int64_t h,void*cell,void*reaches,int64_t water,int64_t atom){int64_t rows=0;guarded([&]{rows=prepareGroups(h,host3(cell),host3(reaches),water,atom);});return rows;}
extern "C" void _mlir_ciface_mdrtPrototypeSearch(int64_t h){guarded([&]{stage(h,false);});}
extern "C" void _mlir_ciface_mdrtPrototypeForcePrepare(int64_t h,void*x,void*cell,void*reaches,int64_t pw,int64_t water,int64_t atom){guarded([&]{prepareForce(h,x,host3(cell),host3(reaches),pw,water,atom);});}
extern "C" void _mlir_ciface_mdrtPrototypeForcePublish(int64_t h,void*f,int64_t fw){guarded([&]{publishForce(h,f,fw);});}
extern "C" void mdrtPrototypeRelease(int64_t h){guarded([&]{auto&s=state(h);auto info=sessionInfo(s.session);std::fprintf(stdout,"prototype cadence builds %llu %llu %llu prunes %llu %llu %llu evaluations %llu\n",(unsigned long long)info.builds[0],(unsigned long long)info.builds[1],(unsigned long long)info.builds[2],(unsigned long long)info.prunes[0],(unsigned long long)info.prunes[1],(unsigned long long)info.prunes[2],(unsigned long long)info.evaluations);states.erase(h);});}

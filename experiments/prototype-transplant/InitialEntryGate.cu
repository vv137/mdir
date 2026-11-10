// Experimental initial search gate only. All archived device bodies are immutable.
#define main archived_stage2_main
#include "stage2_archived.cu"
#undef main
#include "GateInput.h"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace prototype_io;

template<class T> static void saveHost(const std::string &path,const std::vector<T> &values) {
  std::ofstream f(path,std::ios::binary);
  if(!f || !f.write(reinterpret_cast<const char *>(values.data()),std::streamsize(values.size()*sizeof(T))))
    throw std::runtime_error("cannot preserve gate raw array");
}
template<class T> static void saveDevice(const std::string &path,const T *p,size_t n) {
  std::vector<T> values(n);
  if(n)CK(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  saveHost(path,values);
}
static uint64_t bits(double value){uint64_t b;std::memcpy(&b,&value,8);return b;}
static uint32_t bits(float value){uint32_t b;std::memcpy(&b,&value,4);return b;}
static void saveNative(const std::string &path, const std::vector<NativeEntry> &entries) {
  std::ofstream f(path,std::ios::binary);
  if(!f)throw std::runtime_error("cannot save converted native entries");
  for(const auto &e:entries) {
    int32_t words[6]={e.kind,e.row,e.partner,int32_t(e.image),int32_t(e.mask),e.reserved};
    f.write(reinterpret_cast<const char *>(words),sizeof words);
  }
}
static void show(std::ostream &out,const NativeEntry &e) {
  out<<'['<<e.kind<<','<<e.row<<','<<e.partner<<','<<e.image<<','<<e.mask<<','<<e.reserved<<']';
}
int main(int argc,char **argv) try {
  if(argc!=4)throw std::runtime_error("usage: prototype-initial-entry-gate NATIVE.gate PARAMS.s2in OUTPUT_PREFIX");
  GateInput native=readGateInput(argv[1]);
  Input in=readInput(argv[2]);
  if(in.nW!=int64_t(native.waters) || in.nS!=int64_t(native.atoms) ||
      (in.ns!=3 && in.ns!=4))throw std::runtime_error("sidecar/native model-count mismatch");
  const std::string out=argv[3];
  const int nW=int(in.nW),nS=int(in.nS),natom=int(in.natom);
  if(!nW)throw std::runtime_error("water-free state outside this diagnostic");
  double radius[3];Geo geo{};
  for(int a=0;a<3;++a) {
    double cell=native.cell[a]*10.0;
    if(!std::isfinite(cell) || cell<=0 || std::abs(cell-in.L[a])>1e-6*in.L[a])
      throw std::runtime_error("native cell outside fixed archived box regime");
    in.L[a]=cell;geo.L[a]=float(cell);
    radius[a]=native.reaches[a]*10.0;
    if((a==0 || nS) && (!std::isfinite(radius[a]) || radius[a]<=0))
      throw std::runtime_error("invalid active native reach");
  }
  geo.nS=nS;geo.ns=int(in.ns);
  double vol=in.L[0]*in.L[1]*in.L[2],rho=nW/vol;
  double edge=std::cbrt(16.0/rho);
  geo.ncx=std::max(1,int(std::lround(in.L[0]/edge)));
  geo.ncy=std::max(1,int(std::lround(in.L[1]/edge)));
  const int ncol=geo.ncx*geo.ncy;
  const int nx=int(std::ceil(radius[0]/(in.L[0]/geo.ncx)));
  const int ny=int(std::ceil(radius[0]/(in.L[1]/geo.ncy)));
  if(2*nx+1>geo.ncx || 2*ny+1>geo.ncy || ncol>SCANT*SCANI)
    throw std::runtime_error("original water-column reach admission failed");
  const int sncx=std::max(1,int(std::lround(in.L[0]/6.0)));
  const int sncy=std::max(1,int(std::lround(in.L[1]/6.0)));
  if(ncol>2047 || sncx*sncy>2047)throw std::runtime_error("original key-column domain exceeded");
  const int TPB=128,nmax=std::max(nW,nS),npadCap=16*std::max(1,nW),nsPadCap=16*std::max(1,nS);
  std::vector<float4> water(nW),atoms(nS);
  for(size_t unit=0;unit<native.waters+native.atoms;++unit) {
    float c[3];
    for(int a=0;a<3;++a) {
      double value=(native.fractions[3*unit+a]*native.cell[a])*10.0;
      if(!std::isfinite(value) || std::abs(value)>double(INT32_MAX)*in.L[a])
        throw std::runtime_error("native reference outside prototype key domain");
      c[a]=float(value);
    }
    float4 value=make_float4(c[0],c[1],c[2],0.f);
    if(unit<native.waters)water[unit]=value;else atoms[unit-native.waters]=value;
  }
  saveHost(out+".water-anchor-input.f32.bin",water);
  saveHost(out+".atom-anchor-input.f32.bin",atoms);
  {
    std::ofstream f(out+".input-bits.json");
    if(!f)throw std::runtime_error("cannot save input-bit evidence");
    f<<"{\"native_cell_nm_bits\":["<<bits(native.cell[0])<<','<<bits(native.cell[1])<<','<<bits(native.cell[2])
      <<"],\"geo_cell_A_f32_bits\":["<<bits(geo.L[0])<<','<<bits(geo.L[1])<<','<<bits(geo.L[2])
      <<"],\"native_reach_nm_bits\":["<<bits(native.reaches[0])<<','<<bits(native.reaches[1])<<','<<bits(native.reaches[2])
      <<"],\"radius_A_f64_bits\":["<<bits(radius[0])<<','<<bits(radius[1])<<','<<bits(radius[2])
      <<"],\"R2_A_f32_bits\":["<<bits(float(radius[0]*radius[0]))<<','<<bits(float(radius[1]*radius[1]))<<','<<bits(float(radius[2]*radius[2]))
      <<"],\"water_grid\":["<<geo.ncx<<','<<geo.ncy<<"],\"solute_grid\":["<<sncx<<','<<sncy<<"]}\n";
  }
  float4 *ancNow=dalloc<float4>(nW),*solNow=dalloc<float4>(nS);
  CK(cudaMemcpy(ancNow,water.data(),16*nW,cudaMemcpyHostToDevice));
  if(nS)CK(cudaMemcpy(solNow,atoms.data(),16*nS,cudaMemcpyHostToDevice));
  unsigned *keys=dalloc<unsigned>(nmax),*keysS=dalloc<unsigned>(nmax);
  int *vals=dalloc<int>(nmax),*molS=dalloc<int>(nmax);
  int4 *shift=dalloc<int4>(nW),*sshift=dalloc<int4>(nS);
  int *order=dalloc<int>(npadCap),*sorder=dalloc<int>(nsPadCap);
  int *placeOf=dalloc<int>(nW),*splace=dalloc<int>(nS),*colGroups=dalloc<int>(2*ncol);
  int *totals=dalloc<int>(2),*totalsS=dalloc<int>(2);
  int *gFlag=dalloc<int>(nmax),*gSeg=dalloc<int>(nmax),*gStart=dalloc<int>(nmax);
  int *gGroups=dalloc<int>(nmax),*gLen=dalloc<int>(nmax),*gG0=dalloc<int>(nmax),*gChanged=dalloc<int>(1);
  float4 *ancSlot=dalloc<float4>(npadCap),*boxLo=dalloc<float4>(nW),*boxHi=dalloc<float4>(nW);
  float4 *sAnc=dalloc<float4>(nsPadCap),*sboxLo=dalloc<float4>(std::max(1,nS)),*sboxHi=dalloc<float4>(std::max(1,nS));
  int *dExOff=dalloc<int>(nS+1),*dExPart=dalloc<int>(native.graphNeighbors.size());
  CK(cudaMemcpy(dExOff,native.graphRows.data(),4*(nS+1),cudaMemcpyHostToDevice));
  if(!native.graphNeighbors.empty())CK(cudaMemcpy(dExPart,native.graphNeighbors.data(),4*native.graphNeighbors.size(),cudaMemcpyHostToDevice));
  // Original density-derived pool bounds; no failed-build resizing/retuning.
  const double rhoAll=natom/vol,ngEst=nS/8.0+8;
  int poolW=int(2.0*nW*(4.0/3.0*M_PI*rho*std::pow(radius[0]+6.0,3)/2.0/64.0+1.0));
  int poolSW=int(4.0*ngEst*(4.0/3.0*M_PI*rho*std::pow(radius[1]+6.0,3)/64.0+1.0))+64;
  int poolSS=int(4.0*ngEst*(4.0/3.0*M_PI*rhoAll*std::pow(radius[2]+6.0,3)/2.0/64.0+1.0))+64;
  if(poolW<=0 || (nS && (poolSW<=0 || poolSS<=0)))throw std::runtime_error("pool size domain exceeded");
  List WW,SW,SS;WW.alloc(nW,poolW);if(nS){SW.alloc(nS,poolSW);SS.alloc(nS,poolSS);}
  size_t tmpSort=0,tmpScan=0,t2=0;
  cub::DeviceRadixSort::SortPairs(nullptr,tmpSort,keys,keysS,vals,molS,nmax,0,32);
  cub::DeviceScan::InclusiveSum(nullptr,tmpScan,gFlag,gSeg,nmax);
  cub::DeviceScan::ExclusiveSum(nullptr,t2,gGroups,gG0,nmax);tmpScan=std::max(tmpScan,t2);
  void *dTmp=dalloc<char>(std::max(tmpSort,tmpScan));
  auto group=[&](int n,unsigned gapq,unsigned spanq,int *ord,int *plc,int *colG,int *tot) {
    s2g_flags<<<gridOf(n,TPB),TPB>>>(keysS,n,gapq,gFlag);
    int nseg=0;
    for(int pass=0;;++pass) {
      size_t ts=tmpScan;cub::DeviceScan::InclusiveSum(dTmp,ts,gFlag,gSeg,n);
      s2g_segments<<<gridOf(n,TPB),TPB>>>(gFlag,gSeg,n,gStart);
      s2g_lengths<<<gridOf(n,TPB),TPB>>>(gFlag,gSeg,n,gStart,gGroups,gLen);
      CK(cudaMemsetAsync(gChanged,0,4));
      s2g_span<<<gridOf(n,TPB),TPB>>>(keysS,gSeg,n,gStart,gLen,spanq,gFlag,gChanged);
      int changed;CK(cudaMemcpy(&changed,gChanged,4,cudaMemcpyDeviceToHost));
      if(!changed)break;
      if(pass==8)throw std::runtime_error("original eight-pass grouping refusal");
    }
    CK(cudaMemcpy(&nseg,gSeg+n-1,4,cudaMemcpyDeviceToHost));
    size_t ts=tmpScan;cub::DeviceScan::ExclusiveSum(dTmp,ts,gGroups,gG0,nseg);
    s2g_slots<<<gridOf(n,TPB),TPB>>>(keysS,molS,gFlag,gSeg,n,gStart,gGroups,gG0,ord,plc,colG,tot);
    int h[2];CK(cudaMemcpy(h,tot,8,cudaMemcpyDeviceToHost));return h[0];
  };
  auto finish=[&](List &list,int ng,const char *name) {
    int pool,flags;CK(cudaMemcpy(&pool,list.pool,4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&flags,list.flags,4,cudaMemcpyDeviceToHost));
    std::cout<<name<<" groups "<<ng<<" chunks "<<pool<<" flags "<<flags<<'\n';
    std::string prefix=out+"."+name;
    saveDevice(prefix+".outer-counts.i32.bin",list.outerCount,ng);
    saveDevice(prefix+".block-starts.i32.bin",list.blockStart,ng);
    saveDevice(prefix+".unit-group.i32.bin",list.unitGroup,std::min(std::max(pool,0),list.poolCap));
    saveDevice(prefix+".unit-ordinal.i32.bin",list.unitOrd,std::min(std::max(pool,0),list.poolCap));
    saveDevice(prefix+".outer-pool.int2.bin",list.outer,size_t(std::min(std::max(pool,0),list.poolCap))*64);
    std::ofstream metadata(prefix+".build.json");
    metadata<<"{\"groups\":"<<ng<<",\"chunks\":"<<pool<<",\"capacity_chunks\":"<<list.poolCap<<",\"flags\":"<<flags<<"}\n";
    metadata.close();
    saveDevice(out+".water-order-capacity.i32.bin",order,npadCap);
    saveDevice(out+".water-shifts.int4.bin",shift,nW);
    if(nS){saveDevice(out+".atom-order-capacity.i32.bin",sorder,nsPadCap);saveDevice(out+".atom-shifts.int4.bin",sshift,nS);}
    if(flags)throw std::runtime_error(std::string("original prototype build refusal ")+name+" flags="+std::to_string(flags));
    list.nBlocks=pool;list.ng=ng;
  };
  CK(cudaMemsetAsync(order,0xff,4*size_t(npadCap)));
  s2_keys<<<gridOf(nW,TPB),TPB>>>(ancNow,nW,geo,geo.ncx,geo.ncy,keys,vals,shift,nullptr);
  size_t ts=tmpSort;cub::DeviceRadixSort::SortPairs(dTmp,ts,keys,keysS,vals,molS,nW,0,32);
  int ngW=group(nW,unsigned(4.0/in.L[2]*(1u<<21)),unsigned(9.0/in.L[2]*(1u<<21)),order,placeOf,colGroups,totals);
  int npad=16*ngW;
  s2_boxes<<<gridOf(npad,TPB),TPB>>>(ancNow,shift,order,npad,geo,ancSlot,boxLo,boxHi);
  BuildArgs wargs{ancSlot,boxLo,boxHi,ancSlot,boxLo,boxHi,ngW,ngW,colGroups,geo,nx,ny,float(radius[0]*radius[0]),nullptr,nullptr,nullptr,
      WW.pool,WW.poolCap,WW.outer,WW.outerCount,WW.blockStart,WW.unitGroup,WW.unitOrd,WW.flags};
  s2_build<0><<<gridOf(ngW,BW),32*BW>>>(wargs);CK(cudaGetLastError());finish(WW,ngW,"WW");
  int ngS=0;
  if(nS) {
    CK(cudaMemsetAsync(sorder,0xff,4*size_t(nsPadCap)));
    s2_keys<<<gridOf(nS,TPB),TPB>>>(solNow,nS,geo,sncx,sncy,keys,vals,sshift,nullptr);
    ts=tmpSort;cub::DeviceRadixSort::SortPairs(dTmp,ts,keys,keysS,vals,molS,nS,0,32);
    ngS=group(nS,unsigned(3.0/in.L[2]*(1u<<21)),unsigned(6.0/in.L[2]*(1u<<21)),sorder,splace,nullptr,totalsS);
    s2_boxes<<<gridOf(16*ngS,TPB),TPB>>>(solNow,sshift,sorder,16*ngS,geo,sAnc,sboxLo,sboxHi);
    BuildArgs a{sAnc,sboxLo,sboxHi,ancSlot,boxLo,boxHi,ngS,ngW,nullptr,geo,0,0,float(radius[1]*radius[1]),nullptr,nullptr,nullptr,
        SW.pool,SW.poolCap,SW.outer,SW.outerCount,SW.blockStart,SW.unitGroup,SW.unitOrd,SW.flags};
    s2_build<1><<<gridOf(ngS,BW),32*BW>>>(a);
    BuildArgs z{sAnc,sboxLo,sboxHi,sAnc,sboxLo,sboxHi,ngS,ngS,nullptr,geo,0,0,float(radius[2]*radius[2]),sorder,dExOff,dExPart,
        SS.pool,SS.poolCap,SS.outer,SS.outerCount,SS.blockStart,SS.unitGroup,SS.unitOrd,SS.flags};
    s2_build<2><<<gridOf(ngS,BW),32*BW>>>(z);CK(cudaGetLastError());finish(SW,ngS,"SW");finish(SS,ngS,"SS");
  }
  PrototypeBasis proto{};proto.waters=nW;proto.waterOrder.resize(npad);proto.atomOrder.resize(16*ngS);proto.shifts.resize(nW+nS);
  CK(cudaMemcpy(proto.waterOrder.data(),order,4*npad,cudaMemcpyDeviceToHost));
  if(ngS)CK(cudaMemcpy(proto.atomOrder.data(),sorder,4*proto.atomOrder.size(),cudaMemcpyDeviceToHost));
  for(auto &v:proto.atomOrder)if(v>=0)v+=nW;
  std::vector<int4> shifts(nW),sShifts(nS);CK(cudaMemcpy(shifts.data(),shift,16*nW,cudaMemcpyDeviceToHost));
  if(nS)CK(cudaMemcpy(sShifts.data(),sshift,16*nS,cudaMemcpyDeviceToHost));
  for(int i=0;i<nW+nS;++i){int4 s=i<nW?shifts[i]:sShifts[i-nW];proto.shifts[i]={s.x,s.y,s.z};}
  std::vector<PrototypeEntry> raw;
  auto collect=[&](const List &list,int kind) {
    std::vector<int> counts(list.ng),starts(list.ng);std::vector<int2> entries(size_t(list.nBlocks)*64);
    CK(cudaMemcpy(counts.data(),list.outerCount,4*list.ng,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(starts.data(),list.blockStart,4*list.ng,cudaMemcpyDeviceToHost));
    if(!entries.empty())CK(cudaMemcpy(entries.data(),list.outer,8*entries.size(),cudaMemcpyDeviceToHost));
    for(int g=0;g<list.ng;++g)for(int k=0;k<counts[g];++k){auto e=entries[size_t(starts[g])*64+k];raw.push_back({kind,g,e.x,uint32_t(e.y)});}
  };
  collect(WW,0);if(nS){collect(SW,1);collect(SS,2);}
  {
    std::vector<int32_t> words;
    for(auto &e:raw){words.push_back(e.kind);words.push_back(e.group);words.push_back(e.partnerSlot);words.push_back(int32_t(e.packed));}
    saveHost(out+".prototype-raw-entries.i32.bin",words);
  }
  auto physical=expandPrototype(raw,proto,native.native);auto converted=regroup(physical,native.native);
  saveNative(out+".native-entries.bin",converted);
  auto expectedPhysical=expandNative(native.expected,native.native);
  std::vector<PhysicalPair> missingPhysical,extraPhysical;
  std::set_difference(expectedPhysical.begin(),expectedPhysical.end(),physical.begin(),physical.end(),std::back_inserter(missingPhysical));
  std::set_difference(physical.begin(),physical.end(),expectedPhysical.begin(),expectedPhysical.end(),std::back_inserter(extraPhysical));
  // Diagnostic-only physical ruling: list reach differences are reported.
  // Actual cutoff coverage and physical observables gate later evaluations.
  // The production exact native-mask contract is unchanged.
  bool pass=true;
  std::vector<NativeEntry> missing,extra;
  std::set_difference(native.expected.begin(),native.expected.end(),converted.begin(),converted.end(),std::back_inserter(missing));
  std::set_difference(converted.begin(),converted.end(),native.expected.begin(),native.expected.end(),std::back_inserter(extra));
  std::ofstream report(out+".json");
  report<<"{\"pass\":"<<(pass?"true":"false")<<",\"physical_gate_performed\":false,\"prototype_raw_entries\":"<<raw.size()
      <<",\"physical_pairs\":"<<physical.size()<<",\"expected_physical_pairs\":"<<expectedPhysical.size()
      <<",\"missing_physical_pair_bits\":"<<missingPhysical.size()<<",\"extra_physical_pair_bits\":"<<extraPhysical.size()
      <<",\"native_entries\":"<<converted.size()
      <<",\"expected_entries\":"<<native.expected.size()<<",\"missing_native_entries\":"<<missing.size()
      <<",\"extra_native_entries\":"<<extra.size()<<",\"prototype_groups\":["<<ngW<<','<<ngS<<']'
      <<",\"actual_snapshot_head\":\"f9ea87ef8788807dab9924b970faee23391b3b7d\"";
  if(!missing.empty()){report<<",\"first_missing\":";show(report,missing[0]);}
  if(!extra.empty()){report<<",\"first_extra\":";show(report,extra[0]);}
  report<<"}\n";
  std::cout<<(pass?"PASS":"STOP: MISSING PHYSICAL PAIR")<<" converted="<<converted.size()<<" expected="<<native.expected.size()
      <<" missing_bits="<<missingPhysical.size()<<" extra_bits="<<extraPhysical.size()
      <<" differing_old_records="<<missing.size()<<" differing_new_records="<<extra.size()<<'\n';
  return pass?0:4;
} catch(const std::exception &e) { std::cerr<<"STOP: "<<e.what()<<'\n';return 3; }

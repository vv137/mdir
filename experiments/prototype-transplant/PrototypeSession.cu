// Never-merged diagnostic: host lifecycle and format adaptation only.
#define main archived_stage2_main
#include "stage2_archived.cu"
#undef main
#include "PrototypeSession.h"
#include "PhysicalCoverage.h"
#include <stdexcept>

namespace prototype_io {
struct Session {
  Input in;
  int nW, nS, ns, natom, ntS, nmax, npadCap, nsPadCap;
  int ngW=0, ngS=0, npad=0, nsPad=0, sncx=0, sncy=0, nx=0, ny=0;
  Geo geo{}; Params p{}; Table table;
  bool allocated=false, innerReady=false, packedReady=false;
  int pruningMode=0;
  std::vector<void *> allocations;
  std::vector<double> innerFractions;
  uint64_t evaluations=0;
  List WW{}, SW{}, SS{};
  float4 *ancNow=nullptr,*solNow=nullptr,*ancSlot=nullptr,*boxLo=nullptr,*boxHi=nullptr;
  float4 *sAnc=nullptr,*sboxLo=nullptr,*sboxHi=nullptr,*pos=nullptr,*ancFrame=nullptr,*spos=nullptr;
  float4 *pruneWater=nullptr,*pruneAtom=nullptr,*dLjO=nullptr,*dLjTab=nullptr,*sLJ=nullptr,*dTab=nullptr;
  float *fW=nullptr,*fS=nullptr,*fatom=nullptr,*dQ=nullptr,*sq=nullptr;
  int *vals=nullptr,*molS=nullptr,*colGroups=nullptr,*totals=nullptr,*totalsS=nullptr;
  int *order=nullptr,*placeOf=nullptr,*sorder=nullptr,*splace=nullptr,*stype=nullptr;
  int *gFlag=nullptr,*gSeg=nullptr,*gStart=nullptr,*gGroups=nullptr,*gLen=nullptr,*gG0=nullptr,*gChanged=nullptr;
  int *dType=nullptr,*dExOff=nullptr,*dExPart=nullptr;
  unsigned *keys=nullptr,*keysS=nullptr;
  int4 *shift=nullptr,*sshift=nullptr;
  void *dTmp=nullptr;
  size_t tmpSort=0,tmpScan=0,tmpSel=0;
  explicit Session(const std::string &path):in(readInput(path.c_str())),nW(in.nW),nS(in.nS),ns(in.ns),
      natom(in.natom),ntS(in.ntS),nmax(std::max(nW,nS)),npadCap(16*std::max(1,nW)),nsPadCap(16*std::max(1,nS)) {
    if(nW<=0 || nS<0 || (ns!=3 && ns!=4) || ntS<0 || ntS>32 ||
       int64_t(nW)*ns+nS!=natom) throw std::runtime_error("prototype sidecar dimensions outside diagnostic domain");
  }
  template<class T> T *alloc(size_t n) {auto *v=dalloc<T>(n);allocations.push_back(v);return v;}
  static void freeList(List &l) {
    void *values[]={l.pool,l.flags,l.outerCount,l.innerCount,l.blockStart,
        l.unitGroup,l.unitOrd,l.activeBlocks,l.nActive,l.outer,l.inner};
    for(void *v:values)if(v)cudaFree(v);
  }
  ~Session(){if(allocated){freeList(WW);if(nS){freeList(SW);freeList(SS);}}for(void *v:allocations)cudaFree(v);}
  void allocate(const double cellNm[3],const double reachesNm[3],
      const std::vector<int32_t> &rows,const std::vector<int32_t> &neighbors) {
    double radius[3];
    for(int a=0;a<3;++a){double cell=10*cellNm[a];
      if(!std::isfinite(cell)||cell<=0||std::abs(cell-in.L[a])>1e-6*in.L[a])
        throw std::runtime_error("prototype requires the fixed archived orthorhombic box");
      in.L[a]=cell;geo.L[a]=float(cell);radius[a]=10*reachesNm[a];p.L[a]=float(cell);}
    geo.nS=nS;geo.ns=ns;
    double volume=in.L[0]*in.L[1]*in.L[2],rho=nW/volume,edge=std::cbrt(16/rho);
    geo.ncx=std::max(1,int(std::lround(in.L[0]/edge)));geo.ncy=std::max(1,int(std::lround(in.L[1]/edge)));
    int ncol=geo.ncx*geo.ncy;
    nx=int(std::ceil(radius[0]/(in.L[0]/geo.ncx)));ny=int(std::ceil(radius[0]/(in.L[1]/geo.ncy)));
    sncx=std::max(1,int(std::lround(in.L[0]/6)));sncy=std::max(1,int(std::lround(in.L[1]/6)));
    if(2*nx+1>geo.ncx||2*ny+1>geo.ncy||ncol>SCANT*SCANI||ncol>2047||sncx*sncy>2047)
      throw std::runtime_error("prototype column admission failed");
    if(rows.size()!=size_t(nS+1)||rows.front()!=0||rows.back()!=int(neighbors.size()))
      throw std::runtime_error("prototype SS graph shape mismatch");
    ancNow=alloc<float4>(nW);solNow=alloc<float4>(nS);
    keys=alloc<unsigned>(nmax);keysS=alloc<unsigned>(nmax);vals=alloc<int>(nmax);molS=alloc<int>(nmax);
    colGroups=alloc<int>(2*ncol);totals=alloc<int>(2);totalsS=alloc<int>(2);
    gFlag=alloc<int>(nmax);gSeg=alloc<int>(nmax);gStart=alloc<int>(nmax);gGroups=alloc<int>(nmax);
    gLen=alloc<int>(nmax);gG0=alloc<int>(nmax);gChanged=alloc<int>(1);
    order=alloc<int>(npadCap);placeOf=alloc<int>(nW);shift=alloc<int4>(nW);
    sorder=alloc<int>(nsPadCap);splace=alloc<int>(nS);sshift=alloc<int4>(nS);stype=alloc<int>(nsPadCap);
    ancSlot=alloc<float4>(npadCap);boxLo=alloc<float4>(nW);boxHi=alloc<float4>(nW);
    sAnc=alloc<float4>(nsPadCap);sboxLo=alloc<float4>(std::max(1,nS));sboxHi=alloc<float4>(std::max(1,nS));
    pos=alloc<float4>(3*size_t(npadCap));ancFrame=alloc<float4>(npadCap);spos=alloc<float4>(nsPadCap);
    pruneWater=alloc<float4>(npadCap);pruneAtom=alloc<float4>(nsPadCap);
    fW=alloc<float>(9*size_t(npadCap));fS=alloc<float>(3*size_t(nsPadCap));fatom=alloc<float>(3*size_t(natom));
    dType=alloc<int>(nS);dQ=alloc<float>(nS);dLjO=alloc<float4>(nS);dLjTab=alloc<float4>(std::max(1,ntS*ntS));
    sq=alloc<float>(nsPadCap);sLJ=alloc<float4>(nsPadCap);
    dExOff=alloc<int>(nS+1);dExPart=alloc<int>(neighbors.size());
    CK(cudaMemcpy(dExOff,rows.data(),4*rows.size(),cudaMemcpyHostToDevice));
    if(!neighbors.empty())CK(cudaMemcpy(dExPart,neighbors.data(),4*neighbors.size(),cudaMemcpyHostToDevice));
    if(nS){std::vector<float> q(nS);std::vector<float4> lj(nS),lt(std::max(1,ntS*ntS));
      for(int i=0;i<nS;++i){q[i]=float(in.q[i]);lj[i]=make_float4(float(12*in.AOS[i]),float(6*in.BOS[i]),float(in.AOS[i]),float(in.BOS[i]));}
      for(int i=0;i<ntS*ntS;++i)lt[i]=make_float4(float(12*in.LA[i]),float(6*in.LB[i]),float(in.LA[i]),float(in.LB[i]));
      CK(cudaMemcpy(dQ,q.data(),4*nS,cudaMemcpyHostToDevice));CK(cudaMemcpy(dType,in.type.data(),4*nS,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dLjO,lj.data(),16*nS,cudaMemcpyHostToDevice));CK(cudaMemcpy(dLjTab,lt.data(),16*lt.size(),cudaMemcpyHostToDevice));}
    double rhoAll=natom/volume,ngEst=nS/8.0+8;
    int poolW=int(2.0*nW*(4.0/3.0*M_PI*rho*std::pow(radius[0]+6,3)/2/64+1));
    int poolSW=int(4*ngEst*(4.0/3.0*M_PI*rho*std::pow(radius[1]+6,3)/64+1))+64;
    int poolSS=int(4*ngEst*(4.0/3.0*M_PI*rhoAll*std::pow(radius[2]+6,3)/2/64+1))+64;
    if(poolW<=0||(nS&&(poolSW<=0||poolSS<=0)))throw std::runtime_error("prototype pool capacity domain exceeded");
    WW.alloc(nW,poolW);if(nS){SW.alloc(nS,poolSW);SS.alloc(nS,poolSS);}allocated=true;
    cub::DeviceRadixSort::SortPairs(nullptr,tmpSort,keys,keysS,vals,molS,nmax,0,32);
    cub::DeviceScan::InclusiveSum(nullptr,tmpScan,gFlag,gSeg,nmax);size_t t=0;
    cub::DeviceScan::ExclusiveSum(nullptr,t,gGroups,gG0,nmax);tmpScan=std::max(tmpScan,t);
    ActivePred pred{WW.unitGroup,WW.unitOrd,WW.innerCount};
    cub::DeviceSelect::If(nullptr,tmpSel,thrust::counting_iterator<int>(0),WW.activeBlocks,WW.nActive,std::max(poolW,std::max(poolSW,poolSS)),pred);
    dTmp=alloc<char>(std::max(tmpSort,std::max(tmpScan,tmpSel)));
    p.rc2=float(in.rc*in.rc);p.A=float(in.AOO);p.B=float(in.BOO);p.A12=12*p.A;p.B6=6*p.B;
    float ke=float(in.KE),q[4];for(int i=0;i<4;++i)q[i]=float(in.qw[i]);
    for(int i=0;i<4;++i){p.kq[i]=ke*q[i];for(int j=0;j<4;++j)p.k[i][j]=ke*q[i]*q[j];}
    p.a=float(in.a);p.oma=1-2*p.a;p.invh=float(1/in.rOH);p.invs=float(1/in.gs);p.beta=float(in.beta);
    table=makeTable(in.rc,in.beta);dTab=alloc<float4>(table.c.size());
    CK(cudaMemcpy(dTab,table.c.data(),16*table.c.size(),cudaMemcpyHostToDevice));
    innerFractions.resize(3*size_t(nW+nS));
  }
  int group(int n,unsigned gap,unsigned span,int *ord,int *plc,int *columns,int *total) {
    s2g_flags<<<gridOf(n,128),128>>>(keysS,n,gap,gFlag);
    int segments=0;
    for(int pass=0;;++pass){size_t bytes=tmpScan;cub::DeviceScan::InclusiveSum(dTmp,bytes,gFlag,gSeg,n);
      s2g_segments<<<gridOf(n,128),128>>>(gFlag,gSeg,n,gStart);
      s2g_lengths<<<gridOf(n,128),128>>>(gFlag,gSeg,n,gStart,gGroups,gLen);CK(cudaMemsetAsync(gChanged,0,4));
      s2g_span<<<gridOf(n,128),128>>>(keysS,gSeg,n,gStart,gLen,span,gFlag,gChanged);
      int changed;CK(cudaMemcpy(&changed,gChanged,4,cudaMemcpyDeviceToHost));if(!changed)break;
      if(pass==8)throw std::runtime_error("original grouping span admission failed");}
    CK(cudaMemcpy(&segments,gSeg+n-1,4,cudaMemcpyDeviceToHost));size_t bytes=tmpScan;
    cub::DeviceScan::ExclusiveSum(dTmp,bytes,gGroups,gG0,segments);
    s2g_slots<<<gridOf(n,128),128>>>(keysS,molS,gFlag,gSeg,n,gStart,gGroups,gG0,ord,plc,columns,total);
    int h[2];CK(cudaMemcpy(h,total,8,cudaMemcpyDeviceToHost));return h[0];
  }
  void finish(List &l,int groups) {
    int count,flags;CK(cudaMemcpy(&count,l.pool,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&flags,l.flags,4,cudaMemcpyDeviceToHost));
    if(flags||count<0||count>l.poolCap)throw std::runtime_error("original prototype build refusal");
    l.nBlocks=count;l.ng=groups;++l.builds;
  }
  void select(List &l) {size_t bytes=tmpSel;ActivePred pred{l.unitGroup,l.unitOrd,l.innerCount};
    cub::DeviceSelect::If(dTmp,bytes,thrust::counting_iterator<int>(0),l.activeBlocks,l.nActive,l.nBlocks,pred);++l.prunes;}
};

Session *createSession(const std::string &sidecar){return new Session(sidecar);}
void destroySession(Session *s){delete s;}
SessionInfo sessionInfo(const Session *s){return {s->nW,s->nS,s->ns,s->natom,
    {uint64_t(s->WW.builds),uint64_t(s->SW.builds),uint64_t(s->SS.builds)},
    {uint64_t(s->WW.prunes),uint64_t(s->SW.prunes),uint64_t(s->SS.prunes)},s->evaluations};}
SessionDeviceView sessionDeviceView(const Session *s) {
  auto list=[](const List &v)->SessionDeviceList{return {v.outer,v.inner,v.outerCount,v.innerCount,
      v.blockStart,v.unitGroup,v.unitOrd,v.activeBlocks,v.nActive,v.ng,v.nBlocks};};
  return {s->order,s->sorder,s->shift,s->sshift,s->ngW,s->ngS,s->npad,s->nsPad,
      {list(s->WW),list(s->SW),list(s->SS)}};
}

void buildSession(Session *s,bool water,bool atom,const double *refs,const double cellNm[3],const double reachesNm[3],
    const std::vector<int32_t> &rows,const std::vector<int32_t> &neighbors) {
  if(!s->allocated)s->allocate(cellNm,reachesNm,rows,neighbors);
  for(int a=0;a<3;++a)if(cellNm[a]*10!=s->in.L[a])throw std::runtime_error("prototype fixed-cell diagnostic received a changed box");
  std::vector<float4> w(s->nW),a(s->nS);
  for(int i=0;i<s->nW+s->nS;++i){float x[3];for(int k=0;k<3;++k)x[k]=float((refs[3*i+k]*cellNm[k])*10);
    auto v=make_float4(x[0],x[1],x[2],0);if(i<s->nW)w[i]=v;else a[i-s->nW]=v;}
  CK(cudaMemcpy(s->ancNow,w.data(),16*w.size(),cudaMemcpyHostToDevice));
  if(s->nS)CK(cudaMemcpy(s->solNow,a.data(),16*a.size(),cudaMemcpyHostToDevice));
  double r[3];for(int k=0;k<3;++k)r[k]=10*reachesNm[k];
  if(water){CK(cudaMemsetAsync(s->colGroups,0,8*s->geo.ncx*s->geo.ncy));CK(cudaMemsetAsync(s->order,0xff,4*s->npadCap));
    CK(cudaMemsetAsync(s->WW.pool,0,4));CK(cudaMemsetAsync(s->WW.flags,0,4));
    s2_keys<<<gridOf(s->nW,128),128>>>(s->ancNow,s->nW,s->geo,s->geo.ncx,s->geo.ncy,s->keys,s->vals,s->shift,nullptr);
    size_t bytes=s->tmpSort;cub::DeviceRadixSort::SortPairs(s->dTmp,bytes,s->keys,s->keysS,s->vals,s->molS,s->nW,0,32);
    s->ngW=s->group(s->nW,unsigned(4/s->in.L[2]*(1u<<21)),unsigned(9/s->in.L[2]*(1u<<21)),s->order,s->placeOf,s->colGroups,s->totals);s->npad=16*s->ngW;
    s2_boxes<<<gridOf(s->npad,128),128>>>(s->ancNow,s->shift,s->order,s->npad,s->geo,s->ancSlot,s->boxLo,s->boxHi);
    BuildArgs args{s->ancSlot,s->boxLo,s->boxHi,s->ancSlot,s->boxLo,s->boxHi,s->ngW,s->ngW,s->colGroups,s->geo,s->nx,s->ny,float(r[0]*r[0]),nullptr,nullptr,nullptr,
      s->WW.pool,s->WW.poolCap,s->WW.outer,s->WW.outerCount,s->WW.blockStart,s->WW.unitGroup,s->WW.unitOrd,s->WW.flags};
    s2_build<0><<<gridOf(s->ngW,BW),32*BW>>>(args);CK(cudaGetLastError());s->finish(s->WW,s->ngW);}
  if(s->nS&&(water||atom)){
    if(atom){CK(cudaMemsetAsync(s->sorder,0xff,4*s->nsPadCap));
      s2_keys<<<gridOf(s->nS,128),128>>>(s->solNow,s->nS,s->geo,s->sncx,s->sncy,s->keys,s->vals,s->sshift,nullptr);
      size_t bytes=s->tmpSort;cub::DeviceRadixSort::SortPairs(s->dTmp,bytes,s->keys,s->keysS,s->vals,s->molS,s->nS,0,32);
      s->ngS=s->group(s->nS,unsigned(3/s->in.L[2]*(1u<<21)),unsigned(6/s->in.L[2]*(1u<<21)),s->sorder,s->splace,nullptr,s->totalsS);s->nsPad=16*s->ngS;
      s2s_attr<<<gridOf(s->nsPad,128),128>>>(s->sorder,s->nsPad,s->dQ,s->dType,s->dLjO,s->sq,s->stype,s->sLJ);
      s2_boxes<<<gridOf(s->nsPad,128),128>>>(s->solNow,s->sshift,s->sorder,s->nsPad,s->geo,s->sAnc,s->sboxLo,s->sboxHi);}
    CK(cudaMemsetAsync(s->SW.pool,0,4));CK(cudaMemsetAsync(s->SW.flags,0,4));
    BuildArgs args{s->sAnc,s->sboxLo,s->sboxHi,s->ancSlot,s->boxLo,s->boxHi,s->ngS,s->ngW,nullptr,s->geo,0,0,float(r[1]*r[1]),nullptr,nullptr,nullptr,
      s->SW.pool,s->SW.poolCap,s->SW.outer,s->SW.outerCount,s->SW.blockStart,s->SW.unitGroup,s->SW.unitOrd,s->SW.flags};
    s2_build<1><<<gridOf(s->ngS,BW),32*BW>>>(args);CK(cudaGetLastError());s->finish(s->SW,s->ngS);
    if(atom){CK(cudaMemsetAsync(s->SS.pool,0,4));CK(cudaMemsetAsync(s->SS.flags,0,4));
      BuildArgs ss{s->sAnc,s->sboxLo,s->sboxHi,s->sAnc,s->sboxLo,s->sboxHi,s->ngS,s->ngS,nullptr,s->geo,0,0,float(r[2]*r[2]),s->sorder,s->dExOff,s->dExPart,
        s->SS.pool,s->SS.poolCap,s->SS.outer,s->SS.outerCount,s->SS.blockStart,s->SS.unitGroup,s->SS.unitOrd,s->SS.flags};
      s2_build<2><<<gridOf(s->ngS,BW),32*BW>>>(ss);CK(cudaGetLastError());s->finish(s->SS,s->ngS);}}
  CK(cudaDeviceSynchronize());
  s->packedReady=false;
}

void pruneSession(Session *s,bool water,bool atom,const double *current,const double innerNm[3]) {
  if(!s->allocated)throw std::runtime_error("prototype prune before build");
  if(s->pruningMode==2)throw std::runtime_error("cannot mix host and prepared prune reference APIs");
  s->pruningMode=1;
  if(!s->innerReady){water=true;atom=s->nS!=0;}
  for(int i=0;i<s->nW+s->nS;++i)if(i<s->nW?water:atom)
    std::copy(current+3*i,current+3*i+3,s->innerFractions.begin()+3*i);
  std::vector<int> wo(s->npad),so(s->nsPad);std::vector<int4> ws(s->nW),as(s->nS);
  CK(cudaMemcpy(wo.data(),s->order,4*wo.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(ws.data(),s->shift,16*ws.size(),cudaMemcpyDeviceToHost));
  if(s->nS){CK(cudaMemcpy(so.data(),s->sorder,4*so.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(as.data(),s->sshift,16*as.size(),cudaMemcpyDeviceToHost));}
  auto packed=[&](const std::vector<int> &order,const std::vector<int4> &shift,int offset){std::vector<float4> out(order.size());
    for(size_t k=0;k<order.size();++k){int m=order[k];if(m<0){out[k]=make_float4(-2e4f,-2e4f,-2e4f,0);continue;}
      float c[3];int4 sh=shift[m];int si[3]={sh.x,sh.y,sh.z};
      for(int a=0;a<3;++a)c[a]=float(s->innerFractions[3*(m+offset)+a]*s->in.L[a]+double(si[a])*s->geo.L[a]);
      out[k]=make_float4(c[0],c[1],c[2],1);}
    return out;};
  auto pw=packed(wo,ws,0);CK(cudaMemcpy(s->pruneWater,pw.data(),16*pw.size(),cudaMemcpyHostToDevice));
  if(s->nS){auto pa=packed(so,as,s->nW);CK(cudaMemcpy(s->pruneAtom,pa.data(),16*pa.size(),cudaMemcpyHostToDevice));}
  if(water){double r=10*innerNm[0];s2_prune<<<gridOf(s->ngW,4),128>>>(s->pruneWater,s->pruneWater,s->ngW,s->geo,float(r*r),s->WW.outer,s->WW.outerCount,s->WW.blockStart,s->WW.inner,s->WW.innerCount);s->select(s->WW);}
  if(s->nS&&(water||atom)){double r=10*innerNm[1];s2_prune<<<gridOf(s->ngS,4),128>>>(s->pruneAtom,s->pruneWater,s->ngS,s->geo,float(r*r),s->SW.outer,s->SW.outerCount,s->SW.blockStart,s->SW.inner,s->SW.innerCount);s->select(s->SW);}
  if(s->nS&&atom){double r=10*innerNm[2];s2_prune<<<gridOf(s->ngS,4),128>>>(s->pruneAtom,s->pruneAtom,s->ngS,s->geo,float(r*r),s->SS.outer,s->SS.outerCount,s->SS.blockStart,s->SS.inner,s->SS.innerCount);s->select(s->SS);}
  CK(cudaGetLastError());CK(cudaDeviceSynchronize());s->innerReady=true;
}

void exportSession(Session *s,bool inner,PrototypeBasis &basis,std::vector<PrototypeEntry> &out) {
  basis={};basis.waters=s->nW;basis.waterOrder.resize(s->npad);basis.atomOrder.resize(s->nsPad);basis.shifts.resize(s->nW+s->nS);
  CK(cudaMemcpy(basis.waterOrder.data(),s->order,4*s->npad,cudaMemcpyDeviceToHost));
  std::vector<int4> ws(s->nW),as(s->nS);CK(cudaMemcpy(ws.data(),s->shift,16*s->nW,cudaMemcpyDeviceToHost));
  if(s->nS){CK(cudaMemcpy(basis.atomOrder.data(),s->sorder,4*s->nsPad,cudaMemcpyDeviceToHost));CK(cudaMemcpy(as.data(),s->sshift,16*s->nS,cudaMemcpyDeviceToHost));}
  for(int &i:basis.atomOrder)if(i>=0)i+=s->nW;
  for(int i=0;i<s->nW+s->nS;++i){auto v=i<s->nW?ws[i]:as[i-s->nW];basis.shifts[i]={v.x,v.y,v.z};}
  out.clear();
  auto read=[&](const List &list,int kind){std::vector<int> counts(list.ng),starts(list.ng);std::vector<int2> pool(64*size_t(list.nBlocks));
    CK(cudaMemcpy(counts.data(),inner?list.innerCount:list.outerCount,4*counts.size(),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(starts.data(),list.blockStart,4*starts.size(),cudaMemcpyDeviceToHost));
    if(!pool.empty())CK(cudaMemcpy(pool.data(),inner?list.inner:list.outer,8*pool.size(),cudaMemcpyDeviceToHost));
    for(int g=0;g<list.ng;++g)for(int k=0;k<counts[g];++k){auto e=pool[64*size_t(starts[g])+k];out.push_back({kind,g,e.x,uint32_t(e.y)});}};
  read(s->WW,0);if(s->nS){read(s->SW,1);read(s->SS,2);}
}

void prepareSession(Session *s,const double *x) {
  if(!s->allocated)throw std::runtime_error("prototype pack before build");
  s2w_pack<double><<<gridOf(s->npad,128),128>>>(x,s->order,s->shift,s->npad,s->geo,s->pos,s->ancFrame);
  if(s->nS)s2s_pack<double><<<gridOf(s->nsPad,128),128>>>(x,s->sorder,s->sshift,s->sq,s->nsPad,s->geo,s->spos);
  CK(cudaGetLastError());s->packedReady=true;
}

void prunePreparedSession(Session *s,bool water,bool atom,const double innerNm[3]) {
  if(!s->packedReady)throw std::runtime_error("prototype prepared prune requires current packed coordinates");
  if(s->pruningMode==1)throw std::runtime_error("cannot mix host and prepared prune reference APIs");
  s->pruningMode=2;
  if(!s->innerReady){water=true;atom=s->nS!=0;}
  if(water)CK(cudaMemcpyAsync(s->pruneWater,s->ancFrame,16*size_t(s->npad),cudaMemcpyDeviceToDevice));
  if(s->nS&&atom)CK(cudaMemcpyAsync(s->pruneAtom,s->spos,16*size_t(s->nsPad),cudaMemcpyDeviceToDevice));
  if(water){double r=10*innerNm[0];s2_prune<<<gridOf(s->ngW,4),128>>>(s->pruneWater,s->pruneWater,s->ngW,s->geo,float(r*r),s->WW.outer,s->WW.outerCount,s->WW.blockStart,s->WW.inner,s->WW.innerCount);s->select(s->WW);}
  if(s->nS&&(water||atom)){double r=10*innerNm[1];s2_prune<<<gridOf(s->ngS,4),128>>>(s->pruneAtom,s->pruneWater,s->ngS,s->geo,float(r*r),s->SW.outer,s->SW.outerCount,s->SW.blockStart,s->SW.inner,s->SW.innerCount);s->select(s->SW);}
  if(s->nS&&atom){double r=10*innerNm[2];s2_prune<<<gridOf(s->ngS,4),128>>>(s->pruneAtom,s->pruneAtom,s->ngS,s->geo,float(r*r),s->SS.outer,s->SS.outerCount,s->SS.blockStart,s->SS.inner,s->SS.innerCount);s->select(s->SS);}
  CK(cudaGetLastError());CK(cudaDeviceSynchronize());s->innerReady=true;
}

const float *forcePreparedSession(Session *s) {
  if(!s->innerReady||!s->packedReady)throw std::runtime_error("prototype evaluation requires prune and current packed coordinates");
  CK(cudaMemsetAsync(s->fW,0,36*size_t(s->npad)));if(s->nS)CK(cudaMemsetAsync(s->fS,0,12*size_t(s->nsPad)));
  CK(cudaMemsetAsync(s->fatom,0,12*size_t(s->natom)));
  auto vw=s->WW.view();
  if(s->nS){auto vs=s->SW.view(),vv=s->SS.view();int grid=gridOf(int64_t(s->WW.nBlocks)+s->SW.nBlocks+s->SS.nBlocks,4);size_t shared=16*size_t(std::max(1,s->ntS*s->ntS));
    if(s->ns==4)all_force<4,false><<<grid,128,shared>>>(s->pos,s->spos,s->sLJ,s->stype,s->dLjTab,s->ntS,vw,vs,vv,s->fW,s->fS,s->p,float(s->in.KE),s->dTab,s->table.base,s->table.n,nullptr,nullptr);
    else all_force<3,false><<<grid,128,shared>>>(s->pos,s->spos,s->sLJ,s->stype,s->dLjTab,s->ntS,vw,vs,vv,s->fW,s->fS,s->p,float(s->in.KE),s->dTab,s->table.base,s->table.n,nullptr,nullptr);
  }else if(s->ns==4)ww_force<4,false><<<gridOf(s->WW.nBlocks,4),128>>>(s->pos,vw,s->fW,s->p,s->dTab,s->table.base,s->table.n,nullptr,nullptr);
  else ww_force<3,false><<<gridOf(s->WW.nBlocks,4),128>>>(s->pos,vw,s->fW,s->p,s->dTab,s->table.base,s->table.n,nullptr,nullptr);
  s2w_unpack<<<gridOf(9LL*s->npad,128),128>>>(s->fW,s->order,s->npad,s->geo,s->fatom);
  if(s->nS)s2s_unpack<<<gridOf(3LL*s->nsPad,128),128>>>(s->fS,s->sorder,s->nsPad,s->fatom);
  CK(cudaGetLastError());CK(cudaDeviceSynchronize());++s->evaluations;s->packedReady=false;return s->fatom;
}
const float *forceSession(Session *s,const double *x){prepareSession(s,x);return forcePreparedSession(s);}
SessionPhysicalData sessionPhysicalData(const Session*s) {
 if(!s->innerReady||!s->packedReady)throw std::runtime_error("physical coverage requires prepared inner list");
 SessionPhysicalData d{};d.waters=s->nW;d.atoms=s->nS;d.sites=s->ns;d.types=s->ntS;
 for(int k=0;k<3;++k)d.cell[k]=s->in.L[k];
 d.cutoff=s->in.rc;d.dOE=s->in.dOE;for(int k=0;k<4;++k)d.qWater[k]=s->in.qw[k];
 d.aOO=s->in.AOO;d.bOO=s->in.BOO;d.charges=s->in.q;d.aOS=s->in.AOS;d.bOS=s->in.BOS;
 d.aSS=s->in.LA;d.bSS=s->in.LB;d.atomTypes=s->in.type;d.exRows=s->dExOff;d.exNeighbors=s->dExPart;return d;
}

}

#pragma once
// Parameter-only provenance audit. No force, energy, search, or reach changes.
#include <cstdint>
#include <fstream>
#include <map>
#include <limits>
static void checkPrototypeParameters(const model::PreparedModel &p) {
  const char *path=std::getenv("MDIR_PROTOTYPE_SIDECAR");
  require(path,"prototype parameter sidecar is required");
  std::ifstream in(path,std::ios::binary);
  int64_t h[8];double d[17];in.read(reinterpret_cast<char*>(h),sizeof h);
  in.read(reinterpret_cast<char*>(d),sizeof d);
  require(bool(in),"prototype sidecar header is incomplete");
  const auto &t=*p.system.topology;
  int64_t n=p.system.getNumParticles(),w=t.settles.size(),sites=t.virtualSites.empty()?3:4,a=n-sites*w;
  require(h[0]==n&&h[1]==w&&h[2]==sites&&h[3]==a,"prototype sidecar/model dimensions differ");
  require(a>=0&&h[4]>=0&&h[4]<=32,"prototype solute type domain differs");
  size_t checked=0, roundingBoundaryDifferences=0;
  double maxReconstructionRelativeError=0;
  auto same=[&](double actual,double archived){
    if(float(actual)!=float(archived)) {
      double scale=std::max(std::abs(actual),std::abs(archived));
      double error=std::abs(actual-archived);
      // Reconstructing LJ coefficients through unit conversion can cross an
      // exact f32 midpoint by f64 rounding. Inputs remain unchanged.
      if(std::isfinite(actual)&&std::isfinite(archived)&&
         error<=128*std::numeric_limits<double>::epsilon()*scale) {
        ++roundingBoundaryDifferences;
        maxReconstructionRelativeError=std::max(maxReconstructionRelativeError,error/scale);
        ++checked;
        return;
      }
      std::cerr<<std::setprecision(17)<<"parameter provenance mismatch "<<checked
          <<" native "<<actual<<" archived "<<archived<<std::endl;
      require(false,"prototype sidecar f32 parameter differs from native model");
    }
    ++checked;
  };
  for(unsigned k=0;k<3;++k)same(10*p.system.box[k],d[k]);
  require(w>0,"prototype requires water");
  int oxygen=t.settles[0].oxygen;
  require(oxygen==a,"prototype file order must be solutes then OHH[EP] waters");
  for(int64_t j=0;j<w;++j) {
    require(t.settles[j].oxygen==a+sites*j,"prototype water file order differs");
    for(int k=0;k<sites;++k)same(t.charges[a+sites*j+k],d[3+k]);
    same(10*t.settles[j].distanceOH,d[12]);
  }
  for(const auto &site:t.virtualSites) {
    require(site.kind==driver::Topology::VirtualSite::AmberWater,"prototype needs Amber water sites");
    same(10*site.a,d[16]);
  }
  auto read=[&](auto *data,size_t count){in.read(reinterpret_cast<char*>(data),count*sizeof(*data));require(bool(in),"prototype sidecar array is incomplete");};
  std::vector<double>q(a),ao(a),bo(a);std::vector<int32_t>types(a);
  read(q.data(),a);read(types.data(),a);read(ao.data(),a);read(bo.data(),a);
  std::vector<double>la(h[4]*h[4]),lb(la.size());read(la.data(),la.size());read(lb.data(),lb.size());
  size_t nt=t.getNumTypes();
  auto coefficients=[&](unsigned i,unsigned j) {
    size_t ij=t.types[i]*nt+t.types[j];double sigma=10*t.sigma[ij];
    double s2=sigma*sigma,s6=s2*s2*s2,b=4*(t.epsilon[ij]/4.184)*s6;
    return std::array<double,2>{b*s6,b};
  };
  auto oo=coefficients(oxygen,oxygen);same(oo[0],d[7]);same(oo[1],d[8]);
  std::map<int,unsigned>representative;
  for(int64_t i=0;i<a;++i) {
    same(t.charges[i],q[i]);require(types[i]>=0&&types[i]<h[4],"prototype compact type is invalid");
    representative.emplace(types[i],i);auto os=coefficients(i,oxygen);same(os[0],ao[i]);same(os[1],bo[i]);
  }
  require(representative.size()==size_t(h[4]),"prototype compact types do not cover native solutes");
  for(const auto &[i,atomI]:representative)for(const auto &[j,atomJ]:representative){
    auto ss=coefficients(atomI,atomJ);same(ss[0],la[i*h[4]+j]);same(ss[1],lb[i*h[4]+j]);
  }
  same(p.control.cutoffDistance,d[11]);
  same(10*physical::coulomb/4.184,d[10]);
  double beta=p.control.pmeAlpha;
  if(beta==0) {double lo=0,hi=100;for(unsigned i=0;i<200;++i){double mid=.5*(lo+hi);if(std::erfc(mid*p.control.cutoffDistance)>p.control.pmeAlphaTolerance)lo=mid;else hi=mid;}beta=.5*(lo+hi);}
  same(beta,d[9]);
  std::cout<<"parameter provenance: "<<checked<<" coefficients/geometry values checked; "
      <<roundingBoundaryDifferences<<" f32 midpoint differences from f64 reconstruction, max relative "
      <<std::setprecision(17)<<maxReconstructionRelativeError<<"; "
      <<w<<" waters, "<<a<<" solutes, "<<sites<<" sites"<<std::endl;
}

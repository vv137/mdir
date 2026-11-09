// Independent all-pairs f64 LJ/Coulomb and analytic bisector transpose.
// The projected-force comparison removes the span of the three squared
// distance constraint gradients of each water, in the Euclidean metric.
// It does not share the compiler's pair body, site emitter or list search.
#include "mdir/Compiler/Simulation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

using namespace mdir;
using V = std::array<double, 3>;
static void require(bool ok, const char *message) {
  if (!ok) { llvm::errs() << message << '\n'; std::exit(1); }
}
template <class T> static T take(llvm::Expected<T> result) {
  if (!result) { llvm::errs() << llvm::toString(result.takeError()) << '\n'; std::exit(1); }
  return std::move(*result);
}
static void succeed(llvm::Error error) {
  if (error) { llvm::errs() << llvm::toString(std::move(error)) << '\n'; std::exit(1); }
}
static V add(V a, V b) { for (unsigned k=0;k!=3;++k) a[k]+=b[k]; return a; }
static V sub(V a, V b) { for (unsigned k=0;k!=3;++k) a[k]-=b[k]; return a; }
static V scale(V a, double b) { for (double &v:a) v*=b; return a; }
static double dot(V a, V b) { double s=0; for(unsigned k=0;k!=3;++k) s+=a[k]*b[k]; return s; }
static V vectorAt(const std::vector<double> &a, size_t i) { return {a[3*i],a[3*i+1],a[3*i+2]}; }
static V image(V d, const double *box) {
  for(unsigned k=0;k!=3;++k) d[k]-=box[k]*std::nearbyint(d[k]/box[k]);
  return d;
}
struct Oracle {
  double energy=0, ljAbsolute=0, coulombAbsolute=0;
  double pairVirial=0, siteVirial=0, pairSiteVirialAbsolute=0;
  std::vector<double> forces;
  std::array<unsigned,3> classPairs{};
};
static Oracle oracle(const driver::System &system, const std::vector<double> &coordinates,
                     double cutoff) {
  const auto &t=*system.topology;
  size_t n=system.getNumParticles(), nt=t.getNumTypes();
  std::vector<V> x(n), f(n);
  std::vector<bool> water(n,false);
  for(const auto &s:t.settles) for(unsigned j=0;j!=3;++j) water[s.oxygen+j]=true;
  for(const auto &s:t.virtualSites) water[s.site]=true;
  for(size_t i=0;i!=n;++i) x[i]=vectorAt(coordinates,i);
  // Form the actual Amber bisector from the instantaneous OHH geometry.
  for(const auto &s:t.virtualSites) {
    require(s.kind==driver::Topology::VirtualSite::AmberWater,"oracle requires Amber water sites");
    V u=image(sub(x[s.j],x[s.i]),system.box), v=image(sub(x[s.k],x[s.i]),system.box);
    V direction=add(scale(u,1/std::sqrt(dot(u,u))),scale(v,1/std::sqrt(dot(v,v))));
    x[s.site]=add(x[s.i],scale(direction,s.a/std::sqrt(dot(direction,direction))));
  }
  std::set<std::pair<unsigned,unsigned>> excluded(t.exclusions.begin(),t.exclusions.end());
  Oracle result;
  const double coulomb=driver::units::coulomb*driver::units::energy*driver::units::length;
  // Enumerate images independently, rather than choosing the list's image.
  for(unsigned i=0;i!=n;++i) for(unsigned j=i+1;j!=n;++j) {
    if(excluded.count({i,j})) continue;
    for(int a=-1;a<=1;++a) for(int b=-1;b<=1;++b) for(int c=-1;c<=1;++c) {
      V d=sub(x[i],x[j]);
      d=sub(d,{a*system.box[0],b*system.box[1],c*system.box[2]});
      double r2=dot(d,d);
      if(r2>=cutoff*cutoff) continue;
      require(r2>0,"oracle encountered overlapping sites");
      ++result.classPairs[water[i]&&water[j]?0:water[i]||water[j]?1:2];
      double r=std::sqrt(r2), q=coulomb*t.charges[i]*t.charges[j];
      unsigned type=t.types[i]*nt+t.types[j];
      double sr2=t.sigma[type]*t.sigma[type]/r2;
      double sr6=sr2*sr2*sr2, sr12=sr6*sr6, epsilon=t.epsilon[type];
      result.energy+=q/r+4*epsilon*(sr12-sr6);
      result.ljAbsolute+=std::abs(4*epsilon*(sr12-sr6));
      result.coulombAbsolute+=std::abs(q/r);
      V pair=scale(d,q/(r2*r)+24*epsilon*(2*sr12-sr6)/r2);
      double pairTrace=dot(d,pair);
      result.pairVirial+=pairTrace;
      result.pairSiteVirialAbsolute+=std::abs(pairTrace);
      f[i]=add(f[i],pair); f[j]=sub(f[j],pair);
    }
  }
  for(const auto &s:t.virtualSites) {
    V u=image(sub(x[s.j],x[s.i]),system.box),v=image(sub(x[s.k],x[s.i]),system.box);
    double ru=std::sqrt(dot(u,u)),rv=std::sqrt(dot(v,v));
    V a=scale(u,1/ru),b=scale(v,1/rv),w=add(a,b);
    double rw=std::sqrt(dot(w,w)); V axis=scale(w,1/rw);
    V across=sub(f[s.site],scale(axis,dot(axis,f[s.site])));
    V fj=scale(sub(across,scale(a,dot(a,across))),s.a/(ru*rw));
    V fk=scale(sub(across,scale(b,dot(b,across))),s.a/(rv*rw));
    V fo=sub(sub(f[s.site],fj),fk);
    const unsigned atoms[3]={s.i,s.j,s.k};
    const V transferred[3]={fo,fj,fk};
    for(unsigned k=0;k!=3;++k) {
      double term=dot(image(sub(x[atoms[k]],x[s.site]),system.box),transferred[k]);
      result.siteVirial+=term;
      result.pairSiteVirialAbsolute+=std::abs(term);
    }
    f[s.i]=add(f[s.i],fo);
    f[s.j]=add(f[s.j],fj); f[s.k]=add(f[s.k],fk); f[s.site]={0,0,0};
  }
  for(V value:f) result.forces.insert(result.forces.end(),value.begin(),value.end());
  return result;
}
struct ConstraintVirial { double trace=0, absolute=0; };
// c=|x_j-x_i|^2/2 gives G a + |v_j-v_i|^2=0. Solve
// (G M^-1 G^T) lambda = -(G M^-1 F + b) for all three water
// constraints. The resulting G^T lambda includes the velocity curvature.
// Oxygen-relative force arms define the reported absolute contributions.
static ConstraintVirial constraintVirial(const driver::System &system,
    const std::vector<double> &x, const std::vector<double> &v,
    const std::vector<double> &f) {
  ConstraintVirial result;
  for(const auto &water:system.topology->settles) {
    unsigned o=water.oxygen;
    double g[3][9]={}, inverse[9]={}, augmented[3][4]={};
    const unsigned pairs[3][2]={{0,1},{0,2},{1,2}};
    for(unsigned k=0;k!=9;++k) {
      double mass=system.masses[o+k/3];
      require(mass>0,"constraint oracle requires positive atomic masses");
      inverse[k]=1/mass;
    }
    for(unsigned q=0;q!=3;++q) {
      unsigned i=o+pairs[q][0],j=o+pairs[q][1];
      V d=image(sub(vectorAt(x,j),vectorAt(x,i)),system.box);
      V dv=sub(vectorAt(v,j),vectorAt(v,i));
      augmented[q][3]=-dot(dv,dv);
      for(unsigned k=0;k!=3;++k) {
        g[q][3*pairs[q][0]+k]=-d[k];
        g[q][3*pairs[q][1]+k]=d[k];
      }
      for(unsigned k=0;k!=9;++k) augmented[q][3]-=g[q][k]*inverse[k]*f[3*o+k];
    }
    for(unsigned i=0;i!=3;++i) for(unsigned j=0;j!=3;++j)
      for(unsigned k=0;k!=9;++k) augmented[i][j]+=g[i][k]*inverse[k]*g[j][k];
    for(unsigned i=0;i!=3;++i) {
      unsigned pivot=i;
      for(unsigned j=i+1;j!=3;++j) if(std::abs(augmented[j][i])>std::abs(augmented[pivot][i])) pivot=j;
      for(unsigned k=0;k!=4;++k) std::swap(augmented[i][k],augmented[pivot][k]);
      double divisor=augmented[i][i];
      require(std::abs(divisor)>1e-15,"singular constraint virial system");
      for(unsigned k=i;k!=4;++k) augmented[i][k]/=divisor;
      for(unsigned j=0;j!=3;++j) if(j!=i) {
        double multiplier=augmented[j][i];
        for(unsigned k=i;k!=4;++k) augmented[j][k]-=multiplier*augmented[i][k];
      }
    }
    for(unsigned member=0;member!=3;++member) {
      V reaction{};
      for(unsigned k=0;k!=3;++k) for(unsigned q=0;q!=3;++q)
        reaction[k]+=g[q][3*member+k]*augmented[q][3];
      V arm=image(sub(vectorAt(x,o+member),vectorAt(x,o)),system.box);
      double term=dot(arm,reaction);
      result.trace+=term;result.absolute+=std::abs(term);
    }
  }
  return result;
}
static std::vector<double> project(const driver::System &system,
                                 const std::vector<double> &x, std::vector<double> f) {
  for(const auto &water:system.topology->settles) {
    unsigned o=water.oxygen;
    double g[3][9]={};
    const unsigned pairs[3][2]={{0,1},{0,2},{1,2}};
    for(unsigned q=0;q!=3;++q) {
      V d=image(sub(vectorAt(x,o+pairs[q][1]),vectorAt(x,o+pairs[q][0])),system.box);
      for(unsigned k=0;k!=3;++k) { g[q][3*pairs[q][0]+k]=-d[k]; g[q][3*pairs[q][1]+k]=d[k]; }
    }
    double augmented[3][4]={};
    for(unsigned i=0;i!=3;++i) {
      for(unsigned j=0;j!=3;++j) for(unsigned k=0;k!=9;++k) augmented[i][j]+=g[i][k]*g[j][k];
      for(unsigned k=0;k!=9;++k) augmented[i][3]+=g[i][k]*f[3*o+k];
    }
    for(unsigned i=0;i!=3;++i) {
      unsigned pivot=i;
      for(unsigned j=i+1;j!=3;++j) if(std::abs(augmented[j][i])>std::abs(augmented[pivot][i])) pivot=j;
      for(unsigned k=0;k!=4;++k) std::swap(augmented[i][k],augmented[pivot][k]);
      double divisor=augmented[i][i]; require(std::abs(divisor)>1e-15,"singular constraint projection");
      for(unsigned k=i;k!=4;++k) augmented[i][k]/=divisor;
      for(unsigned j=0;j!=3;++j) if(j!=i) {
        double multiplier=augmented[j][i];
        for(unsigned k=i;k!=4;++k) augmented[j][k]-=multiplier*augmented[i][k];
      }
    }
    for(unsigned k=0;k!=9;++k) for(unsigned q=0;q!=3;++q) f[3*o+k]-=g[q][k]*augmented[q][3];
  }
  return f;
}
struct Error { double maximum=0,rms=0; };
static Error error(const std::vector<double> &a,const std::vector<double> &b) {
  require(a.size()==b.size()&&!a.empty(),"force layout differs");
  Error e;
  for(size_t i=0;i!=a.size();++i) {
    require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite force in comparison");
    double d=a[i]-b[i];e.maximum=std::max(e.maximum,std::abs(d)); e.rms+=d*d;
  }
  e.rms=std::sqrt(e.rms/a.size());return e;
}
static uint64_t bits(double value) {
  uint64_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
int main(int argc,char **argv) {
  bool withSolutes=argc==3&&std::strcmp(argv[2],"TIP3P-SOLUTES")==0;
  bool virialCheck=argc==3&&std::strcmp(argv[2],"OPC-VIRIAL")==0;
  require(argc==2||(argc==3&&(std::strcmp(argv[2],"TIP3P")==0||withSolutes||virialCheck)),
          "expected control file and optional TIP3P, TIP3P-SOLUTES or OPC-VIRIAL");
  bool threeSite=argc==3&&!virialCheck;
  const char *name=withSolutes?"TIP3P+solutes":threeSite?"TIP3P":"OPC";
  llvm::InitializeNativeTarget();llvm::InitializeNativeTargetAsmPrinter();
  model::PreparedModel p;
  p.control=take(driver::readControl(argv[1]));
  p.control.truncation=driver::Truncation::None;
  p.control.topologyDispersion=driver::DispersionCorrection::None;
  p.control.topologyDispersionGiven=true;
  p.system=take(driver::readSystem(p.control));
  driver::resolveNeighborStructure(p.control,p.system);
  p.execution.target=p.control.target;p.execution.precision=p.control.precision;
  require(!p.control.pme&&!p.control.ljpme&&!p.control.reactionField,"oracle requires plain cutoff");
  require(p.system.tilt[0]==0&&p.system.tilt[1]==0&&p.system.tilt[2]==0,"oracle fixture requires an orthorhombic cell");
  require(p.system.topology->virtualSites.size()==(threeSite?0u:3u)&&
          p.system.getNumParticles()==(withSolutes?11u:threeSite?9u:12u)&&p.system.topology->settles.size()==3,
          "expected three waters of the requested model");
  auto native=take(compiler::Simulation::create(p,false));
  require(take(native->run(1,{},true))==1,"template run did not take one step");
  auto state=take(native->getState());
  require(bool(state.energies),"template step gave no energy");
  std::cout<<"frame position bits:";
  for(double value:state.positions) std::cout<<' '<<bits(value);
  std::cout<<std::endl<<"frame cell bits:";
  for(double value:state.box) std::cout<<' '<<bits(value);
  for(double value:state.tilt) std::cout<<' '<<bits(value);
  std::cout<<std::endl;
  if(virialCheck) {
    std::cout<<"frame velocity bits:";
    for(double value:state.velocities) std::cout<<' '<<bits(value);
    std::cout<<std::endl;
  }
  p.system.positions=state.positions;p.system.velocities=state.velocities;
  auto templateFrameModel=p;
  p.control.waterTemplate=driver::WaterTemplateChoice::Off;
  auto doubleModel=p;
  doubleModel.control.precision=driver::Precision::Double;
  doubleModel.execution.precision=driver::Precision::Double;
  doubleModel.control.deterministic=doubleModel.execution.deterministic=true;
  auto generalDouble=take(compiler::Simulation::create(doubleModel,false));
  succeed(generalDouble->evaluate());auto reference=take(generalDouble->getState());
  require(bool(reference.energies),"general double gave no energy");
  Oracle independent=oracle(p.system,state.positions,p.control.cutoffDistance*driver::units::length);
  std::cout<<"CPU oracle class pair counts: WW "<<independent.classPairs[0]
           <<"; SW "<<independent.classPairs[1]<<"; SS "<<independent.classPairs[2]<<std::endl;
  if(withSolutes) require(independent.classPairs[0]&&independent.classPairs[1]&&independent.classPairs[2],
                         "charged-solute fixture must exercise all three class pairs");
  std::cout<<std::setprecision(17)<<"CPU oracle pair sums: net "<<independent.energy
           <<"; LJ absolute "<<independent.ljAbsolute
           <<"; Coulomb absolute "<<independent.coulombAbsolute<<std::endl;
  std::cout<<"CPU oracle pair-sum bits: "<<bits(independent.energy)<<' '
           <<bits(independent.ljAbsolute)<<' '<<bits(independent.coulombAbsolute)<<std::endl;
  double largest=0;
  for(double f:reference.forces) largest=std::max(largest,std::abs(f));
  Error oracleError=error(reference.forces,independent.forces);
  double oracleEnergy=std::abs(reference.energies->potential-independent.energy);
  std::cout<<std::setprecision(17)<<"oracle: energy "<<independent.energy
           <<", general-double "<<reference.energies->potential<<", difference "<<oracleEnergy
           <<"; force max "<<oracleError.maximum<<", RMS "<<oracleError.rms<<", scale "<<largest<<std::endl;
  require(oracleError.maximum<=1e-10*std::max(1.,largest),"general double differs from independent force oracle");
  require(oracleEnergy<=1e-10*std::max(1.,std::abs(independent.energy)),"general double differs from independent energy oracle");
  auto generalMixed=take(compiler::Simulation::create(p,false));
  succeed(generalMixed->evaluate());auto mixed=take(generalMixed->getState());
  require(bool(mixed.energies),"general mixed gave no energy for instrumentation");
  std::cout<<"signed energies: template "<<state.energies->potential
           <<"; general-mixed "<<mixed.energies->potential
           <<"; general-double "<<reference.energies->potential<<std::endl;
  std::cout<<"signed energy bits: "<<bits(state.energies->potential)<<' '
           <<bits(mixed.energies->potential)<<' '<<bits(reference.energies->potential)<<std::endl;
  auto projectedReference=project(p.system,state.positions,reference.forces);
  Error baseline=error(project(p.system,state.positions,mixed.forces),projectedReference);
  Error candidate=error(project(p.system,state.positions,state.forces),projectedReference);
  double maxBound=std::max(1.5*baseline.maximum,1e-5*largest);
  double rmsBound=std::max(1.5*baseline.rms,1e-5*largest);
  double energyDifference=std::abs(state.energies->potential-reference.energies->potential);
  double absolutePairSum=independent.ljAbsolute+independent.coulombAbsolute;
  double energyBound=std::max(1e-5*std::abs(reference.energies->potential),1e-6*absolutePairSum);
  double energyRelative=energyDifference/std::abs(reference.energies->potential);
  std::cout<<"projected forces: general-mixed max "<<baseline.maximum<<", RMS "<<baseline.rms
           <<"; template max "<<candidate.maximum<<", RMS "<<candidate.rms
           <<"; bounds "<<maxBound<<", "<<rmsBound<<"; template energy relative "<<energyRelative<<std::endl;
  std::cout<<"energy absolute error "<<energyDifference<<"; cancellation-aware bound "<<energyBound
           <<"; CPU oracle absolute pair sum "<<absolutePairSum<<std::endl;
  require(candidate.maximum<=maxBound&&candidate.rms<=rmsBound,"template projected forces exceed mixed criterion");
  require(std::isfinite(energyDifference)&&std::isfinite(energyBound)&&energyDifference<=energyBound,
          "template energy exceeds cancellation-aware criterion");
  if(virialCheck) {
    // A fresh no-step entry gives the instantaneous constraint reaction.
    // The carried step's mean impulse virial is a different quantity.
    auto templateFrame=take(compiler::Simulation::create(templateFrameModel,false));
    succeed(templateFrame->evaluate());auto observed=take(templateFrame->getState());
    require(bool(observed.energies),"template no-step entry gave no virial");
    auto constraint=constraintVirial(p.system,state.positions,state.velocities,independent.forces);
    double oracleTotal=independent.pairVirial+independent.siteVirial+constraint.trace;
    double absoluteSum=independent.pairSiteVirialAbsolute+constraint.absolute;
    double wt=observed.energies->virial,wm=mixed.energies->virial,wd=reference.energies->virial;
    double oracleDifference=std::abs(wd-oracleTotal);
    double oracleBound=1e-10*std::max(1.,std::abs(oracleTotal));
    double mixedDifference=std::abs(wm-wd),difference=std::abs(wt-wd);
    double bound=std::max({1.5*mixedDifference,1e-5*std::abs(wd),1e-6*absoluteSum});
    std::cout<<"scalar total virials: template "<<wt<<"; general-mixed "<<wm<<"; general-double "<<wd<<std::endl;
    std::cout<<"scalar total virial bits: "<<bits(wt)<<' '<<bits(wm)<<' '<<bits(wd)<<std::endl;
    std::cout<<"oracle virial: pair "<<independent.pairVirial<<"; site "<<independent.siteVirial
             <<"; constraint "<<constraint.trace<<"; total "<<oracleTotal
             <<"; pair/site absolute "<<independent.pairSiteVirialAbsolute
             <<"; constraint absolute "<<constraint.absolute<<"; absolute sum "<<absoluteSum<<std::endl;
    std::cout<<"oracle virial bits: "<<bits(independent.pairVirial)<<' '<<bits(independent.siteVirial)
             <<' '<<bits(constraint.trace)<<' '<<bits(oracleTotal)<<' '
             <<bits(independent.pairSiteVirialAbsolute)<<' '<<bits(constraint.absolute)<<' '<<bits(absoluteSum)<<std::endl;
    std::cout<<"virial oracle difference "<<oracleDifference<<"; oracle bound "<<oracleBound
             <<"; mixed difference "<<mixedDifference<<"; template difference "<<difference<<"; bound "<<bound<<std::endl;
    require(std::isfinite(oracleDifference)&&oracleDifference<=oracleBound,"general double differs from total-virial oracle");
    require(std::isfinite(difference)&&std::isfinite(bound)&&difference<=bound,"template total virial exceeds approved criterion");
    std::cout<<"OPC physical total virial with constraint reaction passed"<<std::endl;
  }
  std::cout<<name<<" independent physical pair oracle and projected template comparison passed"<<std::endl;
}

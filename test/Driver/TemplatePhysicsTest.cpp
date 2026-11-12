#include "TemplatePhysicsOracle.h"

int main(int argc,char **argv) {
  std::string mode=argc==3?argv[2]:"OPC";
  bool withSolutes=mode=="TIP3P-SOLUTES"||mode=="TIP3P-SOLUTES-VIRIAL";
  bool virialCheck=mode=="OPC-VIRIAL"||mode=="TIP3P-VIRIAL"||mode=="TIP3P-SOLUTES-VIRIAL";
  require((argc==2||argc==3)&&(mode=="OPC"||mode=="TIP3P"||withSolutes||virialCheck),
          "expected control file and OPC, TIP3P or TIP3P-SOLUTES, optionally suffixed -VIRIAL");
  bool threeSite=mode.find("TIP3P")==0;
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
    // A fresh non-Brownian entry reports the D264 molecular virial.
    // Compare directly in that convention, retaining atomic diagnostics.
    auto templateFrame=take(compiler::Simulation::create(templateFrameModel,false));
    succeed(templateFrame->evaluate());auto observed=take(templateFrame->getState());
    require(bool(observed.energies),"template no-step entry gave no virial");
    auto constraint=constraintVirial(p.system,state.positions,state.velocities,independent.forces);
    auto moments=molecularMoments(p.system,state.positions,state.velocities,independent.forces);
    double atomicTotal=independent.pairVirial+independent.siteVirial+constraint.trace;
    double oracleTotal=independent.pairVirial+independent.siteVirial+moments.removal;
    double absoluteSum=independent.pairSiteVirialAbsolute+moments.removalAbsolute;
    double wt=observed.energies->virial,wm=mixed.energies->virial,wd=reference.energies->virial;
    double oracleDifference=std::abs(wd-oracleTotal);
    double oracleBound=1e-10*std::max(1.,std::abs(oracleTotal));
    double mixedDifference=std::abs(wm-wd),difference=std::abs(wt-wd);
    double bound=std::max({1.5*mixedDifference,1e-5*std::abs(wd),1e-6*absoluteSum});
    std::cout<<"scalar molecular virials: template "<<wt<<"; general-mixed "<<wm<<"; general-double "<<wd<<std::endl;
    std::cout<<"scalar molecular virial bits: "<<bits(wt)<<' '<<bits(wm)<<' '<<bits(wd)<<std::endl;
    std::cout<<"oracle virial: pair "<<independent.pairVirial<<"; site "<<independent.siteVirial
             <<"; internal-force removal "<<moments.removal<<"; molecular total "<<oracleTotal
             <<"; pair/site absolute "<<independent.pairSiteVirialAbsolute
             <<"; removal absolute "<<moments.removalAbsolute<<"; molecular absolute sum "<<absoluteSum<<std::endl;
    std::cout<<"oracle virial bits: "<<bits(independent.pairVirial)<<' '<<bits(independent.siteVirial)
             <<' '<<bits(moments.removal)<<' '<<bits(oracleTotal)<<' '
             <<bits(independent.pairSiteVirialAbsolute)<<' '<<bits(moments.removalAbsolute)<<' '<<bits(absoluteSum)<<std::endl;
    std::cout<<"atomic diagnostics: total "<<atomicTotal<<"; full constraint reaction "<<constraint.trace
             <<"; constraint absolute "<<constraint.absolute<<"; twice internal kinetic "<<moments.twiceInternalKinetic<<std::endl;
    std::cout<<"atomic diagnostic bits: "<<bits(atomicTotal)<<' '<<bits(constraint.trace)<<' '
             <<bits(constraint.absolute)<<' '<<bits(moments.twiceInternalKinetic)<<std::endl;
    std::cout<<"virial oracle difference "<<oracleDifference<<"; oracle bound "<<oracleBound
             <<"; mixed difference "<<mixedDifference<<"; template difference "<<difference<<"; bound "<<bound<<std::endl;
    require(std::isfinite(oracleDifference)&&oracleDifference<=oracleBound,"general double differs from molecular-virial oracle");
    require(std::isfinite(difference)&&std::isfinite(bound)&&difference<=bound,"template molecular virial exceeds approved criterion");
    std::cout<<name<<" physical molecular virial passed"<<std::endl;
  }
  std::cout<<name<<" independent physical pair oracle and projected template comparison passed"<<std::endl;
}

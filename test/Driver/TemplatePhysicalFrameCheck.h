#pragma once
// Shared test-only physical comparisons; keep the approved bounds unchanged.
#include "TemplatePhysicsOracle.h"

static void installFrame(model::PreparedModel &p, const compiler::SimulationState &s) {
  p.system.positions=s.positions;p.system.velocities=s.velocities;
  std::copy(s.box.begin(),s.box.end(),p.system.box);
  std::copy(s.tilt.begin(),s.tilt.end(),p.system.tilt);
}
static model::PreparedModel route(model::PreparedModel p, unsigned choice) {
  p.control.waterTemplate=choice==0?driver::WaterTemplateChoice::On:driver::WaterTemplateChoice::Off;
  p.control.precision=p.execution.precision=choice==2?driver::Precision::Double:driver::Precision::Mixed;
  p.control.deterministic=p.execution.deterministic=choice==2;
  return p;
}
static void sameFrame(model::PreparedModel p,const compiler::SimulationState &frame) {
  installFrame(p,frame);
  // Potential, forces and molecular virial have no velocity-time offset.
  // Evaluate each route at the exact same x/v/cell with the VV no-step
  // convention used by the independent physical fixture, without coupling.
  p.control.integrator=driver::Integrator::VelocityVerlet;
  p.control.thermostat=p.control.barostat=false;
  std::array<compiler::SimulationState,3> s;
  for(unsigned r=0;r!=3;++r) {
    auto sim=take(compiler::Simulation::create(route(p,r),false));
    succeed(sim->evaluate());s[r]=take(sim->getState());
    require(bool(s[r].energies),"fresh frame evaluation has no energy");
  }
  auto ref=oracle(p.system,frame.positions,p.control.cutoffDistance*driver::units::length);
  auto moments=molecularMoments(p.system,frame.positions,frame.velocities,ref.forces);
  double largest=0;for(double f:s[2].forces) largest=std::max(largest,std::abs(f));
  auto oe=error(s[2].forces,ref.forces);
  double uo=std::abs(s[2].energies->potential-ref.energy);
  double wo=std::abs(s[2].energies->virial-(ref.pairVirial+ref.siteVirial+moments.removal));
  double wOracleBound=1e-10*std::max(1.,std::abs(ref.pairVirial+ref.siteVirial+moments.removal));
  double fOracleBound=1e-10*std::max(1.,largest);
  double uOracleBound=1e-10*std::max(1.,std::abs(ref.energy));
  auto record=[&](const std::string &name,double oracleValue,double generalValue,
                  double bound,double absolute,size_t terms) {
    std::cout<<"fresh "<<frame.step<<' '<<name
        <<" oracle "<<oracleValue<<" bits "<<bits(oracleValue)
        <<" general-double "<<generalValue<<" bits "<<bits(generalValue)
        <<" bound "<<bound<<" bits "<<bits(bound)
        <<" absolute-sum "<<absolute<<" bits "<<bits(absolute)
        <<" terms "<<terms<<std::endl;
  };
  record("energy",ref.energy,s[2].energies->potential,uOracleBound,
      ref.ljAbsolute+ref.coulombAbsolute,2*ref.pairTerms);
  record("molecular-virial",ref.pairVirial+ref.siteVirial+moments.removal,
      s[2].energies->virial,wOracleBound,
      ref.pairSiteVirialAbsolute+moments.removalAbsolute,
      ref.pairTerms+ref.siteTerms+moments.removalTerms);
  for(size_t i=0;i!=ref.forces.size();++i)
    record("force-"+std::to_string(i/3)+"-"+std::to_string(i%3),ref.forces[i],
        s[2].forces[i],fOracleBound,ref.forceAbsolute[i],ref.forceTerms[i]);
  auto projected=project(p.system,frame.positions,s[2].forces);
  auto baseline=error(project(p.system,frame.positions,s[1].forces),projected);
  auto candidate=error(project(p.system,frame.positions,s[0].forces),projected);
  double fb=std::max(1.5*baseline.maximum,1e-5*largest);
  double rb=std::max(1.5*baseline.rms,1e-5*largest);
  double ud=std::abs(s[0].energies->potential-s[2].energies->potential);
  double ub=std::max(1e-5*std::abs(s[2].energies->potential),1e-6*(ref.ljAbsolute+ref.coulombAbsolute));
  double wd=std::abs(s[0].energies->virial-s[2].energies->virial);
  double wb=std::max({1.5*std::abs(s[1].energies->virial-s[2].energies->virial),
      1e-5*std::abs(s[2].energies->virial),1e-6*(ref.pairSiteVirialAbsolute+moments.removalAbsolute)});
  std::cout<<"frame "<<frame.step<<" oracle force "<<oe.maximum<<" energy "<<uo
           <<" virial "<<wo<<"; projected max "<<candidate.maximum<<" / "<<fb
           <<" RMS "<<candidate.rms<<" / "<<rb<<"; energy "<<ud<<" / "<<ub
           <<"; molecular virial "<<wd<<" / "<<wb<<std::endl;
  require(oe.maximum<=fOracleBound&&uo<=uOracleBound&&wo<=wOracleBound,
          "trajectory frame differs from independent physical oracle");
  require(candidate.maximum<=fb&&candidate.rms<=rb&&ud<=ub&&wd<=wb,
          "trajectory frame exceeds unchanged physical bounds");
}

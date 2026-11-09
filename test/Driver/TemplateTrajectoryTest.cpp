// Short, identically initialized trajectories; same-frame pair accuracy
// remains a separate check with the approved physical-oracle envelopes.
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
static double residual(const driver::System &system, const compiler::SimulationState &s) {
  double worst=0;
  for(const auto &w:system.topology->settles) {
    const unsigned pairs[3][2]={{0,1},{0,2},{1,2}};
    for(unsigned k=0;k!=3;++k) {
      V d=image(sub(vectorAt(s.positions,w.oxygen+pairs[k][0]),
                    vectorAt(s.positions,w.oxygen+pairs[k][1])),s.box.data());
      double length=k==2?w.distanceHH:w.distanceOH;
      worst=std::max(worst,std::abs(dot(d,d)/(length*length)-1));
    }
  }
  return worst;
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
  require(oe.maximum<=1e-10*std::max(1.,largest)&&uo<=1e-10*std::max(1.,std::abs(ref.energy))&&wo<=wOracleBound,
          "trajectory frame differs from independent physical oracle");
  require(candidate.maximum<=fb&&candidate.rms<=rb&&ud<=ub&&wd<=wb,
          "trajectory frame exceeds unchanged physical bounds");
}
int main(int argc,char **argv) {
  require(argc==2,"expected short-trajectory control file");
  llvm::InitializeNativeTarget();llvm::InitializeNativeTargetAsmPrinter();
  model::PreparedModel p;
  p.control=take(driver::readControl(argv[1]));
  p.control.truncation=driver::Truncation::None;
  p.control.topologyDispersion=driver::DispersionCorrection::None;p.control.topologyDispersionGiven=true;
  p.system=take(driver::readSystem(p.control));driver::resolveNeighborStructure(p.control,p.system);
  p.execution.target=p.control.target;p.execution.precision=p.control.precision;
  require(p.control.timestep==.001&&!p.control.pme,"expected 1 fs plain-cutoff short fixture");
  auto prep=route(p,2);prep.control.integrator=driver::Integrator::VelocityVerlet;
  prep.control.thermostat=prep.control.barostat=false;
  auto initial=take(compiler::Simulation::create(prep,false));
  require(take(initial->run(1,{},true))==1,"initial constraint preparation failed");
  installFrame(p,take(initial->getState()));initial.reset();
  std::array<std::unique_ptr<compiler::Simulation>,3> sims;
  for(unsigned r=0;r!=3;++r) sims[r]=take(compiler::Simulation::create(route(p,r),false));
  std::cout<<std::setprecision(17);
  for(unsigned sample=0;sample!=10;++sample) {
    std::array<compiler::SimulationState,3> s;
    for(unsigned r=0;r!=3;++r) {
      require(take(sims[r]->run(10,{},true))==10,"short trajectory ended early");
      s[r]=take(sims[r]->getState());require(bool(s[r].energies),"trajectory has no energy");
      std::cout<<"route "<<r<<" step "<<s[r].step<<" offset "<<s[r].velocityOffset
          <<" U "<<s[r].energies->potential<<" K "<<s[r].energies->kinetic
          <<" conserved "<<s[r].energies->conserved<<" W "<<s[r].energies->virial
          <<" pressure "<<s[r].energies->pressure<<" volume "<<s[r].energies->volume
          <<" relative squared-bond residual "<<residual(p.system,s[r])<<std::endl;
      std::cout<<"position bits "<<r;for(double x:s[r].positions) std::cout<<' '<<bits(x);std::cout<<std::endl;
      std::cout<<"velocity bits "<<r;for(double v:s[r].velocities) std::cout<<' '<<bits(v);std::cout<<std::endl;
      std::cout<<"cell bits "<<r;for(double x:s[r].box) std::cout<<' '<<bits(x);std::cout<<std::endl;
    }
    std::array<Error,2> xe,ve;
    for(unsigned r=0;r!=2;++r) {
      auto x=s[r].positions;
      for(size_t i=0;i<x.size();i+=3) {
        V d=image(sub(vectorAt(x,i/3),vectorAt(s[2].positions,i/3)),s[2].box.data());
        for(unsigned k=0;k!=3;++k) x[i+k]=s[2].positions[i+k]+d[k];
      }
      xe[r]=error(x,s[2].positions);ve[r]=error(s[r].velocities,s[2].velocities);
    }
    double xs=*std::max_element(s[2].box.begin(),s[2].box.end()),vs=1;
    for(double v:s[2].velocities) vs=std::max(vs,std::abs(v));
    double xb=std::max(3*xe[1].maximum,1e-5*xs),xr=std::max(3*xe[1].rms,1e-5*xs);
    double vb=std::max(3*ve[1].maximum,1e-5*vs),vr=std::max(3*ve[1].rms,1e-5*vs);
    double vd=std::abs(s[0].energies->volume-s[2].energies->volume);
    double vbound=std::max(3*std::abs(s[1].energies->volume-s[2].energies->volume),1e-5*s[2].energies->volume);
    std::cout<<"trajectory "<<s[0].step<<" x max "<<xe[0].maximum<<" / "<<xb<<" RMS "<<xe[0].rms<<" / "<<xr
        <<"; v max "<<ve[0].maximum<<" / "<<vb<<" RMS "<<ve[0].rms<<" / "<<vr<<"; volume "<<vd<<" / "<<vbound<<std::endl;
    require(xe[0].maximum<=xb&&xe[0].rms<=xr&&ve[0].maximum<=vb&&ve[0].rms<=vr&&vd<=vbound,
            "short trajectory exceeds approved factor-three criterion");
    sameFrame(p,s[0]);
  }
  std::cout<<"short template trajectory and ten fresh physical frames passed"<<std::endl;
}

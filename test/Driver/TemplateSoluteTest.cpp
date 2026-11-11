// Same-frame forces of a solvated protein with heavy-atom solute units.
// A template trajectory under its own coupling supplies the frames; at
// each, the template, the general mixed and the general double routes are
// evaluated from the same positions and cell. With PME there is no
// independent pair oracle here: general double is the reference, general
// mixed the baseline, and the coverage diagnostic is a separate test.
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
int main(int argc,char **argv) {
  require(argc==3,"expected a control file and a number of frames");
  unsigned frames=unsigned(std::atoi(argv[2]));
  llvm::InitializeNativeTarget();llvm::InitializeNativeTargetAsmPrinter();
  model::PreparedModel p;
  p.control=take(driver::readControl(argv[1]));
  p.system=take(driver::readSystem(p.control));driver::resolveNeighborStructure(p.control,p.system);
  p.execution.target=p.control.target;p.execution.precision=p.control.precision;
  require(!p.system.topology->shakes.empty(),"expected constrained X-H bonds of a solute");
  std::array<compiler::CodeStore,3> code;
  auto trajectory=take(compiler::Simulation::create(route(p,0),true,&code[0]));
  unsigned stride=frames?unsigned(p.control.numSteps/frames):0;
  require(stride>0,"expected at least one step per frame");
  std::cout<<std::setprecision(17);
  for(unsigned frame=0;frame!=frames;++frame) {
    require(take(trajectory->run(stride,{},true))==stride,"template trajectory ended early");
    auto state=take(trajectory->getState());
    auto q=p;installFrame(q,state);
    q.control.integrator=driver::Integrator::VelocityVerlet;
    q.control.thermostat=q.control.barostat=false;
    std::array<compiler::SimulationState,3> s;
    for(unsigned r=0;r!=3;++r) {
      auto sim=take(compiler::Simulation::create(route(q,r),true,&code[r]));
      succeed(sim->evaluate());s[r]=take(sim->getState());
      require(bool(s[r].energies),"fresh frame evaluation has no energy");
    }
    double largest=0;for(double f:s[2].forces) largest=std::max(largest,std::abs(f));
    auto reference=project(q.system,state.positions,s[2].forces);
    auto baseline=error(project(q.system,state.positions,s[1].forces),reference);
    auto candidate=error(project(q.system,state.positions,s[0].forces),reference);
    // The solute's own atoms, unprojected: their forces are atom-wise.
    std::vector<bool> water(q.system.getNumParticles(),false);
    for(const auto &w:q.system.topology->settles)
      for(unsigned k=0;k!=4&&w.oxygen+k<water.size();++k)
        if(k<3||q.system.masses[w.oxygen+k]==0) water[w.oxygen+k]=true;
    Error soluteBase,soluteCandidate;size_t solute=0;
    for(size_t i=0;i!=water.size();++i) if(!water[i]) {
      for(unsigned k=0;k!=3;++k) {
        double b=std::abs(s[1].forces[3*i+k]-s[2].forces[3*i+k]);
        double c=std::abs(s[0].forces[3*i+k]-s[2].forces[3*i+k]);
        soluteBase.maximum=std::max(soluteBase.maximum,b);soluteBase.rms+=b*b;
        soluteCandidate.maximum=std::max(soluteCandidate.maximum,c);soluteCandidate.rms+=c*c;
      }
      ++solute;
    }
    soluteBase.rms=std::sqrt(soluteBase.rms/(3*solute));
    soluteCandidate.rms=std::sqrt(soluteCandidate.rms/(3*solute));
    double fb=std::max(1.5*baseline.maximum,1e-5*largest),rb=std::max(1.5*baseline.rms,1e-5*largest);
    double sb=std::max(1.5*soluteBase.maximum,1e-5*largest),srb=std::max(1.5*soluteBase.rms,1e-5*largest);
    double ud=std::abs(s[0].energies->potential-s[2].energies->potential);
    double ub=std::max(1.5*std::abs(s[1].energies->potential-s[2].energies->potential),
                       1e-6*std::abs(s[2].energies->potential));
    std::cout<<"frame "<<state.step<<" volume "<<state.energies->volume
        <<"; projected max "<<candidate.maximum<<" / "<<fb<<" RMS "<<candidate.rms<<" / "<<rb
        <<"; solute max "<<soluteCandidate.maximum<<" / "<<sb<<" RMS "<<soluteCandidate.rms<<" / "<<srb
        <<"; energy "<<ud<<" / "<<ub<<"; general-mixed max "<<baseline.maximum
        <<" energy "<<std::abs(s[1].energies->potential-s[2].energies->potential)<<std::endl;
    require(candidate.maximum<=fb&&candidate.rms<=rb,"template projected forces exceed the mixed criterion");
    require(soluteCandidate.maximum<=sb&&soluteCandidate.rms<=srb,"template solute forces exceed the mixed criterion");
    require(std::isfinite(ud)&&ud<=ub,"template energy exceeds the mixed criterion");
  }
  std::cout<<"solvated solute template frames passed"<<std::endl;
}

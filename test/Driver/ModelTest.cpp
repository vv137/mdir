// Native model parity, ownership, validation and MD-unit adaptation.
#include "mdir/Driver/Model.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdlib>
#include <limits>
using namespace mdir;
static void require(bool ok, const char *what) {
  if (!ok) { llvm::errs() << what << "\n"; std::exit(1); }
}
template<class T> static T take(llvm::Expected<T> result) {
  if (!result) { llvm::errs() << llvm::toString(result.takeError()) << "\n"; std::exit(1); }
  return std::move(*result);
}
static void reject(llvm::Expected<model::PreparedModel> result,
                   model::ModelError::Kind expected) {
  require(!result, "bad model accepted");
  bool matched = false;
  llvm::handleAllErrors(result.takeError(), [&](const model::ModelError &e) { matched = e.kind == expected; });
  require(matched, "wrong model error category");
}
int main(int argc, char **argv) {
  require(argc == 2, "expected a control file");
  auto c = take(driver::readControl(argv[1]));
  auto fileSystem = take(driver::readSystem(c));
  driver::Cell charmmCell;
  if (!c.charmmStructureFile.empty())
    charmmCell = take(driver::makeCell(c.box[0]*0.1,c.box[1]*0.1,c.box[2]*0.1,
                                      c.angles[0],c.angles[1],c.angles[2]));
  auto data = !c.charmmStructureFile.empty()
      ? take(model::loadCharmm(c.charmmStructureFile,c.charmmCoordinateFile,
                              c.charmmParameterFiles,charmmCell))
      : c.prmtopFile.empty()
          ? take(model::loadGromacs(c.gromacsTopologyFile,c.gromacsCoordinateFile,
                                    c.gromacsIncludes,c.gromacsDefines))
          : take(model::loadAmber(c.prmtopFile,c.amberCoordinateFile));
  auto s = data.makeSystem();
  auto state = data.makeState();
  s.cutoff = 0.8; s.pairlistDistance = 0.9; s.switchDistance = 0.8;
  s.truncation = driver::Truncation::None;
  s.rigidHydrogenBonds = c.rigidBonds;
  s.rigidWater = c.fastWater; s.flexibleWater = c.statesFlexible;
  s.electrostatics = c.pme ? model::Electrostatics::PME : model::Electrostatics::Cutoff;
  s.pmeGrid = {28,28,28};
  model::Integrator integrator;
  integrator.timestep = 0.0005;
  integrator.method = c.integrator;
  model::Ensemble ensemble;
  ensemble.temperature = 300;
  ensemble.kind = c.barostat ? model::EnsembleKind::NPT : c.thermostat ? model::EnsembleKind::NVT : model::EnsembleKind::NVE;
  ensemble.comPeriod = c.comPeriod;
  model::Execution execution;
  execution.target = c.target; execution.precision = c.precision;
  model::Schedule schedule;
  schedule.steps = 20; schedule.energyPeriod = 10;
  auto prepare = [&]() { return model::prepare(s,state,integrator,ensemble,execution,schedule); };
  auto prepared = take(prepare());
  auto fileProgram = take(driver::buildProgram(c,fileSystem));
  auto objectProgram = take(prepared.build());
  require(fileProgram.module == objectProgram.module, "file/object semantic IR differs");
  require(fileSystem.positions == prepared.system.positions &&
          fileSystem.velocities == prepared.system.velocities &&
          fileSystem.types == prepared.system.types &&
          fileSystem.masses == prepared.system.masses &&
          fileSystem.numConstraints == prepared.system.numConstraints,
          "file/object initial state or constraints differ");
  require(fileProgram.fields.size() == objectProgram.fields.size() &&
          fileProgram.tables.size() == objectProgram.tables.size() &&
          fileProgram.tupleSets.size() == objectProgram.tupleSets.size(), "metadata counts differ");
  for (size_t i=0; i<fileProgram.fields.size(); ++i)
    require(fileProgram.fields[i].name == objectProgram.fields[i].name &&
            fileProgram.fields[i].values == objectProgram.fields[i].values, "field values differ");
  for (size_t i=0; i<fileProgram.tables.size(); ++i)
    require(fileProgram.tables[i].name == objectProgram.tables[i].name &&
            fileProgram.tables[i].values == objectProgram.tables[i].values, "table values differ");
  for (size_t i=0; i<fileProgram.tupleSets.size(); ++i) {
    auto &a=fileProgram.tupleSets[i]; auto &b=objectProgram.tupleSets[i];
    require(a.members == b.members && a.fields.size() == b.fields.size(), "tuple identities differ");
    for (size_t j=0; j<a.fields.size(); ++j)
      require(a.fields[j].values == b.fields[j].values, "tuple parameters differ");
  }
  require(!data.sources.empty() && !data.sources.front().second.empty(), "missing owned provenance");
  auto stateCopy = state;
  state.positions[0] += 0.25;
  s.topology.masses[0] += 2;
  require(prepared.system.positions == stateCopy.positions, "prepared state aliases caller");
  require(data.topology.masses[0] == fileSystem.masses[0], "loaded physics aliases caller");
  state=stateCopy; s=data.makeSystem();
  s.cutoff=0.8; s.pairlistDistance=0.9; s.switchDistance=0.8; s.truncation=driver::Truncation::None;
  // Explicit velocities are not invented by loading or preparation.
  state.velocities.clear();
  auto noVelocity = take(prepare());
  require(!noVelocity.system.givenVelocities, "preparation invented supplied velocities");
  state = stateCopy;
  auto saved = s;
  auto input = model::ModelError::Input, unsupported = model::ModelError::Unsupported;
  state.positions.pop_back(); reject(prepare(),input); state=stateCopy;
  state.velocities={0}; reject(prepare(),input); state=stateCopy;
  state.positions[0]=std::numeric_limits<double>::quiet_NaN(); reject(prepare(),input); state=stateCopy;
  state.cell.diagonal[0]=0; reject(prepare(),input); state=stateCopy;
  s.topology.types[0]=s.topology.getNumTypes(); reject(prepare(),input); s=saved;
  s.topology.charges.pop_back(); reject(prepare(),input); s=saved;
  s.topology.masses[0]=-1; reject(prepare(),input); s=saved;
  s.topology.positions={0}; reject(prepare(),input); s=saved;
  s.topology.residueOf[0]=999999; reject(prepare(),input); s=saved;
  s.topology.sigma.pop_back(); reject(prepare(),input); s=saved;
  s.topology.bonds.push_back({0,unsigned(s.topology.getNumParticles()),1,0.1,false}); reject(prepare(),input); s=saved;
  s.pairlistDistance=0.5; reject(prepare(),input); s=saved;
  s.pmeGrid={4,4,4}; reject(prepare(),input); s=saved;
  auto precision=execution.precision;
  execution.precision=driver::Precision::Single; reject(prepare(),unsupported); execution.precision=precision;
  integrator.method=driver::Integrator::Brownian; reject(prepare(),unsupported); integrator.method=c.integrator;
  driver::PairTerm pair;
  pair.name="custom"; pair.expression="k*(r-r0)^2";
  pair.constants={{"k",100},{"r0",0.2}};
  s.pairTerms={pair};
  driver::TupleTerm bond;
  bond.name="spring"; bond.expression="k*(r-r0)^2"; bond.particles={0,1};
  bond.parameters={{"k",{100}},{"r0",{0.2}}};
  s.tupleTerms={bond};
  auto custom=take(prepare());
  take(custom.build());
  auto expression=take(driver::Expression::parse(custom.control.pairs[0].expression));
  llvm::StringMap<double> values; values["r"]=3; values["k"]=100; values["r0"]=0.2;
  double energy=expression.evaluate(values)*driver::units::energy;
  require(std::abs(energy-1)<1e-13,"custom pair MD energy/coordinate units differ");
  // Independent analytic derivative, E=k(r-r0)^2 at r=0.3 nm: 20 kJ/mol/nm.
  double h=1e-5; values["r"]=3+h;
  double plus=expression.evaluate(values)*driver::units::energy;
  values["r"]=3-h;
  double minus=expression.evaluate(values)*driver::units::energy;
  double derivative=(plus-minus)/(2*h*driver::units::length);
  require(std::abs(derivative-20)<1e-7,"custom force unit conversion differs");
  llvm::outs()<<"analytic energy: "<<energy<<" kJ/mol; derivative: "<<derivative<<" kJ/mol/nm\n";
  s.tupleTerms[0].parameters[0].second.clear(); reject(prepare(),input); s.tupleTerms={bond};
  s.tupleTerms[0].expression="sin("; reject(prepare(),input); s.tupleTerms={bond};
  s.pairTerms[0].expression="missing*r"; reject(prepare(),input); s.pairTerms={pair};
  s.pairTerms[0].constants.push_back({"sin",1}); reject(prepare(),input); s.pairTerms={pair};
  s.pairTerms[0].expression="k*u^2; u=r-r0";
  auto definitions=take(prepare());
  auto defined=take(driver::Expression::parse(definitions.control.pairs[0].expression));
  values["r"]=3;
  require(std::abs(defined.evaluate(values)*driver::units::energy-1)<1e-13,
          "MD coordinate conversion lost local definitions");
  take(definitions.build()); s.pairTerms={pair};
  llvm::outs()<<"energy difference: "<<driver::formatReal(energy-1)<<"; derivative difference: "<<driver::formatReal(derivative-20)<<"\n";
  s.pairTerms[0].mixing["k"]=driver::Mixing::Arithmetic; reject(prepare(),unsupported);
  llvm::outs()<<"file/object parity, ownership and typed validation passed\n";
}

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
  require(argc == 2 || (argc == 3 && (llvm::StringRef(argv[2]) == "--arrays" ||
                                       llvm::StringRef(argv[2]) == "--drawn" ||
                                       llvm::StringRef(argv[2]) == "--selections")),
          "expected a control file and optional --arrays, --drawn, or --selections");
  auto c = take(driver::readControl(argv[1]));
  auto fileSystem = take(driver::readSystem(c));
  if (argc == 3 && llvm::StringRef(argv[2]) == "--selections") {
    // The particles, from 0, that the masks of this control file select as
    // `mdir run` prepares them: the restrained ones (with a mass), those of
    // each term of [[energy.external]], and those that `couple` decouples
    // (D[python-topology], the oracle of Topology.select).
    auto line = [](llvm::StringRef what, auto chosen) {
      llvm::outs() << what << ":";
      for (size_t i : chosen) llvm::outs() << " " << i;
      llvm::outs() << "\n";
    };
    std::vector<size_t> restrained, coupled;
    for (size_t i = 0; i != fileSystem.restraintConstants.size(); ++i)
      if (fileSystem.restraintConstants[i] > 0.0) restrained.push_back(i);
    for (size_t i = 0; i != fileSystem.alchemical.size(); ++i)
      if (fileSystem.alchemical[i]) coupled.push_back(i);
    line("restrained", restrained);
    for (const auto &term : fileSystem.topology->externalTerms)
      line("external " + term.name,
           std::vector<size_t>(term.particles.begin(), term.particles.end()));
    line("couple", coupled);
    return 0;
  }
  if (argc == 3 && llvm::StringRef(argv[2]) == "--drawn") {
    // The velocities that `mdir run` draws for this control file: its
    // readSystem, then assignVelocities (tools/mdir/Run.cpp), as raw f64.
    driver::assignVelocities(c, fileSystem);
    llvm::outs().write(reinterpret_cast<const char *>(fileSystem.velocities.data()),
                       fileSystem.velocities.size() * sizeof(double));
    return 0;
  }
  // As `mdir run` takes the reference of restraints.
  fileSystem.referencePositions = fileSystem.positions;
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
  if (argc == 3) {
    // The D191 native values underlying D192's list interface, independent
    // of Python array shaping, dtype conversion and ownership.
    auto write = [](const std::vector<double> &values) {
      llvm::outs().write(reinterpret_cast<const char *>(values.data()),
                         values.size() * sizeof(double));
    };
    write(state.positions); write(state.velocities);
    for (const auto &row : state.cell.getVectors())
      llvm::outs().write(reinterpret_cast<const char *>(row.data()), sizeof(row));
    return 0;
  }
  s.cutoff = 0.8; s.pairlistDistance = 0.9; s.switchDistance = 0.8;
  s.truncation = driver::Truncation::None;
  s.rigidHydrogenBonds = c.rigidBonds;
  s.rigidWater = c.fastWater; s.flexibleWater = c.statesFlexible;
  s.electrostatics = c.pme ? model::Electrostatics::PME : model::Electrostatics::Cutoff;
  s.pmeGrid = {28,28,28};
  // [[restraints]] as typed restraints in kJ/mol/nm^2 (D198).
  // The control structure holds the constant of the file in kJ/mol/nm^2.
  for (const auto &r : c.restraints)
    s.restraints.push_back({r.selection, r.forceConstant, r.scaling});
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
  require(fileSystem.restraintConstants == prepared.system.restraintConstants &&
          fileSystem.restraintScaling == prepared.system.restraintScaling &&
          fileSystem.referencePositions == prepared.system.referencePositions,
          "file/object restraints differ");
  // The velocities of `mdir run` (readSystem, then assignVelocities), bit for bit.
  {
    auto fileDrawn = fileSystem;
    driver::assignVelocities(c, fileDrawn);
    auto drawn = take(model::drawVelocities(s, state, c.temperature, c.seed));
    require(drawn.velocities == fileDrawn.velocities && drawn.positions == state.positions,
            "drawn velocities differ from those of mdir run");
    require(llvm::any_of(drawn.velocities, [](double v) { return v != 0; }), "no velocities drawn");
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
  s.pairTerms[0].expression="r^2; r=r0"; reject(prepare(),input); s.pairTerms={pair};
  s.tupleTerms[0].expression="r^2; r=r0"; reject(prepare(),input); s.tupleTerms={bond};
  s.pairTerms[0].mixing["k"]=driver::Mixing::Arithmetic; reject(prepare(),unsupported);
  s=saved;
  // Typed restraints are refused as [[restraints]] is.
  s.restraints={{"!:WAT & !@H*", 4184.0, driver::ReferenceScaling::Center}};
  take(prepare());
  s.restraints[0].selection=""; reject(prepare(),input); s=saved;
  s.restraints={{"@1", 0.0, driver::ReferenceScaling::Center}}; reject(prepare(),input);
  s.restraints[0].forceConstant=std::numeric_limits<double>::infinity(); reject(prepare(),input);
  s.restraints={{"@1 &", 1.0, driver::ReferenceScaling::Center}}; reject(prepare(),input);
  s.restraints={{"@1", 1.0, driver::ReferenceScaling::Center},
                {"@1", 1.0, driver::ReferenceScaling::All}}; reject(prepare(),input);
  s.restraints={{"@1", 1.0, driver::ReferenceScaling::Center}};
  s.restraintReference={0.0}; reject(prepare(),input); s=saved;
  auto rejectDraw=[&](double temperature, uint64_t seed) {
    auto drawn=model::drawVelocities(s,state,temperature,seed);
    require(!drawn, "bad draw accepted"); llvm::consumeError(drawn.takeError());
  };
  rejectDraw(-1,1); rejectDraw(std::numeric_limits<double>::quiet_NaN(),1);
  rejectDraw(300,uint64_t(1)<<63);
  // A constant written in kJ/mol/nm^2 is that of the restraint, exactly:
  // it is not converted.
  for (double k : {4184.0, 1000.0, 0.1}) {
    s.restraints={{"@1", k, driver::ReferenceScaling::Center}};
    auto exact=take(prepare());
    require(llvm::all_of(exact.system.restraintConstants, [k](double c) { return c==0 || c==k; }),
            "restraint constant changed by the unit conversion");
  }
  llvm::outs()<<"file/object parity, ownership and typed validation passed\n";
}

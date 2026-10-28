// Shared parameter assembly, kept arrays, constraints, and start values.
// Term-specific collections and constants are owned by the adapters.

#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

llvm::Error Builder::collectParameters() {
  for (const TermBuilder *builder : getTermBuilders())
    if (llvm::Error error = builder->collectBareParameters(*this))
      return error;

  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().scalarParameters)
      if (llvm::Error error = role->collectBareConstants(*this))
        return error;
  return llvm::Error::success();
}

llvm::Error Builder::adoptArrays() {
  program.fields = kept->fields;
  program.tables = kept->tables;
  program.tupleSets = kept->tupleSets;
  program.gradientFields = kept->gradientFields;
  tupleGradients = kept->tupleGradients;
  gradientCharges = kept->gradientCharges;
  // The arrays that the other build left empty hold the values of the
  // tunables: each is made here by the function that makes it in a build
  // from the topology.
  const Topology &topology = *system.topology;
  size_t count = topology.getNumParticles();
  for (const TermBuilder *builder : getTermBuilders())
    builder->refillArrays(*this);
  // Nothing is left without its numbers: an array that a value of a
  // tunable enters and that nothing above makes would be one, and the
  // build fails here.
  auto lacks = [&](const llvm::Twine &what) {
    return makeError("the array '" + what + "' of the program has no "
                     "numbers after the values of the tunables were put in");
  };
  for (const Program::Field &field : program.fields)
    if (field.values.size() != count)
      return lacks(field.name);
  for (const Program::Table &table : program.tables)
    if (table.values.size() != size_t(table.count) * table.getColumns())
      return lacks(table.name);
  for (const Program::TupleSet &set : program.tupleSets) {
    if (set.members.size() % set.arity != 0)
      return lacks(set.name);
    for (const Program::Field &field : set.fields)
      if (field.values.size() != set.size())
        return lacks(set.name + "." + field.name);
  }
  return llvm::Error::success();
}

llvm::Error Builder::collectTopology() {
  HostMark mark("rebuild.build.topology.arrays");
  if (kept) {
    if (llvm::Error error = adoptArrays())
      return error;
  } else {
    collectTopologyArrays();
  }
  if (capture) {
    capture->fields = program.fields;
    capture->tables = program.tables;
    capture->tupleSets = program.tupleSets;
    capture->gradientFields = program.gradientFields;
    capture->tupleGradients = tupleGradients;
    capture->gradientCharges = gradientCharges;
    return llvm::Error::success();
  }
  mark.next("rebuild.build.topology.constants");
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().nonbonded)
      if (llvm::Error error =
              role->collect(*this, getNonbondedData(), getTermBuilders()))
        return error;
  return llvm::Error::success();
}

void Builder::collectTopologyArrays() {
  const Topology &topology = *system.topology;
  size_t count = topology.getNumParticles();

  // The fields of the particles.
  HostMark mark("rebuild.build.topology.fields");
  for (const TermBuilder *builder : termBuildersByParameterRank())
    builder->collectParameters(*this);

  for (const TermBuilder *builder : termBuildersByParameterRank())
    builder->collectTopologyTables(*this);

  auto addSet = [&](StringRef name, unsigned arity) -> Program::TupleSet & {
    Program::TupleSet set;
    set.name = name.str();
    set.arity = arity;
    program.tupleSets.push_back(std::move(set));
    return program.tupleSets.back();
  };
  auto addField = [&](Program::TupleSet &set, StringRef name) {
    Program::Field field;
    field.name = name.str();
    set.fields.push_back(std::move(field));
    return set.fields.size() - 1;
  };

  mark.next("rebuild.build.topology.tuples");
  for (const TermBuilder *builder : getTermBuilders())
    if (builder->getTupleRank() < 0)
      builder->collectTupleSets(*this);
  for (const TermBuilder *builder : getTermBuilders())
    if (builder->getTupleRank() == 0)
      builder->collectTupleSets(*this);
  // The field of zeros that `@tunable` takes for each of those fields.
  if (!program.gradientFields.empty()) {
    Program::Field zero;
    zero.name = "tg_zero";
    zero.values.assign(topology.getNumParticles(), 0.0);
    program.fields.push_back(std::move(zero));
  }
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().nonbonded)
      role->collectGradientFields(*this, getNonbondedData());
  for (const TermBuilder *builder : getTermBuilders())
    if (builder->getTupleRank() == 1)
      builder->collectTupleSets(*this);
  // The virtual sites: the site, then the atoms that it is built from.
  for (auto [kind, name] :
       {std::pair{Topology::VirtualSite::AmberWater, "sites_amber"},
        std::pair{Topology::VirtualSite::Linear, "sites_linear"}}) {
    if (llvm::none_of(topology.virtualSites,
                      [&](const Topology::VirtualSite &site) {
                        return site.kind == kind;
                      }))
      continue;
    Program::TupleSet &set = addSet(name, 4);
    set.reversible = false;
    size_t a = addField(set, "a"), b = addField(set, "b");
    for (const Topology::VirtualSite &site : topology.virtualSites) {
      if (site.kind != kind)
        continue;
      for (unsigned member : {site.site, site.i, site.j, site.k})
        set.members.push_back(member);
      set.fields[a].values.push_back(site.a);
      set.fields[b].values.push_back(site.b);
    }
  }
  if (!topology.settles.empty()) {
    // The oxygen and the two hydrogens of each rigid water.
    Program::TupleSet &set = addSet("settles", 3);
    set.reversible = false;
    size_t oh = addField(set, "doh"), hh = addField(set, "dhh");
    for (const Topology::Settle &settle : topology.settles) {
      for (unsigned k = 0; k != 3; ++k)
        set.members.push_back(settle.oxygen + k);
      set.fields[oh].values.push_back(settle.distanceOH);
      set.fields[hh].values.push_back(settle.distanceHH);
    }
  }
  // The groups of SHAKE, the heavy atom first, in a set for each number of
  // hydrogens.
  for (unsigned count = 1; count <= 3; ++count) {
    if (llvm::none_of(topology.shakes, [&](const Topology::Shake &shake) {
          return shake.hydrogens.size() == count;
        }))
      continue;
    Program::TupleSet &set = addSet("shake" + std::to_string(count),
                                    count + 1);
    set.reversible = false;
    std::vector<size_t> lengths;
    for (unsigned k = 1; k <= count; ++k)
      lengths.push_back(addField(set, "d" + std::to_string(k)));
    for (const Topology::Shake &shake : topology.shakes) {
      if (shake.hydrogens.size() != count)
        continue;
      set.members.push_back(shake.center);
      for (unsigned k = 0; k != count; ++k) {
        set.members.push_back(shake.hydrogens[k]);
        set.fields[lengths[k]].values.push_back(shake.lengths[k]);
      }
    }
  }
  // The anchor of each particle: the oxygen of its rigid water, the heavy
  // atom of its SHAKE group, that of the first atom that places a virtual
  // site, or itself. The order of the particles puts a particle where its
  // anchor is (emitReorder), so that the members of a group stay together,
  // in one warp of a device for most (D110). A virtual site ordered by its
  // own position fell into another cell than its atoms where they were near
  // a face of one: an extra point of OPC in 25 waters of 1000.
  if (!topology.settles.empty() || !topology.shakes.empty() ||
      !topology.virtualSites.empty()) {
    std::vector<int32_t> anchors(count);
    for (size_t i = 0; i != count; ++i)
      anchors[i] = i;
    for (const Topology::Settle &settle : topology.settles)
      for (unsigned k = 0; k != 3; ++k)
        anchors[settle.oxygen + k] = settle.oxygen;
    for (const Topology::Shake &shake : topology.shakes)
      for (int32_t hydrogen : shake.hydrogens)
        anchors[hydrogen] = shake.center;
    for (const Topology::VirtualSite &site : topology.virtualSites)
      anchors[site.site] = anchors[site.i];
    Program::TupleSet &set = addSet("anchors", 2);
    set.reversible = false;
    set.oriented = true;
    for (size_t i = 0; i != count; ++i) {
      set.members.push_back(i);
      set.members.push_back(anchors[i]);
    }
  }
  for (const TermBuilder *builder : getTermBuilders())
    if (builder->getTupleRank() == 2)
      builder->collectTupleSets(*this);
  for (const TermBuilder *builder : getTermBuilders())
    if (builder->getTupleRank() == 3)
      builder->collectTupleSets(*this);
}

llvm::Error Builder::collectTunableGradient() {
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().nonbonded)
      if (llvm::Error error = role->collectGradients(*this, getNonbondedData(),
                                                     getTermBuilders()))
        return error;
  return llvm::Error::success();
}

void Builder::collectStartValues() {
  // Every value of the state at the start that the text would otherwise
  // hold as a constant, under the conditions where emitEntry uses it, in
  // the order of Builder.h (D227). Each is computed
  // as the constant was.
  auto add = [&](std::string name, double value) {
    program.startValues.push_back({std::move(name), value});
  };
  if (isTriclinic()) {
    add("tilt_bx", system.tilt[0]);
    add("tilt_cx", system.tilt[1]);
    add("tilt_cy", system.tilt[2]);
  }
  if (scalesReference())
    for (int k = 0; k != 3; ++k)
      add("rest_edge" + std::to_string(k),
          system.inputBox[k] > 0.0 ? system.inputBox[k] : system.box[k]);
  if (control.minimize)
    return;
  if (control.getCouplingPeriod() > 0 && control.thermostat &&
      control.barostat) {
    // The virial of the corrections times the volume, which is
    // proportional to 1 / V, and their energies times the volume (the self
    // term of particle mesh Ewald does not depend on it), for the exact
    // work.
    double volume = system.box[0] * system.box[1] * system.box[2];
    add("baro_constant",
        (program.dispersionVirial + program.coulombConstantVirial) * volume);
    add("baro_energy_constant",
        (program.dispersionEnergy + program.coulombConstantEnergy -
         program.coulombSelfEnergy) *
            volume);
  }
  if (isRestart() && scalesEveryStep() && system.barostatState.size() == 9)
    for (int k = 0; k != 9; ++k)
      add("bstate" + std::to_string(k), system.barostatState[k]);
  // The observed constants that tunables take (D232).
  for (const ObservedTerm &term : getObservedTerms())
    for (const std::string &constant : term.constants)
      if (control.isObservedTunable(term.name, constant))
        add("obv_" + term.name + "_" + constant,
            getObservedValue(term, constant));
  // Whether the host asks for the derivative in the tunables, 0 unless a
  // simulation sets it for one evaluation, and the tunable constants of the
  // pair terms as the numbers that `@tunable` takes (D230).
  if (program.tunableGradient) {
    add("tunable_gradient", 0.0);
    for (const TermBuilder *builder : getTermBuilders())
      if (const auto *role = builder->getRoles().scalarParameters)
        role->appendInitialConstants(*this, add);
  }
}

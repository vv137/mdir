// The fields, tables, and constants of the program: the topology, the
// dispersion and its tails, PME, LJPME, the prolate splitting, the
// meshes, and the tunable gradients (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

llvm::Error Builder::collectParameters() {
  for (const TermBuilder *builder : registerTermBuilders())
    if (llvm::Error error = builder->collectBareParameters(*this))
      return error;

  for (const TermBuilder *builder : registerTermBuilders())
    if (llvm::Error error = builder->collectBareConstants(*this))
      return error;
  return llvm::Error::success();
}

void Builder::fillCharges(Program::Field &field) const {
  field.values = system.topology->charges;
}

void Builder::fillTypeTable(Program::Table &table) const {
  const Topology &topology = *system.topology;
  if (table.name == "lj_sigma") {
    table.values = topology.sigma;
  } else if (table.name == "lj_epsilon") {
    table.values = topology.epsilon;
  } else {
    // The weights of a seed of the derivatives in the tables (D230).
    unsigned seed = 0;
    StringRef(table.name).drop_front(3).getAsInteger(10, seed);
    table.values = system.tunableSeedWeights[seed];
  }
}


void Builder::fillPairProducts(Program::Field &field) const {
  // f q_i q_j s_C of each 1-4 pair.
  const Topology &topology = *system.topology;
  std::vector<double> products;
  products.reserve(topology.pairs.size());
  for (const Topology::Pair &pair : topology.pairs)
    products.push_back(coulombInternal * topology.charges[pair.i] *
                       topology.charges[pair.j] * pair.scaleCoulomb);
  field.values = std::move(products);
}

/// Whether `name` is the table of the weights of a seed, `tgw<c>`.
static bool isSeedTable(StringRef name) {
  unsigned seed;
  return name.consume_front("tgw") && !name.getAsInteger(10, seed);
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
  for (Program::Field &field : program.fields)
    if (field.values.empty() && field.name == "q")
      fillCharges(field);
  for (Program::Table &table : program.tables)
    if (table.values.empty()) {
      bool filled = llvm::any_of(registerTermBuilders(), [&](const TermBuilder *builder) {
        return builder->refillTable(*this, table);
      });
      if (!filled && (table.name == "lj_sigma" || table.name == "lj_epsilon" ||
                      isSeedTable(table.name)))
        fillTypeTable(table);
    }
  for (Program::TupleSet &set : program.tupleSets) {
    if (set.name == "pairs14")
      for (Program::Field &field : set.fields)
        if (field.values.empty() && field.name == "qq")
          fillPairProducts(field);
    for (const TermBuilder *builder : registerTermBuilders())
      builder->refillTupleSet(*this, set);
  }
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
  if (control.pme)
    if (llvm::Error error = collectPME())
      return error;
  if (control.ljpme)
    if (llvm::Error error = collectLJPME())
      return error;
  return collectTopologyConstants();
}

void Builder::collectTopologyArrays() {
  const Topology &topology = *system.topology;
  size_t count = topology.getNumParticles();

  // The fields of the particles.
  HostMark mark("rebuild.build.topology.fields");
  Program::Field charges;
  charges.name = "q";
  fillCharges(charges);
  program.fields.push_back(std::move(charges));
  Program::Field types;
  types.name = "type";
  types.isInteger = true;
  for (unsigned type : topology.types)
    types.values.push_back(type);
  program.fields.push_back(std::move(types));
  // The atomic numbers that a learned potential takes
  // (D270), in the order of the particles that the program
  // keeps, as the other fields; a model in stages takes them too
  // (D294).
  if (!control.models.empty() || !control.stagedModels.empty()) {
    Program::Field numbers;
    numbers.name = "model_z";
    numbers.isInteger = true;
    for (size_t i = 0; i != count; ++i)
      numbers.values.push_back(i < topology.atomicNumbers.size()
                                   ? topology.atomicNumbers[i]
                                   : 0);
    program.fields.push_back(std::move(numbers));
  }
  (void)count;

  for (const TermBuilder *builder : termBuildersByParameterRank())
    builder->collectParameters(*this);

  // [free_energy] (D161): 1 for each particle that it decouples.
  if (decouples()) {
    Program::Field flags;
    flags.name = "alch";
    for (bool flag : system.alchemical)
      flags.values.push_back(flag ? 1.0 : 0.0);
    program.fields.push_back(std::move(flags));
  }

  // Lennard-Jones for each pair of types.
  unsigned numTypes = topology.getNumTypes();
  for (const char *name : {"lj_sigma", "lj_epsilon"}) {
    Program::Table table;
    table.name = name;
    table.count = numTypes;
    fillTypeTable(table);
    program.tables.push_back(std::move(table));
  }
  // The weights of the seeds of the derivatives in those tables, a table
  // of all ordered pairs of types each, and the field of the particles
  // that each gives (D230).
  if (program.tunableGradient)
    for (auto [c, seed] : llvm::enumerate(control.tunableTableSeeds)) {
      Program::Table table;
      table.name = "tgw" + std::to_string(c);
      table.count = numTypes;
      table.columns = numTypes;
      fillTypeTable(table);
      program.tables.push_back(std::move(table));
      Program::GradientField field;
      field.tunable = seed.tunable;
      for (unsigned a = 0; a != numTypes; ++a)
        field.rows.push_back({seed.sites[a], seed.scales[a], 0.0});
      program.gradientFields.push_back(std::move(field));
    }
  for (const TermBuilder *builder : registerTermBuilders())
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
  if (!topology.bonds.empty()) {
    Program::TupleSet &set = addSet("bonds", 2);
    size_t k = addField(set, "k"), r0 = addField(set, "r0");
    for (const Topology::Bond &bond : topology.bonds) {
      set.members.push_back(bond.i);
      set.members.push_back(bond.j);
      set.fields[k].values.push_back(bond.k);
      set.fields[r0].values.push_back(bond.r0);
    }
  }
  if (!topology.angles.empty()) {
    Program::TupleSet &set = addSet("angles", 3);
    size_t k = addField(set, "k"), t0 = addField(set, "theta0");
    for (const Topology::Angle &angle : topology.angles) {
      for (unsigned member : {angle.i, angle.j, angle.k})
        set.members.push_back(member);
      set.fields[k].values.push_back(angle.force);
      set.fields[t0].values.push_back(angle.theta0);
    }
  }
  if (!topology.ureyBradleys.empty()) {
    Program::TupleSet &set = addSet("urey_bradley", 2);
    size_t k = addField(set, "k"), r0 = addField(set, "r0");
    for (const Topology::UreyBradley &term : topology.ureyBradleys) {
      set.members.push_back(term.i);
      set.members.push_back(term.k);
      set.fields[k].values.push_back(term.force);
      set.fields[r0].values.push_back(term.r0);
    }
  }
  if (!topology.dihedrals.empty()) {
    Program::TupleSet &set = addSet("dihedrals", 4);
    size_t k = addField(set, "k"), n = addField(set, "n"),
           phase = addField(set, "phase");
    for (const Topology::Dihedral &dihedral : topology.dihedrals) {
      for (unsigned member :
           {dihedral.i, dihedral.j, dihedral.k, dihedral.l})
        set.members.push_back(member);
      set.fields[k].values.push_back(dihedral.force);
      set.fields[n].values.push_back(dihedral.n);
      set.fields[phase].values.push_back(dihedral.phase);
    }
  }
  if (!topology.harmonicImpropers.empty()) {
    Program::TupleSet &set = addSet("impropers", 4);
    size_t k = addField(set, "k"), xi0 = addField(set, "xi0");
    for (const Topology::HarmonicImproper &term : topology.harmonicImpropers) {
      for (unsigned member : {term.i, term.j, term.k, term.l})
        set.members.push_back(member);
      set.fields[k].values.push_back(term.force);
      set.fields[xi0].values.push_back(term.xi0);
    }
  }
  for (const TermBuilder *builder : registerTermBuilders())
    builder->collectTupleSets(*this);
  // The field of zeros that `@tunable` takes for each of those fields.
  if (!program.gradientFields.empty()) {
    Program::Field zero;
    zero.name = "tg_zero";
    zero.values.assign(topology.getNumParticles(), 0.0);
    program.fields.push_back(std::move(zero));
  }
  // The derivative in the charges is that of `@tunable` in its field of
  // them, a site for each particle (D230).
  if (program.tunableGradient)
    for (auto [k, declaration] : llvm::enumerate(control.tunableDeclarations))
      if (declaration.kind == Control::TunableKind::Charge) {
        Program::GradientField field;
        field.tunable = static_cast<unsigned>(k);
        for (uint32_t i = 0, e = topology.getNumParticles(); i != e; ++i)
          field.sites.push_back({i, i});
        program.gradientFields.push_back(std::move(field));
        gradientCharges = true;
      }
  if (!topology.pairs.empty()) {
    // The factors enter the parameters: ε s_LJ, and f q_i q_j s_C.
    Program::TupleSet &set = addSet("pairs14", 2);
    size_t sigma = addField(set, "sigma"), epsilon = addField(set, "epsilon"),
           qq = addField(set, "qq");
    // In `@tunable` the pairs take f s_C and the charges of their members,
    // so that the derivative in the charges covers them.
    if (gradientCharges) {
      size_t scale = addField(set, "tgs");
      for (const Topology::Pair &pair : topology.pairs)
        set.fields[scale].values.push_back(coulombInternal * pair.scaleCoulomb);
    }
    for (const Topology::Pair &pair : topology.pairs) {
      set.members.push_back(pair.i);
      set.members.push_back(pair.j);
      set.fields[sigma].values.push_back(pair.sigma);
      set.fields[epsilon].values.push_back(pair.epsilon * pair.scaleLJ);
    }
    fillPairProducts(set.fields[qq]);
  }
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
  if (!topology.cmaps.empty()) {
    // The map of each term, and the bicubic patches of every map: a row
    // of 16 coefficients for each cell.
    Program::TupleSet &set = addSet("cmap", 5);
    set.reversible = false;
    size_t map = addField(set, "map");
    for (const Topology::CMap &cmap : topology.cmaps) {
      for (unsigned member : {cmap.i, cmap.j, cmap.k, cmap.l, cmap.m})
        set.members.push_back(member);
      set.fields[map].values.push_back(cmap.map);
    }
    Program::Table table;
    table.name = "cmap";
    table.values = getCMapCoefficients(topology);
    table.columns = 16;
    table.count = table.values.size() / 16;
    program.tables.push_back(std::move(table));
  }
  if (!topology.exclusions.empty()) {
    Program::TupleSet &set = addSet("excluded", 2);
    for (auto [i, j] : topology.exclusions) {
      set.members.push_back(i);
      set.members.push_back(j);
    }
  }
  // The pairs within the selection of [free_energy] that are not excluded,
  // whose Coulomb particle mesh Ewald adds back (D161).
  if (control.pme && !system.alchemicalPairs.empty()) {
    Program::TupleSet &set = addSet("alchemical_pairs", 2);
    for (auto [i, j] : system.alchemicalPairs) {
      set.members.push_back(i);
      set.members.push_back(j);
    }
  }

}

llvm::Error Builder::collectTopologyConstants() {
  const Topology &topology = *system.topology;
  // The reaction field (D140): its self term, −c f Σ q² / 2, which with
  // the terms of the excluded pairs makes the field act on every pair of
  // charges within the cutoff, those of one molecule as well: the
  // convention of GROMACS [GromacsManual2025], whose Coulomb (SR) MDIR's
  // matches; OpenMM's leaves both out.
  if (control.reactionField) {
    double squares = 0.0;
    for (double q : topology.charges)
      squares += q * q;
    double self =
        -0.5 * getReactionField(control).second * coulombInternal * squares;
    program.reactionField = true;
    program.coulombConstantEnergy = self;
    program.coulombSelfEnergy = self;
  }

  // The correction for the dispersion (Section 1.2 of nonbonded.md): N²
  // times the mean of C6 over the pairs of distinct particles that are not
  // excluded, as GROMACS takes it.
  if (control.topologyDispersion != DispersionCorrection::None) {
    // With [free_energy] the pairs that the selection decouples count
    // times 1 − λ: the tail of their soft-core Lennard-Jones is that of
    // the plain one times 1 − λ (D161).
    double energy = getTopologyDispersion(false);
    if (decouples()) {
      double lambda = getLambda("vdw");
      energy = (1.0 - lambda) * energy + lambda * getTopologyDispersion(true);
    }
    // The estimate of the shift is the tail times 1 − V / (N (4π/3) r_c³):
    // the tail again, which follows the volume, and a part that does not
    // (#224).
    double shift = energy;
    double shiftFixed = energy * getShiftFixedFactor();
    // Under "POTENTIAL_SHIFT" the energy that the run reports is shifted,
    // and the correction adds the estimate of the shift, which changes no
    // force and so no virial (D210).
    bool shifted = control.truncation == Truncation::Shift;
    program.dispersionEnergy = energy + (shifted ? shift : 0.0);
    program.dispersionFixedEnergy = shifted ? shiftFixed : 0.0;
    program.dispersionVirial = 6.0 * energy;
    for (const TermBuilder *builder : registerTermBuilders())
      if (llvm::Error error = builder->collectDispersionConstants(*this))
        return error;
    collectObservedTails();
  }
  if (control.hasFreeEnergy)
    collectFreeEnergyConstants();
  return llvm::Error::success();
}

double Builder::getTopologyDispersion(bool decoupled) const {
  // The correction and the estimate of its shift go with the
  // Lennard-Jones (D243).
  if (dependent && !dependent->lennardJones)
    return 0.0;
  const Topology &topology = *system.topology;
  unsigned numTypes = topology.getNumTypes();
  std::vector<double> numbers(numTypes, 0.0);
  for (unsigned type : topology.types)
    numbers[type] += 1.0;
  auto c6 = [&](unsigned a, unsigned b) {
    double sigma = topology.sigma[a * numTypes + b];
    return 4.0 * topology.epsilon[a * numTypes + b] * std::pow(sigma, 6);
  };
  double sum = 0.0;
  for (unsigned a = 0; a != numTypes; ++a)
    for (unsigned b = 0; b != numTypes; ++b)
      sum += numbers[a] * (numbers[b] - (a == b ? 1.0 : 0.0)) * c6(a, b);
  for (auto [i, j] : topology.exclusions)
    sum -= 2.0 * c6(topology.types[i], topology.types[j]);
  // Without the pairs of a particle of the selection and one of the rest,
  // over the same number of pairs.
  if (decoupled) {
    std::vector<double> inside(numTypes, 0.0), outside(numTypes, 0.0);
    for (size_t i = 0, e = topology.types.size(); i != e; ++i)
      (system.alchemical[i] ? inside : outside)[topology.types[i]] += 1.0;
    for (unsigned a = 0; a != numTypes; ++a)
      for (unsigned b = 0; b != numTypes; ++b)
        sum -= 2.0 * inside[a] * outside[b] * c6(a, b);
  }
  double n = static_cast<double>(topology.getNumParticles());
  double pairs = n * (n - 1.0) - 2.0 * topology.exclusions.size();
  double mean = pairs > 0.0 ? sum / pairs : 0.0;
  double rc = control.cutoffDistance * units::length;
  double volume = system.box[0] * system.box[1] * system.box[2];
  return -2.0 * M_PI / (3.0 * volume) * n * n * mean / (rc * rc * rc);
}

double Builder::getTopologyShift(bool decoupled) const {
  // A shift to 0 at the cutoff takes u(r_c) from each pair within it. A
  // particle has ρ (4π/3) r_c³ − 1 neighbors within r_c at the density
  // ρ = N / V, the particle itself taken out, as the compressibility sum
  // rule makes ∫ ρ (g − 1) dV = −1 for a liquid of low compressibility
  // [HansenMcDonald2013] (and as GROMACS counts them). In the convention of
  // the tail, N² / 2 pairs of the mean C6, the estimate is
  // (N/2) ⟨u(r_c)⟩ (ρ (4π/3) r_c³ − 1); of the r⁻⁶ part alone, as the
  // tail, u(r_c) = −⟨C6⟩ / r_c⁶, which is the tail E times
  // 1 − V / (N (4π/3) r_c³).
  return getTopologyDispersion(decoupled) * getShiftFactor();
}

double Builder::getShiftFactor() const {
  // 1 − 1 / (ρ (4π/3) r_c³): the neighbors within r_c less the particle
  // itself, over ρ (4π/3) r_c³ (D210).
  double rc = control.cutoffDistance * units::length;
  double volume = system.box[0] * system.box[1] * system.box[2];
  double n = static_cast<double>(system.topology->getNumParticles());
  return 1.0 - volume / (n * 4.0 * M_PI / 3.0 * rc * rc * rc);
}

void Builder::collectObservedTails() {
  // The columns of `observe` of a pair term take its tail and the estimate
  // of its shift: its energy and its derivatives in the observed constants,
  // proportional to 1 / V but the part of the estimate that does not depend
  // on the volume (D210, #224).
  program.observableVolumeConstants.assign(control.observables.size(), 0.0);
  program.observableFixedConstants.assign(control.observables.size(), 0.0);
  for (const TermBuilder *builder : registerTermBuilders())
    builder->collectObservedConstants(*this);
}

std::pair<double, double> Builder::getPMEConstants(double lambda) const {
  const Topology &topology = *system.topology;
  double squares = 0.0, net = 0.0;
  for (size_t i = 0, e = topology.charges.size(); i != e; ++i) {
    double q = topology.charges[i];
    if (decouples() && system.alchemical[i])
      q *= 1.0 - lambda;
    squares += q * q;
    net += q;
  }
  // −f s Σ q² and −f π Q² b / V; for the Gaussian s = β/√π and
  // b = 1/(2β²) (D266).
  double self = -coulombInternal * program.pmeSelfFactor * squares;
  double volume = system.box[0] * system.box[1] * system.box[2];
  double background = -coulombInternal * M_PI * net * net *
                      program.pmeBackgroundFactor / volume;
  return {self, background};
}

void Builder::collectFreeEnergyConstants() {
  // At each state, the constants that depend on λ: the self term of
  // particle mesh Ewald, which does not depend on the volume, and the
  // background of a net charge and the correction for the dispersion,
  // proportional to 1 / V, at the volume of the file. Their derivatives
  // follow at the state of the run: in λ of the Coulomb, 2 s s' times the
  // part of the selection in the self term, and in the background as its
  // net charge Q(λ) = Q_0 + s Q_1 says; in λ of the Lennard-Jones, the
  // difference of the two corrections (D161).
  const Control::FreeEnergy &energy = control.freeEnergy;
  size_t states = energy.getNumStates();
  // These quantities are of the shifted potential, so the correction adds
  // the estimate of the shift to the tail (D210).
  bool dispersion = control.topologyDispersion != DispersionCorrection::None;
  // The estimate is the tail times 1 − V / (N (4π/3) r_c³): the tail
  // again, proportional to 1 / V, and a part that does not depend on the
  // volume (#224).
  double coupled = 0.0, decoupled = 0.0;
  if (dispersion && decouples()) {
    coupled = getTopologyDispersion(false);
    decoupled = getTopologyDispersion(true);
  } else if (dispersion) {
    coupled = decoupled = getTopologyDispersion(false);
  }
  double coupledFixed = coupled * getShiftFixedFactor();
  double decoupledFixed = decoupled * getShiftFixedFactor();
  coupled *= 2.0;
  decoupled *= 2.0;
  for (size_t k = 0; k != states; ++k) {
    double fixed = 0.0, scaled = 0.0;
    if (program.pme && decouples()) {
      auto [self, background] = getPMEConstants(energy.get("coulomb", k));
      fixed += self;
      scaled += background;
    }
    double lambda = decouples() ? energy.get("vdw", k) : 0.0;
    scaled += (1.0 - lambda) * coupled + lambda * decoupled;
    fixed += (1.0 - lambda) * coupledFixed + lambda * decoupledFixed;
    if (dispersion) {
      scaled += getPairTails(getLambdaValues(k));
      fixed += getPairTails(getLambdaValues(k), /*fixed=*/true);
    }
    program.stateFixedEnergies.push_back(fixed);
    program.stateVolumeEnergies.push_back(scaled);
  }
  for (const auto &[name, values] : energy.lambdas) {
    double fixed = 0.0, scaled = 0.0;
    if (decouples() && name == "coulomb" && program.pme) {
      // The derivatives of quadratics in λ, from three points.
      double lambda = getLambda("coulomb");
      auto [self0, background0] = getPMEConstants(0.0);
      auto [self1, background1] = getPMEConstants(1.0);
      auto [selfh, backgroundh] = getPMEConstants(0.5);
      auto slope = [&](double at0, double ath, double at1) {
        // f(λ) = a + b λ + c λ², f'(λ) = b + 2 c λ.
        double c = 2.0 * (at0 + at1 - 2.0 * ath);
        double b = at1 - at0 - c;
        return b + 2.0 * c * lambda;
      };
      fixed = slope(self0, selfh, self1);
      scaled = slope(background0, backgroundh, background1);
    }
    if (decouples() && name == "vdw") {
      scaled = decoupled - coupled;
      fixed += decoupledFixed - coupledFixed;
    }
    for (unsigned index = 0, e = system.pairTails.size(); index != e;
         ++index) {
      scaled += getPairTailDerivative(index, "lambda_" + name, getLambda(name));
      fixed += getPairTailDerivative(index, "lambda_" + name, getLambda(name),
                                     /*shift=*/true, /*fixed=*/true);
    }
    program.lambdaFixedDerivatives.push_back(fixed);
    program.lambdaVolumeDerivatives.push_back(scaled);
  }
}

/// The aliases |j| ≤ P3MAliases of each index of an edge in the sums of
/// influence "P3M" (D304; docs/pme.md, Section 1.1). The
/// kernels of a device take it as a constant (tables_p3m of
/// scripts/generate-pme-gpu-template.py): change both together.
static constexpr int64_t P3MAliases = 2;

/// The smallest even number of points, at least `least`, whose prime
/// factors are 2, 3, 5, and 7, which the FFT handles fast. On an even grid
/// the influence function of sander agrees with MDIR's (docs/pme.md,
/// Section 1.1).
static int64_t getSmoothSize(int64_t least) {
  for (int64_t size = least;; ++size) {
    if (size % 2 != 0)
      continue;
    int64_t rest = size;
    for (int64_t factor : {2, 3, 5, 7})
      while (rest % factor == 0)
        rest /= factor;
    if (rest == 1)
      return size;
  }
}

llvm::Error Builder::collectPME() {
  const Topology &topology = *system.topology;
  double rc = control.cutoffDistance * units::length;

  if (control.pmeKernel == PMEKernel::Prolate) {
    if (llvm::Error error = collectProlate())
      return error;
  }

  // β: given, or such that erfc(β rc) is the tolerance, by bisection.
  double beta = control.pmeAlpha / units::length;
  if (program.pmeProlate) {
    beta = 0.0;
  } else if (beta == 0.0) {
    double low = 0.0, high = 1.0;
    while (std::erfc(high * rc) > control.pmeAlphaTolerance)
      high *= 2.0;
    for (int i = 0; i != 100; ++i) {
      double middle = 0.5 * (low + high);
      if (std::erfc(middle * rc) > control.pmeAlphaTolerance)
        low = middle;
      else
        high = middle;
    }
    beta = 0.5 * (low + high);
  }

  int64_t grid[3];
  if (program.pmeProlate) {
    for (int k = 0; k != 3; ++k)
      grid[k] = program.pmeGrid[k];
  } else {
    if (llvm::Error error = collectMesh("pme_moduli", control.pmeGrid,
                                        control.pmeMaxSpacing,
                                        control.pmeOrder, control.pmeInfluence,
                                        grid))
      return error;
    program.pmeP3M = control.pmeInfluence == PMEInfluence::P3M;
    program.pmeOrder = control.pmeOrder;
    program.pmeSelfFactor = beta / std::sqrt(M_PI);
    program.pmeBackgroundFactor = 1.0 / (2.0 * beta * beta);
  }

  // The grid holds charges in fixed point at the scale 2^40, up to about
  // 8e6 e at a point (D70); a charge beyond 100 e is not of a molecule.
  for (auto [index, q] : llvm::enumerate(topology.charges))
    if (!(std::fabs(q) < 100.0))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the charge of the atom %zu is %g e; particle mesh Ewald holds "
          "charges of less than 100 e",
          index + 1, q);

  // The self term, and the background that neutralizes a net charge,
  // whose virial is its energy on the diagonal; with the charges of the
  // selection of [free_energy] scaled as at the state of the run (D161).
  program.pmeBeta = beta;
  auto [self, background] = getPMEConstants(getLambda("coulomb"));
  program.pme = true;
  program.coulombConstantEnergy = self + background;
  program.coulombSelfEnergy = self;
  program.coulombConstantVirial = 3.0 * background;
  program.pmeBeta = beta;
  for (int k = 0; k != 3; ++k)
    program.pmeGrid[k] = grid[k];
  // The particles of a charge that can be nonzero: the test is on the
  // charge of the topology, which [free_energy] scales by 1 − a λ, so that
  // a charge of zero is zero at every λ (D161).
  collectListed("pme_listed", topology.charges, {Control::TunableKind::Charge});
  return llvm::Error::success();
}

llvm::Error Builder::collectProlate() {
  // Ewald summation with prolates [Liang2026]: the splitting
  // 1/r = (1 − Φ(r/r_c))/r + Φ(r/r_c)/r with Φ(y) = ∫₀ʸ χ, χ = ψ₀ᶜ/C₀, and
  // the spreading by the window φ₁(d) = ψ₀^{c₁}(2d/P) on P points.
  double rc = control.cutoffDistance * units::length;
  double epsilon = control.pmeAlphaTolerance;
  double c = control.pmeBandwidth > 0.0
                 ? control.pmeBandwidth
                 : prolate::bandwidthForTolerance(epsilon);
  // c₁ "slightly larger" than c [Liang2026]; c(ε/2) as the paper's code.
  double c1 = control.pmeWindowBandwidth > 0.0
                  ? control.pmeWindowBandwidth
                  : prolate::bandwidthForTolerance(0.5 * epsilon);
  int64_t points = control.pmeOrder;
  prolate::Function psi(c);
  prolate::Window window = prolate::getWindow(c1, points);
  prolate::Influence influence = prolate::getInfluence(psi);

  // The edges along which the grid is spaced, as collectMesh takes them.
  double edges[3];
  for (int k = 0; k != 3; ++k) {
    double edge = system.inputBox[k] > 0.0 ? system.inputBox[k] : system.box[k];
    if (isTriclinic())
      edge = k == 0 ? system.box[0]
             : k == 1 ? std::hypot(system.tilt[0], system.box[1])
                      : std::sqrt(system.tilt[1] * system.tilt[1] +
                                  system.tilt[2] * system.tilt[2] +
                                  system.box[2] * system.box[2]);
    edges[k] = edge;
  }
  // The grid: given; or from the largest spacing given; or the modes of
  // eq. 15 of [Liang2026], n_f = ⌈c L / (π r_c)⌉, raised until the
  // aliasing estimate of its eq. 10, relative to the sum it perturbs, is at
  // most ε/2 (D266, as ruled on issue #334). The factor 1/2 is
  // calibrated against the relative force error Δ of its eq. 16 over the
  // Coulomb forces of the topology (with its exclusions), measured on JAC and
  // Cellulose against the Gaussian at erfc 1e-13 on 0.33 Å
  // (scripts/benchmarks/prolate-ewald/grid_sweep.py): there the estimate is
  // 2.5 to 3 times below Δ, Δ of a fine grid is 0.6 to 0.9 ε, and ε/2 puts
  // every ε and P measured within 1.3 times that floor; ε alone left P = 5
  // and 7 at 2.7 to 2.9 times it.
  int64_t grid[3];
  for (int k = 0; k != 3; ++k) {
    grid[k] = control.pmeGrid[k];
    if (grid[k] == 0) {
      double spacing = control.pmeMaxSpacing > 0.0
                           ? control.pmeMaxSpacing * units::length
                           : M_PI * rc / c;
      grid[k] = getSmoothSize(
          static_cast<int64_t>(std::ceil(edges[k] / spacing - 1e-9)));
    }
  }
  bool automatic = control.pmeGrid[0] == 0 && control.pmeMaxSpacing == 0.0;
  double aliasing = prolate::getAliasing(psi, window, edges, grid, rc);
  double bound = 0.5 * epsilon;
  int64_t start[3] = {grid[0], grid[1], grid[2]};
  double spacing = 0.0;
  for (int k = 0; k != 3; ++k)
    spacing = std::max(spacing, edges[k] / static_cast<double>(grid[k]));
  while (automatic && aliasing > bound) {
    // Twice the grid of eq. 15 on every axis and still above: the estimate
    // is near the floor that the tail of the window sets, which a finer
    // grid lowers only slowly (at ε = 1e-9 with 8 points: 8.4e-8 at twice
    // the grid, 6.4e-10 at ten times, against 5e-10).
    if (grid[0] >= 2 * start[0] && grid[1] >= 2 * start[1] &&
        grid[2] >= 2 * start[2])
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the aliasing estimate of the prolate kernel with %lld points of "
          "the window stays above half the tolerance %g of [pme] on every "
          "grid: it is %g at %lld × %lld × %lld points, twice the grid of "
          "eq. 15, and falls only slowly beyond it, toward the floor that "
          "the tail of the window sets; %s",
          static_cast<long long>(points), epsilon, aliasing,
          static_cast<long long>(grid[0]), static_cast<long long>(grid[1]),
          static_cast<long long>(grid[2]),
          points < 8 ? "give more points of the window in 'order' (up to 8), "
                       "a looser 'tolerance', or 'grid'"
                     : "8 points are the most; give a looser 'tolerance', or "
                       "the grid in 'grid'");
    // One spacing for the three axes, made finer by the least that adds a
    // point to some axis, and by at least 2%, so that the grid stays in
    // proportion to the edges.
    double finer = 0.0;
    for (int k = 0; k != 3; ++k)
      finer = std::max(finer, edges[k] / static_cast<double>(grid[k] + 1));
    spacing = std::min(finer, 0.98 * spacing);
    for (int k = 0; k != 3; ++k)
      grid[k] = getSmoothSize(
          static_cast<int64_t>(std::ceil(edges[k] / spacing - 1e-9)));
    aliasing = prolate::getAliasing(psi, window, edges, grid, rc);
  }
  for (int k = 0; k != 3; ++k)
    if (grid[k] < 2 * points)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the grid of particle mesh Ewald has %lld points along an edge, "
          "fewer than twice the %lld points of the window",
          static_cast<long long>(grid[k]), static_cast<long long>(points));

  // The table of the op: rows 0 to 2 the factor B_a(k) = 1/φ̂₁(2πk/K_a)² of
  // each edge [Liang2026, eq. 9]; row 3 the Chebyshev series of the
  // weights of the window and then of their slopes in u; row 4 the number
  // of terms D of a weight, the number of them that f32 takes, the number
  // of terms n of χ̂, the scale x/|m| =
  // 2π r_c / c, then the series of χ̂ and of k χ̂'(k) in 2x² − 1.
  int64_t longest = std::max({grid[0], grid[1], grid[2]});
  int64_t columns = std::max({longest, 2 * points * window.degree,
                              4 + 2 * influence.terms});
  Program::Table table;
  table.name = "pme_moduli";
  table.count = 5;
  table.columns = columns;
  table.values.assign(5 * columns, 0.0);
  for (int k = 0; k != 3; ++k)
    for (int64_t i = 0; i != grid[k]; ++i) {
      int64_t m = i <= grid[k] / 2 ? i : i - grid[k];
      double hat = window.transform(2.0 * M_PI * static_cast<double>(m) /
                                    static_cast<double>(grid[k]));
      table.values[k * columns + i] = 1.0 / (hat * hat);
    }
  std::copy(window.values.begin(), window.values.end(),
            table.values.begin() + 3 * columns);
  std::copy(window.slopes.begin(), window.slopes.end(),
            table.values.begin() + 3 * columns + points * window.degree);
  double *row = table.values.data() + 4 * columns;
  row[0] = static_cast<double>(window.degree);
  row[1] = static_cast<double>(window.degreeSingle);
  row[2] = static_cast<double>(influence.terms);
  row[3] = 2.0 * M_PI * rc / c;
  std::copy(influence.value.begin(), influence.value.end(), row + 4);
  std::copy(influence.slope.begin(), influence.slope.end(),
            row + 4 + influence.terms);
  program.tables.push_back(std::move(table));
  // The parts of the splitting that the pair terms take.
  for (auto [name, part] :
       {std::pair{"pme_far", prolate::Part::Far},
        std::pair{"pme_near", prolate::Part::Near}}) {
    Program::Table split;
    split.name = name;
    auto [cells, terms] = getProlateTableShape(control.precision);
    split.count = cells;
    split.columns = terms;
    split.values = prolate::getSplittingTable(psi, part, cells, terms);
    program.tables.push_back(std::move(split));
  }

  program.pmeFarSeries = prolate::getFarSeries(psi);
  program.pmeProlate = true;
  program.pmeBandwidth = c;
  program.pmeWindowBandwidth = c1;
  program.pmeAliasing = aliasing;
  program.pmeOrder = points;
  for (int k = 0; k != 3; ++k)
    program.pmeGrid[k] = grid[k];
  // −f χ(0)/(2 r_c) Σ q², from Φ(r/r_c)/r → χ(0)/r_c, and the background
  // −f π Q² r_c² ⟨x²⟩ / V, from χ̂(k) = 1 − k² ⟨x²⟩ / 2 + O(k⁴).
  program.pmeSelfFactor = psi.value(0.0) / psi.halfIntegral() / (2.0 * rc);
  program.pmeBackgroundFactor = rc * rc * psi.secondMoment();

  return llvm::Error::success();
}

void Builder::collectListed(StringRef name, llvm::ArrayRef<double> values,
                            llvm::ArrayRef<Control::TunableKind> kinds) {
  // A tunable may move a value across zero, and the text of the program,
  // which an update keeps, says whether it has the list. Without the
  // declarations of the tunables, any may.
  if (control.tunables &&
      (control.tunableDeclarations.empty() ||
       llvm::any_of(control.tunableDeclarations,
                    [&](const Control::TunableDeclaration &declaration) {
                      return llvm::is_contained(kinds, declaration.kind);
                    })))
    return;
  Program::Field field;
  field.name = name.str();
  field.isInteger = true;
  for (auto [place, value] : llvm::enumerate(values))
    if (value != 0.0)
      field.values.push_back(static_cast<double>(place));
  if (field.values.size() == values.size())
    return;
  // Then −1 − the place of each of the others, whose forces of zero the
  // gathering stores.
  for (auto [place, value] : llvm::enumerate(values))
    if (value == 0.0)
      field.values.push_back(-1.0 - static_cast<double>(place));
  program.fields.push_back(std::move(field));
}

llvm::Error Builder::collectMesh(StringRef name, const int64_t (&given)[3],
                                 double spacing, int64_t order,
                                 PMEInfluence kind, int64_t (&grid)[3]) {
  // The grid: given, or no wider than the largest spacing in the cell of
  // the file of coordinates. The grid stays as the barostat changes the
  // cell, finer as the cell shrinks and coarser as it grows.
  for (int k = 0; k != 3; ++k) {
    grid[k] = given[k];
    double edge = system.inputBox[k] > 0.0 ? system.inputBox[k] : system.box[k];
    // A triclinic cell is spaced along its vectors: |a|, |b|, |c|.
    if (isTriclinic())
      edge = k == 0 ? system.box[0]
             : k == 1 ? std::hypot(system.tilt[0], system.box[1])
                      : std::sqrt(system.tilt[1] * system.tilt[1] +
                                  system.tilt[2] * system.tilt[2] +
                                  system.box[2] * system.box[2]);
    if (grid[k] == 0)
      grid[k] = getSmoothSize(static_cast<int64_t>(std::ceil(
          edge / (spacing * units::length) - 1e-9)));
    if (grid[k] < 2 * order)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the grid of particle mesh Ewald has %lld points along an edge, "
          "fewer than twice the order %lld",
          static_cast<long long>(grid[k]), static_cast<long long>(order));
  }

  // The factors of the influence function along each edge, which do not
  // depend on the cell: |b(k)|² of the B-splines, times the factor of the
  // aliasing if it is taken. The program computes the rest from the cell.
  int64_t longest = std::max({grid[0], grid[1], grid[2]});
  Program::Table table;
  table.name = name.str();
  table.columns = longest;
  if (kind == PMEInfluence::P3M) {
    // Eq. 2.28 of [Ballenegger2012], optimal for the forces of analytic
    // differentiation, with the Gaussian in a rectangular cell: a product
    // of factors of the edges over a sum of their mean squares, whose
    // weights over the aliases |j| ≤ P3MAliases of each index do not
    // depend on the cell (docs/pme.md, Section 1.1; D304).
    // Row a holds 1 / Σ_j Û², row 3 + a (2J + 1) + j + J the weight
    // Û²(n + jK) / Σ_j Û² of alias j of edge a.
    if (isTriclinic())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "influence = \"P3M\" takes a rectangular cell");
    int64_t span = 2 * P3MAliases + 1;
    table.count = 3 + 3 * span;
    table.values.assign(table.count * longest, 0.0);
    influence::Window window = influence::getBSpline(order);
    for (int k = 0; k != 3; ++k) {
      influence::EdgeWeights rows =
          influence::getEdgeWeights(window, grid[k], P3MAliases);
      std::copy(rows.inverse.begin(), rows.inverse.end(),
                table.values.begin() + k * longest);
      for (int64_t j = 0; j != span; ++j)
        std::copy(rows.weights.begin() + j * grid[k],
                  rows.weights.begin() + (j + 1) * grid[k],
                  table.values.begin() + (3 + k * span + j) * longest);
    }
    program.tables.push_back(std::move(table));
    return llvm::Error::success();
  }
  table.count = 3;
  table.values.assign(3 * longest, 0.0);
  for (int k = 0; k != 3; ++k) {
    std::vector<double> moduli = influence::getSplineModuli(grid[k], order);
    if (kind == PMEInfluence::Optimal) {
      std::vector<double> factors = influence::getAliasFactors(grid[k], order);
      for (int64_t i = 0; i != grid[k]; ++i)
        moduli[i] *= factors[i];
    }
    std::copy(moduli.begin(), moduli.end(),
              table.values.begin() + k * longest);
  }
  program.tables.push_back(std::move(table));
  return llvm::Error::success();
}

llvm::Error Builder::collectLJPME() {
  const Topology &topology = *system.topology;
  double rc = control.cutoffDistance * units::length;

  // β: given, or such that g(β rc) is the tolerance, by bisection; g
  // falls from 1 at 0.
  double beta = control.ljpmeAlpha / units::length;
  if (beta == 0.0) {
    double low = 0.0, high = 1.0;
    while (getDispersionScreen(high * rc) > control.ljpmeTolerance)
      high *= 2.0;
    for (int i = 0; i != 100; ++i) {
      double middle = 0.5 * (low + high);
      if (getDispersionScreen(middle * rc) > control.ljpmeTolerance)
        low = middle;
      else
        high = middle;
    }
    beta = 0.5 * (low + high);
  }

  int64_t grid[3];
  if (llvm::Error error =
          collectMesh("ljpme_moduli", control.ljpmeGrid,
                      control.ljpmeMaxSpacing, control.ljpmeOrder,
                      PMEInfluence::SPME, grid))
    return error;

  // The coefficient of each type, c = 2 √ε σ³ of the type with itself, so
  // that c_a c_b is the C6 of the pair by the geometric rule of both
  // parameters; the grid takes those of the particles, the direct terms
  // and the excluded pairs those of the pairs of types. For the pairs of
  // the Lorentz–Berthelot rule or set apart (NBFIX) the direct terms hold
  // the Lennard-Jones of the pair itself within the cutoff and take out
  // what the grid adds there [Wennberg2013].
  unsigned numTypes = topology.getNumTypes();
  std::vector<double> coefficients(numTypes);
  for (unsigned a = 0; a != numTypes; ++a) {
    double sigma = topology.sigma[a * numTypes + a];
    double epsilon = topology.epsilon[a * numTypes + a];
    coefficients[a] = 2.0 * std::sqrt(std::max(epsilon, 0.0)) *
                      sigma * sigma * sigma;
  }
  Program::Field field;
  field.name = "ljpme_c";
  double squares = 0.0;
  for (unsigned type : topology.types) {
    field.values.push_back(coefficients[type]);
    squares += coefficients[type] * coefficients[type];
  }
  std::vector<double> perParticle;
  for (unsigned type : topology.types)
    perParticle.push_back(coefficients[type]);
  program.fields.push_back(std::move(field));
  // The particles of a coefficient that is not zero, which the grid of the
  // dispersion spreads and gathers alone.
  collectListed("ljpme_listed", perParticle,
                {Control::TunableKind::Sigma, Control::TunableKind::Epsilon,
                 Control::TunableKind::SigmaPair,
                 Control::TunableKind::EpsilonPair});
  std::vector<double> products(numTypes * numTypes);
  for (unsigned a = 0; a != numTypes; ++a)
    for (unsigned b = 0; b != numTypes; ++b)
      products[a * numTypes + b] = coefficients[a] * coefficients[b];
  program.tables.push_back({"ljpme_c6", numTypes, std::move(products)});

  // The self term: the grid sums −c_i² (1 − g(β r)) / r⁶ over each
  // particle with itself, −c_i² β⁶ / 6 at r = 0, half of it each.
  double beta3 = beta * beta * beta;
  program.ljpme = true;
  program.ljpmeSelfEnergy = beta3 * beta3 / 12.0 * squares;
  program.ljpmeBeta = beta;
  for (int k = 0; k != 3; ++k)
    program.ljpmeGrid[k] = grid[k];
  return llvm::Error::success();
}

llvm::Error Builder::collectTunableGradient() {
  // Each tunable has a rule for its derivative, or the program provably
  // does not read it, or the compile fails (D230): nothing
  // is left out without a word.
  bool dispersion = control.topologyDispersion != DispersionCorrection::None;
  // The estimate of what the shift takes within the cutoff, which the
  // energy of the run holds only under the shift (D210).
  if (dispersion && control.truncation != Truncation::Shift) {
    double tail = getTopologyDispersion(false);
    program.gradientShiftEnergy = tail;
    program.gradientShiftFixedEnergy = tail * getShiftFixedFactor();
    for (const TermBuilder *builder : registerTermBuilders())
      builder->collectGradientShiftConstants(*this);
  }
  for (auto [k, declaration] : llvm::enumerate(control.tunableDeclarations)) {
    Program::GradientOutcome outcome = Program::GradientOutcome::Zero;
    bool registered = false;
    for (const TermBuilder *builder : registerTermBuilders()) {
      auto result = builder->collectTunableOutcome(*this, static_cast<unsigned>(k));
      if (!result)
        return result.takeError();
      if (*result) {
        outcome = **result;
        registered = true;
        break;
      }
    }
    if (!registered) {
      switch (declaration.kind) {
      case Control::TunableKind::Charge: {
        auto refuse = [&](const llvm::Twine &what) {
          return makeError("System.tunable_gradient: the derivative of the "
                           "energy in the tunable '" + declaration.name +
                           "' (charges) " + what + " is not implemented yet");
        };
        if (control.implicitSolvent != Control::ImplicitSolvent::None)
          return refuse("with an implicit solvent");
        // The tail of a pair term that reads the charges follows them through
        // its classes of particles (D209).
        for (auto [index, term] : llvm::enumerate(control.pairs)) {
          Expression expression = llvm::cantFail(
              Expression::parse(term.expression, control.functions));
          bool reads = llvm::is_contained(expression.getNames(), "q1") ||
                       llvm::is_contained(expression.getNames(), "q2");
          if (reads && dispersion && index < system.pairTails.size() &&
              !system.pairTails[index].empty())
            return refuse("through the tail of the pair term '" + term.name +
                          "', which reads the charges,");
        }
        // The Python model has no reaction field; its self term and its
        // excluded pairs would follow the charges as well (D140).
        if (control.reactionField)
          return refuse("with the reaction field");
        // Nothing that the host adds to the energy follows the charges under
        // a Coulomb cutoff. With particle mesh Ewald the self term,
        // −f β Σ q² / √π, and the background of a net charge Q,
        // −f π Q² / (2 V β²), do (getPMEConstants): −2 f β q_i / √π, which
        // does not depend on the cell, and −f π Q / (V β²)
        // (D231).
        const std::vector<double> &charges = system.topology->charges;
        program.gradientChargeFixed.assign(charges.size(), 0.0);
        program.gradientChargeVolume.assign(charges.size(), 0.0);
        if (program.pme) {
          // −2 f s q_i and −2 f π Q b / V, with s and b of the kernel.
          double net = 0.0;
          for (double q : charges)
            net += q;
          double volume = system.box[0] * system.box[1] * system.box[2];
          for (size_t i = 0; i != charges.size(); ++i) {
            program.gradientChargeFixed[i] =
                -2.0 * coulombInternal * program.pmeSelfFactor * charges[i];
            program.gradientChargeVolume[i] = -2.0 * coulombInternal * M_PI *
                                              net * program.pmeBackgroundFactor /
                                              volume;
          }
        }
        outcome = Program::GradientOutcome::Rule;
        break;
      }
      case Control::TunableKind::Sigma:
      case Control::TunableKind::Epsilon:
      case Control::TunableKind::SigmaPair:
      case Control::TunableKind::EpsilonPair:
        // The Lennard-Jones of the topology reads the tables; so would a
        // pair term that names sigma or epsilon, whose kernel and tail have
        // no seeds yet.
        for (const PairTerm &term : control.pairs) {
          Expression expression = llvm::cantFail(
              Expression::parse(term.expression, control.functions));
          for (const char *name : {"sigma", "epsilon", "sigma1", "sigma2",
                                   "epsilon1", "epsilon2"})
            if (llvm::is_contained(expression.getNames(), name))
              return makeError(
                  "System.tunable_gradient: the derivative of the energy in "
                  "the tunable '" + declaration.name + "' through the pair "
                  "term '" + term.name + "', which reads '" + name +
                  "', is not implemented yet");
        }
        outcome = Program::GradientOutcome::Rule;
        break;
      default:
        llvm_unreachable("a registered adapter handles this tunable kind");
      }
    }
    program.gradientOutcomes.push_back(outcome);
  }
  // What the correction for the dispersion and the estimate of its shift
  // add to the rows of the seeds of the tables. The correction is
  // K Σ_ab n_ab C6_ab over the ordered pairs of types, C6 = 4 ε σ⁶, with
  // n_ab the ordered pairs of particles of the types less the excluded
  // ones (getTopologyDispersion), and the estimate of the shift is the
  // correction times a factor of the cell (D210). With G_ab its
  // derivative in the entry (a, b) of a table, the row a of a seed of
  // weights w takes Σ_b (G_ab + G_ba) w_ab: the entry (a, b) follows the
  // value of the type a through its first place and (b, a) through its
  // second.
  if (dispersion && !control.tunableTableSeeds.empty()) {
    const Topology &topology = *system.topology;
    unsigned numTypes = topology.getNumTypes();
    std::vector<double> numbers(numTypes, 0.0), pairs(numTypes * numTypes);
    for (unsigned type : topology.types)
      numbers[type] += 1.0;
    for (unsigned a = 0; a != numTypes; ++a)
      for (unsigned b = 0; b != numTypes; ++b)
        pairs[a * numTypes + b] =
            numbers[a] * (numbers[b] - (a == b ? 1.0 : 0.0));
    for (auto [i, j] : topology.exclusions)
      pairs[topology.types[i] * numTypes + topology.types[j]] -= 2.0;
    double n = static_cast<double>(topology.getNumParticles());
    double ordered = n * (n - 1.0) - 2.0 * topology.exclusions.size();
    double rc = control.cutoffDistance * units::length;
    double volume = system.box[0] * system.box[1] * system.box[2];
    double factor = ordered > 0.0
                        ? -2.0 * M_PI / (3.0 * volume) * n * n /
                              (ordered * rc * rc * rc)
                        : 0.0;
    // The tail and the part of the estimate of the shift that follows the
    // volume, twice the tail, and the part that does not (#224).
    double scaledPart = 2.0, fixedPart = getShiftFixedFactor();
    for (auto [c, seed] : llvm::enumerate(control.tunableTableSeeds)) {
      const std::vector<double> &weights = system.tunableSeedWeights[c];
      auto slope = [&](unsigned a, unsigned b) {
        double sigma = topology.sigma[a * numTypes + b];
        double epsilon = topology.epsilon[a * numTypes + b];
        double c6 = seed.sigma ? 24.0 * epsilon * std::pow(sigma, 5)
                               : 4.0 * std::pow(sigma, 6);
        return factor * pairs[a * numTypes + b] * c6;
      };
      for (unsigned a = 0; a != numTypes; ++a) {
        double sum = 0.0;
        for (unsigned b = 0; b != numTypes; ++b)
          if (weights[a * numTypes + b] != 0.0)
            sum += (slope(a, b) + slope(b, a)) * weights[a * numTypes + b];
        program.gradientFields[c].rows[a].volumeConstant = sum * scaledPart;
        program.gradientFields[c].rows[a].fixedConstant = sum * fixedPart;
      }
    }
  }
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
    for (auto [k, constant] : llvm::enumerate(control.tunableConstants))
      for (const auto &[name, value] : control.pairs[constant.first].constants)
        if (name == constant.second)
          add("tg_c" + std::to_string(k), value);
  }
}

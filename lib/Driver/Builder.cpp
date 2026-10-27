// Builds the program of a run: the order of its parts, and the entry
// points of mdir/Driver/Builder.h (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

std::string Builder::getFieldParameters() const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += ", %p_" + field.name + ": " + getFieldType(field).str();
  for (const Program::Table &table : program.tables)
    text += ", %t_" + table.name + ": " + table.getType().str();
  for (const Program::TupleSet &set : program.tupleSets) {
    text += ", %r_" + set.name + ": !rel_" + set.name;
    for (const Program::Field &field : set.fields)
      text += ", %f_" + set.name + "_" + field.name + ": !of_" + set.name;
  }
  return text;
}

std::string Builder::getFieldValues(StringRef prefix) const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += (", " + prefix + field.name).str();
  // Tables do not change with the order of the particles; they keep their
  // names everywhere.
  for (const Program::Table &table : program.tables)
    text += ", %t_" + table.name;
  // Neither do the tuples of a topology: the program does not put the
  // particles in a new order when it has them.
  // The members of the tuples follow the order of the particles: their
  // names are those of the fields with `%r` for `%p`.
  std::string relations = ("%r" + prefix.drop_front(2)).str();
  for (const Program::TupleSet &set : program.tupleSets) {
    text += ", " + relations + set.name;
    for (const Program::Field &field : set.fields)
      text += ", %f_" + set.name + "_" + field.name;
  }
  return text;
}

void Builder::emitReorder(StringRef indent, StringRef from,
                          StringRef stateTo, StringRef otherTo,
                          bool withForces, StringRef velocities,
                          StringRef forcesTo) {
  StringRef order = "!mdrt.permutation<@atoms>";
  // A particle goes where the anchor of its constraint group is (D110):
  // the positions that the order takes are those of the anchors, from the
  // pairs of the anchors in the order the particles are in now.
  std::string ordered = ("%x" + from).str();
  if (llvm::any_of(program.tupleSets, [](const Program::TupleSet &set) {
        return set.name == "anchors";
      })) {
    ordered = ("%xanchor" + stateTo).str();
    os << indent << "%ranchor" << stateTo << " = md_exec.renumber %ro_anchors, %id"
       << from << " : !rel_anchors, !ids\n"
       << indent << ordered << " = md.gather_tuples %ranchor" << stateTo << ", %x"
       << from << ", " << cellName << " coordinates(displacement(1, 0)) gather(%x" << from
       << " : !vec) {\n"
       << indent << "^bb0(%xanchor_d: vector<3xf64>, %xanchor_0: vector<3xf64>, "
       << "%xanchor_1: vector<3xf64>):\n"
       << indent << "  %xanchor_none = arith.constant dense<0.0> : vector<3xf64>\n"
       << indent << "  md.yield %xanchor_1, %xanchor_none : vector<3xf64>, vector<3xf64>\n"
       << indent << "} : !rel_anchors, !vec -> !vec\n";
  }
  os << indent << "%order" << stateTo << " = md_exec.spatial_order "
     << ordered << ", " << cellName << ", %id" << from << " width("
     << formatReal(program.orderWidth) << ")\n"
     << indent << "    : !vec, !ids -> " << order << "\n";
  auto permute = [&](const llvm::Twine &result, const llvm::Twine &field,
                     StringRef type) {
    os << indent << result << " = md_exec.permute " << field << ", %order"
       << stateTo << "\n"
       << indent << "    : " << type << ", " << order << " -> " << type
       << "\n";
  };
  permute("%x" + stateTo, "%x" + from, "!vec");
  permute(velocities, "%v" + from, "!vec");
  if (withForces) {
    if (forcesTo.empty())
      permute("%f" + stateTo, "%f" + from, "!vec");
    else
      permute(forcesTo, "%f" + from, "!vec");
  }
  permute("%m" + otherTo, "%m" + from, "!real");
  for (const Program::Field &field : program.fields) {
    // A list of the places of particles is made again from the permuted
    // field that it lists (D262), which comes
    // before it.
    StringRef name = field.name;
    if (name == "pme_listed" || name == "ljpme_listed") {
      StringRef source = name == "pme_listed" ? "q" : "ljpme_c";
      os << indent << "%p" << otherTo << "_" << name << " = md_exec.nonzero %p"
         << otherTo << "_" << source << " : !real -> !ids\n";
      continue;
    }
    permute("%p" + otherTo + "_" + field.name,
            "%p" + from + "_" + field.name, getFieldType(field));
  }
  permute("%id" + otherTo, "%id" + from, "!ids");
  emitRenumber(indent, "%id" + otherTo, "%r" + otherTo + "_");
}

void Builder::emitRenumber(StringRef indent, const llvm::Twine &ids,
                           const llvm::Twine &prefix) {
  // From the members in the order of the files.
  for (const Program::TupleSet &set : program.tupleSets)
    os << indent << prefix << set.name << " = md_exec.renumber %ro_"
       << set.name << ", " << ids << " : !rel_" << set.name << ", !ids\n";
}

std::string Builder::getFieldTypes() const {
  std::string text;
  for (const Program::Field &field : program.fields)
    text += ", " + getFieldType(field).str();
  for (const Program::Table &table : program.tables)
    text += ", " + table.getType().str();
  for (const Program::TupleSet &set : program.tupleSets) {
    text += ", !rel_" + set.name;
    for (size_t i = 0, e = set.fields.size(); i != e; ++i)
      text += ", !of_" + set.name;
  }
  return text;
}

std::vector<const Program::TupleSet *> Builder::getRigidGroups() const {
  std::vector<const Program::TupleSet *> groups = getShakeSets();
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      groups.insert(groups.begin(), &set);
  return groups;
}

llvm::Error Builder::build() {
  // Every front end resolves `AUTO` for its system first
  // (resolveNeighborStructure, D261).
  if (control.neighborStructure == NeighborStructure::Auto)
    return makeError("the neighbor structure 'AUTO' was not resolved; this "
                     "is a defect of mdir");
  if (const char *spread = std::getenv("MDIR_SITE_SPREAD");
      spread && *spread && StringRef(spread) != "derived" &&
      StringRef(spread) != "written")
    return makeError("MDIR_SITE_SPREAD is '" + llvm::Twine(spread) +
                     "'; expected 'derived' or 'written'");
  // The neighbor matrix of a triclinic cell holds every pair that has an
  // image within its reach, whatever the reach: its build tests the images
  // beyond that of the one pass where the reach is more than half of the
  // least of a_x, b_y, c_z (docs/triclinic.md, Section 2). It counts at
  // most 64 images a side along a lattice vector, which holds every image
  // within the reach while the reach is at most 63.5 times the least of
  // a_x, b_y, c_z, and so, the diagonal being at least twice the cutoff
  // (I2), while the pairlist distance is at most 127 times the cutoff:
  // a bound that does not depend on the cell, so that a cell that a
  // barostat scales needs no other test than I2.
  if (isTriclinic() &&
      control.neighborStructure == NeighborStructure::Matrix &&
      control.pairlistDistance > 127.0 * control.cutoffDistance)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "'pairlist_distance', %g Å, exceeds 127 times 'cutoff', %g Å, the "
        "most that the neighbor matrix of a triclinic cell takes",
        control.pairlistDistance, control.cutoffDistance);
  // The groups hold an entry for every image of a pair within their reach
  // while the reach is less than the least edge of the cell, the least of
  // a_x, b_y, c_z of a triclinic one: their lists take the image of a
  // candidate nearest to the center of the box of a group and those a
  // lattice vector away along each axis, which are the images within three
  // halves of the cell of that center, and a box and the reach span at
  // most half the cell and the reach (docs/neighbor-groups.md, Section
  // 5.2). The runtime asks the same of a cell that a barostat scales
  // (`leastReachEdge`). The 0.999 is for the margin by which the builds
  // widen the reach, 3e-6 of the sum of the edges.
  if (control.periodic &&
      control.neighborStructure == NeighborStructure::Groups) {
    double least = std::min({system.box[0], system.box[1], system.box[2]});
    double reach = control.pairlistDistance * units::length;
    if (reach > 0.999 * least)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "'pairlist_distance', %g Å, exceeds 0.999 of the least %s of the "
          "cell, %g Å, the most that the groups take",
          control.pairlistDistance,
          isTriclinic() ? "of a_x, b_y, c_z" : "edge",
          0.999 * least / units::length);
  }
  // A pair is taken once, in the minimum image, which holds every image
  // within the cutoff only while the cell is wider than twice the cutoff:
  // for a triclinic cell, its diagonal (I2 of docs/triclinic.md).
  static const char axes[] = "xyz";
  for (int k = 0; k != 3; ++k)
    if (system.box[k] < 2.0 * control.cutoffDistance * units::length)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the cell is %.4f Å along %c, less than twice the cutoff, %.4f Å",
          system.box[k] / units::length, axes[k],
          control.cutoffDistance);
  program.entry = "mdir_run";
  // Tunable parameters (D213) are values of the buffers of a
  // program of segments, of the Python simulation that updates them.
  if (control.tunables) {
    if (!system.topology)
      return makeError("tunable parameters are for a model with a topology");
    if (control.hasFreeEnergy || control.ljpme || hasCoordinateOutput())
      return makeError("tunable parameters do not take [free_energy], "
                       "LJPME, or pulls yet");
    program.tunable = true;
    // The derivative in the tunables is a branch of the start of a program
    // that takes steps (D230).
    program.tunableGradient =
        control.tunableGradient && !control.minimize;
  }
  switch (control.precision) {
  case Precision::Single:
    program.state = program.mass = Element::F32;
    program.force = program.parameter = Element::F32;
    break;
  case Precision::Mixed:
    program.state = program.mass = Element::F64;
    program.force = program.parameter = Element::F32;
    // In the deterministic mode the forces that a step carries to the next
    // are stored as the state is (#102). Buffers of forces in f32 at the
    // host calls would make the forces that a loop of steps carries f32
    // where the loop's result reaches such a call (a run whose every step
    // writes energies) and f64 where it does not (md-exec-assign-precision
    // gives fields stored together the type of a buffer that holds one),
    // and the trajectory would depend on the schedule of the energies.
    if (control.deterministic)
      program.force = Element::F64;
    break;
  case Precision::Double:
    program.state = program.mass = Element::F64;
    program.force = program.parameter = Element::F64;
    break;
  }
  // Velocity Verlet begins a step with the forces of the step before.
  program.writesForces = !control.minimize;
  program.takesForces =
      ((isRestart() && !recomputes()) || branchesStart()) && !control.minimize;

  program.skin =
      (control.pairlistDistance - control.cutoffDistance) * units::length;
  if (control.prunedDistance != 0.0)
    program.pruneSkin =
        (control.prunedDistance - control.cutoffDistance) * units::length;
  // Cells of half the reach of a neighbor structure: particles that are
  // neighbors are then a few cells apart in memory.
  // A program of frames keeps the particles in the order of the input
  // (D251): an order chosen at every frame would renumber
  // the tuples and build their incidence structures on the host at every
  // frame, which is most of what the start of an activation costs, and the
  // neighbor structures bin the particles themselves.
  program.reorders = control.reorder && !control.minimize && !control.frames;
  program.orderWidth = 0.5 * control.pairlistDistance * units::length;
  program.neighborWidth = control.neighborWidth;
  if (program.neighborWidth == 0) {
    // Half as many again as the particle with the most neighbors has at
    // the start, and a few more. A uniform density would underestimate a
    // box with room in it, such as a solvated molecule that tleap makes.
    double reach = control.pairlistDistance * units::length;
    int64_t most = countMostNeighbors(
        llvm::ArrayRef(system.positions)
            .take_front(3 * system.getNumParticles()),
        system.box, reach);
    int64_t width = static_cast<int64_t>(std::ceil(1.5 * most)) + 16;
    // Rounded up to four significant bits, a multiple of an eighth of the
    // power of two below it and of 8: at most an eighth more room. The
    // count moves by a few neighbors from one equilibrated state to the
    // next, and the width is part of the lowered program, so a finer
    // rounding made stages that differ only in their start compile anew
    // (D227). The width only sets the room at first.
    int64_t step = 8;
    while (16 * step <= width)
      step *= 2;
    program.neighborWidth = (width + step - 1) / step * step;
  }

  // A program of segments has no outputs of its own: its loops are those
  // of the periods of coupling and of the steps (D196), or of the steps of
  // a minimization (D202).
  if (control.segments && scalesEveryStep())
    return makeError("a program of segments with a barostat that scales the "
                     "cell every step is not supported");
  if (control.segments && (control.energyPeriod > 0 ||
                           control.framePeriod > 0 ||
                           control.checkpointPeriod > 0))
    return makeError("a program of segments takes no energies, frames, or "
                     "checkpoints");

  // The loops of the schedule; a minimization has its own
  // (emitMinimization).
  if (!control.minimize)
    setSchedule();
  if (control.segments && !control.minimize) {
    bool couples = levels[0].name == "couple";
    program.segmentPeriod = couples ? control.getCouplingPeriod() : 0;
    program.closingSteps = getClosingSteps();
    // The second nest: one interval of energy of one period, which its
    // step of energy closes, as `mdir run` ends an interval of its log.
    if (couples) {
      levels.push_back({"energy", 1});
      levels.push_back({"couple", 0});
      levels.push_back({"step", 0});
    } else {
      levels.push_back({"energy", 1});
      levels.push_back({"step", 0});
    }
  }

  HostMark mark("rebuild.build.topology");
  if (system.topology) {
    if (llvm::Error error = collectTopology())
      return error;
  } else if (llvm::Error error = collectParameters()) {
    return error;
  }
  // A learned potential takes every particle with its element, and ML1 has
  // no constraints and no virtual sites beside it
  // (docs/metatomic-ml1.md, Section 6.9).
  if (!control.stagedModels.empty()) {
    if (!system.topology)
      return makeError("a model in stages of [[energy.model]] needs a "
                       "topology, which gives the elements of the particles");
    if (hasSites() || !system.topology->virtualSites.empty())
      return makeError("a model in stages takes no virtual sites yet");
  }
  if (!control.models.empty()) {
    if (!system.topology)
      return makeError("a term of [[energy.model]] needs a topology, which "
                       "gives the elements of the particles");
    if (hasConstraints())
      return makeError("a term of [[energy.model]] takes no constraints "
                       "yet: give a flexible model (no SETTLE or SHAKE)");
    if (hasSites() || !system.topology->virtualSites.empty())
      return makeError("a term of [[energy.model]] takes no virtual sites "
                       "yet");
    if (control.precision == Precision::Single)
      return makeError("a term of [[energy.model]] takes the mixed or the "
                       "double precision");
    // A frame evaluator is chosen when a Python simulation is made, after
    // the checks of the control: its forces and virial come from
    // evaluations of their own, which do not take the model.
    if (control.frames)
      return makeError("a term of [[energy.model]] is not evaluated by the "
                       "frame evaluator yet");
    if (control.tunables || control.tunableGradient)
      return makeError("a term of [[energy.model]] has no derivative in a "
                       "tunable parameter: tunables are refused");
    // The pair list is a value of the program, at the model's cutoff
    // (D277).
    llvm::Expected<int> mode = getPairsMode();
    if (!mode)
      return mode.takeError();
    modelEdges = *mode != MDIR_MLIP_PAIRS_SEARCH;
    if (modelEdges)
      for (const Control::ModelTerm &term : control.models) {
        llvm::Expected<double> cutoff = getModelCutoff(term);
        if (!cutoff)
          return cutoff.takeError();
        modelCutoffs.push_back(*cutoff);
      }
  }
  // A build of the arrays alone ends with them.
  if (capture)
    return llvm::Error::success();
  mark.next("rebuild.build.sets");
  // The polynomials of the tabulated functions, a row of 4ⁿ numbers for
  // each cell of a function of n arguments (D138, D165), or the values of a
  // discrete one.
  for (const TabulatedFunction &function : control.functions) {
    Program::Table table;
    table.name = function.getTableName();
    if (llvm::any_of(program.tables, [&](const Program::Table &other) {
          return other.name == table.name;
        }))
      return makeError("the table of the function '" + function.name +
                       "' takes the name of another; rename the function");
    table.values = function.getCoefficients();
    table.columns = function.getColumns();
    table.count = table.values.size() / table.columns;
    program.tables.push_back(std::move(table));
  }

  os << "!vec   = !md.field<@atoms, 3 x f64>\n"
     << "!real  = !md.field<@atoms, f64>\n"
     << "!ids   = !md.field<@atoms, i32>\n"
     << "!table = !md.table<2, f64, symmetric>\n"
     << "!grid = !md.table<2, f64>\n"
     << "!pairs = !md.relation<@atoms, 2, unordered>\n";
  for (const Program::TupleSet &set : program.tupleSets) {
    os << "!rel_" << set.name << " = !md.relation<@atoms, " << set.arity
       << ", " << set.getOrientation() << ", @" << set.name << ">\n"
       << "!of_" << set.name << " = !md.field<@" << set.name << ", f64>\n";
  }
  os << "\nmd.particle_set @atoms\n";
  for (const Program::TupleSet &set : program.tupleSets)
    os << "md.tuple_set @" << set.name << " on(@atoms) arity(" << set.arity
       << ") orientation(" << set.getOrientation() << ")"
       << (set.arity > 1 && set.isDisjoint() ? " disjoint" : "") << "\n";
  // The groups of the constraints share no atom with one another either
  // (D83): the bonds of a rigid water are gone before SHAKE groups the
  // bonds of hydrogen, and a hydrogen is in one group only (findShakes).
  // What the groups promise is checked here all the same, and without the
  // promise the loops keep their order.
  std::vector<const Program::TupleSet *> groups;
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      groups.push_back(&set);
  for (const Program::TupleSet *set : getShakeSets())
    groups.push_back(set);
  std::vector<int32_t> members;
  for (const Program::TupleSet *set : groups)
    members.insert(members.end(), set->members.begin(), set->members.end());
  if (groups.size() >= 2 && Program::TupleSet::hasNoRepeat({&members})) {
    os << "md.disjoint_union @constraints on(@atoms) of [";
    llvm::interleaveComma(groups, os, [&](const Program::TupleSet *set) {
      os << "@" << set->name;
    });
    os << "]\n";
  }
  os << "\n";
  mark.next("rebuild.build.text");
  if (llvm::Error error = emitStagedModels())
    return error;
  if (system.topology) {
    emitTopologyPotential("energy", AllTerms);
    // Each term on its own, for the log at the start.
    if (!isRestart())
      for (auto [name, term] :
           {std::pair<StringRef, unsigned>{"term_lj", LennardJones},
            {"term_coulomb", Coulomb},
            {"term_bonds", Bonds},
            {"term_angles", Angles},
            {"term_dihedrals", Dihedrals},
            {"term_lj14", LennardJones14},
            {"term_coulomb14", Coulomb14},
            {"term_cmap", CMaps},
            {"term_excluded", CoulombExcluded | CoulombWithin},
            {"term_reciprocal", CoulombReciprocal},
            {"term_urey_bradley", UreyBradleys},
            {"term_impropers", HarmonicImpropers},
            {"term_lj_excluded", LennardJonesExcluded},
            {"term_lj_reciprocal", LennardJonesReciprocal}})
        emitTopologyPotential(name, term);
    if (!isRestart())
      for (const TermBuilder *builder : termBuildersByPotentialRank())
        if (builder->potentialRank() < -2)
          builder->emitTermPotentials(*this);
    if (!isRestart())
      for (size_t k = 0, e = control.pairs.size(); k != e; ++k)
        emitTopologyPotential("term_pair" + std::to_string(k), PairTerms, -1,
                              static_cast<int>(k));
    if (!isRestart())
      for (const TermBuilder *builder : termBuildersByPotentialRank())
        if (builder->potentialRank() >= -2)
          builder->emitTermPotentials(*this);
  }
  else if (llvm::Error error = emitPotential())
    return error;
  // What depends on λ, as a function of it, for dH/dλ and the energies of
  // the other states (D161).
  if (system.topology && !control.freeEnergyFile.empty()) {
    lambdaArguments = true;
    shiftsAtCutoff = true;
    emitTopologyPotential("alchemical", AllTerms | Alchemical);
    shiftsAtCutoff = false;
    lambdaArguments = false;
  }
  // The potential whose derivatives in the tunables the host asks for:
  // that of the energy, with the tunable constants as arguments
  // (D230).
  if (program.tunableGradient) {
    if (llvm::Error error = collectTunableGradient())
      return error;
    // It is the potential that the forces sample: where the run cuts a
    // pair term at the cutoff, shifted to 0 there (D210).
    gradientArguments = true;
    shiftsAtCutoff = true;
    emitTopologyPotential("tunable", AllTerms);
    shiftsAtCutoff = false;
    gradientArguments = false;
  }
  // The energies and derivatives of `observe`, of the potential that the
  // forces sample (D210).
  shiftsAtCutoff = true;
  emitObservedPotentials();
  shiftsAtCutoff = false;
  for (const TermBuilder *term : registerTermBuilders())
    term->emitAuxiliaryPotentials(*this);
  emitSitePotentials();
  emitPrograms();
  emitEntry();
  if (!frameLoopError.empty())
    return makeError(frameLoopError);
  return llvm::Error::success();
}

DependentTerms mdir::driver::getDependentTerms(const Control &control,
                                               const System &system) {
  DependentTerms kept;
  auto reads = [&](StringRef expression, std::initializer_list<StringRef> names) {
    auto parsed = Expression::parse(expression, control.functions);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return true;
    }
    for (StringRef name : names)
      if (llvm::is_contained(parsed->getNames(), name.str()))
        return true;
    return false;
  };
  auto addPair = [&](unsigned index) {
    if (!kept.hasPair(index))
      kept.pairs.push_back(index);
  };
  for (const Control::TunableDeclaration &declaration :
       control.tunableDeclarations)
    switch (declaration.kind) {
    case Control::TunableKind::Charge:
      kept.coulomb = true;
      for (auto [index, term] : llvm::enumerate(control.pairs))
        if (reads(term.expression, {"q1", "q2"}))
          addPair(static_cast<unsigned>(index));
      break;
    case Control::TunableKind::Sigma:
    case Control::TunableKind::Epsilon:
    case Control::TunableKind::SigmaPair:
    case Control::TunableKind::EpsilonPair:
      kept.lennardJones = true;
      break;
    case Control::TunableKind::PairConstant:
      if (declaration.termIndex < control.pairs.size() &&
          reads(control.pairs[declaration.termIndex].expression,
                {declaration.parameter}))
        addPair(declaration.termIndex);
      break;
    case Control::TunableKind::TupleParameter:
      if (system.topology &&
          declaration.termIndex < system.topology->tupleTerms.size() &&
          reads(system.topology->tupleTerms[declaration.termIndex].expression,
                {declaration.parameter}) &&
          !kept.hasTuple(declaration.termIndex))
        kept.tuples.push_back(declaration.termIndex);
      break;
    }
  std::sort(kept.pairs.begin(), kept.pairs.end());
  std::sort(kept.tuples.begin(), kept.tuples.end());
  if (kept.lennardJones)
    kept.names.push_back("lennard_jones");
  if (kept.coulomb)
    kept.names.push_back("coulomb");
  for (unsigned index : kept.pairs)
    kept.names.push_back("pair:" + control.pairs[index].name);
  for (unsigned index : kept.tuples)
    kept.names.push_back("tuple:" + system.topology->tupleTerms[index].name);
  return kept;
}

void mdir::driver::keepDependentTerms(Control &control, const System &system) {
  control.dependentTerms = true;
  DependentTerms kept = getDependentTerms(control, system);
  // Without tunable charges there are no electrostatics, so no mesh.
  if (!kept.coulomb) {
    control.pme = false;
    control.reactionField = false;
  }
  // The observed columns of the terms that stay.
  auto stays = [&](const Control::Observable &observable) {
    for (unsigned index : kept.pairs)
      if (control.pairs[index].name == observable.term)
        return true;
    for (unsigned index : kept.tuples)
      if (system.topology->tupleTerms[index].name == observable.term)
        return true;
    return false;
  };
  llvm::erase_if(control.observables,
                 [&](const Control::Observable &o) { return !stays(o); });
}

llvm::Expected<Program> mdir::driver::buildProgram(const Control &control,
                                                   const System &system,
                                                   const BuildKept *kept) {
  Program program;
  Builder builder(control, system, program);
  builder.kept = kept;
  if (llvm::Error error = builder.build())
    return std::move(error);
  printParsedModule(program);
  return std::move(program);
}

llvm::Expected<std::shared_ptr<BuildKept>>
mdir::driver::buildArrays(const Control &control, const System &system) {
  auto arrays = std::make_shared<BuildKept>();
  Program program;
  Builder builder(control, system, program);
  builder.capture = arrays.get();
  if (llvm::Error error = builder.build())
    return std::move(error);
  return arrays;
}

namespace {
/// Whether two lists of numbers are the same to the bit; a number that is
/// not a number equals itself here.
template <typename T>
bool sameBits(const std::vector<T> &a, const std::vector<T> &b) {
  return a.size() == b.size() &&
         (a.empty() ||
          std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
template <typename T>
bool sameBits(const SharedArray<T> &a, const SharedArray<T> &b) {
  return a.shares(b) || sameBits(a.get(), b.get());
}
bool sameBits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
} // namespace

bool mdir::driver::keepArraysOf(BuildKept &arrays, const Program &program) {
  // The lists are those of the program, which may hold more after them
  // (the tables of the mesh and of the tabulated functions).
  if (arrays.fields.size() > program.fields.size() ||
      arrays.tables.size() > program.tables.size() ||
      arrays.tupleSets.size() != program.tupleSets.size() ||
      arrays.gradientFields.size() != program.gradientFields.size())
    return false;
  arrays.kept = arrays.filled = 0;
  auto keep = [&](Program::Field &field, const Program::Field &other) {
    if (field.name != other.name || field.isInteger != other.isInteger)
      return false;
    if (sameBits(field.values, other.values)) {
      field.values = other.values;
      ++arrays.kept;
    } else {
      field.values.clear();
      ++arrays.filled;
    }
    return true;
  };
  for (size_t k = 0; k != arrays.fields.size(); ++k)
    if (!keep(arrays.fields[k], program.fields[k]))
      return false;
  for (size_t k = 0; k != arrays.tables.size(); ++k) {
    Program::Table &table = arrays.tables[k];
    const Program::Table &other = program.tables[k];
    if (table.name != other.name || table.count != other.count ||
        table.columns != other.columns)
      return false;
    if (sameBits(table.values, other.values)) {
      ++arrays.kept;
    } else {
      table.values.clear();
      ++arrays.filled;
    }
  }
  for (size_t k = 0; k != arrays.tupleSets.size(); ++k) {
    Program::TupleSet &set = arrays.tupleSets[k];
    const Program::TupleSet &other = program.tupleSets[k];
    // The members of a tuple set hold no value of a tunable.
    if (set.name != other.name || set.arity != other.arity ||
        set.reversible != other.reversible || set.oriented != other.oriented ||
        set.fields.size() != other.fields.size() ||
        !sameBits(set.members, other.members))
      return false;
    set.members = other.members;
    ++arrays.kept;
    for (size_t f = 0; f != set.fields.size(); ++f)
      if (!keep(set.fields[f], other.fields[f]))
        return false;
  }
  for (size_t k = 0; k != arrays.gradientFields.size(); ++k) {
    Program::GradientField &field = arrays.gradientFields[k];
    const Program::GradientField &other = program.gradientFields[k];
    if (field.tunable != other.tunable || field.rows.size() != other.rows.size() ||
        !sameBits(field.sites, other.sites))
      return false;
    for (size_t r = 0; r != field.rows.size(); ++r)
      if (field.rows[r].site != other.rows[r].site ||
          !sameBits(field.rows[r].scale, other.rows[r].scale))
        return false;
    field.sites = other.sites;
    ++arrays.kept;
  }
  return true;
}

std::pair<size_t, size_t> mdir::driver::countArrays(const BuildKept &arrays) {
  return {arrays.kept, arrays.filled};
}

bool mdir::driver::spoilArray(BuildKept &arrays) {
  for (Program::Field &field : arrays.fields)
    if (!field.values.empty() && !field.isInteger) {
      field.values.edit()[0] += 1.0;
      return true;
    }
  return false;
}

std::string mdir::driver::describeDifference(const Program &a,
                                             const Program &b) {
  auto fields = [&](const std::vector<Program::Field> &x,
                    const std::vector<Program::Field> &y,
                    const std::string &where) -> std::string {
    if (x.size() != y.size())
      return "the number of the fields of " + where;
    for (size_t k = 0; k != x.size(); ++k)
      if (x[k].name != y[k].name || x[k].isInteger != y[k].isInteger ||
          !sameBits(x[k].values, y[k].values))
        return "the field '" + x[k].name + "' of " + where;
    return "";
  };
  if (a.module != b.module)
    return "the text";
  if (a.entry != b.entry || a.state != b.state || a.force != b.force ||
      a.mass != b.mass || a.parameter != b.parameter ||
      a.takesForces != b.takesForces || a.writesForces != b.writesForces ||
      a.pme != b.pme || a.reactionField != b.reactionField ||
      a.tunableGradient != b.tunableGradient || a.ljpme != b.ljpme ||
      a.neighborWidth != b.neighborWidth || a.reorders != b.reorders ||
      a.segmentPeriod != b.segmentPeriod || a.closingSteps != b.closingSteps ||
      a.tunable != b.tunable || a.gradientOutcomes != b.gradientOutcomes ||
      a.gradientArguments != b.gradientArguments ||
      !std::equal(std::begin(a.pmeGrid), std::end(a.pmeGrid), b.pmeGrid) ||
      !std::equal(std::begin(a.ljpmeGrid), std::end(a.ljpmeGrid), b.ljpmeGrid))
    return "the description of the program";
  if (std::string where = fields(a.fields, b.fields, "the particles");
      !where.empty())
    return where;
  if (a.tables.size() != b.tables.size())
    return "the number of the tables";
  for (size_t k = 0; k != a.tables.size(); ++k)
    if (a.tables[k].name != b.tables[k].name ||
        a.tables[k].count != b.tables[k].count ||
        a.tables[k].columns != b.tables[k].columns ||
        !sameBits(a.tables[k].values, b.tables[k].values))
      return "the table '" + a.tables[k].name + "'";
  if (a.tupleSets.size() != b.tupleSets.size())
    return "the number of the tuple sets";
  for (size_t k = 0; k != a.tupleSets.size(); ++k) {
    const Program::TupleSet &x = a.tupleSets[k], &y = b.tupleSets[k];
    if (x.name != y.name || x.arity != y.arity ||
        x.reversible != y.reversible || x.oriented != y.oriented ||
        !sameBits(x.members, y.members))
      return "the tuple set '" + x.name + "'";
    if (std::string where = fields(x.fields, y.fields, "'" + x.name + "'");
        !where.empty())
      return where;
  }
  struct Scalar {
    const char *name;
    double a, b;
  };
  for (const Scalar &scalar : std::initializer_list<Scalar>{
           {"dispersionEnergy", a.dispersionEnergy, b.dispersionEnergy},
           {"dispersionVirial", a.dispersionVirial, b.dispersionVirial},
           {"dispersionFixedEnergy", a.dispersionFixedEnergy,
            b.dispersionFixedEnergy},
           {"coulombConstantEnergy", a.coulombConstantEnergy,
            b.coulombConstantEnergy},
           {"coulombConstantVirial", a.coulombConstantVirial,
            b.coulombConstantVirial},
           {"coulombSelfEnergy", a.coulombSelfEnergy, b.coulombSelfEnergy},
           {"gradientShiftEnergy", a.gradientShiftEnergy, b.gradientShiftEnergy},
           {"gradientShiftFixedEnergy", a.gradientShiftFixedEnergy,
            b.gradientShiftFixedEnergy},
           {"pmeBeta", a.pmeBeta, b.pmeBeta},
           {"ljpmeSelfEnergy", a.ljpmeSelfEnergy, b.ljpmeSelfEnergy},
           {"ljpmeBeta", a.ljpmeBeta, b.ljpmeBeta},
           {"skin", a.skin, b.skin},
           {"pruneSkin", a.pruneSkin, b.pruneSkin},
           {"orderWidth", a.orderWidth, b.orderWidth}})
    if (!sameBits(scalar.a, scalar.b))
      return std::string("the value '") + scalar.name + "'";
  struct List {
    const char *name;
    const std::vector<double> &a, &b;
  };
  for (const List &list : std::initializer_list<List>{
           {"stateFixedEnergies", a.stateFixedEnergies, b.stateFixedEnergies},
           {"stateVolumeEnergies", a.stateVolumeEnergies, b.stateVolumeEnergies},
           {"lambdaFixedDerivatives", a.lambdaFixedDerivatives,
            b.lambdaFixedDerivatives},
           {"lambdaVolumeDerivatives", a.lambdaVolumeDerivatives,
            b.lambdaVolumeDerivatives},
           {"observableVolumeConstants", a.observableVolumeConstants,
            b.observableVolumeConstants},
           {"observableFixedConstants", a.observableFixedConstants,
            b.observableFixedConstants},
           {"gradientChargeFixed", a.gradientChargeFixed, b.gradientChargeFixed},
           {"gradientChargeVolume", a.gradientChargeVolume,
            b.gradientChargeVolume}})
    if (!sameBits(list.a, list.b))
      return std::string("the values '") + list.name + "'";
  if (a.gradientSlots.size() != b.gradientSlots.size())
    return "the number of the derivatives in the tunable constants";
  for (size_t k = 0; k != a.gradientSlots.size(); ++k) {
    const Program::GradientSlot &x = a.gradientSlots[k], &y = b.gradientSlots[k];
    if (x.tunable != y.tunable || x.site != y.site ||
        !sameBits(x.volumeConstant, y.volumeConstant) ||
        !sameBits(x.fixedConstant, y.fixedConstant))
      return "the derivative in the tunable constant " + std::to_string(k);
  }
  if (a.gradientFields.size() != b.gradientFields.size())
    return "the number of the fields of the derivatives";
  for (size_t k = 0; k != a.gradientFields.size(); ++k) {
    const Program::GradientField &x = a.gradientFields[k],
                                 &y = b.gradientFields[k];
    bool same = x.tunable == y.tunable && sameBits(x.sites, y.sites) &&
                x.rows.size() == y.rows.size();
    for (size_t r = 0; same && r != x.rows.size(); ++r)
      same = x.rows[r].site == y.rows[r].site &&
             sameBits(x.rows[r].scale, y.rows[r].scale) &&
             sameBits(x.rows[r].volumeConstant, y.rows[r].volumeConstant) &&
             sameBits(x.rows[r].fixedConstant, y.rows[r].fixedConstant);
    if (!same)
      return "the field " + std::to_string(k) + " of the derivatives";
  }
  if (a.startValues.size() != b.startValues.size())
    return "the number of the start values";
  for (size_t k = 0; k != a.startValues.size(); ++k)
    if (a.startValues[k].name != b.startValues[k].name ||
        !sameBits(a.startValues[k].value, b.startValues[k].value))
      return "the start value '" + a.startValues[k].name + "'";
  return "";
}

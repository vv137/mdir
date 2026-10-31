// The programs of a step, the kinetic energies, the noise, the couplings
// and the barostat, and the program of the minimization
// (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"
#include "PotentialIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "BarostatIR.h"
#include "DynamicsIR.h"
#include "CouplingIR.h"
#include "NoiseIR.h"
#include "KineticIR.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

/// The kinetic energy of the velocities `velocities` along each axis,
/// Σ m v ⊙ v / 2, a vector of three.
void Builder::emitKineticVector(StringRef result, StringRef velocities,
                                  StringRef masses, StringRef indent) {
  emitIRFragment(
      indent, {result}, {"vector<3xf64>"}, {velocities, masses}, {"!vec", "!real"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildKineticVector(builder, values[0], values[1])};
      });
}

void Builder::emitKineticEnergy(StringRef result, StringRef velocities,
                                  StringRef masses, StringRef indent) {
  emitIRFragment(
      indent, {result}, {"f64"}, {velocities, masses}, {"!vec", "!real"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildKineticEnergy(builder, values[0], values[1])};
      });
}

llvm::Error Builder::emitPrograms() {
  if (control.minimize) {
    emitDescend();
    return llvm::Error::success();
  }
  // Virtual sites are placed after the positions move, and the forces on
  // them are moved to their atoms before the velocities do. Rigid waters
  // are constrained before the sites are placed.
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();

  // Velocity Verlet (Swope et al., J. Chem. Phys. 76, 637 (1982)) stores
  // the velocities of the time of the positions; leapfrog, those half a
  // step behind (dynamics.md, Section 6). Both carry the forces of the
  // positions, so that a step evaluates once, at its end. The step of
  // energy returns the energy and the virial of the new positions, and
  // with leapfrog also the velocities of their time, which the energies
  // and the barostat take.
  //
  // With the barostat of Trotter type (D92), the last step of a period of
  // coupling, `step_trotter`, scales the cell in the middle of its drift
  // by the factor `%mu`: it drifts half, scales the positions by μ (each
  // rigid group with its center of mass) and the velocities by 1/μ, and
  // drifts the other half ([Bernetti2020], SI Sec. V.C, eq. S12a-d; its
  // eq. S13a, which writes the four as one, leaves out the scaling of
  // q(t) and has Δt for Δt/2). Leapfrog drifts with the velocities of the
  // middle of the step as velocity Verlet does, so the scaling is the same.
  // It returns as the step of energy does, and the kinetic energy of the
  // velocities that it scaled.
  bool leapfrog = isLeapfrog();
  bool trotter = usesTrotter();
  // The steps of the barostat of Trotter type need the virial, and the
  // energy only where the log takes it: `step_virial` gives the pressure of
  // the strain, `step_trotter` scales, `step_trotter_energy` scales at the
  // end of an interval between energies.
  // With constraints, the steps whose rows the log writes measure the half
  // steps (`measures`); the barostat that closes a period without scaling
  // within the drift then takes `step_coupling`, a step of energy that
  // does not.
  struct Kind {
    const char *name;
    bool energy, virial, scales, measures;
  };
  llvm::SmallVector<Kind> kinds = {{"step", false, false, false, false},
                                   {"step_energy", true, true, false, true}};
  // The thermostats control the optimal temperature (D263):
  // with constraints the step that closes a period of coupling measures
  // the half steps as well.
  bool coupledHalves = measuresCoupling();
  if (trotter) {
    if (!scalesEveryStep())
      kinds.push_back({"step_virial", false, true, false, false});
    kinds.push_back(
        {"step_trotter", false, countsAfterScaling(), true, coupledHalves});
    kinds.push_back({"step_trotter_energy", true, true, true, true});
  } else if (control.barostat && measuresHalfSteps()) {
    kinds.push_back({"step_coupling", true, true, false, coupledHalves});
  } else if (coupledHalves) {
    kinds.push_back({"step_thermostat", false, false, false, true});
  }
  for (const Kind &kind : kinds) {
    bool withEnergy = kind.energy, withVirial = kind.virial;
    bool scales = kind.scales;
    // The steps of the rows of the log, and those that close a period of
    // a thermostat, return the kinetic energy of each axis at the half
    // steps around their end, measured (D203, D263).
    bool measured = measuresHalfSteps() && kind.measures;
    // The steps with the virial also return the diagonal of the virial of
    // the groups, which the pressure (D264) and the
    // count of the work of a scaling of Trotter type (D116) take.
    // The steps of the rows of the log also report the rigid waters, the
    // solvent (D203).
    bool solvent = measured && withEnergy && reportsSolvent() &&
                   StringRef(kind.name) != "step_coupling";
    // Brownian dynamics (D163b) neither conserves a volume of phase space
    // nor reverses; it couples to the bath.
    bool brownian = control.isBrownian();
    // Leapfrog with constraints. A kick of a whole step leaves a part
    // along the constraints that the first half kick of velocity Verlet,
    // from the projected velocities of the time of the positions, does
    // not. The constraints of the positions take the drifts of both to
    // the same place, and the velocities of the change are then the same,
    // unless the velocities change within the drift: the friction of
    // Langevin dynamics, or the scaling of the barostat of Trotter type,
    // then acts on that part, and the stored velocities differ by
    // (1 - c)/2 or (1/mu - 1)/2 of it. There every program of a step
    // takes the form of velocity Verlet (`projectsState`). Elsewhere the
    // state of every program is that of the plain step, and a step with
    // the virial takes the drift of velocity Verlet on the side, for the
    // virial of the constraints alone (`sideVirial`). Either way a step
    // leaves the same state, to the bit, whatever else it returns.
    {
      auto &context = termIR.getContext();
      mlir::OpBuilder builder(&context);
      builder.setInsertionPointToEnd(termIR.getModule().getBody());
      auto types = buildParameterTypes(context, program);
      std::vector<DescentSiteIR> siteSets;
      for (const Program::TupleSet *selected : getSiteSets()) {
        unsigned slot = program.fields.size() + program.tables.size();
        for (const auto &set : program.tupleSets) {
          if (&set == selected) break;
          slot += 1 + set.fields.size();
        }
        auto fieldSlot = [&](StringRef name) {
          auto found = llvm::find_if(selected->fields, [&](const auto &field) { return field.name == name; });
          assert(found != selected->fields.end() && "site parameter must exist");
          return slot + 1 + unsigned(std::distance(selected->fields.begin(), found));
        };
        siteSets.push_back({slot, fieldSlot("a"), fieldSlot("b"), selected->name == "sites_amber",
                           spreadsSitesDerived(), "spread_" + selected->name});
      }
      auto relationSlot = [&](const Program::TupleSet &selected) {
        unsigned slot = program.fields.size() + program.tables.size();
        for (const auto &set : program.tupleSets) {
          if (&set == &selected) return slot;
          slot += 1 + set.fields.size();
        }
        llvm_unreachable("constraint set must belong to the program");
      };
      auto fieldSlot = [&](const Program::TupleSet &set, StringRef name) {
        auto found = llvm::find_if(set.fields, [&](const auto &field) { return field.name == name; });
        assert(found != set.fields.end() && "constraint parameter must exist");
        return relationSlot(set) + 1 + unsigned(std::distance(set.fields.begin(), found));
      };
      std::vector<DescentConstraintGroupIR> constraintGroups;
      ConstraintPositionConfig config{false, control.analyticBonds,
          control.precision == Precision::Double ? 1.0e-8 : 1.0e-4,
          control.precision == Precision::Double ? 1.0e-12 : 1.0e-6};
      if (settles) {
        const auto &set = *llvm::find_if(program.tupleSets, [](const auto &set) { return set.name == "settles"; });
        constraintGroups.push_back({relationSlot(set), 3, {fieldSlot(set, "doh"), fieldSlot(set, "dhh")},
            {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}}, config, true, control.precision == Precision::Double});
      }
      for (const Program::TupleSet *set : shakeSets) {
        DescentConstraintGroupIR group{relationSlot(*set), set->arity, {}, {}, config, false, false};
        for (unsigned member = 1; member != set->arity; ++member) {
          group.parameters.push_back(fieldSlot(*set, "d" + std::to_string(member)));
          group.bonds.push_back({0, member, member - 1});
        }
        constraintGroups.push_back(std::move(group));
      }
      StepNoiseIR noise;
      noise.active = control.isLangevin() || brownian;
      noise.brownian = brownian;
      noise.friction = control.friction;
      double kT = units::boltzmann * control.temperature;
      if (brownian) {
        noise.parameters = {control.seed, 1.0,
            std::sqrt(2.0 * kT / (control.friction * control.timestep))};
      } else if (control.isLangevin()) {
        double damping = std::exp(-control.friction * control.timestep);
        noise.parameters = {control.seed, damping, std::sqrt((1.0 - damping * damping) * kT)};
      }
      auto nativeProgram = buildNativeStep(
          builder, {kind.name, withEnergy, withVirial, scales, measured, solvent}, types, leapfrog, control.usesTime,
          siteSets, scalesReference(), hasHostEvaluation(),
          [&](mlir::OpBuilder &at, const PotentialStateIR &state, EnergyResults evaluation,
              mlir::Value scale) {
            auto field = [&](StringRef name) -> mlir::Value {
              auto found = llvm::find_if(program.fields, [&](const auto &field) { return field.name == name; });
              assert(found != program.fields.end() && "evaluation field must exist");
              return state.fields[std::distance(program.fields.begin(), found)];
            };
            for (const TermBuilder *term : getTermBuilders())
              if (const auto *role = term->getRoles().postEvaluation;
                  role && role->hasForceCorrection(*this)) {
                auto result = role->build(*this, at,
                    {state.positions, state.cell, evaluation.forces, evaluation.energy,
                     evaluation.virial, scale, field});
                evaluation = {result.energy, result.forces, result.virial};
              }
            for (const TermBuilder *term : getTermBuilders())
              if (const auto *role = term->getRoles().hostEvaluation;
                  role && role->applies(*this)) {
                auto result = role->build(*this, at,
                    {state.positions, state.cell, evaluation.forces, evaluation.energy,
                     evaluation.virial, scale, field});
                evaluation = {result.energy, result.forces, result.virial};
              }
            return evaluation;
          }, noise, constraintGroups, testsConstraintsApart());
      if (!nativeProgram) return nativeProgram.takeError();
      auto ownedProgram = *nativeProgram;
      emitIRFragment("", {}, {}, {}, {},
          [ownedProgram](mlir::OpBuilder &where, mlir::ValueRange) -> TermIRValues {
            ownedProgram->moveBefore(where.getInsertionBlock(), where.getInsertionPoint());
            return {};
          });
      continue;
    }
  }
  return llvm::Error::success();
}

std::string Builder::emitNoiseValue(StringRef indent) {
  if (!needsStepNumber())
    return "";
  std::string name = "%noise" + std::to_string(numNoiseSteps++);
  emitIRFragment(
      indent, {name}, {"i64"}, {"%noise_memory", "%c0", "%noise_one"},
      {"memref<1xi64>", "index", "i64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        Value last = memref::LoadOp::create(builder, loc, values[0], ValueRange{values[1]});
        Value next = arith::AddIOp::create(builder, loc, last, values[2]);
        memref::StoreOp::create(builder, loc, next, values[0], ValueRange{values[1]});
        return {next};
      });
  return name;
}

std::string Builder::emitMolecularDiagonal(StringRef indent,
                                           StringRef diagonal,
                                        StringRef positions, StringRef forces,
                                        StringRef cell, StringRef masses,
                                        StringRef relations, StringRef tag) {
  std::string result = ("%bgm" + tag).str();
  std::vector<std::string> relationNames, relationTypes;
  std::vector<unsigned> arities;
  for (const auto *group : getRigidGroups()) {
    relationNames.push_back((relations + group->name).str());
    relationTypes.push_back("!rel_" + group->name);
    arities.push_back(group->arity);
  }
  if (arities.empty())
    return diagonal.str();
  llvm::SmallVector<StringRef> inputs = {diagonal, positions, forces, cell, masses};
  llvm::SmallVector<StringRef> types = {"vector<3xf64>", "!vec", "!vec", "!md.cell", "!real"};
  for (size_t i = 0; i != arities.size(); ++i) {
    inputs.push_back(relationNames[i]);
    types.push_back(relationTypes[i]);
  }
  emitIRFragment(
      indent, {result}, {"vector<3xf64>"}, inputs, types,
      [arities](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        std::vector<RigidGroupIR> groups;
        for (auto [index, arity] : llvm::enumerate(arities))
          groups.push_back({values[5 + index], arity});
        MolecularDiagonalInputs input{values[0], values[1], values[2], values[3],
                                        values[4], groups};
        return {buildMolecularDiagonal(builder, input)};
      });
  return result;
}

Builder::GroupKinetic Builder::emitGroupKinetic(StringRef indent,
                                                StringRef positions,
                                                StringRef velocities,
                                                StringRef forces,
                                                StringRef tag) {
  std::string t = tag.str();
  std::string relations = ("%r" + StringRef(fieldPrefix).drop_front(2)).str();
  std::vector<std::string> relationNames, relationTypes;
  std::vector<unsigned> arities;
  for (const auto *group : getRigidGroups()) {
    relationNames.push_back(relations + group->name);
    relationTypes.push_back("!rel_" + group->name);
    arities.push_back(group->arity);
  }
  llvm::SmallVector<StringRef> inputs = {velocities, forces, massName, "%dt"};
  llvm::SmallVector<StringRef> types = {"!vec", "!vec", "!real", "f64"};
  bool grouped = !arities.empty();
  if (grouped) {
    inputs.append({positions, cellName});
    types.append({"!vec", "!md.cell"});
  }
  bool com = control.comPeriod > 0;
  if (com) {
    inputs.push_back("%total_mass");
    types.push_back("vector<3xf64>");
  }
  for (size_t i = 0; i != arities.size(); ++i) {
    inputs.push_back(relationNames[i]);
    types.push_back(relationTypes[i]);
  }
  GroupKinetic result{"%gka" + t, "%gft" + t, "%gkh" + t};
  emitIRFragment(
      indent, {result.kinetic, result.forceTerm, result.half},
      {"vector<3xf64>", "vector<3xf64>", "vector<3xf64>"}, inputs, types,
      [arities, com, grouped](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        size_t base = grouped ? 6 : 4;
        mlir::Value totalMass = com ? values[base++] : mlir::Value{};
        std::vector<RigidGroupIR> groups;
        for (auto [index, arity] : llvm::enumerate(arities))
          groups.push_back({values[base + index], arity});
        GroupKineticInputs input{grouped ? values[4] : mlir::Value{}, values[0],
                                   values[1], grouped ? values[5] : mlir::Value{},
                                   values[2], values[3], totalMass, groups};
        auto result = buildGroupKinetic(builder, input);
        return {result.kinetic, result.forceTerm, result.half};
      });
  return result;
}

std::string Builder::emitGroupScaling(StringRef indent, StringRef positions,
                                      StringRef mu, StringRef oneMinusMu,
                                      StringRef cell, StringRef masses,
                                      StringRef relations, StringRef tag) {
  std::string result = ("%xc" + tag).str();
  std::vector<std::string> relationNames, relationTypes;
  std::vector<unsigned> arities;
  for (const auto *group : getRigidGroups()) {
    relationNames.push_back((relations + group->name).str());
    relationTypes.push_back("!rel_" + group->name);
    arities.push_back(group->arity);
  }
  bool grouped = !arities.empty();
  llvm::SmallVector<StringRef> inputs = {positions, mu};
  llvm::SmallVector<StringRef> types = {"!vec", "vector<3xf64>"};
  if (grouped) {
    inputs.append({oneMinusMu, cell, masses});
    types.append({"vector<3xf64>", "!md.cell", "!real"});
  }
  for (size_t i = 0; i != arities.size(); ++i) {
    inputs.push_back(relationNames[i]);
    types.push_back(relationTypes[i]);
  }
  emitIRFragment(
      indent, {result}, {"!vec"}, inputs, types,
      [arities, grouped](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        std::vector<RigidGroupIR> groups;
        for (auto [index, arity] : llvm::enumerate(arities))
          groups.push_back({values[5 + index], arity});
        GroupScalingInputs input{values[0], values[1],
                                   grouped ? values[2] : mlir::Value{},
                                   grouped ? values[3] : mlir::Value{},
                                   grouped ? values[4] : mlir::Value{}, groups};
        return {buildGroupScaling(builder, input)};
      });
  return result;
}

void Builder::emitStrain(StringRef indent, StringRef kinetic,
                         StringRef diagonal, StringRef tag, StringRef step) {
  std::string t = tag.str();
  StrainCoupling coupling = control.anisotropic ? StrainCoupling::Anisotropic
      : control.semiIsotropic ? StrainCoupling::SemiIsotropic
                              : StrainCoupling::Isotropic;
  unsigned cellSize = getCellSize();
  std::vector<std::string> inputs = {
      kinetic.str(), diagonal.str(), "%box_memory", "%baro_constant", "%c_third",
      "%c_bar", "%c_unit3", "%seed", step.str(), "%baro_target", "%baro_kt", "%baro_rate"};
  std::vector<std::string> types = {
      "vector<3xf64>", "vector<3xf64>", getBoxMemoryType(), "f64", "f64", "f64",
      "vector<3xf64>", "i64", "i64", "f64", "f64", "f64"};
  for (unsigned k = 0; k != cellSize; ++k) {
    inputs.push_back("%c_edge" + std::to_string(k));
    types.push_back("index");
  }
  if (coupling == StrainCoupling::Anisotropic) {
    for (unsigned k = 0; k != 3; ++k) {
      inputs.push_back("%baro_beta_" + std::to_string(k));
      types.push_back("f64");
    }
  } else {
    inputs.push_back("%baro_beta");
    types.push_back("f64");
    if (coupling == StrainCoupling::SemiIsotropic) {
      inputs.insert(inputs.end(), {"%baro_beta_z", "%couple_half", "%baro_tension"});
      types.insert(types.end(), {"f64", "f64", "f64"});
    }
  }
  std::vector<std::string> outputs = {"%bv" + t, "%bwc" + t, "%bwcb" + t,
                                      "%bs3" + t, "%mu" + t, "%muinv" + t};
  std::vector<std::string> outputTypes = {"f64", "f64", "vector<3xf64>",
                                          "vector<3xf64>", "vector<3xf64>", "vector<3xf64>"};
  for (unsigned k = 0; k != cellSize; ++k) {
    outputs.push_back("%bn" + t + "_" + std::to_string(k));
    outputTypes.push_back("f64");
  }
  llvm::SmallVector<StringRef> in(inputs.begin(), inputs.end()),
      inTypes(types.begin(), types.end()), out(outputs.begin(), outputs.end()),
      outTypes(outputTypes.begin(), outputTypes.end());
  // The same pressure/noise law on each supported cell coupling, retaining
  // the stored H diag(mu) column order [Bernetti2020], D119 and D163c.
  emitIRFragment(
      indent, out, outTypes, in, inTypes,
      [coupling, cellSize](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        size_t beta = 12 + cellSize;
        std::array<mlir::Value, 3> betas{values[beta], {}, {}};
        mlir::Value half, tension;
        if (coupling == StrainCoupling::Anisotropic) {
          betas[1] = values[beta + 1];
          betas[2] = values[beta + 2];
        } else if (coupling == StrainCoupling::SemiIsotropic) {
          betas[2] = values[beta + 1];
          half = values[beta + 2];
          tension = values[beta + 3];
        }
        StrainIRInputs input{values[0], values[1], values[2], values.slice(12, cellSize),
            values[3], values[4], values[5], values[6], values[7], values[8],
            values[9], values[10], values[11], betas, half, tension};
        auto result = buildBarostatStrain(builder, input, coupling);
        TermIRValues outputs{result.volume, result.constantTrace, result.constantDiagonal,
                               result.logScale, result.scale, result.inverse};
        llvm::append_range(outputs, result.edges);
        return outputs;
      });
}

Builder::TrotterScaling Builder::emitTrotterStrain(
    StringRef indent, StringRef positions, StringRef velocities,
    StringRef forces, StringRef groups, StringRef tag, StringRef step,
    StringRef givenKinetic) {
  std::string t = tag.str();
  // The kinetic energy of the pressure, axis by axis: that of the centers
  // of the groups at the half steps, without the center of mass.
  std::string kinetic = givenKinetic.str();
  if (kinetic.empty())
    kinetic =
        emitGroupKinetic(indent, positions, velocities, forces, "t" + t).half;
  emitStrain(indent, kinetic, groups, t, step);
  return finishTrotterStrain(indent, groups, t);
}

std::string Builder::emitKineticWithoutCenter(StringRef indent,
                                              StringRef velocities,
                                              StringRef masses,
                                              StringRef tag) {
  std::string t = tag.str();
  std::string kinetic = "%tk" + t;
  emitKineticVector(kinetic, velocities, masses, indent);
  if (control.comPeriod > 0) {
    std::string result = "%tkt" + t;
    emitIRFragment(
        indent, {result}, {"vector<3xf64>"},
        {kinetic, velocities, masses, "%total_mass", "%c_half3"},
        {"vector<3xf64>", "!vec", "!real", "vector<3xf64>", "vector<3xf64>"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value moment = buildMomentum(builder, values[1], values[2]);
          mlir::Value center = mlir::arith::DivFOp::create(builder, loc, moment, values[3]);
          mlir::Value product = mlir::arith::MulFOp::create(builder, loc, moment, center);
          mlir::Value com = mlir::arith::MulFOp::create(builder, loc, values[4], product);
          return {mlir::arith::SubFOp::create(builder, loc, values[0], com)};
        });
    kinetic = result;
  }
  return kinetic;
}

void Builder::emitStoreTrotterState(StringRef indent, StringRef diagonal,
                                    StringRef groups, StringRef kinetic) {
  // Checkpoints keep the host copy of the same nine values.
  emitIRFragment(
      indent, {}, {}, {"%trotter_memory", diagonal, groups, kinetic},
      {"memref<9xf64>", "vector<3xf64>", "vector<3xf64>", "vector<3xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        buildStoreBarostatState(builder, values[0], values[1], values[2], values[3]);
        return {};
      });
}

Builder::TrotterScaling Builder::finishTrotterStrain(StringRef indent,
                                                     StringRef groups,
                                                     StringRef tag) {
  std::string t = tag.str();
  TrotterScaling scaling;
  scaling.mu = "%mu" + t;
  scaling.muinv = "%muinv" + t;
  scaling.logMu = "%bs3" + t;
  scaling.workBefore = "%tw" + t;
  scaling.newVolume = "%tvn" + t;
  std::string edge0 = "%bn" + t + "_0", edge1 = "%bn" + t + "_1",
              edge2 = "%bn" + t + "_2", constantDiagonal = "%bwcb" + t;
  emitIRFragment(
      indent, {scaling.workBefore, scaling.newVolume}, {"vector<3xf64>", "f64"},
      {groups, constantDiagonal, edge0, edge1, edge2},
      {"vector<3xf64>", "vector<3xf64>", "f64", "f64", "f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        mlir::Value work = mlir::arith::AddFOp::create(builder, loc, values[0], values[1]);
        mlir::Value xy = mlir::arith::MulFOp::create(builder, loc, values[2], values[3]);
        mlir::Value volume = mlir::arith::MulFOp::create(builder, loc, xy, values[4]);
        return {work, volume};
      });
  scaling.cell = "%tcell" + t;
  emitCellOf(indent, scaling.cell, "%bn" + t);
  if (scalesReference()) {
    scaling.scale = "%tscale" + t;
    emitReferenceScale(indent, scaling.scale, "%bn" + t + "_0",
                       "%bn" + t + "_1", "%bn" + t + "_2");
  }
  return scaling;
}

std::string Builder::emitKineticExcess(StringRef indent, StringRef tag,
                                       StringRef measured, StringRef forces,
                                       StringRef full) {
  std::string t = tag.str();
  std::string result = "%kex" + t;
  if (!measured.empty()) {
    emitKineticVector("%kexf" + t, full, massName, indent);
    std::string fullKinetic = "%kexf" + t;
    emitIRFragment(
        indent, {result}, {"vector<3xf64>"}, {measured, fullKinetic},
        {"vector<3xf64>", "vector<3xf64>"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {mlir::arith::SubFOp::create(builder, builder.getUnknownLoc(),
                                               values[0], values[1])};
        });
    return result;
  }
  // Under Langevin dynamics with constraints neither is measured; Brownian
  // dynamics has no momenta.
  if (hasConstraints() || control.isBrownian())
    return "";
  // Without constraints, K(v - dt F / 2m) and K(v + dt F / 2m) average to
  // K(v) + (dt^2 / 8) sum F^2 / m [Jung2018].
  emitIRFragment(
      indent, {result}, {"vector<3xf64>"}, {forces, massName, "%dt"},
      {"!vec", "!real", "f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildKineticExcess(builder, values[0], values[1], values[2])};
      });
  return result;
}

Builder::Coupled
Builder::emitCoupling(StringRef indent, StringRef positions,
                      StringRef velocities, StringRef forces,
                      StringRef energy, StringRef tag, StringRef step,
                      StringRef diagonal, StringRef groups,
                      StringRef optimal, const TrotterScaling *trotter) {
  std::string newForces = forces.str();
  bool removesMotion = control.comPeriod > 0;
  std::string t = tag.str();
  std::string momentum = "%pc" + t, center = "%vcm" + t, centerEnergy = "%kcm" + t;
  llvm::SmallVector<StringRef> centerInputs, centerTypes;
  llvm::SmallVector<StringRef> centerOutputs{centerEnergy}, centerOutputTypes{"f64"};
  if (removesMotion) {
    centerInputs = {velocities, massName, "%total_mass", "%couple_half"};
    centerTypes = {"!vec", "!real", "vector<3xf64>", "f64"};
    centerOutputs = {momentum, center, centerEnergy};
    centerOutputTypes = {"vector<3xf64>", "vector<3xf64>", "f64"};
  }
  emitIRFragment(
      indent, centerOutputs, centerOutputTypes, centerInputs, centerTypes,
      [removesMotion](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        CenterMotionInputs input;
        if (removesMotion)
          input = {values[0], values[1], values[2], values[3]};
        auto result = buildCenterMotion(builder, input);
        if (removesMotion)
          return {result.momentum, result.velocity, result.energy};
        return {result.energy};
      });
  // The energy that the coupling takes: that of the center of mass, and
  // what the thermostat takes from the rest.
  std::string bath = "%kcm" + t;
  std::string kineticAxes;
  if (control.barostat) {
    // The kinetic energy of each axis, which the pressure of each axis
    // takes, and of the center of mass; their sums for the thermostat.
    os << indent << "// The kinetic energy of each axis.\n";
    emitKineticVector("%kv" + t, velocities, massName, indent);
    emitSum3("%kc" + t, "%kv" + t, indent);
    std::string vector = "%kv" + t;
    if (removesMotion) {
      std::string result = "%ktv" + t, axes = "%kv" + t;
      emitIRFragment(
          indent, {result}, {"vector<3xf64>"}, {axes, momentum, center, "%c_half3"},
          {"vector<3xf64>", "vector<3xf64>", "vector<3xf64>", "vector<3xf64>"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildAxesWithoutCenter(builder, values[0], values[1], values[2], values[3])};
          });
      vector = "%ktv" + t;
    }
    kineticAxes = vector;
  }
  if (rescalesVelocities()) {
    std::string totalEnergy = "%kc" + t;
    if (!control.barostat)
      emitKineticEnergy(totalEnergy, velocities, massName, indent);
    bool noseHoover = control.isNoseHoover(), hasOptimal = !optimal.empty();
    llvm::SmallVector<StringRef> inputs{totalEnergy, step}, types{"f64", "i64"};
    if (removesMotion) {
      inputs.push_back(centerEnergy);
      types.push_back("f64");
    }
    if (hasOptimal) {
      inputs.push_back(optimal);
      types.push_back("f64");
    }
    if (!noseHoover) {
      inputs.append({"%seed", "%target_kinetic", "%freedom", "%decay"});
      types.append({"i64", "f64", "f64", "f64"});
    }
    std::string factor = "%alpha" + t, squared = "%alpha2" + t, heat = "%heat" + t;
    // The controller takes optimal temperature without COM [Jung2019];
    // its heat uses the actual kinetic energy of the velocities scaled.
    emitIRFragment(
        indent, {factor, squared, heat}, {"f64", "f64", "f64"}, inputs, types,
        [removesMotion, hasOptimal, noseHoover](mlir::OpBuilder &builder,
                                                mlir::ValueRange values) -> TermIRValues {
          size_t index = 2;
          mlir::Value centerEnergy = removesMotion ? values[index++] : mlir::Value{};
          mlir::Value optimal = hasOptimal ? values[index++] : mlir::Value{};
          ThermostatIRInputs input{values[0], centerEnergy, optimal, values[1],
              noseHoover ? mlir::Value{} : values[index],
              noseHoover ? mlir::Value{} : values[index + 1],
              noseHoover ? mlir::Value{} : values[index + 2],
              noseHoover ? mlir::Value{} : values[index + 3]};
          auto result = buildThermostat(builder, input, noseHoover);
          return {result.factor, result.factorSquared, result.heat};
        });
    bath = heat;
  }
  // Stochastic cell rescaling (Bernetti and Bussi 2020): the pressure of
  // the step, with the virials of the correction for the dispersion and of
  // the background of a net charge at this volume, gives the change ε of
  // the logarithm of the volume. The positions and the cell scale by
  // μ = exp(ε/3), the velocities by 1/μ (dynamics.md, Section 2.4).
  std::string newPositions = positions.str();
  if (control.barostat) {
    // The kinetic energy of the velocities of the step, without the center
    // of mass, whose change the scaling counts.
    std::string kinetic = kineticAxes;
    // The pressure of the groups that the barostat scales, with the kinetic
    // energy of their centers at the half steps, which the forces give
    // with or without constraints (D264).
    GroupKinetic centers;
    if (!trotter || scalesEveryStep())
      centers = emitGroupKinetic(indent, positions, velocities, forces,
                                 "c" + t);
    if (!trotter)
      emitStrain(indent, centers.half, groups, t, step);
    // The energy that the scaling gives the system, summed over the axes:
    // exactly in the velocities, (1/μ_a² − 1) K_a, and in the positions
    // either exactly, from the energy of the scaled positions (below), or
    // to first order, −(μ_a − 1) W_aa, with W the virial of the groups and
    // of the constant terms.
    bool exact = control.barostatWork == BarostatWork::Exact;
    // The kinetic energy of each axis after the thermostat.
    std::string after = kinetic;
    if (rescalesVelocities()) {
      std::string factor = "%alpha2" + t, scaled = "%knv" + t;
      emitIRFragment(
          indent, {scaled}, {"vector<3xf64>"}, {factor, kinetic}, {"f64", "vector<3xf64>"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildScaleKinetic(builder, values[0], values[1])};
          });
      after = "%knv" + t;
    }
    std::string relations =
        ("%r" + StringRef(fieldPrefix).drop_front(2)).str();
    if (trotter) {
      // The scaling was made within the drift of the step (Trotter type,
      // [Bernetti2020], SI Sec. V.C): the velocities of its middle by 1/μ_a
      // along axis a, which changes their kinetic energy by
      // Σ (1/μ_a² − 1) K_a, and the positions by μ_a, which changes the
      // potential energy by −Σ ln μ_a W_aa to first order, W the virial of
      // the groups, Σ X ⊙ F_group, with those of the constant terms. W is
      // taken as the mean of those of the evaluations before and after the
      // scaling, which makes the count exact to second order in the strain
      // (D92). The virial of the step with twice the internal kinetic
      // energy would not do: the constraints of the step that scales
      // straddle the scaling, and the count was biased by the square of the
      // strain (D116).
      // To first order, from the virial before the scaling only.
      if (!countsAfterScaling()) {
        std::string work = "%bwork" + t;
        emitIRFragment(
            indent, {work}, {"f64"}, {trotter->workBefore, trotter->logMu},
            {"vector<3xf64>", "vector<3xf64>"},
            [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
              return {buildVirialWork(builder, values[0], values[1])};
            });
      }
      std::string groupsAfter =
          countsAfterScaling() ? trotter->groupsAfter : "";
      // With a scaling every step, the state that the next one takes its
      // pressure from: the kinetic energy of the centers at the half steps
      // after the thermostat, which scales K_g by α² and leaves the term of
      // the forces.
      if (scalesEveryStep()) {
        std::string stored = centers.half;
        if (rescalesVelocities()) {
          std::string factor = "%alpha2" + t, result = "%gsth" + t;
          emitIRFragment(
              indent, {result}, {"vector<3xf64>"},
              {factor, centers.kinetic, centers.forceTerm},
              {"f64", "vector<3xf64>", "vector<3xf64>"},
              [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
                return {buildScaledGroupKinetic(builder, values[0], values[1], values[2])};
              });
          stored = "%gsth" + t;
        }
        emitStoreTrotterState(indent, diagonal, groupsAfter, stored);
      }
      if (countsAfterScaling()) {
        std::string work = "%bwork" + t;
        emitIRFragment(
            indent, {work}, {"f64"},
            {"%baro_constant", trotter->newVolume, "%c_third", groupsAfter,
             trotter->workBefore, "%c_half3", trotter->logMu},
            {"f64", "f64", "f64", "vector<3xf64>", "vector<3xf64>",
             "vector<3xf64>", "vector<3xf64>"},
            [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
              MeanVirialWorkInputs input{values[0], values[1], values[2], values[3],
                                           values[4], values[5], values[6]};
              return {buildMeanVirialWork(builder, input)};
            });
      }
      std::string work = "%bwork" + t, result = "%btake" + t;
      emitIRFragment(
          indent, {result}, {"f64"},
          {trotter->muinv, "%c_unit3", trotter->kineticHalf, work, bath},
          {"vector<3xf64>", "vector<3xf64>", "vector<3xf64>", "f64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildTrotterBath(builder, values[0], values[1], values[2], values[3], values[4])};
          });
      bath = "%btake" + t;
    }
    if (!trotter) {
      std::string result = "%bm1" + t, scale = "%mu" + t;
      emitIRFragment(
          indent, {result}, {"vector<3xf64>"}, {"%c_unit3", scale},
          {"vector<3xf64>", "vector<3xf64>"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildOneMinusScale(builder, values[0], values[1])};
          });
    }
    if (!exact && !trotter) {
      // To first order, isotropic only (Control): the trace of the virial
      // of the groups, a third to each axis.
      std::string work = "%bwork" + t, constant = "%bwc" + t, oneMinus = "%bm1" + t;
      emitIRFragment(
          indent, {work}, {"f64"}, {groups, constant, "%c_third", oneMinus},
          {"vector<3xf64>", "f64", "f64", "vector<3xf64>"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildFirstOrderVirialWork(builder, values[0], values[1], values[2], values[3])};
          });
    }
    if (!trotter) {
      std::string inverse = "%muinv" + t, change = "%bdk" + t;
      emitIRFragment(
          indent, {change}, {"f64"}, {inverse, "%c_unit3", after},
          {"vector<3xf64>", "vector<3xf64>", "vector<3xf64>"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildKineticScaleChange(builder, values[0], values[1], values[2])};
          });
    }
    if (!exact && !trotter) {
      std::string work = "%bwork" + t, change = "%bdk" + t, result = "%btake" + t;
      emitIRFragment(
          indent, {result}, {"f64"}, {work, change, bath}, {"f64", "f64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildBathAfterWork(builder, values[0], values[1], values[2])};
          });
      bath = "%btake" + t;
    }
    if (!trotter)
      newPositions = emitGroupScaling(indent, positions, "%mu" + t,
                                      "%bm1" + t, cellName, massName,
                                      relations, t);
    if (exact && !trotter) {
      // The scaled positions in the new cell, with the virtual sites placed
      // on their atoms, and their energy and forces, which the next step
      // takes: the change of the potential energy is counted exactly, and
      // no step begins with the forces of other positions (D77).
      std::string cell = "%bcell" + t;
      emitCellOf(indent, cell, "%bn" + t);
      std::string outerCell = cellName, outerScale = scaleName;
      cellName = cell;
      if (scalesReference()) {
        scaleName = "%bscale" + t;
        emitReferenceScale(indent, scaleName, "%bn" + t + "_0",
                           "%bn" + t + "_1", "%bn" + t + "_2");
      }
      if (hasSites()) {
        std::string placed = "%xcs" + t;
        emitPlaceSites(indent, newPositions, placed, relations);
        newPositions = placed;
      }
      std::string held = hasForceCorrections() ? "p" : "";
      std::string raw = hasSites() ? "e" : "";
      // The work takes the energy, and the next step the forces; the
      // virial of the scaled positions is not needed.
      std::string u = "%bu" + t, f = "%bf" + t;
      // The time of the step that the scaling follows (D145).
      std::string time = "%btime" + t;
      if (control.usesTime)
        emitIRFragment(
            indent, {time}, {"f64"}, {"%noise_memory", "%c0", "%dt"},
            {"memref<1xi64>", "index", "f64"},
            [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
              mlir::Value step = mlir::memref::LoadOp::create(
                  builder, builder.getUnknownLoc(), values[0], mlir::ValueRange{values[1]});
              return {buildSimulationTime(builder, step, values[2])};
            });
      // A learned potential adds its energy and forces in the new cell
      // (D270).
      bool hostTerms = hasHostEvaluation();
      std::string modelled = !hostTerms || !held.empty() ? "" : "c";
      emitEnergyEvaluation(indent, newPositions, cell, fieldPrefix, time,
                             u + held + modelled, f + held + raw + modelled, "");
      if (hasSites())
        emitSpreadSites(indent, newPositions, f + held + "e", f + held,
                        relations);
      if (hasForceCorrections())
        emitForceCorrections(indent, newPositions, fieldPrefix, f + "p",
                       hostTerms ? f + "c" : f, u + "p", hostTerms ? u + "c" : u);
      if (hostTerms)
        emitHostEvaluation(indent, newPositions, cell, fieldPrefix, f + "c", f,
                       u + "c", u, "", "b" + t);
      cellName = outerCell;
      scaleName = outerScale;
      newForces = f;
      // The constant terms of the energy, the correction for the
      // dispersion and the background of a net charge, are proportional to
      // 1 / V and not in `md.evaluate`: their change, C (1/V' - 1/V), is
      // part of the work too, or the conserved energy would carry
      // E_c(V) - E_c(V_0).
      std::string e0 = "%bn" + t + "_0", e1 = "%bn" + t + "_1", e2 = "%bn" + t + "_2",
                  volume = "%bv" + t, change = "%bdk" + t, result = "%btake" + t;
      emitIRFragment(
          indent, {result}, {"f64"},
          {e0, e1, e2, "%c_unit", volume, "%baro_energy_constant", u, energy, change, bath},
          {"f64", "f64", "f64", "f64", "f64", "f64", "f64", "f64", "f64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            ExactBathInputs input{values.take_front(3), values[3], values[4], values[5],
                                     values[6], values[7], values[8], values[9]};
            return {buildExactBath(builder, input)};
          });
      bath = "%btake" + t;
    }
  }
  bool thermal = rescalesVelocities(), barostatScale = control.barostat && !trotter;
  std::string factor = "%alpha" + t, inverse = "%muinv" + t, result = "%vc" + t;
  llvm::SmallVector<StringRef> inputs{velocities, bath}, types{"!vec", "f64"};
  if (removesMotion) {
    inputs.push_back(center);
    types.push_back("vector<3xf64>");
  }
  if (thermal) {
    inputs.push_back(factor);
    types.push_back("f64");
  }
  if (barostatScale) {
    inputs.push_back(inverse);
    types.push_back("vector<3xf64>");
  }
  emitIRFragment(
      indent, {result}, {"!vec"}, inputs, types,
      [removesMotion, thermal, barostatScale](mlir::OpBuilder &builder,
                                              mlir::ValueRange values) -> TermIRValues {
        size_t index = 2;
        mlir::Value center = removesMotion ? values[index++] : mlir::Value{};
        mlir::Value factor = thermal ? values[index++] : mlir::Value{};
        mlir::Value inverse = barostatScale ? values[index++] : mlir::Value{};
        return {buildCoupledVelocities(builder, values[0], center, factor, inverse, values[1])};
      });
  return {newPositions, "%vc" + t, newForces};
}

void Builder::emitDescend() {
  // The step moves each particle along g, its force over its mass without
  // the parts along the constraints, by `%h` times g over the 16-norm of
  // g. That norm is no less than the largest |g|, so that no particle
  // moves farther than `%h`, and it is taken of g over its root mean
  // square, which keeps the powers finite. The constraints weigh the
  // particles by their masses, so that the step stays downhill once they
  // take the groups back to their shapes. Virtual sites have no mass and
  // do not move.
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  {
    auto &context = termIR.getContext();
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToEnd(termIR.getModule().getBody());
    auto types = buildParameterTypes(context, program);
    auto relationSlot = [&](const Program::TupleSet &selected) {
      unsigned slot = program.fields.size() + program.tables.size();
      for (const auto &set : program.tupleSets) {
        if (&set == &selected) return slot;
        slot += 1 + set.fields.size();
      }
      llvm_unreachable("constraint set must belong to the program");
    };
    auto fieldSlot = [&](const Program::TupleSet &set, StringRef name) {
      auto found = llvm::find_if(set.fields, [&](const auto &field) { return field.name == name; });
      assert(found != set.fields.end() && "constraint parameter must exist");
      return relationSlot(set) + 1 + unsigned(std::distance(set.fields.begin(), found));
    };
    std::vector<DescentConstraintGroupIR> groups;
    ConstraintPositionConfig config{false, control.analyticBonds,
        control.precision == Precision::Double ? 1.0e-8 : 1.0e-4,
        control.precision == Precision::Double ? 1.0e-12 : 1.0e-6};
    if (settles) {
      const auto &set = *llvm::find_if(program.tupleSets, [](const auto &set) { return set.name == "settles"; });
      groups.push_back({relationSlot(set), 3, {fieldSlot(set, "doh"), fieldSlot(set, "dhh")},
          {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}}, config, true, control.precision == Precision::Double});
    }
    for (const Program::TupleSet *set : shakeSets) {
      DescentConstraintGroupIR group{relationSlot(*set), set->arity, {}, {}, config, false, false};
      for (unsigned member = 1; member != set->arity; ++member) {
        group.parameters.push_back(fieldSlot(*set, "d" + std::to_string(member)));
        group.bonds.push_back({0, member, member - 1});
      }
      groups.push_back(std::move(group));
    }
    std::vector<DescentSiteIR> siteSets;
    for (const Program::TupleSet *set : getSiteSets())
      siteSets.push_back({relationSlot(*set), fieldSlot(*set, "a"), fieldSlot(*set, "b"),
                         set->name == "sites_amber", spreadsSitesDerived(), "spread_" + set->name});
    auto nativeProgram = buildNativeDescent(
        builder, types, static_cast<double>(system.getNumParticles()), control.usesTime,
        groups, siteSets, hasHostEvaluation(),
        [&](mlir::OpBuilder &at, const PotentialStateIR &state, EnergyResults evaluation) {
          auto field = [&](StringRef name) -> mlir::Value {
            auto found = llvm::find_if(program.fields, [&](const auto &field) { return field.name == name; });
            assert(found != program.fields.end() && "evaluation field must exist");
            return state.fields[std::distance(program.fields.begin(), found)];
          };
          mlir::Value virial;
          if (hasForceCorrections()) {
            auto vec = mlir::VectorType::get({9}, at.getF64Type());
            virial = mlir::arith::ConstantOp::create(at, at.getUnknownLoc(),
                mlir::DenseElementsAttr::get(vec, 0.0));
          }
          for (const TermBuilder *term : getTermBuilders())
            if (const auto *role = term->getRoles().postEvaluation;
                role && role->hasForceCorrection(*this)) {
              auto result = role->build(*this, at,
                  {state.positions, state.cell, evaluation.forces, evaluation.energy, virial, {}, field});
              evaluation.forces = result.forces;
              evaluation.energy = result.energy;
              virial = result.virial;
            }
          for (const TermBuilder *term : getTermBuilders())
            if (const auto *role = term->getRoles().hostEvaluation; role && role->applies(*this)) {
              auto result = role->build(*this, at,
                  {state.positions, state.cell, evaluation.forces, evaluation.energy, {}, {}, field});
              evaluation.forces = result.forces;
              evaluation.energy = result.energy;
            }
          return evaluation;
        });
    // The module owns this complete native body. Only its top-level placement
    // crosses the temporary join; no argument or operation in it is reparsed.
    emitIRFragment("", {}, {}, {}, {},
        [nativeProgram](mlir::OpBuilder &where, mlir::ValueRange) -> TermIRValues {
          nativeProgram->moveBefore(where.getInsertionBlock(), where.getInsertionPoint());
          return {};
        });
    return;
  }

}

void Builder::emitCellOf(StringRef indent, StringRef name, StringRef prefix) {
  std::vector<std::string> edges;
  for (int k = 0; k != getCellSize(); ++k)
    edges.push_back((prefix + "_" + std::to_string(k)).str());
  llvm::SmallVector<StringRef> inputs(edges.begin(), edges.end());
  llvm::SmallVector<StringRef> types(edges.size(), "f64");
  emitIRFragment(
      indent, {name}, {"!md.cell"}, inputs, types,
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildCell(builder, values)};
      });
}

void Builder::emitReferenceScale(StringRef indent, StringRef result,
                                  StringRef e0, StringRef e1, StringRef e2) {
  emitIRFragment(
      indent, {result}, {"vector<3xf64>"}, {e0, e1, e2, "%rest_edges"},
      {"f64", "f64", "f64", "vector<3xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildReferenceScale(builder, values.take_front(3), values[3])};
      });
}

Builder::DiagonalOutputs Builder::emitDiagonal(
    StringRef result, StringRef virial, StringRef indent) {
  DiagonalOutputs outputs{result.str(), (result + "_0").str(),
                            (result + "_4").str(), (result + "_8").str()};
  emitIRFragment(
      indent, {outputs.vector, outputs.xx, outputs.yy, outputs.zz},
      {"vector<3xf64>", "f64", "f64", "f64"}, {virial}, {"vector<9xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto diagonal = buildVirialDiagonal(builder, values[0]);
        return {diagonal.vector, diagonal.xx, diagonal.yy, diagonal.zz};
      });
  return outputs;
}

void Builder::emitTrace(StringRef result, StringRef virial, StringRef indent) {
  emitIRFragment(
      indent, {result}, {"f64"}, {virial}, {"vector<9xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildVirialTrace(builder, values[0])};
      });
}

void Builder::emitSum3(StringRef result, StringRef vector, StringRef indent) {
  emitIRFragment(
      indent, {result}, {"f64"}, {vector}, {"vector<3xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildSum3(builder, values[0])};
      });
}

void Builder::emitEnergyEvaluation(
    StringRef indent, StringRef x, StringRef cell, StringRef prefix, StringRef time,
    StringRef energy, StringRef forces, StringRef virial) {
  PotentialConnections input(program, x, cell, prefix, time, control.usesTime);
  EnergyRequests request{!energy.empty(), !forces.empty(), !virial.empty()};
  llvm::SmallVector<StringRef> outputs, types;
  if (request.energy) { outputs.push_back(energy); types.push_back("f64"); }
  if (request.forces) { outputs.push_back(forces); types.push_back("!vec"); }
  if (request.virial) { outputs.push_back(virial); types.push_back("vector<9xf64>"); }
  emitIRFragment(
      indent, outputs, types, input.names(), input.signatures(),
      [request, fields = input.fields, timed = input.timed](
          mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        PotentialStateIR state{values[0], values[1], values.slice(2, fields),
                                 timed ? values[2 + fields] : mlir::Value{}};
        auto result = buildEnergyEvaluation(builder, state, request);
        TermIRValues outputs;
        if (request.energy) outputs.push_back(result.energy);
        if (request.forces) outputs.push_back(result.forces);
        if (request.virial) outputs.push_back(result.virial);
        return outputs;
      });
}

void Builder::emitKick(StringRef indent, StringRef result, StringRef velocity,
                         StringRef force, StringRef mass, StringRef interval) {
  emitIRFragment(indent, {result}, {"!vec"}, {velocity, force, mass, interval},
      {"!vec", "!vec", "!real", "f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildKick(builder, values[0], values[1], values[2], values[3])};
      });
}

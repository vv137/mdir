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
          getConstraintPositionTolerance(control.precision == Precision::Double),
          control.precision == Precision::Double ? 1.0e-12 : 1.0e-6};
      if (settles) {
        const auto &set = *llvm::find_if(program.tupleSets, [](const auto &set) { return set.name == "settles"; });
        constraintGroups.push_back({relationSlot(set), 3, {fieldSlot(set, "doh"), fieldSlot(set, "dhh")},
            {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}}, config, true, control.precision == Precision::Double});
        if (control.waterTemplate == WaterTemplateChoice::On)
          constraintGroups.back().position.templateWaterEpoch = true;
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

      continue;
    }
  }
  return llvm::Error::success();
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
        getConstraintPositionTolerance(control.precision == Precision::Double),
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
    buildNativeDescent(
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
    // The complete native body stays in the module that owns it.

    return;
  }

}

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

void Builder::emitPrograms() {
  if (control.minimize) {
    emitDescend();
    return;
  }
  // Virtual sites are placed after the positions move, and the forces on
  // them are moved to their atoms before the velocities do. Rigid waters
  // are constrained before the sites are placed.
  bool sites = hasSites();
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  bool constraints = settles || !shakeSets.empty();

  // With restraints the names of an evaluation have `p`, and the
  // restraints give those that the step goes on with.
  std::string held = hasForceCorrections() ? "p" : "";
  // The velocities that leave none along a constrained bond (the second
  // half of RATTLE), with the virial of the impulses if `virial` is given.
  // Returns the velocities and the virial.
  auto project = [&](StringRef x, StringRef input, StringRef base,
                     std::string virial) {
    std::string current = input.str();
    unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
    auto next = [&]() {
      ++step;
      return step == steps ? base.str()
                           : (base + "c" + std::to_string(step)).str();
    };
    if (settles) {
      std::string result = next();
      std::string w = step == steps ? "%w1c" : "%w1c1";
      std::string sum = emitSettleVelocities("  ", x, current, result,
                                             virial, w);
      if (!virial.empty())
        virial = sum;
      current = result;
    }
    for (const Program::TupleSet *set : shakeSets) {
      std::string result = next();
      std::string w = "%w1s_" + set->name;
      std::string sum = emitShakeVelocities("  ", x, current, *set, result,
                                            virial, w);
      if (!virial.empty())
        virial = sum;
      current = result;
    }
    return std::make_pair(current, virial);
  };

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
    bool returnsCurrent = leapfrog && (withVirial || scales || measured);
    // The steps with the virial also return the diagonal of the virial of
    // the groups, which the pressure (D264) and the
    // count of the work of a scaling of Trotter type (D116) take.
    bool molecular = withVirial && !control.isBrownian();
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
    bool projectsState =
        leapfrog && constraints && (control.isLangevin() || scales);
    bool sideVirial = leapfrog && constraints && withVirial && !projectsState;
    std::string effects =
        constraints ? "{read = [\"dyn.stop\"], write = [\"dyn.stop\"]}" : "{}";
    if (hasHostEvaluation())
      effects = constraints
                    ? "{read = [\"dyn.stop\", \"<Default>\"], write = "
                      "[\"dyn.stop\", \"<Default>\"]}"
                    : "{read = [\"<Default>\"], write = [\"<Default>\"]}";
    os << "dyn.program @" << kind.name
       << "(%x: !vec, %v: !vec, %f: !vec, %m: !real,\n"
       << "    %cell: !md.cell, %dt: f64"
       << (scales ? ", %mu: vector<3xf64>" : "") << getScaleParameter()
       << getFieldParameters() << getNoiseParameter() << ")\n"
       << "    -> (!vec, !vec, !vec" << (withEnergy ? ", f64" : "")
       << (withVirial ? ", vector<9xf64>" : "")
       << (molecular ? ", vector<3xf64>" : "")
       << (returnsCurrent ? ", !vec" : "") << (scales ? ", vector<3xf64>" : "")
       << (measured ? ", vector<3xf64>" : "")
       << (solvent ? ", vector<3xf64>" : "") << ")\n"
       << "    attributes {"
       << (leapfrog ? "velocity_offset = -0.5,\n                " : "")
       // The effects of the body, which dyn.step takes (D355):
       // the deferred stops and the hosted evaluation's ordinary memory.
       << "effects = " << effects << ",\n                "
       << (brownian ? "provides = [\"thermostatting\"]} {\n"
                    : "provides = [\"symplectic\", \"time_reversible\"]} {\n");
    emitIRFragment(
        "  ", {"%half"}, {"f64"}, {"%dt"}, {"f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value half = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.5));
          return {mlir::arith::MulFOp::create(builder, loc, half, values[0])};
        });
    // With constraints, a build of the neighbor structure in the step that
    // finds positions that are not numbers records its stop, and the step
    // runs to its end: a failed set of constraints then stops the run with
    // its own message, and the recorded stop follows the tests otherwise
    // (D278, the maintainer's ruling on PR #381).
    if (constraints)
      emitIRFragment(
          "  ", {}, {}, {}, {},
          [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
            mlir::func::CallOp::create(builder, builder.getUnknownLoc(), "mdrtDeferStops",
                                         mlir::TypeRange{}, mlir::ValueRange{});
            return {};
          });
    // The time at the end of the step, at which its forces are evaluated
    // (D145).
    if (control.usesTime)
      emitIRFragment(
          "  ", {"%time"}, {"f64"}, {"%noise_step", "%dt"}, {"i64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildSimulationTime(builder, values[0], values[1])};
          });
    if (brownian) {
      // The velocity of the step, which the drift takes over Δt; the
      // velocities stored are those of the displacement, Δx / Δt.
      emitBrownianVelocity("  ", "%f", "%v1");
    } else if (!leapfrog) {
      emitKick("  ", "%v1", "%v", "%f", "%m", "%half");
    } else if (projectsState) {
      // The velocities of the time of the positions, as velocity Verlet
      // has them, with none along the constraints, and its first half
      // kick: the friction or the scaling of the middle of the drift
      // would act on what a kick of a whole step leaves along the
      // constraints.
      emitKick("  ", "%v0u", "%v", "%f", "%m", "%half");
      std::string current = project("%x", "%v0u", "%v0", "").first;
      emitKick("  ", "%v1", current, "%f", "%m", "%half");
    } else {
      emitKick("  ", "%v1", "%v", "%f", "%m", "%dt");
      if (sideVirial) {
        // The first half kick of velocity Verlet, from the velocities of
        // the time of the positions with none along the constraints: its
        // drift, `%x1dp` below, ends where the constraints give the
        // virial of the first half of the step (D76). The constraints
        // bring both drifts to the same place, and the state is that of
        // the plain step: nothing of this feeds it.
        emitKick("  ", "%v0u", "%v", "%f", "%m", "%half");
        std::string current = project("%x", "%v0u", "%v0", "").first;
        emitKick("  ", "%v1p", current, "%f", "%m", "%half");
      }
    }
    std::string drifting = "%v1";
    std::string drifted = sites || constraints ? "%x1d" : "%x1";
    // The positions that the constraints of the drift below take as their
    // reference, and the length of that drift.
    std::string reference = "%x", length = "%dt";
    // Langevin dynamics with constraints: the g-BAOAB step of
    // [Leimkuhler2016] (its Appendix, one drift in each A), which projects
    // the velocities on the constraints after each part of the step
    // (D265). B: the half kick, projected. A: a drift of
    // Δt/2, the constraints of the positions with those before the drift as
    // the reference, the velocities of the displacement, projected at the
    // new positions. O: the friction and the noise, projected. Under the
    // barostat of Trotter type the scaling follows. A: as the first, by the
    // block of the constraints below. Then the forces and the half kick.
    bool geodesic = constraints && control.isLangevin();
    if (geodesic) {
      std::string kicked = project("%x", "%v1", "%vgk", "").first;
      emitDrift("  ", "%xga", "%x", kicked, "%half");
      std::string current = "%xga";
      unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
      auto next = [&]() {
        ++step;
        return step == steps ? std::string("%xgc")
                             : "%xgc" + std::to_string(step);
      };
      if (settles) {
        std::string result = next();
        emitSettlePositions("  ", "%x", current, "%dxg", result);
        current = result;
      }
      for (const Program::TupleSet *set : shakeSets) {
        std::string result = next();
        emitShakePositions("  ", "%x", current, *set, result);
        current = result;
      }
      emitIRFragment(
          "  ", {"%vga"}, {"!vec"}, {kicked, current, "%xga", "%half"},
          {"!vec", "!vec", "!vec", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildCorrectedHalfVelocity(builder, values[0], values[1], values[2], values[3])};
          });
      // The projection of the A at the new positions and that after O are
      // one: the projection is linear at fixed positions, so that of
      // c v + σ R is c times that of v plus that of σ R, and the projection
      // after O alone gives what both give.
      emitLangevin("  ", "%vga", "%vgo");
      std::string thermal = project("%xgc", "%vgo", "%vgp", "").first;
      reference = "%xgc";
      drifting = thermal;
      if (scales) {
        // The scale of each axis, μ (D119): each rigid group with its
        // center, which keeps its shape, and the velocities by 1/μ; the
        // constraints of the second A take back what an anisotropic
        // scaling leaves along them.
        emitDriftScale("  ");
        reference = emitGroupScaling("  ", "%xgc", "%mu", "%mu_m1", "%cell",
                                     "%m", "%r_", "_t");
        emitKineticVector("%khalf", thermal, "%m", "  ");
        emitVelocityScaling("  ", "%v1t", thermal, "%muinv");
        drifting = "%v1t";
      }
      emitDrift("  ", drifted, reference, drifting, "%half");
      length = "%half";
    } else if (scales) {
      // The scale of each axis, μ, the same for the three with isotropic
      // coupling (D119).
      emitDrift("  ", "%xh", "%x", "%v1", "%half");
      emitDriftScale("  ");
      // Langevin dynamics acts before the scaling, which takes the kinetic
      // energy of the velocities that it scales.
      std::string middle = "%v1";
      if (control.isLangevin()) {
        emitLangevin("  ", "%v1", "%v1o");
        middle = "%v1o";
      }
      std::string scaled = emitGroupScaling("  ", "%xh", "%mu", "%mu_m1",
                                            "%cell", "%m", "%r_", "_t");
      // The kinetic energy of each axis of the velocities that the
      // scaling takes.
      emitKineticVector("%khalf", middle, "%m", "  ");
      emitVelocityScaling("  ", "%v1t", middle, "%muinv");
      emitDrift("  ", drifted, scaled, "%v1t", "%half");
      drifting = "%v1t";
    } else if (control.isLangevin()) {
      // The middle scheme: half a drift, the friction and the noise, and
      // the other half ([Zhang2019]; BAOAB of [Leimkuhler2013]).
      emitDrift("  ", "%xh", "%x", "%v1", "%half");
      emitLangevin("  ", "%v1", "%v1o");
      emitDrift("  ", drifted, "%xh", "%v1o", "%half");
      drifting = "%v1o";
    } else {
      emitDrift("  ", drifted, "%x", "%v1", "%dt");
      if (sideVirial)
        emitDrift("  ", "%x1dp", "%x", "%v1p", "%dt");
    }
    // The constraints back to their lengths, and the velocities that take
    // the atoms there over the step (the first half of RATTLE).
    std::string velocities = drifting;
    std::vector<std::pair<const Program::TupleSet *, std::string>> changes;
    // Each set of constraints tests the lengths it gives in its own kernel,
    // which yields whether one fails; with MDIR_CONSTRAINT_TEST=separate a
    // loop of its own tests them below (D278).
    bool apart = testsConstraintsApart();
    std::vector<std::pair<int, std::string>> failures;
    if (constraints) {
      std::string current = "%x1d";
      std::string constrained = sites ? "%x1s" : "%x1";
      unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
      auto next = [&]() {
        ++step;
        return step == steps ? constrained
                             : "%x1c" + std::to_string(step);
      };
      if (settles) {
        std::string result = next();
        if (!apart)
          failures.push_back({0, "%broken_settles"});
        emitSettlePositions("  ", reference, current, "%dx1", result,
                            apart ? "" : "%broken_settles");
        changes.push_back({program.tupleSets.data(), "%dx1"});
        for (const Program::TupleSet &set : program.tupleSets)
          if (set.name == "settles")
            changes.back().first = &set;
        current = result;
      }
      for (const Program::TupleSet *set : shakeSets) {
        std::string result = next();
        std::string failed = "%broken_" + set->name;
        if (!apart)
          failures.push_back({static_cast<int>(set->arity) - 1, failed});
        emitShakePositions("  ", reference, current, *set, result,
                           apart ? "" : failed);
        changes.push_back({set, result + "_change"});
        current = result;
      }
      emitIRFragment(
          "  ", {"%v1c"}, {"!vec"}, {drifting, current, "%x1d", length},
          {"!vec", "!vec", "!vec", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildCorrectedHalfVelocity(builder, values[0], values[1], values[2], values[3])};
          });
      velocities = "%v1c";
      if (sideVirial) {
        // What the constraints add to the drift of velocity Verlet: their
        // change of this drift, and what this drift is ahead of that one,
        // which lies along the constraints. Each set reads its own
        // particles.
        emitIRFragment(
            "  ", {"%dxp"}, {"!vec"}, {"%x1d", "%x1dp"}, {"!vec", "!vec"},
            [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
              return {buildFieldDifference(builder, values[0], values[1])};
            });
        unsigned index = 0;
        for (auto &[set, change] : changes) {
          std::string whole = "%dxp" + std::to_string(index++);
          emitAddChange("  ", change, "%dxp", whole);
          change = whole;
        }
      }
    }
    // With MDIR_CONSTRAINT_TEST=separate, each set of constraints tested at
    // the positions it gave, by a loop of its own, after the loops that
    // update the velocities from them, so that the run of the constraints
    // stays one kernel with those loops (D110); the flags are read after the
    // evaluation (D278).
    if (constraints && apart)
      failures = emitConstraintTests("  ", sites ? "%x1s" : "%x1",
                                     "%broken", /*flag=*/true);
    if (sites)
      emitPlaceSites("  ", constraints ? "%x1s" : "%x1d", "%x1", "%r_");
    std::string raw = sites ? "e" : "";
    // With a learned potential the forces and the energy of the evaluation
    // are `%f1c` and `%u1c`, and those of the model are added to give `%f1`
    // and `%u1` (D270).
    bool hostTerms = hasHostEvaluation();
    std::string modelled = !hostTerms || !held.empty() ? "" : "c";
    std::string energyName = withEnergy ? "%u1" + held + modelled : "";
    std::string forceName = "%f1" + held + raw + modelled;
    std::string virialName = withEnergy || withVirial ? "%w1" + held + raw : "";
    emitEnergyEvaluation("  ", "%x1", "%cell", "%p_", "%time", energyName, forceName, virialName);
    std::string virial = "%w1" + held;
    if (sites)
      virial = emitSpreadSites("  ", "%x1", "%f1" + held + "e", "%f1" + held,
                               "%r_", withVirial ? "%w1" + held + "e" : "",
                               "%w1" + held);
    if (hasForceCorrections()) {
      emitForceCorrections("  ", "%x1", "%p_", "%f1p", hostTerms ? "%f1c" : "%f1",
                     withEnergy ? "%u1p" : "",
                     withEnergy ? (hostTerms ? "%u1c" : "%u1") : "",
                     withVirial ? virial : "", withVirial ? "%w1" : "");
      virial = "%w1";
    }
    if (hostTerms)
      virial = emitHostEvaluation("  ", "%x1", "%cell", "%p_", "%f1c", "%f1",
                              withEnergy ? "%u1c" : "", "%u1",
                              withVirial ? virial : "", "1");
    emitConstraintStops("  ", failures);
    std::string groups;
    if (molecular) {
      emitDiagonal("%gd1", virial, "  ");
      groups = emitMolecularDiagonal("  ", "%gd1", "%x1", "%f1", "%cell",
                                     "%m", "%r_", "1");
    }
    if (leapfrog && !withVirial && !scales && !measured) {
      emitDynamicsReturn("  ", {"%x1", velocities, "%f1"}, {"!vec", "!vec", "!vec"});
      os << "}\n\n";
      continue;
    }
    if (brownian) {
      // No kick: the velocities are those of the displacement, with the
      // constraints. The virial is that of the forces alone.
      llvm::SmallVector<StringRef> values{"%x1", velocities, "%f1"}, types{"!vec", "!vec", "!vec"};
      if (withEnergy) { values.push_back("%u1"); types.push_back("f64"); }
      if (withVirial) { values.push_back(virial); types.push_back("vector<9xf64>"); }
      emitDynamicsReturn("  ", values, types);
      os << "}\n\n";
      continue;
    }
    // The second half kick, to the velocities of the time of the new
    // positions: those of the next step with velocity Verlet, and those of
    // the energies with leapfrog.
    emitKick("  ", constraints ? "%v2u" : "%v2", velocities, "%f1", "%m", "%half");
    // The virial of the constraints: the mean of those of the two halves of
    // the step, or, under Langevin dynamics, whose friction and noise act
    // within the drift, that of their forces at its end (D135).
    bool instantaneous = control.isLangevin();
    if (constraints && withVirial && !instantaneous) {
      unsigned index = 0;
      for (auto [set, change] : changes)
        virial = emitConstraintVirial("  ", *set, "%x", change, virial,
                                      "%w1x" + std::to_string(index++));
    }
    if (constraints) {
      std::string halves =
          project("%x1", "%v2u", "%v2",
                  withVirial && !instantaneous ? virial : "")
              .second;
      if (withVirial)
        virial = instantaneous
                     ? emitConstraintForceVirial("  ", "%x1", "%v2", "%f1",
                                                 virial)
                     : halves;
    }
    std::string stored = leapfrog ? velocities : "%v2";
    // The steps of energy also report the rigid waters, the solvent.
    if (measured)
      emitHalfStepKinetic(velocities, leapfrog && !geodesic, solvent,
                          geodesic);
    if (withVirial || scales || measured) {
      llvm::SmallVector<StringRef> values{"%x1", stored, "%f1"}, types{"!vec", "!vec", "!vec"};
      if (withEnergy) { values.push_back("%u1"); types.push_back("f64"); }
      if (withVirial) { values.push_back(virial); types.push_back("vector<9xf64>"); }
      if (molecular) { values.push_back(groups); types.push_back("vector<3xf64>"); }
      if (returnsCurrent) { values.push_back("%v2"); types.push_back("!vec"); }
      if (scales) { values.push_back("%khalf"); types.push_back("vector<3xf64>"); }
      if (measured) { values.push_back("%khs"); types.push_back("vector<3xf64>"); }
      if (solvent) { values.push_back("%kws"); types.push_back("vector<3xf64>"); }
      emitDynamicsReturn("  ", values, types);
    } else {
      emitDynamicsReturn("  ", {"%x1", "%v2", "%f1"}, {"!vec", "!vec", "!vec"});
    }
    os << "}\n\n";
  }
}

void Builder::emitHalfStepKinetic(StringRef velocities, bool leapfrog,
                                  bool solvent, bool geodesic) {
  // The kinetic energy of each axis at the half steps around the end of
  // the step, K_half = [K(t - dt/2) + K(t + dt/2)] / 2 [Jung2019], from the
  // velocities after their constraints: `velocities`, those that this step
  // drifted with, and those that the next step will drift with, which are
  // taken here as that step takes them, by a kick, a drift, and the
  // constraints of the positions. The forces alone do not give them with
  // constraints (D203). Nothing here feeds the state
  // that the step returns. Under g-BAOAB (`geodesic`, D265)
  // the half steps are the drifts of the two A of half a step around the
  // end of the step: that of this step, and that of the next without its
  // noise, which comes after it.
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  StringRef length = geodesic ? "%half" : "%dt";
  emitIRFragment(
      "  ", {"%v3k", "%x3d"}, {"!vec", "!vec"},
      {leapfrog ? velocities : StringRef("%v2"), "%f1", "%m",
       leapfrog ? StringRef("%dt") : StringRef("%half"), "%x1", length},
      {"!vec", "!vec", "!real", "f64", "!vec", "f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        mlir::Value kick = mdir::dyn::KickOp::create(
            builder, loc, values[0].getType(), values[0], values[1], values[2], values[3]);
        mlir::Value drift = mdir::dyn::DriftOp::create(
            builder, loc, values[4].getType(), values[4], kick, values[5]);
        return {kick, drift};
      });
  std::string current = "%x3d";
  unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
  auto next = [&]() {
    ++step;
    return step == steps ? std::string("%x3c")
                         : "%x3c" + std::to_string(step);
  };
  if (settles) {
    std::string result = next();
    emitSettlePositions("  ", "%x1", current, "%dx3", result);
    current = result;
  }
  for (const Program::TupleSet *set : shakeSets) {
    std::string result = next();
    emitShakePositions("  ", "%x1", current, *set, result);
    current = result;
  }
  emitIRFragment(
      "  ", {"%v3c", "%khs"}, {"!vec", "vector<3xf64>"},
      {"%v3k", current, "%x3d", length, velocities, "%m"},
      {"!vec", "!vec", "!vec", "f64", "!vec", "!real"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mlir::Value corrected = buildCorrectedHalfVelocity(
            builder, values[0], values[1], values[2], values[3]);
        mlir::Value kinetic = buildHalfStepKinetic(
            builder, values[4], corrected, values[5]);
        return {corrected, kinetic};
      });
  if (!solvent)
    return;
  // The kinetic energies of the rigid waters, K and K_half, as the first
  // two elements of a vector. Its otherwise unused coordinate is retained.
  emitIRFragment(
      "  ", {"%kws"}, {"vector<3xf64>"},
      {"%r_settles", "%x1", "%cell", "%v2", velocities, "%v3c", "%m"},
      {"!rel_settles", "!vec", "!md.cell", "!vec", "!vec", "!vec", "!real"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        SolventKineticInputs input{values[0], values[1], values[2], values[3],
                                     values[4], values[5], values[6]};
        return {buildSolventKinetic(builder, input)};
      });
}

void Builder::emitLangevin(StringRef indent, StringRef velocities,
                           StringRef result) {
  // v' = c v + sqrt((1 - c^2) k_B T / m) R, with c = exp(-gamma dt) and R
  // three standard normal numbers.
  double c = std::exp(-control.friction * control.timestep);
  double kT = units::boltzmann * control.temperature;
  emitNoise(indent, velocities, result, c, std::sqrt((1.0 - c * c) * kT));
}

void Builder::emitBrownianVelocity(StringRef indent, StringRef forces,
                                   StringRef result) {
  // The step of Ermak and McCammon, J. Chem. Phys. 69, 1352 (1978), without
  // hydrodynamic interactions: Δx = Δt F / (m γ) + √(2 k_B T Δt /
  // (m γ)) R, written as a velocity that the positions drift by over Δt. A
  // particle of mass 0 (a virtual site, placed after the drift) does not
  // drift; the step does not read the velocities of the step before.
  double gamma = control.friction;
  double kT = units::boltzmann * control.temperature;
  std::string drift = (result + "_drift").str();
  emitIRFragment(
      indent, {drift}, {"!vec"}, {forces, "%m"}, {"!vec", "!real"},
      [gamma](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildBrownianDrift(builder, values[0], values[1], gamma)};
      });
  emitNoise(indent, drift, result, 1.0,
            std::sqrt(2.0 * kT / (gamma * control.timestep)));
}

void Builder::emitNoise(StringRef indent, StringRef velocities,
                        StringRef result, double c, double spread) {
  // The normal numbers R come from one block of Philox 4x32-10
  // [Salmon2011] under the key of A13: the seed; the counter of the step,
  // the number of the particle, and stream 2 in the high byte of the last
  // word. The four words give four uniform numbers, (w + 1/2) 2^-32, and
  // the method of Box and Muller two pairs of normal numbers, of which the
  // first three are taken. A particle of mass 0 keeps its velocity.
  NoiseIRConfig config{control.seed, c, spread};
  emitIRFragment(
      indent, {result}, {"!vec"},
      {velocities, "%m", "%noise_ids", "%noise_step"},
      {"!vec", "!real", "!ids", "i64"},
      [config](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        NoiseIRInputs input{values[0], values[1], values[2], values[3]};
        return {buildNoise(builder, input, config)};
      });
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
  bool sites = hasSites();
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  bool constraints = settles || !shakeSets.empty();
  if (!hasForceCorrections() && !hasHostEvaluation()) {
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
        builder, types, static_cast<double>(system.getNumParticles()), control.usesTime, groups, siteSets);
    // The module owns this complete native body. Only its top-level placement
    // crosses the temporary join; no argument or operation in it is reparsed.
    emitIRFragment("", {}, {}, {}, {},
        [nativeProgram](mlir::OpBuilder &where, mlir::ValueRange) -> TermIRValues {
          nativeProgram->moveBefore(where.getInsertionBlock(), where.getInsertionPoint());
          return {};
        });
    return;
  }

  os << "dyn.program @descend(%x: !vec, %f: !vec, %m: !real, "
        "%cell: !md.cell,\n    %h: f64"
     << getFieldParameters() << ")\n    -> (!vec, !vec, f64)"
     << " attributes {effects = "
     << (hasHostEvaluation()
             ? "{read = [\"<Default>\"], write = [\"<Default>\"]}"
             : "{}")
     << "} {\n";
  emitIRFragment(
      "  ", {"%zero", "%tiny", "%count"}, {"f64", "f64", "f64"}, {}, {},
      [count = static_cast<double>(system.getNumParticles())](
          mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        mlir::Value zero = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
        mlir::Value tiny = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0e-300));
        mlir::Value particles = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(count));
        return {zero, tiny, particles};
      });
  emitConstrainedDescent("  ", "%x", "%f", "%g");
  std::string trial = sites || constraints ? "%x1d" : "%x1";
  emitIRFragment(
      "  ", {trial}, {"!vec"}, {"%x", "%g", "%h", "%count", "%tiny"},
      {"!vec", "!vec", "f64", "f64", "f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildDescentPositions(builder, values[0], values[1], values[2],
                                        values[3], values[4])};
      });
  // The rigid groups back to their shapes, and the sites where their
  // atoms put them.
  if (constraints) {
    std::string current = "%x1d";
    std::string constrained = "%x1k";
    unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
    auto next = [&]() {
      ++step;
      return step == steps ? constrained : "%x1c" + std::to_string(step);
    };
    if (settles) {
      std::string result = next();
      emitSettlePositions("  ", "%x", current, "%dx1", result);
      current = result;
    }
    for (const Program::TupleSet *set : shakeSets) {
      std::string result = next();
      emitShakePositions("  ", "%x", current, *set, result);
      current = result;
    }
    // A trial whose groups the constraints did not take back to their
    // shapes, a step too long for SETTLE or M-SHAKE, is not taken: the
    // energy is evaluated where the step began, so that no position that
    // is not a number reaches the neighbor structure, and is returned as
    // +inf, so that the step shrinks. Taken, it would leave the surface of
    // the constraints, and every later trial would hold the change that
    // takes the group back, which does not shrink with the step
    // (D259, #343).
    emitConstraintFailures("  ", "%x1k", "%broken");
    std::string chosen = sites ? "%x1s" : "%x1";
    emitIRFragment(
        "  ", {chosen, "%intact"}, {"!vec", "i1"},
        {"%x1k", "%x", "%broken", "%zero"}, {"!vec", "!vec", "f64", "f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto result = buildConstrainedTrial(builder, values[0], values[1],
                                                values[2], values[3]);
          return {result.positions, result.intact};
        });
  }
  if (sites)
    emitPlaceSites("  ", constraints ? "%x1s" : "%x1d", "%x1", "%r_");
  std::string raw = sites ? "e" : "";
  std::string held = hasForceCorrections() ? "p" : "";
  // A learned potential adds its energy and forces to `%u1c` and `%f1c`
  // (D270).
  bool hostTerms = hasHostEvaluation();
  std::string modelled = !hostTerms || !held.empty() ? "" : "c";
  emitEnergyEvaluation("  ", "%x1", "%cell", "%p_", "%zero",
                         "%u1" + held + modelled, "%f1" + held + raw + modelled, "");
  if (sites)
    emitSpreadSites("  ", "%x1", "%f1" + held + "e", "%f1" + held, "%r_");
  if (hasForceCorrections()) {
    emitIRFragment(
        "  ", {"%w1z"}, {"vector<9xf64>"}, {}, {},
        [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          auto vec = mlir::VectorType::get({9}, builder.getF64Type());
          return {mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(),
                                                  mlir::DenseElementsAttr::get(vec, 0.0))};
        });
    emitForceCorrections("  ", "%x1", "%p_", "%f1p", hostTerms ? "%f1c" : "%f1",
                   "%u1p", hostTerms ? "%u1c" : "%u1", "%w1z", "%w1r");
  }
  if (hostTerms)
    emitHostEvaluation("  ", "%x1", "%cell", "%p_", "%f1c", "%f1", "%u1c", "%u1",
                   "", "1");
  std::string energy = "%u1";
  if (constraints) {
    emitIRFragment(
        "  ", {"%u1c"}, {"f64"}, {"%u1", "%intact"}, {"f64", "i1"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {buildTrialEnergy(builder, values[0], values[1])};
        });
    energy = "%u1c";
  }
  emitDynamicsReturn("  ", {"%x1", "%f1", energy}, {"!vec", "!vec", "f64"});
  os << "}\n\n";
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
void Builder::emitDrift(StringRef indent, StringRef result, StringRef positions,
                          StringRef velocities, StringRef interval) {
  emitIRFragment(indent, {result}, {"!vec"}, {positions, velocities, interval},
      {"!vec", "!vec", "f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildDrift(builder, values[0], values[1], values[2])};
      });
}
void Builder::emitDriftScale(StringRef indent) {
  emitIRFragment(indent, {"%mu_m1", "%muinv"}, {"vector<3xf64>", "vector<3xf64>"},
      {"%mu"}, {"vector<3xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto result = buildDriftScale(builder, values[0]);
        return {result.oneMinus, result.inverse};
      });
}
void Builder::emitVelocityScaling(StringRef indent, StringRef result,
                                     StringRef velocities, StringRef scale) {
  emitIRFragment(indent, {result}, {"!vec"}, {velocities, scale}, {"!vec", "vector<3xf64>"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildVelocityScaling(builder, values[0], values[1])};
      });
}
void Builder::emitDynamicsReturn(StringRef indent, llvm::ArrayRef<StringRef> values,
                                    llvm::ArrayRef<StringRef> types) {
  emitIRFragment(indent, {}, {}, values, types,
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mdir::dyn::ReturnOp::create(builder, builder.getUnknownLoc(), values);
        return {};
      });
}

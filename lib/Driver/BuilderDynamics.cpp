// The programs of a step, the kinetic energies, the noise, the couplings
// and the barostat, and the program of the minimization
// (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"
#include "CouplingIR.h"
#include "NoiseIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

/// The kinetic energy of the velocities `velocities` along each axis,
/// Σ m v ⊙ v / 2, a vector of three.
static void emitKineticVector(llvm::raw_ostream &os, StringRef result,
                              StringRef velocities, StringRef masses,
                              StringRef indent) {
  os << indent << result << " = md.sum_particles gather(" << velocities
     << ", " << masses << " : !vec, !real) {\n"
     << indent << "^bb0(%kv_v: vector<3xf64>, %kv_m: f64):\n"
     << indent << "  %kv_c = arith.constant 5.0e-01 : f64\n"
     << indent << "  %kv_hm = arith.mulf %kv_c, %kv_m : f64\n"
     << indent << "  %kv_hmb = vector.broadcast %kv_hm : f64 to vector<3xf64>\n"
     << indent << "  %kv_sq = arith.mulf %kv_v, %kv_v : vector<3xf64>\n"
     << indent << "  %kv_ke = arith.mulf %kv_hmb, %kv_sq : vector<3xf64>\n"
     << indent << "  md.yield %kv_ke : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n";
}

void Builder::emitPrograms() {
  if (control.minimize) {
    emitDescend();
    return;
  }
  std::string evaluate = "md.evaluate @energy(%x1, %cell" + getFieldValues() +
                         getTimeValue("%time") + ")";
  std::string signature =
      "(!vec, !md.cell" + getFieldTypes() + getTimeType() + ")";
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
                    : "provides = [\"symplectic\", \"time_reversible\"]} {\n")
       << "  %c = arith.constant 5.0e-01 : f64\n"
       << "  %half = arith.mulf %c, %dt : f64\n";
    // With constraints, a build of the neighbor structure in the step that
    // finds positions that are not numbers records its stop, and the step
    // runs to its end: a failed set of constraints then stops the run with
    // its own message, and the recorded stop follows the tests otherwise
    // (D278, the maintainer's ruling on PR #381).
    if (constraints)
      os << "  func.call @mdrtDeferStops() : () -> ()\n";
    // The time at the end of the step, at which its forces are evaluated
    // (D145).
    if (control.usesTime)
      os << "  %time_steps = arith.sitofp %noise_step : i64 to f64\n"
         << "  %time = arith.mulf %time_steps, %dt : f64\n";
    if (brownian) {
      // The velocity of the step, which the drift takes over Δt; the
      // velocities stored are those of the displacement, Δx / Δt.
      emitBrownianVelocity("  ", "%f", "%v1");
    } else if (!leapfrog) {
      os << "  %v1 = dyn.kick %v, %f, %m, %half : !vec\n";
    } else if (projectsState) {
      // The velocities of the time of the positions, as velocity Verlet
      // has them, with none along the constraints, and its first half
      // kick: the friction or the scaling of the middle of the drift
      // would act on what a kick of a whole step leaves along the
      // constraints.
      os << "  %v0u = dyn.kick %v, %f, %m, %half : !vec\n";
      std::string current = project("%x", "%v0u", "%v0", "").first;
      os << "  %v1 = dyn.kick " << current << ", %f, %m, %half : !vec\n";
    } else {
      os << "  %v1 = dyn.kick %v, %f, %m, %dt : !vec\n";
      if (sideVirial) {
        // The first half kick of velocity Verlet, from the velocities of
        // the time of the positions with none along the constraints: its
        // drift, `%x1dp` below, ends where the constraints give the
        // virial of the first half of the step (D76). The constraints
        // bring both drifts to the same place, and the state is that of
        // the plain step: nothing of this feeds it.
        os << "  %v0u = dyn.kick %v, %f, %m, %half : !vec\n";
        std::string current = project("%x", "%v0u", "%v0", "").first;
        os << "  %v1p = dyn.kick " << current << ", %f, %m, %half : !vec\n";
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
      os << "  %xga = dyn.drift %x, " << kicked << ", %half : !vec\n";
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
      os << "  %gone = arith.constant 1.0 : f64\n"
         << "  %grate = arith.divf %gone, %half : f64\n"
         << "  %vga = md.map_particles gather(" << kicked << ", " << current
         << ", %xga : !vec, !vec, !vec) {\n"
         << "  ^bb0(%vs_v: vector<3xf64>, %vs_c: vector<3xf64>, "
            "%vs_u: vector<3xf64>):\n"
         << "    %vs_d = arith.subf %vs_c, %vs_u : vector<3xf64>\n"
         << "    %vs_r = vector.broadcast %grate : f64 to vector<3xf64>\n"
         << "    %vs_dv = arith.mulf %vs_r, %vs_d : vector<3xf64>\n"
         << "    %vs_sum = arith.addf %vs_v, %vs_dv : vector<3xf64>\n"
         << "    md.yield %vs_sum : vector<3xf64>\n"
         << "  } : !vec\n";
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
        os << "  %mu_unit = arith.constant dense<1.0> : vector<3xf64>\n"
           << "  %mu_m1 = arith.subf %mu_unit, %mu : vector<3xf64>\n"
           << "  %muinv = arith.divf %mu_unit, %mu : vector<3xf64>\n";
        reference = emitGroupScaling("  ", "%xgc", "%mu", "%mu_m1", "%cell",
                                     "%m", "%r_", "_t");
        emitKineticVector(os, "%khalf", thermal, "%m", "  ");
        os << "  %v1t = md.map_particles gather(" << thermal << " : !vec) {\n"
           << "  ^bb0(%v_i: vector<3xf64>):\n"
           << "    %v_s = arith.mulf %muinv, %v_i : vector<3xf64>\n"
           << "    md.yield %v_s : vector<3xf64>\n"
           << "  } : !vec\n";
        drifting = "%v1t";
      }
      os << "  " << drifted << " = dyn.drift " << reference << ", "
         << drifting << ", %half : !vec\n";
      length = "%half";
    } else if (scales) {
      // The scale of each axis, μ, the same for the three with isotropic
      // coupling (D119).
      os << "  %xh = dyn.drift %x, %v1, %half : !vec\n"
         << "  %mu_unit = arith.constant dense<1.0> : vector<3xf64>\n"
         << "  %mu_m1 = arith.subf %mu_unit, %mu : vector<3xf64>\n"
         << "  %muinv = arith.divf %mu_unit, %mu : vector<3xf64>\n";
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
      emitKineticVector(os, "%khalf", middle, "%m", "  ");
      os << "  %v1t = md.map_particles gather(" << middle << " : !vec) {\n"
         << "  ^bb0(%v_i: vector<3xf64>):\n"
         << "    %v_s = arith.mulf %muinv, %v_i : vector<3xf64>\n"
         << "    md.yield %v_s : vector<3xf64>\n"
         << "  } : !vec\n"
         << "  " << drifted << " = dyn.drift " << scaled
         << ", %v1t, %half : !vec\n";
      drifting = "%v1t";
    } else if (control.isLangevin()) {
      // The middle scheme: half a drift, the friction and the noise, and
      // the other half ([Zhang2019]; BAOAB of [Leimkuhler2013]).
      os << "  %xh = dyn.drift %x, %v1, %half : !vec\n";
      emitLangevin("  ", "%v1", "%v1o");
      os << "  " << drifted << " = dyn.drift %xh, %v1o, %half : !vec\n";
      drifting = "%v1o";
    } else {
      os << "  " << drifted << " = dyn.drift %x, %v1, %dt : !vec\n";
      if (sideVirial)
        os << "  %x1dp = dyn.drift %x, %v1p, %dt : !vec\n";
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
      os << "  %one = arith.constant 1.0 : f64\n"
         << "  %rate = arith.divf %one, " << length << " : f64\n"
         << "  %v1c = md.map_particles gather(" << drifting << ", " << current
         << ", %x1d : !vec, !vec, !vec) {\n"
         << "  ^bb0(%vs_v: vector<3xf64>, %vs_c: vector<3xf64>, "
            "%vs_u: vector<3xf64>):\n"
         << "    %vs_d = arith.subf %vs_c, %vs_u : vector<3xf64>\n"
         << "    %vs_r = vector.broadcast %rate : f64 to vector<3xf64>\n"
         << "    %vs_dv = arith.mulf %vs_r, %vs_d : vector<3xf64>\n"
         << "    %vs_sum = arith.addf %vs_v, %vs_dv : vector<3xf64>\n"
         << "    md.yield %vs_sum : vector<3xf64>\n"
         << "  } : !vec\n";
      velocities = "%v1c";
      if (sideVirial) {
        // What the constraints add to the drift of velocity Verlet: their
        // change of this drift, and what this drift is ahead of that one,
        // which lies along the constraints. Each set reads its own
        // particles.
        os << "  %dxp = md.map_particles gather(%x1d, %x1dp : !vec, !vec) {\n"
           << "  ^bb0(%vs_a: vector<3xf64>, %vs_b: vector<3xf64>):\n"
           << "    %vs_d = arith.subf %vs_a, %vs_b : vector<3xf64>\n"
           << "    md.yield %vs_d : vector<3xf64>\n"
           << "  } : !vec\n";
        unsigned index = 0;
        for (auto &[set, change] : changes) {
          std::string whole = "%dxp" + std::to_string(index++);
          os << "  " << whole << " = md.map_particles gather(" << change
             << ", %dxp : !vec, !vec) {\n"
             << "  ^bb0(%vs_a: vector<3xf64>, %vs_b: vector<3xf64>):\n"
             << "    %vs_s = arith.addf %vs_a, %vs_b : vector<3xf64>\n"
             << "    md.yield %vs_s : vector<3xf64>\n"
             << "  } : !vec\n";
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
    if (withEnergy)
      os << "  %u1" << held << modelled << ", %f1" << held << raw << modelled
         << ", %w1" << held << raw << " = " << evaluate
         << "\n      request [energy, forces, virial]\n"
         << "      : " << signature << " -> (f64, !vec, vector<9xf64>)\n";
    else if (withVirial)
      os << "  %f1" << held << raw << modelled << ", %w1" << held << raw
         << " = " << evaluate << "\n      request [forces, virial]\n"
         << "      : " << signature << " -> (!vec, vector<9xf64>)\n";
    else
      os << "  %f1" << held << raw << modelled << " = " << evaluate
         << " request [forces]\n"
         << "      : " << signature << " -> !vec\n";
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
      emitDiagonal(os, "%gd1", virial, "  ");
      groups = emitMolecularDiagonal("  ", "%gd1", "%x1", "%f1", "%cell",
                                     "%m", "%r_", "1");
    }
    if (leapfrog && !withVirial && !scales && !measured) {
      os << "  dyn.return %x1, " << velocities
         << ", %f1 : !vec, !vec, !vec\n}\n\n";
      continue;
    }
    if (brownian) {
      // No kick: the velocities are those of the displacement, with the
      // constraints. The virial is that of the forces alone.
      os << "  dyn.return %x1, " << velocities << ", %f1"
         << (withEnergy ? ", %u1" : "") << (withVirial ? ", " + virial : "")
         << " : !vec, !vec, !vec" << (withEnergy ? ", f64" : "")
         << (withVirial ? ", vector<9xf64>" : "") << "\n}\n\n";
      continue;
    }
    // The second half kick, to the velocities of the time of the new
    // positions: those of the next step with velocity Verlet, and those of
    // the energies with leapfrog.
    os << "  %v2" << (constraints ? "u" : "") << " = dyn.kick " << velocities
       << ", %f1, %m, %half : !vec\n";
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
    if (withVirial || scales || measured)
      os << "  dyn.return %x1, " << stored << ", %f1"
         << (withEnergy ? ", %u1" : "")
         << (withVirial ? ", " + virial : "")
         << (molecular ? ", " + groups : "")
         << (returnsCurrent ? ", %v2" : "") << (scales ? ", %khalf" : "")
         << (measured ? ", %khs" : "") << (solvent ? ", %kws" : "") << "\n"
         << "      : !vec, !vec, !vec" << (withEnergy ? ", f64" : "")
         << (withVirial ? ", vector<9xf64>" : "")
         << (molecular ? ", vector<3xf64>" : "")
         << (returnsCurrent ? ", !vec" : "")
         << (scales ? ", vector<3xf64>" : "")
         << (measured ? ", vector<3xf64>" : "")
         << (solvent ? ", vector<3xf64>" : "") << "\n";
    else
      os << "  dyn.return %x1, %v2, %f1 : !vec, !vec, !vec\n";
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
  if (leapfrog)
    os << "  %v3k = dyn.kick " << velocities << ", %f1, %m, %dt : !vec\n";
  else
    os << "  %v3k = dyn.kick %v2, %f1, %m, %half : !vec\n";
  os << "  %x3d = dyn.drift %x1, %v3k, " << length << " : !vec\n";
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
  os << "  %h3one = arith.constant 1.0 : f64\n"
     << "  %h3rate = arith.divf %h3one, " << length << " : f64\n"
     << "  %v3c = md.map_particles gather(%v3k, " << current
     << ", %x3d : !vec, !vec, !vec) {\n"
     << "  ^bb0(%vs_v: vector<3xf64>, %vs_c: vector<3xf64>, "
        "%vs_u: vector<3xf64>):\n"
     << "    %vs_d = arith.subf %vs_c, %vs_u : vector<3xf64>\n"
     << "    %vs_r = vector.broadcast %h3rate : f64 to vector<3xf64>\n"
     << "    %vs_dv = arith.mulf %vs_r, %vs_d : vector<3xf64>\n"
     << "    %vs_sum = arith.addf %vs_v, %vs_dv : vector<3xf64>\n"
     << "    md.yield %vs_sum : vector<3xf64>\n"
     << "  } : !vec\n"
     << "  %khs = md.sum_particles gather(" << velocities
     << ", %v3c, %m : !vec, !vec, !real) {\n"
     << "  ^bb0(%kh_a: vector<3xf64>, %kh_b: vector<3xf64>, %kh_m: f64):\n"
     << "    %kh_c = arith.constant 2.5e-01 : f64\n"
     << "    %kh_qm = arith.mulf %kh_c, %kh_m : f64\n"
     << "    %kh_qmb = vector.broadcast %kh_qm : f64 to vector<3xf64>\n"
     << "    %kh_aa = arith.mulf %kh_a, %kh_a : vector<3xf64>\n"
     << "    %kh_bb = arith.mulf %kh_b, %kh_b : vector<3xf64>\n"
     << "    %kh_s = arith.addf %kh_aa, %kh_bb : vector<3xf64>\n"
     << "    %kh_ke = arith.mulf %kh_qmb, %kh_s : vector<3xf64>\n"
     << "    md.yield %kh_ke : vector<3xf64>\n"
     << "  } : vector<3xf64>\n";
  if (!solvent)
    return;
  // The kinetic energies of the rigid waters, K and K_half, as the first
  // two elements of a vector.
  os << "  %kws = md.sum_tuples %r_settles, %x1, %cell\n"
     << "    coordinates(displacement(1, 0))\n"
     << "    gather(%v2, " << velocities << ", %v3c, %m"
     << " : !vec, !vec, !vec, !real) {\n"
     << "  ^bb0(%ws_r: vector<3xf64>, %ws_f0: vector<3xf64>, "
        "%ws_f1: vector<3xf64>, %ws_f2: vector<3xf64>, "
        "%ws_a0: vector<3xf64>, %ws_a1: vector<3xf64>, "
        "%ws_a2: vector<3xf64>, %ws_b0: vector<3xf64>, "
        "%ws_b1: vector<3xf64>, %ws_b2: vector<3xf64>, %ws_m0: f64, "
        "%ws_m1: f64, %ws_m2: f64):\n";
  std::string full, half;
  for (int j = 0; j != 3; ++j) {
    std::string n = std::to_string(j);
    os << "    %ws_ff" << n << " = arith.mulf %ws_f" << n << ", %ws_f" << n
       << " : vector<3xf64>\n"
       << "    %ws_fs" << n << " = vector.reduction <add>, %ws_ff" << n
       << " : vector<3xf64> into f64\n"
       << "    %ws_fm" << n << " = arith.mulf %ws_m" << n << ", %ws_fs" << n
       << " : f64\n"
       << "    %ws_aa" << n << " = arith.mulf %ws_a" << n << ", %ws_a" << n
       << " : vector<3xf64>\n"
       << "    %ws_bb" << n << " = arith.mulf %ws_b" << n << ", %ws_b" << n
       << " : vector<3xf64>\n"
       << "    %ws_hh" << n << " = arith.addf %ws_aa" << n << ", %ws_bb" << n
       << " : vector<3xf64>\n"
       << "    %ws_hs" << n << " = vector.reduction <add>, %ws_hh" << n
       << " : vector<3xf64> into f64\n"
       << "    %ws_hm" << n << " = arith.mulf %ws_m" << n << ", %ws_hs" << n
       << " : f64\n";
    if (j == 0) {
      full = "%ws_fm0";
      half = "%ws_hm0";
    } else {
      os << "    %ws_ft" << n << " = arith.addf " << full << ", %ws_fm" << n
         << " : f64\n"
         << "    %ws_ht" << n << " = arith.addf " << half << ", %ws_hm" << n
         << " : f64\n";
      full = "%ws_ft" + n;
      half = "%ws_ht" + n;
    }
  }
  os << "    %ws_c2 = arith.constant 5.0e-01 : f64\n"
     << "    %ws_c4 = arith.constant 2.5e-01 : f64\n"
     << "    %ws_k = arith.mulf %ws_c2, " << full << " : f64\n"
     << "    %ws_kh = arith.mulf %ws_c4, " << half << " : f64\n"
     << "    %ws_z = arith.constant 0.0 : f64\n"
     << "    %ws_v = vector.from_elements %ws_k, %ws_kh, %ws_z : vector<3xf64>\n"
     << "    md.yield %ws_v : vector<3xf64>\n"
     << "  } : !rel_settles, !vec -> vector<3xf64>\n";
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
  return ", " + name + ", " + idName;
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
  for (int k = 0; k != getCellSize(); ++k)
    os << indent << "%be" << t << "_" << k << " = memref.load %box_memory"
       << "[%c_edge" << k << "] : " << getBoxMemoryType() << "\n";
  os << indent << "%bxy" << t << " = arith.mulf %be" << t << "_0, %be" << t
     << "_1 : f64\n"
     << indent << "%bv" << t << " = arith.mulf %bxy" << t << ", %be" << t
     << "_2 : f64\n";
  // The pressure of each axis, in bar: (2 K_a + W_aa + C / (3 V)) / V, with
  // C / V the trace of the virials of the constant terms.
  os << indent << "%bk2" << t << " = arith.addf " << kinetic << ", "
     << kinetic << " : vector<3xf64>\n"
     << indent << "%bw0" << t << " = arith.addf %bk2" << t << ", "
     << diagonal << " : vector<3xf64>\n"
     << indent << "%bwc" << t << " = arith.divf %baro_constant, %bv" << t
     << " : f64\n"
     << indent << "%bwc3" << t << " = arith.mulf %bwc" << t
     << ", %c_third : f64\n"
     << indent << "%bwcb" << t << " = vector.broadcast %bwc3" << t
     << " : f64 to vector<3xf64>\n"
     << indent << "%bw" << t << " = arith.addf %bw0" << t << ", %bwcb" << t
     << " : vector<3xf64>\n"
     << indent << "%bpc" << t << " = arith.divf %c_bar, %bv" << t
     << " : f64\n"
     << indent << "%bpcb" << t << " = vector.broadcast %bpc" << t
     << " : f64 to vector<3xf64>\n"
     << indent << "%bp3" << t << " = arith.mulf %bw" << t << ", %bpcb" << t
     << " : vector<3xf64>\n";
  if (control.anisotropic) {
    // Anisotropic (D163c): each axis by the strain of its own pressure and
    // noise, eq. (9b) of [Bernetti2020] on each.
    std::string strains[3];
    for (int k = 0; k != 3; ++k) {
      std::string a = std::to_string(k);
      strains[k] = "%strain" + a + "_" + t;
      os << indent << "%bpa" << a << "_" << t << " = vector.extract %bp3"
         << t << "[" << k << "] : f64 from vector<3xf64>\n"
         << indent << "%baxis" << a << "_" << t << " = arith.constant " << k
         << " : i64\n"
         << indent << strains[k]
         << " = func.call @mdrtBarostatStrainAxis(%seed, " << step
         << ", %baxis" << a << "_" << t << ", %bpa" << a << "_" << t
         << ", %baro_target, %bv" << t << ", %baro_kt, %baro_beta_" << a
         << ", %baro_rate)\n"
         << indent
         << "    : (i64, i64, i64, f64, f64, f64, f64, f64, f64) -> f64\n";
    }
    os << indent << "%bs3" << t << " = vector.from_elements " << strains[0]
       << ", " << strains[1] << ", " << strains[2] << " : vector<3xf64>\n";
  } else if (!control.semiIsotropic) {
    // Isotropic: the mean pressure gives ε, the change of ln V, by a step
    // of λ = √V (eq. S7 of [Bernetti2020]); each axis scales by ε / 3.
    emitSum3(os, "%bps" + t, "%bp3" + t, indent);
    os << indent << "%bp" << t << " = arith.mulf %bps" << t
       << ", %c_third : f64\n"
       << indent << "%strain" << t
       << " = func.call @mdrtBarostatStrain(%seed, " << step << ", %bp" << t
       << ", %baro_target, %bv" << t
       << ", %baro_kt, %baro_beta, %baro_rate)\n"
       << indent << "    : (i64, i64, f64, f64, f64, f64, f64, f64) -> f64\n"
       << indent << "%bsa" << t << " = arith.mulf %strain" << t
       << ", %c_third : f64\n"
       << indent << "%bs3" << t << " = vector.broadcast %bsa" << t
       << " : f64 to vector<3xf64>\n";
  } else {
    // Semi-isotropic: the mean pressure of x and y gives the change of
    // ln A, that of z the change of ln L, by eqs. (9a) and (9b) of
    // [Bernetti2020] with their own noises (D119); x and y scale by half
    // the first, z by the second.
    os << indent << "%bpx" << t << " = vector.extract %bp3" << t
       << "[0] : f64 from vector<3xf64>\n"
       << indent << "%bpy" << t << " = vector.extract %bp3" << t
       << "[1] : f64 from vector<3xf64>\n"
       << indent << "%bpz" << t << " = vector.extract %bp3" << t
       << "[2] : f64 from vector<3xf64>\n"
       << indent << "%bpxys" << t << " = arith.addf %bpx" << t << ", %bpy"
       << t << " : f64\n"
       << indent << "%bpxy" << t << " = arith.mulf %bpxys" << t
       << ", %couple_half : f64\n"
       << indent << "%strainxy" << t
       << " = func.call @mdrtBarostatStrainArea(%seed, " << step << ", %bpxy"
       << t << ", %baro_target, %bv" << t << ", %be" << t
       << "_2, %baro_kt, %baro_beta, %baro_rate, %baro_tension)\n"
       << indent
       << "    : (i64, i64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64\n"
       << indent << "%strainz" << t
       << " = func.call @mdrtBarostatStrainHeight(%seed, " << step
       << ", %bpz" << t << ", %baro_target, %bv" << t
       << ", %baro_kt, %baro_beta_z, %baro_rate)\n"
       << indent << "    : (i64, i64, f64, f64, f64, f64, f64, f64) -> f64\n"
       << indent << "%bsxy" << t << " = arith.mulf %strainxy" << t
       << ", %couple_half : f64\n"
       << indent << "%bs3" << t << " = vector.from_elements %bsxy" << t
       << ", %bsxy" << t << ", %strainz" << t << " : vector<3xf64>\n";
  }
  os << indent << "%mu" << t << " = math.exp %bs3" << t
     << " : vector<3xf64>\n"
     << indent << "%muinv" << t << " = arith.divf %c_unit3, %mu" << t
     << " : vector<3xf64>\n";
  // The new cell, which the next iteration takes from memory: H diag(μ),
  // the tilts b_x and c_x of the column of x scaled by μ_x and c_y by μ_y
  // (I3 of docs/triclinic.md).
  static const int column[] = {0, 1, 2, 0, 0, 1};
  for (int k = 0; k != getCellSize(); ++k) {
    if (k < 3)
      os << indent << "%bmu" << t << "_" << k << " = vector.extract %mu" << t
         << "[" << k << "] : f64 from vector<3xf64>\n";
    os << indent << "%bn" << t << "_" << k << " = arith.mulf %be" << t
       << "_" << k << ", %bmu" << t << "_" << column[k] << " : f64\n"
       << indent << "memref.store %bn" << t << "_" << k
       << ", %box_memory[%c_edge" << k << "] : " << getBoxMemoryType()
       << "\n";
  }
  os << indent << "func.call @mdrtSetBox(%bn" << t << "_0, %bn" << t
     << "_1, %bn" << t << "_2) : (f64, f64, f64) -> ()\n";
  if (isTriclinic())
    os << indent << "func.call @mdrtSetTilt(%bn" << t << "_3, %bn" << t
       << "_4, %bn" << t << "_5) : (f64, f64, f64) -> ()\n";
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
  emitKineticVector(os, kinetic, velocities, masses, indent);
  if (control.comPeriod > 0) {
    os << indent << "%tpc" << t << " = md.sum_particles gather("
       << velocities << ", " << masses << " : !vec, !real) {\n"
       << indent << "^bb0(%v_i: vector<3xf64>, %m_i: f64):\n"
       << indent << "  %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << indent << "  %p = arith.mulf %mb, %v_i : vector<3xf64>\n"
       << indent << "  md.yield %p : vector<3xf64>\n"
       << indent << "} : vector<3xf64>\n"
       << indent << "%tvcm" << t << " = arith.divf %tpc" << t
       << ", %total_mass : vector<3xf64>\n"
       << indent << "%tpvs" << t << " = arith.mulf %tpc" << t << ", %tvcm"
       << t << " : vector<3xf64>\n"
       << indent << "%tkcm" << t << " = arith.mulf %c_half3, %tpvs" << t
       << " : vector<3xf64>\n"
       << indent << "%tkt" << t << " = arith.subf " << kinetic << ", %tkcm"
       << t << " : vector<3xf64>\n";
    kinetic = "%tkt" + t;
  }
  return kinetic;
}

void Builder::emitStoreTrotterState(StringRef indent, StringRef diagonal,
                                    StringRef groups, StringRef kinetic) {
  std::string t = std::to_string(trotterStores++);
  StringRef vectors[] = {diagonal, groups, kinetic};
  std::string values[9];
  for (int v = 0; v != 3; ++v)
    for (int k = 0; k != 3; ++k) {
      int slot = 3 * v + k;
      values[slot] = "%tms" + t + "_" + std::to_string(slot);
      os << indent << values[slot] << " = vector.extract " << vectors[v]
         << "[" << k << "] : f64 from vector<3xf64>\n"
         << indent << "%tmi" << t << "_" << slot << " = arith.constant "
         << slot << " : index\n"
         << indent << "memref.store " << values[slot]
         << ", %trotter_memory[%tmi" << t << "_" << slot
         << "] : memref<9xf64>\n";
    }
  // The checkpoints keep it, so a run that continues takes the same.
  os << indent << "func.call @mdrtSetBarostatState(";
  for (int slot = 0; slot != 9; ++slot)
    os << (slot ? ", " : "") << values[slot];
  os << ") : (f64, f64, f64, f64, f64, f64, f64, f64, f64) -> ()\n";
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
  os << indent << scaling.workBefore << " = arith.addf " << groups
     << ", %bwcb" << t << " : vector<3xf64>\n";
  scaling.newVolume = "%tvn" + t;
  os << indent << "%tvn0" << t << " = arith.mulf %bn" << t << "_0, %bn" << t
     << "_1 : f64\n"
     << indent << scaling.newVolume << " = arith.mulf %tvn0" << t << ", %bn"
     << t << "_2 : f64\n";
  scaling.cell = "%tcell" + t;
  emitCellOf(indent, scaling.cell, "%bn" + t);
  if (scalesReference()) {
    scaling.scale = "%tscale" + t;
    emitReferenceScale(os, indent, scaling.scale, "%bn" + t + "_0",
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
    emitKineticVector(os, "%kexf" + t, full, massName, indent);
    os << indent << result << " = arith.subf " << measured << ", %kexf" << t
       << " : vector<3xf64>\n";
    return result;
  }
  // Under Langevin dynamics with constraints neither is measured; Brownian
  // dynamics has no momenta.
  if (hasConstraints() || control.isBrownian())
    return "";
  // Without constraints, K(v - dt F / 2m) and K(v + dt F / 2m) average to
  // K(v) + (dt^2 / 8) sum F^2 / m [Jung2018].
  os << indent << "%kexg" << t << " = md.sum_particles gather(" << forces
     << ", " << massName << " : !vec, !real) {\n"
     << indent << "^bb0(%kx_f: vector<3xf64>, %kx_m: f64):\n"
     << indent << "  %kx_sq = arith.mulf %kx_f, %kx_f : vector<3xf64>\n"
     << indent << "  %kx_zero = arith.constant 0.0 : f64\n"
     << indent << "  %kx_one = arith.constant 1.0 : f64\n"
     << indent << "  %kx_massless = arith.cmpf oeq, %kx_m, %kx_zero : f64\n"
     << indent << "  %kx_safe = arith.select %kx_massless, %kx_one, %kx_m : f64\n"
     << indent << "  %kx_inv = arith.divf %kx_one, %kx_safe : f64\n"
     << indent << "  %kx_w = arith.select %kx_massless, %kx_zero, %kx_inv : f64\n"
     << indent << "  %kx_wb = vector.broadcast %kx_w : f64 to vector<3xf64>\n"
     << indent << "  %kx_g = arith.mulf %kx_sq, %kx_wb : vector<3xf64>\n"
     << indent << "  md.yield %kx_g : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n"
     << indent << "%kexd" << t << " = arith.mulf %dt, %dt : f64\n"
     << indent << "%kexc" << t << " = arith.constant 1.25e-01 : f64\n"
     << indent << "%kexs" << t << " = arith.mulf %kexd" << t << ", %kexc"
     << t << " : f64\n"
     << indent << "%kexb" << t << " = vector.broadcast %kexs" << t
     << " : f64 to vector<3xf64>\n"
     << indent << result << " = arith.mulf %kexg" << t << ", %kexb" << t
     << " : vector<3xf64>\n";
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
  if (removesMotion) {
    // The velocity of the center of mass.
    os << indent << "%pc" << t << " = md.sum_particles gather(" << velocities
       << ", " << massName << " : !vec, !real) {\n"
       << indent << "^bb0(%v_i: vector<3xf64>, %m_i: f64):\n"
       << indent << "  %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << indent << "  %p = arith.mulf %mb, %v_i : vector<3xf64>\n"
       << indent << "  md.yield %p : vector<3xf64>\n"
       << indent << "} : vector<3xf64>\n"
       << indent << "%vcm" << t << " = arith.divf %pc" << t
       << ", %total_mass : vector<3xf64>\n";
  }
  // The kinetic energy of the center of mass, P·V / 2, which the removal
  // takes away.
  if (removesMotion)
    os << indent << "%pvs" << t << " = arith.mulf %pc" << t << ", %vcm" << t
       << " : vector<3xf64>\n"
       << indent << "%pv" << t << " = vector.reduction <add>, %pvs" << t
       << " : vector<3xf64> into f64\n"
       << indent << "%kcm" << t << " = arith.mulf %couple_half, %pv" << t
       << " : f64\n";
  else
    os << indent << "%kcm" << t << " = arith.constant 0.0 : f64\n";
  // The energy that the coupling takes: that of the center of mass, and
  // what the thermostat takes from the rest.
  std::string bath = "%kcm" + t;
  std::string kineticAxes;
  if (control.barostat) {
    // The kinetic energy of each axis, which the pressure of each axis
    // takes, and of the center of mass; their sums for the thermostat.
    os << indent << "// The kinetic energy of each axis.\n";
    emitKineticVector(os, "%kv" + t, velocities, massName, indent);
    emitSum3(os, "%kc" + t, "%kv" + t, indent);
    std::string vector = "%kv" + t;
    if (removesMotion) {
      os << indent << "%pvv" << t << " = arith.mulf %pc" << t << ", %vcm" << t
         << " : vector<3xf64>\n"
         << indent << "%kcmv" << t << " = arith.mulf %c_half3, %pvv" << t
         << " : vector<3xf64>\n"
         << indent << "%ktv" << t << " = arith.subf %kv" << t << ", %kcmv"
         << t << " : vector<3xf64>\n";
      vector = "%ktv" + t;
    }
    kineticAxes = vector;
  }
  if (rescalesVelocities()) {
    std::string kinetic = "%kc" + t;
    if (!control.barostat)
      emitKineticEnergy(os, kinetic, velocities, massName, indent);
    if (removesMotion) {
      os << indent << "%kt" << t << " = arith.subf %kc" << t << ", %kcm" << t
         << " : f64\n";
      kinetic = "%kt" + t;
    }
    // The factor takes the kinetic energy of the optimal temperature
    // [Jung2019] without that of the center of mass (D263);
    // it scales every velocity, and the bath takes the change of the
    // kinetic energy of the velocities scaled, exactly.
    std::string controlled = kinetic;
    if (!optimal.empty()) {
      controlled = optimal.str();
      if (removesMotion) {
        os << indent << "%kot" << t << " = arith.subf " << optimal << ", %kcm"
           << t << " : f64\n";
        controlled = "%kot" + t;
      }
    }
    if (control.isNoseHoover()) {
      // A Nose-Hoover chain (D163a) moves on the host over the period.
      os << indent << "%alpha" << t << " = func.call @mdrtNoseHooverFactor("
         << step << ", " << controlled << ") : (i64, f64) -> f64\n";
    } else {
      os << indent << "%alpha" << t << " = func.call @mdrtBussiFactor(%seed, "
         << step << ", " << controlled
         << ", %target_kinetic, %freedom, %decay)\n"
         << indent << "    : (i64, i64, f64, f64, f64, f64) -> f64\n";
    }
    os << indent << "%alpha2" << t << " = arith.mulf %alpha" << t
       << ", %alpha" << t << " : f64\n"
       << indent << "%kn" << t << " = arith.mulf %alpha2" << t << ", "
       << kinetic << " : f64\n"
       << indent << "%heat" << t << " = arith.subf %kc" << t << ", %kn" << t
       << " : f64\n";
    bath = "%heat" + t;
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
      os << indent << "%alpha2b" << t << " = vector.broadcast %alpha2" << t
         << " : f64 to vector<3xf64>\n"
         << indent << "%knv" << t << " = arith.mulf %alpha2b" << t << ", "
         << kinetic << " : vector<3xf64>\n";
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
        os << indent << "%bwl" << t << " = arith.mulf "
           << trotter->workBefore << ", " << trotter->logMu
           << " : vector<3xf64>\n";
        emitSum3(os, "%bwork" + t, "%bwl" + t, indent);
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
          os << indent << "%gst2" << t << " = vector.broadcast %alpha2" << t
             << " : f64 to vector<3xf64>\n"
             << indent << "%gsts" << t << " = arith.mulf %gst2" << t << ", "
             << centers.kinetic << " : vector<3xf64>\n"
             << indent << "%gsth" << t << " = arith.addf %gsts" << t << ", "
             << centers.forceTerm << " : vector<3xf64>\n";
          stored = "%gsth" + t;
        }
        emitStoreTrotterState(indent, diagonal, groupsAfter, stored);
      }
      if (countsAfterScaling()) {
        os << indent << "%bwca" << t << " = arith.divf %baro_constant, "
           << trotter->newVolume << " : f64\n"
           << indent << "%bwca3" << t << " = arith.mulf %bwca" << t
           << ", %c_third : f64\n"
           << indent << "%bwcab" << t << " = vector.broadcast %bwca3" << t
           << " : f64 to vector<3xf64>\n"
           << indent << "%bwb" << t << " = arith.addf " << groupsAfter
           << ", %bwcab"
           << t << " : vector<3xf64>\n"
           << indent << "%bws" << t << " = arith.addf " << trotter->workBefore
           << ", %bwb" << t << " : vector<3xf64>\n"
           << indent << "%bwm" << t << " = arith.mulf %bws" << t
           << ", %c_half3 : vector<3xf64>\n"
           << indent << "%bwl" << t << " = arith.mulf %bwm" << t << ", "
           << trotter->logMu << " : vector<3xf64>\n";
        emitSum3(os, "%bwork" + t, "%bwl" + t, indent);
      }
      os << indent << "%bmi2" << t << " = arith.mulf " << trotter->muinv
         << ", " << trotter->muinv << " : vector<3xf64>\n"
         << indent << "%bmi21" << t << " = arith.subf %bmi2" << t
         << ", %c_unit3 : vector<3xf64>\n"
         << indent << "%bdkv" << t << " = arith.mulf %bmi21" << t << ", "
         << trotter->kineticHalf << " : vector<3xf64>\n";
      emitSum3(os, "%bdk" + t, "%bdkv" + t, indent);
      os
         << indent << "%bdku" << t << " = arith.subf %bdk" << t << ", %bwork"
         << t << " : f64\n"
         << indent << "%btake" << t << " = arith.subf " << bath << ", %bdku"
         << t << " : f64\n";
      bath = "%btake" + t;
    }
    if (!trotter)
      os << indent << "%bm1" << t << " = arith.subf %c_unit3, %mu" << t
         << " : vector<3xf64>\n";
    if (!exact && !trotter) {
      // To first order, isotropic only (Control): the trace of the virial
      // of the groups, a third to each axis.
      emitSum3(os, "%dtr" + t, groups, indent);
      os << indent << "%bwt" << t << " = arith.addf %dtr" << t
         << ", %bwc" << t << " : f64\n"
         << indent << "%bwt3" << t << " = arith.mulf %bwt" << t
         << ", %c_third : f64\n"
         << indent << "%bwtb" << t << " = vector.broadcast %bwt3" << t
         << " : f64 to vector<3xf64>\n"
         << indent << "%bwl" << t << " = arith.mulf %bm1" << t << ", %bwtb"
         << t << " : vector<3xf64>\n";
      emitSum3(os, "%bwork" + t, "%bwl" + t, indent);
    }
    if (!trotter) {
      os << indent << "%bmi2" << t << " = arith.mulf %muinv" << t
         << ", %muinv" << t << " : vector<3xf64>\n"
         << indent << "%bmi21" << t << " = arith.subf %bmi2" << t
         << ", %c_unit3 : vector<3xf64>\n"
         << indent << "%bdkv" << t << " = arith.mulf %bmi21" << t << ", "
         << after << " : vector<3xf64>\n";
      emitSum3(os, "%bdk" + t, "%bdkv" + t, indent);
    }
    if (!exact && !trotter) {
      os << indent << "%bgain" << t << " = arith.addf %bwork" << t << ", %bdk"
         << t << " : f64\n"
         << indent << "%btake" << t << " = arith.subf " << bath << ", %bgain"
         << t << " : f64\n";
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
        emitReferenceScale(os, indent, scaleName, "%bn" + t + "_0",
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
        os << indent << time << "_steps_i = memref.load %noise_memory[%c0]"
           << " : memref<1xi64>\n"
           << indent << time << "_steps = arith.sitofp " << time
           << "_steps_i : i64 to f64\n"
           << indent << time << " = arith.mulf " << time << "_steps, %dt"
           << " : f64\n";
      // A learned potential adds its energy and forces in the new cell
      // (D270).
      bool hostTerms = hasHostEvaluation();
      std::string modelled = !hostTerms || !held.empty() ? "" : "c";
      os << indent << u << held << modelled << ", " << f << held << raw
         << modelled << " = md.evaluate @energy(" << newPositions << ", "
         << cell << getFieldValues(fieldPrefix) << getTimeValue(time)
         << ")\n"
         << indent << "    request [energy, forces]\n"
         << indent << "    : (!vec, !md.cell" << getFieldTypes()
         << getTimeType() << ") -> (f64, !vec)\n";
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
      os << indent << "%bnxy" << t << " = arith.mulf %bn" << t << "_0, %bn"
         << t << "_1 : f64\n"
         << indent << "%bnv" << t << " = arith.mulf %bnxy" << t << ", %bn" << t
         << "_2 : f64\n"
         << indent << "%bnvi" << t << " = arith.divf %c_unit, %bnv" << t
         << " : f64\n"
         << indent << "%bovi" << t << " = arith.divf %c_unit, %bv" << t
         << " : f64\n"
         << indent << "%bdvi" << t << " = arith.subf %bnvi" << t << ", %bovi"
         << t << " : f64\n"
         << indent << "%bdc" << t << " = arith.mulf %baro_energy_constant, %bdvi"
         << t << " : f64\n"
         << indent << "%bdue" << t << " = arith.subf " << u << ", " << energy
         << " : f64\n"
         << indent << "%bdu" << t << " = arith.addf %bdue" << t << ", %bdc" << t
         << " : f64\n"
         << indent << "%bgain" << t << " = arith.addf %bdu" << t << ", %bdk"
         << t << " : f64\n"
         << indent << "%btake" << t << " = arith.subf " << bath << ", %bgain"
         << t << " : f64\n";
      bath = "%btake" + t;
    }
  }
  os << indent << "func.call @mdrtAddBath(" << bath << ") : (f64) -> ()\n";
  os << indent << "%vc" << t << " = md.map_particles gather(" << velocities
     << " : !vec) {\n"
     << indent << "^bb0(%v_i: vector<3xf64>):\n";
  std::string value = "%v_i";
  if (removesMotion) {
    os << indent << "  %d = arith.subf %v_i, %vcm" << t
       << " : vector<3xf64>\n";
    value = "%d";
  }
  if (rescalesVelocities()) {
    os << indent << "  %ab = vector.broadcast %alpha" << t
       << " : f64 to vector<3xf64>\n"
       << indent << "  %s = arith.mulf %ab, " << value
       << " : vector<3xf64>\n";
    value = "%s";
  }
  if (control.barostat && !trotter) {
    os << indent << "  %v_scaled = arith.mulf %muinv" << t << ", " << value
       << " : vector<3xf64>\n";
    value = "%v_scaled";
  }
  os << indent << "  md.yield " << value << " : vector<3xf64>\n"
     << indent << "} : !vec\n";
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
  std::string evaluate = "md.evaluate @energy(%x1, %cell" + getFieldValues() +
                         getTimeValue("%zero") + ")";
  std::string signature =
      "(!vec, !md.cell" + getFieldTypes() + getTimeType() + ")";
  os << "dyn.program @descend(%x: !vec, %f: !vec, %m: !real, "
        "%cell: !md.cell,\n    %h: f64"
     << getFieldParameters() << ")\n    -> (!vec, !vec, f64)"
     << " attributes {effects = "
     << (hasHostEvaluation()
             ? "{read = [\"<Default>\"], write = [\"<Default>\"]}"
             : "{}")
     << "} {\n"
     << "  %zero = arith.constant 0.0 : f64\n"
     << "  %tiny = arith.constant 1.0e-300 : f64\n"
     << "  %count = arith.constant "
     << formatReal(static_cast<double>(system.getNumParticles())) << " : f64\n";
  emitConstrainedDescent("  ", "%x", "%f", "%g");
  os << "  %square = md.sum_particles gather(%g : !vec) {\n"
     << "  ^bb0(%f_i: vector<3xf64>):\n"
     << "    %sq = arith.mulf %f_i, %f_i : vector<3xf64>\n"
     << "    %s = vector.reduction <add>, %sq : vector<3xf64> into f64\n"
     << "    md.yield %s : f64\n"
     << "  } : f64\n"
     << "  %mean = arith.divf %square, %count : f64\n"
     << "  %mean_safe = arith.maximumf %mean, %tiny : f64\n"
     << "  %rms = math.sqrt %mean_safe : f64\n"
     << "  %power = md.sum_particles gather(%g : !vec) {\n"
     << "  ^bb0(%f_i: vector<3xf64>):\n"
     << "    %sq = arith.mulf %f_i, %f_i : vector<3xf64>\n"
     << "    %s = vector.reduction <add>, %sq : vector<3xf64> into f64\n"
     << "    %r2 = arith.divf %s, %mean_safe : f64\n"
     << "    %r4 = arith.mulf %r2, %r2 : f64\n"
     << "    %r8 = arith.mulf %r4, %r4 : f64\n"
     << "    %r16 = arith.mulf %r8, %r8 : f64\n"
     << "    md.yield %r16 : f64\n"
     << "  } : f64\n"
     << "  %root2 = math.sqrt %power : f64\n"
     << "  %root4 = math.sqrt %root2 : f64\n"
     << "  %root8 = math.sqrt %root4 : f64\n"
     << "  %root16 = math.sqrt %root8 : f64\n"
     << "  %norm = arith.mulf %rms, %root16 : f64\n"
     << "  %scale = arith.divf %h, %norm : f64\n"
     << "  %x1" << (sites || constraints ? "d" : "")
     << " = md.map_particles gather(%x, %g : !vec, !vec) {\n"
     << "  ^bb0(%x_i: vector<3xf64>, %f_i: vector<3xf64>):\n"
     << "    %sb = vector.broadcast %scale : f64 to vector<3xf64>\n"
     << "    %d = arith.mulf %sb, %f_i : vector<3xf64>\n"
     << "    %moved = arith.addf %x_i, %d : vector<3xf64>\n"
     << "    md.yield %moved : vector<3xf64>\n"
     << "  } : !vec\n";
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
    os << "  %intact = arith.cmpf oeq, %broken, %zero : f64\n"
       << "  " << (sites ? "%x1s" : "%x1")
       << " = md.map_particles gather(%x1k, %x : !vec, !vec) {\n"
       << "  ^bb0(%new_i: vector<3xf64>, %old_i: vector<3xf64>):\n"
       << "    %zero_i = arith.constant 0.0 : f64\n"
       << "    %intact_i = arith.cmpf oeq, %broken, %zero_i : f64\n"
       << "    %chosen = arith.select %intact_i, %new_i, %old_i "
          ": vector<3xf64>\n"
       << "    md.yield %chosen : vector<3xf64>\n"
       << "  } : !vec\n";
  }
  if (sites)
    emitPlaceSites("  ", constraints ? "%x1s" : "%x1d", "%x1", "%r_");
  std::string raw = sites ? "e" : "";
  std::string held = hasForceCorrections() ? "p" : "";
  // A learned potential adds its energy and forces to `%u1c` and `%f1c`
  // (D270).
  bool hostTerms = hasHostEvaluation();
  std::string modelled = !hostTerms || !held.empty() ? "" : "c";
  os << "  %u1" << held << modelled << ", %f1" << held << raw << modelled
     << " = " << evaluate << "\n      request [energy, forces]\n"
     << "      : " << signature << " -> (f64, !vec)\n";
  if (sites)
    emitSpreadSites("  ", "%x1", "%f1" + held + "e", "%f1" + held, "%r_");
  if (hasForceCorrections()) {
    os << "  %w1z = arith.constant dense<0.0> : vector<9xf64>\n";
    emitForceCorrections("  ", "%x1", "%p_", "%f1p", hostTerms ? "%f1c" : "%f1",
                   "%u1p", hostTerms ? "%u1c" : "%u1", "%w1z", "%w1r");
  }
  if (hostTerms)
    emitHostEvaluation("  ", "%x1", "%cell", "%p_", "%f1c", "%f1", "%u1c", "%u1",
                   "", "1");
  std::string energy = "%u1";
  if (constraints) {
    os << "  %never = arith.constant 0x7FF0000000000000 : f64\n"
       << "  %u1c = arith.select %intact, %u1, %never : f64\n";
    energy = "%u1c";
  }
  os << "  dyn.return %x1, %f1, " << energy << " : !vec, !vec, f64\n}\n\n";
}

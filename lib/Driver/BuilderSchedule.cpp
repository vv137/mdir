// The schedule of @mdir_run: the levels of its loops, the branches of
// the start, the parts of a program of segments, the frames, and the
// minimization (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"
#include "ScheduleIR.h"
#include "CouplingIR.h"
#include "EntryRuntimeIR.h"
#include "SimpleFrameEntryIR.h"
#include "mdir/Dialect/Dyn/DynOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "PotentialIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "BarostatIR.h"
#include "DynamicsIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

void Builder::emitStartBranch(std::string chunk,
                              llvm::ArrayRef<std::string> results,
                              llvm::ArrayRef<std::string> inside,
                              llvm::ArrayRef<std::string> otherwise,
                              bool segment) {
  if (chunk.empty() && results.empty())
    return;
  std::string types;
  for (size_t i = 0; i != results.size(); ++i)
    types += (i ? ", " : "") + std::string("!vec");
  // A loop of one iteration on the first call and none on the others: the
  // passes that place fields know the loops that carry them.
  if (!emittedFirstTrips) {
    emitIRFragment(
        "  ", {"%first_trips"}, {"index"}, {"%is_first", "%c1", "%c0"}, {"i1", "index", "index"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {mlir::arith::SelectOp::create(builder, builder.getUnknownLoc(), values[0], values[1], values[2])};
        });
    emittedFirstTrips = true;
  }
  os << "  ";
  if (!results.empty()) {
    llvm::interleaveComma(results, os);
    os << " = ";
  }
  std::string loop = "%i_first" + std::to_string(startBranches++);
  os << "scf.for " << loop << " = %c0 to %first_trips step %c1";
  if (!results.empty()) {
    os << "\n      iter_args(";
    for (size_t i = 0; i != otherwise.size(); ++i)
      os << (i ? ", " : "") << otherwise[i] << "_a = " << otherwise[i];
    os << ") -> (" << types << ")";
  }
  os << " {\n" << chunk;
  if (!results.empty()) {
    os << "    scf.yield ";
    llvm::interleaveComma(inside, os);
    os << " : " << types << "\n";
  }
  // The loop that carries values asks for its arguments in the place of
  // those values (carryLoopValues).
  std::vector<std::string> attributes;
  if (!otherwise.empty())
    attributes.push_back("mdir.carries");
  if (segment)
    attributes.push_back("mdrt.segment");
  os << "  }";
  if (!attributes.empty()) {
    os << " {";
    llvm::interleaveComma(attributes, os);
    os << "}";
  }
  os << "\n";
}

void Builder::emitLevel(unsigned level, StringRef indent) {
  // The state that the loops carry: positions, velocities, and forces.
  std::string state = "!vec, !vec, !vec";
  auto getValues = [&](const llvm::Twine &suffix) {
    std::string x = ("%x" + suffix).str(), v = ("%v" + suffix).str(),
                f = ("%f" + suffix).str();
    return x + ", " + v + ", " + f;
  };
  auto getInits = [&](const llvm::Twine &inside, const llvm::Twine &outside) {
    std::string text;
    for (StringRef name : {"x", "v", "f"}) {
      text += (text.empty() ? "" : ", ") +
              ("%" + name + inside + " = %" + name + outside).str();
    }
    return text;
  };
  // A segment begins with the particles in the order of their positions.
  auto isReordered = [&](unsigned index) {
    return program.reorders && levels[index].name == "segment";
  };

  std::string inner = (indent + "  ").str();
  std::string here = std::to_string(level);
  std::string outside = level == nestBegin
                            ? nestOutside
                            : isReordered(level - 1)
                                  ? "s"
                                  : "a" + std::to_string(level - 1);
  const Level &current = levels[level];
  bool isStepLoop = level + 1 == getNestEnd();
  bool reorders = isReordered(level);

  // With a new order in every iteration, the loop carries the fields that
  // do not change otherwise as well.
  std::string moreResults, moreInits, moreTypes, moreYielded;
  if (reorders) {
    moreResults = ", %me" + here;
    moreInits = ", %ma" + here + " = " + massName;
    moreTypes = ", !real";
    moreYielded = ", %ms";
    for (const Program::Field &field : program.fields) {
      moreResults += ", %pe" + here + "_" + field.name;
      moreInits +=
          ", %pa" + here + "_" + field.name + " = " + fieldPrefix + field.name;
      moreTypes += ", " + getFieldType(field).str();
      moreYielded += ", %ps_" + field.name;
    }
    moreResults += ", %ide" + here;
    moreInits += ", %ida" + here + " = " + idName;
    moreTypes += ", !ids";
    moreYielded += ", %ids";
  }

  os << indent << getValues("e" + here) << moreResults << " = scf.for %i"
     << here << " = %c0 to %n" << here << " step %c1\n"
     << indent << "    iter_args(" << getInits("a" + here, outside)
     << moreInits << ")\n"
     << indent << "    -> (" << state << moreTypes << ") {\n";

  // With a barostat the cell changes: each iteration takes it from where
  // the barostat keeps it.
  std::string outerCell = cellName, outerScale = scaleName;
  if (changesCell()) {
    cellName = "%cell" + here;
    emitBoxEdges(inner, "%edge" + here);
    emitCellOf(inner, cellName, "%edge" + here);
    if (scalesReference()) {
      scaleName = "%rest_scale" + here;
      emitReferenceScale(inner, scaleName, "%edge" + here + "_0",
                         "%edge" + here + "_1", "%edge" + here + "_2");
    }
  }

  std::string outerMass = massName, outerPrefix = fieldPrefix,
              outerId = idName;
  if (reorders) {
    emitReorder(inner, "a" + here, "s", "s", /*withForces=*/true, "%vs");
    massName = "%ms";
    fieldPrefix = "%ps_";
    idName = "%ids";
  }

  if (isStepLoop) {
    emitDynamicsStep(inner, "step", "%xa" + here, "%va" + here, "%fa" + here,
                         cellName, "", {"%xb" + here, "%vb" + here, "%fb" + here},
                         {"!vec", "!vec", "!vec"});
    os << inner << "scf.yield " << getValues("b" + here) << " : " << state
       << "\n";
  } else {
    emitLevel(level + 1, inner);
    std::string below = std::to_string(level + 1);
    std::string last = "e" + below;
    // The barostat may have changed the cell in the loop below; the steps
    // after it take the cell from where the barostat keeps it.
    if (changesCell()) {
      cellName = "%cellr" + here;
      emitBoxEdges(inner, "%edger" + here);
      emitCellOf(inner, cellName, "%edger" + here);
      if (scalesReference()) {
        scaleName = "%rest_scaler" + here;
        emitReferenceScale(inner, scaleName, "%edger" + here + "_0",
                           "%edger" + here + "_1", "%edger" + here + "_2");
      }
    }

    // Count the completed step from the enclosing loop state.
    auto emitStep = [&]() {
      bool split = current.name == "couple" && level != nestBegin &&
                   levels[level - 1].name == "energy";
      unsigned upto = split ? level - 1 : level;
      std::vector<std::string> inputs, signatures;
      for (unsigned i = nestBegin; i <= upto; ++i) {
        inputs.push_back("%i" + std::to_string(i)); signatures.push_back("index");
        if (i != nestBegin) { inputs.push_back("%n" + std::to_string(i)); signatures.push_back("index"); }
      }
      unsigned loops = upto - nestBegin + 1, tail = inputs.size();
      inputs.push_back("%i" + here); signatures.push_back("index");
      if (split) { inputs.push_back("%per" + std::to_string(level - 1)); signatures.push_back("index"); }
      inputs.push_back("%per" + here); signatures.push_back("index");
      inputs.push_back("%c1"); signatures.push_back("index");
      inputs.push_back(nestStart); signatures.push_back("i64");
      llvm::SmallVector<StringRef> names, types;
      for (const auto &name : inputs) names.push_back(name);
      for (const auto &type : signatures) types.push_back(type);
      emitIRFragment(
          inner, {"%step" + here}, {"i64"}, names, types,
          [loops, tail, split](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            llvm::SmallVector<LoopIterationIR> iterations;
            unsigned slot = 0;
            for (unsigned i = 0; i != loops; ++i) {
              mlir::Value index = values[slot++], extent;
              if (i) extent = values[slot++];
              iterations.push_back({index, extent});
            }
            unsigned at = tail;
            mlir::Value current = values[at++], outerPeriod;
            if (split) outerPeriod = values[at++];
            return {buildStepNumber(builder, iterations, current, outerPeriod,
                                    values[at], values[at + 1], values[at + 2])};
          });
    };
    // The velocities as the step after the coupling takes them, and the
    // state that the loop yields.
    // With leapfrog and a barostat, the coupling takes the velocities of
    // the time of the positions, `current`, and the stored ones are half a
    // kick behind those it leaves: the steps are then those of velocity
    // Verlet, and the energy that the coupling gives is that which the log
    // counts (D76).
    auto getCoupled = [&](StringRef x, StringRef v, StringRef f,
                          StringRef energy, StringRef trace,
                          StringRef groups, StringRef optimal,
                          StringRef current = "",
                          const TrotterScaling *trotter = nullptr) {
      Coupled coupled =
          emitCoupling(inner, x, current.empty() ? v : current, f, energy,
                       here, "%step" + here, trace, groups, optimal,
                       trotter);
      std::string velocities = coupled.velocities;
      if (!current.empty()) {
        std::string behind = "%vh" + here;
        emitKick(inner, behind, velocities, coupled.forces, massName, "%half_back");
        velocities = behind;
      }
      return coupled.positions + ", " + velocities + ", " + coupled.forces;
    };
    auto emitOptimalEnergy = [&](StringRef at, StringRef result,
                                 StringRef kinetic, StringRef excess) {
      emitIRFragment(
          at, {result}, {"f64"}, {kinetic, excess}, {"f64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value weight = mlir::arith::ConstantOp::create(
                builder, loc, builder.getF64FloatAttr(2.0 / 3.0));
            mlir::Value change = mlir::arith::MulFOp::create(builder, loc, weight, values[1]);
            return {mlir::arith::AddFOp::create(builder, loc, values[0], change)};
          });
    };
    // The kinetic energy of the optimal temperature at the end of a step,
    // K + (2/3)(K_half − K) [Jung2019], which the thermostat controls
    // (D263): from `full`, the velocities of the time of the
    // positions, and K_half measured (`measured`) with constraints or from
    // the forces `f` without. None without a thermostat that rescales.
    auto emitOptimal = [&](StringRef tag, StringRef full, StringRef measured,
                           StringRef f) -> std::string {
      if (!rescalesVelocities())
        return "";
      std::string o = ("%ko" + tag).str();
      std::string excess =
          emitKineticExcess(inner, ("o" + tag).str(), measured, f, full);
      if (excess.empty())
        return "";
      emitKineticEnergy(o + "k", full, massName, inner);
      emitSum3(o + "g", excess, inner);
      emitOptimalEnergy(inner, o, o + "k", o + "g");
      return o;
    };
    // The velocities of the time of the positions of a plain step of
    // leapfrog, half a kick ahead of those it stores.
    auto emitAhead = [&](StringRef tag, StringRef v, StringRef f) {
      std::string name = ("%vah" + tag).str();
      emitIRFragment(
          inner, {name}, {"!vec"}, {"%dt", v, f, massName},
          {"f64", "!vec", "!vec", "!real"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value half = mlir::arith::ConstantOp::create(
                builder, loc, builder.getF64FloatAttr(0.5));
            mlir::Value interval = mlir::arith::MulFOp::create(builder, loc, half, values[0]);
            return {buildKick(builder, values[1], values[2], values[3], interval)};
          });
      return name;
    };
    bool couplesBelow = levels[level + 1].name == "couple";
    // The last two steps of a period with the barostat of Trotter type
    // (D92): the step of energy whose pressure gives the strain, and the
    // step that scales the cell within its drift, in the new cell; their
    // results are named with `name`, as those of a step of energy, with
    // the kinetic energy of the scaled velocities `%kh<name>`. Returns the
    // scaling.
    auto emitTrotterSteps = [&](StringRef name, bool withEnergy) {
      std::string a = "j" + here, n = name.str();
      bool leapfrog = isLeapfrog();
      std::string kinetic, groups, trace, x, v;
      if (scalesEveryStep()) {
        // The pressure of the state that the step before left (D92): the
        // diagonals of the virial and of the virial of the groups, and the
        // kinetic energy of each axis.
        std::string m = "%tm" + here;
        std::string vectors[3];
        for (int v = 0; v != 3; ++v)
          vectors[v] = m + "_v" + std::to_string(v);
        emitIRFragment(
            inner, {vectors[0], vectors[1], vectors[2]},
            {"vector<3xf64>", "vector<3xf64>", "vector<3xf64>"},
            {"%trotter_memory"}, {"memref<9xf64>"},
            [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
              auto loc = builder.getUnknownLoc();
              llvm::SmallVector<mlir::Value> loaded;
              for (int slot = 0; slot != 9; ++slot) {
                mlir::Value index = mlir::arith::ConstantIndexOp::create(builder, loc, slot);
                loaded.push_back(mlir::memref::LoadOp::create(builder, loc, values[0], index));
              }
              auto vec = mlir::VectorType::get({3}, builder.getF64Type());
              TermIRValues result;
              for (int axis = 0; axis != 3; ++axis)
                result.push_back(mlir::vector::FromElementsOp::create(
                    builder, loc, vec, mlir::ValueRange(loaded).slice(3 * axis, 3)));
              return result;
            });
        trace = vectors[0];
        groups = vectors[1];
        kinetic = vectors[2];
        x = "%x" + last;
        v = "%v" + last;
      } else {
        // The step whose pressure gives the strain.
        std::vector<std::string> results{"%x" + a, "%v" + a, "%f" + a, "%w" + a, "%gw" + a};
        std::vector<std::string> types{"!vec", "!vec", "!vec", "vector<9xf64>", "vector<3xf64>"};
        if (leapfrog) { results.push_back("%vc" + a); types.push_back("!vec"); }
        llvm::SmallVector<StringRef> names, signatures;
        for (const auto &name : results) names.push_back(name);
        for (const auto &type : types) signatures.push_back(type);
        emitDynamicsStep(inner, "step_virial", "%x" + last, "%v" + last, "%f" + last, cellName, "", names, signatures);
        trace = "%dg" + a;
        emitDiagonal(trace, "%w" + a, inner);
        groups = "%gw" + a;
        x = "%x" + a;
        v = "%v" + a;
      }
      emitStep();
      TrotterScaling scaling =
          scalesEveryStep()
              ? emitTrotterStrain(inner, "", "", "", groups, here,
                                  "%step" + here, kinetic)
              : emitTrotterStrain(inner, "%x" + a,
                                  leapfrog ? "%vc" + a : "%v" + a,
                                  "%f" + a, groups, here, "%step" + here);
      std::string outerScale = scaleName;
      if (!scaling.scale.empty())
        scaleName = scaling.scale;
      bool virial = withEnergy || countsAfterScaling();
      bool measuredHere =
          measuresHalfSteps() && (withEnergy || measuresCoupling());
      std::vector<std::string> results{"%x" + n, "%v" + n, "%f" + n};
      std::vector<std::string> types{"!vec", "!vec", "!vec"};
      if (withEnergy) { results.push_back("%u" + n); types.push_back("f64"); }
      if (virial) { results.push_back("%w" + n); types.push_back("vector<9xf64>"); }
      if (virial) { results.push_back("%gw" + n); types.push_back("vector<3xf64>"); }
      if (leapfrog) { results.push_back("%vc" + n); types.push_back("!vec"); }
      if (true) { results.push_back("%kh" + n); types.push_back("vector<3xf64>"); }
      if (measuredHere) { results.push_back("%khs" + n); types.push_back("vector<3xf64>"); }
      if (withEnergy && reportsSolvent()) { results.push_back("%kws" + n); types.push_back("vector<3xf64>"); }
      llvm::SmallVector<StringRef> names, signatures;
      for (const auto &name : results) names.push_back(name);
      for (const auto &type : types) signatures.push_back(type);
      emitDynamicsStep(inner, withEnergy ? "step_trotter_energy" : "step_trotter", x, v, "%f" + (scalesEveryStep() ? last : a), scaling.cell, scaling.mu, names, signatures);
      scaleName = outerScale;
      scaling.kineticHalf = "%kh" + n;
      if (virial)
        scaling.groupsAfter = "%gw" + n;
      return scaling;
    };

    if (current.name == "couple") {
      // The last step of the period, and the coupling after it. The
      // barostat needs the virial of the step.
      std::string trace;
      if (usesTrotter()) {
        std::string k = "k" + here;
        TrotterScaling scaling = emitTrotterSteps(k, /*withEnergy=*/false);
        if (countsAfterScaling()) {
          trace = "%dg" + k;
          emitDiagonal(trace, "%w" + k, inner);
        }
        std::string optimal = emitOptimal(
            k, isLeapfrog() ? "%vc" + k : "%v" + k,
            measuresCoupling() ? "%khs" + k : "", "%f" + k);
        std::string coupled =
            getCoupled("%x" + k, "%v" + k, "%f" + k, "", trace, "", optimal,
                       isLeapfrog() ? "%vc" + k : "", &scaling);
        os << inner << "scf.yield " << coupled << " : " << state << "\n";
        os << indent << "}\n";
        cellName = outerCell;
        scaleName = outerScale;
        return;
      }
      if (control.barostat) {
        // With leapfrog the pressure takes the velocities of the time of
        // the positions, which the step of energy returns.
        std::string current = isLeapfrog() ? "%vck" + here : "";
        bool measured = measuresCoupling();
        std::vector<std::string> results{"%x" + ("k" + here), "%v" + ("k" + here), "%f" + ("k" + here), "%uk" + here, "%wk" + here, "%gwk" + here};
        std::vector<std::string> types{"!vec", "!vec", "!vec", "f64", "vector<9xf64>", "vector<3xf64>"};
        if (isLeapfrog()) { results.push_back(current); types.push_back("!vec"); }
        if (measured) { results.push_back("%khsk" + here); types.push_back("vector<3xf64>"); }
        llvm::SmallVector<StringRef> names, signatures;
        for (const auto &name : results) names.push_back(name);
        for (const auto &type : types) signatures.push_back(type);
        emitDynamicsStep(inner, measuresHalfSteps() ? "step_coupling" : "step_energy", "%x" + last, "%v" + last, "%f" + last, cellName, "", names, signatures);
        trace = "%dgk" + here;
        emitDiagonal(trace, "%wk" + here, inner);
        emitStep();
        std::string optimal = emitOptimal(
            "k" + here, isLeapfrog() ? current : "%vk" + here,
            measured ? "%khsk" + here : "", "%fk" + here);
        std::string coupled =
            getCoupled("%xk" + here, "%vk" + here, "%fk" + here,
                       "%uk" + here, trace, "%gwk" + here, optimal, current);
        os << inner << "scf.yield " << coupled << " : " << state << "\n";
        os << indent << "}\n";
        cellName = outerCell;
        scaleName = outerScale;
        return;
      }
      // With constraints the step that closes the period of the thermostat
      // measures the half steps (D263).
      bool measured = measuresCoupling();
      bool ahead = isLeapfrog() && rescalesVelocities();
      std::vector<std::string> results{"%x" + ("k" + here), "%v" + ("k" + here), "%f" + ("k" + here)};
      std::vector<std::string> types{"!vec", "!vec", "!vec"};
      if (measured && isLeapfrog()) { results.push_back("%vck" + here); types.push_back("!vec"); }
      if (measured) { results.push_back("%khsk" + here); types.push_back("vector<3xf64>"); }
      llvm::SmallVector<StringRef> names, signatures;
      for (const auto &name : results) names.push_back(name);
      for (const auto &type : types) signatures.push_back(type);
      emitDynamicsStep(inner, measured ? "step_thermostat" : "step", "%x" + last, "%v" + last, "%f" + last, cellName, "", names, signatures);
      emitStep();
      std::string full = "%vk" + here;
      if (ahead)
        full = measured ? "%vck" + here
                        : emitAhead("k" + here, "%vk" + here, "%fk" + here);
      std::string optimal = emitOptimal("k" + here, full,
                                        measured ? "%khsk" + here : "",
                                        "%fk" + here);
      std::string coupled = getCoupled("%xk" + here, "%vk" + here,
                                       "%fk" + here, "", "", "", optimal);
      os << inner << "scf.yield " << coupled << " : " << state << "\n";
      os << indent << "}\n";
      cellName = outerCell;
      scaleName = outerScale;
      return;
    }

    if (current.name == "energy" && couplesBelow) {
      // The last period of the interval: its steps but the last, which
      // the step of energy takes.
      std::string steps = std::to_string(getNestEnd() - 1);
      os << inner << getValues("q" + here) << " = scf.for %j" << here
         << " = %c0 to %n" << steps << " step %c1\n"
         << inner << "    iter_args(" << getInits("p" + here, last) << ")\n"
         << inner << "    -> (" << state << ") {\n";
      emitDynamicsStep(inner + "  ", "step", "%xp" + here, "%vp" + here, "%fp" + here,
                           cellName, "", {"%xr" + here, "%vr" + here, "%fr" + here},
                           {"!vec", "!vec", "!vec"});
      os << inner << "  scf.yield " << getValues("r" + here) << " : "
         << state << "\n"
         << inner << "}\n";
      last = "q" + here;
    }

    if (current.name == "energy") {
      // The last step of the interval, and the energies after it.
      // With leapfrog the step of energy also returns the velocities of
      // the time of the positions, which the kinetic energy takes.
      std::string virialName = "%w", energyName = "%u", now = "%vn";
      // With the barostat of Trotter type the last period ends with the
      // step that scales the cell (D92).
      bool scalesHere = usesTrotter() && couplesBelow;
      TrotterScaling scaling;
      if (scalesHere) {
        scaling = emitTrotterSteps("l", /*withEnergy=*/true);
        virialName = "%wl";
        energyName = "%ul";
        now = "%vcl";
      } else {
        std::vector<std::string> results{"%xl", "%vl", "%fl", "%u", "%w"};
        std::vector<std::string> types{"!vec", "!vec", "!vec", "f64", "vector<9xf64>"};
        if (!control.isBrownian()) { results.push_back("%gwl"); types.push_back("vector<3xf64>"); }
        if (isLeapfrog()) { results.push_back("%vn"); types.push_back("!vec"); }
        if (measuresHalfSteps()) { results.push_back("%khsl"); types.push_back("vector<3xf64>"); }
        if (reportsSolvent()) { results.push_back("%kwsl"); types.push_back("vector<3xf64>"); }
        llvm::SmallVector<StringRef> names, signatures;
        for (const auto &name : results) names.push_back(name);
        for (const auto &type : types) signatures.push_back(type);
        emitDynamicsStep(inner, "step_energy", "%x" + last, "%v" + last, "%f" + last, cellName, "", names, signatures);
      }
      std::string full = isLeapfrog() ? now : "%vl";
      emitKineticEnergy("%k", full, massName, inner);
      // K_half - K, measured from the velocities of the half steps with
      // constraints and from the forces without (D203).
      std::string excess = emitKineticExcess(
          inner, "l" + here, measuresHalfSteps() ? "%khsl" : "", "%fl", full);
      if (excess.empty())
        emitIRFragment(
            inner, {"%g"}, {"f64"}, {}, {},
            [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
              return {mlir::arith::ConstantOp::create(
                  builder, builder.getUnknownLoc(), builder.getF64FloatAttr(0.0))};
            });
      else
        emitSum3("%g", excess, inner);
      if (reportsSolvent())
        emitWriteSolvent(inner, "%kwsl");
      // The pressure of the log is that of the barostat, of the groups
      // (D264): the trace of their virial and the
      // kinetic energy of their centers at the half steps.
      if (control.isBrownian()) {
        emitTrace("%tr", virialName, inner);
        emitIRFragment(
            inner, {"%kph"}, {"f64"}, {"%k", "%g"}, {"f64", "f64"},
            [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
              return {mlir::arith::AddFOp::create(
                  builder, builder.getUnknownLoc(), values[0], values[1])};
            });
      } else {
        emitSum3("%tr", "%gwl", inner);
        GroupKinetic centers =
            emitGroupKinetic(inner, "%xl", full, "%fl", "l" + here);
        emitSum3("%kph", centers.half, inner);
      }
      if (couplesBelow && control.barostat)
        emitDiagonal("%dg", virialName, inner);
      if (!scalesHere)
        emitStep();
      emitIRFragment(
          inner, {}, {}, {"%step" + here, energyName, "%k", "%g", "%tr", "%kph"}, {"i64", "f64", "f64", "f64", "f64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            mlir::func::CallOp::create(builder, builder.getUnknownLoc(), "mdrtWriteEnergies",
                mlir::TypeRange{}, values);
            return {};
          });
      // The cell of the positions of the row: with the barostat of Trotter
      // type the step has scaled them, and they are in the cell after the
      // scaling, not in that of the start of the interval (#218). The
      // potentials of the outputs take the minimum image and the frame of
      // the cell from it.
      std::string rowCell = scalesHere ? scaling.cell : cellName;
      std::string outputStep = "%step" + here;
      for (const TermBuilder *term : getTermBuilders())
        if (const auto *role = term->getRoles().coordinateOutput)
          role->emitOutput(*this, {inner, "%xl", rowCell, fieldPrefix,
                                   outputStep, "", here, true});
      if (!control.freeEnergyFile.empty()) {
        // dH/dλ and the energies of the other states at the positions
        // after the step (D161).
        std::string time = "%fe_time" + here;
        if (control.usesTime)
          emitIRFragment(
              inner, {time}, {"f64"}, {"%step" + here, "%dt"}, {"i64", "f64"},
              [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
                return {buildSimulationTime(builder, values[0], values[1])};
              });
        emitFreeEnergyOutput(inner, "%xl", rowCell, fieldPrefix,
                             "%step" + here, time);
      }
      if (!control.observables.empty()) {
        // The observed terms at the positions after the step (D189).
        std::string time = "%ob_time" + here;
        if (control.usesTime)
          emitIRFragment(
              inner, {time}, {"f64"}, {"%step" + here, "%dt"}, {"i64", "f64"},
              [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
                return {buildSimulationTime(builder, values[0], values[1])};
              });
        emitObservablesOutput(inner, "%xl", rowCell, fieldPrefix,
                              "%step" + here, time);
      }
      // Without a periodic cell (D142), whether the particles have spread
      // so far that images interact, at every row of the log as well as at
      // the frames and checkpoints.
      if (!control.periodic && !framesAtEnergies)
        emitIRFragment(
          inner, {}, {}, {"%step" + here, "%xl", idName}, {"i64", "!vec", "!ids"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
                mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtCheckSpread"), values, mlir::ValueRange{}, mlir::UnitAttr{});
            return {};
          });
      std::string optimal;
      if (couplesBelow && rescalesVelocities() && !excess.empty()) {
        emitOptimalEnergy(inner, "%kol", "%k", "%g");
        optimal = "%kol";
      }
      std::string yielded =
          couplesBelow
              ? getCoupled("%xl", "%vl", "%fl", energyName,
                           control.barostat ? "%dg" : "",
                           control.barostat ? "%gwl" : "", optimal,
                           isLeapfrog() && control.barostat ? now : "",
                           scalesHere ? &scaling : nullptr)
              : getValues("l");
      // The frame of the positions that the interval leaves, as a loop
      // over frames writes them.
      if (framesAtEnergies)
        emitIRFragment(
          inner, {}, {}, {"%step" + here, StringRef(yielded).split(',').first, idName}, {"i64", "!vec", "!ids"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
                mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteFrame"), values, mlir::ValueRange{}, mlir::UnitAttr{});
            return {};
          });
      else if (control.segments)
        emitFrameIfDue(inner, "%step" + here,
                       StringRef(yielded).split(',').first, here);
      os << inner << "scf.yield " << yielded << " : " << state << "\n";
    } else {
      if (current.name == "frame") {
        emitStep();
        emitIRFragment(
          inner, {}, {}, {"%step" + here, "%x" + last, idName}, {"i64", "!vec", "!ids"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
                mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteFrame"), values, mlir::ValueRange{}, mlir::UnitAttr{});
            return {};
          });
      }
      if (current.name == "segment") {
        // The state as the next step needs it.
        emitStep();
        emitIRFragment(
          inner, {}, {}, {"%step" + here, "%x" + last, "%v" + last, "%f" + last, idName}, {"i64", "!vec", "!vec", "!vec", "!ids"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
                mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteCheckpoint"), values, mlir::ValueRange{}, mlir::UnitAttr{});
            return {};
          });
      }
      os << inner << "scf.yield " << getValues(last) << moreYielded << " : "
         << state << moreTypes << "\n";
    }
  }

  os << indent << "}";
  if (current.name == "segment")
    os << " {mdrt.segment}";
  else if (level == 0 && !control.segments)
    // The evaluation at the start is a segment of its own: the steps begin
    // with neighbor structures of their own, built at the first step, as
    // the first segment between checkpoints and the steps of a program of
    // segments do (D218).
    os << " {mdrt.fresh}";
  os << "\n";

  massName = outerMass;
  fieldPrefix = outerPrefix;
  idName = outerId;
  cellName = outerCell;
  scaleName = outerScale;
  // What the loop has left is in the order of its last iteration.
  if (reorders) {
    massName = "%me" + here;
    fieldPrefix = "%pe" + here + "_";
    idName = "%ide" + here;
    emitRenumber(indent, idName, "%re" + here + "_");
  }
}

void Builder::emitMinimization(StringRef velocities) {
  // The positions of the file on the surface of the constraints first. A
  // step is taken from positions on it and is constrained again, so a
  // start off it would put into every trial the change that takes the
  // groups to their shapes, which does not shrink with the step: where
  // the file has close contacts (a bilayer from packmol, bonds to
  // hydrogens 0.02 Å from their lengths) that change alone can raise the
  // energy, and no step is ever taken. A segment that continues a
  // minimization begins where the last one ended, on that surface
  // (D202).
  std::string x0 = "%x0";
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  os.flush();
  size_t projectionMark = program.module.size();
  if ((settles || !shakeSets.empty()) && !isRestart()) {
    std::string current = "%x0";
    unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
    std::string constrained = hasSites() ? "%x0ks" : "%x0k";
    auto next = [&]() {
      ++step;
      return step == steps ? constrained : "%x0k" + std::to_string(step);
    };
    if (settles) {
      std::string result = next();
      emitSettlePositions("  ", "%x0", current, "%dx0k", result);
      current = result;
    }
    for (const Program::TupleSet *set : shakeSets) {
      std::string result = next();
      emitShakePositions("  ", "%x0", current, *set, result);
      current = result;
    }
    if (hasSites())
      emitPlaceSites("  ", "%x0ks", "%x0k", "%r_");
    x0 = "%x0k";
    if (branchesStart()) {
      os.flush();
      std::string chunk = program.module.substr(projectionMark);
      program.module.resize(projectionMark);
      emitStartBranch(std::move(chunk), {"%x0c"}, {"%x0k"}, {"%x0"});
      x0 = "%x0c";
    }
  }
  // The energy and the forces at the start, and the terms.
  std::string raw = hasSites() ? "e" : "";
  std::string held = hasForceCorrections() ? "p" : "";
  bool hostTerms = hasHostEvaluation();
  std::string modelled = !hostTerms || !held.empty() ? "" : "c";
  emitEnergyEvaluation("  ", x0, "%cell", "%p_", "%time0",
                           "%u0" + held + modelled,
                           "%f0" + held + raw + modelled, "");
  if (hasSites())
    emitSpreadSites("  ", x0, "%f0" + held + "e", "%f0" + held, "%r_");
  if (hasForceCorrections()) {
    emitIRFragment(
        "  ", {"%w0z"}, {"vector<9xf64>"}, {}, {},
        [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          auto vec = mlir::VectorType::get({9}, builder.getF64Type());
          return {mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(),
                                                  mlir::DenseElementsAttr::get(vec, 0.0))};
        });
    emitForceCorrections("  ", x0, "%p_", "%f0p", hostTerms ? "%f0c" : "%f0", "%u0p",
                   hostTerms ? "%u0c" : "%u0", "%w0z", "%w0r");
  }
  // A learned potential (D270).
  if (hostTerms)
    emitHostEvaluation("  ", x0, "%cell", "%p_", "%f0c", "%f0", "%u0c", "%u0",
                   "", "0");
  // The terms at the start, which a continued segment does not log.
  if (branchesStart()) {
    os.flush();
    size_t termsMark = program.module.size();
    emitTerms(x0);
    os.flush();
    std::string chunk = program.module.substr(termsMark);
    program.module.resize(termsMark);
    emitStartBranch(std::move(chunk), {}, {}, {});
  } else if (!isRestart()) {
    emitTerms(x0);
  }
  // The length of the first step: that of the control file, or the one
  // that the last segment left.
  std::string h0 = control.segments ? "%first_size" : "%h0";
  if (!control.segments)
    emitIRFragment(
        "  ", {"%h0"}, {"f64"}, {}, {},
        [step = control.minimizeStep * units::length](
            mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          return {mlir::arith::ConstantOp::create(
              builder, builder.getUnknownLoc(), builder.getF64FloatAttr(step))};
        });
  // A step grows to at most 0.1 Å (D260): longer ones turn
  // rigid groups beyond what the constraints take back, and are not taken
  // (D259); on 103,292 OPC waters the largest force fell
  // below 5 kcal/mol/Å in 500 steps rather than 3300 with a cap of 1 Å.
  emitIRFragment(
      "  ", {"%h_most", "%grow", "%shrink", "%one", "%zero"},
      {"f64", "f64", "f64", "f64", "f64"}, {}, {},
      [cap = Control::minimizeStepCap * units::length](
          mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
        TermIRValues result;
        for (double value : {cap, 1.2, 0.2, 1.0, 0.0})
          result.push_back(mlir::arith::ConstantOp::create(
              builder, builder.getUnknownLoc(), builder.getF64FloatAttr(value)));
        return result;
      });
  // The log has the forces without their parts along the constraints,
  // those that the minimization lowers.
  auto emitReported = [&](StringRef indent, StringRef x, StringRef f,
                          StringRef tag) {
    std::string direction = ("%gr" + tag).str();
    emitConstrainedDescent(indent, x, f, direction);
    std::string reported = ("%fr" + tag).str();
    emitIRFragment(
        indent, {reported}, {"!vec"}, {direction, "%m"}, {"!vec", "!real"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          auto vec = mlir::VectorType::get({3}, builder.getF64Type());
          auto map = mdir::md::MapParticlesOp::create(
              builder, loc, values[0].getType(), values);
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto args = builder.createBlock(&map.getKernel(), {},
                {vec, builder.getF64Type()}, {loc, loc})->getArguments();
            mlir::Value mass = mlir::vector::BroadcastOp::create(builder, loc, vec, args[1]);
            mlir::Value force = mlir::arith::MulFOp::create(builder, loc, mass, args[0]);
            mdir::md::YieldOp::create(builder, loc, mlir::ValueRange{force});
          }
          return {map.getResult()};
        });
    return reported;
  };
  std::string reported = emitReported("  ", x0, "%f0", "0");
  emitIRFragment(
      "  ", {}, {}, {"%start", "%u0", h0, reported, "%id"},
      {"i64", "f64", "f64", "!vec", "!ids"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
            mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteMinimization"),
            values, mlir::ValueRange{}, mlir::UnitAttr{});
        return {};
      });
  // The host compares the largest force of each row with the tolerance
  // (D219), and the length of the next step with the floor
  // (D274), and keeps, in a buffer of its own, whether a
  // row met either and the step of that row (the last step otherwise).
  // The loops read it where an iteration begins: the steps and the rows
  // after that row are loops of no iteration. A row is where the host
  // waits for the device anyway, so the check takes no wait of its own.
  bool stops = stopsMinimization();
  if (stops) {
    emitIRFragment(
        "  ", {"%conv", "%conv_at0", "%conv_at1", "%conv_no"},
        {"memref<2xi64>", "index", "index", "i64"}, {"%start"}, {"i64"},
        [steps = control.numSteps](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto stop = buildMinimizationStop(builder, values[0], steps);
          return {stop.memory, stop.flagIndex, stop.stepIndex, stop.notDone};
        });
  }

  // A step is taken if it lowers the energy, and the next one is longer;
  // otherwise the next one is shorter, from where the step began. The
  // loops: over the intervals between frames (one if there are none),
  // over the intervals between energies in each, and over the steps.
  // A segment (D202) is one interval of the steps that
  // the entry takes, with a row at its end.
  os.flush();
  size_t partMark = program.module.size();
  std::string state = "!vec, !vec, f64, f64";
  if (control.segments) {
    emitIRFragment(
        "  ", {"%n0", "%n1", "%n2", "%per0"}, {"index", "index", "index", "index"},
        {"%count_outer", "%c0"}, {"i64", "index"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value n0 = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
          mlir::Value n1 = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
          mlir::Value n2 = mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(), values[0]);
          mlir::Value period = mlir::arith::AddIOp::create(builder, loc, n2, values[1]);
          return {n0, n1, n2, period};
        });
  } else {
    int64_t period = control.energyPeriod;
    int64_t framePeriod =
        control.framePeriod > 0 ? control.framePeriod : control.numSteps;
    std::array<int64_t, 4> counts{control.numSteps / framePeriod,
                                   framePeriod / period, period, framePeriod};
    emitIRFragment(
        "  ", {"%n0", "%n1", "%n2", "%per0"}, {"index", "index", "index", "index"}, {}, {},
        [counts](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          TermIRValues result;
          for (int64_t count : counts)
            result.push_back(mlir::arith::ConstantIndexOp::create(builder, builder.getUnknownLoc(), count));
          return result;
        });
  }
  os << "  %xe0, %fe0, %ue0, %he0 = scf.for %i0 = %c0 to %n0 step %c1\n"
     << "      iter_args(%xa0 = " << x0
     << ", %fa0 = %f0, %ua0 = %u0, %ha0 = " << h0 << ")\n"
     << "      -> (" << state << ") {\n"
     << "    %xe1, %fe1, %ue1, %he1 = scf.for %i1 = %c0 to %n1 step %c1\n"
     << "        iter_args(%xa1 = %xa0, %fa1 = %fa0, %ua1 = %ua0, "
        "%ha1 = %ha0)\n"
     << "        -> (" << state << ") {\n";
  // After the row that converged, no step and no row.
  std::string n2 = "%n2";
  if (stops) {
    emitIRFragment(
        "      ", {"%n2_open", "%row_open"}, {"index", "index"},
        {"%conv", "%conv_at0", "%conv_at1", "%conv_no", "%n2", "%c0", "%c1"},
        {"memref<2xi64>", "index", "index", "i64", "index", "index", "index"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto bounds = buildMinimizationBounds(builder,
              {values[0], values[1], values[2], values[3]}, values[4], values[5], values[6]);
          return {bounds.steps, bounds.rows};
        });
    n2 = "%n2_open";
  }
  os << "      %xe2, %fe2, %ue2, %he2 = scf.for %i2 = %c0 to " << n2
     << " step %c1\n"
     << "          iter_args(%xa2 = %xa1, %fa2 = %fa1, %ua2 = %ua1, "
        "%ha2 = %ha1)\n"
     << "          -> (" << state << ") {\n";
  PotentialConnections descentFields(program, "%xa2", "%cell", "%p_", "", false);
  std::vector<std::string> descentInputs{"%xa2", "%fa2", "%m", "%cell", "%ha2"};
  std::vector<std::string> descentTypes{"!vec", "!vec", "!real", "!md.cell", "f64"};
  for (unsigned slot = 2; slot != descentFields.values.size(); ++slot) {
    descentInputs.push_back(descentFields.values[slot]);
    descentTypes.push_back(descentFields.types[slot]);
  }
  llvm::SmallVector<StringRef> descentOperands, descentSignatures;
  for (const auto &input : descentInputs) descentOperands.push_back(input);
  for (const auto &type : descentTypes) descentSignatures.push_back(type);
  emitProgramCall("        ", "descend", {"%xt", "%ft", "%ut"},
                     {"!vec", "!vec", "f64"}, descentOperands, descentSignatures);
  emitIRFragment(
      "        ", {"%xb", "%fb", "%ub", "%hb"}, {"!vec", "!vec", "f64", "f64"},
      {"%ut", "%ua2", "%ha2", "%grow", "%h_most", "%shrink", "%one", "%zero",
       "%xt", "%xa2", "%ft", "%fa2"},
      {"f64", "f64", "f64", "f64", "f64", "f64", "f64", "f64",
       "!vec", "!vec", "!vec", "!vec"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        mlir::Value lower = mlir::arith::CmpFOp::create(
            builder, loc, mlir::arith::CmpFPredicate::OLT, values[0], values[1]);
        mlir::Value longer = mlir::arith::MulFOp::create(builder, loc, values[2], values[3]);
        mlir::Value capped = mlir::arith::MinimumFOp::create(builder, loc, longer, values[4]);
        mlir::Value shorter = mlir::arith::MulFOp::create(builder, loc, values[2], values[5]);
        mlir::Value energy = mlir::arith::SelectOp::create(builder, loc, lower, values[0], values[1]);
        mlir::Value interval = mlir::arith::SelectOp::create(builder, loc, lower, capped, shorter);
        mlir::Value keep = mlir::arith::SelectOp::create(builder, loc, lower, values[6], values[7]);
        auto vec = mlir::VectorType::get({3}, builder.getF64Type());
        TermIRValues fields;
        for (unsigned slot : {8u, 10u}) {
          auto map = mdir::md::MapParticlesOp::create(
              builder, loc, values[slot].getType(), values.slice(slot, 2));
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto args = builder.createBlock(&map.getKernel(), {}, {vec, vec}, {loc, loc})->getArguments();
            mlir::Value half = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.5));
            mlir::Value taken = mlir::arith::CmpFOp::create(
                builder, loc, mlir::arith::CmpFPredicate::OGT, keep, half);
            mlir::Value chosen = mlir::arith::SelectOp::create(builder, loc, taken, args[0], args[1]);
            mdir::md::YieldOp::create(builder, loc, mlir::ValueRange{chosen});
          }
          fields.push_back(map.getResult());
        }
        return {fields[0], fields[1], energy, interval};
      });
  os << "        scf.yield %xb, %fb, %ub, %hb : " << state << "\n"
     << "      }\n";
  emitIRFragment(
      "      ", {"%step"}, {"i64"}, {"%i0", "%per0", "%i1", "%c1", "%n2", "%start"},
      {"index", "index", "index", "index", "index", "i64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        LoopIterationIR loop{values[0], {}};
        return {buildStepNumber(builder, {loop}, values[2], values[1], values[4], values[3], values[5])};
      });
  // A row, unless an earlier one converged: a loop of one iteration or
  // none, as emitFrameIfDue.
  std::string rowIndent = stops ? "        " : "      ";
  if (stops)
    os << "      scf.for %i_row = %c0 to %row_open step %c1 {\n";
  reported = emitReported(rowIndent, "%xe2", "%fe2", "1");
  emitIRFragment(
      rowIndent, {}, {}, {"%step", "%ue2", "%he2", reported, "%id"},
      {"i64", "f64", "f64", "!vec", "!ids"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
            mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteMinimization"),
            values, mlir::ValueRange{}, mlir::UnitAttr{});
        return {};
      });
  if (stops) {
    emitIRFragment(
        "        ", {}, {}, {"%conv"}, {"memref<2xi64>"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
              mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtCheckMinimization"),
              values, mlir::ValueRange{}, mlir::UnitAttr{});
          return {};
        });
    os << "      }\n";
  }
  os << "      scf.yield %xe2, %fe2, %ue2, %he2 : " << state << "\n"
     << "    }\n";
  if (control.framePeriod > 0) {
    emitIRFragment(
        "    ", {"%frame_step"}, {"i64"}, {"%i0", "%c1", "%per0", "%start"},
        {"index", "index", "index", "i64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          LoopIterationIR loop{values[0], {}};
          return {buildStepNumber(builder, {loop}, {}, {}, values[2], values[1], values[3])};
        });
    std::string frameIndent = "    ";
    if (stops) {
      // The frame at the row that converged, and none after it.
      emitIRFragment(
          "    ", {"%conv_fn"}, {"index"},
          {"%conv", "%conv_at0", "%conv_at1", "%conv_no", "%frame_step", "%c0", "%c1"},
          {"memref<2xi64>", "index", "index", "i64", "i64", "index", "index"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildMinimizationFrameBound(builder,
                {values[0], values[1], values[2], values[3]}, values[4], values[5], values[6])};
          });
      os << "    scf.for %i_frame = %c0 to %conv_fn step %c1 {\n";
      frameIndent = "      ";
    }
    emitIRFragment(
        frameIndent, {}, {}, {"%frame_step", "%xe1", "%id"}, {"i64", "!vec", "!ids"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
              mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteFrame"),
              values, mlir::ValueRange{}, mlir::UnitAttr{});
          return {};
        });
    if (stops)
      os << "    }\n";
  }
  os << "    scf.yield %xe1, %fe1, %ue1, %he1 : " << state << "\n"
     << "  }\n";
  if (control.segments) {
    // The parts of the minimization continue one another in one
    // activation (D215).
    os.flush();
    std::string chunk = program.module.substr(partMark);
    program.module.resize(partMark);
    emitPartLoop(std::move(chunk), {x0, "%f0", "%u0", h0},
                 {"!vec", "!vec", "f64", "f64"},
                 {"%xe0", "%fe0", "%ue0", "%he0"},
                 {x0, velocities.str(), "%f0"});
    os << "  return\n}\n";
    return;
  }
  // The checkpoint holds the positions, and velocities of 0, at the last
  // step or at the row that converged.
  if (control.checkpointPeriod > 0) {
    if (stops)
      emitIRFragment(
          "  ", {"%end"}, {"i64"}, {"%conv", "%conv_at1"}, {"memref<2xi64>", "index"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(), values[0], values[1])};
          });
    else
      emitIRFragment(
          "  ", {"%end"}, {"i64"}, {"%start"}, {"i64"},
          [steps = control.numSteps](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value count = mlir::arith::ConstantIntOp::create(builder, loc, steps, 64);
            return {mlir::arith::AddIOp::create(builder, loc, values[0], count)};
          });
    emitIRFragment(
        "  ", {}, {}, {"%end", "%xe0", "%v0", "%id"}, {"i64", "!vec", "!vec", "!ids"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
              mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteCheckpoint"),
              values, mlir::ValueRange{}, mlir::UnitAttr{});
          return {};
        });
  }
  emitIRFragment(
      "  ", {}, {}, {"%xe0", "%v0", "%id"}, {"!vec", "!vec", "!ids"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
            mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtFinish"),
            values, mlir::ValueRange{}, mlir::UnitAttr{});
        return {};
      });
  os << "  return\n}\n";
}


llvm::Error Builder::emitEntry() {
  collectStartValues();
  StringRef state = getName(program.state);
  StringRef force = getName(program.force);
  StringRef mass = getName(program.mass);
  StringRef parameter = getName(program.parameter);

  RuntimeDeclarationConfig declarations{
      program.state == Element::F64, program.force == Element::F64,
      program.tunableGradient, bool(control.frames), control.segments,
      static_cast<int64_t>(getPartCountsSize()), bool(control.minimize),
      stopsMinimization(), reportsSolvent(),
      hasConstraints() && !control.minimize, rescalesVelocities(),
      control.isNoseHoover(), control.getCouplingPeriod() > 0,
      bool(control.barostat), scalesEveryStep(), control.checkpointPeriod > 0,
      program.writesForces};
  // Each selected entry is complete native IR, including every loop body.
  // Other physical families keep their existing entry until their scopes move.
  bool nativeFrames = control.frames && !control.minimize &&
      !isRestart() &&
      !hasHostEvaluation();
  if (nativeFrames) {
    mlir::OpBuilder ir(&termIR.getContext());
    auto *moduleBody = termIR.getModule().getBody();
    ir.setInsertionPointToEnd(moduleBody);
    size_t before = std::distance(moduleBody->begin(), moduleBody->end());
    emitRuntimeDeclarations(ir, declarations, false);
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().hostEvaluation)
        role->buildDeclarations(*this, ir);
    emitRuntimeDeclarations(ir, declarations, true);
    SimpleFrameEntryIRConfig config;
    config.branchesStart = branchesStart();
    config.givenForces = branchesStart() && program.takesForces;
    config.timed = control.usesTime;
    config.stepMemory = needsStepNumber();
    config.leapfrog = isLeapfrog();
    config.brownian = control.isBrownian();
    config.coupled = control.getCouplingPeriod() > 0;
    config.barostat = bool(control.barostat);
    config.changesCell = changesCell();
    config.scalesEveryStep = scalesEveryStep();
    config.removesMotion = control.comPeriod > 0;
    if (scalesReference()) {
      std::array<unsigned, 3> slots;
      for (unsigned index = 0; index != 3; ++index) {
        std::string name = "rest_edge" + std::to_string(index);
        auto found = llvm::find_if(program.startValues,
            [&](const auto &value) { return value.name == name; });
        assert(found != program.startValues.end() && "reference edge must be an entry value");
        slots[index] = std::distance(program.startValues.begin(), found);
      }
      config.referenceEdges = slots;
    }
    if (config.coupled && control.thermostat) {
      double freedom = system.getDegreesOfFreedom();
      double target = 0.5 * freedom * units::boltzmann * control.temperature;
      double decay = std::exp(-static_cast<double>(control.getCouplingPeriod()) *
          control.timestep / control.tauT);
      config.thermostat = FrameThermostatIR{static_cast<int64_t>(control.seed), freedom, target, decay};
      if (control.barostat) {
        double bar = 1.01325;
        config.barostatParameters = BarostatParameters{
            {control.compressibilities[0] / bar, control.compressibilities[1] / bar,
             control.compressibilities[2] / bar},
            control.pressure * bar, control.compressibility / bar,
            static_cast<double>(control.barostatPeriod) * control.timestep / control.tauP,
            units::boltzmann * control.temperature, control.compressibilityZ / bar,
            control.surfaceTension * control.surfaces * units::dynePerCmToBarNm};
      }
    }
    config.cellEdges = getCellSize();
    config.topology = bool(system.topology);
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().postEvaluation)
        config.storedTerms += role->getStoredTermCount(*this);
    for (const Program::TupleSet *selected : getSiteSets()) {
      unsigned slot = program.fields.size() + program.tables.size();
      for (const auto &set : program.tupleSets) {
        if (&set == selected) break;
        slot += 1 + set.fields.size();
      }
      auto fieldSlot = [&](StringRef name) {
        auto found = llvm::find_if(selected->fields,
            [&](const auto &field) { return field.name == name; });
        assert(found != selected->fields.end() && "site parameter must exist");
        return slot + 1 + unsigned(std::distance(selected->fields.begin(), found));
      };
      config.sites.push_back({slot, fieldSlot("a"), fieldSlot("b"),
          selected->name == "sites_amber", spreadsSitesDerived(), "spread_" + selected->name});
    }
    auto groups = getShakeSets();
    for (const auto &set : program.tupleSets)
      if (set.name == "settles") groups.insert(groups.begin(), &set);
    for (const Program::TupleSet *selected : groups) {
      unsigned slot = program.fields.size() + program.tables.size();
      for (const auto &set : program.tupleSets) {
        if (&set == selected) break;
        slot += 1 + set.fields.size();
      }
      auto fieldSlot = [&](StringRef name) {
        auto found = llvm::find_if(selected->fields,
            [&](const auto &field) { return field.name == name; });
        assert(found != selected->fields.end() && "constraint parameter must exist");
        return slot + 1 + unsigned(std::distance(selected->fields.begin(), found));
      };
      FrameConstraintIR group{slot, selected->arity, {}, selected->name == "settles"};
      if (group.water) {
        group.parameters = {fieldSlot("doh"), fieldSlot("dhh")};
      } else {
        for (unsigned member = 1; member != selected->arity; ++member)
          group.parameters.push_back(fieldSlot("d" + std::to_string(member)));
      }
      config.constraints.push_back(std::move(group));
    }
    if (config.topology)
      for (const auto &definition : getTermDefinitions())
        config.terms.push_back(definition.name);
    if (!control.freeEnergyFile.empty()) {
      config.freeEnergy.emplace(FreeEnergyOutputConfig{
          {}, static_cast<size_t>(control.freeEnergy.state), getNumPotentialArguments()});
      for (const auto &component : control.freeEnergy.lambdas)
        config.freeEnergy->lambdas.push_back(component.second);
    }
    config.observableCount = control.observables.size();
    config.observableBaseArgument = getNumPotentialArguments();
    const auto observedTerms = getObservedTerms();
    for (auto [index, term] : llvm::enumerate(observedTerms)) {
      ObservableOutputTermIR output;
      output.potential = "observe" + std::to_string(index);
      for (const auto &constant : term.constants) {
        bool tunable = control.isObservedTunable(term.name, constant);
        std::optional<unsigned> slot;
        if (tunable) {
          slot = config.observableTunables.size();
          std::string name = "obv_" + term.name + "_" + constant;
          auto found = llvm::find_if(program.startValues,
              [&](const auto &value) { return value.name == name; });
          assert(found != program.startValues.end() && "observed tunable must be an entry value");
          config.observableTunables.push_back(std::distance(program.startValues.begin(), found));
        }
        output.constants.push_back({tunable ? 0.0 : getObservedValue(term, constant), slot});
      }
      for (auto column : term.columns)
        output.columns.push_back({column.first, column.second});
      config.observables.push_back(std::move(output));
    }
    if (program.tunableGradient) {
      auto &gradient = config.gradient.emplace();
      auto startSlot = [&](StringRef name) -> unsigned {
        auto found = llvm::find_if(program.startValues,
            [&](const auto &value) { return value.name == name; });
        assert(found != program.startValues.end() && "gradient input must be an entry value");
        return std::distance(program.startValues.begin(), found);
      };
      gradient.request = startSlot("tunable_gradient");
      size_t constants = control.tunableConstants.size(), seeds = getNumGradientSeeds();
      for (size_t index = 0; index != constants; ++index)
        gradient.constants.push_back(startSlot("tg_c" + std::to_string(index)));
      gradient.seeds = seeds;
      if (seeds) {
        auto found = llvm::find_if(program.fields,
            [](const auto &field) { return field.name == "tg_zero"; });
        assert(found != program.fields.end() && "gradient seed field must exist");
        gradient.seedField = std::distance(program.fields.begin(), found);
      }
      unsigned base = getNumPotentialArguments(), chargeArgument = 2;
      for (const auto &field : program.fields) {
        if (field.name == "q") break;
        ++chargeArgument;
      }
      program.gradientArguments.clear();
      for (auto [index, slot] : llvm::enumerate(program.gradientSlots)) {
        unsigned argument = base + gradientColumns[index];
        gradient.output.scalarArguments.push_back(argument);
        program.gradientArguments.push_back({argument, slot.tunable});
      }
      for (auto [index, field] : llvm::enumerate(program.gradientFields)) {
        unsigned argument = index < seeds ? base + constants + index : chargeArgument;
        gradient.output.fieldArguments.push_back(argument);
        program.gradientArguments.push_back({argument, field.tunable});
      }
    }
    if (!control.segments) {
      for (const auto &level : levels) config.counts.push_back(level.count);
      int64_t steps = 1;
      for (unsigned index = levels.size(); index-- != 0;) {
        config.periods.push_back(steps);
        steps *= levels[index].count;
        if (index && levels[index - 1].name == "couple")
          steps += usesTrotter() && !scalesEveryStep() ? 2 : 1;
        else if (index && levels[index - 1].name == "energy")
          steps += levels[index].name == "couple" ? control.getCouplingPeriod() : 1;
      }
    }
    auto entry = buildSimpleFrameEntry(ir, program, config,
        [&](mlir::OpBuilder &at, const PotentialStateIR &state, EnergyResults evaluation,
            mlir::Value scale) {
          llvm::SmallVector<mlir::Value> stored;
          auto field = [&](StringRef name) -> mlir::Value {
            auto found = llvm::find_if(program.fields,
                [&](const auto &field) { return field.name == name; });
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
              llvm::append_range(stored, result.storedTerms);
            }
          return FrameCompletionIR{evaluation, std::move(stored)};
        }, [&](mlir::OpBuilder &at, const PotentialStateIR &state, mlir::Value step) {
          auto tuple = [&](StringRef name) {
            unsigned offset = program.fields.size() + program.tables.size();
            for (const auto &set : program.tupleSets) {
              if (set.name == name) return std::make_pair(&set, offset);
              offset += 1 + set.fields.size();
            }
            llvm_unreachable("output relation must exist");
          };
          auto relation = [&](StringRef name) { return state.fields[tuple(name).second]; };
          auto field = [&](StringRef set, StringRef name) {
            auto [metadata, offset] = tuple(set);
            auto found = llvm::find_if(metadata->fields,
                [&](const auto &field) { return field.name == name; });
            assert(found != metadata->fields.end() && "output parameter must exist");
            return state.fields[offset + 1 + std::distance(metadata->fields.begin(), found)];
          };
          TermNativeOutput point{state, step, relation, field};
          for (const TermBuilder *term : getTermBuilders())
            if (const auto *role = term->getRoles().coordinateOutput)
              role->buildOutput(*this, at, point);
        });
    if (!entry) return entry.takeError();
    llvm::SmallVector<mlir::Operation *> operations;
    for (auto &operation : llvm::drop_begin(*moduleBody, before))
      operations.push_back(&operation);
    emitIRFragment("", {}, {}, {}, {},
        [operations = std::move(operations)](mlir::OpBuilder &where, mlir::ValueRange) -> TermIRValues {
          for (auto *operation : operations)
            operation->moveBefore(where.getInsertionBlock(), where.getInsertionPoint());
          return {};
        });
    return llvm::Error::success();
  }
  emitIRFragment("", {}, {}, {}, {},
                 [declarations](mlir::OpBuilder &builder, mlir::ValueRange) {
                   emitRuntimeDeclarations(builder, declarations, false);
                   return TermIRValues{};
                 });
  auto emitIR = [this](StringRef indent, llvm::ArrayRef<StringRef> results,
                       llvm::ArrayRef<StringRef> resultTypes,
                       llvm::ArrayRef<StringRef> operands,
                       llvm::ArrayRef<StringRef> operandTypes,
                       TermIRRecipe recipe) {
    emitIRFragment(indent, results, resultTypes, operands, operandTypes,
                   std::move(recipe));
  };
  for (const TermBuilder *builder : getTermBuilders())
    if (const TermHostEvaluation *role = builder->getRoles().hostEvaluation)
      role->emitDeclarations(*this, emitIR);
  emitIRFragment("", {}, {}, {}, {},
                 [declarations](mlir::OpBuilder &builder, mlir::ValueRange) {
                   emitRuntimeDeclarations(builder, declarations, true);
                   return TermIRValues{};
                 });
  os << "\n";

  os << "func.func @" << program.entry << "(\n"
     << "    %positions: memref<?x3x" << state << ">, %velocities: memref<?x3x"
     << state << ">,\n";
  if (program.takesForces)
    os << "    %forces: memref<?x3x" << force << ">,\n";
  // The buffers of the model are read and never written
  // (D258): `md-exec-assign-storage` takes none of them for
  // another field or for a copy back from a device, and
  // `check-read-only-arguments` fails the compile where the lowered entry
  // could write one. A simulation hands them to every start without a copy.
  StringRef readOnly = " {mdrt.read_only}";
  os << "    %masses: memref<?x" << mass << ">" << readOnly;
  for (const Program::Field &field : program.fields)
    os << ", %b_" << field.name << ": memref<?x"
       << (field.isInteger ? StringRef("i32") : parameter) << ">" << readOnly;
  for (const Program::Table &table : program.tables)
    os << ", %bt_" << table.name << ": memref<?x?xf64>" << readOnly;
  for (const Program::TupleSet &set : program.tupleSets) {
    os << ",\n    %br_" << set.name << ": memref<?x" << set.arity << "xi32>"
       << readOnly;
    for (const Program::Field &field : set.fields)
      os << ", %bf_" << set.name << "_" << field.name << ": memref<?xf64>"
         << readOnly;
  }
  os << ",\n    %identities: memref<?xi32>,\n"
     << "    %lx: f64, %ly: f64, %lz: f64, %dt: f64, %start: i64";
  // The counts of the loops of a program of segments (D196):
  // the iterations of the outer loop, the plain steps in an iteration of the
  // loop over the periods of coupling, and plain steps after the loops; then
  // whether the segment ends with a plain step of energy, and whether it
  // ends with a period that closes with a step of energy, with its plain
  // steps.
  // The counts of the loops of each part come from the host at the end
  // of the part before it (emitPartLoop). A minimization in segments takes
  // the length of its first step (D202).
  if (control.segments && control.minimize)
    os << ", %first_size: f64";
  // Whether the call begins the run, nonzero, or continues the state that
  // the last call left (D211).
  if (branchesStart())
    os << ", %first_call: i64";
  for (const Program::StartValue &value : program.startValues)
    os << ", %" << value.name << ": f64";
  os << ") {\n";

  emitIRFragment(
      "  ", {"%c0", "%c1"}, {"index", "index"}, {}, {},
      [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
        mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
        return {zero, one};
      });
  if (branchesStart())
    emitIRFragment(
        "  ", {"%is_first"}, {"i1"}, {"%first_call"}, {"i64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value none = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
          return {mlir::arith::CmpIOp::create(
              builder, loc, mlir::arith::CmpIPredicate::ne, values[0], none)};
        });
  // The time of the first step, which the evaluations at the start take
  // (D145); a minimization takes 0.
  if (control.usesTime) {
    if (control.minimize)
      emitIRFragment(
          "  ", {"%time0"}, {"f64"}, {}, {},
          [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
            return {mlir::arith::ConstantOp::create(
                builder, builder.getUnknownLoc(), builder.getF64FloatAttr(0.0))};
          });
    else
      emitIRFragment(
          "  ", {"%time0"}, {"f64"}, {"%start", "%dt"}, {"i64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildSimulationTime(builder, values[0], values[1])};
          });
  }
  // The number of the last step taken, which keys the random numbers of
  // Langevin dynamics (D135).
  if (needsStepNumber())
    emitIRFragment(
        "  ", {"%noise_memory", "%noise_one"}, {"memref<1xi64>", "i64"},
        {"%start", "%c0"}, {"i64", "index"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value memory = mlir::memref::AllocaOp::create(
              builder, loc, mlir::MemRefType::get({1}, builder.getI64Type()));
          mlir::memref::StoreOp::create(builder, loc, values[0], memory, values[1]);
          mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
          return {memory, one};
        });
  // A minimization has loops of its own (emitMinimization), and a program
  // of segments takes its counts in the loop over parts (emitPartCounts).
  if (!control.segments) {
    for (auto [index, level] : llvm::enumerate(levels)) {
      std::string result = "%n" + std::to_string(index);
      emitIRFragment(
          "  ", {result}, {"index"}, {}, {},
          [count = level.count](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
            return {mlir::arith::ConstantIndexOp::create(builder, builder.getUnknownLoc(), count)};
          });
    }
  }
  // The number of steps in one iteration of each loop. An iteration of
  // the loop over energy intervals takes one step after its loop over
  // steps.
  // An iteration of the loop over the periods of coupling takes one as
  // well (two with the barostat of Trotter type), and one over energy intervals a whole period after its loop over
  // periods.
  int64_t steps = 1;
  for (unsigned i = control.segments ? 0 : levels.size(); i-- != 0;) {
    std::string result = "%per" + std::to_string(i);
    emitIRFragment(
        "  ", {result}, {"index"}, {}, {},
        [steps](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          return {mlir::arith::ConstantIndexOp::create(builder, builder.getUnknownLoc(), steps)};
        });
    steps *= levels[i].count;
    if (i == 0)
      continue;
    // With the barostat of Trotter type, two steps (setSchedule).
    if (levels[i - 1].name == "couple")
      steps += usesTrotter() && !scalesEveryStep() ? 2 : 1;
    else if (levels[i - 1].name == "energy")
      steps += levels[i].name == "couple" ? control.getCouplingPeriod() : 1;
  }

  // A field's logical element type is independent of its storage precision.
  auto emitFieldBuffer = [&](StringRef result, StringRef input,
                             StringRef storage, StringRef signature,
                             unsigned components, bool integer,
                             StringRef set = "atoms") {
    emitIRFragment(
        "  ", {result}, {signature}, {input}, {storage},
        [components, integer, set = set.str()](mlir::OpBuilder &builder,
                                               mlir::ValueRange values) -> TermIRValues {
          mlir::Type element = integer ? mlir::Type(builder.getI32Type())
                                       : mlir::Type(builder.getF64Type());
          auto type = mdir::md::FieldType::get(
              builder.getContext(), mlir::FlatSymbolRefAttr::get(builder.getContext(), set),
              components, element);
          return {mdir::mdrt::FromBufferOp::create(builder, builder.getUnknownLoc(), type, values[0])};
        });
  };
  // The fields as the buffers hold them, and in the order of the positions
  // if the run keeps that order.
  // A program of segments takes the forces and the velocities of the last
  // segment as %fg and %vg, which a later call continues from.
  bool givenForces = (isRestart() || branchesStart()) && program.takesForces;
  std::string velocities =
      (branchesStart() && !control.minimize) ||
              (isLeapfrog() && !isRestart())
          ? "%vg"
          : "%v0";
  std::string givenForcesName =
      branchesStart() ? "%fg" : "%f0";
  std::string given = program.reorders ? "_in" : "";
  if (changesCell()) {
    static const char *const initial[] = {"%lx", "%ly", "%lz", "%tilt_bx", "%tilt_cx", "%tilt_cy"};
    unsigned edgeCount = getCellSize();
    std::vector<std::string> outputs{"%box_memory"}, signatures{getBoxMemoryType()};
    if (scalesEveryStep()) {
      outputs.push_back("%trotter_memory");
      signatures.push_back("memref<9xf64>");
    }
    for (unsigned k = 0; k != edgeCount; ++k) {
      outputs.push_back("%c_edge" + std::to_string(k));
      signatures.push_back("index");
    }
    llvm::SmallVector<StringRef> names, types;
    for (const auto &name : outputs) names.push_back(name);
    for (const auto &type : signatures) types.push_back(type);
    llvm::SmallVector<StringRef> edges(initial, initial + edgeCount);
    llvm::SmallVector<StringRef> edgeTypes(edgeCount, "f64");
    emitIRFragment(
        "  ", names, types, edges, edgeTypes,
        [edgeCount, trotter = scalesEveryStep()](mlir::OpBuilder &builder,
                                                  mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value memory = mlir::memref::AllocaOp::create(
              builder, loc, mlir::MemRefType::get({int64_t(edgeCount)}, builder.getF64Type()));
          TermIRValues results{memory};
          if (trotter)
            results.push_back(mlir::memref::AllocaOp::create(
                builder, loc, mlir::MemRefType::get({9}, builder.getF64Type())));
          for (unsigned k = 0; k != edgeCount; ++k) {
            mlir::Value index = mlir::arith::ConstantIndexOp::create(builder, loc, k);
            mlir::memref::StoreOp::create(builder, loc, values[k], memory, index);
            results.push_back(index);
          }
          return results;
        });
  }
  if (scalesReference())
    // The edges of the cell of the file, which the reference positions of
    // the restraints are for, arguments of the entry (collectStartValues);
    // they scale with the cell, axis by axis.
    emitIRFragment(
        "  ", {"%rest_edges"}, {"vector<3xf64>"},
        {"%rest_edge0", "%rest_edge1", "%rest_edge2"}, {"f64", "f64", "f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto vec = mlir::VectorType::get({3}, builder.getF64Type());
          return {mlir::vector::FromElementsOp::create(builder, builder.getUnknownLoc(), vec, values)};
        });
  // A program of frames (D251) takes the cell and the
  // positions of each frame in its loop over frames, where this text goes:
  // the boundary gives the buffer of the host that holds the positions of
  // the frame and writes the six numbers of its cell. The buffer is a value
  // of the iteration, so that nothing takes the positions of one frame for
  // those of another; the buffer %positions of the entry is not read.
  std::string frameHead;
  os.flush();
  size_t frameMark = program.module.size();
  if (control.frames) {
    unsigned edges = getCellSize();
    std::vector<std::string> outputs{"%frame_positions"};
    std::vector<std::string> types{("memref<?x3x" + state + ">").str()};
    std::vector<std::string> inputs{"%frame_cell", "%frame_asks"};
    std::vector<std::string> signatures{"memref<6xf64>", "memref<2xf64>"};
    for (unsigned k = 0; k != edges; ++k) {
      outputs.push_back("%frame_cell_" + std::to_string(k));
      types.push_back("f64");
    }
    for (unsigned k = 0; k != 2; ++k) {
      outputs.push_back("%frame_ask_" + std::to_string(k));
      types.push_back("f64");
    }
    for (unsigned k = 0; k != std::max(edges, 2u); ++k) {
      inputs.push_back("%frame_at" + std::to_string(k));
      signatures.push_back("index");
    }
    llvm::SmallVector<StringRef> resultNames, resultTypes, names, operandTypes;
    for (const auto &name : outputs) resultNames.push_back(name);
    for (const auto &type : types) resultTypes.push_back(type);
    for (const auto &name : inputs) names.push_back(name);
    for (const auto &type : signatures) operandTypes.push_back(type);
    emitIRFragment(
        "  ", resultNames, resultTypes, names, operandTypes,
        [edges, state64 = program.state == Element::F64](
            mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Type element = state64 ? mlir::Type(builder.getF64Type()) : mlir::Type(builder.getF32Type());
          auto storage = mlir::MemRefType::get({mlir::ShapedType::kDynamic, 3}, element);
          auto boundary = mlir::func::CallOp::create(
              builder, loc, "mdrtFrameBoundary", mlir::TypeRange{storage}, values.take_front(2));
          TermIRValues results{boundary.getResult(0)};
          for (unsigned k = 0; k != edges; ++k)
            results.push_back(mlir::memref::LoadOp::create(builder, loc, values[0], values[2 + k]));
          for (unsigned k = 0; k != 2; ++k)
            results.push_back(mlir::memref::LoadOp::create(builder, loc, values[1], values[2 + k]));
          return results;
        });
    if (scalesReference())
      emitReferenceScale("  ", "%rest_scale", "%frame_cell_0",
                         "%frame_cell_1", "%frame_cell_2");
    emitCellOf("  ", "%cell", "%frame_cell");
  } else {
    if (scalesReference())
      emitReferenceScale("  ", "%rest_scale", "%lx", "%ly", "%lz");
    llvm::SmallVector<StringRef> edges{"%lx", "%ly", "%lz"};
    if (isTriclinic())
      edges.append({"%tilt_bx", "%tilt_cx", "%tilt_cy"});
    llvm::SmallVector<StringRef> edgeTypes(edges.size(), "f64");
    emitIRFragment(
        "  ", {"%cell"}, {"!md.cell"}, edges, edgeTypes,
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {buildCell(builder, values)};
        });
  }
  std::string positionName = "%x" + std::string(program.reorders ? "_in" : "0") +
                             (hasSites() ? "u" : "");
  emitFieldBuffer(positionName, control.frames ? "%frame_positions" : "%positions",
                  ("memref<?x3x" + state + ">").str(), "!vec", 3, false);
  if (control.frames) {
    os.flush();
    frameHead = program.module.substr(frameMark);
    program.module.resize(frameMark);
  }
  emitFieldBuffer(program.reorders ? "%v_in" : velocities, "%velocities",
                  ("memref<?x3x" + state + ">").str(), "!vec", 3, false);
  emitFieldBuffer("%m" + given, "%masses", ("memref<?x" + mass + ">").str(),
                  "!real", 1, false);
  for (const Program::Field &field : program.fields) {
    StringRef element = field.isInteger ? StringRef("i32") : parameter;
    emitFieldBuffer("%p" + given + "_" + field.name, "%b_" + field.name,
                    ("memref<?x" + element + ">").str(), getFieldType(field),
                    1, field.isInteger);
  }
  for (const Program::Table &table : program.tables) {
    emitIRFragment(
        "  ", {"%t_" + table.name}, {table.getType()}, {"%bt_" + table.name},
        {"memref<?x?xf64>"},
        [symmetric = table.isSquare()](mlir::OpBuilder &builder,
                                         mlir::ValueRange values) -> TermIRValues {
          auto type = mdir::md::TableType::get(
              builder.getContext(), 2, builder.getF64Type(), symmetric);
          return {mdir::mdrt::FromBufferOp::create(builder, builder.getUnknownLoc(), type, values[0])};
        });
  }
  for (const Program::TupleSet &set : program.tupleSets) {
    std::string result = "%r" + std::string(program.reorders ? "o" : "") + "_" + set.name;
    std::string relation = "!rel_" + set.name;
    auto orientation = set.arity == 2 && !set.oriented
                           ? mdir::md::Orientation::Unordered
                           : set.reversible ? mdir::md::Orientation::Reversal
                                            : mdir::md::Orientation::Ordered;
    emitIRFragment(
        "  ", {result}, {relation}, {"%br_" + set.name},
        {"memref<?x" + std::to_string(set.arity) + "xi32>"},
        [name = set.name, arity = set.arity, orientation](
            mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto ctx = builder.getContext();
          auto type = mdir::md::RelationType::get(
              ctx, mlir::FlatSymbolRefAttr::get(ctx, "atoms"), arity, orientation,
              mlir::FlatSymbolRefAttr::get(ctx, name));
          return {mdir::mdrt::FromBufferOp::create(builder, builder.getUnknownLoc(), type, values[0])};
        });
    for (const Program::Field &field : set.fields)
      emitFieldBuffer("%f_" + set.name + "_" + field.name,
                      "%bf_" + set.name + "_" + field.name, "memref<?xf64>",
                      "!of_" + set.name, 1, false, set.name);
  }
  emitFieldBuffer("%id" + given, "%identities", "memref<?xi32>", "!ids", 1, true);
  if (givenForces)
    emitFieldBuffer(program.reorders ? "%f_in" : givenForcesName, "%forces",
                    ("memref<?x3x" + force + ">").str(), "!vec", 3, false);
  os.flush();
  frameMark = program.module.size();
  // The virtual sites where their atoms put them, whatever the file says.
  if (hasSites()) {
    StringRef positions = program.reorders ? "%x_in" : "%x0";
    emitPlaceSites("  ", (positions + "u").str(), positions,
                   program.reorders ? "%ro_" : "%r_");
  }
  if (program.reorders)
    emitReorder("  ", "_in", "0", "", givenForces && !control.frames,
                velocities, branchesStart() ? "%fg" : "");
  if (control.frames) {
    os.flush();
    frameHead += program.module.substr(frameMark);
    program.module.resize(frameMark);
  }
  if (control.minimize) {
    emitMinimization(velocities);
    return llvm::Error::success();
  }

  if (control.getCouplingPeriod() > 0) {
    // What the coupling of the velocities takes: the total mass; and for
    // the thermostat the key of the random numbers, the degrees of freedom,
    // the mean kinetic energy at the temperature of the bath, and how much
    // of the kinetic energy is kept over one period.
    emitIRFragment(
        "  ", {"%couple_half", "%total_mass"}, {"f64", "vector<3xf64>"},
        {"%m" + given}, {"!real"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto result = buildCouplingMass(builder, values[0]);
          return {result.half, result.totalMass};
        });
    if (control.thermostat) {
      // The degrees of freedom of the log. The momentum stays 0 with or
      // without its removal, from velocities that begin with none.
      double freedom = system.getDegreesOfFreedom();
      double target =
          0.5 * freedom * units::boltzmann * control.temperature;
      double decay = std::exp(
          -static_cast<double>(control.getCouplingPeriod()) *
          control.timestep / control.tauT);
      if (control.barostat) {
        // The barostat, in bar: the target pressure, the compressibility,
        // Δt_p / τ_p, and k_B T of the bath. The virial and the energy of
        // the corrections times the volume are arguments of the entry
        // (collectStartValues).
        double bar = 1.01325;
        BarostatParameters parameters{
            {control.compressibilities[0] / bar, control.compressibilities[1] / bar,
             control.compressibilities[2] / bar},
            control.pressure * bar, control.compressibility / bar,
            static_cast<double>(control.barostatPeriod) * control.timestep / control.tauP,
            units::boltzmann * control.temperature, control.compressibilityZ / bar,
            control.surfaceTension * control.surfaces * units::dynePerCmToBarNm};
        emitIRFragment(
            "  ", {"%baro_beta_0", "%baro_beta_1", "%baro_beta_2", "%baro_target",
                   "%baro_beta", "%baro_rate", "%baro_kt", "%c_bar", "%c_three",
                   "%c_third", "%c_unit", "%c_unit3", "%c_half3", "%baro_beta_z", "%baro_tension"},
            {"f64", "f64", "f64", "f64", "f64", "f64", "f64", "f64", "f64", "f64",
             "f64", "vector<3xf64>", "vector<3xf64>", "f64", "f64"}, {}, {},
            [parameters](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
              auto p = buildBarostatParameters(builder, parameters);
              return {p.axisCompressibility[0], p.axisCompressibility[1], p.axisCompressibility[2],
                      p.pressure, p.compressibility, p.rate, p.thermal, p.bar, p.three, p.third,
                      p.unit, p.unitVector, p.halfVector, p.zCompressibility, p.tension};
            });
      }
      emitIRFragment(
          "  ", {"%seed", "%freedom", "%target_kinetic", "%decay"},
          {"i64", "f64", "f64", "f64"}, {}, {},
          [seed = static_cast<int64_t>(control.seed), freedom, target, decay](
              mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value key = mlir::arith::ConstantIntOp::create(builder, loc, seed, 64);
            TermIRValues result{key};
            for (double value : {freedom, target, decay})
              result.push_back(mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value)));
            return result;
          });
    }
  }

  if (isLeapfrog() && control.barostat)
    emitIRFragment(
        "  ", {"%half_back"}, {"f64"}, {"%dt"}, {"f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value factor = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(-0.5));
          return {mlir::arith::MulFOp::create(builder, loc, factor, values[0])};
        });

  // The energy, the forces, and the virial of the state at the start, as
  // %u0, %f0, and %w0; returns the name of the virial.
  auto emitStartForces = [&]() -> std::string {
    StringRef raw = hasSites() ? "e" : "";
    std::string held = hasForceCorrections() ? "p" : "";
    // With a learned potential, `%f0c` and `%u0c` before its terms
    // (D270).
    bool hostTerms = hasHostEvaluation();
    std::string modelled = !hostTerms || !held.empty() ? "" : "c";
    emitEnergyEvaluation("  ", "%x0", "%cell", "%p_", "%time0",
                             "%u0" + held + modelled,
                             "%f0" + held + raw.str() + modelled,
                             "%w0" + held + raw.str());
    std::string virial = "%w0" + held;
    if (hasSites())
      virial = emitSpreadSites("  ", "%x0", "%f0" + held + "e", "%f0" + held,
                               "%r_", "%w0" + held + "e", "%w0" + held);
    if (hasForceCorrections()) {
      emitForceCorrections("  ", "%x0", "%p_", "%f0p", hostTerms ? "%f0c" : "%f0",
                     "%u0p", hostTerms ? "%u0c" : "%u0", virial, "%w0");
      virial = "%w0";
    }
    if (hostTerms)
      virial = emitHostEvaluation("  ", "%x0", "%cell", "%p_", "%f0c", "%f0",
                              "%u0c", "%u0", virial, "0");
    return virial;
  };

  // A run that begins from the checkpoint of a run of other physics
  // evaluates the forces of its first step rather than take those of the
  // checkpoint (D172).
  if (recomputes())
    emitStartForces();

  // A program of segments does the work of the start in a branch on
  // %is_first; a later call takes the forces and the velocities given
  // (D211).
  os.flush();
  size_t startMark = program.module.size();
  if (!isRestart()) {
    // The energies at the start.
    std::string virial = emitStartForces();
    if (control.frames) {
      // The forces and the virial of the frame, where the host asks for
      // them (D252): the gradients of the energy in the
      // positions and in the strain with the opposite sign. The forces are
      // those of the potential, the virtual sites spread and the restraints
      // added, without the constraints; the virial is that of these forces.
      //
      // They come from an evaluation of their own, in a loop of one
      // iteration or none (the form of the frame of a reporter,
      // emitFrameIfDue), and not from the evaluation above. A frame reads
      // of that one the energy, the trace and the diagonal of the virial,
      // and the square of the forces; a call here that took its forces or
      // the nine components of its virial changed what its kernels compute
      // and how its loops are joined, in every frame, asked or not: measured
      // on the CPU in double precision, 46.6 to 47.2 ms a frame of the
      // dipeptide against 45.1 ms. With the evaluation in the branch the
      // text of a frame without the request is that of before but for the
      // branch, and so are its numbers, to the bit, whatever is asked.
      // One evaluation serves both: where the virial is asked the forces
      // are handed over from it. The forces alone have a loop of their
      // own: one loop for either, with the two calls in loops inside it,
      // was measured too, and a frame that asked for nothing took 45.35 to
      // 45.65 ms in it where `main` takes 44.84 to 45.30 ms and this text
      // 44.75 to 45.04 ms.
      StringRef raw = hasSites() ? "e" : "";
      std::string held = hasForceCorrections() ? "p" : "";
      auto emitAsked = [&](StringRef tag, bool withVirial) {
        std::string f = ("%f" + tag + "0").str(), w = ("%w" + tag + "0").str();
        emitEnergyEvaluation("    ", "%x0", "%cell", "%p_", "%time0", "",
                                 f + held + raw.str(),
                                 withVirial ? w + held + raw.str() : std::string());
        std::string tensor = withVirial ? w + held + raw.str() : std::string();
        if (hasSites())
          tensor = emitSpreadSites(
              "    ", "%x0", f + held + "e", f + held, "%r_", tensor,
              withVirial ? w + held : std::string());
        if (hasForceCorrections()) {
          emitForceCorrections("    ", "%x0", "%p_", f + "p", f, "", "", tensor,
                         withVirial ? w : std::string());
          if (withVirial)
            tensor = w;
        }
        return std::make_pair(f, tensor);
      };
      auto emitForces = [&](StringRef indent, StringRef force) {
        emitIRFragment(
            indent, {}, {}, {force, idName}, {"!vec", "!ids"},
            [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
              mdir::mdrt::HostCallOp::create(
                  builder, builder.getUnknownLoc(), mlir::TypeRange{},
                  mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteFrameForces"),
                  values, mlir::ValueRange{}, mlir::UnitAttr{});
              return {};
            });
      };
      emitIRFragment(
          "  ", {"%frame_nf", "%frame_nw", "%frame_no"}, {"index", "index", "index"},
          {"%frame_ask_0", "%frame_ask_1", "%c1", "%c0"}, {"f64", "f64", "index", "index"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value zero = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
            mlir::Value force = mlir::arith::CmpFOp::create(builder, loc, mlir::arith::CmpFPredicate::ONE, values[0], zero);
            mlir::Value virial = mlir::arith::CmpFOp::create(builder, loc, mlir::arith::CmpFPredicate::ONE, values[1], zero);
            mlir::Value nf = mlir::arith::SelectOp::create(builder, loc, force, values[2], values[3]);
            mlir::Value nw = mlir::arith::SelectOp::create(builder, loc, virial, values[2], values[3]);
            mlir::Value no = mlir::arith::SelectOp::create(builder, loc, virial, values[3], nf);
            return {nf, nw, no};
          });
      os << "  scf.for %frame_if = %c0 to %frame_no step %c1 {\n";
      emitForces("    ", emitAsked("g", false).first);
      os << "  }\n"
         << "  scf.for %frame_iw = %c0 to %frame_nw step %c1 {\n";
      auto [forces, tensor] = emitAsked("w", true);
      os << "    scf.for %frame_ib = %c0 to %frame_nf step %c1 {\n";
      emitForces("      ", forces);
      os << "    }\n";
      emitIRFragment(
          "    ", {}, {}, {tensor}, {"vector<9xf64>"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            llvm::SmallVector<mlir::Value> entries;
            for (int k = 0; k != 9; ++k)
              entries.push_back(mlir::vector::ExtractOp::create(builder, loc, values[0], k));
            mlir::func::CallOp::create(builder, loc, "mdrtWriteFrameVirial", mlir::TypeRange{}, entries);
            return {};
          });
      os << "  }\n";
    }
    emitTerms();
    // The diagonal of the virial, which semi-isotropic coupling takes axis
    // by axis, for comparison with other programs (D119).
    if (system.topology) {
      auto diagonal = emitDiagonal("%dgw0", virial, "  ");
      emitIRFragment(
          "  ", {}, {}, {diagonal.xx, diagonal.yy, diagonal.zz},
          {"f64", "f64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            mlir::func::CallOp::create(builder, builder.getUnknownLoc(),
                                        "mdrtWriteVirial", mlir::TypeRange{}, values);
            return {};
          });
    }
    emitKineticEnergy("%k0", velocities, "%m", "  ");
    // K_half - K from the forces without constraints; with them there is
    // no step before the start to measure, and the row takes K
    // (D203).
    std::string startExcess =
        emitKineticExcess("  ", "s0", "", "%f0", velocities);
    if (startExcess.empty())
      emitIRFragment(
          "  ", {"%g0"}, {"f64"}, {}, {},
          [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
            return {mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), builder.getF64FloatAttr(0.0))};
          });
    else
      emitSum3("%g0", startExcess, "  ");
    std::string trace = "%tr0", pressureKinetic = "%kph0";
    if (control.frames || control.isBrownian()) {
      // The frame evaluator reports the atomic virial of the forces.
      emitTrace("%tr0", virial, "  ");
      if (hasConstraints())
        trace = emitStartConstraintTrace("%x0", "%f0", velocities, trace);
      emitIRFragment(
          "  ", {"%kph0"}, {"f64"}, {"%k0", "%g0"}, {"f64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {mlir::arith::AddFOp::create(builder, builder.getUnknownLoc(), values[0], values[1])};
          });
    } else {
      // The pressure of the groups, whose kinetic energy at the half steps
      // the forces give with constraints as well
      // (D264).
      emitDiagonal("%dgp0", virial, "  ");
      std::string groups = emitMolecularDiagonal(
          "  ", "%dgp0", "%x0", "%f0", "%cell", "%m", "%r_", "p0");
      emitSum3("%tr0", groups, "  ");
      GroupKinetic centers =
          emitGroupKinetic("  ", "%x0", velocities, "%f0", "s0");
      emitSum3("%kph0", centers.half, "  ");
    }
    emitIRFragment(
        "  ", {}, {}, {"%start", "%u0", "%k0", "%g0", trace, pressureKinetic},
        {"i64", "f64", "f64", "f64", "f64", "f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          mlir::func::CallOp::create(builder, builder.getUnknownLoc(), "mdrtWriteEnergies", mlir::TypeRange{}, values);
          return {};
        });
    for (const TermBuilder *term : getTermBuilders())
      if (const auto *role = term->getRoles().coordinateOutput)
        role->emitOutput(*this, {"  ", "%x0", "%cell", "%p_", "%start",
                                 "%time0", "", false});
    emitFreeEnergyOutput("  ", "%x0", "%cell", "%p_", "%start", "%time0");
    emitObservablesOutput("  ", "%x0", "%cell", "%p_", "%start", "%time0");
    emitTunableGradientOutput("  ", "%x0", "%cell", "%p_", "%time0");
    // The state that the first scaling takes its pressure from (D92), by
    // axes.
    if (scalesEveryStep()) {
      emitDiagonal("%dg0", virial, "  ");
      std::string diagonal = "%dg0";
      if (hasConstraints())
        diagonal = emitStartConstraintTrace("%x0", "%f0", velocities,
                                            diagonal, /*axes=*/true);
      emitStoreTrotterState(
          "  ", diagonal,
          emitMolecularDiagonal("  ", "%dg0", "%x0", "%f0", "%cell", "%m",
                                "%r_", "s0"),
          emitGroupKinetic("  ", "%x0", velocities, "%f0", "t0").half);
    }

    if (isLeapfrog() && branchesStart()) {
      // v(-dt/2) = v(0) - (dt/2) F(0) / m on the first call; a call that
      // evaluates the forces anew after an update of the tunables
      // (%first_call = 2, D213) or at the start of a stage of other
      // physics from a checkpoint (D223) has the
      // velocities of the half step already.
      emitIRFragment(
          "  ", {"%v0"}, {"!vec"}, {"%dt", "%first_call", "%vg", "%f0", "%m"},
          {"f64", "i64", "!vec", "!vec", "!real"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value back = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(-0.5));
            mlir::Value interval = mlir::arith::MulFOp::create(builder, loc, back, values[0]);
            mlir::Value none = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
            mlir::Value first = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
            mlir::Value begins = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::eq, values[1], first);
            mlir::Value behind = mlir::arith::SelectOp::create(builder, loc, begins, interval, none);
            return {buildKick(builder, values[2], values[3], values[4], behind)};
          });
    } else if (isLeapfrog()) {
      emitIRFragment(
          "  ", {"%v0"}, {"!vec"}, {"%dt", "%vg", "%f0", "%m"},
          {"f64", "!vec", "!vec", "!real"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value back = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(-0.5));
            mlir::Value behind = mlir::arith::MulFOp::create(builder, loc, back, values[0]);
            return {buildKick(builder, values[1], values[2], values[3], behind)};
          });
    }
  }
  if (control.frames) {
    // The loop over frames, which has no end of its own, as the loop over
    // parts has none: each iteration waits at its boundary for the cell and
    // the positions of a frame, places the virtual sites, and evaluates
    // what the start of a run evaluates, in a segment of its own, so that
    // the neighbor structures are built anew (D218).
    // This marker follows the evaluation recipes inside the frame iteration.
    // Its guard sees their native operations, not their assembly connections.
    emitIRFragment(
        "  ", {}, {}, {}, {},
        [](mlir::OpBuilder &builder, mlir::ValueRange)
            -> llvm::Expected<TermIRValues> {
          if (llvm::Error error = checkFrameEvaluation(
                  *builder.getInsertionBlock()->getParentOp()))
            return std::move(error);
          return TermIRValues{};
        });
    os.flush();
    std::string chunk = frameHead + "  scf.for %i_frame_once = %c0 to %c1 "
                                    "step %c1 {\n" +
                        program.module.substr(startMark) +
                        "  } {mdrt.segment}\n";
    program.module.resize(startMark);
    // What this assembly assumes of the text, which would otherwise break
    // without a sign (#302).
    frameLoopError = checkFrameLoop(chunk);
    std::vector<unsigned> indices;
    for (int k = getCellSize(); k < 2; ++k) indices.push_back(k);
    for (int k = 0; k != getCellSize(); ++k) indices.push_back(k);
    std::vector<std::string> outputs{"%frame_cell", "%frame_asks"};
    for (unsigned index : indices) outputs.push_back("%frame_at" + std::to_string(index));
    outputs.push_back("%frame_end");
    llvm::SmallVector<StringRef> names, types{"memref<6xf64>", "memref<2xf64>"};
    for (const auto &name : outputs) names.push_back(name);
    types.append(indices.size() + 1, "index");
    emitIRFragment(
        "  ", names, types, {}, {},
        [indices](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value cell = mlir::memref::AllocaOp::create(builder, loc, mlir::MemRefType::get({6}, builder.getF64Type()));
          mlir::Value asks = mlir::memref::AllocaOp::create(builder, loc, mlir::MemRefType::get({2}, builder.getF64Type()));
          TermIRValues results{cell, asks};
          for (unsigned index : indices)
            results.push_back(mlir::arith::ConstantIndexOp::create(builder, loc, index));
          results.push_back(mlir::arith::ConstantIndexOp::create(builder, loc, INT64_MAX));
          return results;
        });
    os << "  scf.for %i_frame = %c0 to %frame_end step %c1 {\n"
       << chunk << "  }\n  return\n}\n";
    return llvm::Error::success();
  }
  if (branchesStart()) {
    os.flush();
    std::string chunk = program.module.substr(startMark);
    program.module.resize(startMark);
    // The evaluation at the start is a segment of its own, as it is in
    // `mdir run`, whose first segment of checkpoints (mdrt.segment) builds
    // its neighbor structures anew at the first step after it: a structure
    // of the start does not carry into the steps, so that both build at the
    // same steps and sum the forces of a configuration in the same order
    // (#121, D218).
    emitStartBranch(std::move(chunk), {"%f0", "%v0"},
                    {"%f0", isLeapfrog() ? "%v0" : "%vg"}, {"%fg", "%vg"},
                    /*segment=*/true);
  }

  if (isRestart() && scalesEveryStep() &&
      system.barostatState.size() == 9) {
    // The state that the first scaling takes its pressure from, as the
    // checkpoint keeps it (D92).
    // Its numbers are arguments of the entry (collectStartValues).
    std::string vectors[3]{"%bstatev0", "%bstatev1", "%bstatev2"};
    std::vector<std::string> inputs;
    for (unsigned k = 0; k != 9; ++k) inputs.push_back("%bstate" + std::to_string(k));
    llvm::SmallVector<StringRef> names, types(9, "f64");
    for (const auto &name : inputs) names.push_back(name);
    emitIRFragment(
        "  ", {vectors[0], vectors[1], vectors[2]},
        {"vector<3xf64>", "vector<3xf64>", "vector<3xf64>"}, names, types,
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          auto vec = mlir::VectorType::get({3}, builder.getF64Type());
          TermIRValues results;
          for (unsigned axis = 0; axis != 3; ++axis)
            results.push_back(mlir::vector::FromElementsOp::create(
                builder, builder.getUnknownLoc(), vec, values.slice(3 * axis, 3)));
          return results;
        });
    emitStoreTrotterState("  ", vectors[0], vectors[1], vectors[2]);
  } else if (isRestart() && scalesEveryStep()) {
    // A checkpoint of a run that did not scale every step holds no virial:
    // the state that the first scaling takes its pressure from is evaluated
    // once (D92). With leapfrog the stored velocities are half a kick
    // behind those of the time of the positions.
    StringRef raw = hasSites() ? "e" : "";
    std::string held = hasForceCorrections() ? "p" : "";
    // A learned potential adds its terms to `%ubsc`, `%fbsc`, and the virial
    // (D270).
    bool hostTerms = hasHostEvaluation();
    std::string modelled = !hostTerms || !held.empty() ? "" : "c";
    emitEnergyEvaluation("  ", "%x0", "%cell", "%p_", "%time0",
                             "%ubs" + held + modelled,
                             "%fbs" + held + raw.str() + modelled,
                             "%wbs" + held + raw.str());
    std::string virial = "%wbs" + held;
    if (hasSites())
      virial = emitSpreadSites("  ", "%x0", "%fbs" + held + "e", "%fbs" + held,
                               "%r_", "%wbs" + held + "e", "%wbs" + held);
    if (hasForceCorrections()) {
      emitForceCorrections("  ", "%x0", "%p_", "%fbsp", hostTerms ? "%fbsc" : "%fbs",
                     "%ubsp", hostTerms ? "%ubsc" : "%ubs", virial, "%wbs");
      virial = "%wbs";
    }
    if (hostTerms)
      virial = emitHostEvaluation("  ", "%x0", "%cell", "%p_", "%fbsc", "%fbs",
                              "%ubsc", "%ubs", virial, "bs");
    std::string current = velocities;
    if (isLeapfrog()) {
      emitIRFragment(
          "  ", {"%vs_now"}, {"!vec"}, {"%dt", velocities, "%fbs", "%m"},
          {"f64", "!vec", "!vec", "!real"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            auto loc = builder.getUnknownLoc();
            mlir::Value half = mlir::arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.5));
            mlir::Value interval = mlir::arith::MulFOp::create(builder, loc, half, values[0]);
            return {buildKick(builder, values[1], values[2], values[3], interval)};
          });
      current = "%vs_now";
    }
    emitDiagonal("%dgs", virial, "  ");
    std::string diagonal = "%dgs";
    if (hasConstraints())
      diagonal = emitStartConstraintTrace("%x0", "%fbs", current, diagonal,
                                          /*axes=*/true);
    emitStoreTrotterState(
        "  ", diagonal,
        emitMolecularDiagonal("  ", "%dgs", "%x0", "%fbs", "%cell", "%m",
                              "%r_", "s0"),
        emitKineticWithoutCenter("  ", current, "%m", "s0"));
  }

  std::string last = "e0";
  if (control.segments) {
    os.flush();
    size_t partMark = program.module.size();
    nestEnd = levels[0].name == "couple" ? 2 : 1;
    emitLevel(0, "  ");
    // The plain steps of the period in which the segment ends, after the
    // loops (D196); the next segment closes the period. The
    // barostat may have changed the cell in the loops.
    std::string outerCell = cellName, outerScale = scaleName;
    if (changesCell()) {
      cellName = "%cell_tail";
      emitBoxEdges("  ", "%edge_tail");
      emitCellOf("  ", cellName, "%edge_tail");
      if (scalesReference()) {
        scaleName = "%rest_scale_tail";
        emitReferenceScale("  ", scaleName, "%edge_tail_0",
                           "%edge_tail_1", "%edge_tail_2");
      }
    }
    os << "  %xt, %vt, %ft = scf.for %i_tail = %c0 to %n_tail step %c1\n"
       << "      iter_args(%xta = %xe0, %vta = %ve0, %fta = %fe0)\n"
       << "      -> (!vec, !vec, !vec) {\n";
    emitDynamicsStep("    ", "step", "%xta", "%vta", "%fta", cellName, "",
                         {"%xtb", "%vtb", "%ftb"}, {"!vec", "!vec", "!vec"});
    os << "    scf.yield %xtb, %vtb, %ftb : !vec, !vec, !vec\n"
       << "  }\n";
    last = emitSegmentEnergyStep("%xt", "%vt", "%ft");
    cellName = outerCell;
    scaleName = outerScale;
    // Intervals that end with a step of energy, as an interval of the log
    // of `mdir run` ends, after the steps of the segment before them: one
    // period that closes, or the intervals of the reports
    // (D207).
    unsigned second = levels[0].name == "couple" ? 2 : 1;
    emitIRFragment(
        "  ", {"%start_e"}, {"i64"}, {"%n0", "%per0", "%n_tail", "%n_plain", "%start"},
        {"index", "index", "index", "index", "i64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {buildSegmentStep(builder, values[0], values[1], values[2], values[3], values[4])};
        });
    nestBegin = second;
    nestEnd = levels.size();
    nestOutside = last;
    nestStart = "%start_e";
    emitLevel(second, "  ");
    last = "e" + std::to_string(second);
    os.flush();
    std::string chunk = program.module.substr(partMark);
    program.module.resize(partMark);
    emitPartLoop(std::move(chunk), {"%x0", "%v0", "%f0"},
                 {"!vec", "!vec", "!vec"},
                 {"%x" + last, "%v" + last, "%f" + last},
                 {"%x0", "%v0", "%f0"});
    os << "  return\n}\n";
    return llvm::Error::success();
  }
  emitLevel(0, "  ");
  emitIRFragment(
      "  ", {}, {}, {"%x" + last, "%v" + last, idName}, {"!vec", "!vec", "!ids"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
            mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtFinish"),
            values, mlir::ValueRange{}, mlir::UnitAttr{});
        return {};
      });
  os << "  return\n}\n";
  return llvm::Error::success();
}

void Builder::emitPartCounts() {
  // The step that the part begins after, and the counts of its loops: the
  // iterations of the outer loop, the plain steps in an iteration of the
  // loop over the periods of coupling, and plain steps after the loops;
  // then whether the part ends with a plain step of energy, and whether it
  // ends with a period that closes with a step of energy, with its plain
  // steps (D196). The second nest is %count_energy_close intervals that end
  // with a step of energy, of %count_energy_periods + 1 periods of coupling
  // each (or of %count_energy_inner + 1 steps without coupling); a frame is
  // written at a step of energy whose number is a multiple of
  // %frame_period (0: none) (D207). A minimization takes its steps in
  // %count_outer (D202).
  static const char *const countNames[] = {
      "%start$part", "%count_outer", "%count_inner", "%count_tail",
      "%count_energy_plain", "%count_energy_close", "%count_energy_inner",
      "%count_energy_periods", "%frame_period"};
  unsigned count = control.minimize ? 2 : getPartCountsSize();
  bool couples = !control.minimize && levels[0].name == "couple";
  llvm::SmallVector<StringRef> names(countNames, countNames + count);
  llvm::SmallVector<StringRef> types(count, "i64");
  if (!control.minimize) {
    if (couples)
      names.append({"%n0", "%n1", "%n2", "%n3", "%n4", "%n_tail", "%n_plain",
                    "%per1", "%per0", "%per4", "%per3", "%per2"});
    else
      names.append({"%n0", "%n1", "%n2", "%n_tail", "%n_plain", "%per0", "%per2", "%per1"});
    types.append(names.size() - count, "index");
  }
  std::string memoryType = "memref<" + std::to_string(getPartCountsSize()) + "xi64>";
  llvm::SmallVector<StringRef> inputs{"%part_counts"}, inputTypes{memoryType};
  if (!control.minimize) { inputs.push_back("%c1"); inputTypes.push_back("index"); }
  emitIRFragment(
      "    ", names, types, inputs, inputTypes,
      [count, couples, minimize = bool(control.minimize), closing = getClosingSteps()](
          mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        TermIRValues counts;
        for (unsigned k = 0; k != count; ++k) {
          mlir::Value index = mlir::arith::ConstantIndexOp::create(builder, loc, k);
          counts.push_back(mlir::memref::LoadOp::create(builder, loc, values[0], index));
        }
        if (minimize) return counts;
        auto cast = [&](mlir::Value value) -> mlir::Value {
          return mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(), value);
        };
        auto one = [&]() -> mlir::Value {
          return mlir::arith::ConstantIndexOp::create(builder, loc, 1);
        };
        auto add = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
          return mlir::arith::AddIOp::create(builder, loc, a, b);
        };
        mlir::Value n0 = cast(counts[1]);
        counts.push_back(n0);
        mlir::Value n1, n2, n3, n4;
        if (couples) {
          n1 = cast(counts[2]); n2 = cast(counts[5]); n3 = cast(counts[7]); n4 = cast(counts[6]);
          counts.append({n1, n2, n3, n4});
        } else {
          n1 = cast(counts[5]); n2 = cast(counts[6]);
          counts.append({n1, n2});
        }
        mlir::Value tail = cast(counts[3]), plain = cast(counts[4]);
        counts.append({tail, plain});
        if (couples) {
          mlir::Value per1 = one();
          mlir::Value last = mlir::arith::ConstantIndexOp::create(builder, loc, closing);
          mlir::Value per0 = add(n1, last);
          mlir::Value per4 = one();
          mlir::Value per3 = add(n4, last);
          mlir::Value periods = add(n3, values[1]);
          mlir::Value per2 = mlir::arith::MulIOp::create(builder, loc, periods, per3);
          counts.append({per1, per0, per4, per3, per2});
        } else {
          mlir::Value per0 = one(), per2 = one();
          mlir::Value per1 = add(n2, values[1]);
          counts.append({per0, per2, per1});
        }
        return counts;
      });
}

void Builder::emitPartLoop(std::string chunk,
                           llvm::ArrayRef<std::string> carried,
                           llvm::ArrayRef<std::string> types,
                           llvm::ArrayRef<std::string> yielded,
                           llvm::ArrayRef<std::string> boundary) {
  // The chunk is written as it is: its uses of the values carried become
  // the loop's arguments, and its uses of %start the step that the part
  // begins after, when the module is parsed (`mdir.carries`,
  // `mdir.part_start`, ProgramModule.cpp, #568). The builder renamed them
  // in the text before.
  std::string typeList;
  for (size_t i = 0; i != types.size(); ++i)
    typeList += (i ? ", " : "") + types[i];
  std::string counts =
      "memref<" + std::to_string(getPartCountsSize()) + "xi64>";
  // The loop has no end of its own: the host ends the activation at a
  // boundary (Simulation.cpp).
  emitIRFragment(
      "  ", {"%part_counts", "%part_end"}, {counts, "index"}, {}, {},
      [count = getPartCountsSize()](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        mlir::Value memory = mlir::memref::AllocaOp::create(
            builder, loc, mlir::MemRefType::get({int64_t(count)}, builder.getI64Type()));
        mlir::Value end = mlir::arith::ConstantIndexOp::create(builder, loc, INT64_MAX);
        return {memory, end};
      });
  os << "  ";
  for (size_t i = 0; i != carried.size(); ++i)
    os << (i ? ", " : "") << carried[i] << "$parts";
  os << " = scf.for %i_part = %c0 to %part_end step %c1\n"
     << "      iter_args(";
  for (size_t i = 0; i != carried.size(); ++i)
    os << (i ? ", " : "") << carried[i] << "$part = " << carried[i];
  os << ") -> (" << typeList << ") {\n";
  // The components of the state that are not numbers, which fail the part
  // (D196): the positions, velocities, and forces, in whatever type they
  // are stored. x - x is not a number for an infinity as well.
  std::string x = boundary[0] + "$part", v = boundary[1],
              f = boundary[2] + "$part";
  if (llvm::is_contained(carried, boundary[1]))
    v += "$part";
  emitIRFragment(
      "    ", {}, {}, {x, v, f, idName, "%part_counts"},
      {"!vec", "!vec", "!vec", "!ids", counts},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        auto vec = mlir::VectorType::get({3}, builder.getF64Type());
        auto sum = mdir::md::SumParticlesOp::create(builder, loc, vec, values.take_front(3));
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          auto args = builder.createBlock(&sum.getKernel(), {}, {vec, vec, vec}, {loc, loc, loc})->getArguments();
          mlir::Value one = mlir::arith::ConstantOp::create(builder, loc, mlir::DenseElementsAttr::get(vec, 1.0));
          mlir::Value zero = mlir::arith::ConstantOp::create(builder, loc, mlir::DenseElementsAttr::get(vec, 0.0));
          llvm::SmallVector<mlir::Value> bad;
          for (mlir::Value field : args) {
            mlir::Value difference = mlir::arith::SubFOp::create(builder, loc, field, field);
            mlir::Value finite = mlir::arith::CmpFOp::create(builder, loc, mlir::arith::CmpFPredicate::ORD, difference, difference);
            bad.push_back(mlir::arith::SelectOp::create(builder, loc, finite, zero, one));
          }
          mlir::Value first = mlir::arith::AddFOp::create(builder, loc, bad[0], bad[1]);
          mlir::Value all = mlir::arith::AddFOp::create(builder, loc, first, bad[2]);
          mdir::md::YieldOp::create(builder, loc, mlir::ValueRange{all});
        }
        mlir::Value invalid = buildSum3(builder, sum.getResult());
        mdir::mdrt::HostCallOp::create(
            builder, loc, mlir::TypeRange{},
            mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtPartBoundary"),
            mlir::ValueRange{values[0], values[1], values[2], values[3], invalid, values[4]},
            mlir::ValueRange{}, builder.getUnitAttr());
        return {};
      });
  emitPartCounts();
  // %start of the chunk is the step of the part (carryLoopValues).
  os << "    %start$mark = builtin.unrealized_conversion_cast %start$part, "
        "%start : i64, i64 to i64 {mdir.part_start}\n";
  os << chunk << "    scf.yield ";
  llvm::interleaveComma(yielded, os);
  os << " : " << typeList << "\n  } {mdir.carries}\n";
}

void Builder::emitFrameIfDue(StringRef indent, StringRef step,
                             StringRef positions, StringRef tag) {
  // A frame of a reporter (D207) where the number of the step
  // is a multiple of the frame period that the entry takes; only then are
  // the positions copied.
  emitIRFragment(
      indent, {}, {}, {"%frame_period", step, positions, idName, "%c0", "%c1"},
      {"i64", "i64", "!vec", "!ids", "index", "index"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto loc = builder.getUnknownLoc();
        mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
        mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
        mlir::Value hasPeriod = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::sgt, values[0], zero);
        mlir::Value safe = mlir::arith::SelectOp::create(builder, loc, hasPeriod, values[0], one);
        mlir::Value remainder = mlir::arith::RemSIOp::create(builder, loc, values[1], safe);
        mlir::Value exact = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::eq, remainder, zero);
        mlir::Value due = mlir::arith::AndIOp::create(builder, loc, hasPeriod, exact);
        mlir::Value wide = mlir::arith::ExtUIOp::create(builder, loc, builder.getI64Type(), due);
        mlir::Value trips = mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(), wide);
        auto loop = mlir::scf::ForOp::create(builder, loc, values[4], trips, values[5]);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(loop.getBody());
          mdir::mdrt::HostCallOp::create(
              builder, loc, mlir::TypeRange{},
              mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtWriteFrame"),
              values.slice(1, 3), mlir::ValueRange{}, mlir::UnitAttr{});
        }
        return {};
      });
}

std::string Builder::emitSegmentEnergyStep(StringRef x, StringRef v,
                                           StringRef f) {
  // At most one plain step, of energy, as `mdir run` takes at a row of its
  // log without coupling: its energies go to mdrtWriteEnergies.
  emitIRFragment(
      "  ", {"%step_plain"}, {"i64"}, {"%n0", "%per0", "%n_tail", "%c1", "%start"},
      {"index", "index", "index", "index", "i64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildSegmentStep(builder, values[0], values[1], values[2], values[3], values[4])};
      });
  os << "  %xp, %vp, %fp = scf.for %i_plain = %c0 to %n_plain step %c1\n"
     << "      iter_args(%xpa = " << x << ", %vpa = " << v << ", %fpa = "
     << f << ")\n"
     << "      -> (!vec, !vec, !vec) {\n";
  std::vector<std::string> results{"%xpl", "%vpl", "%fpl", "%upl", "%wpl"};
  std::vector<std::string> types{"!vec", "!vec", "!vec", "f64", "vector<9xf64>"};
  if (!control.isBrownian()) { results.push_back("%gwpl"); types.push_back("vector<3xf64>"); }
  if (isLeapfrog()) { results.push_back("%vpn"); types.push_back("!vec"); }
  if (measuresHalfSteps()) { results.push_back("%khspl"); types.push_back("vector<3xf64>"); }
  if (reportsSolvent()) { results.push_back("%kwspl"); types.push_back("vector<3xf64>"); }
  llvm::SmallVector<StringRef> names, signatures;
  for (const auto &name : results) names.push_back(name);
  for (const auto &type : types) signatures.push_back(type);
  emitDynamicsStep("    ", "step_energy", "%xpa", "%vpa", "%fpa", cellName, "", names, signatures);
  if (reportsSolvent())
    emitWriteSolvent("    ", "%kwspl");
  std::string plainFull = isLeapfrog() ? "%vpn" : "%vpl";
  emitKineticEnergy("%kpl", plainFull, massName, "    ");
  std::string plainExcess = emitKineticExcess(
      "    ", "pl", measuresHalfSteps() ? "%khspl" : "", "%fpl", plainFull);
  if (plainExcess.empty())
    emitIRFragment(
        "    ", {"%gpl"}, {"f64"}, {}, {},
        [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          return {mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), builder.getF64FloatAttr(0.0))};
        });
  else
    emitSum3("%gpl", plainExcess, "    ");
  // The pressure of the groups (D264).
  if (control.isBrownian()) {
    emitTrace("%trpl", "%wpl", "    ");
    emitIRFragment(
        "    ", {"%kphpl"}, {"f64"}, {"%kpl", "%gpl"}, {"f64", "f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {mlir::arith::AddFOp::create(builder, builder.getUnknownLoc(), values[0], values[1])};
        });
  } else {
    emitSum3("%trpl", "%gwpl", "    ");
    GroupKinetic centers =
        emitGroupKinetic("    ", "%xpl", plainFull, "%fpl", "pl");
    emitSum3("%kphpl", centers.half, "    ");
  }
  emitIRFragment(
      "    ", {}, {}, {"%step_plain", "%upl", "%kpl", "%gpl", "%trpl", "%kphpl"},
      {"i64", "f64", "f64", "f64", "f64", "f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mlir::func::CallOp::create(builder, builder.getUnknownLoc(), "mdrtWriteEnergies", mlir::TypeRange{}, values);
        return {};
      });
  if (!control.observables.empty()) {
    // The observed terms at the positions after the step (D189), as at a
    // row of the second nest.
    if (control.usesTime)
      emitIRFragment(
          "    ", {"%ob_time_pl"}, {"f64"}, {"%step_plain", "%dt"}, {"i64", "f64"},
          [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
            return {buildSimulationTime(builder, values[0], values[1])};
          });
    emitObservablesOutput("    ", "%xpl", cellName, fieldPrefix, "%step_plain",
                          "%ob_time_pl");
  }
  if (!control.periodic)
    emitIRFragment(
        "    ", {}, {}, {"%step_plain", "%xpl", idName}, {"i64", "!vec", "!ids"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          mdir::mdrt::HostCallOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{},
              mlir::FlatSymbolRefAttr::get(builder.getContext(), "mdrtCheckSpread"),
              values, mlir::ValueRange{}, mlir::UnitAttr{});
          return {};
        });
  emitFrameIfDue("    ", "%step_plain", "%xpl", "pl");
  os << "    scf.yield %xpl, %vpl, %fpl : !vec, !vec, !vec\n"
     << "  }\n";
  return "p";
}

void Builder::setSchedule() {
  // A period of zero means no output of that kind, and no loop for it.
  int64_t steps = control.numSteps;
  if (control.checkpointPeriod > 0) {
    levels.push_back({"segment", steps / control.checkpointPeriod});
    steps = control.checkpointPeriod;
  }
  // Frames more frequent than energies are written at the end of each
  // interval between energies, which then has their period.
  int64_t energyPeriod = control.getEnergyLoopPeriod();
  framesAtEnergies = energyPeriod != control.energyPeriod;
  if (control.framePeriod > 0 && !framesAtEnergies) {
    levels.push_back({"frame", steps / control.framePeriod});
    steps = control.framePeriod;
  }
  // The velocities are coupled at the end of the last step of a period of
  // coupling, which is taken after the loop over steps, as the last step
  // of an interval between energies is. The loop over the periods of an
  // interval between energies leaves the last period to the interval,
  // which ends it with its step of energy.
  // With the barostat of Trotter type the last two steps of a period are
  // taken after its loop: the step whose pressure gives the strain, and
  // the step that scales the cell within its drift (D92).
  int64_t coupling = control.getCouplingPeriod();
  int64_t taken = usesTrotter() && !scalesEveryStep() ? 2 : 1;
  if (energyPeriod > 0) {
    levels.push_back({"energy", steps / energyPeriod});
    steps = energyPeriod;
    if (coupling > 0) {
      levels.push_back({"couple", steps / coupling - 1});
      levels.push_back({"step", coupling - taken});
    } else {
      levels.push_back({"step", steps - 1});
    }
  } else if (coupling > 0) {
    levels.push_back({"couple", steps / coupling});
    levels.push_back({"step", coupling - taken});
  } else {
    levels.push_back({"step", steps});
  }
}

void Builder::emitProgramCall(
    StringRef indent, StringRef symbol, llvm::ArrayRef<StringRef> results,
    llvm::ArrayRef<StringRef> resultTypes, llvm::ArrayRef<StringRef> operands,
    llvm::ArrayRef<StringRef> operandTypes) {
  emitIRFragment(
      indent, results, resultTypes, operands, operandTypes,
      [symbol = symbol.str()](mlir::OpBuilder &builder, mlir::ValueRange values)
          -> llvm::Expected<TermIRValues> {
        auto module = builder.getInsertionBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
        auto callee = module.lookupSymbol<mdir::dyn::ProgramOp>(symbol);
        if (!callee)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "a generated step has no program definition");
        auto call = mdir::dyn::StepOp::create(
            builder, builder.getUnknownLoc(), callee.getFunctionType().getResults(), symbol, values);
        TermIRValues result;
        llvm::append_range(result, call.getResults());
        return result;
      });
}
void Builder::emitDynamicsStep(
    StringRef indent, StringRef symbol, StringRef x, StringRef v, StringRef f,
    StringRef cell, StringRef mu, llvm::ArrayRef<StringRef> results,
    llvm::ArrayRef<StringRef> resultTypes) {
  PotentialConnections fields(program, x, cell, fieldPrefix, "", false);
  std::vector<std::string> inputs{x.str(), v.str(), f.str(), massName, cell.str(), "%dt"};
  std::vector<std::string> types{"!vec", "!vec", "!vec", "!real", "!md.cell", "f64"};
  if (!mu.empty()) { inputs.push_back(mu.str()); types.push_back("vector<3xf64>"); }
  if (scalesReference()) { inputs.push_back(scaleName); types.push_back("vector<3xf64>"); }
  for (unsigned slot = 2; slot != fields.values.size(); ++slot) {
    inputs.push_back(fields.values[slot]); types.push_back(fields.types[slot]);
  }
  std::string noise = emitNoiseValue(indent);
  if (!noise.empty()) {
    inputs.push_back(noise); types.push_back("i64");
    inputs.push_back(idName); types.push_back("!ids");
  }
  llvm::SmallVector<StringRef> operands, signatures;
  for (const auto &input : inputs) operands.push_back(input);
  for (const auto &type : types) signatures.push_back(type);
  emitProgramCall(indent, symbol, results, resultTypes, operands, signatures);
}

void Builder::emitBoxEdges(StringRef indent, StringRef prefix) {
  unsigned count = getCellSize();
  std::vector<std::string> outputs, inputs{"%box_memory"};
  std::vector<std::string> inputTypes{getBoxMemoryType()};
  for (unsigned k = 0; k != count; ++k) {
    outputs.push_back((prefix + "_" + std::to_string(k)).str());
    inputs.push_back("%c_edge" + std::to_string(k)); inputTypes.push_back("index");
  }
  llvm::SmallVector<StringRef> names, operands, types, resultTypes(count, "f64");
  for (const auto &name : outputs) names.push_back(name);
  for (const auto &name : inputs) operands.push_back(name);
  for (const auto &type : inputTypes) types.push_back(type);
  emitIRFragment(indent, names, resultTypes, operands, types,
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        TermIRValues result;
        for (mlir::Value index : values.drop_front())
          result.push_back(mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(), values[0], index));
        return result;
      });
}

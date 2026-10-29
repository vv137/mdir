// The schedule of @mdir_run: the levels of its loops, the branches of
// the start, the parts of a program of segments, the frames, and the
// minimization (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"

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
    os << "  %first_trips = arith.select %is_first, %c1, %c0 : index\n";
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

/// The trace of the virial `virial`, which enters the pressure.
static void emitTrace(llvm::raw_ostream &os, StringRef result,
                      StringRef virial, StringRef indent) {
  for (StringRef part : {"0", "4", "8"})
    os << indent << result << "_" << part << " = vector.extract " << virial
       << "[" << part << "] : f64 from vector<9xf64>\n";
  os << indent << result << "_04 = arith.addf " << result << "_0, " << result
     << "_4 : f64\n"
     << indent << result << " = arith.addf " << result << "_04, " << result
     << "_8 : f64\n";
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
    for (int k = 0; k != getCellSize(); ++k)
      os << inner << "%edge" << here << "_" << k << " = memref.load "
         << "%box_memory[%c_edge" << k << "] : " << getBoxMemoryType()
         << "\n";
    emitCellOf(inner, cellName, "%edge" + here);
    if (scalesReference()) {
      scaleName = "%rest_scale" + here;
      emitReferenceScale(os, inner, scaleName, "%edge" + here + "_0",
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
    std::string noise1 = emitNoiseValue(inner);
    os << inner << getValues("b" + here) << " = dyn.step @step(";
    os << "%xa" << here << ", %va" << here << ", %fa" << here << ", "
         << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise1
         << ")\n"
         << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
         << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec)\n";
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
      for (int k = 0; k != getCellSize(); ++k)
        os << inner << "%edger" << here << "_" << k << " = memref.load "
           << "%box_memory[%c_edge" << k << "] : " << getBoxMemoryType()
           << "\n";
      emitCellOf(inner, cellName, "%edger" + here);
      if (scalesReference()) {
        scaleName = "%rest_scaler" + here;
        emitReferenceScale(os, inner, scaleName, "%edger" + here + "_0",
                           "%edger" + here + "_1", "%edger" + here + "_2");
      }
    }

    // The iteration of the loop of `upto` that is under way, counted over
    // the whole run.
    auto emitIteration = [&](unsigned upto) {
      std::string counted;
      for (unsigned i = nestBegin; i <= upto; ++i) {
        std::string index = "%i" + std::to_string(i);
        if (counted.empty()) {
          counted = index;
          continue;
        }
        std::string scaled = "%s" + here + "_" + std::to_string(i);
        os << inner << scaled << "m = arith.muli " << counted << ", %n" << i
           << " : index\n";
        os << inner << scaled << " = arith.addi " << scaled << "m, " << index
           << " : index\n";
        counted = scaled;
      }
      return counted;
    };
    // The number of the step that has just been taken.
    auto emitStep = [&]() {
      if (current.name == "couple" && level != nestBegin &&
          levels[level - 1].name == "energy") {
        // The interval between energies holds one period more than its
        // loop over periods: the steps before the interval, and those of
        // the periods of this interval so far.
        std::string outer = std::to_string(level - 1);
        std::string counted = emitIteration(level - 1);
        os << inner << "%before" << here << " = arith.muli " << counted
           << ", %per" << outer << " : index\n";
        os << inner << "%done" << here << " = arith.addi %i" << here
           << ", %c1 : index\n";
        os << inner << "%within" << here << " = arith.muli %done" << here
           << ", %per" << here << " : index\n";
        os << inner << "%steps" << here << " = arith.addi %before" << here
           << ", %within" << here << " : index\n";
      } else {
        std::string counted = emitIteration(level);
        os << inner << "%done" << here << " = arith.addi " << counted
           << ", %c1 : index\n";
        os << inner << "%steps" << here << " = arith.muli %done" << here
           << ", %per" << here << " : index\n";
      }
      os << inner << "%since" << here << " = arith.index_cast %steps"
         << here << " : index to i64\n";
      os << inner << "%step" << here << " = arith.addi %since" << here
         << ", " << nestStart << " : i64\n";
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
        os << inner << behind << " = dyn.kick " << velocities << ", "
           << coupled.forces << ", " << massName
           << ", %half_back : !vec\n";
        velocities = behind;
      }
      return coupled.positions + ", " + velocities + ", " + coupled.forces;
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
      emitKineticEnergy(os, o + "k", full, massName, inner);
      emitSum3(os, o + "g", excess, inner);
      os << inner << o << "c = arith.constant " << formatReal(2.0 / 3.0)
         << " : f64\n"
         << inner << o << "e = arith.mulf " << o << "c, " << o << "g : f64\n"
         << inner << o << " = arith.addf " << o << "k, " << o << "e : f64\n";
      return o;
    };
    // The velocities of the time of the positions of a plain step of
    // leapfrog, half a kick ahead of those it stores.
    auto emitAhead = [&](StringRef tag, StringRef v, StringRef f) {
      std::string name = ("%vah" + tag).str();
      os << inner << name << "c = arith.constant 5.0e-01 : f64\n"
         << inner << name << "t = arith.mulf " << name << "c, %dt : f64\n"
         << inner << name << " = dyn.kick " << v << ", " << f << ", "
         << massName << ", " << name << "t : !vec\n";
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
        std::string loaded[9];
        for (int slot = 0; slot != 9; ++slot) {
          loaded[slot] = m + "_" + std::to_string(slot);
          os << inner << m << "_i" << slot << " = arith.constant " << slot
             << " : index\n"
             << inner << loaded[slot] << " = memref.load %trotter_memory["
             << m << "_i" << slot << "] : memref<9xf64>\n";
        }
        std::string vectors[3];
        for (int v = 0; v != 3; ++v) {
          vectors[v] = m + "_v" + std::to_string(v);
          os << inner << vectors[v] << " = vector.from_elements "
             << loaded[3 * v] << ", " << loaded[3 * v + 1] << ", "
             << loaded[3 * v + 2] << " : vector<3xf64>\n";
        }
        trace = vectors[0];
        groups = vectors[1];
        kinetic = vectors[2];
        x = "%x" + last;
        v = "%v" + last;
      } else {
        // The step whose pressure gives the strain.
        std::string noise2 = emitNoiseValue(inner);
        os << inner << getValues(a) << ", %w" << a << ", %gw" << a
           << (leapfrog ? ", %vc" + a : "") << " = dyn.step @step_virial(%x"
           << last << ", %v" << last << ", %f" << last << ", " << massName
           << ", " << cellName << ", %dt" << getScaleValue()
           << getFieldValues(fieldPrefix) << noise2 << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getScaleType() << getFieldTypes() << getNoiseType()
           << ") -> (!vec, !vec, !vec, vector<9xf64>, vector<3xf64>"
           << (leapfrog ? ", !vec" : "") << ")\n";
        trace = "%dg" + a;
        emitDiagonal(os, trace, "%w" + a, inner);
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
      std::string noise3 = emitNoiseValue(inner);
      os << inner << "%x" << n << ", %v" << n << ", %f" << n
         << (withEnergy ? ", %u" + n : "") << (virial ? ", %w" + n : "")
         << (virial ? ", %gw" + n : "")
         << (leapfrog ? ", %vc" + n : "") << ", %kh" << n
         << (measuredHere ? ", %khs" + n : "")
         << (withEnergy && reportsSolvent() ? ", %kws" + n : "")
         << " = dyn.step @step_trotter" << (withEnergy ? "_energy" : "")
         << "(" << x << ", " << v << ", %f"
         << (scalesEveryStep() ? last : a) << ", " << massName << ", "
         << scaling.cell << ", %dt, " << scaling.mu << getScaleValue()
         << getFieldValues(fieldPrefix) << noise3 << ")\n"
         << inner
         << "    : (!vec, !vec, !vec, !real, !md.cell, f64, vector<3xf64>"
         << getScaleType() << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec"
         << (withEnergy ? ", f64" : "")
         << (virial ? ", vector<9xf64>, vector<3xf64>" : "")
         << (leapfrog ? ", !vec" : "") << ", vector<3xf64>"
         << (measuredHere ? ", vector<3xf64>" : "")
         << (withEnergy && reportsSolvent() ? ", vector<3xf64>" : "")
         << ")\n";
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
          emitDiagonal(os, trace, "%w" + k, inner);
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
        std::string noise4 = emitNoiseValue(inner);
        os << inner << getValues("k" + here) << ", %uk" << here << ", %wk"
           << here << ", %gwk" << here
           << (isLeapfrog() ? ", " + current : "")
           << (measured ? ", %khsk" + here : "")
           << " = dyn.step @step_"
           << (measuresHalfSteps() ? "coupling" : "energy") << "(%x" << last
           << ", %v" << last << ", %f" << last << ", " << massName << ", "
           << cellName
           << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise4 << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << getNoiseType()
           << ") -> (!vec, !vec, !vec, f64, vector<9xf64>, vector<3xf64>"
           << (isLeapfrog() ? ", !vec" : "")
           << (measured ? ", vector<3xf64>" : "") << ")\n";
        trace = "%dgk" + here;
        emitDiagonal(os, trace, "%wk" + here, inner);
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
      std::string noise5 = emitNoiseValue(inner);
      os << inner << getValues("k" + here)
         << (measured && isLeapfrog() ? ", %vck" + here : "")
         << (measured ? ", %khsk" + here : "")
         << " = dyn.step @step" << (measured ? "_thermostat" : "") << "(";
      os << "%x" << last << ", %v" << last << ", %f" << last << ", "
           << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise5
           << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec"
           << (measured && isLeapfrog() ? ", !vec" : "")
           << (measured ? ", vector<3xf64>" : "") << ")\n";
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
      std::string noise6 = emitNoiseValue((inner + "  "));
      os << inner << "  " << getValues("r" + here) << " = dyn.step @step(";
      os << "%xp" << here << ", %vp" << here << ", %fp" << here << ", "
           << massName << ", " << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix) << noise6
           << ")\n"
           << inner << "      : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
           << getFieldTypes() << getNoiseType() << ") -> (!vec, !vec, !vec)\n";
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
        std::string noise7 = emitNoiseValue(inner);
        os << inner << "%xl, %vl, %fl, %u, %w"
           << (control.isBrownian() ? "" : ", %gwl")
           << (isLeapfrog() ? ", %vn" : "")
           << (measuresHalfSteps() ? ", %khsl" : "")
           << (reportsSolvent() ? ", %kwsl" : "")
           << " = dyn.step @step_energy(%x"
           << last << ", %v" << last << ", %f" << last << ", " << massName
           << ", " << cellName << ", %dt" << getScaleValue()
           << getFieldValues(fieldPrefix) << noise7 << ")\n"
           << inner << "    : (!vec, !vec, !vec, !real, !md.cell, f64"
           << getScaleType() << getFieldTypes() << getNoiseType()
           << ") -> (!vec, !vec, !vec, f64, vector<9xf64>"
           << (control.isBrownian() ? "" : ", vector<3xf64>")
           << (isLeapfrog() ? ", !vec" : "")
           << (measuresHalfSteps() ? ", vector<3xf64>" : "")
           << (reportsSolvent() ? ", vector<3xf64>" : "") << ")\n";
      }
      std::string full = isLeapfrog() ? now : "%vl";
      emitKineticEnergy(os, "%k", full, massName, inner);
      // K_half - K, measured from the velocities of the half steps with
      // constraints and from the forces without (D203).
      std::string excess = emitKineticExcess(
          inner, "l" + here, measuresHalfSteps() ? "%khsl" : "", "%fl", full);
      if (excess.empty())
        os << inner << "%g = arith.constant 0.0 : f64\n";
      else
        emitSum3(os, "%g", excess, inner);
      if (reportsSolvent())
        emitWriteSolvent(inner, "%kwsl");
      // The pressure of the log is that of the barostat, of the groups
      // (D264): the trace of their virial and the
      // kinetic energy of their centers at the half steps.
      if (control.isBrownian()) {
        emitTrace(os, "%tr", virialName, inner);
        os << inner << "%kph = arith.addf %k, %g : f64\n";
      } else {
        emitSum3(os, "%tr", "%gwl", inner);
        GroupKinetic centers =
            emitGroupKinetic(inner, "%xl", full, "%fl", "l" + here);
        emitSum3(os, "%kph", centers.half, inner);
      }
      if (couplesBelow && control.barostat)
        emitDiagonal(os, "%dg", virialName, inner);
      if (!scalesHere)
        emitStep();
      os << inner << "func.call @mdrtWriteEnergies(%step" << here << ", "
         << energyName
         << ", %k, %g, %tr, %kph) : (i64, f64, f64, f64, f64, f64) -> ()\n";
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
          os << inner << time << "_steps = arith.sitofp %step" << here
             << " : i64 to f64\n"
             << inner << time << " = arith.mulf " << time << "_steps, %dt"
             << " : f64\n";
        emitFreeEnergyOutput(inner, "%xl", rowCell, fieldPrefix,
                             "%step" + here, time);
      }
      if (!control.observables.empty()) {
        // The observed terms at the positions after the step (D189).
        std::string time = "%ob_time" + here;
        if (control.usesTime)
          os << inner << time << "_steps = arith.sitofp %step" << here
             << " : i64 to f64\n"
             << inner << time << " = arith.mulf " << time << "_steps, %dt"
             << " : f64\n";
        emitObservablesOutput(inner, "%xl", rowCell, fieldPrefix,
                              "%step" + here, time);
      }
      // Without a periodic cell (D142), whether the particles have spread
      // so far that images interact, at every row of the log as well as at
      // the frames and checkpoints.
      if (!control.periodic && !framesAtEnergies)
        os << inner << "mdrt.host_call @mdrtCheckSpread(%step" << here
           << ", %xl, " << idName << ") : (i64, !vec, !ids)\n";
      std::string optimal;
      if (couplesBelow && rescalesVelocities() && !excess.empty()) {
        os << inner << "%kolc = arith.constant " << formatReal(2.0 / 3.0)
           << " : f64\n"
           << inner << "%kole = arith.mulf %kolc, %g : f64\n"
           << inner << "%kol = arith.addf %k, %kole : f64\n";
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
        os << inner << "mdrt.host_call @mdrtWriteFrame(%step" << here << ", "
           << StringRef(yielded).split(',').first << ", " << idName
           << ") : (i64, !vec, !ids)\n";
      else if (control.segments)
        emitFrameIfDue(inner, "%step" + here,
                       StringRef(yielded).split(',').first, here);
      os << inner << "scf.yield " << yielded << " : " << state << "\n";
    } else {
      if (current.name == "frame") {
        emitStep();
        os << inner << "mdrt.host_call @mdrtWriteFrame(%step" << here
           << ", %x" << last << ", " << idName << ") : (i64, !vec, !ids)\n";
      }
      if (current.name == "segment") {
        // The state as the next step needs it.
        emitStep();
        os << inner << "mdrt.host_call @mdrtWriteCheckpoint(%step" << here
           << ", " << getValues(last) << ", " << idName << ")\n"
           << inner << "    : (i64, " << state << ", !ids)\n";
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
  os << "  %u0" << held << modelled << ", %f0" << held << raw << modelled
     << " = md.evaluate @energy(" << x0 << ", %cell" << getFieldValues()
     << getTimeValue("%time0") << ")\n"
     << "      request [energy, forces]\n"
     << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
     << ") -> (f64, !vec)\n";
  if (hasSites())
    emitSpreadSites("  ", x0, "%f0" + held + "e", "%f0" + held, "%r_");
  if (hasForceCorrections()) {
    os << "  %w0z = arith.constant dense<0.0> : vector<9xf64>\n";
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
    os << "  %h0 = arith.constant "
       << formatReal(control.minimizeStep * units::length) << " : f64\n";
  // A step grows to at most 0.1 Å (D260): longer ones turn
  // rigid groups beyond what the constraints take back, and are not taken
  // (D259); on 103,292 OPC waters the largest force fell
  // below 5 kcal/mol/Å in 500 steps rather than 3300 with a cap of 1 Å.
  os << "  %h_most = arith.constant "
     << formatReal(Control::minimizeStepCap * units::length)
     << " : f64\n"
     << "  %grow = arith.constant 1.2 : f64\n"
     << "  %shrink = arith.constant 0.2 : f64\n"
     << "  %one = arith.constant 1.0 : f64\n"
     << "  %zero = arith.constant 0.0 : f64\n"
     ;
  // The log has the forces without their parts along the constraints,
  // those that the minimization lowers.
  auto emitReported = [&](StringRef indent, StringRef x, StringRef f,
                          StringRef tag) {
    std::string direction = ("%gr" + tag).str();
    emitConstrainedDescent(indent, x, f, direction);
    std::string reported = ("%fr" + tag).str();
    os << indent << reported << " = md.map_particles gather(" << direction
       << ", %m : !vec, !real) {\n"
       << indent << "^bb0(%g_i: vector<3xf64>, %m_i: f64):\n"
       << indent << "  %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << indent << "  %f_i = arith.mulf %mb, %g_i : vector<3xf64>\n"
       << indent << "  md.yield %f_i : vector<3xf64>\n"
       << indent << "} : !vec\n";
    return reported;
  };
  std::string reported = emitReported("  ", x0, "%f0", "0");
  os << "  mdrt.host_call @mdrtWriteMinimization(%start, %u0, " << h0 << ", "
     << reported << ", %id)\n"
     << "      : (i64, f64, f64, !vec, !ids)\n";
  // The host compares the largest force of each row with the tolerance
  // (D219), and the length of the next step with the floor
  // (D274), and keeps, in a buffer of its own, whether a
  // row met either and the step of that row (the last step otherwise).
  // The loops read it where an iteration begins: the steps and the rows
  // after that row are loops of no iteration. A row is where the host
  // waits for the device anyway, so the check takes no wait of its own.
  bool stops = stopsMinimization();
  if (stops) {
    os << "  %conv = memref.alloca() : memref<2xi64>\n"
       << "  %conv_at0 = arith.constant 0 : index\n"
       << "  %conv_at1 = arith.constant 1 : index\n"
       << "  %conv_no = arith.constant 0 : i64\n"
       << "  %conv_steps = arith.constant " << control.numSteps << " : i64\n"
       << "  %conv_last = arith.addi %start, %conv_steps : i64\n"
       << "  memref.store %conv_no, %conv[%conv_at0] : memref<2xi64>\n"
       << "  memref.store %conv_last, %conv[%conv_at1] : memref<2xi64>\n"
       << "  mdrt.host_call @mdrtCheckMinimization(%conv) : (memref<2xi64>)\n";
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
    os << "  %n0 = arith.constant 1 : index\n"
       << "  %n1 = arith.constant 1 : index\n"
       << "  %n2 = arith.index_cast %count_outer : i64 to index\n"
       << "  %per0 = arith.addi %n2, %c0 : index\n";
  } else {
    int64_t period = control.energyPeriod;
    int64_t framePeriod =
        control.framePeriod > 0 ? control.framePeriod : control.numSteps;
    os << "  %n0 = arith.constant " << control.numSteps / framePeriod
       << " : index\n"
       << "  %n1 = arith.constant " << framePeriod / period << " : index\n"
       << "  %n2 = arith.constant " << period << " : index\n"
       << "  %per0 = arith.constant " << framePeriod << " : index\n";
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
    os << "      %conv_done = memref.load %conv[%conv_at0] : memref<2xi64>\n"
       << "      %conv_open = arith.cmpi eq, %conv_done, %conv_no : i64\n"
       << "      %n2_open = arith.select %conv_open, %n2, %c0 : index\n"
       << "      %row_open = arith.select %conv_open, %c1, %c0 : index\n";
    n2 = "%n2_open";
  }
  os << "      %xe2, %fe2, %ue2, %he2 = scf.for %i2 = %c0 to " << n2
     << " step %c1\n"
     << "          iter_args(%xa2 = %xa1, %fa2 = %fa1, %ua2 = %ua1, "
        "%ha2 = %ha1)\n"
     << "          -> (" << state << ") {\n"
     << "        %xt, %ft, %ut = dyn.step @descend(%xa2, %fa2, %m, %cell, "
        "%ha2" << getFieldValues() << ")\n"
     << "            : (!vec, !vec, !real, !md.cell, f64" << getFieldTypes()
     << ") -> (!vec, !vec, f64)\n"
     << "        %lower = arith.cmpf olt, %ut, %ua2 : f64\n"
     << "        %longer = arith.mulf %ha2, %grow : f64\n"
     << "        %capped = arith.minimumf %longer, %h_most : f64\n"
     << "        %shorter = arith.mulf %ha2, %shrink : f64\n"
     << "        %ub = arith.select %lower, %ut, %ua2 : f64\n"
     << "        %hb = arith.select %lower, %capped, %shorter : f64\n"
     << "        %keep = arith.select %lower, %one, %zero : f64\n";
  // The fields are chosen particle by particle, so that each has storage
  // of its own.
  for (StringRef field : {"x", "f"})
    os << "        %" << field << "b = md.map_particles gather(%" << field
       << "t, %" << field << "a2 : !vec, !vec) {\n"
       << "        ^bb0(%new_i: vector<3xf64>, %old_i: vector<3xf64>):\n"
       << "          %half = arith.constant 5.0e-01 : f64\n"
       << "          %taken = arith.cmpf ogt, %keep, %half : f64\n"
       << "          %chosen = arith.select %taken, %new_i, %old_i "
          ": vector<3xf64>\n"
       << "          md.yield %chosen : vector<3xf64>\n"
       << "        } : !vec\n";
  os << "        scf.yield %xb, %fb, %ub, %hb : " << state << "\n"
     << "      }\n"
     << "      %before = arith.muli %i0, %per0 : index\n"
     << "      %done = arith.addi %i1, %c1 : index\n"
     << "      %within = arith.muli %done, %n2 : index\n"
     << "      %steps = arith.addi %before, %within : index\n"
     << "      %since = arith.index_cast %steps : index to i64\n"
     << "      %step = arith.addi %since, %start : i64\n";
  // A row, unless an earlier one converged: a loop of one iteration or
  // none, as emitFrameIfDue.
  std::string rowIndent = stops ? "        " : "      ";
  if (stops)
    os << "      scf.for %i_row = %c0 to %row_open step %c1 {\n";
  reported = emitReported(rowIndent, "%xe2", "%fe2", "1");
  os << rowIndent << "mdrt.host_call @mdrtWriteMinimization(%step, %ue2, "
     << "%he2, " << reported << ", %id)\n"
     << rowIndent << "    : (i64, f64, f64, !vec, !ids)\n";
  if (stops)
    os << "        mdrt.host_call @mdrtCheckMinimization(%conv) "
          ": (memref<2xi64>)\n"
       << "      }\n";
  os << "      scf.yield %xe2, %fe2, %ue2, %he2 : " << state << "\n"
     << "    }\n";
  if (control.framePeriod > 0) {
    os << "    %frames = arith.addi %i0, %c1 : index\n"
       << "    %frame_steps = arith.muli %frames, %per0 : index\n"
       << "    %frame_since = arith.index_cast %frame_steps : index to i64\n"
       << "    %frame_step = arith.addi %frame_since, %start : i64\n";
    std::string frameIndent = "    ";
    if (stops) {
      // The frame at the row that converged, and none after it.
      os << "    %conv_fdone = memref.load %conv[%conv_at0] : memref<2xi64>\n"
         << "    %conv_fstep = memref.load %conv[%conv_at1] : memref<2xi64>\n"
         << "    %conv_fopen = arith.cmpi eq, %conv_fdone, %conv_no : i64\n"
         << "    %conv_fat = arith.cmpi eq, %conv_fstep, %frame_step : i64\n"
         << "    %conv_fdue = arith.ori %conv_fopen, %conv_fat : i1\n"
         << "    %conv_fn = arith.select %conv_fdue, %c1, %c0 : index\n"
         << "    scf.for %i_frame = %c0 to %conv_fn step %c1 {\n";
      frameIndent = "      ";
    }
    os << frameIndent << "mdrt.host_call @mdrtWriteFrame(%frame_step, %xe1, "
       << "%id) : (i64, !vec, !ids)\n";
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
      os << "  %end = memref.load %conv[%conv_at1] : memref<2xi64>\n";
    else
      os << "  %c_total = arith.constant " << control.numSteps << " : i64\n"
         << "  %end = arith.addi %start, %c_total : i64\n";
    os << "  mdrt.host_call @mdrtWriteCheckpoint(%end, %xe0, %v0, %id)\n"
       << "      : (i64, !vec, !vec, !ids)\n";
  }
  os << "  mdrt.host_call @mdrtFinish(%xe0, %v0, %id) : (!vec, !vec, !ids)\n";
  os << "  return\n}\n";
}

void Builder::emitEntry() {
  collectStartValues();
  StringRef state = getName(program.state);
  StringRef force = getName(program.force);
  StringRef mass = getName(program.mass);
  StringRef parameter = getName(program.parameter);

  os << "func.func private @mdrtWriteEnergies(i64, f64, f64, f64, f64, f64)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWritePull(i64, memref<?xf64>, memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteFreeEnergy(i64, memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteObservables(i64, memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteTerms(memref<?xf64>)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << (program.tunableGradient
             ? "func.func private @mdrtWriteTunableGradient(memref<?xf64>)\n"
               "    attributes {llvm.emit_c_interface}\n"
               "func.func private @mdrtWriteTunableGradientField(i64, "
               "memref<?xf64>, memref<?xi32>)\n"
               "    attributes {llvm.emit_c_interface}\n"
             : "")
     << "func.func private @mdrtWriteVirial(f64, f64, f64)\n"
     << "    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtWriteFrame(i64, memref<?x3x" << state
     << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n"
     << "func.func private @mdrtCheckSpread(i64, memref<?x3x" << state
     << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n"
     ;
  // A program of segments hands its state to the host where it is at the
  // end of each part, and takes the counts of the next part there
  // (D215); a run ends with its state in the buffers of the
  // host.
  if (control.frames)
    // The boundary of a frame also says what the host asks of the frame
    // (D252): its forces, its virial. The two calls below
    // hand them over, each in a branch that a frame without the request
    // does not take.
    os << "func.func private @mdrtFrameBoundary(memref<6xf64>, memref<2xf64>)"
          " -> memref<?x3x"
       << state << ">\n    attributes {llvm.emit_c_interface}\n"
       << "func.func private @mdrtWriteFrameForces(memref<?x3x" << force
       << ">, memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n"
       << "func.func private @mdrtWriteFrameVirial(f64, f64, f64, f64, f64, "
          "f64, f64, f64, f64)\n    attributes {llvm.emit_c_interface}\n";
  else if (control.segments)
    os << "func.func private @mdrtPartBoundary(memref<?x3x" << state
       << ">, memref<?x3x" << state << ">, memref<?x3x" << force
       << ">, memref<?xi32>, f64, memref<" << getPartCountsSize()
       << "xi64>)\n    attributes {llvm.emit_c_interface}\n";
  else
    os << "func.func private @mdrtFinish(memref<?x3x" << state
       << ">, memref<?x3x" << state << ">, memref<?xi32>)\n"
       << "    attributes {llvm.emit_c_interface}\n";
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
  if (control.minimize)
    os << "func.func private @mdrtWriteMinimization(i64, f64, f64, memref<?x3x"
       << force << ">, memref<?xi32>)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (stopsMinimization())
    os << "func.func private @mdrtCheckMinimization(memref<2xi64>)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (reportsSolvent())
    os << "func.func private @mdrtWriteSolvent(f64, f64)\n";
  if (hasConstraints() && !control.minimize)
    // With their effects, which a program that calls them states
    // (D355): each records or delivers a stop of the run.
    os << "func.func private @mdrtStopConstraintFailure(i64, i64)\n"
       << "    attributes {dyn.effects = {write = [\"dyn.stop\"]}}\n"
       << "func.func private @mdrtDeferStops()\n"
       << "    attributes {dyn.effects = {write = [\"dyn.stop\"]}}\n"
       << "func.func private @mdrtStopPending()\n"
       << "    attributes {dyn.effects = {read = [\"dyn.stop\"], "
          "write = [\"dyn.stop\"]}}\n";
  if (rescalesVelocities() && control.isNoseHoover())
    os << "func.func private @mdrtNoseHooverFactor(i64, f64) -> f64\n";
  else if (rescalesVelocities())
    os << "func.func private @mdrtBussiFactor(i64, i64, f64, f64, f64, f64) "
          "-> f64\n";
  if (control.getCouplingPeriod() > 0)
    os << "func.func private @mdrtAddBath(f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (control.barostat)
    os << "func.func private @mdrtBarostatStrain(i64, i64, f64, f64, f64, f64, "
          "f64, f64) -> f64\n"
       << "func.func private @mdrtBarostatStrainArea(i64, i64, f64, f64, f64, "
          "f64, f64, f64, f64, f64) -> f64\n"
       << "func.func private @mdrtBarostatStrainHeight(i64, i64, f64, f64, "
          "f64, f64, f64, f64) -> f64\n"
       << "func.func private @mdrtBarostatStrainAxis(i64, i64, i64, f64, f64, "
          "f64, f64, f64, f64) -> f64\n"
       << "func.func private @mdrtSetBox(f64, f64, f64)\n"
       << "    attributes {llvm.emit_c_interface}\n"
       << "func.func private @mdrtSetTilt(f64, f64, f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (scalesEveryStep())
    os << "func.func private @mdrtSetBarostatState(f64, f64, f64, f64, f64, "
          "f64, f64, f64, f64)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  if (control.checkpointPeriod > 0) {
    os << "func.func private @mdrtWriteCheckpoint(i64, memref<?x3x" << state
       << ">, memref<?x3x" << state << ">";
    if (program.writesForces)
      os << ", memref<?x3x" << force << ">";
    os << ", memref<?xi32>)\n    attributes {llvm.emit_c_interface}\n";
  }
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

  os << "  %c0 = arith.constant 0 : index\n"
     << "  %c1 = arith.constant 1 : index\n";
  if (branchesStart())
    os << "  %first_none = arith.constant 0 : i64\n"
       << "  %is_first = arith.cmpi ne, %first_call, %first_none : i64\n";
  // The time of the first step, which the evaluations at the start take
  // (D145); a minimization takes 0.
  if (control.usesTime) {
    if (control.minimize)
      os << "  %time0 = arith.constant 0.0 : f64\n";
    else
      os << "  %time0_steps = arith.sitofp %start : i64 to f64\n"
         << "  %time0 = arith.mulf %time0_steps, %dt : f64\n";
  }
  // The number of the last step taken, which keys the random numbers of
  // Langevin dynamics (D135).
  if (needsStepNumber())
    os << "  %noise_memory = memref.alloca() : memref<1xi64>\n"
       << "  memref.store %start, %noise_memory[%c0] : memref<1xi64>\n"
       << "  %noise_one = arith.constant 1 : i64\n";
  // A minimization has loops of its own (emitMinimization), and a program
  // of segments takes its counts in the loop over parts (emitPartCounts).
  if (!control.segments) {
    for (auto [index, level] : llvm::enumerate(levels))
      os << "  %n" << index << " = arith.constant " << level.count
         << " : index\n";
  }
  // The number of steps in one iteration of each loop. An iteration of
  // the loop over energy intervals takes one step after its loop over
  // steps.
  // An iteration of the loop over the periods of coupling takes one as
  // well (two with the barostat of Trotter type), and one over energy intervals a whole period after its loop over
  // periods.
  int64_t steps = 1;
  for (unsigned i = control.segments ? 0 : levels.size(); i-- != 0;) {
    os << "  %per" << i << " = arith.constant " << steps << " : index\n";
    steps *= levels[i].count;
    if (i == 0)
      continue;
    // With the barostat of Trotter type, two steps (setSchedule).
    if (levels[i - 1].name == "couple")
      steps += usesTrotter() && !scalesEveryStep() ? 2 : 1;
    else if (levels[i - 1].name == "energy")
      steps += levels[i].name == "couple" ? control.getCouplingPeriod() : 1;
  }

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
    // Where the barostat keeps the cell, on the host.
    os << "  %box_memory = memref.alloca() : " << getBoxMemoryType() << "\n";
    // With a scaling every step, the diagonals of the virial and of that of
    // the rigid groups, and the kinetic energy of each axis without the
    // center of mass, of the state that the last step left (D92, D119).
    if (scalesEveryStep())
      os << "  %trotter_memory = memref.alloca() : memref<9xf64>\n";
    // A triclinic cell keeps its tilts after its diagonal (I3 of
    // docs/triclinic.md: the barostat scales them with their columns).
    // The tilts at the start are arguments of the entry
    // (collectStartValues).
    static const char *const initial[] = {"%lx", "%ly", "%lz",
                                          "%tilt_bx", "%tilt_cx", "%tilt_cy"};
    for (int k = 0; k != getCellSize(); ++k)
      os << "  %c_edge" << k << " = arith.constant " << k << " : index\n"
         << "  memref.store " << initial[k] << ", %box_memory[%c_edge" << k
         << "] : " << getBoxMemoryType() << "\n";
  }
  if (scalesReference())
    // The edges of the cell of the file, which the reference positions of
    // the restraints are for, arguments of the entry (collectStartValues);
    // they scale with the cell, axis by axis.
    os << "  %rest_edges = vector.from_elements %rest_edge0, %rest_edge1, "
          "%rest_edge2 : vector<3xf64>\n";
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
    os << "  %frame_positions = func.call @mdrtFrameBoundary(%frame_cell, "
          "%frame_asks) : (memref<6xf64>, memref<2xf64>) -> memref<?x3x"
       << state << ">\n";
    for (int k = 0; k != getCellSize(); ++k)
      os << "  %frame_cell_" << k << " = memref.load %frame_cell[%frame_at"
         << k << "] : memref<6xf64>\n";
    // What the host asks of this frame: its forces, its virial; each a
    // number that is not 0 if it is asked.
    for (int k = 0; k != 2; ++k)
      os << "  %frame_ask_" << k << " = memref.load %frame_asks[%frame_at" << k
         << "] : memref<2xf64>\n";
    if (scalesReference())
      emitReferenceScale(os, "  ", "%rest_scale", "%frame_cell_0",
                         "%frame_cell_1", "%frame_cell_2");
    emitCellOf("  ", "%cell", "%frame_cell");
  } else {
    if (scalesReference())
      emitReferenceScale(os, "  ", "%rest_scale", "%lx", "%ly", "%lz");
    if (isTriclinic()) {
      os << "  %cell = md.triclinic_cell %lx, %ly, %lz, %tilt_bx, %tilt_cx, "
            "%tilt_cy\n";
    } else
      os << "  %cell = md.orthorhombic_cell %lx, %ly, %lz\n";
  }
  os << "  %x" << (program.reorders ? "_in" : "0") << (hasSites() ? "u" : "")
     << " = mdrt.from_buffer "
     << (control.frames ? "%frame_positions" : "%positions")
     << " : memref<?x3x" << state << "> to !vec\n";
  if (control.frames) {
    os.flush();
    frameHead = program.module.substr(frameMark);
    program.module.resize(frameMark);
  }
  os << "  " << (program.reorders ? "%v_in" : velocities)
     << " = mdrt.from_buffer %velocities : memref<?x3x" << state
     << "> to !vec\n"
     << "  %m" << given << " = mdrt.from_buffer %masses : memref<?x" << mass
     << "> to !real\n";
  for (const Program::Field &field : program.fields)
    os << "  %p" << given << "_" << field.name << " = mdrt.from_buffer %b_"
       << field.name << " : memref<?x"
       << (field.isInteger ? StringRef("i32") : parameter) << "> to "
       << getFieldType(field) << "\n";
  for (const Program::Table &table : program.tables)
    os << "  %t_" << table.name << " = mdrt.from_buffer %bt_" << table.name
       << " : memref<?x?xf64> to " << table.getType() << "\n";
  for (const Program::TupleSet &set : program.tupleSets) {
    // The members in the order of the files, which a new order of the
    // particles renumbers.
    os << "  %r" << (program.reorders ? "o" : "") << "_" << set.name
       << " = mdrt.from_buffer %br_" << set.name << " : memref<?x"
       << set.arity << "xi32> to !rel_" << set.name << "\n";
    for (const Program::Field &field : set.fields)
      os << "  %f_" << set.name << "_" << field.name
         << " = mdrt.from_buffer %bf_" << set.name << "_" << field.name
         << " : memref<?xf64> to !of_" << set.name << "\n";
  }
  os << "  %id" << given
     << " = mdrt.from_buffer %identities : memref<?xi32> to !ids\n";
  // A run that continues an earlier one has the forces of the step before.
  if (givenForces)
    os << "  " << (program.reorders ? "%f_in" : givenForcesName)
       << " = mdrt.from_buffer %forces : memref<?x3x" << force
       << "> to !vec\n";
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
    return;
  }

  if (control.getCouplingPeriod() > 0) {
    // What the coupling of the velocities takes: the total mass; and for
    // the thermostat the key of the random numbers, the degrees of freedom,
    // the mean kinetic energy at the temperature of the bath, and how much
    // of the kinetic energy is kept over one period.
    os << "  %couple_half = arith.constant 5.0e-01 : f64\n"
       << "  %total_mass = md.sum_particles gather(%m" << given
       << " : !real) {\n"
       << "  ^bb0(%m_i: f64):\n"
       << "    %mb = vector.broadcast %m_i : f64 to vector<3xf64>\n"
       << "    md.yield %mb : vector<3xf64>\n"
       << "  } : vector<3xf64>\n";
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
        for (int k = 0; k != 3; ++k)
          os << "  %baro_beta_" << k << " = arith.constant "
             << formatReal(control.compressibilities[k] / bar) << " : f64\n";
        os << "  %baro_target = arith.constant "
           << formatReal(control.pressure * bar) << " : f64\n"
           << "  %baro_beta = arith.constant "
           << formatReal(control.compressibility / bar) << " : f64\n"
           << "  %baro_rate = arith.constant "
           << formatReal(static_cast<double>(control.barostatPeriod) *
                         control.timestep / control.tauP)
           << " : f64\n"
           << "  %baro_kt = arith.constant "
           << formatReal(units::boltzmann * control.temperature) << " : f64\n";
        os << "  %c_bar = arith.constant 16.6053906717 : f64\n"
           << "  %c_three = arith.constant 3.0 : f64\n"
           << "  %c_third = arith.constant "
           << formatReal(1.0 / 3.0) << " : f64\n"
           << "  %c_unit = arith.constant 1.0 : f64\n"
           << "  %c_unit3 = arith.constant dense<1.0> : vector<3xf64>\n"
           << "  %c_half3 = arith.constant dense<5.0e-01> : vector<3xf64>\n"
           << "  %baro_beta_z = arith.constant "
           << formatReal(control.compressibilityZ / bar) << " : f64\n"
           << "  %baro_tension = arith.constant "
           << formatReal(control.surfaceTension * control.surfaces *
                         units::dynePerCmToBarNm)
           << " : f64\n";
      }
      os << "  %seed = arith.constant " << static_cast<int64_t>(control.seed)
         << " : i64\n"
         << "  %freedom = arith.constant " << formatReal(freedom) << " : f64\n"
         << "  %target_kinetic = arith.constant " << formatReal(target)
         << " : f64\n"
         << "  %decay = arith.constant " << formatReal(decay) << " : f64\n";
    }
  }

  if (isLeapfrog() && control.barostat)
    os << "  %c_half_back = arith.constant -5.0e-01 : f64\n"
       << "  %half_back = arith.mulf %c_half_back, %dt : f64\n";

  // The energy, the forces, and the virial of the state at the start, as
  // %u0, %f0, and %w0; returns the name of the virial.
  auto emitStartForces = [&]() -> std::string {
    StringRef raw = hasSites() ? "e" : "";
    std::string held = hasForceCorrections() ? "p" : "";
    // With a learned potential, `%f0c` and `%u0c` before its terms
    // (D270).
    bool hostTerms = hasHostEvaluation();
    std::string modelled = !hostTerms || !held.empty() ? "" : "c";
    os << "  %u0" << held << modelled << ", %f0" << held << raw << modelled
       << ", %w0" << held << raw
       << " = md.evaluate @energy(%x0, %cell" << getFieldValues()
       << getTimeValue("%time0") << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> (f64, !vec, vector<9xf64>)\n";
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
        os << "    " << f << held << raw;
        if (withVirial)
          os << ", " << w << held << raw;
        os << " = md.evaluate @energy(%x0, %cell" << getFieldValues()
           << getTimeValue("%time0") << ")\n"
           << "        request [forces" << (withVirial ? ", virial" : "")
           << "]\n        : (!vec, !md.cell" << getFieldTypes() << getTimeType()
           << ") -> " << (withVirial ? "(!vec, vector<9xf64>)" : "!vec")
           << "\n";
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
      auto emitForces = [&](StringRef indent, StringRef f) {
        os << indent << "mdrt.host_call @mdrtWriteFrameForces(" << f << ", "
           << idName << ") : (!vec, !ids)\n";
      };
      os << "  %frame_none = arith.constant 0.0 : f64\n"
         << "  %frame_wf = arith.cmpf one, %frame_ask_0, %frame_none : f64\n"
         << "  %frame_ww = arith.cmpf one, %frame_ask_1, %frame_none : f64\n"
         << "  %frame_nf = arith.select %frame_wf, %c1, %c0 : index\n"
         << "  %frame_nw = arith.select %frame_ww, %c1, %c0 : index\n"
         // The forces alone: asked without the virial.
         << "  %frame_no = arith.select %frame_ww, %c0, %frame_nf : index\n"
         << "  scf.for %frame_if = %c0 to %frame_no step %c1 {\n";
      emitForces("    ", emitAsked("g", false).first);
      os << "  }\n"
         << "  scf.for %frame_iw = %c0 to %frame_nw step %c1 {\n";
      auto [forces, tensor] = emitAsked("w", true);
      os << "    scf.for %frame_ib = %c0 to %frame_nf step %c1 {\n";
      emitForces("      ", forces);
      os << "    }\n";
      for (int k = 0; k != 9; ++k)
        os << "    %frame_w" << k << " = vector.extract " << tensor << "[" << k
           << "] : f64 from vector<9xf64>\n";
      os << "    func.call @mdrtWriteFrameVirial(";
      for (int k = 0; k != 9; ++k)
        os << (k ? ", " : "") << "%frame_w" << k;
      os << ")\n        : (f64, f64, f64, f64, f64, f64, f64, f64, f64) -> ()\n"
         << "  }\n";
    }
    emitTerms();
    // The diagonal of the virial, which semi-isotropic coupling takes axis
    // by axis, for comparison with other programs (D119).
    if (system.topology) {
      emitDiagonal(os, "%dgw0", virial, "  ");
      os << "  func.call @mdrtWriteVirial(%dgw0_0, %dgw0_4, %dgw0_8)"
            " : (f64, f64, f64) -> ()\n";
    }
    emitKineticEnergy(os, "%k0", velocities, "%m", "  ");
    // K_half - K from the forces without constraints; with them there is
    // no step before the start to measure, and the row takes K
    // (D203).
    std::string startExcess =
        emitKineticExcess("  ", "s0", "", "%f0", velocities);
    if (startExcess.empty())
      os << "  %g0 = arith.constant 0.0 : f64\n";
    else
      emitSum3(os, "%g0", startExcess, "  ");
    std::string trace = "%tr0", pressureKinetic = "%kph0";
    if (control.frames || control.isBrownian()) {
      // The frame evaluator reports the atomic virial of the forces.
      emitTrace(os, "%tr0", virial, "  ");
      if (hasConstraints())
        trace = emitStartConstraintTrace("%x0", "%f0", velocities, trace);
      os << "  %kph0 = arith.addf %k0, %g0 : f64\n";
    } else {
      // The pressure of the groups, whose kinetic energy at the half steps
      // the forces give with constraints as well
      // (D264).
      emitDiagonal(os, "%dgp0", virial, "  ");
      std::string groups = emitMolecularDiagonal(
          "  ", "%dgp0", "%x0", "%f0", "%cell", "%m", "%r_", "p0");
      emitSum3(os, "%tr0", groups, "  ");
      GroupKinetic centers =
          emitGroupKinetic("  ", "%x0", velocities, "%f0", "s0");
      emitSum3(os, "%kph0", centers.half, "  ");
    }
    os << "  func.call @mdrtWriteEnergies(%start, %u0, %k0, %g0, " << trace
       << ", " << pressureKinetic << ")\n"
       << "      : (i64, f64, f64, f64, f64, f64) -> ()\n";
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
      emitDiagonal(os, "%dg0", virial, "  ");
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
      os << "  %back = arith.constant -5.0e-01 : f64\n"
         << "  %back_dt = arith.mulf %back, %dt : f64\n"
         << "  %back_none = arith.constant 0.0 : f64\n"
         << "  %back_first = arith.constant 1 : i64\n"
         << "  %back_begins = arith.cmpi eq, %first_call, %back_first : i64\n"
         << "  %behind = arith.select %back_begins, %back_dt, %back_none"
            " : f64\n"
         << "  %v0 = dyn.kick %vg, %f0, %m, %behind : !vec\n";
    } else if (isLeapfrog()) {
      // v(-dt/2) = v(0) - (dt/2) F(0) / m.
      os << "  %back = arith.constant -5.0e-01 : f64\n"
         << "  %behind = arith.mulf %back, %dt : f64\n"
         << "  %v0 = dyn.kick %vg, %f0, %m, %behind : !vec\n";
    }
  }
  if (control.frames) {
    // The loop over frames, which has no end of its own, as the loop over
    // parts has none: each iteration waits at its boundary for the cell and
    // the positions of a frame, places the virtual sites, and evaluates
    // what the start of a run evaluates, in a segment of its own, so that
    // the neighbor structures are built anew (D218).
    os.flush();
    std::string chunk = frameHead + "  scf.for %i_frame_once = %c0 to %c1 "
                                    "step %c1 {\n" +
                        program.module.substr(startMark) +
                        "  } {mdrt.segment}\n";
    program.module.resize(startMark);
    // What this assembly assumes of the text, which would otherwise break
    // without a sign (#302).
    frameLoopError = checkFrameLoop(chunk);
    os << "  %frame_cell = memref.alloca() : memref<6xf64>\n"
       << "  %frame_asks = memref.alloca() : memref<2xf64>\n";
    // The places of the cell, of which the first two are those of what the
    // host asks as well.
    for (int k = getCellSize(); k < 2; ++k)
      os << "  %frame_at" << k << " = arith.constant " << k << " : index\n";
    for (int k = 0; k != getCellSize(); ++k)
      os << "  %frame_at" << k << " = arith.constant " << k << " : index\n";
    os << "  %frame_end = arith.constant 9223372036854775807 : index\n"
       << "  scf.for %i_frame = %c0 to %frame_end step %c1 {\n"
       << chunk << "  }\n  return\n}\n";
    return;
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
    std::string vectors[3];
    for (int v = 0; v != 3; ++v) {
      std::string names[3];
      for (int k = 0; k != 3; ++k)
        names[k] = "%bstate" + std::to_string(3 * v + k);
      vectors[v] = "%bstatev" + std::to_string(v);
      os << "  " << vectors[v] << " = vector.from_elements " << names[0]
         << ", " << names[1] << ", " << names[2] << " : vector<3xf64>\n";
    }
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
    os << "  %ubs" << held << modelled << ", %fbs" << held << raw << modelled
       << ", %wbs" << held << raw
       << " = md.evaluate @energy(%x0, %cell" << getFieldValues()
       << getTimeValue("%time0") << ")\n"
       << "      request [energy, forces, virial]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> (f64, !vec, vector<9xf64>)\n";
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
      os << "  %ahead = arith.constant 5.0e-01 : f64\n"
         << "  %ahead_dt = arith.mulf %ahead, %dt : f64\n"
         << "  %vs_now = dyn.kick " << velocities
         << ", %fbs, %m, %ahead_dt : !vec\n";
      current = "%vs_now";
    }
    emitDiagonal(os, "%dgs", virial, "  ");
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
      for (int k = 0; k != getCellSize(); ++k)
        os << "  %edge_tail_" << k << " = memref.load %box_memory[%c_edge"
           << k << "] : " << getBoxMemoryType() << "\n";
      emitCellOf("  ", cellName, "%edge_tail");
      if (scalesReference()) {
        scaleName = "%rest_scale_tail";
        emitReferenceScale(os, "  ", scaleName, "%edge_tail_0",
                           "%edge_tail_1", "%edge_tail_2");
      }
    }
    os << "  %xt, %vt, %ft = scf.for %i_tail = %c0 to %n_tail step %c1\n"
       << "      iter_args(%xta = %xe0, %vta = %ve0, %fta = %fe0)\n"
       << "      -> (!vec, !vec, !vec) {\n";
    std::string noise = emitNoiseValue("    ");
    os << "    %xtb, %vtb, %ftb = dyn.step @step(%xta, %vta, %fta, "
       << massName << ", " << cellName << ", %dt" << getScaleValue()
       << getFieldValues(fieldPrefix) << noise << ")\n"
       << "        : (!vec, !vec, !vec, !real, !md.cell, f64"
       << getScaleType() << getFieldTypes() << getNoiseType()
       << ") -> (!vec, !vec, !vec)\n"
       << "    scf.yield %xtb, %vtb, %ftb : !vec, !vec, !vec\n"
       << "  }\n";
    last = emitSegmentEnergyStep("%xt", "%vt", "%ft");
    cellName = outerCell;
    scaleName = outerScale;
    // Intervals that end with a step of energy, as an interval of the log
    // of `mdir run` ends, after the steps of the segment before them: one
    // period that closes, or the intervals of the reports
    // (D207).
    unsigned second = levels[0].name == "couple" ? 2 : 1;
    os << "  %steps_a = arith.muli %n0, %per0 : index\n"
       << "  %steps_t = arith.addi %steps_a, %n_tail : index\n"
       << "  %steps_p = arith.addi %steps_t, %n_plain : index\n"
       << "  %steps_e = arith.index_cast %steps_p : index to i64\n"
       << "  %start_e = arith.addi %start, %steps_e : i64\n";
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
    return;
  }
  emitLevel(0, "  ");
  os << "  mdrt.host_call @mdrtFinish(%x" << last << ", %v" << last << ", "
     << idName << ") : (!vec, !vec, !ids)\n";
  os << "  return\n}\n";
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
  static const char *const names[] = {
      "%start$part",        "%count_outer",        "%count_inner",
      "%count_tail",        "%count_energy_plain", "%count_energy_close",
      "%count_energy_inner", "%count_energy_periods", "%frame_period"};
  static_assert(sizeof(names) / sizeof(names[0]) == 9, "");
  int64_t count = control.minimize ? 2 : getPartCountsSize();
  for (int64_t k = 0; k != count; ++k)
    os << "    %part_at" << k << " = arith.constant " << k << " : index\n"
       << "    " << names[k] << " = memref.load %part_counts[%part_at" << k
       << "] : memref<" << getPartCountsSize() << "xi64>\n";
  if (control.minimize)
    return;
  // The first nest is over the periods of coupling, with plain steps
  // inside, or over steps; the second, if any, is one period that closes
  // with a step of energy.
  bool couples = levels[0].name == "couple";
  os << "    %n0 = arith.index_cast %count_outer : i64 to index\n";
  if (couples)
    os << "    %n1 = arith.index_cast %count_inner : i64 to index\n"
       << "    %n2 = arith.index_cast %count_energy_close : i64 to index\n"
       << "    %n3 = arith.index_cast %count_energy_periods : i64 to index\n"
       << "    %n4 = arith.index_cast %count_energy_inner : i64 to index\n";
  else
    os << "    %n1 = arith.index_cast %count_energy_close : i64 to index\n"
       << "    %n2 = arith.index_cast %count_energy_inner : i64 to index\n";
  os << "    %n_tail = arith.index_cast %count_tail : i64 to index\n"
     << "    %n_plain = arith.index_cast %count_energy_plain : i64 to index\n";
  // The steps of an iteration of each loop. An iteration over a period
  // takes its plain steps and those that close it: one, or two with the
  // barostat of Trotter type. The period of the second nest is the whole of
  // its one interval of energy. Without coupling, the second nest is
  // intervals of %n2 plain steps and a step of energy.
  if (couples)
    os << "    %per1 = arith.constant 1 : index\n"
       << "    %closing = arith.constant " << getClosingSteps() << " : index\n"
       << "    %per0 = arith.addi %n1, %closing : index\n"
       << "    %per4 = arith.constant 1 : index\n"
       << "    %per3 = arith.addi %n4, %closing : index\n"
       << "    %periods2 = arith.addi %n3, %c1 : index\n"
       << "    %per2 = arith.muli %periods2, %per3 : index\n";
  else
    os << "    %per0 = arith.constant 1 : index\n"
       << "    %per2 = arith.constant 1 : index\n"
       << "    %per1 = arith.addi %n2, %c1 : index\n";
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
  os << "  %part_counts = memref.alloca() : " << counts << "\n"
     << "  %part_end = arith.constant 9223372036854775807 : index\n"
     << "  ";
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
  os << "    %nf_s = md.sum_particles gather(" << x << ", " << v << ", " << f
     << " : !vec, !vec, !vec) {\n"
     << "    ^bb0(%nf_x: vector<3xf64>, %nf_v: vector<3xf64>, "
        "%nf_f: vector<3xf64>):\n"
     << "      %nf_one = arith.constant dense<1.0> : vector<3xf64>\n"
     << "      %nf_zero = arith.constant dense<0.0> : vector<3xf64>\n";
  for (StringRef c : {"x", "v", "f"})
    os << "      %nf_d" << c << " = arith.subf %nf_" << c << ", %nf_" << c
       << " : vector<3xf64>\n"
       << "      %nf_k" << c << " = arith.cmpf ord, %nf_d" << c << ", %nf_d"
       << c << " : vector<3xf64>\n"
       << "      %nf_b" << c << " = arith.select %nf_k" << c
       << ", %nf_zero, %nf_one : vector<3xi1>, vector<3xf64>\n";
  os << "      %nf_bxv = arith.addf %nf_bx, %nf_bv : vector<3xf64>\n"
     << "      %nf_b = arith.addf %nf_bxv, %nf_bf : vector<3xf64>\n"
     << "      md.yield %nf_b : vector<3xf64>\n"
     << "    } : vector<3xf64>\n";
  emitSum3(os, "%nf", "%nf_s", "    ");
  os << "    mdrt.host_call @mdrtPartBoundary(" << x << ", " << v << ", " << f
     << ", " << idName << ", %nf, %part_counts) {in_place}\n"
     << "        : (!vec, !vec, !vec, !ids, f64, " << counts << ")\n";
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
  std::string t = ("%frame_" + tag).str();
  os << indent << t << "_z = arith.constant 0 : i64\n"
     << indent << t << "_o = arith.constant 1 : i64\n"
     << indent << t << "_h = arith.cmpi sgt, %frame_period, " << t
     << "_z : i64\n"
     << indent << t << "_s = arith.select " << t << "_h, %frame_period, " << t
     << "_o : i64\n"
     << indent << t << "_r = arith.remsi " << step << ", " << t << "_s : i64\n"
     << indent << t << "_e = arith.cmpi eq, " << t << "_r, " << t
     << "_z : i64\n"
     << indent << t << "_d = arith.andi " << t << "_h, " << t << "_e : i1\n"
     // A loop of one iteration or none: the passes that place the buffers
     // carry their storage through scf.for, not scf.if.
     << indent << t << "_w = arith.extui " << t << "_d : i1 to i64\n"
     << indent << t << "_n = arith.index_cast " << t << "_w : i64 to index\n"
     << indent << "scf.for " << t << "_i = %c0 to " << t << "_n step %c1 {\n"
     << indent << "  mdrt.host_call @mdrtWriteFrame(" << step << ", "
     << positions << ", " << idName << ") : (i64, !vec, !ids)\n"
     << indent << "}\n";
}

std::string Builder::emitSegmentEnergyStep(StringRef x, StringRef v,
                                           StringRef f) {
  // At most one plain step, of energy, as `mdir run` takes at a row of its
  // log without coupling: its energies go to mdrtWriteEnergies.
  os << "  %steps_pa = arith.muli %n0, %per0 : index\n"
     << "  %steps_pt = arith.addi %steps_pa, %n_tail : index\n"
     << "  %steps_pe = arith.addi %steps_pt, %c1 : index\n"
     << "  %steps_pi = arith.index_cast %steps_pe : index to i64\n"
     << "  %step_plain = arith.addi %start, %steps_pi : i64\n";
  os << "  %xp, %vp, %fp = scf.for %i_plain = %c0 to %n_plain step %c1\n"
     << "      iter_args(%xpa = " << x << ", %vpa = " << v << ", %fpa = "
     << f << ")\n"
     << "      -> (!vec, !vec, !vec) {\n";
  std::string noise = emitNoiseValue("    ");
  bool brownian = control.isBrownian();
  os << "    %xpl, %vpl, %fpl, %upl, %wpl" << (brownian ? "" : ", %gwpl")
     << (isLeapfrog() ? ", %vpn" : "")
     << (measuresHalfSteps() ? ", %khspl" : "")
     << (reportsSolvent() ? ", %kwspl" : "")
     << " = dyn.step @step_energy(%xpa, %vpa, %fpa, " << massName << ", "
     << cellName << ", %dt" << getScaleValue() << getFieldValues(fieldPrefix)
     << noise << ")\n"
     << "        : (!vec, !vec, !vec, !real, !md.cell, f64" << getScaleType()
     << getFieldTypes() << getNoiseType()
     << ") -> (!vec, !vec, !vec, f64, vector<9xf64>"
     << (brownian ? "" : ", vector<3xf64>")
     << (isLeapfrog() ? ", !vec" : "")
     << (measuresHalfSteps() ? ", vector<3xf64>" : "")
     << (reportsSolvent() ? ", vector<3xf64>" : "") << ")\n";
  if (reportsSolvent())
    emitWriteSolvent("    ", "%kwspl");
  std::string plainFull = isLeapfrog() ? "%vpn" : "%vpl";
  emitKineticEnergy(os, "%kpl", plainFull, massName, "    ");
  std::string plainExcess = emitKineticExcess(
      "    ", "pl", measuresHalfSteps() ? "%khspl" : "", "%fpl", plainFull);
  if (plainExcess.empty())
    os << "    %gpl = arith.constant 0.0 : f64\n";
  else
    emitSum3(os, "%gpl", plainExcess, "    ");
  // The pressure of the groups (D264).
  if (brownian) {
    emitTrace(os, "%trpl", "%wpl", "    ");
    os << "    %kphpl = arith.addf %kpl, %gpl : f64\n";
  } else {
    emitSum3(os, "%trpl", "%gwpl", "    ");
    GroupKinetic centers =
        emitGroupKinetic("    ", "%xpl", plainFull, "%fpl", "pl");
    emitSum3(os, "%kphpl", centers.half, "    ");
  }
  os << "    func.call @mdrtWriteEnergies(%step_plain, %upl, %kpl, %gpl, "
        "%trpl, %kphpl) : (i64, f64, f64, f64, f64, f64) -> ()\n";
  if (!control.observables.empty()) {
    // The observed terms at the positions after the step (D189), as at a
    // row of the second nest.
    if (control.usesTime)
      os << "    %ob_time_pl_steps = arith.sitofp %step_plain : i64 to f64\n"
         << "    %ob_time_pl = arith.mulf %ob_time_pl_steps, %dt : f64\n";
    emitObservablesOutput("    ", "%xpl", cellName, fieldPrefix, "%step_plain",
                          "%ob_time_pl");
  }
  if (!control.periodic)
    os << "    mdrt.host_call @mdrtCheckSpread(%step_plain, %xpl, " << idName
       << ") : (i64, !vec, !ids)\n";
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

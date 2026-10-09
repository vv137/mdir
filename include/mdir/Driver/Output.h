// What a run writes: the log, files of columns, the trajectory, and
// checkpoints (D149, docs/driver-m0.md, Section 2.8).

#ifndef MDIR_DRIVER_OUTPUT_H
#define MDIR_DRIVER_OUTPUT_H

#include "mdir/Driver/Builder.h"
#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"
#include "mdir/Driver/Trajectory.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mdir {
namespace driver {

/// The log of a run: every line goes to the standard output and, once
/// `open` has named a file, to that file as well. What was printed before
/// goes to the file when it opens, so that the file holds the whole log.
class Log {
public:
  Log() = default;
  Log(const Log &) = delete;
  Log &operator=(const Log &) = delete;
  ~Log();

  /// Prints to the standard output and to the file.
  void print(const char *format, ...) __attribute__((format(printf, 2, 3)));
  /// A log that prints nothing: that of a simulation embedded in another
  /// program (D196), whose state the program reads.
  bool quiet = false;
  /// Opens `path`, appending to what it holds if `appends`, and writes
  /// what was printed before.
  llvm::Error open(const std::string &path, bool appends);
  void flush();
  void close();

private:
  std::FILE *file = nullptr;
  std::string pending;
};

/// A file of columns: a line of the names of the columns, a line of their
/// units, `-` for none, both after `#`, and a row for each output, the
/// step an integer and the other values with six decimals.
class ColumnFile {
public:
  struct Column {
    std::string name;
    std::string unit;
    /// Written as an integer, as the step is.
    bool integer = false;
  };

  ColumnFile() = default;
  ColumnFile(const ColumnFile &) = delete;
  ColumnFile &operator=(const ColumnFile &) = delete;
  ~ColumnFile();

  /// Opens `path` for `columns`, the first of which is the step. With
  /// `keepThrough`, the run continues the file: it keeps the rows up to
  /// that step and appends after them, and the file must begin with the
  /// lines that `columns` give, if it exists. Otherwise the file is
  /// written anew.
  llvm::Error open(const std::string &path, std::vector<Column> columns,
                   std::optional<int64_t> keepThrough);
  bool isOpen() const { return file != nullptr; }
  /// Writes a row: the step and a value for each column after it.
  void write(int64_t step, llvm::ArrayRef<double> values);
  void close();

  /// The first two lines of a file of `columns`.
  static std::string getHeader(llvm::ArrayRef<Column> columns);

private:
  std::FILE *file = nullptr;
  std::vector<Column> columns;
};

/// The output of the run that is under way. The functions that compiled
/// code calls write to it.
struct Output {
  Log log;
  /// Records a checkpoint stop before its direct process exit (D168).
  std::function<llvm::Error(int64_t, const char *)> recordStop;
  int64_t lastMinimizationStep = 0;
  /// The last row of a minimization, in kJ/mol and nm: the energy with its
  /// constant parts, the RMS and the largest force without their parts
  /// along the constraints, the particle of the largest (zero-based), and
  /// the length of the next step (D202).
  struct MinimizationRow {
    int64_t step = -1;
    double energy = 0.0, rmsForce = 0.0, maxForce = 0.0, stepSize = 0.0;
    int64_t maxForceParticle = 0;
  } lastMinimization;
  /// The rows of the log as columns (D149), or none.
  ColumnFile energies;
  /// The terms over centers (D145), or none, and the number of
  /// coordinates of each term.
  ColumnFile pull;
  std::vector<int64_t> pullCounts;
  /// [free_energy] (D161): dH/dλ and the differences of the energy to the
  /// other states, or none; the state of the run, and the constant
  /// energies of Program at each state and their derivatives.
  ColumnFile freeEnergy;
  int64_t freeEnergyState = 0;
  /// The energies of terms and their derivatives in constants of theirs
  /// (D189), or none.
  ColumnFile observables;
  /// For a program that embeds the run (D232): the steps
  /// between the rows of `observables`, which the host writes where they
  /// are due, 0 for none; and the values of the last step of energy, in
  /// kJ/mol and per unit of each constant, with the tails at the volume of
  /// the cell, `step` -1 before one.
  int64_t observablesPeriod = 0;
  struct ObservableRow {
    int64_t step = -1;
    std::vector<double> values;
  } lastObservables;
  std::vector<double> stateFixedEnergies, stateVolumeEnergies;
  std::vector<double> lambdaFixedDerivatives, lambdaVolumeDerivatives;
  /// What the correction for the dispersion adds to each column of
  /// `observe`, in kJ/mol at the volume `firstVolume`, proportional to 1 / V
  /// (D209); empty if nothing.
  std::vector<double> observableVolumeConstants;
  /// And what it adds that does not depend on the volume (#224).
  std::vector<double> observableFixedConstants;
  /// The derivatives of the energy in the sites of the tunables that the
  /// entry handed over at its last evaluation of them
  /// (mdrtWriteTunableGradient, D230), in kJ/mol per unit
  /// of each, and the volume of the cell then; `tunableGradientWritten` is
  /// set by the call.
  std::vector<double> tunableGradient;
  /// The fields of the particles that it handed over before them
  /// (mdrtWriteTunableGradientField), each in the order of the input.
  std::vector<std::vector<double>> tunableGradientFields;
  double tunableGradientVolume = 0.0;
  bool tunableGradientWritten = false;
  std::unique_ptr<TrajectoryWriter> trajectory;
  bool hasTrajectory = false;

  /// The types that the buffers of the state and of the forces hold.
  Element state = Element::F64;
  Element force = Element::F64;

  /// The step and the time that the run begins with.
  int64_t firstStep = 0;
  double firstTime = 0.0;
  /// The steps between the rows of the log; energies computed between
  /// them, at more frequent frames, are not shown.
  int64_t energyPeriod = 0;
  double timestep = 0.0;
  double degreesOfFreedom = 0.0;
  /// Those of the rigid waters (System::getSolventDegreesOfFreedom); the
  /// solute has the rest. The kinetic energies of the waters at the step
  /// of the next row, K and K_half, if `hasSolvent`, and for each row
  /// since the start, the temperatures of the solute and of the solvent:
  /// the optimal estimate and that of the velocities of the step
  /// (D203).
  double solventFreedom = 0.0;
  bool hasSolvent = false;
  /// Whether a program embeds the run (a Python simulation, D196): its
  /// state takes the energies of every step of energy, rows or not
  /// (D207).
  bool embedded = false;
  /// A simulation with tunable parameters (D213): the version
  /// of their values, which the energy file takes as its last column, or
  /// -1 for none; and a step at which the energies are evaluated anew after
  /// an update, which takes no row, or -1.
  int64_t tunablesVersion = -1;
  int64_t quietStep = -1;
  double solventKinetic = 0.0, solventHalf = 0.0;
  std::vector<double> soluteOptimal, solventOptimal, soluteFull,
      solventFull;
  /// The last step of the run, and where it stops if a signal or the wall
  /// time asks for a stop: at the next checkpoint before the last step
  /// (D131). `began` is when the run began, `lastCheckpoint` when the last
  /// checkpoint was written, and `longestSegment` the longest time in s
  /// between two checkpoints, which the wall time `maxWalltime` in s, if
  /// not 0, must leave for the next.
  int64_t endStep = 0;
  double maxWalltime = 0.0;
  std::chrono::steady_clock::time_point began;
  std::chrono::steady_clock::time_point lastCheckpoint;
  double longestSegment = 0.0;
  /// The trajectory without the directory, which checkpoints record.
  std::string trajectoryName;

  /// The volume of the cell, in nm^3, and the edges, which a barostat
  /// changes (mdrtSetBox).
  double volume = 0.0;
  double box[3] = {0.0, 0.0, 0.0};
  /// The volume that the constants below are for; they are proportional to
  /// 1 / V but the self term.
  double firstVolume = 0.0;
  /// The correction for the dispersion beyond the cutoff, in kJ/mol: what
  /// it adds to the potential energy and to the trace of the virial.
  double dispersionEnergy = 0.0;
  double dispersionVirial = 0.0;
  /// The part of the estimate of the shift that does not depend on the
  /// volume (Program::dispersionFixedEnergy, #224).
  double dispersionFixedEnergy = 0.0;
  /// Particle mesh Ewald: the self term and the background of a net
  /// charge, which the program does not compute.
  bool pme = false;
  bool reactionField = false;
  /// Whether the cell is periodic; without one (D142), the reach of the
  /// neighbor structures in nm, within which no image may come.
  bool periodic = true;
  double listReach = 0.0;
  double coulombConstantEnergy = 0.0;
  double coulombConstantVirial = 0.0;
  /// Of which the self term, which does not depend on the volume.
  double coulombSelfEnergy = 0.0;
  /// Particle mesh Ewald of the dispersion (D162): its self term, which
  /// does not depend on the volume either.
  bool ljpme = false;
  double ljpmeSelfEnergy = 0.0;

  /// The constants at the volume `volume`.
  double getDispersionEnergy() const {
    return dispersionEnergy * firstVolume / volume + dispersionFixedEnergy;
  }
  double getDispersionVirial() const {
    return dispersionVirial * firstVolume / volume;
  }
  double getCoulombConstantEnergy() const {
    return coulombSelfEnergy +
           (coulombConstantEnergy - coulombSelfEnergy) * firstVolume / volume;
  }
  double getCoulombConstantVirial() const {
    return coulombConstantVirial * firstVolume / volume;
  }

  /// Whether the velocities are coupled, and the energy that the coupling
  /// has taken from the system so far, in kJ/mol. The total energy with it
  /// is conserved.
  bool couples = false;
  double bath = 0.0;
  /// A Nose-Hoover chain (D163a): the positions of its thermostats, then
  /// their velocities, their masses Q_j, k_B T of the bath, the degrees of
  /// freedom, and the time that one action of the chain spans, the period
  /// of the coupling, in the units of the run. Empty without one.
  std::vector<double> chain, chainMasses;
  double chainKT = 0.0, chainFreedom = 0.0, chainTime = 0.0;
  /// The equal parts that the action over `chainTime` is split into.
  int chainSubsteps = 1;
  /// The energy of the chain: Σ Q_j v_j² / 2 + N_f k_B T ξ_1 + k_B T Σ ξ_j.
  double getChainEnergy() const;
  /// Whether a barostat changes the cell, which the log then shows.
  bool changesCell = false;
  /// Brownian dynamics (D163b): the velocities are displacements over a
  /// step, so the log has no kinetic energy, temperature, or total, and the
  /// pressure takes the kinetic energy of the bath, `bathKinetic`,
  /// N_f k_B T / 2, that of the momenta that the overdamped limit leaves
  /// Maxwellian.
  bool overdamped = false;
  double bathKinetic = 0.0;
  /// Whether the run minimizes the energy, whose log has the forces in
  /// place of the kinetic energy (mdrtWriteMinimization).
  bool minimizes = false;
  /// The largest force at which a minimization ends, in kJ/mol/nm (0:
  /// none), and the step of the first row whose largest force is below it
  /// (-1: none yet) (D219).
  double minimizeTolerance = 0.0;
  int64_t convergedStep = -1;
  /// The shortest edge of the cell that the cutoff allows, twice it; a run
  /// whose barostat takes the cell below it stops.
  double leastEdge = 0.0;
  /// The shortest edge of the cell that the reach of the groups allows, the
  /// pairlist distance over 0.999 (0: no such bound, the neighbor matrix):
  /// their lists hold every image within the reach only in a cell wider
  /// than the reach (D[group-images]); a run whose barostat takes the cell
  /// below it stops.
  double leastReachEdge = 0.0;

  /// The energies at the first and at the last output, in kJ/mol. With
  /// coupling, those that are conserved.
  bool hasEnergies = false;
  double firstTotal = 0.0;
  double lastTotal = 0.0;

  /// The steps of the outputs of the energies and when each was written,
  /// in seconds of a steady clock: the rate of the run past its start
  /// (the set-up and the first build), as GROMACS reports it with
  /// `-resethway`.
  std::vector<std::pair<int64_t, double>> energyTimes;

  /// Where checkpoints go, and what they hold beside the state.
  std::string checkpointPath;
  Checkpoint checkpoint;
  int64_t numCheckpoints = 0;

  /// Takes the state when the run ends.
  System *system = nullptr;
  /// The last row of energies, in kJ/mol, K, bar, and nm^3 (the total
  /// without the bath, `conserved` with it), for a program
  /// that embeds the run (D196); `step` is -1 before one.
  struct EnergyRow {
    int64_t step = -1;
    double potential = 0.0, kinetic = 0.0, total = 0.0, temperature = 0.0,
           virial = 0.0, pressure = 0.0, conserved = 0.0, volume = 0.0;
  } lastEnergies;
  /// If set, a failure of the run is reported to it, and the run goes on to
  /// the end of its segment, rather than the process exiting: a simulation
  /// embedded in another program (D196).
  std::function<void(const std::string &)> fail;

  double getTime(int64_t step) const {
    return firstTime + static_cast<double>(step - firstStep) * timestep;
  }
};

/// Sets the output that the functions below write to.
void setOutput(Output *output);

/// Without a periodic cell (D142), fails the run of `output` if the
/// positions `x` of the step `step`, in the order of the input, have spread
/// so far that images of the particles could interact; as a program does at
/// its end (mdrtFinish).
void checkParticleSpread(const Output &output, const std::vector<double> &x,
                         int64_t step);

/// The signal that has asked the run to stop, or 0. A handler of SIGTERM
/// and SIGINT sets it; the run stops at its next checkpoint (D131).
extern volatile std::sig_atomic_t stopSignal;

/// The exit status of a run that stopped at a checkpoint before its last
/// step on a signal or at the wall time: EX_TEMPFAIL of sysexits.h, a
/// failure that a later attempt may get past (D131).
constexpr int StoppedStatus = 75;

void writeLogHeader(Output &output);

/// Writes a frame of the state to the trajectory of `output`, whose writer
/// is exact (H5MDWriter): three numbers per particle in the order of the
/// input, in nm, nm/ps, and kJ/mol/nm, with the potential energy of the
/// step if the last row of energies is of it, and the version of the
/// tunables. A failure of the write fails the run.
void writeExactFrame(Output &output, const double *positions,
                     const double *velocities, const double *forces,
                     int64_t step, double time);

/// The columns of the file of the energies: those of the rows of the log.
std::vector<ColumnFile::Column> getEnergyColumns(const Output &output);

/// The most backups that a run keeps of one output (D149).
constexpr int MaxBackups = 99;

/// The name under which a run that is not continued keeps the output
/// `path` of an earlier run before it writes its own (D149): `#<name>.<n>#`
/// in the directory of `path`, n the least number from 1 that no file
/// takes. Empty if all MaxBackups are taken. The naming and the limit of
/// 99 are those of the backups of GROMACS [GromacsManual2025].
std::string getBackupPath(llvm::StringRef path);

/// Fails if `path` exists and all MaxBackups of it are taken.
llvm::Error checkBackup(const std::string &path);

/// Renames `path` to getBackupPath(path) if it exists. Returns the new
/// name, empty if there was no file, or an error if MaxBackups are taken
/// or the rename fails.
llvm::Expected<std::string> backUpOutput(const std::string &path);

/// The name of the part `part` of the output `path`: `run.dcd` becomes
/// `run.part0002.dcd` (D130, D149).
std::string getPartPath(llvm::StringRef path, int64_t part);

} // namespace driver
} // namespace mdir

/// The functions that compiled code calls. Buffers arrive as pointers to
/// descriptors, as the C interface of MLIR passes them.
///
/// The particles of a buffer are in the order that the run keeps them in.
/// `ids` holds the number of each: the place of the particle in the files
/// of the run. What is written is in the order of these numbers.
extern "C" {
/// `kinetic` is the kinetic energy of the velocities of the step, and
/// `excess` the mean of those of the half steps before and after it less
/// `kinetic`, K_half - K (D203).
void _mlir_ciface_mdrtWriteEnergies(int64_t step, double potential,
                                    double kinetic, double excess,
                                    double virial);
void _mlir_ciface_mdrtWriteFrame(int64_t step, void *positions, void *ids);
/// Without a periodic cell (D142), stops the run if the positions at the
/// step `step` have spread so far that images interact.
void _mlir_ciface_mdrtCheckSpread(int64_t step, void *positions, void *ids);
/// The energy of each term of a topology at the start, in kJ/mol:
/// Lennard-Jones, Coulomb, bonds, angles, dihedrals, the pairs three bonds
/// apart, Lennard-Jones and Coulomb, and CMAP.
void _mlir_ciface_mdrtWriteTerms(void *terms);
/// The diagonal of the virial of the forces at the start, without those of
/// the constraints, in kJ/mol; the constant terms are added here.
void _mlir_ciface_mdrtWriteVirial(double xx, double yy, double zz);
/// The energy that a coupling of the velocities has just taken from the
/// system, in kJ/mol.
void _mlir_ciface_mdrtAddBath(double energy);
/// The factor that a Nose-Hoover chain scales the velocities by over the
/// period of coupling that ends at step `step`, from their kinetic energy
/// `kinetic` in kJ/mol (D163a). The chain moves on; the change of its
/// energy goes to the bath. A chain that its factorization no longer
/// follows stops the run (D206).
double mdrtNoseHooverFactor(int64_t step, double kinetic);
/// The kinetic energies of the rigid waters at the step of the next row of
/// the log, of the velocities of the step and K_half, in kJ/mol
/// (D203).
void mdrtWriteSolvent(double kinetic, double half);
/// The coordinates of the terms over centers of groups at the step `step`
/// (D145), in Å and radians, and the energy and the forces of each term,
/// in kcal/mol and per Å or radian.
void _mlir_ciface_mdrtWriteFreeEnergy(int64_t step, void *values);
/// The columns of `[output] observe` at the step `step` (D189), in kJ/mol
/// and kJ/mol per unit of a constant.
void _mlir_ciface_mdrtWriteObservables(int64_t step, void *values);
/// Takes the derivatives of the energy in the sites of the tunables, in
/// the order of Program::gradientSlots (D230).
void _mlir_ciface_mdrtWriteTunableGradient(void *values);
/// Takes the field `index` of Program::gradientFields, a value of f64 for
/// each particle, whose numbers `ids` holds.
void _mlir_ciface_mdrtWriteTunableGradientField(int64_t index, void *values,
                                                void *ids);
void _mlir_ciface_mdrtWritePull(int64_t step, void *coordinates,
                                void *terms);
/// A step of a minimization: the potential energy and the length of the
/// next step, in kJ/mol and nm, and the forces, of which the log has the
/// root mean square over the particles with mass and the largest.
void _mlir_ciface_mdrtWriteMinimization(int64_t step, double energy,
                                        double size, void *forces,
                                        void *ids);
/// Whether the minimization has converged (D219): sets
/// `state[0]` to 1 and `state[1]` to the step of the last row if the
/// largest force of that row is below the tolerance. `state` is a buffer of
/// two i64 on the host.
void _mlir_ciface_mdrtCheckMinimization(void *state);
/// The cell after a barostat has changed it: its edges in nm.
void _mlir_ciface_mdrtSetBox(double lx, double ly, double lz);
/// The tilts b_x, c_x, c_y of a triclinic cell that a barostat has scaled,
/// after mdrtSetBox (docs/triclinic-m2.md).
void _mlir_ciface_mdrtSetTilt(double bx, double cx, double cy);
/// The state that the next scaling of a barostat that scales the cell every
/// step takes its pressure from (D92), for the checkpoints.
/// The state of the last scaling of a barostat that scales every step
/// (D92, D119): the diagonals of the virial and of the virial of the rigid
/// groups, and the kinetic energy of each axis without the center of mass.
void _mlir_ciface_mdrtSetBarostatState(double w0, double w1, double w2,
                                       double g0, double g1, double g2,
                                       double k0, double k1, double k2);
void _mlir_ciface_mdrtWriteCheckpoint(int64_t step, void *positions,
                                      void *velocities, void *ids);
void _mlir_ciface_mdrtWriteCheckpointWithForces(int64_t step,
                                                void *positions,
                                                void *velocities,
                                                void *forces, void *ids);
void _mlir_ciface_mdrtFinish(void *positions, void *velocities, void *ids);
}

#endif // MDIR_DRIVER_OUTPUT_H

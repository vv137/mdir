// Primary, excluded, reciprocal, and within-selection nonbonded terms (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

namespace mdir::driver::builder {
class NonbondedWork {
public:
  NonbondedWork(TermParameters &context, TermNonbondedData data,
                llvm::ArrayRef<const TermBuilder *> builders = {})
      : context(context), program(data), control(context.getControl()),
        system(context.getSystem()), dependent(data.dependent),
        builders(builders) {}
  TermParameters &context;
  TermNonbondedData program;
  const Control &control;
  const System &system;
  const DependentTerms *dependent;
  llvm::ArrayRef<const TermBuilder *> builders;
  auto getTermBuilders() const { return builders; }
  bool decouples() const { return !system.alchemical.empty(); }
  bool isTriclinic() const {
    return system.tilt[0] != 0.0 || system.tilt[1] != 0.0 ||
           system.tilt[2] != 0.0;
  }
  double getLambda(StringRef name) const {
    return control.freeEnergy.get(name, control.freeEnergy.state);
  }
  double getShiftFixedFactor() const { return context.getShiftFixedFactor(); }
  double getPairTails(const llvm::StringMap<double> &changes,
                      bool fixed = false) const {
    return getCustomPairTails(context, changes, fixed);
  }
  double getPairTailDerivative(unsigned index, StringRef name, double value,
                               bool shift = true, bool fixed = false) const {
    return getCustomPairTailDerivative(context, index, name, value, shift,
                                       fixed);
  }
  llvm::StringMap<double> getLambdaValues(size_t k) const {
    llvm::StringMap<double> values;
    for (const auto &[name, list] : control.freeEnergy.lambdas)
      values["lambda_" + name] = control.freeEnergy.get(name, k);
    return values;
  }
  llvm::Error collectTopologyConstants();
  double getTopologyDispersion(bool) const;
  void collectPrimaryDispersionConstants();
  void collectPrimaryGradientShiftConstants();
  void collectObservedTails();
  std::pair<double, double> getPMEConstants(double) const;
  void collectFreeEnergyConstants();
  llvm::Error collectPME();
  llvm::Error collectProlate();
  void collectListed(StringRef, llvm::ArrayRef<double>,
                     llvm::ArrayRef<Control::TunableKind>);
  llvm::Error collectMesh(StringRef, const int64_t (&)[3], double, int64_t,
                          PMEInfluence, int64_t (&)[3]);
  llvm::Error collectLJPME();
  llvm::Error collectTunableGradient();
  llvm::Error prepare() {
    if (control.pme)
      if (auto error = collectPME())
        return error;
    if (control.ljpme)
      if (auto error = collectLJPME())
        return error;
    return collectTopologyConstants();
  }
};
} // namespace mdir::driver::builder
static void fillCharges(const TermInputs &context, Program::Field &field) {
  field.values = context.getTopology().charges;
}

static void fillTypeTable(const TermInputs &context, Program::Table &table) {
  const Topology &topology = context.getTopology();
  const auto &system = context.getSystem();
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

/// Whether `name` is the table of the weights of a seed, `tgw<c>`.
static bool isSeedTable(StringRef name) {
  unsigned seed;
  return name.consume_front("tgw") && !name.getAsInteger(10, seed);
}

double
mdir::driver::builder::getTopologyShiftFactor(const TermInputs &context) {
  const auto &control = context.getControl();
  const auto &system = context.getSystem();
  // 1 − 1 / (ρ (4π/3) r_c³): the neighbors within r_c less the particle
  // itself, over ρ (4π/3) r_c³ (D210).
  double rc = control.cutoffDistance * units::length;
  double volume = system.box[0] * system.box[1] * system.box[2];
  double n = static_cast<double>(system.topology->getNumParticles());
  return 1.0 - volume / (n * 4.0 * M_PI / 3.0 * rc * rc * rc);
}

/// A part of the splitting of the prolate kernel at `x` = r / r_c, from the
/// table `table` (prolate::getSplittingTable): the polynomial of the cell
/// of x in t = 2 (x cells − k) − 1, the cell clamped to the table. The
/// place in the cell carries the derivative; the index and the lookups do
/// not, as in a tabulated function of the control file (D138).
static mlir::Value buildProlateSplitting(mlir::OpBuilder &builder,
                                         mlir::Value table, mlir::Value x,
                                         int64_t prolateCells,
                                         int64_t prolateTerms) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  auto constant = [&](double value) -> Value {
    return arith::ConstantOp::create(builder, loc,
                                     builder.getF64FloatAttr(value));
  };
  Value cells = constant(static_cast<double>(prolateCells));
  Value top = constant(static_cast<double>(prolateCells - 1));
  Value nil = constant(0.0);
  Value one = constant(1.0);
  Value two = constant(2.0);
  Value u = arith::MulFOp::create(builder, loc, x, cells);
  Value floor = math::FloorOp::create(builder, loc, u);
  Value low = arith::MaximumFOp::create(builder, loc, floor, nil);
  Value place = arith::MinimumFOp::create(builder, loc, low, top);
  Value s = arith::SubFOp::create(builder, loc, u, place);
  Value s2 = arith::MulFOp::create(builder, loc, two, s);
  Value t = arith::SubFOp::create(builder, loc, s2, one);
  Value row =
      arith::FPToSIOp::create(builder, loc, builder.getI32Type(), place);
  llvm::SmallVector<Value> coefficients;
  for (int64_t j = 0; j != prolateTerms; ++j) {
    Value column =
        arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(j));
    coefficients.push_back(mdir::md::LookupOp::create(
        builder, loc, builder.getF64Type(), table, ValueRange{row, column}));
  }
  Value value = coefficients.back();
  for (int64_t j = prolateTerms - 2; j >= 0; --j) {
    Value product = arith::MulFOp::create(builder, loc, value, t);
    value = arith::AddFOp::create(builder, loc, product, coefficients[j]);
  }
  return value;
}

/// The reciprocal share of 1/r at r: erf(βr)/r, or the prolate
/// Φ(r/r_c)/r = (Φ(x)/x)/r_c within the cutoff and 1/r beyond it (D266).
static mlir::Value buildShieldedCoulomb(mlir::OpBuilder &builder,
                                        mlir::Value source, mlir::Value r,
                                        bool prolate, double cutoff,
                                        int64_t cells, int64_t terms) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  if (!prolate) {
    Value br = arith::MulFOp::create(builder, loc, source, r);
    Value erf = math::ErfOp::create(builder, loc, br);
    return arith::DivFOp::create(builder, loc, erf, r);
  }
  Value inverse = arith::ConstantOp::create(
      builder, loc, builder.getF64FloatAttr(1.0 / cutoff));
  Value xr = arith::MulFOp::create(builder, loc, r, inverse);
  Value polynomial = buildProlateSplitting(builder, source, xr, cells, terms);
  Value near = arith::MulFOp::create(builder, loc, polynomial, inverse);
  Value one =
      arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0));
  Value rInverse = arith::DivFOp::create(builder, loc, one, r);
  Value inside =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT, xr, one);
  return arith::SelectOp::create(builder, loc, inside, near, rInverse);
}

llvm::Error NonbondedWork::collectTopologyConstants() {
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
    for (const TermBuilder *builder :
         termBuildersByParameterRank(getTermBuilders()))
      if (llvm::Error error = builder->collectDispersionConstants(context))
        return error;
    collectObservedTails();
  }
  if (control.hasFreeEnergy)
    collectFreeEnergyConstants();
  return llvm::Error::success();
}

void NonbondedWork::collectPrimaryDispersionConstants() {
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
}
void NonbondedWork::collectPrimaryGradientShiftConstants() {
  double tail = getTopologyDispersion(false);
  program.gradientShiftEnergy = tail;
  program.gradientShiftFixedEnergy = tail * getShiftFixedFactor();
}

double NonbondedWork::getTopologyDispersion(bool decoupled) const {
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

void NonbondedWork::collectObservedTails() {
  // The columns of `observe` of a pair term take its tail and the estimate
  // of its shift: its energy and its derivatives in the observed constants,
  // proportional to 1 / V but the part of the estimate that does not depend
  // on the volume (D210, #224).
  program.observableVolumeConstants.assign(control.observables.size(), 0.0);
  program.observableFixedConstants.assign(control.observables.size(), 0.0);
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().scalarParameters)
      role->collectObservedConstants(context);
}

std::pair<double, double> NonbondedWork::getPMEConstants(double lambda) const {
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

void NonbondedWork::collectFreeEnergyConstants() {
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
    for (unsigned index = 0, e = system.pairTails.size(); index != e; ++index) {
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

llvm::Error NonbondedWork::collectPME() {
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
    if (llvm::Error error =
            collectMesh("pme_moduli", control.pmeGrid, control.pmeMaxSpacing,
                        control.pmeOrder, control.pmeInfluence, grid))
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

llvm::Error NonbondedWork::collectProlate() {
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
      edge = k == 0   ? system.box[0]
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
  int64_t columns =
      std::max({longest, 2 * points * window.degree, 4 + 2 * influence.terms});
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
  for (auto [name, part] : {std::pair{"pme_far", prolate::Part::Far},
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

void NonbondedWork::collectListed(StringRef name, llvm::ArrayRef<double> values,
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

llvm::Error NonbondedWork::collectMesh(StringRef name,
                                       const int64_t (&given)[3],
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
      edge = k == 0   ? system.box[0]
             : k == 1 ? std::hypot(system.tilt[0], system.box[1])
                      : std::sqrt(system.tilt[1] * system.tilt[1] +
                                  system.tilt[2] * system.tilt[2] +
                                  system.box[2] * system.box[2]);
    if (grid[k] == 0)
      grid[k] = getSmoothSize(static_cast<int64_t>(
          std::ceil(edge / (spacing * units::length) - 1e-9)));
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
    std::copy(moduli.begin(), moduli.end(), table.values.begin() + k * longest);
  }
  program.tables.push_back(std::move(table));
  return llvm::Error::success();
}

llvm::Error NonbondedWork::collectLJPME() {
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
  if (llvm::Error error = collectMesh(
          "ljpme_moduli", control.ljpmeGrid, control.ljpmeMaxSpacing,
          control.ljpmeOrder, PMEInfluence::SPME, grid))
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
    coefficients[a] =
        2.0 * std::sqrt(std::max(epsilon, 0.0)) * sigma * sigma * sigma;
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

llvm::Error NonbondedWork::collectTunableGradient() {
  // Each tunable has a rule for its derivative, or the program provably
  // does not read it, or the compile fails (D230): nothing
  // is left out without a word.
  bool dispersion = control.topologyDispersion != DispersionCorrection::None;
  // The estimate of what the shift takes within the cutoff, which the
  // energy of the run holds only under the shift (D210).
  if (dispersion && control.truncation != Truncation::Shift) {
    for (const TermBuilder *builder :
         termBuildersByParameterRank(getTermBuilders()))
      builder->collectGradientShiftConstants(context);
  }
  for (auto [k, declaration] : llvm::enumerate(control.tunableDeclarations)) {
    Program::GradientOutcome outcome = Program::GradientOutcome::Zero;
    bool registered = false;
    for (const TermBuilder *builder : getTermBuilders()) {
      auto result =
          builder->collectTunableOutcome(context, static_cast<unsigned>(k));
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
                           "energy in the tunable '" +
                           declaration.name + "' (charges) " + what +
                           " is not implemented yet");
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
            program.gradientChargeVolume[i] =
                -2.0 * coulombInternal * M_PI * net *
                program.pmeBackgroundFactor / volume;
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
          for (const char *name :
               {"sigma", "epsilon", "sigma1", "sigma2", "epsilon1", "epsilon2"})
            if (llvm::is_contained(expression.getNames(), name))
              return makeError(
                  "System.tunable_gradient: the derivative of the energy in "
                  "the tunable '" +
                  declaration.name +
                  "' through the pair "
                  "term '" +
                  term.name + "', which reads '" + name +
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
    double factor = ordered > 0.0 ? -2.0 * M_PI / (3.0 * volume) * n * n /
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

namespace mdir::driver::builder {
class NonbondedTermBuilder final : public TermBuilder, public TermNonbonded {
public:
  NonbondedTermBuilder() {
    roles.nonbonded = this;
    tupleRank = 3;
    pairExclusion = "excluded";
  }
  bool rejectsTunables(const TermInputs &context) const override {
    return context.getControl().ljpme;
  }
  int parameterRank() const override { return -4; }
  int topologyRank() const override { return 1; }
  bool
  needsPairNeighborhood(const TermInputs &,
                        const PotentialSelection &selection) const override {
    return selection.terms & (TermInputs::LennardJones | TermInputs::Coulomb);
  }
  void collectParameters(TermParameters &context) const override {
    const auto &topology = context.getTopology();
    Program::Field charges;
    charges.name = "q";
    charges.values = topology.charges;
    context.getFields().push_back(std::move(charges));
    Program::Field types;
    types.name = "type";
    types.isInteger = true;
    for (unsigned type : topology.types)
      types.values.push_back(type);
    context.getFields().push_back(std::move(types));
  }
  void collectTopologyTables(TermParameters &context) const override {
    const auto &topology = context.getTopology();
    const auto &control = context.getControl();
    const auto &system = context.getSystem();
    auto decouples = [&] { return !system.alchemical.empty(); };

    // [free_energy] (D161): 1 for each particle that it decouples.
    if (decouples()) {
      Program::Field flags;
      flags.name = "alch";
      for (bool flag : system.alchemical)
        flags.values.push_back(flag ? 1.0 : 0.0);
      context.getFields().push_back(std::move(flags));
    }

    // Lennard-Jones for each pair of types.
    unsigned numTypes = topology.getNumTypes();
    for (const char *name : {"lj_sigma", "lj_epsilon"}) {
      Program::Table table;
      table.name = name;
      table.count = numTypes;
      fillTypeTable(context, table);
      context.getTables().push_back(std::move(table));
    }
    // The weights of the seeds of the derivatives in those tables, a table
    // of all ordered pairs of types each, and the field of the particles
    // that each gives (D230).
    if (context.hasTunableGradient())
      for (auto [c, seed] : llvm::enumerate(control.tunableTableSeeds)) {
        Program::Table table;
        table.name = "tgw" + std::to_string(c);
        table.count = numTypes;
        table.columns = numTypes;
        fillTypeTable(context, table);
        context.getTables().push_back(std::move(table));
        Program::GradientField field;
        field.tunable = seed.tunable;
        for (unsigned a = 0; a != numTypes; ++a)
          field.rows.push_back({seed.sites[a], seed.scales[a], 0.0});
        context.getGradientFields().push_back(std::move(field));
      }
  }
  void collectTupleSets(TermParameters &context) const override {
    const auto &topology = context.getTopology();
    const auto &control = context.getControl();
    const auto &system = context.getSystem();
    if (!topology.exclusions.empty()) {
      Program::TupleSet &set =
          addTupleSet(context.getTupleSets(), "excluded", 2);
      for (auto [i, j] : topology.exclusions) {
        set.members.push_back(i);
        set.members.push_back(j);
      }
    }
    // The pairs within the selection of [free_energy] that are not excluded,
    // whose Coulomb particle mesh Ewald adds back (D161).
    if (control.pme && !system.alchemicalPairs.empty()) {
      Program::TupleSet &set =
          addTupleSet(context.getTupleSets(), "alchemical_pairs", 2);
      for (auto [i, j] : system.alchemicalPairs) {
        set.members.push_back(i);
        set.members.push_back(j);
      }
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    const TermDefinition data[] = {
        {0, 0, "term_lj", TermInputs::LennardJones},
        {1, 0, "term_coulomb", TermInputs::Coulomb},
        {8, 0, "term_excluded",
         TermInputs::CoulombExcluded | TermInputs::CoulombWithin},
        {9, 0, "term_reciprocal", TermInputs::CoulombReciprocal},
        {12, 0, "term_lj_excluded", TermInputs::LennardJonesExcluded},
        {13, 0, "term_lj_reciprocal", TermInputs::LennardJonesReciprocal},
    };
    definitions.insert(definitions.end(), std::begin(data), std::end(data));
  }
  void refillArrays(TermParameters &context) const override {
    for (auto &field : context.getFields())
      if (field.values.empty() && field.name == "q")
        fillCharges(context, field);
    for (auto &table : context.getTables())
      if (table.values.empty() &&
          (table.name == "lj_sigma" || table.name == "lj_epsilon" ||
           isSeedTable(table.name)))
        fillTypeTable(context, table);
  }
  llvm::Error
  collect(TermParameters &context, TermNonbondedData data,
          llvm::ArrayRef<const TermBuilder *> builders) const override {
    state.emplace(data);
    return NonbondedWork(context, data, builders).prepare();
  }
  void collectGradientFields(TermParameters &context,
                             TermNonbondedData data) const override {
    const auto &control = context.getControl();
    const auto &topology = context.getTopology();
    // The derivative in the charges is that of `@tunable` in its field of
    // them, a site for each particle (D230).
    if (context.hasTunableGradient())
      for (auto [k, declaration] : llvm::enumerate(control.tunableDeclarations))
        if (declaration.kind == Control::TunableKind::Charge) {
          Program::GradientField field;
          field.tunable = static_cast<unsigned>(k);
          for (uint32_t i = 0, e = topology.getNumParticles(); i != e; ++i)
            field.sites.push_back({i, i});
          context.getGradientFields().push_back(std::move(field));
          data.gradientCharges = true;
        }
  }
  llvm::Error collectGradients(
      TermParameters &context, TermNonbondedData data,
      llvm::ArrayRef<const TermBuilder *> builders) const override {
    state.emplace(data);
    return NonbondedWork(context, data, builders).collectTunableGradient();
  }
  llvm::Error
  collectDispersionConstants(TermParameters &context) const override {
    assert(state &&
           "nonbonded constants require their bounded data projection");
    NonbondedWork(context, *state).collectPrimaryDispersionConstants();
    return llvm::Error::success();
  }
  void collectGradientShiftConstants(TermParameters &context) const override {
    assert(state &&
           "nonbonded gradients require their bounded data projection");
    NonbondedWork(context, *state).collectPrimaryGradientShiftConstants();
  }
  unsigned getPotentialMask(const TermInputs &context,
                            const DependentTerms *dependent,
                            bool alchemicalOnly) const override {
    unsigned mask = TermInputs::LennardJones | TermInputs::Coulomb |
                    TermInputs::LennardJonesExcluded |
                    TermInputs::LennardJonesReciprocal |
                    TermInputs::CoulombExcluded |
                    TermInputs::CoulombReciprocal | TermInputs::CoulombWithin;
    if (dependent) {
      unsigned kept = 0;
      if (dependent->lennardJones)
        kept |= TermInputs::LennardJones | TermInputs::LennardJonesExcluded |
                TermInputs::LennardJonesReciprocal;
      if (dependent->coulomb)
        kept |= TermInputs::Coulomb | TermInputs::CoulombExcluded |
                TermInputs::CoulombReciprocal | TermInputs::CoulombWithin;
      mask &= kept;
    }
    if (alchemicalOnly) {
      unsigned kept = TermInputs::LennardJones | TermInputs::Coulomb |
                      TermInputs::CoulombExcluded | TermInputs::CoulombWithin |
                      TermInputs::CoulombReciprocal;
      mask &= context.getSystem().alchemical.empty() ? 0 : kept;
    }
    return mask;
  }

  void emit(TermEmission &context, TermNonbondedData program,
            const PotentialSelection &selection,
            const TermPotentialAddition &add) const override {
    auto &os = context.getOutput();
    const auto &control = context.getControl();
    const auto &system = context.getSystem();
    unsigned terms = selection.terms;
    bool alchemicalOnly = selection.alchemicalOnly;
    bool gradientArguments = context.hasGradientArguments();
    bool lambdaArguments = program.lambdaArguments;
    bool shiftsAtCutoff = program.shiftsAtCutoff;
    double cutoff = control.cutoffDistance * units::length;
    auto decouples = [&] { return !system.alchemical.empty(); };
    auto getLambda = [&](StringRef name) {
      return control.freeEnergy.get(name, control.freeEnergy.state);
    };
    Truncation truncation = context.getPairTruncation();
    auto has = [&](StringRef name) {
      return llvm::any_of(context.getTupleSets(),
                          [&](const auto &set) { return set.name == name; });
    };
    auto hasField = [&](StringRef name) {
      return llvm::any_of(context.getFields(), [&](const auto &field) {
        return field.name == name;
      });
    };
    auto gathersAll = [&](StringRef name) {
      for (const auto &field : context.getFields())
        if (field.name == name)
          return 2 * static_cast<size_t>(
                         llvm::count_if(field.values.get(),
                                        [](double v) { return v >= 0.0; })) >
                 field.values.size();
      return false;
    };
    auto emitLennardJones = [&](StringRef sigma, StringRef epsilon,
                                StringRef result, Truncation modifier,
                                StringRef distance = "%r") {
      mdir::driver::builder::emitLennardJones(context, control, sigma, epsilon,
                                              result, modifier, distance);
    };
    // Which λ the kernels of the pairs take: as arguments, or as constants
    // of the state of the run that are not 0.
    bool scalesCoulomb =
        decouples() && (lambdaArguments || getLambda("coulomb") != 0.0);
    bool scalesVdw =
        decouples() && (lambdaArguments || getLambda("vdw") != 0.0);
    bool lj = terms & TermInputs::LennardJones,
         coulomb = terms & TermInputs::Coulomb;
    if (lj || coulomb) {
      // [free_energy] (D161): a pair with one particle in the selection,
      // `%al_x` = 1, is decoupled. Its Lennard-Jones is that of the
      // soft-core distance r_A, r_A^6 = alpha sigma^6 lambda^p + r^6, times
      // 1 - lambda [Beutler1994], and its Coulomb is times 1 - lambda: the
      // charge of the particle in the selection scaled. The pairs within the
      // selection stay.
      bool flags = (lj && scalesVdw) || (coulomb && scalesCoulomb) ||
                   (alchemicalOnly && decouples());
      // In `@tunable`, σ and ε of a pair take the seeds of the derivatives
      // in their tables (D230, Control::TunableTableSeed).
      size_t seeds =
          gradientArguments && lj ? control.tunableTableSeeds.size() : 0;
      std::string seedFields, seedTypes, seedArguments;
      for (size_t c = 0; c != seeds; ++c) {
        seedFields += ", %tg_d" + std::to_string(c);
        seedTypes += ", !real";
        seedArguments += ", %tgd" + std::to_string(c) + "_i: f64, %tgd" +
                         std::to_string(c) + "_j: f64";
      }
      os << "  %u_nonbonded = md.sum_relation %n, %x, %cell gather(%p_type, "
            "%p_q"
         << (flags ? ", %p_alch" : "") << seedFields << " : !ids, !real"
         << (flags ? ", !real" : "") << seedTypes << ")\n"
         << "      exchange(symmetric" << (seeds ? ", asserted" : "") << ") {\n"
         << "  ^bb0(%r: f64, %d: vector<3xf64>, %type_i: i32, %type_j: i32, "
            "%q_i: f64, %q_j: f64"
         << (flags ? ", %al_i: f64, %al_j: f64" : "") << seedArguments
         << "):\n";
      if (flags)
        os << "    %al_ij = arith.mulf %al_i, %al_j : f64\n"
           << "    %al_sum = arith.addf %al_i, %al_j : f64\n"
           << "    %al_twice = arith.addf %al_ij, %al_ij : f64\n"
           << "    %cross = arith.subf %al_sum, %al_twice : f64\n"
           << "    %al_one = arith.constant 1.0 : f64\n";
      std::string value;
      if (lj) {
        bool seedsSigma = false, seedsEpsilon = false;
        for (size_t c = 0; c != seeds; ++c)
          (control.tunableTableSeeds[c].sigma ? seedsSigma : seedsEpsilon) =
              true;
        os << "    %sigma" << (seedsSigma ? "_table" : "")
           << " = md.lookup %t_lj_sigma[%type_i, %type_j] : !table, "
              "i32, i32 -> f64\n"
           << "    %epsilon" << (seedsEpsilon ? "_table" : "")
           << " = md.lookup %t_lj_epsilon[%type_i, %type_j] : "
              "!table, i32, i32 -> f64\n";
        // The value of the table plus d_i w[a, b] + d_j w[b, a] for each
        // seed: the value itself, d being zero.
        for (bool sigma : {true, false}) {
          std::string value = sigma ? "%sigma_table" : "%epsilon_table";
          size_t left = 0;
          for (size_t c = 0; c != seeds; ++c)
            left += control.tunableTableSeeds[c].sigma == sigma;
          for (size_t c = 0; c != seeds; ++c) {
            if (control.tunableTableSeeds[c].sigma != sigma)
              continue;
            std::string n = std::to_string(c);
            std::string next = --left
                                   ? "%tgs" + n
                                   : std::string(sigma ? "%sigma" : "%epsilon");
            os << "    %tgw" << n << "_ij = md.lookup %t_tgw" << n
               << "[%type_i, %type_j] : !grid, i32, i32 -> f64\n"
               << "    %tgw" << n << "_ji = md.lookup %t_tgw" << n
               << "[%type_j, %type_i] : !grid, i32, i32 -> f64\n"
               << "    %tgs" << n << "_i = arith.mulf %tgd" << n << "_i, %tgw"
               << n << "_ij : f64\n"
               << "    %tgs" << n << "_j = arith.mulf %tgd" << n << "_j, %tgw"
               << n << "_ji : f64\n"
               << "    %tgs" << n << "_ij = arith.addf %tgs" << n << "_i, %tgs"
               << n << "_j : f64\n"
               << "    " << next << " = arith.addf " << value << ", %tgs" << n
               << "_ij : f64\n";
            value = next;
          }
        }
        if (scalesVdw) {
          // The distance r_A of a decoupled pair, r of the others. A pair
          // without a σ, which has no Lennard-Jones, takes 0.3 nm in r_A,
          // so that r_A stays above 0 where its particles overlap.
          double alpha = control.freeEnergy.softCoreAlpha;
          std::string distance = "%r";
          if (alpha > 0.0) {
            bool root = truncation != Truncation::None &&
                        truncation != Truncation::Shift;
            llvm::SmallVector<StringRef> results =
                root ? llvm::SmallVector<StringRef>{"%sc_r"}
                     : llvm::SmallVector<StringRef>{"%sc_asl", "%sc_r6",
                                                    "%sc_i3"};
            llvm::SmallVector<StringRef> types =
                root ? llvm::SmallVector<StringRef>{"f64"}
                     : llvm::SmallVector<StringRef>{"f64", "f64", "i32"};
            context.emitIRFragment(
                "    ", results, types,
                {"%sigma", "%r", "%lambda_vdw", "%cross"},
                {"f64", "f64", "f64", "f64"},
                [alpha, power = control.freeEnergy.softCorePower,
                 root](mlir::OpBuilder &builder,
                       mlir::ValueRange values) -> TermIRValues {
                  using namespace mlir;
                  auto loc = builder.getUnknownLoc();
                  auto constant = [&](double value) -> Value {
                    return arith::ConstantOp::create(
                        builder, loc, builder.getF64FloatAttr(value));
                  };
                  auto mul = [&](Value a, Value b) -> Value {
                    return arith::MulFOp::create(builder, loc, a, b);
                  };
                  Value alphaValue = constant(alpha);
                  Value zero = constant(0.0);
                  Value floor = constant(0.3);
                  Value has = arith::CmpFOp::create(
                      builder, loc, arith::CmpFPredicate::OGT, values[0], zero);
                  Value sigma = arith::SelectOp::create(builder, loc, has,
                                                        values[0], floor);
                  Value i3 = arith::ConstantOp::create(
                      builder, loc, builder.getI32IntegerAttr(3));
                  Value s3 = math::FPowIOp::create(builder, loc, sigma, i3);
                  Value s6 = mul(s3, s3);
                  Value r3 = math::FPowIOp::create(builder, loc, values[1], i3);
                  Value r6 = mul(r3, r3);
                  Value lambda = values[2];
                  if (power == 2)
                    lambda = mul(lambda, lambda);
                  Value as = mul(alphaValue, s6);
                  Value asl = mul(as, lambda);
                  Value x = arith::AddFOp::create(builder, loc, r6, asl);
                  if (!root)
                    return {asl, r6, i3};
                  Value sixth = constant(1.0 / 6.0);
                  Value ra = math::PowFOp::create(builder, loc, x, sixth);
                  Value half = constant(0.5);
                  Value cross = arith::CmpFOp::create(
                      builder, loc, arith::CmpFPredicate::OGT, values[3], half);
                  return {arith::SelectOp::create(builder, loc, cross, ra,
                                                  values[1])};
                });
            if (root)
              distance = "%sc_r";
          }
          bool plain =
              truncation == Truncation::None || truncation == Truncation::Shift;
          if (alpha > 0.0 && plain) {
            // The unswitched soft-core potential needs no sixth root
            // [Beutler1994]; retain its original arithmetic sequence.
            context.emitIRFragment(
                "    ", {"%lj_full"}, {"f64"},
                {"%sigma", "%epsilon", "%cross", "%sc_asl", "%sc_r6", "%sc_i3"},
                {"f64", "f64", "f64", "f64", "f64", "i32"},
                [truncation, cutoff](mlir::OpBuilder &builder,
                                     mlir::ValueRange values) -> TermIRValues {
                  using namespace mlir;
                  auto loc = builder.getUnknownLoc();
                  auto constant = [&](double value) -> Value {
                    return arith::ConstantOp::create(
                        builder, loc, builder.getF64FloatAttr(value));
                  };
                  auto mul = [&](Value a, Value b) -> Value {
                    return arith::MulFOp::create(builder, loc, a, b);
                  };
                  auto sub = [&](Value a, Value b) -> Value {
                    return arith::SubFOp::create(builder, loc, a, b);
                  };
                  Value xs = mul(values[3], values[2]);
                  Value xx = arith::AddFOp::create(builder, loc, values[4], xs);
                  Value g3 =
                      math::FPowIOp::create(builder, loc, values[0], values[5]);
                  Value g6 = mul(g3, g3);
                  Value q6 = arith::DivFOp::create(builder, loc, g6, xx);
                  Value q12 = mul(q6, q6);
                  Value a = q12, b = q6;
                  if (truncation == Truncation::Shift) {
                    double cut6 = std::pow(cutoff, -6.0);
                    Value rc6 = constant(cut6);
                    Value rc12 = constant(cut6 * cut6);
                    Value g12 = mul(g6, g6);
                    Value c12 = mul(g12, rc12);
                    Value c6 = mul(g6, rc6);
                    a = sub(q12, c12);
                    b = sub(q6, c6);
                  }
                  Value c4 = constant(4.0);
                  Value e4 = mul(c4, values[1]);
                  Value t = sub(a, b);
                  return {mul(e4, t)};
                });
          } else {
            emitLennardJones("%sigma", "%epsilon", "%lj_full", truncation,
                             distance);
          }
          context.emitIRFragment(
              "    ", {"%lj"}, {"f64"},
              {"%cross", "%lambda_vdw", "%al_one", "%lj_full"},
              {"f64", "f64", "f64", "f64"},
              [](mlir::OpBuilder &builder,
                 mlir::ValueRange values) -> TermIRValues {
                auto loc = builder.getUnknownLoc();
                mlir::Value cl = mlir::arith::MulFOp::create(
                    builder, loc, values[0], values[1]);
                mlir::Value scale =
                    mlir::arith::SubFOp::create(builder, loc, values[2], cl);
                return {mlir::arith::MulFOp::create(builder, loc, scale,
                                                    values[3])};
              });
        } else {
          emitLennardJones("%sigma", "%epsilon", "%lj", truncation);
        }
        value = "%lj";
        if (program.ljpme) {
          // The dispersion that the grid sums within the cutoff,
          // −c_i c_j (1 − g(β r)) / r⁶, taken out again, so that the pair
          // has its own Lennard-Jones there [Essmann1995, Wennberg2013]
          // (D162); shifted with it to 0 at the cutoff if it is.
          double beta = program.ljpmeBeta;
          double shift = truncation == Truncation::Shift
                             ? (1.0 - getDispersionScreen(beta * cutoff)) /
                                   std::pow(cutoff, 6.0)
                             : 0.0;
          context.emitIRFragment(
              "    ", {"%ljp"}, {"f64"},
              {"%t_ljpme_c6", "%type_i", "%type_j", "%r", "%lj"},
              {"!table", "i32", "i32", "f64", "f64"},
              [beta, shift,
               truncation](mlir::OpBuilder &builder,
                           mlir::ValueRange operands) -> TermIRValues {
                using namespace mlir;
                auto loc = builder.getUnknownLoc();
                auto constant = [&](double value) -> Value {
                  return arith::ConstantOp::create(
                      builder, loc, builder.getF64FloatAttr(value));
                };
                auto mul = [&](Value a, Value b) -> Value {
                  return arith::MulFOp::create(builder, loc, a, b);
                };
                Value c6 = mdir::md::LookupOp::create(
                    builder, loc, builder.getF64Type(), operands[0],
                    ValueRange{operands[1], operands[2]});
                Value betaValue = constant(beta);
                Value r = operands[3];
                Value x = mul(betaValue, r);
                Value x2 = mul(x, x);
                Value nx2 = arith::NegFOp::create(builder, loc, x2);
                Value e = math::ExpOp::create(builder, loc, nx2);
                Value x4 = mul(x2, x2);
                Value half = constant(0.5);
                Value h4 = mul(half, x4);
                Value one = constant(1.0);
                Value p1 = arith::AddFOp::create(builder, loc, one, x2);
                Value polynomial = arith::AddFOp::create(builder, loc, p1, h4);
                Value g = mul(e, polynomial);
                Value kept = arith::SubFOp::create(builder, loc, one, g);
                Value r2 = mul(r, r);
                Value r4 = mul(r2, r2);
                Value r6 = mul(r4, r2);
                Value cr = arith::DivFOp::create(builder, loc, c6, r6);
                Value correction = mul(cr, kept);
                if (truncation == Truncation::Shift) {
                  Value shiftValue = constant(shift);
                  Value cs = mul(c6, shiftValue);
                  correction =
                      arith::SubFOp::create(builder, loc, correction, cs);
                }
                return {arith::AddFOp::create(builder, loc, operands[4],
                                              correction)};
              });
          value = "%ljp";
        }
      }
      if (coulomb) {
        // The clamped distance prevents a decoupled overlapping pair from
        // producing zero times infinity. Reaction-field r² keeps physical r.
        auto reaction = program.reactionField ? getReactionField(control)
                                              : std::pair<double, double>{};
        bool prolateSeries = takesProlateSeries(control) && !scalesCoulomb;
        auto [prolateCells, prolateTerms] =
            getProlateTableShape(control.precision);
        llvm::SmallVector<StringRef> operands = {"%q_i", "%q_j", "%r"};
        llvm::SmallVector<StringRef> operandTypes = {"f64", "f64", "f64"};
        if (scalesCoulomb) {
          operands.append({"%cross", "%al_one", "%lambda_coulomb"});
          operandTypes.append({"f64", "f64", "f64"});
        }
        if (program.pme && program.pmeProlate && !prolateSeries) {
          operands.push_back("%t_pme_far");
          operandTypes.push_back("!grid");
        }
        context.emitIRFragment(
            "    ", {"%coulomb"}, {"f64"}, operands, operandTypes,
            [scalesCoulomb, pme = program.pme, prolate = program.pmeProlate,
             prolateSeries, farSeries = program.pmeFarSeries,
             beta = program.pmeBeta, shift = control.pmeShift || shiftsAtCutoff,
             reactionField = program.reactionField, reaction, cutoff,
             shiftsAtCutoff, prolateCells,
             prolateTerms](mlir::OpBuilder &builder,
                           mlir::ValueRange values) -> TermIRValues {
              using namespace mlir;
              auto loc = builder.getUnknownLoc();
              auto constant = [&](double value) -> Value {
                return arith::ConstantOp::create(
                    builder, loc, builder.getF64FloatAttr(value));
              };
              auto mul = [&](Value a, Value b) -> Value {
                return arith::MulFOp::create(builder, loc, a, b);
              };
              auto sub = [&](Value a, Value b) -> Value {
                return arith::SubFOp::create(builder, loc, a, b);
              };
              auto div = [&](Value a, Value b) -> Value {
                return arith::DivFOp::create(builder, loc, a, b);
              };
              Value f = constant(coulombInternal);
              Value qq = mul(values[0], values[1]);
              if (scalesCoulomb) {
                Value cl = mul(values[3], values[5]);
                Value scale = sub(values[4], cl);
                qq = mul(qq, scale);
              }
              Value fqq = mul(f, qq);
              Value r = values[2], rq = r;
              if (scalesCoulomb) {
                Value least = constant(1.0e-4);
                rq = arith::MaximumFOp::create(builder, loc, r, least);
              }
              if (pme && prolate) {
                // Direct prolate kernel [Liang2026, eq. 3]. Its cutoff shift is
                // zero.
                Value inverse = constant(1.0 / cutoff);
                Value xr = mul(rq, inverse);
                Value far;
                if (prolateSeries) {
                  Value two = constant(2.0);
                  Value one = constant(1.0);
                  Value twice = mul(two, xr);
                  Value t = sub(twice, one);
                  Value mean = mdir::md::ChebyshevOp::create(
                      builder, loc, t, builder.getF64ArrayAttr(farSeries));
                  Value rest = sub(one, xr);
                  far = mul(rest, mean);
                } else {
                  far = buildProlateSplitting(builder, values.back(), xr,
                                              prolateCells, prolateTerms);
                }
                Value screened = div(far, rq);
                return {mul(fqq, screened)};
              }
              if (pme) {
                // Gaussian Ewald direct kernel, optionally shifted at the
                // cutoff.
                Value betaValue = constant(beta);
                Value br = mul(betaValue, rq);
                Value erfc = math::ErfcOp::create(builder, loc, br);
                Value screened = div(erfc, rq);
                if (shift) {
                  Value shiftValue =
                      constant(std::erfc(beta * cutoff) / cutoff);
                  screened = sub(screened, shiftValue);
                }
                return {mul(fqq, screened)};
              }
              if (reactionField) {
                // Reaction field f q q (1/r + k r² - c), zero at cutoff (D140).
                Value one = constant(1.0);
                Value k = constant(reaction.first);
                Value c = constant(reaction.second);
                Value inverse = div(one, rq);
                Value rr = mul(r, r);
                Value field = mul(k, rr);
                Value near =
                    arith::AddFOp::create(builder, loc, inverse, field);
                Value rf = sub(near, c);
                return {mul(fqq, rf)};
              }
              if (shiftsAtCutoff) {
                Value one = constant(1.0);
                Value inverse = div(one, rq);
                Value rcInverse = constant(1.0 / cutoff);
                Value cut = sub(inverse, rcInverse);
                return {mul(fqq, cut)};
              }
              return {div(fqq, rq)};
            });
        if (value.empty()) {
          value = "%coulomb";
        } else {
          os << "    %k = arith.addf " << value << ", %coulomb : f64\n";
          value = "%k";
        }
      }
      // In `@alchemical` the pairs that λ leaves alone add nothing.
      if (alchemicalOnly) {
        os << "    %masked = arith.mulf " << value << ", %cross : f64\n";
        value = "%masked";
      }
      os << "    md.yield " << value << " : f64\n"
         << "  } : !pairs, !vec -> f64\n";
      add("nonbonded");
    }

    if (program.pme && (terms & TermInputs::CoulombExcluded) &&
        has("excluded")) {
      // The excluded pairs take their share of the reciprocal sum out again:
      // −f q_i q_j erf(β r) / r, with the charges that the reciprocal sum
      // takes, those of the selection of [free_energy] times 1 − λ (D161).
      bool flags = scalesCoulomb || (alchemicalOnly && decouples());
      llvm::SmallVector<StringRef> operands = {"%r_excluded", "%x", "%cell",
                                               "%p_q"};
      llvm::SmallVector<StringRef> types = {"!rel_excluded", "!vec", "!md.cell",
                                            "!real"};
      if (flags) {
        operands.append({"%p_alch", "%lambda_coulomb"});
        types.append({"!real", "f64"});
      }
      if (program.pmeProlate) {
        operands.push_back("%t_pme_near");
        types.push_back("!grid");
      }
      auto [cells, polynomialTerms] = getProlateTableShape(control.precision);
      context.emitIRFragment(
          "  ", {"%u_excluded"}, {"f64"}, operands, types,
          [flags, alchemicalOnly, prolate = program.pmeProlate,
           beta = program.pmeBeta, cutoff, cells,
           polynomialTerms](mlir::OpBuilder &builder,
                            mlir::ValueRange values) -> TermIRValues {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            DenseI32ArrayAttr coordinateKinds;
            DenseI64ArrayAttr coordinateMembers;
            mdir::md::Coordinate distance{mdir::md::CoordinateKind::Distance,
                                          {0, 1}};
            mdir::md::getCoordinateAttrs(builder, {distance}, coordinateKinds,
                                         coordinateMembers);
            auto sum = mdir::md::SumTuplesOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], values.slice(3, flags ? 2 : 1), ValueRange{},
                coordinateKinds, coordinateMembers);
            llvm::SmallVector<Type> blockTypes(flags ? 5 : 3,
                                               builder.getF64Type());
            llvm::SmallVector<Location> locations(blockTypes.size(), loc);
            Block *block = builder.createBlock(&sum.getKernel(), {}, blockTypes,
                                               locations);
            auto args = block->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(value));
            };
            auto mul = [&](Value a, Value b) -> Value {
              return arith::MulFOp::create(builder, loc, a, b);
            };
            auto sub = [&](Value a, Value b) -> Value {
              return arith::SubFOp::create(builder, loc, a, b);
            };
            Value f = constant(-coulombInternal);
            Value betaValue = constant(beta);
            Value qq = mul(args[1], args[2]);
            if (flags) {
              Value one = constant(1.0);
              Value li = mul(args[3], values[5]);
              Value si = sub(one, li);
              Value lj = mul(args[4], values[5]);
              Value sj = sub(one, lj);
              Value scale = mul(si, sj);
              qq = mul(qq, scale);
            }
            Value fqq = mul(f, qq);
            Value source = prolate ? values.back() : betaValue;
            Value shielded =
                buildShieldedCoulomb(builder, source, args[0], prolate, cutoff,
                                     cells, polynomialTerms);
            Value energy = mul(fqq, shielded);
            if (alchemicalOnly) {
              Value both = mul(args[3], args[4]);
              energy = mul(energy, both);
            }
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            return {sum.getResult()};
          });
      add("excluded");
    }
    if (program.pme && (terms & TermInputs::CoulombWithin) && scalesCoulomb &&
        has("alchemical_pairs")) {
      // The pairs within the selection that are not excluded keep their
      // TermInputs::Coulomb, which the reciprocal sum of the scaled charges
      // takes times (1 − λ)²: (1 − (1 − λ)²) f q_i q_j erf(β r) / r adds the
      // rest (D161).
      llvm::SmallVector<StringRef> operands = {
          "%r_alchemical_pairs", "%x", "%cell", "%p_q", "%lambda_coulomb"};
      llvm::SmallVector<StringRef> types = {"!rel_alchemical_pairs", "!vec",
                                            "!md.cell", "!real", "f64"};
      if (program.pmeProlate) {
        operands.push_back("%t_pme_near");
        types.push_back("!grid");
      }
      auto [cells, polynomialTerms] = getProlateTableShape(control.precision);
      context.emitIRFragment(
          "  ", {"%u_within"}, {"f64"}, operands, types,
          [prolate = program.pmeProlate, beta = program.pmeBeta, cutoff, cells,
           polynomialTerms](mlir::OpBuilder &builder,
                            mlir::ValueRange values) -> TermIRValues {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            DenseI32ArrayAttr coordinateKinds;
            DenseI64ArrayAttr coordinateMembers;
            mdir::md::Coordinate distance{mdir::md::CoordinateKind::Distance,
                                          {0, 1}};
            mdir::md::getCoordinateAttrs(builder, {distance}, coordinateKinds,
                                         coordinateMembers);
            auto sum = mdir::md::SumTuplesOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], values.slice(3, 1), ValueRange{}, coordinateKinds,
                coordinateMembers);
            llvm::SmallVector<Type> blockTypes(3, builder.getF64Type());
            llvm::SmallVector<Location> locations(blockTypes.size(), loc);
            Block *block = builder.createBlock(&sum.getKernel(), {}, blockTypes,
                                               locations);
            auto args = block->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(value));
            };
            auto mul = [&](Value a, Value b) -> Value {
              return arith::MulFOp::create(builder, loc, a, b);
            };
            auto sub = [&](Value a, Value b) -> Value {
              return arith::SubFOp::create(builder, loc, a, b);
            };
            Value f = constant(coulombInternal);
            Value betaValue = constant(beta);
            Value one = constant(1.0);
            Value scale = sub(one, values[4]);
            Value square = mul(scale, scale);
            Value weight = sub(one, square);
            Value qq = mul(args[1], args[2]);
            Value fqq = mul(f, qq);
            Value fw = mul(fqq, weight);
            Value source = prolate ? values.back() : betaValue;
            Value shielded =
                buildShieldedCoulomb(builder, source, args[0], prolate, cutoff,
                                     cells, polynomialTerms);
            Value energy = mul(fw, shielded);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            return {sum.getResult()};
          });
      add("within");
    }
    if (program.reactionField && (terms & TermInputs::CoulombExcluded) &&
        !alchemicalOnly && has("excluded")) {
      // The field acts on the excluded pairs within the cutoff as well:
      // f q_i q_j (k r² − c), as GROMACS has it (D140).
      auto [k, c] = getReactionField(control);
      context.emitIRFragment(
          "  ", {"%u_excluded"}, {"f64"},
          {"%r_excluded", "%x", "%cell", "%p_q"},
          {"!rel_excluded", "!vec", "!md.cell", "!real"},
          [k, c, cutoff](mlir::OpBuilder &builder,
                         mlir::ValueRange values) -> TermIRValues {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            DenseI32ArrayAttr coordinateKinds;
            DenseI64ArrayAttr coordinateMembers;
            mdir::md::Coordinate distance{mdir::md::CoordinateKind::Distance,
                                          {0, 1}};
            mdir::md::getCoordinateAttrs(builder, {distance}, coordinateKinds,
                                         coordinateMembers);
            auto sum = mdir::md::SumTuplesOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], values.slice(3, 1), ValueRange{}, coordinateKinds,
                coordinateMembers);
            llvm::SmallVector<Type> blockTypes(3, builder.getF64Type());
            llvm::SmallVector<Location> locations(blockTypes.size(), loc);
            Block *block = builder.createBlock(&sum.getKernel(), {}, blockTypes,
                                               locations);
            auto args = block->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(value));
            };
            auto mul = [&](Value a, Value b) -> Value {
              return arith::MulFOp::create(builder, loc, a, b);
            };
            Value f = constant(coulombInternal);
            Value krf = constant(k);
            Value crf = constant(c);
            Value rc = constant(cutoff);
            Value zero = constant(0.0);
            Value qq = mul(args[1], args[2]);
            Value fqq = mul(f, qq);
            Value rr = mul(args[0], args[0]);
            Value field = mul(krf, rr);
            Value rf = arith::SubFOp::create(builder, loc, field, crf);
            Value all = mul(fqq, rf);
            Value inside = arith::CmpFOp::create(
                builder, loc, arith::CmpFPredicate::OLT, args[0], rc);
            Value energy =
                arith::SelectOp::create(builder, loc, inside, all, zero);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            return {sum.getResult()};
          });
      add("excluded");
    }
    if (program.ljpme && (terms & TermInputs::LennardJonesExcluded) &&
        has("excluded")) {
      // The excluded pairs take their share of the grid of the dispersion
      // out again: c_i c_j P(3, β² r²) / r⁶ with P(3, y) = 1 − exp(−y) (1 +
      // y + y²/2) = 1 − g(β r) (D162). Below y = 1, P is taken from its
      // series exp(−y) y³ Σ_k y^k / (k + 3)!, which has no cancellation;
      // the pairs of a molecule are close, where 1 − g is small.
      // Horner series through k = 10; the omitted tail is below 1e-10.
      double factorial = 6.0;
      std::vector<double> coefficients;
      for (int k = 0; k <= 10; ++k) {
        coefficients.push_back(1.0 / factorial);
        factorial *= k + 4;
      }
      context.emitIRFragment(
          "  ", {"%u_lj_excluded"}, {"f64"},
          {"%r_excluded", "%x", "%cell", "%p_type", "%t_ljpme_c6"},
          {"!rel_excluded", "!vec", "!md.cell", "!ids", "!table"},
          [beta = program.ljpmeBeta,
           coefficients](mlir::OpBuilder &builder,
                         mlir::ValueRange values) -> TermIRValues {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            DenseI32ArrayAttr coordinateKinds;
            DenseI64ArrayAttr coordinateMembers;
            mdir::md::Coordinate distance{mdir::md::CoordinateKind::Distance,
                                          {0, 1}};
            mdir::md::getCoordinateAttrs(builder, {distance}, coordinateKinds,
                                         coordinateMembers);
            auto sum = mdir::md::SumTuplesOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], values.slice(3, 1), ValueRange{}, coordinateKinds,
                coordinateMembers);
            llvm::SmallVector<Type> blockTypes = {builder.getF64Type(),
                                                  builder.getI32Type(),
                                                  builder.getI32Type()};
            llvm::SmallVector<Location> locations(blockTypes.size(), loc);
            Block *block = builder.createBlock(&sum.getKernel(), {}, blockTypes,
                                               locations);
            auto args = block->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(value));
            };
            auto mul = [&](Value a, Value b) -> Value {
              return arith::MulFOp::create(builder, loc, a, b);
            };
            auto add = [&](Value a, Value b) -> Value {
              return arith::AddFOp::create(builder, loc, a, b);
            };
            Value r = args[0];
            Value c6 = mdir::md::LookupOp::create(
                builder, loc, builder.getF64Type(), values[4],
                ValueRange{args[1], args[2]});
            Value betaValue = constant(beta);
            Value bx = mul(betaValue, r);
            Value y = mul(bx, bx);
            Value ny = arith::NegFOp::create(builder, loc, y);
            Value ey = math::ExpOp::create(builder, loc, ny);
            Value one = constant(1.0);
            Value half = constant(0.5);
            Value polynomial = constant(coefficients[10]);
            for (int k = 9; k >= 0; --k) {
              Value a = constant(coefficients[k]);
              Value product = mul(polynomial, y);
              polynomial = add(product, a);
            }
            Value y2 = mul(y, y);
            Value y3 = mul(y2, y);
            Value ey3 = mul(ey, y3);
            Value series = mul(ey3, polynomial);
            Value hy2 = mul(half, y2);
            Value p1 = add(one, y);
            Value p = add(p1, hy2);
            Value g = mul(ey, p);
            Value closed = arith::SubFOp::create(builder, loc, one, g);
            Value small = arith::CmpFOp::create(
                builder, loc, arith::CmpFPredicate::OLT, y, one);
            Value kept =
                arith::SelectOp::create(builder, loc, small, series, closed);
            Value r2 = mul(r, r);
            Value r4 = mul(r2, r2);
            Value r6 = mul(r4, r2);
            Value cr = arith::DivFOp::create(builder, loc, c6, r6);
            Value energy = mul(cr, kept);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            return {sum.getResult()};
          });
      add("lj_excluded");
    }
    if (program.ljpme && (terms & TermInputs::LennardJonesReciprocal)) {
      // The grid of the dispersion (D162), of the coefficients c_i.
      bool listed = hasField("ljpme_listed");
      os << "  %u_lj_reciprocal, %f_lj_reciprocal, %w_lj_reciprocal = "
            "md.reciprocal %x, %p_ljpme_c, %cell, %t_ljpme_moduli"
         << (listed ? " listed(%p_ljpme_listed)" : "") << "\n"
         << "      grid([" << program.ljpmeGrid[0] << ", "
         << program.ljpmeGrid[1] << ", " << program.ljpmeGrid[2] << "]) order("
         << control.ljpmeOrder << ") beta(" << formatReal(program.ljpmeBeta)
         << ") coulomb(1.0) dispersion"
         << (listed && gathersAll("ljpme_listed") ? " gather_all" : "") << "\n"
         << "      : !vec, !real, !grid" << (listed ? ", !ids" : "")
         << " -> f64, !vec, vector<9xf64>\n";
      add("lj_reciprocal");
    }
    if (program.pme && (terms & TermInputs::CoulombReciprocal)) {
      // The charges of the selection of [free_energy] times 1 − λ (D161).
      std::string charges = "%p_q";
      if (scalesCoulomb) {
        os << "  %q_scaled = md.map_particles gather(%p_q, %p_alch : !real, "
              "!real) {\n"
           << "  ^bb0(%q: f64, %al: f64):\n"
           << "    %one = arith.constant 1.0 : f64\n"
           << "    %al_l = arith.mulf %al, %lambda_coulomb : f64\n"
           << "    %al_s = arith.subf %one, %al_l : f64\n"
           << "    %qs = arith.mulf %q, %al_s : f64\n"
           << "    md.yield %qs : f64\n"
           << "  } : !real\n";
        charges = "%q_scaled";
      }
      // The particles of a charge that can be nonzero alone
      // (D262).
      bool listed = hasField("pme_listed");
      os << "  %u_reciprocal, %f_reciprocal, %w_reciprocal = md.reciprocal %x, "
         << charges << ", %cell, %t_pme_moduli"
         << (listed ? " listed(%p_pme_listed)" : "") << "\n"
         << "      grid([" << program.pmeGrid[0] << ", " << program.pmeGrid[1]
         << ", " << program.pmeGrid[2] << "]) order(" << program.pmeOrder
         << ") beta(" << formatReal(program.pmeBeta) << ") coulomb("
         << formatReal(coulombInternal) << ")"
         << (program.pmeProlate ? " prolate" : "")
         << (program.pmeP3M ? " p3m" : "")
         << (listed && gathersAll("pme_listed") ? " gather_all" : "") << "\n"
         << "      : !vec, !real, !grid" << (listed ? ", !ids" : "")
         << " -> f64, !vec, vector<9xf64>\n";
      add("reciprocal");
    }
  }

private:
  mutable std::optional<TermNonbondedData> state;
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createNonbondedTermBuilder() {
  return std::make_unique<NonbondedTermBuilder>();
}

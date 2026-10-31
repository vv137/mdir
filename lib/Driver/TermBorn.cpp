// Generalized Born and its dependent nonpolar surface contribution
// (D359, D361, #683). Computed Born radii stay private to this adapter.
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

struct BornIRConfig {
  bool hct = false, energy = false;
  double cutoff = 0.0, radiusCutoff = 0.0, reachCutoff = 0.0;
  double alpha = 0.0, beta = 0.0, gamma = 0.0;
  double kappa = 0.0, tau = 0.0, inverseSolute = 0.0, inverseSolvent = 0.0;
};

static TermIRValues buildBornContribution(mlir::OpBuilder &builder, mlir::ValueRange values, const BornIRConfig &config) {
        using namespace mlir;
        auto *ctx = builder.getContext();
        auto loc = builder.getUnknownLoc();
        auto f64 = builder.getF64Type();
        auto vec = VectorType::get({3}, f64);
        Type real = values[2].getType();
        auto pairsType = mdir::md::RelationType::get(
            ctx, cast<mdir::md::FieldType>(real).getParticleSet(), 2,
            mdir::md::Orientation::Unordered, FlatSymbolRefAttr{});
        auto constant = [&](double value) -> Value {
          return arith::ConstantOp::create(builder, loc,
                                           builder.getF64FloatAttr(value));
        };
        auto add = [&](Value a, Value b) -> Value {
          return arith::AddFOp::create(builder, loc, a, b);
        };
        auto sub = [&](Value a, Value b) -> Value {
          return arith::SubFOp::create(builder, loc, a, b);
        };
        auto mul = [&](Value a, Value b) -> Value {
          return arith::MulFOp::create(builder, loc, a, b);
        };
        auto div = [&](Value a, Value b) -> Value {
          return arith::DivFOp::create(builder, loc, a, b);
        };
        Value neighborhood = mdir::md::NeighborhoodOp::create(
            builder, loc, pairsType, values[0], values[1], Value{},
            builder.getF64FloatAttr(config.cutoff));
        Value reach = neighborhood;
        if (config.radiusCutoff > 0.0)
          reach = mdir::md::NeighborhoodOp::create(
              builder, loc, pairsType, values[0], values[1], Value{},
              builder.getF64FloatAttr(config.reachCutoff));
        // Descreening [Hawkins1996], integrated only over shells inside the
        // smooth radius cutoff when one is selected (D152).
        auto integral = mdir::md::GatherRelationOp::create(
            builder, loc, TypeRange{real}, reach, values[0], values[1],
            ValueRange{values[2], values[3]}, mdir::md::Exchange::None);
        {
          OpBuilder::InsertionGuard guard(builder);
          llvm::SmallVector<Type> types = {f64, vec, f64, f64, f64, f64};
          llvm::SmallVector<Location> locations(types.size(), loc);
          Block *block =
              builder.createBlock(&integral.getKernel(), {}, types, locations);
          auto args = block->getArguments();
          Value r = args[0], ri = args[2], sj = args[5];
          Value zero = constant(0.0), one = constant(1.0), two = constant(2.0);
          Value quarter = constant(0.25), half = constant(0.5);
          Value upper = add(r, sj);
          if (config.radiusCutoff > 0.0) {
            Value rgb = constant(config.radiusCutoff);
            upper = arith::MinimumFOp::create(builder, loc, upper, rgb);
          }
          Value gap = sub(r, sj);
          Value agap = math::AbsFOp::create(builder, loc, gap);
          Value lower = arith::MaximumFOp::create(builder, loc, ri, agap);
          Value l = div(one, lower), u = div(one, upper);
          Value l2 = mul(l, l), u2 = mul(u, u);
          Value t0 = sub(l, u), du = sub(u2, l2);
          Value qr = mul(quarter, r), t1 = mul(qr, du);
          Value ratio = div(u, l);
          Value logarithm = math::LogOp::create(builder, loc, ratio);
          Value hr = div(half, r), t2 = mul(hr, logarithm);
          Value sj2 = mul(sj, sj), qs = mul(quarter, sj2);
          Value qsr = div(qs, r), dl = sub(l2, u2), t3 = mul(qsr, dl);
          Value a0 = add(t0, t1), a1 = add(a0, t2), a2 = add(a1, t3);
          Value inner = sub(sj, r);
          Value within = arith::CmpFOp::create(
              builder, loc, arith::CmpFPredicate::OLT, ri, inner);
          Value iri = div(one, ri), e0 = sub(iri, l), e1 = mul(two, e0),
                a3 = add(a2, e1);
          Value a4 = arith::SelectOp::create(builder, loc, within, a3, a2);
          Value reaches = arith::CmpFOp::create(
              builder, loc, arith::CmpFPredicate::OLT, lower, upper);
          Value result =
              arith::SelectOp::create(builder, loc, reaches, a4, zero);
          mdir::md::YieldOp::create(builder, loc, ValueRange{result});
        }
        // HCT [Hawkins1996] or OBC [Onufriev2004] radii.
        llvm::SmallVector<Value> fields = {integral.getResult(), values[2]};
        if (!config.hct)
          fields.push_back(values[4]);
        auto radii =
            mdir::md::MapParticlesOp::create(builder, loc, real, fields);
        {
          OpBuilder::InsertionGuard guard(builder);
          llvm::SmallVector<Type> types(fields.size(), f64);
          llvm::SmallVector<Location> locations(types.size(), loc);
          Block *block =
              builder.createBlock(&radii.getKernel(), {}, types, locations);
          auto args = block->getArguments();
          Value half = constant(0.5), one = constant(1.0);
          Value result;
          if (config.hct) {
            Value hi = mul(half, args[0]), iri = div(one, args[1]);
            Value den = sub(iri, hi);
            result = div(one, den);
          } else {
            Value alpha = constant(config.alpha), beta = constant(config.beta),
                  gamma = constant(config.gamma);
            Value hi = mul(half, args[0]), psi = mul(hi, args[1]);
            Value psi2 = mul(psi, psi), psi3 = mul(psi2, psi);
            Value a = mul(alpha, psi), b = mul(beta, psi2),
                  c = mul(gamma, psi3);
            Value s0 = sub(a, b), s1 = add(s0, c);
            Value t = math::TanhOp::create(builder, loc, s1);
            Value tr = div(t, args[2]), iri = div(one, args[1]);
            Value den = sub(iri, tr);
            result = div(one, den);
          }
          mdir::md::YieldOp::create(builder, loc, ValueRange{result});
        }
        // Generalized Born pair/self energy with salt screening
        // [Srinivasan1999] (D152), at the same ordered insertion site.
        TermIRValues results = {radii.getResult()};
        if (config.energy) {
          auto screening = [&](double factor, Value distance) -> Value {
            if (config.kappa == 0.0)
              return constant(-factor * config.tau * coulombInternal);
            Value cf = constant(-factor * coulombInternal);
            Value ein = constant(config.inverseSolute),
                  eout = constant(config.inverseSolvent);
            Value nkappa = constant(-config.kappa), kd = mul(nkappa, distance);
            Value ekd = math::ExpOp::create(builder, loc, kd);
            Value screened = mul(ekd, eout), tau = sub(ein, screened);
            return mul(cf, tau);
          };
          auto pairs = mdir::md::SumRelationOp::create(
              builder, loc, f64, neighborhood, values[0], values[1],
              ValueRange{values[5], radii.getResult()},
              mdir::md::Exchange::Symmetric, mdir::md::ExchangeBasis::Proof,
              mdir::md::Truncation::None, FloatAttr{});
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types = {f64, vec, f64, f64, f64, f64};
            llvm::SmallVector<Location> locations(types.size(), loc);
            Block *block =
                builder.createBlock(&pairs.getKernel(), {}, types, locations);
            auto args = block->getArguments();
            Value quarter = constant(0.25), bb = mul(args[4], args[5]),
                  r2 = mul(args[0], args[0]);
            Value bb4 = div(quarter, bb), x0 = mul(r2, bb4);
            Value nx = arith::NegFOp::create(builder, loc, x0);
            Value ex = math::ExpOp::create(builder, loc, nx);
            Value be = mul(bb, ex), s = add(r2, be);
            Value fgb = math::SqrtOp::create(builder, loc, s);
            Value ct = screening(1.0, fgb), qq = mul(args[2], args[3]);
            Value cq = mul(ct, qq), energy = div(cq, fgb);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
          }
          auto self = mdir::md::SumParticlesOp::create(
              builder, loc, f64, ValueRange{values[5], radii.getResult()});
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types(2, f64);
            llvm::SmallVector<Location> locations(types.size(), loc);
            Block *block =
                builder.createBlock(&self.getKernel(), {}, types, locations);
            auto args = block->getArguments();
            Value ct = screening(0.5, args[1]), qq = mul(args[0], args[0]);
            Value cq = mul(ct, qq), energy = div(cq, args[1]);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
          }
          results.push_back(add(pairs.getResult(), self.getResult()));
        }
        return results;
}

static mlir::Value buildSurfaceContribution(mlir::OpBuilder &builder, mlir::ValueRange values, double tension) {
          using namespace mlir;
          auto loc = builder.getUnknownLoc();
          auto sum = mdir::md::SumParticlesOp::create(
              builder, loc, builder.getF64Type(), values);
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types(2, builder.getF64Type());
            llvm::SmallVector<Location> locations(2, loc);
            Block *block =
                builder.createBlock(&sum.getKernel(), {}, types, locations);
            auto args = block->getArguments();
            Value c = arith::ConstantOp::create(
                builder, loc, builder.getF64FloatAttr(4.0 * M_PI * tension));
            Value probe = arith::ConstantOp::create(
                builder, loc, builder.getF64FloatAttr(0.14));
            Value six = arith::ConstantIntOp::create(builder, loc, 6, 32);
            Value rp = arith::AddFOp::create(builder, loc, args[0], probe);
            Value rp2 = arith::MulFOp::create(builder, loc, rp, rp);
            Value ratio = arith::DivFOp::create(builder, loc, args[0], args[1]);
            Value r6 = math::FPowIOp::create(builder, loc, ratio, six);
            Value a = arith::MulFOp::create(builder, loc, c, rp2);
            Value energy = arith::MulFOp::create(builder, loc, a, r6);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
          }
          return sum.getResult();
}

namespace mdir::driver::builder {
class BornTermBuilder final : public TermBuilder {
public:
  BornTermBuilder() {
    potentialMask = TermInputs::GeneralizedBorn | TermInputs::Surface;
    nativePotentialMask = potentialMask;
    alchemicalMask = 0;
  }
  int parameterRank() const override { return -2; }
  int potentialRank() const override { return -1; }
  void collectParameters(TermParameters &context) const override;
  llvm::Error buildTopologyPotential(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    if (!(selection.terms & (TermInputs::GeneralizedBorn | TermInputs::Surface)) ||
        context.getControl().implicitSolvent == Control::ImplicitSolvent::None) return llvm::Error::success();
    auto config = configure(context, selection.terms);
    TermIRValues values{state.physical.positions, state.physical.cell,
        state.physical.field("gb_offset"), state.physical.field("gb_scaled"), state.physical.field("gb_radius")};
    if (config.energy) values.push_back(state.physical.field("q"));
    auto born = buildBornContribution(builder, values, config);
    if (config.energy) add(born[1]);
    if ((selection.terms & TermInputs::Surface) && context.getControl().surfaceAreaEnergy > 0.0) {
      double tension = context.getControl().surfaceAreaEnergy * units::energy / (units::length * units::length);
      add(buildSurfaceContribution(builder,
          mlir::ValueRange{state.physical.field("gb_radius"), born[0]}, tension));
    }

    return llvm::Error::success();
  }
  void emitTopologyPotential(
      TermEmission &context, const PotentialSelection &selection,
      const TermPotentialAddition & add) const override {
    if ((selection.terms & (TermInputs::GeneralizedBorn | TermInputs::Surface)) &&
        context.getControl().implicitSolvent != Control::ImplicitSolvent::None)
      emitBorn(context, selection.terms, add);
  }

  void appendTermDefinitions(
      const TermInputs &context,
      std::vector<TermDefinition> &definitions) const override {
    const auto &terms = context.getSystem().bornTermNames;
    if (terms.empty())
      return;
    definitions.push_back({16, 0, "term_born", TermInputs::GeneralizedBorn});
    if (terms.size() > 1)
      definitions.push_back({16, 1, "term_surface", TermInputs::Surface});
  }
private:
  static BornIRConfig configure(const TermInputs &, unsigned);
  static void emitBorn(TermEmission &, unsigned,
                       const TermPotentialAddition &);
  static double getDebyeKappa(const Control &);
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createBornTermBuilder() {
  return std::make_unique<BornTermBuilder>();
}

void BornTermBuilder::collectParameters(TermParameters &context) const {
  const Control &control = context.getControl();
  const Topology &topology = context.getTopology();
  size_t count = topology.getNumParticles();
  // Generalized Born (D144): the radius less its offset, 0.009 nm, the
  // radius that screens the others, that times the scale of the particle,
  // and the radius itself, in nm.
  if (control.implicitSolvent != Control::ImplicitSolvent::None) {
    Program::Field offset, scaled, radius;
    offset.name = "gb_offset";
    scaled.name = "gb_scaled";
    radius.name = "gb_radius";
    for (size_t i = 0; i != count; ++i) {
      double rho = topology.bornRadii[i];
      offset.values.push_back(rho - 0.009);
      scaled.values.push_back(topology.bornScreens[i] * (rho - 0.009));
      radius.values.push_back(rho);
    }
    context.getFields().push_back(std::move(offset));
    context.getFields().push_back(std::move(scaled));
    context.getFields().push_back(std::move(radius));
  }
}

void BornTermBuilder::emitBorn(TermEmission &context, unsigned terms,
                       const TermPotentialAddition & add) {
  const Control &control = context.getControl();
  BornIRConfig config = configure(context, terms);
  llvm::SmallVector<StringRef> outputs = {"%gb_born"}, resultTypes = {"!real"};
  llvm::SmallVector<StringRef> operands = {"%x", "%cell", "%p_gb_offset",
                                           "%p_gb_scaled", "%p_gb_radius"};
  llvm::SmallVector<StringRef> operandTypes = {"!vec", "!md.cell", "!real",
                                               "!real", "!real"};
  if (config.energy) {
    outputs.push_back("%u_born");
    resultTypes.push_back("f64");
    operands.push_back("%p_q");
    operandTypes.push_back("!real");
  }
  context.emitIRFragment(
      "  ", outputs, resultTypes, operands, operandTypes,
      [config](mlir::OpBuilder &builder,
               mlir::ValueRange values) -> TermIRValues {
        return buildBornContribution(builder, values, config);
      });
  // Add the Born contribution before constructing the dependent surface
  // energy, preserving the outer accumulation order as well as loop order.
  if (config.energy)
    add("born");
  if ((terms & TermInputs::Surface) && control.surfaceAreaEnergy > 0.0) {
    double tension = control.surfaceAreaEnergy * units::energy /
                     (units::length * units::length);
    context.emitIRFragment(
        "  ", {"%u_surface"}, {"f64"}, {"%p_gb_radius", "%gb_born"},
        {"!real", "!real"},
        [tension](mlir::OpBuilder &builder,
                  mlir::ValueRange values) -> TermIRValues {
          return {buildSurfaceContribution(builder, values, tension)};
        });
    add("surface");
  }
}

BornIRConfig BornTermBuilder::configure(const TermInputs &context, unsigned terms) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  BornIRConfig config;
  config.cutoff = control.cutoffDistance * units::length;
  config.radiusCutoff = control.bornRadiusCutoff * units::length;
  config.reachCutoff = config.cutoff;
  if (config.radiusCutoff > 0.0) {
    double largest = 0.0;
    for (size_t i = 0, e = system.topology->bornRadii.size(); i != e; ++i)
      largest = std::max(largest, system.topology->bornScreens[i] *
                                      (system.topology->bornRadii[i] - 0.009));
    config.reachCutoff = std::min(config.radiusCutoff + largest, config.cutoff);
  }
  config.hct = control.implicitSolvent == Control::ImplicitSolvent::HCT;
  bool first = control.implicitSolvent == Control::ImplicitSolvent::OBC1;
  config.alpha = first ? 0.8 : 1.0;
  config.beta = first ? 0.0 : 0.8;
  config.gamma = first ? 2.909125 : 4.85;
  config.energy = terms & TermInputs::GeneralizedBorn;
  if (config.energy) {
    config.inverseSolute = 1.0 / control.soluteDielectric;
    config.inverseSolvent = 1.0 / control.solventDielectric;
    config.tau = config.inverseSolute - config.inverseSolvent;
    config.kappa = getDebyeKappa(control);
  }
  return config;
}

double BornTermBuilder::getDebyeKappa(const Control &control) {
  // κ² = 2 N_A e² c / (ε0 ε_r k_B T) for a 1:1 salt of c mol/L, with the
  // constants of CODATA 2018 in SI units, c in mol/m³; κ in nm⁻¹.
  if (control.saltConcentration <= 0.0)
    return 0.0;
  const double avogadro = 6.02214076e23, charge = 1.602176634e-19,
               permittivity = 8.8541878128e-12, boltzmann = 1.380649e-23;
  double concentration = control.saltConcentration * 1000.0;
  double kappa2 = 2.0 * avogadro * charge * charge * concentration /
                  (permittivity * control.solventDielectric * boltzmann *
                   control.temperature);
  return 0.73 * std::sqrt(kappa2) * 1e-9;
}

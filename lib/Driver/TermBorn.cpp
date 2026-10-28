// Generalized Born and its dependent nonpolar surface contribution
// (D359, D361, #683). Computed Born radii stay private to this adapter.
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class BornTermBuilder final : public TermBuilder {
public:
  int parameterRank() const override { return -2; }
  int potentialRank() const override { return -1; }
  void collectParameters(TermParameters &context) const override;
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
  const System &system = context.getSystem();
  llvm::raw_ostream &os = context.getOutput();
  // Every pair within the cutoff, those that the topology excludes as
  // well.
  double cutoff = control.cutoffDistance * units::length;
  os << "  %ngb = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
     << ") : !vec -> !pairs\n";
  // With a cutoff R of the radii (D152), the descreening reaches R and the
  // largest screened radius beyond it, which System checked is within the
  // cutoff.
  double radiusCutoff = control.bornRadiusCutoff * units::length;
  std::string reach = "%ngb";
  if (radiusCutoff > 0.0) {
    double largest = 0.0;
    for (size_t i = 0, e = system.topology->bornRadii.size(); i != e; ++i)
      largest = std::max(largest, system.topology->bornScreens[i] *
                                      (system.topology->bornRadii[i] - 0.009));
    reach = "%ngb_radii";
    os << "  %ngb_radii = md.neighborhood %x, %cell cutoff("
       << formatReal(std::min(radiusCutoff + largest, cutoff))
       << ") : !vec -> !pairs\n";
  }
  // The integral of the descreening of i by the sphere of j of the radius
  // s_j [Hawkins1996]: with L = max(ρ̃_i, |r − s_j|) and U = r + s_j,
  //   ½ (1/L − 1/U + r/4 (1/U² − 1/L²) + ln(L/U) / (2r) + s_j²/(4r) (1/L² − 1/U²)),
  // and 1/ρ̃_i − 1/L more where i lies within the sphere of j, 0 where the
  // sphere does not reach it, L ≥ U. A cutoff R of the radii takes the
  // integral over the shells within R alone, U = min(r + s_j, R), so that
  // the radii do not jump where a sphere crosses it: the smooth cutoff
  // `rgbmax` of Amber's pmemd (gb_ene.F90), written here from the integral
  // rather than its series (D152).
  os << "  %gb_integral = md.gather_relation " << reach << ", %x, %cell gather("
        "%p_gb_offset, %p_gb_scaled : !real, !real)\n"
     << "      exchange(none) {\n"
     << "  ^bb0(%r: f64, %d: vector<3xf64>, %ri: f64, %rj: f64, %si: f64, "
        "%sj: f64):\n"
     << "    %zero = arith.constant 0.0 : f64\n"
     << "    %one = arith.constant 1.0 : f64\n"
     << "    %two = arith.constant 2.0 : f64\n"
     << "    %quarter = arith.constant 0.25 : f64\n"
     << "    %half = arith.constant 0.5 : f64\n"
     << "    %upper" << (radiusCutoff > 0.0 ? "_sphere" : "")
     << " = arith.addf %r, %sj : f64\n";
  if (radiusCutoff > 0.0)
    os << "    %rgb = arith.constant " << formatReal(radiusCutoff)
       << " : f64\n"
       << "    %upper = arith.minimumf %upper_sphere, %rgb : f64\n";
  os << "    %gap = arith.subf %r, %sj : f64\n"
     << "    %agap = math.absf %gap : f64\n"
     << "    %lower = arith.maximumf %ri, %agap : f64\n"
     << "    %l = arith.divf %one, %lower : f64\n"
     << "    %u = arith.divf %one, %upper : f64\n"
     << "    %l2 = arith.mulf %l, %l : f64\n"
     << "    %u2 = arith.mulf %u, %u : f64\n"
     << "    %t0 = arith.subf %l, %u : f64\n"
     << "    %du = arith.subf %u2, %l2 : f64\n"
     << "    %qr = arith.mulf %quarter, %r : f64\n"
     << "    %t1 = arith.mulf %qr, %du : f64\n"
     << "    %ratio = arith.divf %u, %l : f64\n"
     << "    %log = math.log %ratio : f64\n"
     << "    %hr = arith.divf %half, %r : f64\n"
     << "    %t2 = arith.mulf %hr, %log : f64\n"
     << "    %sj2 = arith.mulf %sj, %sj : f64\n"
     << "    %qs = arith.mulf %quarter, %sj2 : f64\n"
     << "    %qsr = arith.divf %qs, %r : f64\n"
     << "    %dl = arith.subf %l2, %u2 : f64\n"
     << "    %t3 = arith.mulf %qsr, %dl : f64\n"
     << "    %a0 = arith.addf %t0, %t1 : f64\n"
     << "    %a1 = arith.addf %a0, %t2 : f64\n"
     << "    %a2 = arith.addf %a1, %t3 : f64\n"
     << "    %inner = arith.subf %sj, %r : f64\n"
     << "    %within = arith.cmpf olt, %ri, %inner : f64\n"
     << "    %iri = arith.divf %one, %ri : f64\n"
     << "    %e0 = arith.subf %iri, %l : f64\n"
     << "    %e1 = arith.mulf %two, %e0 : f64\n"
     << "    %a3 = arith.addf %a2, %e1 : f64\n"
     << "    %a4 = arith.select %within, %a3, %a2 : f64\n"
     << "    %reach = arith.cmpf olt, %lower, %upper : f64\n"
     << "    %k = arith.select %reach, %a4, %zero : f64\n"
     << "    md.yield %k : f64\n"
     << "  } : !pairs, !vec -> !real\n";
  // The Born radii of Hawkins, Cramer, and Truhlar [Hawkins1996],
  // B = 1 / (1/ρ̃ − I/2) (the factor ½ of the integral).
  if (control.implicitSolvent == Control::ImplicitSolvent::HCT)
    os << "  %gb_born = md.map_particles gather(%gb_integral, %p_gb_offset "
          ": !real, !real) {\n"
       << "  ^bb0(%i: f64, %ri: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %hi = arith.mulf %half, %i : f64\n"
       << "    %iri = arith.divf %one, %ri : f64\n"
       << "    %den = arith.subf %iri, %hi : f64\n"
       << "    %bi = arith.divf %one, %den : f64\n"
       << "    md.yield %bi : f64\n"
       << "  } : !real\n";
  // Those of Onufriev, Bashford, and Case [Onufriev2004]: with ψ = I ρ̃ / 2,
  // B = 1 / (1/ρ̃ − tanh(α ψ − β ψ² + γ ψ³) / ρ).
  bool first = control.implicitSolvent == Control::ImplicitSolvent::OBC1;
  double alpha = first ? 0.8 : 1.0, beta = first ? 0.0 : 0.8,
         gamma = first ? 2.909125 : 4.85;
  if (control.implicitSolvent != Control::ImplicitSolvent::HCT)
    os << "  %gb_born = md.map_particles gather(%gb_integral, %p_gb_offset, "
          "%p_gb_radius : !real, !real, !real) {\n"
       << "  ^bb0(%i: f64, %ri: f64, %r0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %alpha = arith.constant " << formatReal(alpha) << " : f64\n"
       << "    %beta = arith.constant " << formatReal(beta) << " : f64\n"
       << "    %gamma = arith.constant " << formatReal(gamma) << " : f64\n"
       << "    %hi = arith.mulf %half, %i : f64\n"
       << "    %psi = arith.mulf %hi, %ri : f64\n"
       << "    %psi2 = arith.mulf %psi, %psi : f64\n"
       << "    %psi3 = arith.mulf %psi2, %psi : f64\n"
       << "    %a = arith.mulf %alpha, %psi : f64\n"
       << "    %b = arith.mulf %beta, %psi2 : f64\n"
       << "    %c = arith.mulf %gamma, %psi3 : f64\n"
       << "    %s0 = arith.subf %a, %b : f64\n"
       << "    %s1 = arith.addf %s0, %c : f64\n"
       << "    %t = math.tanh %s1 : f64\n"
       << "    %tr = arith.divf %t, %r0 : f64\n"
       << "    %iri = arith.divf %one, %ri : f64\n"
       << "    %den = arith.subf %iri, %tr : f64\n"
       << "    %bi = arith.divf %one, %den : f64\n"
       << "    md.yield %bi : f64\n"
       << "  } : !real\n";
  if (terms & TermInputs::GeneralizedBorn) {
    // −τ f q_i q_j / f_GB over the pairs and −τ f q_i² / (2 B_i) for each
    // particle, f_GB = sqrt(r² + B_i B_j exp(−r² / (4 B_i B_j))) and
    // τ = 1/ε_solute − 1/ε_solvent. A salt screens the solvent with the
    // Debye length 1/κ, τ = 1/ε_solute − exp(−κ f_GB)/ε_solvent, with κ
    // scaled by 0.73 for the layer around the solute that excludes the ions
    // [Srinivasan1999] (D152); f_GB is B_i for a particle.
    double tau = 1.0 / control.soluteDielectric -
                 1.0 / control.solventDielectric;
    double kappa = getDebyeKappa(control);
    // Emits `%ct`, f τ with the factor `factor` of f, for the distance
    // `distance`.
    auto emitScreening = [&](double factor, StringRef distance) {
      if (kappa == 0.0) {
        os << "    %ct = arith.constant "
           << formatReal(-factor * tau * coulombInternal) << " : f64\n";
        return;
      }
      os << "    %cf = arith.constant " << formatReal(-factor * coulombInternal)
         << " : f64\n"
         << "    %ein = arith.constant "
         << formatReal(1.0 / control.soluteDielectric) << " : f64\n"
         << "    %eout = arith.constant "
         << formatReal(1.0 / control.solventDielectric) << " : f64\n"
         << "    %nkappa = arith.constant " << formatReal(-kappa) << " : f64\n"
         << "    %kd = arith.mulf %nkappa, " << distance << " : f64\n"
         << "    %ekd = math.exp %kd : f64\n"
         << "    %screened = arith.mulf %ekd, %eout : f64\n"
         << "    %tau = arith.subf %ein, %screened : f64\n"
         << "    %ct = arith.mulf %cf, %tau : f64\n";
    };
    os << "  %u_gb_pairs = md.sum_relation %ngb, %x, %cell gather(%p_q, "
          "%gb_born : !real, !real)\n"
       << "      exchange(symmetric) {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64, %bi: f64, "
          "%bj: f64):\n"
       << "    %quarter = arith.constant 0.25 : f64\n"
       << "    %bb = arith.mulf %bi, %bj : f64\n"
       << "    %r2 = arith.mulf %r, %r : f64\n"
       << "    %bb4 = arith.divf %quarter, %bb : f64\n"
       << "    %x0 = arith.mulf %r2, %bb4 : f64\n"
       << "    %nx = arith.negf %x0 : f64\n"
       << "    %ex = math.exp %nx : f64\n"
       << "    %be = arith.mulf %bb, %ex : f64\n"
       << "    %s = arith.addf %r2, %be : f64\n"
       << "    %fgb = math.sqrt %s : f64\n";
    emitScreening(1.0, "%fgb");
    os << "    %qq = arith.mulf %qi, %qj : f64\n"
       << "    %cq = arith.mulf %ct, %qq : f64\n"
       << "    %e = arith.divf %cq, %fgb : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !pairs, !vec -> f64\n"
       << "  %u_gb_self = md.sum_particles gather(%p_q, %gb_born : !real, "
          "!real) {\n"
       << "  ^bb0(%qi: f64, %bi: f64):\n";
    emitScreening(0.5, "%bi");
    os << "    %qq = arith.mulf %qi, %qi : f64\n"
       << "    %cq = arith.mulf %ct, %qq : f64\n"
       << "    %e = arith.divf %cq, %bi : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : f64\n"
       << "  %u_born = arith.addf %u_gb_pairs, %u_gb_self : f64\n";
    add("born");
  }
  if ((terms & TermInputs::Surface) && control.surfaceAreaEnergy > 0.0) {
    // The nonpolar term of Schaefer et al., 4π γ (ρ + 0.14 nm)² (ρ/B)⁶,
    // with γ from kcal/(mol Å²) in kJ/(mol nm²).
    double tension = control.surfaceAreaEnergy * units::energy /
                     (units::length * units::length);
    os << "  %u_surface = md.sum_particles gather(%p_gb_radius, %gb_born : "
          "!real, !real) {\n"
       << "  ^bb0(%r0: f64, %bi: f64):\n"
       << "    %c = arith.constant " << formatReal(4.0 * M_PI * tension)
       << " : f64\n"
       << "    %probe = arith.constant 0.14 : f64\n"
       << "    %six = arith.constant 6 : i32\n"
       << "    %rp = arith.addf %r0, %probe : f64\n"
       << "    %rp2 = arith.mulf %rp, %rp : f64\n"
       << "    %ratio = arith.divf %r0, %bi : f64\n"
       << "    %r6 = math.fpowi %ratio, %six : f64, i32\n"
       << "    %a = arith.mulf %c, %rp2 : f64\n"
       << "    %e = arith.mulf %a, %r6 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : f64\n";
    add("surface");
  }
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

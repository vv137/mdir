// Positional-restraint fields and contributions (D359, D361, #683).
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class RestraintTermBuilder final : public TermBuilder {
public:
  int parameterRank() const override { return -1; }
  void collectParameters(TermParameters &context) const override;
  bool hasForceCorrection(const TermInputs &context) const override {
    return !context.getSystem().restraintConstants.empty();
  }
  void emitForceCorrection(TermEmission &context,
                           const TermCorrection &point) const override {
    if (!hasForceCorrection(context))
      return;
    emitCorrection(context, point.scale, point.indent, point.x, point.prefix,
                   point.f, point.fResult, point.u, point.uResult, point.w,
                   point.wResult);
  }
  unsigned getStoredTermCount(const TermInputs &context) const override {
    return hasForceCorrection(context) ? 1 : 0;
  }
  void emitStoredTerms(TermEmission &context, StringRef type,
                       int &index) const override {
    if (!hasForceCorrection(context))
      return;
    context.getOutput() << "  %i_restraints = arith.constant " << index++ << " : index\n"
       << "  memref.store %u0_r, %terms[%i_restraints] : " << type << "\n";
  }
private:
  static void emitCorrection(TermEmission &, StringRef scaleName, StringRef indent,
      StringRef x, StringRef prefix, StringRef f, StringRef fResult,
      StringRef u, StringRef uResult, StringRef w, StringRef wResult);
};
} // namespace mdir::driver::builder

const TermBuilder &mdir::driver::builder::getRestraintTermBuilder() {
  static const RestraintTermBuilder builder;
  return builder;
}

void RestraintTermBuilder::collectParameters(TermParameters &context) const {
  const System &system = context.getSystem();
  const Control &control = context.getControl();
  // The restraints: the constant of each particle, 0 for one that is not
  // restrained, and its reference position, in nm.
  if (!system.restraintConstants.empty()) {
    Program::Field constants;
    constants.name = "rest_k";
    constants.values = system.restraintConstants;
    context.getFields().push_back(std::move(constants));
    // Under a barostat each reference is its center c, which scales with
    // the cell, plus its offset o from it, which stays (D124). The center of
    // the references of `reference_scaling = "CENTER"` is their mean, so
    // that their shape stays: scaled about the origin, the restraints would
    // hold a solute as much smaller as the cell has become. That of `"ALL"`
    // is the reference itself, with no offset. Without a barostat the
    // offsets are the references.
    size_t count = system.getNumParticles(), centered = 0;
    double mean[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i != count; ++i)
      if (system.restraintConstants[i] > 0.0 &&
          system.restraintScaling[i] == ReferenceScaling::Center) {
        ++centered;
        for (int c = 0; c != 3; ++c)
          mean[c] += system.referencePositions[3 * i + c];
      }
    for (int c = 0; c != 3; ++c)
      mean[c] = centered ? mean[c] / centered : 0.0;
    std::vector<double> centers(3 * count, 0.0);
    if (control.barostat)
      for (size_t i = 0; i != count; ++i)
        for (int c = 0; c != 3; ++c)
          centers[3 * i + c] =
              system.restraintScaling[i] == ReferenceScaling::All
                  ? system.referencePositions[3 * i + c]
                  : mean[c];
    for (int c = 0; c != 3; ++c) {
      Program::Field reference;
      reference.name = std::string("rest_") + "xyz"[c];
      for (size_t i = 0; i != count; ++i)
        reference.values.push_back(system.referencePositions[3 * i + c] -
                                   centers[3 * i + c]);
      context.getFields().push_back(std::move(reference));
    }
    for (int c = 0; control.barostat && c != 3; ++c) {
      Program::Field center;
      center.name = std::string("rest_c") + "xyz"[c];
      for (size_t i = 0; i != count; ++i)
        center.values.push_back(centers[3 * i + c]);
      context.getFields().push_back(std::move(center));
    }
  }

}

void RestraintTermBuilder::emitCorrection(TermEmission &context,
                             StringRef scaleName, StringRef indent, StringRef x,
                             StringRef prefix, StringRef f,
                             StringRef fResult, StringRef u,
                             StringRef uResult, StringRef w,
                             StringRef wResult) {
  llvm::raw_ostream &os = context.getOutput();
  std::string fields = (prefix + "rest_k, " + prefix + "rest_x, " + prefix +
                        "rest_y, " + prefix + "rest_z")
                           .str();
  std::string types = "!real, !real, !real, !real";
  std::string arguments = "%k_i: f64, %rx_i: f64, %ry_i: f64, %rz_i: f64";
  if (context.getControl().barostat) {
    fields += (", " + prefix + "rest_cx, " + prefix + "rest_cy, " + prefix +
               "rest_cz")
                  .str();
    types += ", !real, !real, !real";
    arguments += ", %rcx_i: f64, %rcy_i: f64, %rcz_i: f64";
  }
  // d = x − x_ref, and k d. Under a barostat the fields hold the offsets o
  // of the references from their centers c, and the center follows the cell
  // while the offset stays, x_ref = s ⊙ c + o (D124).
  auto emitOffset = [&](StringRef inner) {
    std::string reference = "%ref_i";
    os << inner << reference << " = vector.from_elements %rx_i, %ry_i, %rz_i "
                   ": vector<3xf64>\n";
    if (context.getControl().barostat) {
      os << inner << "%rc_i = vector.from_elements %rcx_i, %rcy_i, %rcz_i "
                     ": vector<3xf64>\n"
         << inner << "%rcs_i = arith.mulf " << scaleName
         << ", %rc_i : vector<3xf64>\n"
         << inner << "%refs_i = arith.addf %rcs_i, %ref_i : vector<3xf64>\n";
      reference = "%refs_i";
    }
    os << inner << "%d_i = arith.subf %x_i, " << reference
       << " : vector<3xf64>\n"
       << inner << "%kb_i = vector.broadcast %k_i : f64 to vector<3xf64>\n"
       << inner << "%kd_i = arith.mulf %kb_i, %d_i : vector<3xf64>\n";
  };
  std::string inner = (indent + "  ").str();
  os << indent << fResult << " = md.map_particles gather(" << x << ", " << f
     << ", " << fields << " : !vec, !vec, " << types << ") {\n"
     << indent << "^bb0(%x_i: vector<3xf64>, %f_i: vector<3xf64>, "
     << arguments << "):\n";
  emitOffset(inner);
  os << inner << "%two_i = arith.constant dense<2.0> : vector<3xf64>\n"
     << inner << "%pull_i = arith.mulf %two_i, %kd_i : vector<3xf64>\n"
     << inner << "%fr_i = arith.subf %f_i, %pull_i : vector<3xf64>\n"
     << inner << "md.yield %fr_i : vector<3xf64>\n"
     << indent << "} : !vec\n";
  if (u.empty() && w.empty())
    return;
  // Σ k d ⊙ d, whose sum is the energy and whose elements times −2 are the
  // diagonal of the virial. Under a barostat the virial is Σ F ⊙ (x − s ⊙ c)
  // = −2 Σ k d ⊙ (d + o), the derivative of the energy when the positions
  // and the centers of the references scale and their offsets do not.
  std::string sum = ((u.empty() ? wResult : uResult) + "_parts").str();
  os << indent << sum << " = md.sum_particles gather(" << x << ", " << fields
     << " : !vec, " << types << ") {\n"
     << indent << "^bb0(%x_i: vector<3xf64>, " << arguments << "):\n";
  emitOffset(inner);
  os << inner << "%kdd_i = arith.mulf %kd_i, %d_i : vector<3xf64>\n"
     << inner << "md.yield %kdd_i : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n";
  std::string virialSum = sum;
  if (!w.empty() && context.getControl().barostat) {
    virialSum = (wResult + "_vparts").str();
    os << indent << virialSum << " = md.sum_particles gather(" << x << ", "
       << fields << " : !vec, " << types << ") {\n"
       << indent << "^bb0(%x_i: vector<3xf64>, " << arguments << "):\n";
    emitOffset(inner);
    os << inner << "%dpo_i = arith.addf %d_i, %ref_i : vector<3xf64>\n"
       << inner << "%kdo_i = arith.mulf %kd_i, %dpo_i : vector<3xf64>\n"
       << inner << "md.yield %kdo_i : vector<3xf64>\n"
       << indent << "} : vector<3xf64>\n";
  }
  if (!u.empty())
    os << indent << uResult << "_r = vector.reduction <add>, " << sum
       << " : vector<3xf64> into f64\n"
       << indent << uResult << " = arith.addf " << u << ", " << uResult
       << "_r : f64\n";
  if (w.empty())
    return;
  std::string zero = (wResult + "_zero").str();
  os << indent << zero << " = arith.constant 0.0 : f64\n"
     << indent << wResult << "_m2 = arith.constant -2.0 : f64\n";
  std::string diagonal[3];
  for (int c = 0; c != 3; ++c) {
    diagonal[c] = (wResult + "_d" + std::to_string(c)).str();
    os << indent << diagonal[c] << "_s = vector.extract " << virialSum
       << "[" << c << "] : f64 from vector<3xf64>\n"
       << indent << diagonal[c] << " = arith.mulf " << diagonal[c] << "_s, "
       << wResult << "_m2 : f64\n";
  }
  os << indent << wResult << "_r = vector.from_elements " << diagonal[0]
     << ", " << zero << ", " << zero << ", " << zero << ", " << diagonal[1]
     << ", " << zero << ", " << zero << ", " << zero << ", " << diagonal[2]
     << " : vector<9xf64>\n"
     << indent << wResult << " = arith.addf " << w << ", " << wResult
     << "_r : vector<9xf64>\n";
}

// Positional-restraint fields and contributions (D359, D361, #683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>

using namespace mdir::driver;
using namespace mdir::driver::builder;

struct RestraintIRConfig {
  bool barostat = false, energy = false, virial = false;
  size_t fields = 0, scaleSlot = 0, energySlot = 0, virialSlot = 0;
};

static TermIRValues buildRestraintCorrection(mlir::OpBuilder &builder, mlir::ValueRange values, const RestraintIRConfig &config) {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        auto f64 = builder.getF64Type();
        auto vec = VectorType::get({3}, f64);
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
        auto sub = [&](Value a, Value b) -> Value {
          return arith::SubFOp::create(builder, loc, a, b);
        };
        // x_ref = s * c + o: only reference centers follow the cell (D124).
        // Return the unscaled offset, d = x-x_ref, and k*d in their old order.
        auto offset = [&](Block *block, unsigned base) -> std::array<Value, 3> {
          Value reference = vector::FromElementsOp::create(
              builder, loc, vec,
              ValueRange{block->getArgument(base + 1),
                         block->getArgument(base + 2),
                         block->getArgument(base + 3)});
          Value scaledReference = reference;
          if (config.barostat) {
            Value center = vector::FromElementsOp::create(
                builder, loc, vec,
                ValueRange{block->getArgument(base + 4),
                           block->getArgument(base + 5),
                           block->getArgument(base + 6)});
            Value scaled = mul(values[config.scaleSlot], center);
            scaledReference = add(scaled, reference);
          }
          Value d = sub(block->getArgument(0), scaledReference);
          Value kb = vector::BroadcastOp::create(builder, loc, vec,
                                                 block->getArgument(base));
          Value kd = mul(kb, d);
          return {reference, d, kd};
        };
        auto force = mdir::md::MapParticlesOp::create(
            builder, loc, values[1].getType(),
            values.take_front(2 + config.fields));
        {
          OpBuilder::InsertionGuard guard(builder);
          llvm::SmallVector<Type> types(2 + config.fields, f64);
          types[0] = types[1] = vec;
          llvm::SmallVector<Location> locations(types.size(), loc);
          Block *block =
              builder.createBlock(&force.getKernel(), {}, types, locations);
          auto terms = offset(block, 2);
          Value two = arith::ConstantOp::create(
              builder, loc, DenseElementsAttr::get(vec, 2.0));
          Value pull = mul(two, terms[2]),
                result = sub(block->getArgument(1), pull);
          mdir::md::YieldOp::create(builder, loc, ValueRange{result});
        }
        TermIRValues result = {force.getResult()};
        if (!config.energy && !config.virial)
          return result;
        llvm::SmallVector<Value> fields = {values[0]};
        llvm::append_range(fields, values.slice(2, config.fields));
        auto sumParts = [&](bool virial) -> Value {
          auto sum =
              mdir::md::SumParticlesOp::create(builder, loc, vec, fields);
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types(1 + config.fields, f64);
            types[0] = vec;
            llvm::SmallVector<Location> locations(types.size(), loc);
            Block *block =
                builder.createBlock(&sum.getKernel(), {}, types, locations);
            auto terms = offset(block, 1);
            Value displacement = terms[1];
            if (virial)
              displacement = add(displacement, terms[0]);
            Value product = mul(terms[2], displacement);
            mdir::md::YieldOp::create(builder, loc, ValueRange{product});
          }
          return sum.getResult();
        };
        Value sum = sumParts(false), virialSum = sum;
        if (config.virial && config.barostat)
          virialSum = sumParts(true);
        if (config.energy) {
          Value energy = vector::ReductionOp::create(
              builder, loc, vector::CombiningKind::ADD, sum);
          Value total = add(values[config.energySlot], energy);
          result.push_back(energy);
          result.push_back(total);
        }
        if (config.virial) {
          Value zero = constant(0.0), minusTwo = constant(-2.0);
          std::array<Value, 3> diagonal;
          for (int c = 0; c != 3; ++c) {
            Value element =
                vector::ExtractOp::create(builder, loc, virialSum, c);
            diagonal[c] = mul(element, minusTwo);
          }
          auto matrix = VectorType::get({9}, f64);
          Value correction = vector::FromElementsOp::create(
              builder, loc, matrix,
              ValueRange{diagonal[0], zero, zero, zero, diagonal[1], zero, zero,
                         zero, diagonal[2]});
          result.push_back(add(values[config.virialSlot], correction));
        }
        return result;
}

namespace mdir::driver::builder {
class RestraintTermBuilder final : public TermBuilder,
                                   public TermPostEvaluation {
public:
  RestraintTermBuilder() { roles.postEvaluation = this; }
  int parameterRank() const override { return -1; }
  void collectParameters(TermParameters &context) const override;
  bool hasForceCorrection(const TermInputs &context) const override {
    return !context.getSystem().restraintConstants.empty();
  }
  TermNativeResults build(const TermInputs &context, mlir::OpBuilder &builder,
                          const TermNativeEvaluation &point) const override {
    RestraintIRConfig config;
    config.barostat = bool(context.getControl().barostat);
    config.energy = bool(point.energy);
    config.virial = bool(point.virial);
    TermIRValues values{point.positions, point.forces};
    for (StringRef name : {"rest_k", "rest_x", "rest_y", "rest_z"})
      values.push_back(point.field(name));
    config.fields = 4;
    if (config.barostat) {
      for (StringRef name : {"rest_cx", "rest_cy", "rest_cz"})
        values.push_back(point.field(name));
      config.fields = 7;
      config.scaleSlot = values.size();
      values.push_back(point.scale);
    }
    if (config.energy) {
      config.energySlot = values.size();
      values.push_back(point.energy);
    }
    if (config.virial) {
      config.virialSlot = values.size();
      values.push_back(point.virial);
    }
    auto result = buildRestraintCorrection(builder, values, config);
    return {result[0], config.energy ? result[2] : mlir::Value{},
            config.virial ? result[config.energy ? 3 : 1] : mlir::Value{}};
  }
  void emitForceCorrection(TermEmission &context,
                           const TermCorrection &point) const override {
    if (!hasForceCorrection(context))
      return;
    emitCorrection(context, point.scale, point.indent, point.x, point.prefix,
                   point.f, point.fResult, point.u, point.uResult, point.w,
                   point.wResult);
    // The initial term report consumes this correction's energy. A hosted
    // evaluation uses a different result name; force-only frame requests do
    // not replace the energy that the report needs. This adapter is per build.
    if (!point.uResult.empty())
      storedEnergy = (point.uResult + "_r").str();
  }
  unsigned getStoredTermCount(const TermInputs &context) const override {
    return hasForceCorrection(context) ? 1 : 0;
  }
  void emitStoredTerms(TermEmission &context, StringRef type,
                       int &index) const override {
    if (!hasForceCorrection(context))
      return;
    if (storedEnergy.empty())
      llvm::report_fatal_error("the builder has no recorded restraint energy");
    int column = index++;
    context.emitIRFragment(
        "  ", {}, {}, {storedEnergy, "%terms"}, {"f64", type},
        [column](mlir::OpBuilder &builder,
                 mlir::ValueRange values) -> TermIRValues {
          auto loc = builder.getUnknownLoc();
          mlir::Value place =
              mlir::arith::ConstantIndexOp::create(builder, loc, column);
          mlir::memref::StoreOp::create(builder, loc, values[0], values[1],
                                        mlir::ValueRange{place});
          return {};
        });
  }
private:
  mutable std::string storedEnergy;
  static void emitCorrection(TermEmission &, StringRef scaleName, StringRef indent,
      StringRef x, StringRef prefix, StringRef f, StringRef fResult,
      StringRef u, StringRef uResult, StringRef w, StringRef wResult);
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createRestraintTermBuilder() {
  return std::make_unique<RestraintTermBuilder>();
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
  RestraintIRConfig config;
  config.barostat = bool(context.getControl().barostat);
  config.energy = !u.empty();
  config.virial = !w.empty();
  std::vector<std::string> names = {x.str(), f.str()}, types = {"!vec", "!vec"};
  auto operand = [&](std::string name, std::string type) {
    size_t slot = names.size();
    names.push_back(std::move(name));
    types.push_back(std::move(type));
    return slot;
  };
  for (StringRef name : {"rest_k", "rest_x", "rest_y", "rest_z"})
    operand((prefix + name).str(), "!real");
  config.fields = 4;
  if (config.barostat) {
    for (StringRef name : {"rest_cx", "rest_cy", "rest_cz"})
      operand((prefix + name).str(), "!real");
    config.fields = 7;
    config.scaleSlot = operand(scaleName.str(), "vector<3xf64>");
  }
  if (config.energy)
    config.energySlot = operand(u.str(), "f64");
  if (config.virial)
    config.virialSlot = operand(w.str(), "vector<9xf64>");
  std::vector<std::string> results = {fResult.str()}, resultTypes = {"!vec"};
  if (config.energy) {
    results.push_back((uResult + "_r").str());
    results.push_back(uResult.str());
    resultTypes.push_back("f64");
    resultTypes.push_back("f64");
  }
  if (config.virial) {
    results.push_back(wResult.str());
    resultTypes.push_back("vector<9xf64>");
  }
  llvm::SmallVector<StringRef> in, inTypes, out, outTypes;
  for (const auto &s : names)
    in.push_back(s);
  for (const auto &s : types)
    inTypes.push_back(s);
  for (const auto &s : results)
    out.push_back(s);
  for (const auto &s : resultTypes)
    outTypes.push_back(s);
  context.emitIRFragment(
      indent, out, outTypes, in, inTypes,
      [config](mlir::OpBuilder &builder,
               mlir::ValueRange values) -> TermIRValues {
        return buildRestraintCorrection(builder, values, config);
      });
}

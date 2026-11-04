// D230 tuple-seed coloring and native parameter perturbations, shared by terms.
#include "BuilderImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

void mdir::driver::builder::collectTupleGradient(
    TermParameters &context, Program::TupleSet &set, unsigned tunable,
    StringRef parameter) {
  unsigned place = 0, count = 0;
  std::vector<double> colors;
  for (unsigned candidate = 0; candidate != set.arity; ++candidate) {
    std::map<unsigned, unsigned> seen;
    std::vector<double> numbers;
    unsigned most = 0;
    for (size_t t = 0; t != set.size(); ++t) {
      unsigned number = seen[set.members[t * set.arity + candidate]]++;
      numbers.push_back(number);
      most = std::max(most, number + 1);
    }
    if (candidate == 0 || most < count) {
      place = candidate;
      count = most;
      colors = std::move(numbers);
    }
  }
  BuildKept::TupleGradient gradient;
  gradient.tunable = tunable;
  gradient.tupleSet = set.name;
  gradient.parameter = parameter.str();
  gradient.place = place;
  gradient.count = count;
  gradient.first = static_cast<unsigned>(context.getGradientFields().size());
  for (unsigned c = 0; c != count; ++c) {
    Program::GradientField field;
    field.tunable = tunable;
    for (size_t t = 0; t != set.size(); ++t)
      if (colors[t] == c)
        field.sites.push_back(
            {static_cast<uint32_t>(t), set.members[t * set.arity + place]});
    context.getGradientFields().push_back(std::move(field));
  }
  size_t field = addTupleField(set, ("tgc_" + parameter).str());
  set.fields[field].values = std::move(colors);
  context.getMutableTupleGradients().push_back(std::move(gradient));
}

mlir::Value mdir::driver::builder::buildTupleGradientParameter(
    mlir::OpBuilder &builder, mlir::Value parameter, mlir::Value selector,
    mlir::ValueRange gathered, unsigned arity,
    const BuildKept::TupleGradient &gradient) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  auto constant = [&](double value) -> Value {
    return arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(value));
  };
  Value zero = constant(0.0);
  for (unsigned c = 0; c != gradient.count; ++c) {
    Value number = constant(static_cast<double>(c));
    Value equal = arith::CmpFOp::create(
        builder, loc, arith::CmpFPredicate::OEQ, selector, number);
    Value derivative = gathered[c * arity + gradient.place];
    Value selected = arith::SelectOp::create(builder, loc, equal, derivative, zero);
    parameter = arith::AddFOp::create(builder, loc, parameter, selected);
  }
  return parameter;
}

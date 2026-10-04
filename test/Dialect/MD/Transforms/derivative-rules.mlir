// All scalar operations emitted by expressions and table interpolation
// have a rule or an explicit structural zero (D[ad-rules]).
// RUN: mdir-opt %s --md-check-derivative-coverage --md-differentiate -o /dev/null
!vec = !md.field<@atoms, 3 x f64>
md.particle_set @atoms
md.potential @rules(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2 : i32
  %square = arith.mulf %a, %a : f64
  %positive = arith.addf %one, %square : f64
  %absf = math.absf %a : f64
  %floor = math.floor %a : f64
  %ceil = math.ceil %a : f64
  %sqrt = math.sqrt %positive : f64
  %exp = math.exp %a : f64
  %log = math.log %positive : f64
  %sin = math.sin %a : f64
  %cos = math.cos %a : f64
  %tan = math.tan %a : f64
  %asin = math.asin %a : f64
  %acos = math.acos %a : f64
  %atan = math.atan %a : f64
  %sinh = math.sinh %a : f64
  %cosh = math.cosh %a : f64
  %tanh = math.tanh %a : f64
  %erf = math.erf %a : f64
  %erfc = math.erfc %a : f64
  %addf = arith.addf %a, %one : f64
  %subf = arith.subf %a, %one : f64
  %mulf = arith.mulf %a, %one : f64
  %divf = arith.divf %a, %one : f64
  %minimumf = arith.minimumf %a, %one : f64
  %maximumf = arith.maximumf %a, %one : f64
  %neg = arith.negf %a : f64
  %pow = math.powf %positive, %a : f64
  %powi = math.fpowi %a, %two : f64, i32
  %atan2 = math.atan2 %a, %one : f64
  %condition = arith.cmpf olt, %a, %one : f64
  %select = arith.select %condition, %a, %one : f64
  %fpint = arith.fptosi %floor : f64 to i32
  %addint = arith.addi %fpint, %two : i32
  %subint = arith.subi %addint, %two : i32
  %mulint = arith.muli %subint, %two : i32
  %remainder = arith.remsi %mulint, %two : i32
  %cmpint = arith.cmpi slt, %mulint, %two : i32
  %or = arith.ori %cmpint, %condition : i1
  %and = arith.andi %or, %condition : i1
  %xor = arith.xori %and, %condition : i1
  %flag = arith.uitofp %and : i1 to f64
  %intfloat = arith.sitofp %mulint : i32 to f64
  %index = arith.index_cast %mulint : i32 to index
  %broadcast = vector.broadcast %a : f64 to vector<3xf64>
  %reduced = vector.reduction <add>, %broadcast : vector<3xf64> into f64
  %assembled = vector.from_elements %a, %one, %square : vector<3xf64>
  %extract = vector.extract %assembled[0] : f64 from vector<3xf64>
  %extractb = vector.extract %broadcast[1] : f64 from vector<3xf64>
  %total0 = arith.addf %absf, %floor : f64
  %total1 = arith.addf %total0, %ceil : f64
  %total2 = arith.addf %total1, %sqrt : f64
  %total3 = arith.addf %total2, %exp : f64
  %total4 = arith.addf %total3, %log : f64
  %total5 = arith.addf %total4, %sin : f64
  %total6 = arith.addf %total5, %cos : f64
  %total7 = arith.addf %total6, %tan : f64
  %total8 = arith.addf %total7, %asin : f64
  %total9 = arith.addf %total8, %acos : f64
  %total10 = arith.addf %total9, %atan : f64
  %total11 = arith.addf %total10, %sinh : f64
  %total12 = arith.addf %total11, %cosh : f64
  %total13 = arith.addf %total12, %tanh : f64
  %total14 = arith.addf %total13, %erf : f64
  %total15 = arith.addf %total14, %erfc : f64
  %total16 = arith.addf %total15, %addf : f64
  %total17 = arith.addf %total16, %subf : f64
  %total18 = arith.addf %total17, %mulf : f64
  %total19 = arith.addf %total18, %divf : f64
  %total20 = arith.addf %total19, %minimumf : f64
  %total21 = arith.addf %total20, %maximumf : f64
  %total22 = arith.addf %total21, %neg : f64
  %total23 = arith.addf %total22, %pow : f64
  %total24 = arith.addf %total23, %powi : f64
  %total25 = arith.addf %total24, %atan2 : f64
  %total26 = arith.addf %total25, %select : f64
  %total27 = arith.addf %total26, %flag : f64
  %total28 = arith.addf %total27, %intfloat : f64
  %total29 = arith.addf %total28, %extract : f64
  %total30 = arith.addf %total29, %extractb : f64
  md.return %total30 : f64
}
md.function @request(%x: !vec, %cell: !md.cell, %a: f64) -> f64 {
  %d = md.evaluate @rules(%x, %cell, %a) request [derivative(2)] : (!vec, !md.cell, f64) -> f64
  md.return %d : f64
}

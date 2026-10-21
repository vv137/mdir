// A write in a stage reduction payload is rejected by the stage purity
// contract before either row-sum strategy can clone the payload.
// RUN: not mdir-opt %s --md-exec-lower-stage-bodies='short-reductions=true' 2>&1 | FileCheck %s
// CHECK: 'memref.store' op in the forward body of the stage 'write', which must be pure: the op has memory effects

md.particle_set @atoms
md.stage @write forward(@write.f) : (vector<3xf64>) -> f64

memref.global "private" @scratch : memref<1xf32> = dense<0.0>

#id = affine_map<(r, f) -> (r, f)>
#row = affine_map<(r, f) -> (r)>

func.func @write.f(%x: tensor<8x3xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %scratch = memref.get_global @scratch : memref<1xf32>
  %empty = tensor.empty() : tensor<8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %sum = linalg.generic {indexing_maps = [#id, #row],
                         iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<8x3xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%a: f32, %o: f32):
    memref.store %a, %scratch[%c0] : memref<1xf32>
    %next = arith.addf %o, %a : f32
    linalg.yield %next : f32
  } -> tensor<8xf32>
  return %sum : tensor<8xf32>
}

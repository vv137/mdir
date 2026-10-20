// An effectful stage payload is rejected before either reduction strategy
// can clone it into a GPU launch. The register strategy must not bypass the
// stage's purity contract.
// RUN: not mdir-opt %s --md-exec-lower-stage-bodies='short-reductions=true' 2>&1 | FileCheck %s
// CHECK: 'func.call' op in the forward body of the stage 'effect', which must be pure: the op has memory effects

md.particle_set @atoms
md.stage @effect forward(@effect.f) : (vector<3xf64>) -> f64

#id = affine_map<(r, f) -> (r, f)>
#row = affine_map<(r, f) -> (r)>

func.func private @touch(f32)
func.func @effect.f(%x: tensor<8x3xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %sum = linalg.generic {indexing_maps = [#id, #row],
                         iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<8x3xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%a: f32, %o: f32):
    func.call @touch(%a) : (f32) -> ()
    %next = arith.addf %o, %a : f32
    linalg.yield %next : f32
  } -> tensor<8xf32>
  return %sum : tensor<8xf32>
}

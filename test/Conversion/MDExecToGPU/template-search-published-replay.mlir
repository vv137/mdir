// REQUIRES: cuda-toolkit
// RUN: mdir-opt %s '--test-md-template-search=reuse-published-groups force-general columns-x=1 columns-y=1' --convert-md-exec-to-gpu | FileCheck %s
// The actual owner supplies exact reaches; do not recompute its group plan.
md.particle_set @atoms
!list = !mdrt.segmented_pairs<@atoms>
func.func private @search_seed(!list, memref<?x3xf64, 1>, memref<?xi32, 1>, vector<3xf64>, i1, i1, vector<3xf64>)
// CHECK-DAG: func.func private @__mdir_template_search_general_orthorhombic
// CHECK-DAG: func.func private @search_seed
// CHECK: call @mdrtTemplatePairsGroupView
// CHECK: call @mdrtTemplatePairsSetImagePath

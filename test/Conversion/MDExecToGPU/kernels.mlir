// The storage form on a device, lowered to kernels.
//
// RUN: mdir-opt %s --convert-md-exec-to-gpu | FileCheck %s
// RUN: mdir-opt %s --convert-md-exec-to-gpu="block-size=64" \
// RUN: | FileCheck %s --check-prefix=SMALL

md.particle_set @atoms

// One thread per particle, in blocks of 128. The threads beyond the last
// particle do nothing.
//
// CHECK-LABEL: func.func @kick(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[F:[a-z0-9]+]]: memref<?x3xf32, 1>, %[[DT:[a-z0-9]+]]: f64)
// SMALL-LABEL: func.func @kick(
func.func @kick(%v: memref<?x3xf64, 1>, %f: memref<?x3xf32, 1>, %dt: f64) {
  // CHECK:      %[[N:[a-z0-9_]+]] = memref.dim %[[V]],
  // CHECK:      %[[BLOCK:[a-z0-9_]+]] = arith.constant 128 : index
  // CHECK:      %[[PADDED:[0-9]+]] = arith.addi %[[N]], %{{[a-z0-9_]+}}
  // CHECK:      %[[GRID:[0-9]+]] = arith.divui %[[PADDED]], %{{[a-z0-9_]+}}
  // CHECK:      gpu.launch blocks(%[[BX:[a-z0-9]+]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}}) in (%{{[a-z0-9_]+}} = %[[GRID]],
  // CHECK-SAME:   threads(%[[TX:[a-z0-9]+]], %{{[a-z0-9]+}}, %{{[a-z0-9]+}}) in (%{{[a-z0-9_]+}} = %[[BLOCK]],
  // CHECK:        %[[BASE:[0-9]+]] = arith.muli %[[BX]],
  // CHECK:        %[[I:[0-9]+]] = arith.addi %[[BASE]], %[[TX]]
  // CHECK:        %[[INSIDE:[0-9]+]] = arith.cmpi ult, %[[I]], %[[N]]
  // CHECK:        scf.if %[[INSIDE]] {
  // CHECK:          memref.load %[[V]][%[[I]],
  // CHECK:          memref.load %[[F]][%[[I]],
  // CHECK:          arith.extf %{{[0-9]+}} : vector<3xf32> to vector<3xf64>
  // CHECK:          vector.broadcast %[[DT]]
  // CHECK:          memref.store %{{[0-9]+}}, %[[V]][%[[I]],
  // CHECK:        gpu.terminator
  // CHECK-NOT:  md_exec
  // SMALL:      arith.constant 64 : index
  md_exec.particle_for ins(%v, %f : memref<?x3xf64, 1>, memref<?x3xf32, 1>)
      outs(%v : memref<?x3xf64, 1>) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf32>):
    %wide = arith.extf %f_i : vector<3xf32> to vector<3xf64>
    %s = vector.broadcast %dt : f64 to vector<3xf64>
    %change = arith.mulf %s, %wide : vector<3xf64>
    %new = arith.addf %v_i, %change : vector<3xf64>
    md_exec.yield %new : vector<3xf64>
  }
  return
}

// A global sum: the kernel stores the contribution of each particle, a
// second kernel adds up chunks of particles, a third adds up the
// chunks, and the host reads the result. The value on the device and the
// buffer of the host that takes it are allocated where the function begins.
// A vector from outside the kernel enters it as its elements.
//
// CHECK-LABEL: func.func @forces(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[F:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[A:[a-z0-9]+]]: memref<?xf64, 1>, %[[B:[a-z0-9]+]]: memref<?xf64, 1>, %[[BOX:[a-z0-9]+]]: vector<3xf64>, %{{[a-z0-9]+}}: index)
func.func @forces(%x: memref<?x3xf64, 1>, %f: memref<?x3xf64, 1>,
                  %a: memref<?xf64, 1>, %b: memref<?xf64, 1>,
                  %cell: !md.cell, %n: index) -> f64 {
  // CHECK:      %[[CELL:[a-z0-9_]+]] = gpu.alloc () : memref<1xf64, 1>
  // CHECK:      %[[HOST:[a-z0-9_]+]] = memref.alloca() : memref<1xf64>
  // CHECK:      %[[COUNTS:[a-z0-9_]+]] = gpu.alloc (%{{[a-z0-9_]+}}) : memref<?xi32, 1>
  // CHECK:      %[[INDEX:[a-z0-9_]+]] = gpu.alloc (%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}) : memref<?x?xi32, 1>
  // CHECK:      %[[REFERENCE:[a-z0-9_]+]] = gpu.alloc (%{{[a-z0-9_]+}}) : memref<?x3xf64, 1>
  %nl0 = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64, 1>)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>

  // CHECK:      call @mdrt_gpu_build_neighbors_matrix(%[[X]], %[[BOX]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[COUNTS]], %[[INDEX]])
  // CHECK:      gpu.memcpy async [%{{[0-9]+}}] %[[REFERENCE]], %[[X]]
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always)
      : !mdrt.neighbors<@atoms>, memref<?x3xf64, 1>

  // The minimum image takes one over the edge lengths, computed once on
  // the host.
  //
  // CHECK:      %[[INVERSE:[0-9]+]] = arith.divf %{{[a-z0-9_]+}}, %[[BOX]] : vector<3xf64>
  // CHECK:      arith.divui
  // CHECK-DAG:  %[[IX:[0-9]+]] = vector.extract %[[INVERSE]][0]
  // CHECK-DAG:  %[[LX:[0-9]+]] = vector.extract %[[BOX]][0]
  // Threads in groups of 16 share the row of a particle and sum over the
  // group by shuffles; the first thread of a group writes.
  //
  // CHECK:      gpu.launch
  // CHECK-DAG:    %[[INSIDE:[0-9]+]] = vector.from_elements %[[IX]],
  // CHECK-DAG:    %[[EDGES:[0-9]+]] = vector.from_elements %[[LX]],
  // CHECK:        arith.divui
  // CHECK:        arith.remui
  // CHECK:        memref.load %[[COUNTS]][
  // CHECK:        scf.for
  // CHECK:          %[[IMAGES:[0-9]+]] = arith.mulf %{{[0-9]+}}, %[[INSIDE]]
  // CHECK:          %[[NEAREST:[0-9]+]] = math.roundeven %[[IMAGES]]
  // CHECK:          arith.mulf %[[NEAREST]], %[[EDGES]]
  // CHECK:        gpu.shuffle xor
  // CHECK:        scf.if
  // CHECK:          memref.store %{{[0-9]+}}, %[[F]][
  // CHECK:        scf.if
  // CHECK:          memref.store %{{[0-9]+}}, %[[A]][
  // CHECK:        gpu.terminator
  //
  // CHECK:      gpu.launch
  // CHECK:          scf.for
  // CHECK:            memref.load %[[A]][
  // CHECK:          memref.store %{{[0-9]+}}, %[[B]][
  // CHECK:        gpu.terminator
  //
  // CHECK:      gpu.launch
  // CHECK:        scf.for
  // CHECK:          memref.load %[[B]][
  // CHECK:        memref.store %{{[0-9]+}}, %[[CELL]][
  // CHECK:        gpu.terminator
  //
  // CHECK:      gpu.memcpy async [%{{[0-9]+}}] %[[HOST]], %[[CELL]]
  // CHECK:      %[[SUM:[0-9]+]] = memref.load %[[HOST]][
  // CHECK:      %[[U:[0-9]+]] = arith.addf %{{[a-z0-9_]+}}, %[[SUM]]
  // CHECK:      return %[[U]]
  %u0 = arith.constant 0.0 : f64
  %u = md_exec.pair_for %nl, %x, %cell outs(%f : memref<?x3xf64, 1>)
      reduce(%u0 : f64) scratch(%a, %b : memref<?xf64, 1>, memref<?xf64, 1>)
      cutoff(1.5) weights [0.5] overwrite [true]
      policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_exec.yield %d, %r2 : vector<3xf64>, f64
  } : !mdrt.neighbors<@atoms>, memref<?x3xf64, 1> -> f64
  return %u : f64
}

// A constant from outside the kernel is a constant of the kernel: the
// kernel knows the exponent of the power. A loop of the host that launches
// kernels releases the stack at the end of every iteration, because the
// arguments of a launch are put on the stack.
//
// CHECK-LABEL: func.func @steps(
func.func @steps(%v: memref<?x3xf64, 1>, %steps: index) {
  // CHECK:      scf.for
  // CHECK-NEXT:   %[[STACK:[0-9]+]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK:        gpu.launch
  // CHECK-DAG:      %[[SCALE:[a-z0-9_]+]] = arith.constant 5.000000e-01 : f64
  // CHECK-DAG:      %[[POWER:[a-z0-9_]+]] = arith.constant 3 : i32
  // CHECK:          scf.if
  // CHECK:            math.fpowi %[[SCALE]], %[[POWER]]
  // CHECK:          gpu.terminator
  // CHECK:        llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  // CHECK-NEXT: }
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %scale = arith.constant 0.5 : f64
  %power = arith.constant 3 : i32
  scf.for %step = %zero to %steps step %one {
    md_exec.particle_for ins(%v : memref<?x3xf64, 1>)
        outs(%v : memref<?x3xf64, 1>) {
    ^bb0(%v_i: vector<3xf64>):
      %cube = math.fpowi %scale, %power : f64, i32
      %s = vector.broadcast %cube : f64 to vector<3xf64>
      %new = arith.mulf %s, %v_i : vector<3xf64>
      md_exec.yield %new : vector<3xf64>
    }
  }
  return
}

// A loop that tells whether the kernel yields true for any particle: a
// thread that yields true sets a flag on the device, and the others write
// nothing. The flag is not set when the function begins, and the host
// clears it where it finds it set. The refresh launches no kernel for its
// test.
//
// CHECK-LABEL: func.func @validity(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[V:[a-z0-9]+]]: memref<?x3xf64, 1>,
// CHECK:         %[[FLAG:[a-z0-9_]+]] = gpu.alloc () : memref<1xi32, 1>
// CHECK:         %[[HOST:[a-z0-9_]+]] = memref.alloca() : memref<1xi32>
// CHECK:         %[[CLEAR:[a-z0-9_]+]] = memref.alloca() : memref<1xi32>
// CHECK:         memref.store %{{[a-z0-9_]+}}, %[[CLEAR]][
// CHECK:         gpu.memcpy async [%{{[0-9]+}}] %[[FLAG]], %[[CLEAR]]
// CHECK:         %[[REFERENCE:[a-z0-9_]+]] = gpu.alloc (%{{[a-z0-9_]+}}) : memref<?x3xf64, 1>
// CHECK:         gpu.launch
// CHECK:           memref.load %[[REFERENCE]][
// CHECK:           %[[FAR:[0-9]+]] = arith.cmpf ugt,
// CHECK:           memref.store %{{[0-9]+}}, %[[X]][
// CHECK:           scf.if %[[FAR]] {
// CHECK:             memref.store %{{[a-z0-9_]+}}, %[[FLAG]][
// CHECK:           gpu.terminator
// CHECK:         gpu.memcpy async [%{{[0-9]+}}] %[[HOST]], %[[FLAG]]
// CHECK:         %[[VALUE:[0-9]+]] = memref.load %[[HOST]][
// CHECK:         %[[SET:[0-9]+]] = arith.cmpi ne, %[[VALUE]],
// CHECK:         scf.if %[[SET]] {
// CHECK:           gpu.memcpy async [%{{[0-9]+}}] %[[FLAG]], %[[CLEAR]]
// CHECK:         %[[MOVED:[0-9]+]] = arith.ori %{{[a-z0-9_]+}}, %[[SET]]
// CHECK-NOT:     gpu.launch
// CHECK:         %[[NEAR:[0-9]+]] = arith.xori %[[MOVED]],
// CHECK:         scf.if
// CHECK:           call @mdrt_gpu_build_neighbors_matrix(

func.func @validity(%x: memref<?x3xf64, 1>, %v: memref<?x3xf64, 1>,
                    %cell: !md.cell, %n: index) {
  %nl = md_exec.empty_neighbors size(%n) positions(memref<?x3xf64, 1>)
      kind(matrix) width(48) : !mdrt.neighbors<@atoms>
  %ref = md_exec.reference_positions %nl
      : !mdrt.neighbors<@atoms> -> memref<?x3xf64, 1>
  %no = arith.constant false
  %moved = md_exec.particle_for
      ins(%x, %v, %ref
          : memref<?x3xf64, 1>, memref<?x3xf64, 1>, memref<?x3xf64, 1>)
      outs(%x : memref<?x3xf64, 1>) reduce(%no : i1) {
  ^bb0(%x_i: vector<3xf64>, %v_i: vector<3xf64>, %r_i: vector<3xf64>):
    %xn = arith.addf %x_i, %v_i : vector<3xf64>
    %d = arith.subf %xn, %r_i : vector<3xf64>
    %sq = arith.mulf %d, %d : vector<3xf64>
    %d2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %limit = arith.constant 0.015625 : f64
    %far = arith.cmpf ugt, %d2, %limit : f64
    md_exec.yield %xn, %far : vector<3xf64>, i1
  } -> i1
  %nl1 = md_exec.refresh_neighbors %nl, %x, %cell moved(%moved)
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(check)
      : !mdrt.neighbors<@atoms>, memref<?x3xf64, 1>
  return
}

// The order of the particles is a call to the template for devices. A
// field in that order is a kernel with one thread for each place.
//
// CHECK-LABEL: func.func @ordered(
// CHECK-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[IDS:[a-z0-9]+]]: memref<?xi32, 1>, %[[ORDER:[a-z0-9]+]]: memref<?xi32, 1>, %[[XS:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[BOX:[a-z0-9]+]]: vector<3xf64>)
// CHECK:         call @mdrt_gpu_spatial_order(%[[X]], %[[BOX]], %{{[a-z0-9_]+}}, %[[IDS]], %[[ORDER]])
// CHECK:         gpu.launch
// CHECK:           scf.if
// CHECK:             %[[FROM:[0-9]+]] = memref.load %[[ORDER]][%[[K:[0-9]+]]]
// CHECK:             %[[J:[0-9]+]] = arith.index_cast %[[FROM]] : i32 to index
// CHECK:             memref.load %[[X]][%[[J]],
// CHECK:             memref.store %{{[0-9]+}}, %[[XS]][%[[K]],
// CHECK:           gpu.terminator
// CHECK-NOT:     md_exec

func.func @ordered(%x: memref<?x3xf64, 1>, %ids: memref<?xi32, 1>,
                   %order: memref<?xi32, 1>, %xs: memref<?x3xf64, 1>,
                   %cell: !md.cell) {
  md_exec.spatial_order %x, %cell, %ids outs(%order : memref<?xi32, 1>)
      width(1.1) : memref<?x3xf64, 1>, memref<?xi32, 1>
  md_exec.permute %x, %order outs(%xs : memref<?x3xf64, 1>)
      : memref<?x3xf64, 1>, memref<?xi32, 1>
  return
}

// A global sum of vectors: the kernel stores the vector of each particle,
// the kernels that add up read and write vectors, and the host reads the
// vector that results.
//
// CHECK-LABEL: func.func @momentum(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[A:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[B:[a-z0-9]+]]: memref<?x3xf64, 1>)
// CHECK:         %[[CELL:[a-z0-9_]+]] = gpu.alloc () : memref<3xf64, 1>
// CHECK:         %[[HOST:[a-z0-9_]+]] = memref.alloca() : memref<3xf64>
// CHECK:         gpu.launch
// CHECK:           memref.load %[[V]][
// CHECK:           memref.store %{{[0-9]+}}, %[[A]][%{{[0-9]+}}, %{{[a-z0-9_]+}}]
// CHECK:         gpu.launch
// CHECK:           scf.for
// CHECK:             memref.load %[[A]][
// CHECK:             arith.addf %{{[a-z0-9]+}}, %{{[0-9]+}} : vector<3xf64>
// CHECK:           memref.store %{{[0-9]+}}, %[[B]][
// CHECK:         gpu.launch
// CHECK:           scf.for
// CHECK:             memref.load %[[B]][
// CHECK:           memref.store %{{[0-9]+}}, %[[CELL]][
// CHECK:         gpu.memcpy async [%{{[0-9]+}}] %[[HOST]], %[[CELL]]
// CHECK:         memref.load %[[HOST]][
// CHECK:         %[[SUM:[0-9]+]] = vector.from_elements
// CHECK:         %[[TOTAL:[0-9]+]] = arith.addf %{{[a-z0-9_]+}}, %[[SUM]] : vector<3xf64>
// CHECK:         return %[[TOTAL]]
func.func @momentum(%v: memref<?x3xf64, 1>, %a: memref<?x3xf64, 1>,
                    %b: memref<?x3xf64, 1>) -> vector<3xf64> {
  %zero = arith.constant dense<0.0> : vector<3xf64>
  %p = md_exec.particle_for ins(%v : memref<?x3xf64, 1>)
      reduce(%zero : vector<3xf64>)
      scratch(%a, %b : memref<?x3xf64, 1>, memref<?x3xf64, 1>) {
  ^bb0(%v_i: vector<3xf64>):
    md_exec.yield %v_i : vector<3xf64>
  } -> vector<3xf64>
  return %p : vector<3xf64>
}

// The sums of one loop are added up together: one kernel for the chunks,
// one for the results of the chunks, and one copy to the host, which takes
// the ten numbers of a number and a vector of nine.
//
// CHECK-LABEL: func.func @together(
// CHECK-SAME:    %[[V:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[A:[a-z0-9]+]]: memref<?xf64, 1>, %[[B:[a-z0-9]+]]: memref<?xf64, 1>, %[[C:[a-z0-9]+]]: memref<?x9xf64, 1>, %[[D:[a-z0-9]+]]: memref<?x9xf64, 1>)
// CHECK:         %[[CELL:[a-z0-9_]+]] = gpu.alloc () : memref<10xf64, 1>
// CHECK:         %[[HOST:[a-z0-9_]+]] = memref.alloca() : memref<10xf64>
// CHECK:         gpu.launch
// CHECK:           memref.store %{{[0-9]+}}, %[[A]][
// CHECK:           memref.store %{{[0-9]+}}, %[[C]][
// CHECK:           gpu.terminator
// CHECK:         gpu.launch
// CHECK:           scf.for
// CHECK:             memref.load %[[A]][
// CHECK:             memref.load %[[C]][
// CHECK:           memref.store %{{[0-9#]+}}, %[[B]][
// CHECK:           memref.store %{{[0-9]+}}, %[[D]][
// CHECK:           gpu.terminator
// CHECK:         gpu.launch
// CHECK:           scf.for
// CHECK:             memref.load %[[B]][
// CHECK:             memref.load %[[D]][
// CHECK:           memref.store %{{[0-9#]+}}, %[[CELL]][
// CHECK:           gpu.terminator
// CHECK-NOT:     gpu.launch
// CHECK:         gpu.memcpy async [%{{[0-9]+}}] %[[HOST]], %[[CELL]]
// CHECK-NOT:     gpu.memcpy
// CHECK:         memref.load %[[HOST]][%{{[a-z0-9_]+}}]
// CHECK:         vector.from_elements
// CHECK:         return
func.func @together(%v: memref<?x3xf64, 1>, %a: memref<?xf64, 1>,
                    %b: memref<?xf64, 1>, %c: memref<?x9xf64, 1>,
                    %d: memref<?x9xf64, 1>) -> (f64, vector<9xf64>) {
  %none = arith.constant 0.0 : f64
  %zero = arith.constant dense<0.0> : vector<9xf64>
  %k, %w = md_exec.particle_for ins(%v : memref<?x3xf64, 1>)
      reduce(%none, %zero : f64, vector<9xf64>)
      scratch(%a, %b, %c, %d : memref<?xf64, 1>, memref<?xf64, 1>,
                               memref<?x9xf64, 1>, memref<?x9xf64, 1>) {
  ^bb0(%v_i: vector<3xf64>):
    %sq = arith.mulf %v_i, %v_i : vector<3xf64>
    %v2 = vector.reduction <add>, %sq : vector<3xf64> into f64
    %nine = vector.broadcast %v2 : f64 to vector<9xf64>
    md_exec.yield %v2, %nine : f64, vector<9xf64>
  } -> f64, vector<9xf64>
  return %k, %w : f64, vector<9xf64>
}

// The template for devices is in the module.
//
// CHECK: func.func private @mdrt_gpu_build_neighbors_matrix(
// CHECK-NOT: md.particle_set

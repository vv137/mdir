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
// second kernel adds up chunks of 256 particles, a third adds up the
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

  // CHECK:      call @mdrt_gpu_build_neighbors_matrix(%[[X]], %[[BOX]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[COUNTS]], %[[INDEX]])
  // CHECK:      gpu.memcpy async [%{{[0-9]+}}] %[[REFERENCE]], %[[X]]
  %nl = md_exec.refresh_neighbors %nl0, %x, %cell
      cutoff(1.5) skin(0.25) cell_width(1.75) policy(always)
      : !mdrt.neighbors<@atoms>, memref<?x3xf64, 1>

  // CHECK:      arith.divui
  // CHECK-NEXT: %[[LX:[0-9]+]] = vector.extract %[[BOX]][0]
  // CHECK-NEXT: %[[LY:[0-9]+]] = vector.extract %[[BOX]][1]
  // CHECK-NEXT: %[[LZ:[0-9]+]] = vector.extract %[[BOX]][2]
  // CHECK-NEXT: gpu.launch
  // CHECK-NEXT:   %[[INSIDE:[0-9]+]] = vector.from_elements %[[LX]], %[[LY]], %[[LZ]]
  // CHECK:        scf.if
  // CHECK:          memref.load %[[COUNTS]][
  // CHECK:          scf.for
  // CHECK:            arith.divf %{{[0-9]+}}, %[[INSIDE]]
  // CHECK:          memref.store %{{[0-9]+}}, %[[F]][
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

// The template for devices is in the module.
//
// CHECK: func.func private @mdrt_gpu_build_neighbors_matrix(
// CHECK-NOT: md.particle_set

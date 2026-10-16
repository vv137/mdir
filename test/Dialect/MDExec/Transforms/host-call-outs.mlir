// RUN: mdir-opt %s --md-exec-fuse-loops | FileCheck %s --check-prefix=FUSE
// RUN: mdir-opt %s --md-exec-fuse-loops --md-exec-assign-storage="memory=device" | FileCheck %s --check-prefix=DEVICE
// RUN: mdir-opt %s --md-exec-fuse-loops --md-exec-assign-storage | FileCheck %s --check-prefix=HOST

// A function of the host that adds to the forces, as a learned potential
// does (D[metatomic-plugin]): it takes them in `outs` and yields the forces
// with its own added. The kick reads that result, so it can neither move
// before the call nor fuse with the loop that computes the other forces,
// and the call is given the buffer of those forces where it is.

!vec  = !md.field<@atoms, 3 x f64>
!real = !md.field<@atoms, f64>

md.particle_set @atoms

func.func private @model(memref<?x3xf64>, memref<?x3xf64>, memref<1xf64>,
                         memref<9xf64>)

// FUSE-LABEL: func.func @kick_after_model(
// FUSE:         %[[FC:[0-9]+]] = md_exec.particle_for
// FUSE:         %[[F:[0-9]+]]:3 = mdrt.host_call @model(%{{[a-z0-9]+}}) outs(%[[FC]] : !md.field<@atoms, 3 x f64>) {in_place}
// FUSE:         %[[V:[0-9]+]] = md_exec.particle_for ins(%{{[a-z0-9]+}}, %[[F]]#0, %{{[a-z0-9]+}} :
// FUSE-NOT:     md_exec.particle_for
// FUSE:         return %[[V]], %[[F]]#0, %[[F]]#1
//
// DEVICE-LABEL: func.func @kick_after_model(
// DEVICE-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[V:[a-z0-9]+]]: memref<?x3xf64, 1>, %[[M:[a-z0-9]+]]: memref<?xf64, 1>
// DEVICE:         md_exec.particle_for ins(%[[X]] : memref<?x3xf64, 1>) outs(%[[FB:[a-z0-9_]+]] : memref<?x3xf64, 1>)
// DEVICE-DAG:     %[[XH:[a-z0-9_]+]] = memref.memory_space_cast %[[X]] : memref<?x3xf64, 1> to memref<?x3xf64>
// DEVICE-DAG:     %[[FH:[a-z0-9_]+]] = memref.memory_space_cast %[[FB]] : memref<?x3xf64, 1> to memref<?x3xf64>
// DEVICE-DAG:     %[[U:[a-z0-9_]+]] = memref.alloca() : memref<1xf64>
// DEVICE-DAG:     %[[W:[a-z0-9_]+]] = memref.alloca() : memref<9xf64>
// DEVICE:         call @model(%[[XH]], %[[FH]], %[[U]], %[[W]])
// DEVICE-NOT:     gpu.memcpy
// DEVICE:         %[[E:[0-9]+]] = memref.load %[[U]]
// DEVICE:         md_exec.particle_for ins(%[[V]], %[[FB]], %[[M]] : memref<?x3xf64, 1>, memref<?x3xf64, 1>, memref<?xf64, 1>)
// DEVICE:         return %{{.*}}, %[[E]]
//
// HOST-LABEL: func.func @kick_after_model(
// HOST-SAME:    %[[X:[a-z0-9]+]]: memref<?x3xf64>
// HOST:         md_exec.particle_for ins(%[[X]] : memref<?x3xf64>) outs(%[[FB:[a-z0-9_]+]] : memref<?x3xf64>)
// HOST:         call @model(%[[X]], %[[FB]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}})
// HOST:         md_exec.particle_for ins(%{{[a-z0-9]+}}, %[[FB]], %{{[a-z0-9]+}} :
func.func @kick_after_model(%x: !vec, %v: !vec, %m: !real, %dt: f64)
    -> (!vec, !vec, f64) {
  %e0 = md_exec.empty : !vec
  %fc = md_exec.particle_for ins(%x : !vec) outs(%e0 : !vec) {
  ^bb0(%x_i: vector<3xf64>):
    %fn = arith.negf %x_i : vector<3xf64>
    md_exec.yield %fn : vector<3xf64>
  } -> !vec
  %f, %u, %w = mdrt.host_call @model(%x) outs(%fc : !vec) {in_place}
      : (!vec) -> (!vec, f64, vector<9xf64>)
  %e1 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f, %m : !vec, !vec, !real)
      outs(%e1 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
    %s  = arith.divf %dt, %m_i : f64
    %sv = vector.broadcast %s : f64 to vector<3xf64>
    %dv = arith.mulf %sv, %f_i : vector<3xf64>
    %vn = arith.addf %v_i, %dv : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  return %v1, %f, %u : !vec, !vec, f64
}

// The same two loops without the call are one.
//
// FUSE-LABEL: func.func @kick_without_model(
// FUSE:         md_exec.particle_for
// FUSE-NOT:     md_exec.particle_for
// FUSE:         return
func.func @kick_without_model(%x: !vec, %v: !vec, %m: !real, %dt: f64)
    -> (!vec, !vec) {
  %e0 = md_exec.empty : !vec
  %f = md_exec.particle_for ins(%x : !vec) outs(%e0 : !vec) {
  ^bb0(%x_i: vector<3xf64>):
    %fn = arith.negf %x_i : vector<3xf64>
    md_exec.yield %fn : vector<3xf64>
  } -> !vec
  %e1 = md_exec.empty : !vec
  %v1 = md_exec.particle_for ins(%v, %f, %m : !vec, !vec, !real)
      outs(%e1 : !vec) {
  ^bb0(%v_i: vector<3xf64>, %f_i: vector<3xf64>, %m_i: f64):
    %s  = arith.divf %dt, %m_i : f64
    %sv = vector.broadcast %s : f64 to vector<3xf64>
    %dv = arith.mulf %sv, %f_i : vector<3xf64>
    %vn = arith.addf %v_i, %dv : vector<3xf64>
    md_exec.yield %vn : vector<3xf64>
  } -> !vec
  return %v1, %f : !vec, !vec
}

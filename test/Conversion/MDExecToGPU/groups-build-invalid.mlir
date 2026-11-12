// RUN: mdir-opt %s --add-groups-build --split-input-file --verify-diagnostics

// A module that defines the build already: the pass replaces only a
// declaration, by which a test calls the build.
// expected-error @+1 {{'func.func' op is the build of the groups already}}
func.func private @mdrt_gpu_build_neighbors_groups() {
  return
}

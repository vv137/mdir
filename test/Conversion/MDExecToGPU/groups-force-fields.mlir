// RUN: not mdir-opt %s --convert-md-exec-to-gpu="arch=sm_89 groups-force-fields=other" 2>&1 | FileCheck %s --check-prefix=LAYOUT
// RUN: not mdir-opt %s --convert-md-exec-to-gpu="arch=sm_89 groups-force-fields=float4" 2>&1 | FileCheck %s --check-prefix=ARCH
// RUN: mdir-opt %s --convert-md-exec-to-gpu="arch=sm_90 groups-force-fields=float4" -o /dev/null
// LAYOUT: error: groups-force-fields takes policy, packed, i32, i64, or float4
// ARCH: error: float4 force reductions require sm_90 or later
module {}

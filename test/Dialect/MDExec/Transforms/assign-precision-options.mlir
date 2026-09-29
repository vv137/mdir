// RUN: not mdir-opt %s --md-exec-assign-precision="mode=half" 2>&1 \
// RUN: | FileCheck %s --check-prefix=MODE
// MODE: expected the mode 'single', 'mixed', or 'double', got 'half'

// RUN: not mdir-opt %s --md-exec-assign-precision="kernel=f16" 2>&1 \
// RUN: | FileCheck %s --check-prefix=ROLE
// ROLE: expected the type 'f32' or 'f64' for the role 'kernel', got 'f16'

func.func @nothing() {
  return
}

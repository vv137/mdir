// A storage choice outside the host/device contract is rejected even when
// the module has no transform to lower. D[fft-value-buffer].
// RUN: not mdir-opt %s '--convert-linear-transform-values=memory=shared' -o /dev/null 2>&1 | FileCheck %s
// CHECK: expected memory 'host' or 'device', got 'shared'
func.func @main() {
  return
}

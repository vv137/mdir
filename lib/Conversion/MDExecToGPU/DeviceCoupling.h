// The coupling step on the device (D[device-coupling], issue #1026).
#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include <functional>

namespace mdir::kernels {
/// What the lowering of a function gives the coupling step: the buffer of
/// the results of its global sums on the device and its copy on the host,
/// a slot of that buffer of its own, a number for a group of readbacks,
/// and the preparation of a launch's values from outside.
struct CouplingResults {
  std::function<std::pair<mlir::Value, mlir::Value>()> cells;
  std::function<int64_t()> slot;
  std::function<int64_t()> readback;
  std::function<void(mlir::gpu::LaunchOp)> bringIn;
  /// The slots that remain in the buffer.
  std::function<int64_t()> room;
  /// A small buffer on the device and its page-locked copy on the host,
  /// for the numbers that the host still reads at a coupling step.
  std::function<std::pair<mlir::Value, mlir::Value>()> scalars;
  static constexpr int64_t numScalars = 8;
};

/// Moves the arithmetic of the coupling step that the host does on the
/// global sums of the step onto the device, ahead of keepReadbacksOnDevice:
/// the factor of stochastic velocity rescaling [Bussi2007] from the draws
/// of the host, which do not depend on the state, and the bath, kept in a
/// slot of the results on the device. Returns the reason that it left the
/// function as it was, or an empty string.
std::string placeCouplingOnDevice(mlir::func::FuncOp function,
                                  CouplingResults &results);
} // namespace mdir::kernels

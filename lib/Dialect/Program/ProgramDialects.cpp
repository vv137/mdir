// One accepted dialect set for program construction, printing, and lowering.
// Extend this list and its libraries in lib/Dialect/Program/CMakeLists.txt
// together; the driver and compiler deliberately consume the same set.
// The driver deliberately parses with the full set, including lowering-only
// dialects such as LLVM and NVVM. Do not narrow its registry independently:
// staged user input must parse wherever a valid program is consumed.
//
// - Dialects: those of the programs that the builder writes and of the
//   templates that the lowerings parse (arith, func, math, memref, scf,
//   vector, gpu, ...), those of a model in stages (linalg, tensor,
//   bufferization), and those that the lowerings produce on the way to
//   LLVM IR (cf, index, affine, ub, omp, nvgpu, NVVM, LLVM, DLTI), and
//   complex, which tensor loads: `gpu-to-llvm` requires the conversion to
//   the LLVM dialect of every dialect loaded, and stops the process
//   without it.
#include "mdir/Dialect/ProgramDialects.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mdir;

/// The dialects of the programs and of their lowerings.
void mdir::insertProgramDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, arith::ArithDialect,
                  bufferization::BufferizationDialect, complex::ComplexDialect,
                  cf::ControlFlowDialect, DLTIDialect, func::FuncDialect,
                  gpu::GPUDialect, index::IndexDialect, linalg::LinalgDialect,
                  LLVM::LLVMDialect, math::MathDialect, memref::MemRefDialect,
                  NVVM::NVVMDialect, nvgpu::NVGPUDialect, omp::OpenMPDialect,
                  scf::SCFDialect, tensor::TensorDialect, ub::UBDialect,
                  vector::VectorDialect, dyn::DynDialect, md::MDDialect,
                  md_exec::MDExecDialect, mdrt::MDRTDialect>();
}

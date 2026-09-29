import os

import lit.formats

from lit.llvm import llvm_config

config.name = "MDIR"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mdir_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in", "lib", "Inputs"]

llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.mdir_tools_dir = os.path.join(config.mdir_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.mdir_tools_dir, config.llvm_tools_dir]
tools = ["mdir-opt", "mlir-opt", "mlir-runner", "FileCheck", "not"]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# The library that provides printF64 and friends to code run by mlir-runner.
config.substitutions.append(
    (
        "%mlir_c_runner_utils",
        os.path.join(
            config.llvm_lib_dir, "libmlir_c_runner_utils" + config.llvm_shlib_ext
        ),
    )
)

# The runtime, and the OpenMP runtime that code lowered through the omp
# dialect needs.
config.substitutions.append(
    (
        "%mdrt",
        os.path.join(
            config.mdir_obj_root, "lib", "libmdrt" + config.llvm_shlib_ext
        ),
    )
)
config.substitutions.append(
    (
        "%openmp",
        os.path.join(config.llvm_lib_dir, "libomp" + config.llvm_shlib_ext),
    )
)

# Tests that run on a GPU need the runtime for NVIDIA GPUs and a device.
# The kernels find the device math library through CUDA_ROOT.
if config.mdir_cuda:
    import subprocess

    try:
        devices = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if devices.returncode == 0 and devices.stdout.strip():
            config.available_features.add("cuda")
    except (OSError, subprocess.SubprocessError):
        pass
    config.environment["CUDA_ROOT"] = config.mdir_cuda_root
# It goes before "%mdrt", which is the beginning of its name.
config.substitutions.insert(
    0,
    (
        "%mdrt_cuda",
        os.path.join(
            config.mdir_obj_root, "lib", "libmdrt_cuda" + config.llvm_shlib_ext
        ),
    ),
)

# The passes of the semantic level, up to code in ordinary functions.
config.substitutions.append(
    (
        "%md_passes",
        "--md-check-exchange --md-differentiate --md-expand-truncation"
        " --md-inline",
    )
)

# The transformations of the execution level. The second set reassociates
# floating-point arithmetic.
md_exec_transforms = "--md-exec-reuse-neighbors --md-exec-fuse-loops --cse"
md_exec_fast_transforms = (
    "--md-exec-reuse-neighbors --md-exec-fuse-loops"
    " --md-exec-simplify-distance --canonicalize --cse"
)
md_exec_lowering = " --md-exec-assign-storage --convert-md-exec-to-loops"
config.substitutions.append(("%md_exec_transforms", md_exec_transforms))
config.substitutions.append(
    ("%md_exec_fast_transforms", md_exec_fast_transforms)
)

# The passes of the execution level, up to loops over buffers, in double
# precision.
config.substitutions.append(
    ("%md_exec_passes", md_exec_transforms + md_exec_lowering)
)
config.substitutions.append(
    (
        "%md_exec_fast_passes",
        md_exec_fast_transforms + md_exec_lowering,
    )
)

# The same for a device, up to ops of the gpu dialect, and the upstream
# pipeline that lowers those. The kernels are embedded as PTX text, which
# the driver compiles when the program starts.
md_exec_gpu_lowering = (
    " --md-exec-assign-storage=memory=device --convert-md-exec-to-gpu"
)
config.substitutions.append(
    ("%md_exec_gpu_passes", md_exec_transforms + md_exec_gpu_lowering)
)
config.substitutions.append(
    (
        "%lower_gpu_to_llvm",
        '--gpu-lower-to-nvvm-pipeline="cubin-format=isa"'
        " --reconcile-unrealized-casts",
    )
)

# Lowers loops over buffers to the LLVM dialect: sequentially, or with the
# parallel loops run by OpenMP.
loops_to_llvm = (
    "--convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm"
    " --convert-vector-to-llvm --expand-strided-metadata"
    " --finalize-memref-to-llvm --convert-arith-to-llvm"
    " --convert-func-to-llvm --convert-cf-to-llvm"
)
config.substitutions.append(
    ("%lower_loops_to_llvm", loops_to_llvm + " --reconcile-unrealized-casts")
)
config.substitutions.append(
    (
        "%lower_loops_to_openmp",
        "--convert-scf-to-openmp --canonicalize "
        + loops_to_llvm
        + " --convert-openmp-to-llvm --reconcile-unrealized-casts",
    )
)

# Lowers functions of scalars and small vectors to the LLVM dialect.
config.substitutions.append(
    (
        "%lower_to_llvm",
        "--convert-math-to-llvm --convert-math-to-libm --convert-vector-to-llvm"
        " --convert-arith-to-llvm --convert-func-to-llvm"
        " --reconcile-unrealized-casts",
    )
)

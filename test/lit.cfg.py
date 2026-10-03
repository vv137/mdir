import os

import lit.formats

from lit.llvm import llvm_config

config.name = "MDIR"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".toml", ".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mdir_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in", "lib", "Inputs"]

# The options of the sanitizers reach the tests of a sanitized build
# (scripts/build-sanitized.sh), and the choice of the device reaches the
# tests that run on one: without it they took the first device whatever
# the shell chose, and shared it with a benchmark there.
llvm_config.with_system_environment(
    ["HOME", "TMP", "TEMP", "ASAN_OPTIONS", "UBSAN_OPTIONS",
     "CUDA_VISIBLE_DEVICES"])
# with_system_environment forwards only values that are not empty; an empty
# choice of devices, which hides every GPU, is forwarded as well, so that
# the tests run on no device rather than on the first.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    config.environment["CUDA_VISIBLE_DEVICES"] = ""
llvm_config.use_default_substitutions()

config.mdir_tools_dir = os.path.join(config.mdir_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.mdir_tools_dir, config.llvm_tools_dir]
tools = [
    "mdir",
    "mdir-opt",
    "mlir-opt",
    "mlir-runner",
    "FileCheck",
    "not",
    "split-file",
]
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

# The Python that runs lit, for the scripts that check results.
import sys

config.substitutions.append(("%python", sys.executable))

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

# Checkpoints need HDF5.
if config.mdir_hdf5:
    config.available_features.add("hdf5")

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
        # nvidia-smi lists every device whatever CUDA_VISIBLE_DEVICES
        # hides; an empty choice hides them all.
        hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        if devices.returncode == 0 and devices.stdout.strip() and not hidden:
            config.available_features.add("cuda")
    except (OSError, subprocess.SubprocessError):
        pass
    config.environment["CUDA_ROOT"] = config.mdir_cuda_root
# Runs of the driver under compute-sanitizer (memcheck, initcheck,
# racecheck) take about a minute; they run when lit is given
# -Dsanitize=1 (docs/principles.md, Section 6).
if "cuda" in config.available_features and lit_config.params.get("sanitize"):
    sanitizer = os.path.join(config.mdir_cuda_root, "bin", "compute-sanitizer")
    if os.path.exists(sanitizer):
        config.available_features.add("compute-sanitizer")
        config.substitutions.append(("%compute_sanitizer", sanitizer))

# Short runs of the systems of the Amber benchmark suite, which
# scripts/benchmarks/amber/bench.py prepares in MDIR_BENCH_DIR
# (docs/principles.md, Section 5).
bench = os.environ.get("MDIR_BENCH_DIR")
if bench and os.path.exists(os.path.join(bench, "jac_nve", "system.parm7")):
    config.available_features.add("amber-suite")
    config.environment["MDIR_BENCH_DIR"] = bench
    config.substitutions.append(("%amber_suite", bench))
    config.substitutions.append(
        (
            "%bench",
            "python3 "
            + os.path.join(
                os.path.dirname(__file__), "..", "scripts", "benchmarks", "amber",
                "bench.py",
            ),
        )
    )

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
md_exec_transforms = "--md-exec-reuse-neighbors --md-exec-expose-validity --md-exec-fuse-loops --cse"
md_exec_fast_transforms = (
    "--md-exec-reuse-neighbors --md-exec-expose-validity --md-exec-fuse-loops"
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

import os

import lit.formats

from lit.llvm import llvm_config

config.name = "MDIR"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".toml", ".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mdir_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in", "mdir_lit.py", "lib", "Inputs"]

# The options of the sanitizers reach the tests of a sanitized build
# (scripts/build-sanitized.sh), and the choice of the device reaches the
# tests that run on one: without it they took the first device whatever
# the shell chose, and shared it with a benchmark there. The cache of the
# CUDA driver, which holds the kernels that it compiles from PTX, reaches
# the tests as well, so that a suite can keep it off a home directory on a
# network file system (#223). Temporary files need nothing here: lit gives
# each run a directory of its own under the TMPDIR that lit was given, and
# sets TMPDIR, TMP, and TEMP of the tests to it.
llvm_config.with_system_environment(
    ["HOME", "TMP", "TEMP", "ASAN_OPTIONS", "UBSAN_OPTIONS",
     "CUDA_VISIBLE_DEVICES", "CUDA_CACHE_PATH"])
# What lit itself was given, for the test that the tests receive it
# (test/Lit/environment.test).
for variable in ("CUDA_CACHE_PATH", "TMPDIR"):
    config.substitutions.append(
        ("%suite_" + variable.lower(), os.environ.get(variable, "")))
# with_system_environment forwards only values that are not empty; an empty
# choice of devices, which hides every GPU, is forwarded as well, so that
# the tests run on no device rather than on the first.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    config.environment["CUDA_VISIBLE_DEVICES"] = ""
# The cache of the chosen skins (D281) is off, so that no choice of
# an earlier run under HOME reaches a test; the tests of the cache give
# their own directory and turn it on.
config.environment["MDIR_SKIN_CACHE"] = "off"
llvm_config.use_default_substitutions()

config.mdir_tools_dir = os.path.join(config.mdir_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.mdir_tools_dir, config.llvm_tools_dir]
tools = [
    "mdir",
    "mdir-opt",
    "mdir-model-test",
    "mdir-output-rows-test",
    "mdir-prolate-test",
    "mdir-influence-test",
    "mdir-neighbor-count-test",
    "mdir-compile-test",
    "mdir-jit-memory-test",
    "mdir-compile-cache-test",
    "mlir-opt",
    "mlir-runner",
    "FileCheck",
    "not",
    "split-file",
    "mlir-tblgen",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
# The headers of MLIR, for mlir-tblgen (test/Docs/dialect-docs.test).
config.substitutions.append(
    ("%mlir_include_dir", os.path.join(os.path.dirname(config.llvm_lib_dir), "include")))

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
# lit itself, for the tests of how the suite is scheduled (test/Lit).
config.substitutions.append(
    ("%lit", sys.executable + " -c 'import lit.main; lit.main.main()'"))

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

if config.mdir_python:
    config.available_features.add("python-api")
    config.substitutions.append(("%mdir_python", config.mdir_python_executable))
    llvm_config.with_environment("PYTHONPATH", os.path.join(config.mdir_obj_root, "python"))
    # OpenMM's unit module, for the tests of unit quantities (D200).
    import subprocess

    if subprocess.run([config.mdir_python_executable, "-c", "import openmm.unit"],
                      capture_output=True).returncode == 0:
        config.available_features.add("openmm")
    # PyTorch, the independent consumer of DLPack views (D220):
    # the interpreter of the tests if it imports torch, or one that lit is
    # given with -Dtorch_python=<interpreter>, which must import the module
    # of this build (the same Python version).
    torch_python = lit_config.params.get("torch_python", config.mdir_python_executable)
    if subprocess.run([torch_python, "-c", "import torch"], capture_output=True).returncode == 0:
        config.available_features.add("torch")
        config.substitutions.append(("%torch_python", torch_python))
        # torch-pme, the independent differentiable PME that the derivative
        # in the charges is checked against (D231).
        if subprocess.run([torch_python, "-c", "import torchpme"],
                          capture_output=True).returncode == 0:
            config.available_features.add("torchpme")

# Checkpoints need HDF5.
if config.mdir_hdf5:
    config.available_features.add("hdf5")

# h5py, an independent reader of the trajectories in H5MD
# (D239), and MDAnalysis if it is there: the interpreter of the
# tests if it imports h5py, or one that lit is given with
# -Dh5md_python=<interpreter>, which need not import the module of this
# build.
if config.mdir_python:
    import subprocess

    h5md_python = lit_config.params.get("h5md_python", config.mdir_python_executable)
    if subprocess.run([h5md_python, "-c", "import h5py"], capture_output=True).returncode == 0:
        config.available_features.add("h5py")
        config.substitutions.append(("%h5md_python", h5md_python))

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
# The tests share the host objects they compile (test/mdir_lit.py).
sys.path.insert(0, os.path.dirname(__file__))
import mdir_lit

mdir_lit.enable_compile_cache(config, lit_config)
# A shard of the suite is taken in the lexical order of the tests (D309).
mdir_lit.check_shards(lit_config)

# One device serves every worker of a suite: at most four tests that need it
# run at once, the others side by side (test/mdir_lit.py).
if "cuda" in config.available_features:
    mdir_lit.serialize_gpu_tests(config, lit_config)
# Runs of the driver under compute-sanitizer (memcheck, initcheck,
# racecheck) take about a minute; they run when lit is given
# -Dsanitize=1 (docs/principles.md, Section 6).
if "cuda" in config.available_features and lit_config.params.get("sanitize"):
    sanitizer = os.path.join(config.mdir_cuda_root, "bin", "compute-sanitizer")
    if os.path.exists(sanitizer):
        config.available_features.add("compute-sanitizer")
        config.substitutions.append(("%compute_sanitizer", sanitizer))

# The plugin of learned potentials (D270): "metatomic" where
# it is built; "mlip-python" where the Python of the environment that it was
# built against (MDIR_MLIP_PYTHON of the build, or -Dmlip_python=) imports
# torch, metatomic, and NequIP; and "allegro-package" where the model package
# of the tests, Allegro-OAM-L-0.1.nequip.zip (docs/metatomic-ml1.md,
# Section 2; never in the repository), is given by MDIR_ALLEGRO_PACKAGE or
# -Dallegro_package=. The tests build their artifact from it.
if config.mdir_metatomic:
    import subprocess

    config.available_features.add("metatomic")
    mlip_python = lit_config.params.get("mlip_python", config.mdir_mlip_python)
    if mlip_python and config.mdir_python and subprocess.run(
            [mlip_python, "-c", "import torch, metatomic.torch, nequip, ase"],
            capture_output=True).returncode == 0:
        config.available_features.add("mlip-python")
        config.substitutions.append(("%mlip_python", mlip_python))
    package = lit_config.params.get("allegro_package",
                                    os.environ.get("MDIR_ALLEGRO_PACKAGE", ""))
    if package and os.path.exists(package):
        config.available_features.add("allegro-package")
        config.substitutions.append(("%allegro_package", package))
    # ML4's baselines (D286, docs/ml4-baselines.md): "ml4-python"
    # where the interpreter of -Dml4_python= or MDIR_ML4_PYTHON imports torch,
    # metatomic, mace-torch, and SchNetPack (scripts/mlip/ml4-requirements.lock),
    # and "ml4-artifacts" where the directory of -Dml4_artifacts= or
    # MDIR_ML4_ARTIFACTS holds the checkpoints that
    # scripts/mlip/fetch-ml4-artifacts.py fetches (never in the repository).
    ml4_python = lit_config.params.get("ml4_python", os.environ.get("MDIR_ML4_PYTHON", ""))
    if ml4_python and config.mdir_python and subprocess.run(
            [ml4_python, "-c", "import torch, metatomic.torch, mace, schnetpack, ase"],
            capture_output=True).returncode == 0:
        config.available_features.add("ml4-python")
        config.substitutions.append(("%ml4_python", ml4_python))
    artifacts = lit_config.params.get("ml4_artifacts",
                                      os.environ.get("MDIR_ML4_ARTIFACTS", ""))
    if artifacts and all(os.path.exists(os.path.join(artifacts, f)) for f in (
            "painn_100k.ckpt", "MACE-OFF23_medium.model")):
        config.available_features.add("ml4-artifacts")
        config.substitutions.append(("%ml4_artifacts", artifacts))

# Models in stages (D294, docs/ml4-ir.md, Section 8):
# "stages-export" where the interpreter of -Djax_python= or MDIR_JAX_PYTHON
# imports JAX and NumPy, and -Dstablehlo_opt= or MDIR_STABLEHLO_OPT names
# StableHLO's stablehlo-opt, which legalizes the exported bodies to linalg
# (scripts/mlip/painn-stages.py export).
import subprocess as _subprocess

jax_python = lit_config.params.get("jax_python", os.environ.get("MDIR_JAX_PYTHON", ""))
stablehlo_opt = lit_config.params.get("stablehlo_opt",
                                      os.environ.get("MDIR_STABLEHLO_OPT", ""))
if jax_python and stablehlo_opt and os.access(stablehlo_opt, os.X_OK) and _subprocess.run(
        [jax_python, "-c", "import jax, numpy"], capture_output=True).returncode == 0:
    config.available_features.add("stages-export")
    config.substitutions.append(("%jax_python", jax_python))
    config.substitutions.append(("%stablehlo_opt", stablehlo_opt))

# The tests that a release adds to the default suite: the whole matrix of
# a test whose default form runs a part of its cases, on every target and
# in every precision. They are named *-release.test and
# *-release-gpu.test, require "release-tests", and run when lit is given
# -Drelease=1, as the suite of scripts/release/publish.sh is
# (docs/workflow.md, Releases). The feature does not depend on a device:
# such a test names the device it needs.
if lit_config.params.get("release"):
    config.available_features.add("release-tests")

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

# Every feature is known now: compare what they leave unsupported with
# test/expected-unsupported.txt (#608, test/mdir_lit.py).
mdir_lit.check_expected_unsupported(config, lit_config)

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
# The lowerings read the algebra of each sum, which md-exec-assign-algebra
# states (D301); these pipelines are in the default mode.
md_exec_lowering = (
    " --md-exec-assign-algebra --md-exec-assign-storage"
    " --convert-md-exec-to-loops"
)
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
    " --md-exec-assign-algebra --md-exec-assign-storage=memory=device"
    " --convert-md-exec-to-gpu"
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

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

# The passes of the semantic level, up to code in ordinary functions.
config.substitutions.append(
    (
        "%md_passes",
        "--md-check-exchange --md-differentiate --md-expand-truncation"
        " --md-inline",
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

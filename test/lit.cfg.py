import os

import lit.formats

from lit.llvm import llvm_config

config.name = "MDIR"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mdir_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in", "lib"]

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

# Lowers functions of scalars and small vectors to the LLVM dialect.
config.substitutions.append(
    (
        "%lower_to_llvm",
        "--convert-math-to-llvm --convert-math-to-libm --convert-vector-to-llvm"
        " --convert-arith-to-llvm --convert-func-to-llvm"
        " --reconcile-unrealized-casts",
    )
)

"""Scheduling of the MDIR tests by lit.

lit runs the tests of a suite in several worker processes, and
CUDA_VISIBLE_DEVICES names one device for all of them. Tests that need the
device take the parallelism group "mdir-gpu", of size 16 unless lit is given
-Dgpu_workers=N, so that at most sixteen of them share the device while the
others run side by side. One at a time, the suite took 1266 s on an RTX 3090;
sharing the device without a bound, 262 s (PR #83).

Most tests that need the device spend their time compiling, on one core of
the host each, and leave the device nearly idle, so the size of the group
sets the wall time (D208, #141). After the Python tests
were split by scenario, a Release suite with Python on one RTX 3090 and a
128-core host took 969, 496, 370, and 262 s with groups of 4, 8, 12, and 16;
at 16 the wall time is that of the slowest test (about 260 s), so a larger
group gains nothing. The device then held at most 4.2 GB. A smaller device,
or a host with fewer cores, takes a smaller -Dgpu_workers=N. The lock of the device
that a suite holds (docs/workflow.md, "Shared machines") keeps other suites
and timing runs off the device; this group orders the tests within one
suite.

lit pickles the configuration for its workers, so the group is chosen by an
instance of a class of this module rather than by a function of lit.cfg.py.
"""
import os
import re

GPU_GROUP = "mdir-gpu"

# lit reads a keyword anywhere on a line, in the comments of any language.
_requires = re.compile(r"REQUIRES:(.*)")
_cuda = re.compile(r"(?<![\w-])cuda(?![\w-])")


def requires_cuda(path):
    """Whether a REQUIRES line of the test at path names the feature cuda."""
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            for line in file:
                match = _requires.search(line)
                if match and _cuda.search(match.group(1)):
                    return True
    except OSError:
        pass
    return False


class GPUGroup:
    """The parallelism group of a test: "mdir-gpu" if it needs CUDA."""

    def __call__(self, test):
        if requires_cuda(test.getSourcePath()):
            return GPU_GROUP
        return None


def serialize_gpu_tests(config, lit_config):
    """Run at most sixteen tests that need CUDA at a time, or gpu_workers
    when lit is given -Dgpu_workers=N."""
    value = lit_config.params.get("gpu_workers", "16")
    try:
        workers = int(value)
    except ValueError:
        workers = 0
    if workers < 1:
        lit_config.fatal(f"gpu_workers must be a positive integer, not '{value}'")
    lit_config.parallelism_groups[GPU_GROUP] = workers
    config.parallelism_group = GPUGroup()
    limit_compile_threads(config, lit_config, workers)


def limit_compile_threads(config, lit_config, workers):
    """Bound the threads with which each test lowers its programs. A process
    lowers with as many threads as the host has (D211), so the tests that
    need the device, `workers` of them at once, would ask for `workers`
    times the cores. Each takes its share of the cores, or compile_threads
    when lit is given -Dcompile_threads=N. A suite took 285, 252, and 262 s
    with 128, 8, and 1 threads to a process (one RTX 3090, 128 cores
    shared with other jobs)."""
    cores = os.cpu_count() or 1
    value = lit_config.params.get("compile_threads",
                                  str(max(1, cores // workers)))
    try:
        threads = int(value)
    except ValueError:
        threads = 0
    if threads < 1:
        lit_config.fatal(
            f"compile_threads must be a positive integer, not '{value}'")
    config.environment["MDIR_COMPILE_THREADS"] = str(threads)


def enable_compile_cache(config, lit_config):
    """Share the host objects of the tests' JIT engines through the compile
    cache (D[compile-cache]), in a directory of the build tree, so that a
    suite of one build shares nothing with that of another. lit -Dcompile_cache=off,
    or MDIR_COMPILE_CACHE=off in the environment, runs the suite without it;
    a test of compilation itself sets MDIR_COMPILE_CACHE=off in its RUN
    lines."""
    off = (lit_config.params.get("compile_cache", "on").lower() == "off" or
           os.environ.get("MDIR_COMPILE_CACHE", "").lower() == "off")
    if off:
        config.environment["MDIR_COMPILE_CACHE"] = "off"
        return
    config.environment["MDIR_COMPILE_CACHE_DIR"] = os.path.join(
        config.mdir_obj_root, "test", "compile-cache")

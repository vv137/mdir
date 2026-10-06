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

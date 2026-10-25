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
    cache (D212), in a directory of the build tree, so that a
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


# The tests expected to be unsupported, by the feature that decides it
# (test/expected-unsupported.txt, #608). A test that becomes unsupported by
# accident, through a feature missing on a node, an interpreter that fails
# to import, or a changed REQUIRES, would otherwise look like a pass in the
# counts of a suite.
_keyword = re.compile(r"(REQUIRES|UNSUPPORTED):(.*)")
_entry = re.compile(r"^(without|with)\s+([\w.+-]+):\s*(\S+)$")


def read_expected(path):
    """The entries of the list, (kind, feature, test): kind "without" (the
    test needs the feature) or "with" (the feature makes the test
    unsupported); test a path under test/, or "*" for every test whose
    REQUIRES names the feature."""
    entries = []
    with open(path, encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            match = _entry.match(line)
            if not match:
                raise ValueError(f"{path}:{number}: not 'without|with "
                                 f"<feature>: <test>': {line}")
            entries.append(match.groups())
    return entries


def read_conditions(path):
    """The REQUIRES and UNSUPPORTED expressions of a test, as lit reads them:
    on any line, comma-separated."""
    requires, unsupported = [], []
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            for line in file:
                match = _keyword.search(line)
                if not match:
                    continue
                exprs = [e.strip() for e in match.group(2).split(",") if e.strip()]
                if match.group(1) == "REQUIRES":
                    requires.extend(exprs)
                else:
                    unsupported.extend(exprs)
    except OSError:
        pass
    return requires, unsupported


def _names(exprs):
    names = set()
    for expr in exprs:
        names |= set(re.findall(r"[\w.+-]+", expr))
    return names - {"true", "false"}


def list_tests(root, suffixes, excludes):
    """The test files under root, as lit discovers them: by suffix, outside
    the excluded names."""
    tests = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in excludes)
        for name in sorted(files):
            if name in excludes or os.path.splitext(name)[1] not in suffixes:
                continue
            tests.append(os.path.relpath(os.path.join(directory, name), root))
    return tests


def compare(root, tests, features, entries, evaluate):
    """For these features: the tests the list expects to be unsupported,
    the unsupported tests it does not expect (with the reason), and the
    tests it expects to be unsupported that run."""
    features = set(features)
    conditions = {t: read_conditions(os.path.join(root, t)) for t in tests}
    expected = set()
    for kind, feature, test in entries:
        holds = (feature not in features) if kind == "without" else (feature in features)
        if not holds:
            continue
        if test == "*":
            expected |= {t for t, (req, _) in conditions.items() if feature in _names(req)}
        else:
            expected.add(test)
    unexpected, running = [], []
    for test in tests:
        req, uns = conditions[test]
        unsupported = (not all(evaluate(e, features) for e in req) or
                       any(evaluate(e, features) for e in uns))
        if unsupported and test not in expected:
            reason = ([f"without {f}" for f in sorted(_names(req) - features)] +
                      [f"with {f}" for f in sorted(_names(uns) & features)])
            unexpected.append(f"{test} ({', '.join(reason)})")
        elif not unsupported and test in expected:
            running.append(test)
    return sorted(expected), unexpected, running


def check_list(root, tests, entries):
    """The list against the tests: every test it names exists and names its
    feature, and every test that names a feature with explicit entries is
    listed under it."""
    problems = []
    conditions = {t: read_conditions(os.path.join(root, t)) for t in tests}
    explicit = {}
    for kind, feature, test in entries:
        if test == "*":
            continue
        explicit.setdefault((kind, feature), set()).add(test)
        if test not in conditions:
            problems.append(f"{kind} {feature}: {test}: no such test")
            continue
        req, uns = conditions[test]
        if feature not in _names(req if kind == "without" else uns):
            word = "REQUIRES" if kind == "without" else "UNSUPPORTED"
            problems.append(f"{kind} {feature}: {test}: its {word} does not name {feature}")
    for (kind, feature), listed in sorted(explicit.items()):
        for test, (req, uns) in sorted(conditions.items()):
            if feature in _names(req if kind == "without" else uns) and test not in listed:
                problems.append(f"{kind} {feature}: {test}: not listed")
    return problems


def _runs_whole_suite(config):
    """Whether lit was given the root of the suite, in the source or the
    build tree, as check-mdir gives it."""
    import sys
    roots = {os.path.realpath(config.test_source_root),
             os.path.realpath(config.test_exec_root)}
    return any(not arg.startswith("-") and os.path.realpath(arg) in roots
               for arg in sys.argv[1:])


def check_expected_unsupported(config, lit_config):
    """Compare the tests that the features of this configuration leave
    unsupported with test/expected-unsupported.txt, before any test runs.
    On a run of the whole suite, or with -Dexpected_unsupported=check, a
    difference is a warning; with -Dexpected_unsupported=strict it is an
    error; -Dexpected_unsupported=off skips the check."""
    from lit.BooleanExpression import BooleanExpression

    mode = lit_config.params.get("expected_unsupported", "")
    if mode not in ("", "check", "strict", "off"):
        lit_config.fatal("expected_unsupported must be check, strict, or "
                         f"off, not '{mode}'")
    if mode == "off" or (not mode and not _runs_whole_suite(config)):
        return
    root = config.test_source_root
    try:
        entries = read_expected(os.path.join(root, "expected-unsupported.txt"))
    except (OSError, ValueError) as error:
        lit_config.fatal(str(error))
    tests = list_tests(root, set(config.suffixes), set(config.excludes))
    expected, unexpected, running = compare(
        root, tests, config.available_features, entries,
        BooleanExpression.evaluate)
    lit_config.note(f"expected unsupported: {len(expected)} of {len(tests)} "
                    "tests for the features of this configuration "
                    "(test/expected-unsupported.txt)")
    problems = ([f"unsupported but not expected: {t}" for t in unexpected] +
                [f"expected unsupported but runs: {t}" for t in running])
    if not problems:
        return
    message = ("the unsupported tests differ from "
               "test/expected-unsupported.txt:\n  " + "\n  ".join(problems))
    if mode == "strict":
        lit_config.fatal(message)
    lit_config.warning(message)
    # Again after the results, where the log of a suite ends.
    import atexit
    import sys

    def repeat():
        sys.stdout.flush()
        print("lit: warning: " + message, file=sys.stderr, flush=True)

    atexit.register(repeat)


if __name__ == "__main__":
    # python3 test/mdir_lit.py [ROOT]: check the list against the REQUIRES
    # and UNSUPPORTED lines of the tests (test/Lit/expected-unsupported.test).
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    excludes = {"CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in",
                "mdir_lit.py", "lib", "Inputs"}
    tests = list_tests(root, {".mlir", ".toml", ".test"}, excludes)
    try:
        entries = read_expected(os.path.join(root, "expected-unsupported.txt"))
    except (OSError, ValueError) as error:
        print(error)
        sys.exit(1)
    problems = check_list(root, tests, entries)
    for problem in problems:
        print(problem)
    print(f"{len(entries)} entries, {len(tests)} tests, {len(problems)} problems")
    sys.exit(1 if problems else 0)

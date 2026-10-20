"""Check lit's compile thread budget under a restricted CPU affinity."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import mdir_lit


def threads(params):
    config = SimpleNamespace(environment={})
    lit_config = SimpleNamespace(params=params)
    mdir_lit.limit_compile_threads(config, lit_config, 16)
    return config.environment["MDIR_COMPILE_THREADS"]


with patch.object(mdir_lit.os, "sched_getaffinity", return_value=set(range(16)), create=True), \
     patch.object(mdir_lit.os, "cpu_count", return_value=64):
    assert threads({}) == "1"
    print("affinity: 1 thread per test")

with patch.object(mdir_lit.os, "sched_getaffinity", side_effect=OSError, create=True), \
     patch.object(mdir_lit.os, "cpu_count", return_value=64):
    assert threads({}) == "4"
    print("fallback: 4 threads per test")

with patch.object(mdir_lit.os, "sched_getaffinity", return_value=set(range(16)), create=True):
    assert threads({"compile_threads": "8"}) == "8"
    print("override: 8 threads per test")

"""Reject malformed ABI selectors before dereferencing a null handle."""
import ctypes
import resource
import signal
import subprocess
import sys

if len(sys.argv) == 4:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    runtime = ctypes.CDLL(sys.argv[1])
    getter = runtime._mlir_ciface_mdrtGroupsBuffer
    getter.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    getter(None, 0, int(sys.argv[2]))
    raise SystemExit("invalid selector unexpectedly returned")

for slot in (-1, 16, 2**63 - 1):
    result = subprocess.run([sys.executable, __file__, sys.argv[1],
                             str(slot), "child"], capture_output=True, text=True)
    assert result.returncode == -signal.SIGABRT, (slot, result)
    assert f"invalid groups buffer slot {slot} (expected 0..15)" in result.stderr
print("Three invalid slots diagnosed before any handle or array access")

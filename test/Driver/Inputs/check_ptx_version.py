"""Loads a PTX module of an ISA version that no driver supports through the
GPU runtime, and checks that the load ends the process with an error that
names the driver's CUDA version, the PTX ISA version, and what to do.

Usage: check_ptx_version.py <libmdrt_cuda.so>
"""
import ctypes
import subprocess
import sys

library = sys.argv[1]

# The kernels are PTX that the driver compiles at load; `.version` is the
# PTX ISA version, here one that no driver knows.
PTX = b""".version 99.9
.target sm_52
.address_size 64

.visible .entry empty()
{
  ret;
}
"""

LOAD = """
import ctypes, sys
runtime = ctypes.CDLL(sys.argv[1])
ptx = ctypes.create_string_buffer(sys.argv[3].encode())
if sys.argv[2] == "jit":
    runtime.mgpuModuleLoadJIT.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    runtime.mgpuModuleLoadJIT(ptx, 2, len(sys.argv[3]) + 1)
else:
    runtime.mgpuModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    runtime.mgpuModuleLoad(ptx, len(sys.argv[3]) + 1)
print("loaded")
"""

for entry in ("jit", "plain"):
    result = subprocess.run(
        [sys.executable, "-c", LOAD, library, entry, PTX.decode()],
        capture_output=True, text=True)
    assert result.returncode == 1, (entry, result.returncode, result.stderr)
    assert "loaded" not in result.stdout, result.stdout
    message = result.stderr
    assert "the NVIDIA driver supports CUDA " in message, message
    assert "cannot compile kernels of PTX ISA 99.9" in message, message
    assert "Update the NVIDIA driver" in message, message
    print(entry + ": " + message.strip().splitlines()[0][:60])
print("unsupported PTX ISA versions are reported")

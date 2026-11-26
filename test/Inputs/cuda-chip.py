"""Print the architecture of the first visible CUDA device."""
import ctypes as C
import sys

try:
    driver = C.CDLL("libcuda.so.1")
except OSError:
    sys.exit(1)
device, major, minor = C.c_int(), C.c_int(), C.c_int()
if (driver.cuInit(0) or driver.cuDeviceGet(C.byref(device), 0)
        or driver.cuDeviceComputeCapability(C.byref(major), C.byref(minor), device)):
    sys.exit(1)
print(f"sm_{major.value}{minor.value}")

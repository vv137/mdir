"""The registry and the pool of the blocks of the device (#622), through
the functions of the runtime: more blocks than 4096 in use in one
activation, all taken back when it closes; a second activation served from
the pool; the budget of the pool in bytes; a request that the driver
refuses for want of memory, made once more after the idle memory is given
back; and the explicit trim.

Usage: device_blocks.py LIBMDRT_CUDA SCENARIO

  registry  5000 blocks of 4 KiB in one activation, then another
  budget    (MDIR_DEVICE_POOL_BYTES=1048576) the idle blocks past 1 MiB go
            back to the driver
  retry     (MDRT_FAIL_ALLOCATION=101) 100 idle blocks, then a request
            that the driver refuses once for want of memory: the pool is
            trimmed and the request made again
"""
import ctypes
import sys

lib = ctypes.CDLL(sys.argv[1])
scenario = sys.argv[2]
lib.mdrtDeviceActivationOpen.restype = ctypes.c_void_p
lib.mdrtDeviceActivationEnter.argtypes = [ctypes.c_void_p]
lib.mdrtDeviceActivationClose.argtypes = [ctypes.c_void_p]
lib.mgpuMemAlloc.restype = ctypes.c_void_p
lib.mgpuMemAlloc.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_bool]
lib.mgpuMemFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.mdrtDeviceTrim.restype = ctypes.c_uint64
NAMES = ("live", "live bytes", "idle", "idle bytes", "driver blocks", "plans", "plan bytes", "free")


def counts():
    values = (ctypes.c_int64 * 8)()
    lib.mdrtDeviceMemoryCounts(values)
    return dict(zip(NAMES, values))


def show(when):
    c = counts()
    print(f"{when}: " + ", ".join(f"{name} {c[name]}" for name in NAMES[:5]))
    return c


def activation(sizes):
    a = lib.mdrtDeviceActivationOpen()
    lib.mdrtDeviceActivationEnter(a)
    pointers = [lib.mgpuMemAlloc(size, None, False) for size in sizes]
    assert all(pointers) and len(set(pointers)) == len(pointers)
    return a, pointers


def close(a):
    lib.mdrtDeviceActivationLeave()
    lib.mdrtDeviceActivationClose(a)


if scenario == "registry":
    sizes = [4096] * 5000
    a, _ = activation(sizes)
    show("5000 blocks in use")
    close(a)
    show("closed")
    a, _ = activation(sizes)
    show("again, from the pool")
    close(a)
    show("closed again")
    print("trimmed", lib.mdrtDeviceTrim(), "bytes")
    show("after the trim")
elif scenario == "budget":
    a, _ = activation([4096] * 1000)
    close(a)
    show("1000 blocks closed under a budget of 1 MiB")
    print("trimmed", lib.mdrtDeviceTrim(), "bytes")
    show("after the trim")
elif scenario == "retry":
    # MDRT_FAIL_ALLOCATION=101: the 101st request to the driver reports that
    # it has no memory. The first 100 make blocks of 4 KiB that go idle; the
    # 101st, of another size, is refused once, the pool is trimmed, and the
    # request is made again.
    a, _ = activation([4096] * 100)
    close(a)
    show("100 idle blocks")
    a, _ = activation([8192])
    show("a request refused once, after the trim")
    close(a)
    lib.mdrtDeviceTrim()
    show("after the trim")
print(f"device blocks {scenario} passed")

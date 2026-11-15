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

The pool's best fit with splitting and coalescing (D416):

  split     a request served from a larger idle arena, cut at its size
            rounded up to 256 bytes; every piece aligned to 256 bytes
  tail      a rest of fewer than 256 bytes is not cut off: the request
            takes the whole piece
  best      the smallest idle piece that holds a request is taken
  coalesce  freed pieces stay unmerged until a request finds no piece that
            holds it; then a piece merges with both idle neighbors, and the
            arena, idle as a whole again, serves a request of its full size
  exact     a request takes the piece that last held a block of its size,
            not another piece of the same extent
  close     the blocks of an activation that closes merge with their idle
            neighbors
  arenas    pieces of two arenas never merge, even where the driver placed
            the arenas next to each other
  trim      a trim returns the idle arenas, not one that holds a block
            in use
  pieces    (MDIR_DEVICE_POOL_BYTES=1048576) the budget returns an arena
            that becomes idle as a whole, not the idle pieces of one in use
  off       (MDIR_DEVICE_POOL_BYTES=0) no block is kept: every freed arena
            goes back to the driver, so nothing is split
  replaced  a buffer of the runtime that a growth replaces goes back to the
            driver if it is an arena of its own, and to the pool if it is
            a piece of a larger one (D312)
  side-alloc, side-free, other-stream
            an allocation or a free of a block of the pool on the second
            stream (between mdrtSideBegin and mdrtSideEnd), or with a stream
            named that is not the runtime's first: the runtime stops the run
  stream-ok an allocation and a free with the runtime's first stream named,
            and after the second stream has ended: accepted
  streams   a piece freed while work on stream A still writes it, merged
            with an idle neighbor, and taken on stream B after the runtime
            changes to B, is written on B only after A's work
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
lib.mdrtMatrixCreate.restype = ctypes.c_int64
lib.mdrtMatrixCreate.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.mdrtMatrixGrow.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.mdrtUseStream.restype = ctypes.c_void_p
lib.mdrtUseStream.argtypes = [ctypes.c_void_p]
lib.mdrtRestoreStream.argtypes = [ctypes.c_void_p]
lib.mgpuMemset32.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
KIB, MIB = 1024, 1048576
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
elif scenario in ("side-alloc", "side-free", "other-stream", "stream-ok"):
    lib.mgpuStreamCreate.restype = ctypes.c_void_p
    first = lib.mgpuStreamCreate()
    if scenario == "side-alloc":
        lib.mdrtSideBegin()
        print("allocating on the second stream", flush=True)
        lib.mgpuMemAlloc(4096, None, False)
    elif scenario == "side-free":
        pointer = lib.mgpuMemAlloc(4096, None, False)
        lib.mdrtSideBegin()
        print("freeing on the second stream", flush=True)
        lib.mgpuMemFree(pointer, None)
    elif scenario == "other-stream":
        other = ctypes.c_void_p()
        assert ctypes.CDLL("libcuda.so.1").cuStreamCreate(ctypes.byref(other), 1) == 0
        print("allocating with another stream named", flush=True)
        lib.mgpuMemAlloc(4096, other, False)
    else:
        pointer = lib.mgpuMemAlloc(4096, first, False)
        lib.mgpuMemFree(pointer, first)
        lib.mdrtSideBegin()
        lib.mdrtSideEnd()
        lib.mdrtSideJoin()
        pointer = lib.mgpuMemAlloc(4096, None, False)
        lib.mgpuMemFree(pointer, None)
        lib.mdrtDeviceTrim()
        show("the first stream named, and after the second stream")
    print("not refused", flush=True)
elif scenario in ("split", "tail", "best", "coalesce", "arenas", "trim",
                  "pieces", "off", "replaced", "streams", "exact", "close"):
    def allocate(size):
        pointer = lib.mgpuMemAlloc(size, None, False)
        assert pointer and pointer % 256 == 0, (size, pointer)
        return pointer

    def free(pointer):
        lib.mgpuMemFree(pointer, None)

    def idle_arenas(*sizes):
        """Arenas of `sizes` bytes, all taken from the driver before any is
        freed, then idle in the pool in this order: their bases."""
        bases = [allocate(size) for size in sizes]
        for base in bases:
            free(base)
        return bases

    def idle_arena(size):
        return idle_arenas(size)[0]

    a = lib.mdrtDeviceActivationOpen()
    lib.mdrtDeviceActivationEnter(a)
    if scenario == "split":
        base = idle_arena(MIB)
        first = allocate(1000)
        second = allocate(300)
        print("taken from the arena:", first == base, second == base + 1024)
        show("two pieces of one arena")
        free(first)
        free(second)
        show("both freed")
    elif scenario == "tail":
        base = idle_arena(4096 + 200)
        taken = allocate(4096)
        print("the whole arena taken:", taken == base)
        show("a rest of 200 bytes not cut off")
        free(taken)
        show("freed")
    elif scenario == "best":
        sizes = (16 * KIB, 64 * KIB, 32 * KIB)
        bases = idle_arenas(*sizes)
        show("three idle arenas")
        taken = allocate(10 * KIB)
        print("the smallest that holds 10 KiB, of 16, 64, and 32 KiB:",
              dict(zip(bases, sizes))[taken] // KIB, "KiB")
        free(taken)
    elif scenario == "coalesce":
        base = idle_arena(MIB)
        left, middle, right = allocate(4096), allocate(4096), allocate(4096)
        free(left)
        free(right)
        show("the left and the right piece freed")
        free(middle)
        show("the middle piece freed")
        whole = allocate(MIB)
        print("the arena serves its full size again:", whole == base)
        show("in use as a whole")
        free(whole)
    elif scenario == "exact":
        base = idle_arena(MIB)
        first = allocate(4000)
        guard = allocate(4096)
        other = allocate(4096)
        guard2 = allocate(4096)
        free(first)
        free(other)
        show("two idle pieces of 4096 bytes")
        again = allocate(4000)
        print("4000 bytes take the piece that held 4000:", again == first, again == other)
        free(again)
        free(guard)
        free(guard2)
        show("all freed, unmerged")
    elif scenario == "arenas":
        # Arenas of 4 KiB until two lie next to each other.
        held = [allocate(4096) for _ in range(64)]
        pairs = [(p, q) for p in held for q in held if p + 4096 == q]
        print("two arenas next to each other:", bool(pairs))
        p, q = pairs[0]
        for pointer in held:
            if pointer not in (p, q):
                free(pointer)
        lib.mdrtDeviceTrim()
        free(p)
        free(q)
        show("two neighboring arenas idle")
        larger = allocate(8192)
        print("8 KiB from neither:", larger not in (p, q))
        show("a request of both")
        free(larger)
    elif scenario == "trim":
        base, other = idle_arenas(MIB, 64 * KIB)
        piece = allocate(4096)
        print("the piece is from the smaller arena:", piece == other)
        show("a piece in use, two arenas")
        print("trimmed", lib.mdrtDeviceTrim(), "bytes")
        show("after the trim")
        lib.mgpuMemset32(piece, 7, 1024, None)
        free(piece)
        print("trimmed", lib.mdrtDeviceTrim(), "bytes")
        show("after the second trim")
    elif scenario == "pieces":
        base = idle_arena(MIB)
        small = allocate(4096)
        large = allocate(MIB - 8 * KIB)
        show("two pieces of an arena of 1 MiB in use")
        other = idle_arena(MIB - 8 * KIB)
        show("a second arena idle")
        free(large)
        show("the large piece freed, past the budget")
        free(small)
        show("the first arena idle as a whole")
    elif scenario == "close":
        base = idle_arena(MIB)
        inner = lib.mdrtDeviceActivationOpen()
        lib.mdrtDeviceActivationEnter(inner)
        pieces = [allocate(4096) for _ in range(3)]
        lib.mdrtDeviceActivationLeave()
        lib.mdrtDeviceActivationClose(inner)
        lib.mdrtDeviceActivationEnter(a)
        show("an activation of three pieces closed")
    elif scenario == "off":
        base = idle_arena(MIB)
        show("an arena freed with the pool off")
        small = allocate(1000)
        show("a smaller request after it")
        free(small)
        show("freed")
    elif scenario == "replaced":
        handle = lib.mdrtMatrixCreate(256, 1)
        show("a matrix in an arena of its own")
        lib.mdrtMatrixGrow(handle, 2)
        show("grown: the old rows back to the driver")
        base = idle_arena(MIB)
        keep = allocate(4096)
        handle = lib.mdrtMatrixCreate(256, 1)
        show("a matrix in a piece of an arena")
        lib.mdrtMatrixGrow(handle, 1000)
        show("grown: the old rows back to the pool")
    elif scenario == "streams":
        import time
        lib.mgpuStreamCreate.restype = ctypes.c_void_p
        lib.mgpuStreamCreate()  # the runtime's context, current here
        cuda = ctypes.CDLL("libcuda.so.1")
        streams = [ctypes.c_void_p(), ctypes.c_void_p()]
        for stream in streams:
            assert cuda.cuStreamCreate(ctypes.byref(stream), 1) == 0  # non-blocking
        host_function = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: time.sleep(0.5))
        n = MIB // 4

        def read(pointer):
            values = (ctypes.c_uint32 * n)()
            assert cuda.cuMemcpyDtoH_v2(values, ctypes.c_uint64(pointer), ctypes.c_size_t(MIB)) == 0
            return set(values)

        previous = lib.mdrtUseStream(streams[0])
        base = idle_arena(MIB)
        early, late = allocate(MIB // 2), allocate(MIB // 4)
        # The late piece is freed with no work pending. On stream A the early
        # piece is written after half a second, then freed.
        free(late)
        assert cuda.cuLaunchHostFunc(streams[0], host_function, None) == 0
        lib.mgpuMemset32(early, 1, n // 2, streams[0])
        free(early)
        show("the early piece freed while stream A writes it")
        inner = lib.mdrtUseStream(streams[1])
        whole = allocate(MIB)
        print("the merged arena taken on stream B:", whole == base)
        lib.mgpuMemset32(whole, 2, n, streams[1])
        assert cuda.cuStreamSynchronize(streams[1]) == 0
        assert cuda.cuStreamSynchronize(streams[0]) == 0
        print("the values after both streams:", sorted(read(whole)))
        lib.mdrtRestoreStream(inner)
        lib.mdrtRestoreStream(previous)
        free(whole)
    close(a)
    lib.mdrtDeviceTrim()
    show("closed and trimmed")
print(f"device blocks {scenario} passed")

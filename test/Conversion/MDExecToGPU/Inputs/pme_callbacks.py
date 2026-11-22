"""Check cuFFT loads on an impulse and a single Nyquist mode."""
import ctypes as c
import math
import sys

rt = c.CDLL(sys.argv[1])
cuda = c.CDLL("libcuda.so.1")
ptr = c.c_void_p
word = c.c_int64


def bind(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = args
    return fn


stream = bind(rt, "mgpuStreamCreate", ptr)()
alloc = bind(cuda, "cuMemAlloc_v2", c.c_int, c.POINTER(ptr), c.c_size_t)
upload = bind(cuda, "cuMemcpyHtoD_v2", c.c_int, ptr, ptr, c.c_size_t)
download = bind(cuda, "cuMemcpyDtoH_v2", c.c_int, ptr, ptr, c.c_size_t)
sync = bind(cuda, "cuStreamSynchronize", c.c_int, ptr)
prepare = bind(rt, "mdrtCudaPMEPrepare", ptr, word, word, word, word, word, ptr)
trials = bind(rt, "mdrtCudaPMETrials", word, ptr)
begin = bind(rt, "mdrtCudaPMEBegin", word, ptr, word)
execute = bind(rt, "mdrtCudaPMEFFT", None, ptr, word, ptr, ptr)
end = bind(rt, "mdrtCudaPMEEnd", None, ptr, word)
release = bind(rt, "mdrtCudaPMERelease", None, ptr)


def device(host):
    addr = ptr()
    assert alloc(c.byref(addr), c.sizeof(host)) == 0
    assert upload(addr, c.byref(host), c.sizeof(host)) == 0
    return addr


class Info(c.Structure):
    _fields_ = [("reciprocals", c.c_uint64)] + [(key, ptr) for key in ("wide", "flag", "tables")]
    _fields_ += [(key, word) for key in ("x", "y", "z")]
    _fields_ += [(key, c.c_double) for key in ("xx", "yx", "yy", "zx", "zy", "zz", "gauss")]


def compare(single, tilted, use_wide):
    k = 8
    scalar = c.c_float if single else c.c_double
    scale = 2**(23 if single else 40)
    # Place a different amplitude in each word array to detect a stale flag.
    narrow = device((c.c_int32 * k**3)(scale if single else 0))
    wide = device((word * k**3)(2 * scale))
    flag = device(c.c_int(use_wide))
    vectors = []
    for axis in range(3):
        frequency = [i if i <= k//2 else i-k for i in range(k)]
        weights = [float(i == (k//2 if axis == 0 else 0)) for i in range(k)]
        vectors.extend(frequency + [n*n for n in frequency] + weights)
    tables = device((scalar * len(vectors))(*vectors))
    divisor = lambda d: ((1 << 32) + d - 1) // d
    reciprocals = divisor(k) | (divisor(k//2 + 1) << 32)
    info = Info(reciprocals, wide, flag, tables, k, k, k, 1, .25, 1, .5, .125, 1, .01)
    metadata = device(info)
    spectrum = (scalar * (2 * k*k*(k//2+1)))()
    mesh = (scalar * k**3)()
    freq, real = device(spectrum), device(mesh)
    state = prepare(k, k, k, single, tilted, metadata)
    count = trials(state)
    modes = set()
    for step in range(count):
        mode = begin(state, step)
        modes.add(mode)
        if mode & 1:
            execute(state, 0, narrow if single else wide, freq)
            assert sync(stream) == 0
            assert download(spectrum, freq, c.sizeof(spectrum)) == 0
            amplitude = 2 if use_wide else 1
            assert all(abs(v - (0 if i % 2 else amplitude)) < 1e-6 for i, v in enumerate(spectrum))
        if mode & 2:
            # A unit spectrum, then a refreshed cell on the same plan.
            for i in range(len(spectrum)):
                spectrum[i] = float(i % 2 == 0)
            assert upload(freq, spectrum, c.sizeof(spectrum)) == 0
            info.xx = 1 + step / 16
            assert upload(metadata, c.byref(info), c.sizeof(info)) == 0
            execute(state, 1, freq, real)
            assert sync(stream) == 0
            assert download(mesh, real, c.sizeof(mesh)) == 0
            squared = (k//2)**2
            weight = 1/squared
            if tilted:
                squared *= info.xx**2 + info.yx**2 + info.zx**2
                weight = math.exp(-info.gauss*squared)/squared
            assert all(abs(v - (-1)**(i//(k*k))*weight) < (1e-6 if single else 1e-12)
                       for i,v in enumerate(mesh))
        end(state, step)
    winner = begin(state, count)
    assert winner in (modes or {0})
    release(state)
    assert trials(prepare(k, k, k, single, tilted, metadata)) == 0
    release(state)
    print(single, tilted, use_wide, sorted(modes), winner)


for single, tilted, wide in ((1, 0, 0), (1, 1, 1), (0, 1, 1)):
    compare(single, tilted, wide)

# Force each policy by delaying one candidate between its two event records.
# A different particle count on the same grid must get an independent trial.
import time

path_prepare = bind(rt, "mdrtCudaPMEPathPrepare", ptr, *([word] * 9))
path_trials = bind(rt, "mdrtCudaPMEPathTrials", word, ptr)
path_begin = bind(rt, "mdrtCudaPMEPathBegin", word, ptr, word)
path_end = bind(rt, "mdrtCudaPMEPathEnd", None, ptr, word)
for wanted in (0, 1):
    key = [40, 40, 40, 10000 + wanted, -1, 1, 0, 8, 0]
    state = path_prepare(*key)
    assert path_trials(state) == 8
    for i in range(8):
        if path_begin(state, i) != wanted:
            time.sleep(0.01)
        path_end(state, i)
    assert path_prepare(*key) == state and path_trials(state) == 0
    assert path_begin(state, 0) == wanted
    for field in (0, 1, 2, 4, 5, 6, 7, 8):
        changed = key.copy()
        changed[field] += 1
        assert path_trials(path_prepare(*changed)) == 8

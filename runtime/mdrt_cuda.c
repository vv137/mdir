/* The MDIR runtime for NVIDIA GPUs.

   These are the functions that the lowering of the upstream gpu dialect
   calls. Their names and signatures are fixed by that lowering. They are
   implemented on the CUDA driver API, so the library needs the driver and
   the header of the toolkit, and nothing else of the toolkit. */

#include <cuda.h>
#include "mdir/Runtime/GroupsABI.h"
#include "mdir/Runtime/CopyAudit.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The kernel that was launched last, and its name, which a failure names:
   a kernel that fails is reported by the driver at the next call that
   waits for it. */
static CUfunction lastFunction = NULL;
static __thread struct MdirDeviceCopyAudit *copyAudit = NULL;
static __thread uintptr_t copyMetadataDestination = 0;
static void auditGenericCopy(void *destination, const void *source, uint64_t size);
void *mdrtDeviceAuditCopies(void *audit) {
  void *previous = copyAudit;
  copyAudit = audit;
  copyMetadataDestination = 0;
  return previous;
}
void mdrtDeviceAuditMetadataCopy(uintptr_t destination) {
  if (copyAudit) copyMetadataDestination = destination;
}
static const char *getKernelName(CUfunction function);
void mdrtSpansFlush(void);
/* The number of arguments of the launch in progress, -1 if unknown. */
static int64_t launchCount = -1;
static void releaseAllSpans(void);

/* A failure of the driver is reported and ends the run: compiled code has no
   way to continue without its device. */
/* What the host runs once before a failure of the driver ends the process
   (mdrtSetFailureHook): the writer of the frames writes those whose copies
   had arrived (D400). */
static void (*failureHook)(void) = NULL;

static void check(CUresult result, const char *what) {
  if (result == CUDA_SUCCESS)
    return;
  static int failing = 0;
  if (failureHook && !failing) {
    failing = 1;
    failureHook();
  }
  const char *name = NULL;
  cuGetErrorName(result, &name);
  fprintf(stderr, "mdrt: %s failed with %s\n", what,
          name ? name : "an unknown error");
  if (lastFunction)
    fprintf(stderr,
            "mdrt: the last kernel launched was %s; the driver reports the "
            "failure of a kernel at a later call, and with MDRT_WAIT=1 each "
            "launch is waited for, so that the report names the kernel that "
            "failed. compute-sanitizer names the access; see "
            "docs/debugging.md\n",
            getKernelName(lastFunction));
  fflush(stderr);
  abort();
}

/*===----------------------------------------------------------------------===
  Profile
  ===----------------------------------------------------------------------===

  With MDRT_PROFILE set, the library counts its calls and the time they
  take, and reports when the program ends. With MDRT_WAIT set as well, it
  reports the time of each kernel: the time from the launch to the end of
  the kernel. With MDRT_TRACE set, it prints each module that it loads, the
  kernels of each module, and each launch with its grid, so that a kernel
  that a tool such as compute-sanitizer reports can be found in the module
  that the pass `gpu-kernel-outlining` prints (MDIR_PRINT_AFTER). */

enum { LAUNCH, WAIT, COPY, ALLOCATE, LOOKUP, NUM_COUNTERS };
static const char *const counterNames[NUM_COUNTERS] = {
    "launches", "waits", "copies", "allocations", "lookups of kernels"};
static int64_t counts[NUM_COUNTERS];
static double seconds[NUM_COUNTERS];
static int profile = -1;

static double now(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (double)time.tv_sec + 1.0e-9 * (double)time.tv_nsec;
}

/* The kernels that were launched. */
enum { MAX_KERNELS = 1024 };
struct Kernel {
  CUfunction function;
  char name[128];
  int64_t count;
  double seconds;
  /* The threads of its last launch, which tell what it runs over. */
  int64_t threads;
  /* Whether it was launched (the accounts of MDIR_DEVICE_MEMORY=2). */
  int launched;
};
static struct Kernel kernels[MAX_KERNELS];
static int numKernels = 0;

/* The kernel that was launched last, and when; -1 if its time is
   accounted for. */
static int lastKernel = -1;
static double lastLaunch = 0.0;

static void report(void) {
  fprintf(stderr, "mdrt: %-40s %10s %12s %12s %10s\n", "", "calls",
          "seconds", "microseconds", "threads");
  for (int i = 0; i != NUM_COUNTERS; ++i)
    fprintf(stderr, "mdrt: %-40s %10lld %12.4f %12.2f\n", counterNames[i],
            (long long)counts[i], seconds[i],
            counts[i] ? 1.0e6 * seconds[i] / (double)counts[i] : 0.0);
  for (int i = 0; i != numKernels; ++i)
    if (kernels[i].count)
      fprintf(stderr, "mdrt: %-40s %10lld %12.4f %12.2f %10lld\n",
              kernels[i].name, (long long)kernels[i].count,
              kernels[i].seconds,
              1.0e6 * kernels[i].seconds / (double)kernels[i].count,
              (long long)kernels[i].threads);
}

/* The copies from the device to the host and the waits of the host for the
   device, counted with MDRT_HOST_READS set and reported at the exit
   (D452): what the tests of the coupling step on the device
   count, whatever submission path the run takes. */
static int hostReads = -1;
static long long readCopies, readBytes, readWaits, flagWaits;

enum { READ_SIZES = 16 };
static long long readSize[READ_SIZES], readSizeCount[READ_SIZES];

static void reportHostReads(void) {
  fprintf(stderr,
          "mdrt: host reads: %lld copies to the host (%lld bytes), %lld "
          "waits for the device, %lld waits for flags\n",
          readCopies, readBytes, readWaits, flagWaits);
  for (int i = 0; i != READ_SIZES && readSizeCount[i]; ++i)
    fprintf(stderr, "mdrt: host reads of %lld bytes: %lld\n", readSize[i],
            readSizeCount[i]);
}

void mdrtCountHostRead(size_t bytes, int waits, int flags) {
  if (hostReads < 0) {
    hostReads = getenv("MDRT_HOST_READS") != NULL;
    if (hostReads)
      atexit(reportHostReads);
  }
  if (!hostReads)
    return;
  if (bytes) {
    ++readCopies;
    readBytes += (long long)bytes;
    for (int i = 0; i != READ_SIZES; ++i)
      if (!readSizeCount[i] || readSize[i] == (long long)bytes) {
        readSize[i] = (long long)bytes;
        ++readSizeCount[i];
        break;
      }
  }
  readWaits += waits;
  flagWaits += flags;
}

/* Notes that `function` has the name `name`. */
static void noteKernel(CUfunction function, const char *name) {
  for (int i = 0; i != numKernels; ++i)
    if (kernels[i].function == function)
      return;
  if (numKernels == MAX_KERNELS)
    return;
  kernels[numKernels].function = function;
  snprintf(kernels[numKernels].name, sizeof(kernels[numKernels].name), "%s",
           name);
  ++numKernels;
}

static int findKernel(CUfunction function) {
  for (int i = 0; i != numKernels; ++i)
    if (kernels[i].function == function)
      return i;
  return -1;
}

static const char *getKernelName(CUfunction function) {
  int i = findKernel(function);
  return i < 0 ? "a kernel without a name" : kernels[i].name;
}

/* The time at which a call begins, or a negative number if the profile is
   off. */
static double begin(void) {
  if (profile < 0) {
    profile = getenv("MDRT_PROFILE") != NULL;
    if (profile)
      atexit(report);
  }
  return profile ? now() : -1.0;
}

static void end(int counter, double start) {
  if (start < 0.0)
    return;
  counts[counter] += 1;
  seconds[counter] += now() - start;
}

static int32_t defaultDevice = 0;
static CUcontext context = NULL;

/* One stream serves all launches and copies, so the device runs them in the
   order in which they were issued. The lowering waits after every launch.
   The library does not: the host goes on issuing work while the device
   runs, and waits only where it reads what the device has computed. With
   MDRT_WAIT set, the library waits after every launch, which tells an
   error of the order apart from other errors and names the kernel that
   fails. */
static CUstream sharedStream = NULL;
/* A caller may supply the legacy default stream (NULL), so a borrowed
   stream cannot be distinguished from an uninitialized owned stream by
   its handle. Scopes are serialized with the rest of the runtime. */
static unsigned borrowedStreamDepth = 0;
static CUevent streamScopeEvent = NULL;
static int waitsAlways = -1;

/* Side streams, on which work that is independent of what follows it on
   the first can run beside it (mdrtSideBegin). Work issued between
   mdrtSideBegin and mdrtSideEnd goes to the second stream, after what the
   first stream had been given; between mdrtSide2Begin and mdrtSide2End
   to the third, beside the first two (D449). mdrtSideJoin
   makes the first stream wait for both. The forks and the joins are
   events recorded and waited for on the streams, so that a capture of the
   first stream into a graph takes the side streams in as branches that
   end at the join (CUDA programming guide, stream capture across
   streams). */
enum { SIDE_LANES = 2 };
static CUstream sideStreams[SIDE_LANES];
static int sideLane = -1;
#define onSide (sideLane >= 0)
static CUevent forkEvent = NULL, joinEvents[SIDE_LANES];
static int joinPending[SIDE_LANES];

/* Whether the device may still be running work that was issued. */
static int isPending = 0;

/* With MDIR_DEVICE_MEMORY=2 (D311), the runtime also
   keeps accounts of the device memory that its pool of blocks does not
   hold: at each event that may take memory of the device outside the pool,
   how much the driver's use of the device grew (cuMemGetInfo, the total
   less the free, which counts every process on the device), and how many
   bytes the runtime asked for there. The events are the making of the
   context, the loads of modules, the first launch of each kernel (the
   driver reserves the local memory of the threads there), the plans of
   cuFFT and their workspaces, the handle of cuBLAS, and the allocations
   outside the pool. The blocks of the pool are an account too: the growth
   at the driver's allocations of blocks against their bytes. */
enum {
  ACCOUNT_CONTEXT,
  ACCOUNT_MODULES,
  ACCOUNT_LAUNCHES,
  ACCOUNT_BLOCKS,
  ACCOUNT_FFT,
  ACCOUNT_BLAS,
  ACCOUNT_TUPLES,
  ACCOUNT_CONSTANTS,
  ACCOUNT_MANAGED,
  ACCOUNT_KEPT,
  ACCOUNT_SPANS,
  NUM_ACCOUNTS
};
static const char *const accountNames[NUM_ACCOUNTS] = {
    "the context",
    "the modules",
    "the first launch of each kernel",
    "the blocks of the pool",
    "the plans of cuFFT",
    "the handle of cuBLAS",
    "the structures of the tuples",
    "the constants of the bodies",
    "the memory shared with the host",
    "the memory of an embedding program",
    "the graphs of spans"};
static int64_t accountGrowth[NUM_ACCOUNTS], accountAsked[NUM_ACCOUNTS];
static int64_t accountEvents[NUM_ACCOUNTS];
static int memoryDetail = -1;
static int isMemoryDetailed(void) {
  if (memoryDetail < 0) {
    const char *value = getenv("MDIR_DEVICE_MEMORY");
    memoryDetail = value && strcmp(value, "2") == 0;
  }
  return memoryDetail;
}
/* The driver's use of the device, in bytes, and the most it was at an
   event; -1 if it cannot be read. */
static int64_t usedMost = 0;
static int64_t getDeviceUsed(void) {
  size_t free = 0, total = 0;
  if (cuMemGetInfo(&free, &total) != CUDA_SUCCESS)
    return -1;
  int64_t used = (int64_t)(total - free);
  if (used > usedMost)
    usedMost = used;
  return used;
}
/* The driver's use before an event of the accounts, or -1 if they are not
   kept. */
static int64_t beginAccount(void) {
  return isMemoryDetailed() ? getDeviceUsed() : -1;
}
static void endAccount(int account, int64_t before, int64_t asked) {
  if (before < 0)
    return;
  int64_t after = getDeviceUsed();
  if (after >= 0)
    accountGrowth[account] += after - before;
  accountAsked[account] += asked;
  accountEvents[account] += 1;
}

/* The driver must support the major version of CUDA that this library was
   built with: under CUDA's minor version compatibility, a driver of that
   major version runs what the toolkit of any of its minor versions makes,
   which is why a release needs a driver of CUDA 13.0 (D177). A driver of an
   older major version crashed inside the driver at the first load of a
   module (#657), so the version is compared before anything is loaded,
   once, as the first context is made, and the run ends with both versions.
   MDRT_DRIVER_VERSION=<n>, for the tests, stands for the version that the
   driver reports, in its encoding (1000 major + 10 minor). */
static void checkDriverVersion(void) {
  int driver = 0;
  const char *forced = getenv("MDRT_DRIVER_VERSION");
  if (forced && *forced)
    driver = (int)strtol(forced, NULL, 10);
  else if (cuDriverGetVersion(&driver) != CUDA_SUCCESS)
    return; /* cuInit reports a driver that does not answer. */
  int needed = CUDA_VERSION / 1000 * 1000;
  if (driver >= needed)
    return;
  fprintf(stderr,
          "mdrt: the NVIDIA driver supports CUDA %d.%d; this build of MDIR, "
          "with CUDA %d.%d, needs a driver that supports CUDA %d.0 or later "
          "(D177). Update the NVIDIA driver, or run on the CPU "
          "(target = \"CPU\")\n",
          driver / 1000, driver % 1000 / 10, CUDA_VERSION / 1000,
          CUDA_VERSION % 1000 / 10, needed / 1000);
  fflush(stderr);
  exit(1);
}

/* Makes the context of the device current for the calling thread. The
   device is the one that MDRT_DEVICE names, among those that
   CUDA_VISIBLE_DEVICES leaves visible, or the first. */
static void enter(void) {
  if (!context) {
    static int checked = 0;
    if (!checked) {
      checkDriverVersion();
      checked = 1;
    }
    check(cuInit(0), "cuInit");
    const char *chosen = getenv("MDRT_DEVICE");
    if (chosen)
      defaultDevice = (int32_t)strtol(chosen, NULL, 10);
    CUdevice device;
    check(cuDeviceGet(&device, defaultDevice), "cuDeviceGet");
    check(cuDevicePrimaryCtxRetain(&context, device),
          "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    /* The driver's use once the context is made: what it was before is
       the idle use of the device, which nvidia-smi reports. */
    if (isMemoryDetailed()) {
      accountGrowth[ACCOUNT_CONTEXT] = getDeviceUsed();
      accountEvents[ACCOUNT_CONTEXT] = 1;
    }
  }
  check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

/* What the products of the bodies of neural stages hold on the device
   (D302, at the end of this file): the copies of the constants
   that `owner` made (an activation, or null outside one; every owner with
   `all`), and the handle of cuBLAS. `gone` says that the driver has shut
   down, and the memory of the device with it. */
static void releaseDeviceConstants(void *owner, int all, int gone);
static void releaseBlas(int gone);

/* A grid with one block per SM fits whenever a single block fits. */
static int cooperativeSMs = -1;

intptr_t mdrtCooperativeGridSize(intptr_t requested) {
  if (cooperativeSMs < 0) {
    enter();
    CUdevice device;
    int supported;
    check(cuCtxGetDevice(&device), "cuCtxGetDevice");
    check(cuDeviceGetAttribute(&supported,
                               CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device),
          "cuDeviceGetAttribute");
    cooperativeSMs = 1;
    if (supported)
      check(cuDeviceGetAttribute(&cooperativeSMs,
                                 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                 device),
            "cuDeviceGetAttribute");
  }
  return requested < cooperativeSMs ? requested : cooperativeSMs;
}

void mgpuSetDefaultDevice(int32_t device) {
  cooperativeSMs = -1;
  defaultDevice = device;
  if (context) {
    /* The handle and the copies belong to the context that is released. */
    int gone = cuCtxSetCurrent(context) != CUDA_SUCCESS;
    releaseDeviceConstants(NULL, 1, gone);
    releaseBlas(gone);
    CUdevice old;
    check(cuCtxGetDevice(&old), "cuCtxGetDevice");
    check(cuDevicePrimaryCtxRelease(old), "cuDevicePrimaryCtxRelease");
    context = NULL;
    sharedStream = NULL;
  }
}

/*===----------------------------------------------------------------------===
  Modules and kernels
  ===----------------------------------------------------------------------===*/

/* The kernels are PTX that the driver compiles at load, so the driver must
   know the PTX ISA version that LLVM wrote; the toolkit supplies only
   libdevice. A driver older than that version fails the load with
   CUDA_ERROR_UNSUPPORTED_PTX_VERSION, or a log that names `.version`. That
   failure ends the run with what to do, not a generic failure. */
static void reportUnsupportedPTX(CUresult result, const void *data,
                                 size_t size, const char *log) {
  if (result != CUDA_ERROR_UNSUPPORTED_PTX_VERSION &&
      !(log && strstr(log, ".version")))
    return;
  /* The `.version` line of the module, within its first kilobyte. */
  char isa[16] = "unknown";
  const char *text = (const char *)data;
  size_t length = size ? size : strnlen(text, 1 << 20);
  for (size_t i = 0; i + 9 < length && i < 1024; ++i)
    if (strncmp(text + i, ".version ", 9) == 0) {
      size_t n = 0;
      for (size_t j = i + 9; j < length && n + 1 < sizeof(isa) &&
                             (text[j] == '.' || (text[j] >= '0' && text[j] <= '9'));
           ++j)
        isa[n++] = text[j];
      isa[n] = '\0';
      break;
    }
  int driver = 0;
  cuDriverGetVersion(&driver);
  fprintf(stderr,
          "mdrt: the NVIDIA driver supports CUDA %d.%d, which cannot compile "
          "kernels of PTX ISA %s, the version that this build of MDIR emits. "
          "Update the NVIDIA driver to one that supports PTX ISA %s, or run "
          "on the CPU (target = \"CPU\")\n",
          driver / 1000, driver % 1000 / 10, isa, isa);
  fflush(stderr);
  exit(1);
}

CUmodule mgpuModuleLoad(void *data, size_t size) {
  enter();
  CUmodule module = NULL;
  int64_t before = beginAccount();
  CUresult result = cuModuleLoadData(&module, data);
  endAccount(ACCOUNT_MODULES, before, 0);
  if (result != CUDA_SUCCESS)
    reportUnsupportedPTX(result, data, size, NULL);
  /* A cubin is compiled for the architecture of the device that the
     compiler asked the driver for (D214). */
  if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
    fprintf(stderr,
            "mdrt: the kernels were compiled for the architecture of another "
            "device than the one that runs them; compile the program with "
            "the device that runs it selected (MDRT_DEVICE or the device of "
            "the execution)\n");
    fflush(stderr);
  }
  check(result, "cuModuleLoadData");
  static int loads = 0;
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE module %p #%d\n", (void *)module, loads++);
  return module;
}

/* Loads a module from PTX text, which the driver compiles. */
CUmodule mgpuModuleLoadJIT(void *data, int optLevel, size_t size) {
  enter();
  char log[4096] = {0};
  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_OPTIMIZATION_LEVEL};
  void *values[] = {(void *)log, (void *)(uintptr_t)sizeof(log),
                    (void *)(uintptr_t)optLevel};
  CUmodule module = NULL;
  int64_t before = beginAccount();
  CUresult result = cuModuleLoadDataEx(&module, data, 3, options, values);
  endAccount(ACCOUNT_MODULES, before, 0);
  if (result != CUDA_SUCCESS) {
    reportUnsupportedPTX(result, data, size, log);
    fprintf(stderr, "mdrt: the driver could not compile a kernel:\n%s\n",
            log);
  }
  check(result, "cuModuleLoadDataEx");
  static int jitLoads = 0;
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE module %p #%d\n", (void *)module, jitLoads++);
  return module;
}

void mgpuModuleUnload(CUmodule module) {
  /* This runs when the program ends, possibly after the driver has shut
     down. */
  mdrtSpansFlush();
  if (isPending && (sharedStream || borrowedStreamDepth))
    cuStreamSynchronize(sharedStream);
  isPending = 0;
  /* The graphs of spans and the layouts of the arguments name the kernels
     of the program; another program may later have kernels at the same
     handles. */
  releaseAllSpans();
  /* The program ends: the copies of its constants made outside an
     activation go with it, since another program may later put its own
     constants at the same addresses of the host. A program that is still
     loaded copies its own again when it next needs them. */
  {
    CUresult current = context ? cuCtxSetCurrent(context) : CUDA_SUCCESS;
    releaseDeviceConstants(NULL, 0,
                           !context || current != CUDA_SUCCESS);
  }
  CUresult result = cuModuleUnload(module);
  if (result != CUDA_ERROR_DEINITIALIZED &&
      result != CUDA_ERROR_INVALID_CONTEXT &&
      result != CUDA_ERROR_CONTEXT_IS_DESTROYED)
    check(result, "cuModuleUnload");
}

CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  double start = begin();
  CUfunction function = NULL;
  check(cuModuleGetFunction(&function, module, name), "cuModuleGetFunction");
  const char *dynamic = strncmp(name, "towers_", 7) == 0
                            ? strstr(name, "_dynamic_")
                            : NULL;
  if (dynamic)
    check(cuFuncSetAttribute(function,
                             CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             atoi(dynamic + 9)), "cuFuncSetAttribute");
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE function %p module %p %s\n", (void *)function,
            (void *)module, name);
  noteKernel(function, name);
  end(LOOKUP, start);
  return function;
}

static void finish(void);
static void endKernel(void);
static int recordLaunch(CUfunction function, intptr_t gridX, intptr_t gridY,
                        intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                        intptr_t blockZ, int32_t sharedMemory,
                        CUstream stream, void **parameters, void **extra,
                        int cooperative, intptr_t clusterX, intptr_t clusterY,
                        intptr_t clusterZ);
void mdrtSpansFlush(void);
static int isRecording(void);

static void launchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                         intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                         intptr_t blockZ, int32_t sharedMemory, CUstream stream,
                         void **parameters, void **extra, int cooperative,
                         intptr_t clusterX, intptr_t clusterY,
                         intptr_t clusterZ) {
  /* A launch over no particles is no launch. The driver rejects it. */
  if (gridX <= 0 || gridY <= 0 || gridZ <= 0)
    return;
  if (recordLaunch(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                   sharedMemory, stream, parameters, extra, cooperative,
                   clusterX, clusterY, clusterZ))
    return;
  /* A launch that is not recorded follows what was. */
  mdrtSpansFlush();
  enter();
  double start = begin();
  static long traceCount = 0;
  if (getenv("MDRT_TRACE"))
    fprintf(stderr, "TRACE launch %ld fn %p grid %ld block %ld\n",
            traceCount++, (void *)function, (long)gridX, (long)blockX);
  int64_t before = -1;
  int launching = -1;
  if (isMemoryDetailed()) {
    launching = findKernel(function);
    if (launching >= 0 && !kernels[launching].launched)
      before = getDeviceUsed();
  }
  if (clusterX > 0 && clusterY > 0 && clusterZ > 0) {
#if CUDA_VERSION >= 12000
    CUlaunchAttribute attrs[2] = {0};
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[0].value.clusterDim.x = clusterX;
    attrs[0].value.clusterDim.y = clusterY;
    attrs[0].value.clusterDim.z = clusterZ;
    attrs[1].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attrs[1].value.cooperative = cooperative;
    CUlaunchConfig config = {.gridDimX = gridX,
                             .gridDimY = gridY,
                             .gridDimZ = gridZ,
                             .blockDimX = blockX,
                             .blockDimY = blockY,
                             .blockDimZ = blockZ,
                             .sharedMemBytes = sharedMemory,
                             .hStream = stream,
                             .attrs = attrs,
                             .numAttrs = 2};
    check(cuLaunchKernelEx(&config, function, parameters, extra),
          "cuLaunchKernelEx");
#else
    check(CUDA_ERROR_NOT_SUPPORTED, "thread block clusters require CUDA 12");
#endif
  } else if (cooperative && (gridX > 1 || gridY > 1 || gridZ > 1)) {
    check(cuLaunchCooperativeKernel(function, gridX, gridY, gridZ, blockX,
                                    blockY, blockZ, sharedMemory, stream,
                                    parameters),
          "cuLaunchCooperativeKernel");
  } else {
    check(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                         sharedMemory, stream, parameters, extra),
          "cuLaunchKernel");
  }
  if (before >= 0) {
    kernels[launching].launched = 1;
    endAccount(ACCOUNT_LAUNCHES, before, 0);
  }
  lastFunction = function;
  isPending = 1;
  if (start >= 0.0) {
    lastKernel = findKernel(function);
    lastLaunch = start;
    if (lastKernel >= 0)
      kernels[lastKernel].threads =
          (int64_t)(gridX * gridY * gridZ * blockX * blockY * blockZ);
  }
  end(LAUNCH, start);
  /* With MDRT_WAIT set, every launch is waited for, so that a failure is
     reported with the kernel that failed. */
  if (waitsAlways < 0)
    waitsAlways = getenv("MDRT_WAIT") != NULL;
  if (waitsAlways) {
    finish();
    endKernel();
  }
}

void mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                      intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                      intptr_t blockZ, int32_t sharedMemory, CUstream stream,
                      void **parameters, void **extra, size_t count) {
  launchCount = (int64_t)count;
  launchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
               sharedMemory, stream, parameters, extra, 0, 0, 0, 0);
  launchCount = -1;
}

void mgpuLaunchKernelCooperative(CUfunction function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t clusterX, intptr_t clusterY,
                                 intptr_t clusterZ, intptr_t blockX,
                                 intptr_t blockY, intptr_t blockZ,
                                 int32_t sharedMemory, CUstream stream,
                                 void **parameters, void **extra) {
  launchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
               sharedMemory, stream, parameters, extra, 1, clusterX, clusterY,
               clusterZ);
}

/*===----------------------------------------------------------------------===
  Streams and events
  ===----------------------------------------------------------------------===*/

CUstream mgpuStreamCreate(void) {
  enter();
  if (!sharedStream && !borrowedStreamDepth)
    check(cuStreamCreate(&sharedStream, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate");
  return onSide ? sideStreams[sideLane] : sharedStream;
}

static void sideBegin(int lane) {
  mgpuStreamCreate();
  /* A recorded span ends where a side stream forks: the work of the
     side streams is issued as launched (D448), for either lane. */
  mdrtSpansFlush();
  if (!forkEvent)
    check(cuEventCreate(&forkEvent, CU_EVENT_DISABLE_TIMING),
          "cuEventCreate");
  if (!sideStreams[lane]) {
    check(cuStreamCreate(&sideStreams[lane], CU_STREAM_NON_BLOCKING),
          "cuStreamCreate");
    check(cuEventCreate(&joinEvents[lane], CU_EVENT_DISABLE_TIMING),
          "cuEventCreate");
  }
  check(cuEventRecord(forkEvent, sharedStream), "cuEventRecord");
  check(cuStreamWaitEvent(sideStreams[lane], forkEvent, 0),
        "cuStreamWaitEvent");
  sideLane = lane;
}

static void sideEnd(int lane) {
  check(cuEventRecord(joinEvents[lane], sideStreams[lane]), "cuEventRecord");
  sideLane = -1;
  joinPending[lane] = 1;
}

void mdrtSideBegin(void) { sideBegin(0); }
void mdrtSideEnd(void) { sideEnd(0); }
void mdrtSide2Begin(void) { sideBegin(1); }
void mdrtSide2End(void) { sideEnd(1); }

void mdrtSideJoin(void) {
  for (int lane = 0; lane != SIDE_LANES; ++lane) {
    if (!joinPending[lane])
      continue;
    check(cuStreamWaitEvent(sharedStream, joinEvents[lane], 0),
          "cuStreamWaitEvent");
    joinPending[lane] = 0;
  }
}

/* Transfer the ordering of the runtime's work without synchronizing the
   host. An event wait observes the most recent record at the time it is
   issued; the same event can therefore serve nested, serialized scopes. */
static void changeStream(CUstream next) {
  mdrtSpansFlush();
  mdrtSideJoin();
  if (next != sharedStream) {
    if (!streamScopeEvent)
      check(cuEventCreate(&streamScopeEvent, CU_EVENT_DISABLE_TIMING),
            "cuEventCreate");
    check(cuEventRecord(streamScopeEvent, sharedStream), "cuEventRecord");
    check(cuStreamWaitEvent(next, streamScopeEvent, 0), "cuStreamWaitEvent");
    sharedStream = next;
  }
}

/* The caller owns `stream` and keeps it alive through the matching
   restore. Launches, copies, FFT plans and side streams all use it. */
void *mdrtUseStream(void *stream) {
  mgpuStreamCreate();
  CUstream previous = sharedStream;
  changeStream((CUstream)stream);
  ++borrowedStreamDepth;
  return (void *)previous;
}

void mdrtRestoreStream(void *previous) {
  changeStream((CUstream)previous);
  --borrowedStreamDepth;
}

void mgpuStreamDestroy(CUstream stream) { (void)stream; }

/*===----------------------------------------------------------------------===
  Recorded spans (D448)
  ===----------------------------------------------------------------------===

  With spans on (mdrtSpansEnable), the work that the host issues on
  the stream of the kernels between two host boundaries is not issued at
  once: the launches, the transforms of cuFFT, the sets of memory and the
  events of the flags are recorded, with the values of their arguments,
  and issued together at the next boundary. A boundary is any call that
  waits for the device, reads or writes memory that the device uses, or
  issues work outside the record: the wait of a flag (mdrtFlagFinish), a
  copy, a change of stream, a growth of a structure. A sequence that was
  recorded before, with the same kernels, grids and argument bytes, is
  issued as the CUDA graph made from it at its second recording; another is
  issued as it was recorded. The device runs the same kernels in the same
  order, with the same arguments, on the one stream, so the results are
  bitwise those of the launches one by one. What the graph removes is the
  gap of the device between two launches and the host's time in each
  launch (#60, #62).

  The record is filled only from the thread of the steps and drained before
  any wait, so no work the host waits for is held back. */

enum { SPAN_KERNEL, SPAN_EVENT, SPAN_FFT, SPAN_MEMSET32, SPAN_MEMSET16 };

struct SpanOp {
  int kind;
  /* A kernel: its function, grid, block, dynamic shared memory, whether it
     is cooperative, its cluster, and its argument bytes in the arena. */
  CUfunction function;
  uint32_t grid[3], block[3], cluster[3];
  uint32_t shared;
  int32_t cooperative;
  uint32_t offset, bytes;
  /* An event of a flag; a transform: its kind (0 D2Z, 1 Z2D, 2 R2C, 3 C2R)
     and plan; a set: its value and count. */
  CUevent event;
  int32_t fft, plan;
  uint64_t a, b, count;
  uint32_t value;
};

/* The layout of the arguments of a kernel (cuFuncGetParamInfo). */
struct ParamLayout {
  CUfunction function;
  uint32_t count, total;
  uint32_t *sizes;
  /* Whether cuFuncGetParamInfo ended at the index past the last argument
     (CUDA_ERROR_INVALID_VALUE) rather than at another error. */
  int complete;
};

struct Span {
  /* The hash of the ops and argument bytes the graph holds now, and of its
     shape: the ops without their argument bytes and grids, which an
     update of the graph's kernel nodes can change. */
  uint64_t hash, shape;
  int numOps;
  struct SpanOp *ops;
  uint32_t numBytes;
  unsigned char *bytes;
  CUgraph graph;
  CUgraphExec exec;
  /* The kernel node of each op of the graph, NULL for the others; whether
     the kernels could all be found in order, so that the graph can take the
     arguments and grids of another span of its shape. */
  CUgraphNode *nodes;
  int updatable;
  /* A graph whose update the driver refused: some of its nodes may hold
     the new arguments, so it matches no record again. */
  int dead;
  uint64_t lastUse;
};

enum { SPAN_CACHE = 1024, SPAN_INDEX = 4096, SPAN_SEEN = 16384, SPAN_VARIANTS = 16,
       LAYOUTS = 4096 };
static int spansOn = 0;
static int spansSuspended = 0;
static struct SpanOp *spanOps = NULL;
static int numSpanOps = 0, roomSpanOps = 0;
static unsigned char *spanBytes = NULL;
static uint32_t numSpanBytes = 0, roomSpanBytes = 0;
static struct Span spanCache[SPAN_CACHE];
static int numSpans = 0;
static uint64_t spanSeen[SPAN_SEEN];
static uint64_t spanClock = 0;
static struct ParamLayout layouts[LAYOUTS];
static int64_t spanCounts[9];

static int issueFFT(int fft, int plan, uint64_t in, uint64_t out);
static void retainFFTPlanHandle(int plan, int delta);

static int spanSplitAtFlags(void) {
  static int split = -1;
  if (split < 0) {
    const char *value = getenv("MDRT_SPANS_SPLIT");
    split = !(value && strcmp(value, "0") == 0);
  }
  return split;
}

/* MDRT_WAIT, MDRT_PROFILE and MDRT_TRACE, which observe each launch, turn
   the record off; the driver applies MDRT_SPANS and records the choice
   (resolved_choices.step_graph.spans). */
static int spansOverride(int on) {
  if (getenv("MDRT_WAIT") || getenv("MDRT_PROFILE") || getenv("MDRT_TRACE"))
    on = 0;
  return on;
}

/* Nothing records from a fork to its join, on any side lane: the first
   stream waits there for work that no span holds. */
static int isRecording(void) {
  int joining = 0;
  for (int lane = 0; lane != SIDE_LANES; ++lane)
    joining |= joinPending[lane];
  return spansOn > 0 && !spansSuspended && !onSide && !joining &&
         !borrowedStreamDepth;
}

static struct ParamLayout *getLayout(CUfunction function) {
  uint64_t h = ((uint64_t)(uintptr_t)function >> 4) * 0x9E3779B97F4A7C15ull;
  for (uint32_t probe = 0; probe != LAYOUTS; ++probe) {
    struct ParamLayout *l = &layouts[(h + probe) % LAYOUTS];
    if (l->function == function)
      return l;
    if (l->function)
      continue;
    size_t offset, size;
    uint32_t count = 0, room = 16;
    uint32_t *sizes = malloc(room * sizeof *sizes), total = 0;
    CUresult result;
    while ((result = cuFuncGetParamInfo(function, count, &offset, &size)) ==
           CUDA_SUCCESS) {
      if (count == room)
        sizes = realloc(sizes, (room *= 2) * sizeof *sizes);
      sizes[count++] = (uint32_t)size;
      total += ((uint32_t)size + 7u) & ~7u;
    }
    l->complete = result == CUDA_ERROR_INVALID_VALUE &&
                  getenv("MDRT_SPANS_LAYOUT_FAULT") == NULL;
    l->function = function;
    l->count = count;
    l->total = total;
    l->sizes = sizes;
    return l;
  }
  return NULL;
}

static struct SpanOp *appendOp(int kind) {
  if (numSpanOps == roomSpanOps) {
    roomSpanOps = roomSpanOps ? 2 * roomSpanOps : 64;
    spanOps = realloc(spanOps, roomSpanOps * sizeof *spanOps);
  }
  struct SpanOp *op = &spanOps[numSpanOps++];
  memset(op, 0, sizeof *op);
  op->kind = kind;
  return op;
}

static unsigned char *appendBytes(uint32_t size) {
  if (numSpanBytes + size > roomSpanBytes) {
    while (numSpanBytes + size > roomSpanBytes)
      roomSpanBytes = roomSpanBytes ? 2 * roomSpanBytes : 16384;
    spanBytes = realloc(spanBytes, roomSpanBytes);
  }
  unsigned char *at = spanBytes + numSpanBytes;
  numSpanBytes += size;
  return at;
}

/* Records a launch; 0 if it must be issued now instead. */
static int recordLaunch(CUfunction function, intptr_t gridX, intptr_t gridY,
                        intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                        intptr_t blockZ, int32_t sharedMemory,
                        CUstream stream, void **parameters, void **extra,
                        int cooperative, intptr_t clusterX, intptr_t clusterY,
                        intptr_t clusterZ) {
  if (!isRecording() || extra || (stream && stream != sharedStream))
    return 0;
  struct ParamLayout *layout = getLayout(function);
  /* A layout that the driver did not give in full, or that disagrees with
     the number of arguments of the launch, is not recorded: the launch goes
     as it is. */
  if (!layout || !layout->complete ||
      (launchCount >= 0 && launchCount != (int64_t)layout->count)) {
    ++spanCounts[8];
    return 0;
  }
  struct SpanOp *op = appendOp(SPAN_KERNEL);
  op->function = function;
  op->grid[0] = (uint32_t)gridX, op->grid[1] = (uint32_t)gridY,
  op->grid[2] = (uint32_t)gridZ;
  op->block[0] = (uint32_t)blockX, op->block[1] = (uint32_t)blockY,
  op->block[2] = (uint32_t)blockZ;
  op->shared = (uint32_t)sharedMemory;
  op->cooperative = cooperative && (gridX > 1 || gridY > 1 || gridZ > 1);
  if (clusterX > 0 && clusterY > 0 && clusterZ > 0) {
    op->cluster[0] = (uint32_t)clusterX, op->cluster[1] = (uint32_t)clusterY,
    op->cluster[2] = (uint32_t)clusterZ;
    op->cooperative = cooperative;
  }
  op->offset = numSpanBytes;
  op->bytes = layout->total;
  unsigned char *at = appendBytes(layout->total);
  /* The padding is zero, so that equal arguments compare equal. */
  memset(at, 0, layout->total);
  for (uint32_t i = 0; i != layout->count; ++i) {
    memcpy(at, parameters[i], layout->sizes[i]);
    at += (layout->sizes[i] + 7u) & ~7u;
  }
  lastFunction = function;
  isPending = 1;
  return 1;
}

/* A hash of 8-byte words (the arguments are mostly pointers and sizes):
   the record of a step is some kilobytes, which a hash of single bytes took
   tens of microseconds to read. */
static uint64_t hashBytes(uint64_t h, const void *data, size_t size) {
  const unsigned char *p = data;
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    uint64_t w;
    memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
  }
  for (; i != size; ++i)
    h = (h ^ p[i]) * 0x100000001B3ull;
  return h;
}

/* Issues the ops of the record on the stream of the kernels; inside a
   capture the events are external, so that the host can wait for them. */
static CUresult issueOps(const struct SpanOp *ops, int numOps,
                         const unsigned char *bytes, int capturing) {
  void *args[1024];
  for (int i = 0; i != numOps; ++i) {
    const struct SpanOp *op = &ops[i];
    CUresult result = CUDA_SUCCESS;
    switch (op->kind) {
    case SPAN_KERNEL: {
      struct ParamLayout *layout = getLayout(op->function);
      if (layout->count > 1024)
        return CUDA_ERROR_INVALID_VALUE;
      const unsigned char *at = bytes + op->offset;
      for (uint32_t k = 0; k != layout->count; ++k) {
        args[k] = (void *)at;
        at += (layout->sizes[k] + 7u) & ~7u;
      }
      if (op->cluster[0]) {
        CUlaunchAttribute attrs[2];
        memset(attrs, 0, sizeof attrs);
        attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attrs[0].value.clusterDim.x = op->cluster[0];
        attrs[0].value.clusterDim.y = op->cluster[1];
        attrs[0].value.clusterDim.z = op->cluster[2];
        attrs[1].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        attrs[1].value.cooperative = op->cooperative;
        CUlaunchConfig config = {.gridDimX = op->grid[0],
                                 .gridDimY = op->grid[1],
                                 .gridDimZ = op->grid[2],
                                 .blockDimX = op->block[0],
                                 .blockDimY = op->block[1],
                                 .blockDimZ = op->block[2],
                                 .sharedMemBytes = op->shared,
                                 .hStream = sharedStream,
                                 .attrs = attrs,
                                 .numAttrs = 2};
        result = cuLaunchKernelEx(&config, op->function, args, NULL);
      } else if (op->cooperative) {
        result = cuLaunchCooperativeKernel(
            op->function, op->grid[0], op->grid[1], op->grid[2], op->block[0],
            op->block[1], op->block[2], op->shared, sharedStream, args);
      } else {
        result = cuLaunchKernel(op->function, op->grid[0], op->grid[1],
                                op->grid[2], op->block[0], op->block[1],
                                op->block[2], op->shared, sharedStream, args,
                                NULL);
      }
      break;
    }
    case SPAN_EVENT:
      result = capturing ? cuEventRecordWithFlags(op->event, sharedStream,
                                                  CU_EVENT_RECORD_EXTERNAL)
                         : cuEventRecord(op->event, sharedStream);
      break;
    case SPAN_FFT:
      if (!issueFFT(op->fft, op->plan, op->a, op->b))
        return CUDA_ERROR_UNKNOWN;
      break;
    case SPAN_MEMSET32:
      result = cuMemsetD32Async((CUdeviceptr)op->a, op->value,
                                (size_t)op->count, sharedStream);
      break;
    case SPAN_MEMSET16:
      result = cuMemsetD16Async((CUdeviceptr)op->a, (unsigned short)op->value,
                                (size_t)op->count, sharedStream);
      break;
    }
    if (result != CUDA_SUCCESS)
      return result;
  }
  return CUDA_SUCCESS;
}

static void releaseSpan(struct Span *s) {
  for (int i = 0; i != s->numOps; ++i)
    if (s->ops[i].kind == SPAN_FFT)
      retainFFTPlanHandle(s->ops[i].plan, -1);
  if (s->exec)
    cuGraphExecDestroy(s->exec);
  if (s->graph)
    cuGraphDestroy(s->graph);
  free(s->ops);
  free(s->bytes);
  free(s->nodes);
  memset(s, 0, sizeof *s);
}

static void rebuildSpanIndex(void);
static void releaseAllSpans(void) {
  /* MDRT_SPANS_UNLOAD_REPORT, for the tests: how many graphs the unload of
     a program released. */
  if (numSpans && getenv("MDRT_SPANS_UNLOAD_REPORT"))
    fprintf(stderr, "mdrt: the unload of a program released %d graphs of spans\n",
            numSpans);
  for (int i = 0; i != numSpans; ++i)
    releaseSpan(&spanCache[i]);
  numSpans = 0;
  rebuildSpanIndex();
  memset(spanSeen, 0, sizeof spanSeen);
  for (int i = 0; i != LAYOUTS; ++i)
    free(layouts[i].sizes);
  memset(layouts, 0, sizeof layouts);
}

/* The shape of the record: its ops without the argument bytes and grids. */
static uint64_t hashShape(void) {
  uint64_t h = 0x84222325CBF29CE4ull;
  for (int i = 0; i != numSpanOps; ++i) {
    struct SpanOp op = spanOps[i];
    memset(op.grid, 0, sizeof op.grid);
    h = hashBytes(h, &op, sizeof op);
  }
  return h;
}

/* Whether graph `s` has the shape of the record: its ops equal but for
   their grids (the hash of the shape may collide). */
static int sameShape(const struct Span *s) {
  for (int i = 0; i != numSpanOps; ++i) {
    struct SpanOp a = s->ops[i], b = spanOps[i];
    memset(a.grid, 0, sizeof a.grid);
    memset(b.grid, 0, sizeof b.grid);
    if (memcmp(&a, &b, sizeof a))
      return 0;
  }
  return 1;
}

/* The kernel nodes of a captured graph in the order of the record's
   kernels: the graph of one stream is a chain, whose kernels are those of
   the record and those of cuFFT, which have other functions. */
static int mapNodes(CUgraph graph, const struct SpanOp *ops, int numOps,
                    CUgraphNode *nodes) {
  memset(nodes, 0, numOps * sizeof *nodes);
  size_t count = 1;
  CUgraphNode node = NULL;
  if (cuGraphGetRootNodes(graph, &node, &count) != CUDA_SUCCESS || count != 1)
    return 0;
  int next = 0;
  while (node) {
    CUgraphNodeType type;
    if (cuGraphNodeGetType(node, &type) != CUDA_SUCCESS)
      return 0;
    if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params;
      if (cuGraphKernelNodeGetParams(node, &params) != CUDA_SUCCESS)
        return 0;
      int k = next;
      while (k != numOps && ops[k].kind != SPAN_KERNEL)
        ++k;
      if (k != numOps && params.func == ops[k].function) {
        nodes[k] = node;
        next = k + 1;
      }
    }
    size_t dependents = 0;
    if (cuGraphNodeGetDependentNodes(node, NULL, NULL, &dependents) != CUDA_SUCCESS ||
        dependents > 1)
      return 0;
    CUgraphNode following = NULL;
    if (dependents == 1 &&
        cuGraphNodeGetDependentNodes(node, &following, NULL, &dependents) !=
            CUDA_SUCCESS)
      return 0;
    node = dependents ? following : NULL;
  }
  for (int k = 0; k != numOps; ++k)
    if (ops[k].kind == SPAN_KERNEL && !nodes[k])
      return 0;
  return 1;
}

/* Makes the graph of the record by capturing its ops; 0 if the capture
   fails, and the ops are then issued as they are. */
static int captureSpan(struct Span *s) {
  CUgraph graph = NULL;
  CUgraphExec exec = NULL;
  if (cuStreamBeginCapture(sharedStream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) !=
      CUDA_SUCCESS)
    return 0;
  CUresult issued = issueOps(spanOps, numSpanOps, spanBytes, 1);
  CUresult ended = cuStreamEndCapture(sharedStream, &graph);
  int64_t before = beginAccount();
  CUresult instantiated = issued == CUDA_SUCCESS && ended == CUDA_SUCCESS && graph
                              ? cuGraphInstantiate(&exec, graph, 0)
                              : CUDA_ERROR_INVALID_VALUE;
  endAccount(ACCOUNT_SPANS, before, 0);
  if (instantiated != CUDA_SUCCESS) {
    if (graph)
      cuGraphDestroy(graph);
    return 0;
  }
  s->graph = graph;
  s->exec = exec;
  s->nodes = malloc(numSpanOps * sizeof *s->nodes);
  s->updatable = mapNodes(graph, spanOps, numSpanOps, s->nodes);
  return 1;
}

/* MDRT_SPANS_UPDATE_FAULT=<n>, for the tests: the n-th update of a kernel
   node is taken as refused by the driver, after it was made. */
static int updateFault(void) {
  static long fault = -2, made = 0;
  if (fault == -2) {
    const char *value = getenv("MDRT_SPANS_UPDATE_FAULT");
    fault = value && *value ? strtol(value, NULL, 10) : -1;
  }
  return fault > 0 && ++made == fault;
}

/* Gives graph `s` the arguments and grids of the record, whose shape is
   its own; 0 if the driver refuses an update. */
static int updateSpan(struct Span *s) {
  void *args[1024];
  for (int i = 0; i != numSpanOps; ++i) {
    const struct SpanOp *op = &spanOps[i], *was = &s->ops[i];
    if (op->kind != SPAN_KERNEL)
      continue;
    if (!memcmp(op->grid, was->grid, sizeof op->grid) &&
        !memcmp(spanBytes + op->offset, s->bytes + was->offset, op->bytes))
      continue;
    struct ParamLayout *layout = getLayout(op->function);
    if (layout->count > 1024)
      return 0;
    const unsigned char *at = spanBytes + op->offset;
    for (uint32_t k = 0; k != layout->count; ++k) {
      args[k] = (void *)at;
      at += (layout->sizes[k] + 7u) & ~7u;
    }
    CUDA_KERNEL_NODE_PARAMS params;
    memset(&params, 0, sizeof params);
    params.func = op->function;
    params.gridDimX = op->grid[0];
    params.gridDimY = op->grid[1];
    params.gridDimZ = op->grid[2];
    params.blockDimX = op->block[0];
    params.blockDimY = op->block[1];
    params.blockDimZ = op->block[2];
    params.sharedMemBytes = op->shared;
    params.kernelParams = args;
    if (cuGraphExecKernelNodeSetParams(s->exec, s->nodes[i], &params) !=
            CUDA_SUCCESS ||
        updateFault())
      return 0;
    ++spanCounts[6];
  }
  return 1;
}

static void keepRecord(struct Span *s, uint64_t hash) {
  s->hash = hash;
  free(s->ops);
  free(s->bytes);
  s->numOps = numSpanOps;
  s->ops = malloc(numSpanOps * sizeof *spanOps);
  memcpy(s->ops, spanOps, numSpanOps * sizeof *spanOps);
  s->numBytes = numSpanBytes;
  s->bytes = malloc(numSpanBytes ? numSpanBytes : 1);
  memcpy(s->bytes, spanBytes, numSpanBytes);
}

/* The caches by hash and by shape: open addressing, entries + 1; an entry
   whose span holds another hash now is passed over, and the tables are
   made again when they fill. */
static int spanIndex[SPAN_INDEX], shapeIndex[SPAN_INDEX];
static int spanIndexUsed = 0;
static void insertIndex(int *table, uint64_t hash, int i) {
  for (int probe = 0;; ++probe)
    if (!table[(hash + probe) % SPAN_INDEX]) {
      table[(hash + probe) % SPAN_INDEX] = i + 1;
      return;
    }
}
static void rebuildSpanIndex(void) {
  memset(spanIndex, 0, sizeof spanIndex);
  memset(shapeIndex, 0, sizeof shapeIndex);
  spanIndexUsed = 0;
  for (int i = 0; i != numSpans; ++i) {
    insertIndex(spanIndex, spanCache[i].hash, i);
    insertIndex(shapeIndex, spanCache[i].shape, i);
    ++spanIndexUsed;
  }
}
static void indexSpan(int i) {
  if (++spanIndexUsed > SPAN_INDEX / 2) {
    rebuildSpanIndex();
    return;
  }
  insertIndex(spanIndex, spanCache[i].hash, i);
  insertIndex(shapeIndex, spanCache[i].shape, i);
}

/* A place in the cache, the graph used least recently giving its own
   once the device has run what was issued from it. */
/* MDRT_SPANS_CACHE=<n>, for the tests: at most n graphs, fewer than
   SPAN_CACHE, so that the cache releases graphs. */
static int getSpanRoom(void) {
  static int room = 0;
  if (!room) {
    const char *value = getenv("MDRT_SPANS_CACHE");
    long n = value && *value ? strtol(value, NULL, 10) : SPAN_CACHE;
    room = n > 0 && n < SPAN_CACHE ? (int)n : SPAN_CACHE;
  }
  return room;
}

static struct Span *takeSpan(void) {
  if (numSpans >= getSpanRoom()) {
    int oldest = 0;
    for (int i = 1; i != numSpans; ++i)
      if (spanCache[i].lastUse < spanCache[oldest].lastUse)
        oldest = i;
    check(cuStreamSynchronize(sharedStream), "cuStreamSynchronize");
    releaseSpan(&spanCache[oldest]);
    spanCache[oldest] = spanCache[--numSpans];
    memset(&spanCache[numSpans], 0, sizeof spanCache[numSpans]);
    rebuildSpanIndex();
    ++spanCounts[5];
  }
  return &spanCache[numSpans++];
}

/* MDRT_SPANS_DEBUG=1: the hash and length of each span, and, for each
   sequence of kernels with the same functions and grids, the argument
   words that differ from its previous recording, with the kernel's name,
   on stderr: what keeps a span from repeating. */
static void debugSpan(void) {
  static int on = -1;
  if (on < 0)
    on = getenv("MDRT_SPANS_DEBUG") != NULL;
  if (!on)
    return;
  uint64_t full = hashBytes(hashBytes(0xCBF29CE484222325ull, spanOps,
                                      numSpanOps * sizeof *spanOps),
                            spanBytes, numSpanBytes);
  fprintf(stderr, "mdrt: span-hash %016llx %d\n", (unsigned long long)full,
          numSpanOps);
  enum { SHAPES = 512 };
  static struct {
    uint64_t hash;
    int numOps;
    struct SpanOp *ops;
    unsigned char *bytes;
    uint32_t numBytes;
  } last[SHAPES];
  uint64_t h = 0xCBF29CE484222325ull;
  for (int i = 0; i != numSpanOps; ++i) {
    h = hashBytes(h, &spanOps[i].kind, sizeof spanOps[i].kind);
    h = hashBytes(h, &spanOps[i].function, sizeof spanOps[i].function);
    h = hashBytes(h, spanOps[i].grid, sizeof spanOps[i].grid);
  }
  int slot = (int)(h % SHAPES);
  if (last[slot].hash == h && last[slot].numOps == numSpanOps &&
      last[slot].numBytes == numSpanBytes)
    for (int i = 0; i != numSpanOps; ++i) {
      const struct SpanOp *a = &last[slot].ops[i], *b = &spanOps[i];
      if (b->kind != SPAN_KERNEL) {
        if (memcmp(a, b, sizeof *a))
          fprintf(stderr, "mdrt: span %016llx op %d (kind %d) differs\n",
                  (unsigned long long)h, i, b->kind);
        continue;
      }
      for (uint32_t w = 0; w + 8 <= b->bytes; w += 8) {
        uint64_t x, y;
        memcpy(&x, last[slot].bytes + a->offset + w, 8);
        memcpy(&y, spanBytes + b->offset + w, 8);
        if (x != y)
          fprintf(stderr,
                  "mdrt: span %016llx op %d %s word %u: %016llx -> %016llx\n",
                  (unsigned long long)h, i, getKernelName(b->function), w / 8,
                  (unsigned long long)x, (unsigned long long)y);
      }
    }
  free(last[slot].ops);
  free(last[slot].bytes);
  last[slot].hash = h;
  last[slot].numOps = numSpanOps;
  last[slot].numBytes = numSpanBytes;
  last[slot].ops = malloc(numSpanOps * sizeof *spanOps);
  memcpy(last[slot].ops, spanOps, numSpanOps * sizeof *spanOps);
  last[slot].bytes = malloc(numSpanBytes ? numSpanBytes : 1);
  memcpy(last[slot].bytes, spanBytes, numSpanBytes);
}

static void flushSpan(void) {
  if (!numSpanOps)
    return;
  debugSpan();
  int kernels = 0;
  for (int i = 0; i != numSpanOps; ++i)
    kernels += spanOps[i].kind == SPAN_KERNEL || spanOps[i].kind == SPAN_FFT;
  struct Span *found = NULL;
  if (kernels >= 2) {
    uint64_t hash = hashBytes(
        hashBytes(0xCBF29CE484222325ull, spanOps,
                  numSpanOps * sizeof *spanOps),
        spanBytes, numSpanBytes);
    uint64_t shape = hashShape();
    /* The same span: its graph as it is. */
    for (int probe = 0; probe != SPAN_INDEX && !found; ++probe) {
      int i = spanIndex[(hash + probe) % SPAN_INDEX] - 1;
      if (i < 0)
        break;
      struct Span *s = &spanCache[i];
      if (!s->dead && s->hash == hash && s->numOps == numSpanOps &&
          s->numBytes == numSpanBytes &&
          !memcmp(s->ops, spanOps, numSpanOps * sizeof *spanOps) &&
          !memcmp(s->bytes, spanBytes, numSpanBytes))
        found = s;
    }
    if (!found) {
      /* The graphs of its shape: a new one while there are fewer than
         SPAN_VARIANTS, then the one used least recently, updated. */
      struct Span *variants[SPAN_VARIANTS];
      int numVariants = 0;
      for (int probe = 0; probe != SPAN_INDEX; ++probe) {
        int i = shapeIndex[(shape + probe) % SPAN_INDEX] - 1;
        if (i < 0)
          break;
        struct Span *s = &spanCache[i];
        int seen = 0;
        for (int k = 0; k != numVariants; ++k)
          seen |= variants[k] == s;
        if (!seen && !s->dead && s->shape == shape && s->numOps == numSpanOps &&
            s->numBytes == numSpanBytes && numVariants != SPAN_VARIANTS &&
            sameShape(s))
          variants[numVariants++] = s;
      }
      uint64_t *seen = &spanSeen[shape % SPAN_SEEN];
      if (numVariants == SPAN_VARIANTS || (numVariants && *seen != shape)) {
        struct Span *oldest = NULL;
        for (int k = 0; k != numVariants; ++k)
          if (variants[k]->updatable &&
              (!oldest || variants[k]->lastUse < oldest->lastUse))
            oldest = variants[k];
        int updated = oldest && updateSpan(oldest);
        if (oldest && !updated) {
          /* Some of its nodes may hold the new arguments: it matches no
             record again, and its place goes when the cache fills. */
          oldest->hash = oldest->shape = 0;
          oldest->dead = 1;
          oldest->updatable = 0;
          ++spanCounts[7];
        }
        if (updated) {
          keepRecord(oldest, hash);
          insertIndex(spanIndex, hash, (int)(oldest - spanCache));
          if (++spanIndexUsed > SPAN_INDEX / 2)
            rebuildSpanIndex();
          found = oldest;
        }
        *seen = shape;
      } else if (*seen == shape) {
        /* The shape was recorded before: make a graph. */
        struct Span *s = takeSpan();
        if (captureSpan(s)) {
          s->shape = shape;
          keepRecord(s, hash);
          for (int i = 0; i != numSpanOps; ++i)
            if (spanOps[i].kind == SPAN_FFT)
              retainFFTPlanHandle(spanOps[i].plan, +1);
          indexSpan((int)(s - spanCache));
          found = s;
          ++spanCounts[2];
        } else {
          --numSpans;
          ++spanCounts[3];
        }
        *seen = 0;
      } else {
        *seen = shape;
      }
    }
  }
  if (found) {
    found->lastUse = ++spanClock;
    check(cuGraphLaunch(found->exec, sharedStream), "cuGraphLaunch");
    ++spanCounts[0];
  } else {
    check(issueOps(spanOps, numSpanOps, spanBytes, 0), "a recorded launch");
    ++spanCounts[1];
  }
  spanCounts[4] += kernels;
  numSpanOps = 0;
  numSpanBytes = 0;
}

/* Issues what was recorded: every boundary of the host calls it first. */
void mdrtSpansFlush(void) {
  if (numSpanOps)
    flushSpan();
}

/* Turns spans on or off for the thread of the steps (the driver, from
   `[execution] step_graph`). Off issues what was recorded. */
void mdrtSpansEnable(int32_t on) {
  mdrtSpansFlush();
  spansOn = spansOverride(on != 0);
}

/* Holds the record off while other code captures the stream itself (the
   TOWERS graph, D439). */
void mdrtSpansSuspend(int32_t suspend) {
  mdrtSpansFlush();
  spansSuspended += suspend ? 1 : -1;
}

/* The counts for the log: graph launches, spans issued as recorded,
   graphs made, captures that failed, kernels and transforms recorded,
   graphs released, kernel nodes updated, updates refused, launches not
   recorded for their layout. */
void mdrtSpansCounts(int64_t counts[9]) {
  for (int i = 0; i != 9; ++i)
    counts[i] = spanCounts[i];
}

/* The descriptor of a memref of one dimension, as the C interface of MLIR
   passes it. */
struct Buffer1 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

/* The host's copy of the results of the global sums of a function and of
   its coupling step, page-locked so that the copy from the device is a
   transfer of the device's engine (D452; the maintainer's
   addition to #1026). Each place of the module that takes one has a number
   of its own and keeps its memory for the process; a larger request
   replaces it. */
enum { HOST_RESULT_PLACES = 1024 };
static void *hostResults[HOST_RESULT_PLACES];
static int64_t hostResultBytes[HOST_RESULT_PLACES];

void _mlir_ciface_mdrtHostResults(struct Buffer1 *result, int64_t place,
                                  int64_t count) {
  mgpuStreamCreate();
  if (place < 0 || place >= HOST_RESULT_PLACES) {
    fprintf(stderr, "mdrt: the module takes more than %d buffers of results\n",
            HOST_RESULT_PLACES);
    exit(1);
  }
  int64_t bytes = count * (int64_t)sizeof(double);
  if (hostResultBytes[place] < bytes) {
    if (hostResults[place]) {
      finish();
      check(cuMemFreeHost(hostResults[place]), "cuMemFreeHost");
    }
    void *host = NULL;
    check(cuMemHostAlloc(&host, (size_t)bytes, 0), "cuMemHostAlloc");
    memset(host, 0, (size_t)bytes);
    hostResults[place] = host;
    hostResultBytes[place] = bytes;
  }
  result->allocated = hostResults[place];
  result->aligned = hostResults[place];
  result->offset = 0;
  result->size = count;
  result->stride = 1;
}

/* A memref of rank 2 as the C interface of MLIR passes it. */
struct Buffer2 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

/* Reads of flags of the device that do not wait for the work issued after
   them (D113): mdrtFlagStart copies the flag into memory of the host that
   the device writes while the host goes on, and marks the copy with an
   event; mdrtFlagFinish waits for that event only. The host can then issue
   the work that does not depend on the flag, such as the reciprocal sum,
   before it waits, and the device does not stand idle while the host
   decides. A run has few flags in flight; the slots are reused in turn. */
enum { FLAG_SLOTS = 32 };
static int32_t *flagSlots = NULL;
static CUevent flagEvents[FLAG_SLOTS];
static volatile int32_t *flagSources[FLAG_SLOTS];
/* The event that each slot waits for: its own, or that of the place of a
   mapped flag while spans record (mdrtSpansEnable). */
static CUevent flagEventOf[FLAG_SLOTS];
static int64_t nextFlagSlot = 0;

/* The flags themselves are in memory of the host that is mapped for the
   device (D118): a kernel stores 1 there across the bus, so a read needs no
   copy on the stream, only an event after the kernels that set it. A copy
   between two kernels held the device for some microseconds each, and the
   two flags of a dual list cost 13 us a step on ubiquitin. The host clears
   a flag when it reads it: no kernel that sets it is issued before the host
   has decided on it. Each place of the module that takes a flag has a
   number of its own and the same memory at every call of its function, so
   that a function called many times, as a step of a minimization is, takes
   no other function's flag. */
enum { MAPPED_FLAGS = 1024 };
static volatile int32_t *mappedFlags = NULL;
static CUevent placeEvents[MAPPED_FLAGS];
static CUdeviceptr mappedDevice = 0;

void _mlir_ciface_mdrtFlagMemory(struct Buffer1 *result, int64_t place) {
  mgpuStreamCreate();
  if (!mappedFlags) {
    void *host = NULL;
    check(cuMemHostAlloc(&host, MAPPED_FLAGS * sizeof(int32_t),
                         CU_MEMHOSTALLOC_DEVICEMAP),
          "cuMemHostAlloc");
    memset(host, 0, MAPPED_FLAGS * sizeof(int32_t));
    check(cuMemHostGetDevicePointer(&mappedDevice, host, 0),
          "cuMemHostGetDevicePointer");
    mappedFlags = (volatile int32_t *)host;
  }
  if (place < 0 || place >= MAPPED_FLAGS) {
    fprintf(stderr, "mdrt: the module takes more than %d flags\n",
            MAPPED_FLAGS);
    exit(1);
  }
  int64_t slot = place;
  mappedFlags[slot] = 0;
  void *device = (void *)(uintptr_t)(mappedDevice + slot * sizeof(int32_t));
  result->allocated = device;
  result->aligned = device;
  result->offset = 0;
  result->size = 1;
  result->stride = 1;
}

int64_t mdrtFlagStart(void *device) {
  mgpuStreamCreate();
  if (!flagSlots) {
    check(cuMemAllocHost((void **)&flagSlots,
                         FLAG_SLOTS * sizeof(*flagSlots)),
          "cuMemAllocHost");
    for (int i = 0; i != FLAG_SLOTS; ++i)
      check(cuEventCreate(&flagEvents[i], CU_EVENT_DISABLE_TIMING),
            "cuEventCreate");
  }
  /* The kernels that set flags run on the first stream. */
  int64_t slot = nextFlagSlot++ % FLAG_SLOTS;
  CUdeviceptr address = (CUdeviceptr)device;
  if (mappedFlags && address >= mappedDevice &&
      address < mappedDevice + MAPPED_FLAGS * sizeof(int32_t)) {
    int64_t place = (int64_t)((address - mappedDevice) / sizeof(int32_t));
    flagSources[slot] = mappedFlags + place;
    if (isRecording()) {
      /* The event of a mapped flag is its place's own, so that a
         recorded sequence repeats from step to step; a place has one
         read in flight at a time. The record is issued here, so that the
         device runs what precedes the flag while the host records the
         work issued before the wait. */
      if (!placeEvents[place])
        check(cuEventCreate(&placeEvents[place], CU_EVENT_DISABLE_TIMING),
              "cuEventCreate");
      flagEventOf[slot] = placeEvents[place];
      struct SpanOp *op = appendOp(SPAN_EVENT);
      op->event = placeEvents[place];
      isPending = 1;
      if (spanSplitAtFlags())
        mdrtSpansFlush();
      return slot;
    }
  } else {
    mdrtSpansFlush();
    flagSources[slot] = NULL;
    if (copyAudit) copyAudit->metadataDownloads += sizeof(*flagSlots);
    mdrtCountHostRead(sizeof(*flagSlots), 0, 0);
    check(cuMemcpyDtoHAsync(&flagSlots[slot], address, sizeof(*flagSlots),
                            sharedStream),
          "cuMemcpyDtoHAsync");
  }
  flagEventOf[slot] = flagEvents[slot];
  check(cuEventRecord(flagEvents[slot], sharedStream), "cuEventRecord");
  isPending = 1;
  return slot;
}

int32_t mdrtFlagFinish(int64_t slot) {
  mdrtSpansFlush();
  mdrtCountHostRead(0, 0, 1);
  double start = begin();
  check(cuEventSynchronize(flagEventOf[slot]), "cuEventSynchronize");
  end(WAIT, start);
  volatile int32_t *source = flagSources[slot];
  if (!source)
    return flagSlots[slot];
  int32_t value = *source;
  *source = 0;
  return value;
}

/* Reads and clears a flag that the kernel of an other flag set, after
   mdrtFlagFinish of that flag has waited for the event after the kernel
   (D278): the loops of a run, one kernel, have a flag
   each and one event. A flag outside the mapped words is copied as
   mdrtFlagStart does and waited for. */
int32_t mdrtFlagTake(void *device) {
  CUdeviceptr address = (CUdeviceptr)device;
  if (mappedFlags && address >= mappedDevice &&
      address < mappedDevice + MAPPED_FLAGS * sizeof(int32_t)) {
    volatile int32_t *source =
        mappedFlags + (address - mappedDevice) / sizeof(int32_t);
    int32_t value = *source;
    *source = 0;
    return value;
  }
  return mdrtFlagFinish(mdrtFlagStart(device));
}

/* Waits until the device has run everything that was issued. */
static void finish(void) {
  mdrtSpansFlush();
  if (!isPending || (!sharedStream && !borrowedStreamDepth))
    return;
  mdrtCountHostRead(0, 1, 0);
  double start = begin();
  for (int lane = 0; lane != SIDE_LANES; ++lane)
    if (sideStreams[lane])
      check(cuStreamSynchronize(sideStreams[lane]), "cuStreamSynchronize");
  check(cuStreamSynchronize(sharedStream), "cuStreamSynchronize");
  end(WAIT, start);
  isPending = 0;
}

/* Adds the time since the last launch to the time of its kernel. */
static void endKernel(void) {
  if (lastKernel < 0)
    return;
  kernels[lastKernel].count += 1;
  kernels[lastKernel].seconds += now() - lastLaunch;
  lastKernel = -1;
}

void mgpuStreamSynchronize(CUstream stream) {
  (void)stream;
  if (waitsAlways < 0)
    waitsAlways = getenv("MDRT_WAIT") != NULL;
  if (waitsAlways) {
    finish();
    endKernel();
  }
}

void mgpuStreamWaitEvent(CUstream stream, CUevent event) {
  mdrtSpansFlush();
  check(cuStreamWaitEvent(stream, event, 0), "cuStreamWaitEvent");
}

CUevent mgpuEventCreate(void) {
  enter();
  CUevent event = NULL;
  check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
  return event;
}

void mgpuEventDestroy(CUevent event) {
  check(cuEventDestroy(event), "cuEventDestroy");
}

void mgpuEventSynchronize(CUevent event) {
  mdrtSpansFlush();
  mdrtCountHostRead(0, 1, 0);
  check(cuEventSynchronize(event), "cuEventSynchronize");
}

void mgpuEventRecord(CUevent event, CUstream stream) {
  mdrtSpansFlush();
  check(cuEventRecord(event, stream), "cuEventRecord");
}

/*===----------------------------------------------------------------------===
  Memory
  ===----------------------------------------------------------------------===*/

/* Memory of the device is kept for reuse when it is freed. All work goes
   to one stream, in order, so a block that is freed may be handed to work
   issued after the free without waiting: that work runs after the work that
   used the block. A change of the stream (mdrtUseStream) makes the new
   stream wait for an event recorded on the old one, so this holds across
   it; the second stream of mdrtSideBegin allocates and frees nothing (D87).
   Freeing memory with the driver would wait for the device, and a build of
   a neighbor structure frees a dozen blocks.

   The pool is a best-fit allocator with splitting and deferred
   coalescing (Wilson, Johnstone, Neely and Boles, "Dynamic storage
   allocation: a survey and critical review", IWMM 1995, LNCS 986), over
   the allocations that it takes from the driver. A request that no idle
   piece can hold takes an allocation of its own size from the driver, an
   arena, which is then cut into pieces. A request takes first an idle
   piece that last held a block of exactly its size, unmerged and uncut
   since, so that the requests that recur at every step keep their pieces
   (and the addresses that the pool gave them before it split). Otherwise
   it takes the smallest idle piece that holds it, cut at its size rounded
   up to POOL_ALIGNMENT bytes when the rest is at least that much, or whole
   otherwise. A freed piece stays as it is. The idle pieces are merged with
   their idle neighbors in their own arena, never across arenas, even ones
   that the driver placed next to each other, only when a request finds no
   idle piece that holds it, at a trim, and when an activation closes.
   Every byte of a merged piece was freed by work issued before the request
   that takes it, so the order of the stream covers each part. The sizes
   that compiled code asks for at every step do not follow the builds: a
   buffer indexed by the places of the groups takes the room of the places,
   not the number that a build occupied (D312).

   Only a whole arena goes back to the driver: an idle piece of an arena
   that holds a block in use stays in the pool until the arena is idle as a
   whole. A buffer of the runtime that a growth replaces goes back to the
   driver if it is an arena of its own, and returns to the pool if it is a
   piece of a larger one (D312).

   The registry of the blocks, live and pooled, is what returns them: an
   activation that closes takes back the blocks that it owns, and a block
   that the registry does not record would stay allocated for the rest of
   the process. So it grows with the blocks, however many there are; it
   held 4096 before #622, and a block past that was handed out untracked
   and never reclaimed. If it cannot grow, the block is given back to the
   driver and the run stops. The idle pieces of the pool are bounded in
   bytes (MDIR_DEVICE_POOL_BYTES, by default half of the device): an arena
   found idle as a whole (a freed block that is an arena of its own, or
   pieces merged) when the idle pieces would exceed the budget goes back to
   the driver, so the idle arenas never exceed it; a free that takes the
   idle pieces past the budget merges them first. A
   request that the driver refuses for want of memory releases the idle
   arenas of the pool and the plans of cuFFT (trimIdle) and is made once
   more; and an embedding program may release them itself
   (mdrtDeviceTrim). The copies of the blocks at the peaks, for
   MDIR_DEVICE_MEMORY=2, keep at most MAX_BLOCKS of them. */
#define MAX_BLOCKS 4096
/* An activation of an entry that an embedding program keeps across its
   calls (a Python simulation, D215): what the activation
   allocates is its own, and returns to the pool when the program closes
   it, so the memory of the device stays that of one activation however
   many parts it runs (#110). Several activations may be open at once, one
   for each live simulation, and one of them is current while its code
   runs. `mdir run` opens none. */
struct Activation {
  /* Memory shared with the host that the activation allocated. */
  CUdeviceptr *managed;
  size_t numManaged, roomManaged;
  /* The structures of neighbors (groups and matrices) that it made: their
     host records are freed when it closes, and their memory of the device
     returns to the pool with its other blocks. */
  void **handles;
  size_t numHandles, roomHandles;
};
static struct Activation *currentActivation = NULL;

/* A block in use, or an idle piece of the pool. */
struct Block {
  CUdeviceptr pointer;
  /* The bytes asked for; of an idle piece, its extent. */
  uint64_t size;
  /* The activation that allocated the block, or null. */
  struct Activation *owner;
  /* What the block was last allocated for: a buffer of the runtime's
     structures, or null for compiled code (MDIR_DEVICE_MEMORY=2). */
  const char *tag;
  /* The bytes of the piece that it occupies from `pointer`: at least
     `size`, and a multiple of POOL_ALIGNMENT unless the piece ends its
     arena. */
  uint64_t extent;
  /* The arena: the allocation of the driver that holds the piece. */
  CUdeviceptr base;
  uint64_t baseSize;
  /* Of an idle piece: the bytes asked for by the block that it last held,
     if it has not been merged or cut since; 0 otherwise. */
  uint64_t asked;
};
/* The alignment of the pieces of an arena: the driver aligns an
   allocation to at least 256 bytes, and the code may rely on that. */
#define POOL_ALIGNMENT 256u
static struct Block *liveBlocks = NULL;
static int numLive = 0, roomLive = 0;
static struct Block *freeBlocks = NULL;
static int numFree = 0, roomFree = 0;
/* The arenas of the pool's allocator that the driver holds, live or idle,
   and their bytes: what a trim of everything gives back
   (mdrtDeviceMemoryCounts). */
static int64_t numDriverBlocks = 0;
static uint64_t reservedBytes = 0;
/* Makes room for one more block in `list`, twice as much as it had; 0 if
   the host has no memory for it. */
static int reserveBlock(struct Block **list, int count, int *room) {
  if (count < *room)
    return 1;
  int grown = *room ? 2 * *room : 1024;
  struct Block *larger = realloc(*list, (size_t)grown * sizeof(struct Block));
  if (!larger)
    return 0;
  *list = larger;
  *room = grown;
  return 1;
}
/* Records a block in use, or stops the run if the registry cannot grow:
   the block is first given back, so that nothing stays untracked. */
static void recordLiveBlock(struct Block block) {
  if (!reserveBlock(&liveBlocks, numLive, &roomLive)) {
    if (block.pointer == block.base && block.extent == block.baseSize)
      cuMemFree(block.pointer);
    fprintf(stderr, "mdrt: out of memory of the host for the registry of "
                    "the blocks of the device; the block is given back\n");
    abort();
  }
  liveBlocks[numLive++] = block;
}
/* The budget of the idle blocks of the pool, in bytes: MDIR_DEVICE_POOL_BYTES,
   or half of the memory of the device. */
static uint64_t getPoolBudget(void) {
  static int read = 0;
  static uint64_t budget = 0;
  if (!read) {
    read = 1;
    const char *value = getenv("MDIR_DEVICE_POOL_BYTES");
    size_t free = 0, total = 0;
    if (value && *value)
      budget = strtoull(value, NULL, 10);
    else if (context && cuCtxSetCurrent(context) == CUDA_SUCCESS &&
             cuMemGetInfo(&free, &total) == CUDA_SUCCESS)
      budget = total / 2;
    else
      budget = UINT64_MAX;
  }
  return budget;
}
/* Returns the idle arenas of the pool and the plans of cuFFT to the driver
   once the device has run its work; the bytes returned. */
static uint64_t trimIdle(void);
/* cuMemAlloc, made once more after a trim if the driver has no memory.
   MDRT_FAIL_ALLOCATION=N, for the tests of the retry, makes the N-th
   request of this process report that the driver has no memory, without
   asking it. */
static CUresult allocateDevice(CUdeviceptr *pointer, size_t size) {
  static int64_t requests = 0, failing = -1;
  if (failing < 0) {
    const char *value = getenv("MDRT_FAIL_ALLOCATION");
    failing = value && *value ? strtoll(value, NULL, 10) : 0;
  }
  CUresult result = ++requests == failing ? CUDA_ERROR_OUT_OF_MEMORY
                                          : cuMemAlloc(pointer, size);
  if (result == CUDA_ERROR_OUT_OF_MEMORY && trimIdle())
    result = cuMemAlloc(pointer, size);
  return result;
}
/* Puts a freed block in the pool, merged with the idle pieces next to it
   in its arena. An arena that is then idle as a whole goes back to the
   driver if the budget has no room for it; otherwise the piece stays idle.
   A buffer that a growth replaced (`replaced`, D312) goes back at once if
   it is an arena of its own. The caller has waited for the work that may
   use it (as it has if `replaced`), or the work after it runs after that
   work. */
static void poolBlock(struct Block block, int gone, int replaced);
/* What the runtime allocates a block for, while it does (null for
   compiled code). */
static const char *allocationTag = NULL;

/* The bytes asked for by the blocks in use and the bytes of the idle
   pieces of the pool, and the peaks since the program started (D294) of
   the bytes in use and of the bytes of the arenas, those in use, idle, or
   rounded up within a piece in use: with MDIR_DEVICE_MEMORY set, they are
   printed when the process ends. */
static uint64_t liveBytes = 0, poolBytes = 0, peakLive = 0, peakReserved = 0;
/* With MDIR_DEVICE_MEMORY=2: the blocks at the peak in use and at the peak
   with the pool (those in the pool after those in use), the driver's use
   at the latter, and the counts of the allocations. */
static struct Block peakLiveBlocks[MAX_BLOCKS];
static int numPeakLive = 0;
static struct Block peakReservedBlocks[2 * MAX_BLOCKS];
static int numPeakReservedLive = 0, numPeakReservedFree = 0;
static int64_t usedAtPeakReserved = -1;
static uint64_t paddingAtPeakReserved = 0;
static int64_t numAllocations = 0, numFromPool = 0, numToDriver = 0;
static int64_t numSplits = 0;
static void noteBytes(void) {
  int detailed = isMemoryDetailed();
  if (liveBytes > peakLive) {
    peakLive = liveBytes;
    if (detailed) {
      int live = numLive < MAX_BLOCKS ? numLive : MAX_BLOCKS;
      memcpy(peakLiveBlocks, liveBlocks, live * sizeof(struct Block));
      numPeakLive = live;
    }
  }
  if (reservedBytes > peakReserved) {
    peakReserved = reservedBytes;
    paddingAtPeakReserved = reservedBytes - liveBytes - poolBytes;
    if (detailed) {
      int live = numLive < MAX_BLOCKS ? numLive : MAX_BLOCKS;
      int idle = numFree < MAX_BLOCKS ? numFree : MAX_BLOCKS;
      memcpy(peakReservedBlocks, liveBlocks, live * sizeof(struct Block));
      memcpy(peakReservedBlocks + live, freeBlocks,
             idle * sizeof(struct Block));
      numPeakReservedLive = live;
      numPeakReservedFree = idle;
      usedAtPeakReserved = getDeviceUsed();
    }
  }
}

/* The blocks of `blocks`, by what they were allocated for and their size,
   the largest totals first: one line for each, and one for each tag for
   the blocks smaller than a MiB. */
struct BlockKind {
  const char *tag;
  uint64_t size;
  int64_t count;
};
static int compareKinds(const void *a, const void *b) {
  const struct BlockKind *x = a, *y = b;
  /* A count of 0 is the blocks under a MiB, whose bytes are the size. */
  uint64_t bx = x->size * (uint64_t)(x->count ? x->count : 1);
  uint64_t by = y->size * (uint64_t)(y->count ? y->count : 1);
  return bx < by ? 1 : bx > by ? -1 : 0;
}
static void printBlocks(const char *state, const struct Block *blocks,
                        int count) {
  static struct BlockKind kinds[2 * MAX_BLOCKS];
  int numKinds = 0;
  for (int i = 0; i != count; ++i) {
    const char *tag = blocks[i].tag ? blocks[i].tag : "compiled code";
    uint64_t size = blocks[i].size < 1048576 ? 0 : blocks[i].size;
    int k = 0;
    while (k != numKinds &&
           !(kinds[k].size == size && strcmp(kinds[k].tag, tag) == 0))
      ++k;
    if (k == numKinds)
      kinds[numKinds++] = (struct BlockKind){tag, size, 0};
    /* The blocks under a MiB are counted together, by their bytes. */
    kinds[k].count += size ? 1 : (int64_t)blocks[i].size;
  }
  for (int k = 0; k != numKinds; ++k)
    if (!kinds[k].size) {
      kinds[k].size = (uint64_t)kinds[k].count;
      kinds[k].count = 0;
    }
  qsort(kinds, numKinds, sizeof(*kinds), compareKinds);
  for (int k = 0; k != numKinds; ++k) {
    int small = kinds[k].count == 0;
    int64_t count = small ? 1 : kinds[k].count;
    fprintf(stderr, "mdrt:   %-7s %10.1f MiB  %5lld x %12llu bytes  %s%s\n",
            state, kinds[k].size * (double)count / 1048576.0,
            (long long)count, (unsigned long long)kinds[k].size,
            kinds[k].tag, small ? " (the blocks under 1 MiB)" : "");
  }
}

static void printDeviceMemory(void) {
  fprintf(stderr,
          "mdrt: device memory (MDIR_DEVICE_MEMORY): peak %.1f MiB allocated, "
          "%.1f MiB with the pool\n",
          peakLive / 1048576.0, peakReserved / 1048576.0);
  if (!isMemoryDetailed())
    return;
  const double mib = 1048576.0;
  int64_t used = context && cuCtxSetCurrent(context) == CUDA_SUCCESS
                     ? getDeviceUsed()
                     : -1;
  fprintf(stderr,
          "mdrt: device memory: %lld allocations of blocks, %lld from the "
          "pool, %lld returned to the driver, %lld split from a larger "
          "piece\n",
          (long long)numAllocations, (long long)numFromPool,
          (long long)numToDriver, (long long)numSplits);
  fprintf(stderr,
          "mdrt: device memory: the driver's use of the device: %.1f MiB "
          "once the context was made, %.1f MiB at the peak with the pool, "
          "%.1f MiB at most at the events, %.1f MiB at the end\n",
          accountGrowth[ACCOUNT_CONTEXT] / mib, usedAtPeakReserved / mib,
          usedMost / mib, used / mib);
  fprintf(stderr, "mdrt: device memory: by account, the growth of the "
                  "driver's use, the bytes asked for, and the events:\n");
  for (int i = 1; i != NUM_ACCOUNTS; ++i)
    if (accountEvents[i])
      fprintf(stderr, "mdrt:   %-36s %10.1f MiB %10.1f MiB %6lld\n",
              accountNames[i], accountGrowth[i] / mib, accountAsked[i] / mib,
              (long long)accountEvents[i]);
  fprintf(stderr,
          "mdrt: device memory at the peak in use (%.1f MiB), by tag and "
          "size:\n",
          peakLive / mib);
  printBlocks("in use", peakLiveBlocks, numPeakLive);
  fprintf(stderr,
          "mdrt: device memory at the peak with the pool (%.1f MiB), by "
          "tag and size:\n",
          peakReserved / mib);
  printBlocks("in use", peakReservedBlocks, numPeakReservedLive);
  printBlocks("pool", peakReservedBlocks + numPeakReservedLive,
              numPeakReservedFree);
  if (paddingAtPeakReserved)
    fprintf(stderr,
            "mdrt:   %-7s %10.1f MiB  %12llu bytes  the pieces in use "
            "beyond the bytes asked for\n",
            "padding", paddingAtPeakReserved / mib,
            (unsigned long long)paddingAtPeakReserved);
}
static void watchDeviceMemory(void) {
  static int registered = 0;
  if (registered)
    return;
  registered = 1;
  const char *value = getenv("MDIR_DEVICE_MEMORY");
  if (value && *value && strcmp(value, "0") != 0)
    atexit(printDeviceMemory);
}

/* Makes the room of a list that grows twice as large. */
static void *growList(void *list, size_t *room, size_t size) {
  *room = *room ? 2 * *room : 8;
  void *grown = realloc(list, *room * size);
  if (!grown) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  return grown;
}

/* Stops the run if a block of the pool is allocated or freed on a stream
   other than the runtime's first one: while the second stream of
   mdrtSideBegin is current, or with a stream named that is not the first.
   The pool hands a freed piece to the next request without waiting, which
   is sound only because both are issued on the first stream, in order
   (D87, premise P4; D416). The lowering passes no stream. */
static void mergeIdle(void);
static void checkPoolStream(CUstream stream, const char *what) {
  if (!onSide && (!stream || stream == sharedStream))
    return;
  fprintf(stderr,
          "mdrt: %s of a block of device memory on a stream other than the "
          "runtime's first stream%s; the pool orders the reuse of its blocks "
          "by the first stream alone\n",
          what, onSide ? " (between mdrtSideBegin and mdrtSideEnd)" : "");
  abort();
}

void *mgpuMemAlloc(uint64_t size, CUstream stream, bool isHostShared) {
  enter();
  CUdeviceptr pointer = 0;
  if (size == 0)
    return NULL;
  double start = begin();
  if (isHostShared) {
    int64_t before = beginAccount();
    check(cuMemAllocManaged(&pointer, size, CU_MEM_ATTACH_GLOBAL),
          "cuMemAllocManaged");
    endAccount(ACCOUNT_MANAGED, before, (int64_t)size);
    struct Activation *a = currentActivation;
    if (a) {
      if (a->numManaged == a->roomManaged)
        a->managed = growList(a->managed, &a->roomManaged, sizeof(*a->managed));
      a->managed[a->numManaged++] = pointer;
    }
    end(ALLOCATE, start);
    return (void *)pointer;
  }
  checkPoolStream(stream, "an allocation");
  watchDeviceMemory();
  struct Block block = {.size = size,
                        .owner = currentActivation,
                        .tag = allocationTag};
  /* First an idle piece that last held a block of exactly this size and
     has not been merged or cut since, the last such in the list: a request
     that recurs at every step takes the piece it freed, as the pool took
     before it split. Otherwise the best fit, the smallest idle piece that
     holds the request (among pieces of one extent, the last in the list);
     if none does, the idle pieces are merged with their idle neighbors and
     the best fit is sought again. */
  int best = -1;
  for (int i = numFree; i-- != 0;)
    if (freeBlocks[i].asked == size) {
      best = i;
      break;
    }
  for (int pass = 0; best < 0 && pass != 2 && numFree; ++pass) {
    if (pass)
      mergeIdle();
    for (int i = numFree; i-- != 0;) {
      uint64_t extent = freeBlocks[i].extent;
      if (extent < size || (best >= 0 && extent >= freeBlocks[best].extent))
        continue;
      best = i;
      if (extent == size)
        break;
    }
  }
  ++numAllocations;
  if (best >= 0) {
    struct Block piece = freeBlocks[best];
    freeBlocks[best] = freeBlocks[--numFree];
    poolBytes -= piece.extent;
    uint64_t used =
        (size + POOL_ALIGNMENT - 1) & ~(uint64_t)(POOL_ALIGNMENT - 1);
    if (piece.asked != size && piece.extent >= used + POOL_ALIGNMENT) {
      /* The rest stays idle; the list has the room of the piece taken. */
      struct Block rest = piece;
      rest.pointer += used;
      rest.extent -= used;
      rest.size = rest.extent;
      rest.asked = 0;
      freeBlocks[numFree++] = rest;
      poolBytes += rest.extent;
      piece.extent = used;
      ++numSplits;
    }
    block.pointer = piece.pointer;
    block.extent = piece.extent;
    block.base = piece.base;
    block.baseSize = piece.baseSize;
    ++numFromPool;
  } else {
    int64_t before = beginAccount();
    check(allocateDevice(&pointer, size), "cuMemAlloc");
    endAccount(ACCOUNT_BLOCKS, before, (int64_t)size);
    ++numDriverBlocks;
    reservedBytes += size;
    block.pointer = block.base = pointer;
    block.extent = block.baseSize = size;
  }
  pointer = block.pointer;
  liveBytes += size;
  recordLiveBlock(block);
  noteBytes();
  end(ALLOCATE, start);
  return (void *)pointer;
}

void mgpuMemFree(void *pointer, CUstream stream) {
  if (!pointer)
    return;
  checkPoolStream(stream, "a free");
  double start = begin();
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = numLive; i-- != 0;) {
    if (liveBlocks[i].pointer != address)
      continue;
    struct Block block = liveBlocks[i];
    liveBlocks[i] = liveBlocks[--numLive];
    liveBytes -= block.size;
    poolBlock(block, 0, 0);
    end(ALLOCATE, start);
    return;
  }
  /* Memory that is not recorded goes back to the driver once the work that
     may use it is done. */
  finish();
  check(cuMemFree(address), "cuMemFree");
  ++numToDriver;
  end(ALLOCATE, start);
}

/* Gives an idle arena back to the driver, once the work that may use it is
   done unless the caller has waited for it. */
static void releaseArena(struct Block arena, int gone, int waited) {
  --numDriverBlocks;
  reservedBytes -= arena.baseSize;
  if (gone)
    return;
  if (!waited)
    finish();
  check(cuMemFree(arena.base), "cuMemFree");
  ++numToDriver;
}

/* Adds an idle piece to the list. With `merge`, it is merged first with
   the idle pieces next to it in its arena. An arena that is idle as a whole
   goes back to the driver if the budget has no room for it. */
static void addIdle(struct Block block, int gone, int replaced, int merge) {
  block.owner = NULL;
  block.size = block.extent;
  /* The idle neighbors in the arena: one ends where the block begins, one
     begins where it ends. Merged pieces are never next to each other, so
     one pass over the list finds both of a piece being merged. The merged
     piece keeps the tag of its larger part. */
  for (int i = numFree; merge && i-- != 0;) {
    struct Block other = freeBlocks[i];
    if (other.base != block.base)
      continue;
    if (other.pointer + other.extent == block.pointer)
      block.pointer = other.pointer;
    else if (block.pointer + block.extent != other.pointer)
      continue;
    if (other.extent > block.extent)
      block.tag = other.tag;
    block.extent += other.extent;
    block.size = block.extent;
    block.asked = 0;
    poolBytes -= other.extent;
    freeBlocks[i] = freeBlocks[--numFree];
  }
  int whole = block.pointer == block.base && block.extent == block.baseSize;
  if (whole && poolBytes + block.extent > getPoolBudget()) {
    /* Past the budget, an idle arena goes back to the driver. */
    releaseArena(block, gone, replaced);
    return;
  }
  if (!reserveBlock(&freeBlocks, numFree, &roomFree)) {
    if (whole) {
      releaseArena(block, gone, replaced);
      return;
    }
    fprintf(stderr, "mdrt: out of memory of the host for the list of the "
                    "idle pieces of the device\n");
    abort();
  }
  freeBlocks[numFree++] = block;
  poolBytes += block.extent;
}

/* A freed block keeps its piece as it is, unmerged, so that the next
   request of its size takes it again: coalescing is deferred until a
   request finds no idle piece that holds it, a trim, the close of an
   activation, or a free that takes the idle pieces past the budget
   (mergeIdle), where an arena that the merge finds idle as a whole goes
   back to the driver. */
static void poolBlock(struct Block block, int gone, int replaced) {
  if (replaced && block.pointer == block.base &&
      block.extent == block.baseSize) {
    releaseArena(block, gone, 1);
    return;
  }
  block.asked = block.size;
  addIdle(block, gone, replaced, 0);
  if (!gone && poolBytes > getPoolBudget())
    mergeIdle();
}

/* Merges every idle piece with its idle neighbors in its arena, and returns
   an arena that is then idle as a whole to the driver if the budget has no
   room for it: the idle pieces are taken out of the list and added again,
   merged. A piece that merges with none keeps its size asked. The work that
   may still use a returned arena is waited for first (releaseArena). */
static void mergeIdle(void) {
  int count = numFree;
  if (!count)
    return;
  struct Block *pieces = malloc((size_t)count * sizeof(struct Block));
  if (!pieces) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  memcpy(pieces, freeBlocks, (size_t)count * sizeof(struct Block));
  numFree = 0;
  poolBytes = 0;
  for (int i = 0; i != count; ++i)
    addIdle(pieces[i], 0, 0, 1);
  free(pieces);
}

/* Returns a buffer of the runtime's structures that a growth replaces to
   the driver, not to the pool (D312): the buffer that replaces it is
   larger, and a later request of the old size is unlikely, so in the pool
   it would stay for the rest of the run. That holds for a buffer that is
   an arena of its own. A piece of a larger arena cannot go back alone: it
   returns to the pool, where a smaller request may take it, and its arena
   follows the budget's rule. The caller has waited for the device, as a
   growth does, so the free does not wait. */
static void releaseReplaced(void *pointer) {
  if (!pointer)
    return;
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = numLive; i-- != 0;) {
    if (liveBlocks[i].pointer != address)
      continue;
    struct Block block = liveBlocks[i];
    liveBytes -= block.size;
    liveBlocks[i] = liveBlocks[--numLive];
    poolBlock(block, 0, 1);
    return;
  }
  check(cuMemFree(address), "cuMemFree");
  ++numToDriver;
}

/* Records a structure of neighbors that the current activation made. */
static void recordCallHandle(void *handle) {
  struct Activation *a = currentActivation;
  if (!a)
    return;
  if (a->numHandles == a->roomHandles)
    a->handles = growList(a->handles, &a->roomHandles, sizeof(*a->handles));
  a->handles[a->numHandles++] = handle;
}

/* Returns true if `pointer` is memory of the device: in a block that the
   library has allocated, all of which it records (#622). It does not ask
   the driver, which reports an error for memory of the host (and tools
   such as compute-sanitizer count it). */
static bool isOnDevice(const void *pointer) {
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = 0; i != numLive; ++i)
    if (address >= liveBlocks[i].pointer &&
        address < liveBlocks[i].pointer + liveBlocks[i].size)
      return true;
  return false;
}

static bool auditDevicePointer(const void *pointer) {
  uintptr_t address = (uintptr_t)pointer;
  for (unsigned i = 0; i != copyAudit->ranges; ++i)
    if (address >= copyAudit->first[i] && address < copyAudit->last[i])
      return true;
  return isOnDevice(pointer);
}
static void auditGenericCopy(void *destination, const void *source, uint64_t size) {
  if (!copyAudit || !size) return;
  if (copyMetadataDestination == (uintptr_t)destination) {
    copyAudit->metadataDownloads += size;
    copyMetadataDestination = 0;
    return;
  }
  copyMetadataDestination = 0;
  bool toDevice = auditDevicePointer(destination);
  bool fromDevice = auditDevicePointer(source);
  if (toDevice && !fromDevice) copyAudit->uploads += size;
  if (fromDevice && !toDevice) copyAudit->downloads += size;
}

/* Copies between the host and the device, in either direction: the driver
   tells them apart by the address. */
void mgpuMemcpy(void *destination, void *source, size_t size,
                CUstream stream) {
  if (size == 0)
    return;
  mdrtSpansFlush();
  auditGenericCopy(destination, source, size);
  if (isOnDevice(source) && !isOnDevice(destination))
    mdrtCountHostRead(size, 0, 0);
  double start = begin();
  check(cuMemcpyAsync((CUdeviceptr)destination, (CUdeviceptr)source, size,
                      stream),
        "cuMemcpyAsync");
  isPending = 1;
  end(COPY, start);

  /* The host reads what it has copied from the device, and it may change
     what it has copied to the device. Either way the copy must be done
     when the function returns. */
  if (!isOnDevice(destination) || !isOnDevice(source))
    finish();
}

/* Completion on one actual producer queue, without draining the other.
   MDRT_WAIT retains its explicitly requested debugging synchronization. */
static void waitScoped(CUstream stream) {
  mdrtSpansFlush();
  double start = begin();
  CUevent completed = mgpuEventCreate();
  mgpuEventRecord(completed, stream);
  mgpuEventSynchronize(completed);
  mgpuEventDestroy(completed);
  end(WAIT, start);
  mgpuStreamSynchronize(stream);
}

/* A checked host boundary of the resident FFT strategy
   (D360). The compiler establishes the source/destination
   extents, lifetime, and producer queue before selecting this entry point.
   Completion is host-visible when it returns, but unrelated side work is
   not drained. Keep isPending conservative: that work may still be live. */
void mdrtCudaMemcpyScoped(uintptr_t destination, uintptr_t source,
                          uint64_t size) {
  if (size == 0)
    return;
  auditGenericCopy((void *)destination, (const void *)source, size);
  if (isOnDevice((void *)source) && !isOnDevice((void *)destination))
    mdrtCountHostRead(size, 0, 0);
  CUstream stream = mgpuStreamCreate();
  mdrtSpansFlush();
  double start = begin();
  check(cuMemcpyAsync((CUdeviceptr)destination, (CUdeviceptr)source,
                      (size_t)size, stream),
        "cuMemcpyAsync");
  isPending = 1;
  end(COPY, start);
  waitScoped(stream);
}

/* An unused resident result also completes before its storage is released. */
void mdrtCudaWaitScoped(void) {
  waitScoped(mgpuStreamCreate());
}

void mgpuMemset32(void *destination, unsigned int value, size_t count,
                  CUstream stream) {
  if (isRecording() && (!stream || stream == sharedStream)) {
    struct SpanOp *op = appendOp(SPAN_MEMSET32);
    op->a = (uint64_t)(uintptr_t)destination;
    op->value = value;
    op->count = count;
    isPending = 1;
    return;
  }
  mdrtSpansFlush();
  check(cuMemsetD32Async((CUdeviceptr)destination, value, count, stream),
        "cuMemsetD32Async");
  isPending = 1;
}

void mgpuMemset16(void *destination, unsigned short value, size_t count,
                  CUstream stream) {
  if (isRecording() && (!stream || stream == sharedStream)) {
    struct SpanOp *op = appendOp(SPAN_MEMSET16);
    op->a = (uint64_t)(uintptr_t)destination;
    op->value = value;
    op->count = count;
    isPending = 1;
    return;
  }
  mdrtSpansFlush();
  check(cuMemsetD16Async((CUdeviceptr)destination, value, count, stream),
        "cuMemsetD16Async");
  isPending = 1;
}

/*===----------------------------------------------------------------------===
  FFT of particle mesh Ewald
  ===----------------------------------------------------------------------===

  The transforms of the template of particle mesh Ewald on a device, by
  cuFFT, on the stream that the kernels run on, so that they run in the
  order of the program without a wait. A plan is kept for each grid. */

#include <cufft.h>
#ifdef MDIR_PME_CALLBACK
#include <cufftXt.h>
#include "pme_callback.inc"
#endif
#include "pme_callback.h"

/* The descriptor of a buffer of one dimension of f64 on the device. */
typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} DeviceBuffer1D;

/* The same, of f32. */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} DeviceBuffer1DF32;

/* A failure that ends the run but need not end the process: the program
   that embeds the run (a simulation of the Python interface, D196) takes
   it through its handler, which records it, and the run goes on to the end
   of its part, whose state the program discards; without a handler the
   failure is printed and the process exits with status 1. The calls that
   fail return without their work. */
static void (*stopHandler)(const char *) = NULL;
void mdrtCudaSetStopHandler(void (*handler)(const char *)) {
  stopHandler = handler;
}

static void stopWith(const char *message) {
  if (stopHandler) {
    stopHandler(message);
    return;
  }
  fprintf(stderr, "mdir: %s\n", message);
  fflush(stderr);
  exit(1);
}

/* The plans of cuFFT of the process, one for each grid, direction, and
   precision that a program has transformed, kept until the process ends
   and shared by the programs that transform the same grid. A callback
   inverse also keys on the table address bound to the plan. A program in
   mixed precision with tunable charges takes four a grid (both directions,
   f32 and f64), so the table grows as programs of other grids come (#323);
   it held 8 before, and the ninth plan ended the process. */
struct FFTPlan {
  int64_t k1, k2, k3;
  int forward, single, graphUsers;
  const float *tables;
  CUdeviceptr callbackInfo;
  cufftHandle plan;
  /* The workspace and callback descriptor, and when the plan was last taken. */
  uint64_t bytes, lastUse;
};
static struct FFTPlan *fftPlans = NULL;
static int64_t numFFTPlans = 0, fftPlansCapacity = 0;
/* The workspaces of the plans together, and the count of the takings
   (#622): past the budget, MDIR_FFT_PLAN_BYTES or a quarter of the memory
   of the device, the plans least recently taken are destroyed, and a trim
   destroys them all; a plan is made anew when its grid comes back. */
static uint64_t fftPlanBytes = 0, fftUses = 0;
static uint64_t getFFTPlanBudget(void) {
  static int read = 0;
  static uint64_t budget = 0;
  if (!read) {
    read = 1;
    const char *value = getenv("MDIR_FFT_PLAN_BYTES");
    size_t free = 0, total = 0;
    if (value && *value)
      budget = strtoull(value, NULL, 10);
    else if (context && cuCtxSetCurrent(context) == CUDA_SUCCESS &&
             cuMemGetInfo(&free, &total) == CUDA_SUCCESS)
      budget = total / 4;
    else
      budget = UINT64_MAX;
  }
  return budget;
}
/* Destroys plan `i` once the work that may use it is done. */
static void destroyFFTPlan(int64_t i) {
  finish();
  cufftDestroy(fftPlans[i].plan);
  if (fftPlans[i].callbackInfo)
    cuMemFree(fftPlans[i].callbackInfo);
  fftPlanBytes -= fftPlans[i].bytes;
  fftPlans[i] = fftPlans[--numFFTPlans];
}

/* Whether `result` is a success; otherwise the run stops, naming `what`. */
static int checkFFT(cufftResult result, const char *what, int64_t k1,
                    int64_t k2, int64_t k3) {
  if (result == CUFFT_SUCCESS)
    return 1;
  char message[256];
  snprintf(message, sizeof message,
           "%s failed with the cuFFT error %d for the grid of %lld x %lld x "
           "%lld points; the run has failed",
           what, (int)result, (long long)k1, (long long)k2, (long long)k3);
  stopWith(message);
  return 0;
}

/* A callback descriptor belongs to the plan. Its address stays fixed while
   the tables kernel refreshes the values before each inverse transform. */
static cufftResult makeFFTPlan(cufftHandle *plan, int k1, int k2, int k3,
                               cufftType type, CUdeviceptr info) {
#ifdef MDIR_PME_CALLBACK
  if (info) {
    cufftResult result = cufftCreate(plan);
    if (result != CUFFT_SUCCESS)
      return result;
    void *caller = (void *)(uintptr_t)info;
    result = cufftXtSetJITCallback(*plan, "mdirGaussianLoad", pmeCallback,
                                   sizeof pmeCallback, CUFFT_CB_LD_COMPLEX,
                                   &caller);
    size_t work;
    if (result == CUFFT_SUCCESS)
      result = cufftMakePlan3d(*plan, k1, k2, k3, type, &work);
    if (result != CUFFT_SUCCESS)
      cufftDestroy(*plan);
    return result;
  }
#endif
  return cufftPlan3d(plan, k1, k2, k3, type);
}

/* The plan of a grid, a direction, and a precision (f32 if `single`), on
   the stream of the kernels, in `plan`; 0 if it cannot be made, and the
   run stops. */
static int getFFTPlan(int64_t k1, int64_t k2, int64_t k3, int forward,
                      int single, const float *tables, cufftHandle *plan) {
  for (int64_t i = 0; i != numFFTPlans; ++i)
    if (fftPlans[i].k1 == k1 && fftPlans[i].k2 == k2 &&
        fftPlans[i].k3 == k3 && fftPlans[i].forward == forward &&
        fftPlans[i].single == single && fftPlans[i].tables == tables) {
      fftPlans[i].lastUse = ++fftUses;
      *plan = fftPlans[i].plan;
      return checkFFT(cufftSetStream(*plan, (cudaStream_t)mgpuStreamCreate()),
                      "cufftSetStream", k1, k2, k3);
    }
  if (numFFTPlans == fftPlansCapacity) {
    int64_t capacity = fftPlansCapacity ? 2 * fftPlansCapacity : 8;
    struct FFTPlan *grown =
        realloc(fftPlans, (size_t)capacity * sizeof(struct FFTPlan));
    if (!grown) {
      char message[160];
      snprintf(message, sizeof message,
               "no memory of the host for a plan of FFT of the grid of %lld x "
               "%lld x %lld points; the run has failed",
               (long long)k1, (long long)k2, (long long)k3);
      stopWith(message);
      return 0;
    }
    fftPlans = grown;
    fftPlansCapacity = capacity;
  }
  cufftType type = single ? (forward ? CUFFT_R2C : CUFFT_C2R)
                          : (forward ? CUFFT_D2Z : CUFFT_Z2D);
  cufftHandle made;
  int64_t before = beginAccount();
  CUdeviceptr info = 0;
#ifdef MDIR_PME_CALLBACK
  if (tables) {
    PMECallbackInfo desc = {tables, (unsigned)k1, (unsigned)k2, (unsigned)k3};
    CUresult allocated = cuMemAlloc(&info, sizeof desc);
    if (allocated == CUDA_ERROR_OUT_OF_MEMORY && trimIdle())
      allocated = cuMemAlloc(&info, sizeof desc);
    check(allocated, "PME callback descriptor");
    check(cuMemcpyHtoD(info, &desc, sizeof desc), "PME callback descriptor");
  }
#endif
  cufftResult result = makeFFTPlan(&made, k1, k2, k3, type, info);
  if (result == CUFFT_ALLOC_FAILED && trimIdle())
    result = makeFFTPlan(&made, k1, k2, k3, type, info);
  if (result != CUFFT_SUCCESS) {
    if (info)
      cuMemFree(info);
    return checkFFT(result, tables ? "cuFFT Gaussian callback plan" : "cufftPlan3d",
                    k1, k2, k3);
  }
  /* Include the caller information owned by a callback plan in its budget. */
  size_t workspace = 0;
  cufftGetSize(made, &workspace);
  if (info)
    workspace += sizeof(PMECallbackInfo);
  if (before >= 0)
    endAccount(ACCOUNT_FFT, before, (int64_t)workspace);
  if (!checkFFT(cufftSetStream(made, (cudaStream_t)mgpuStreamCreate()),
                "cufftSetStream", k1, k2, k3)) {
    cufftDestroy(made);
    if (info)
      cuMemFree(info);
    return 0;
  }
  /* The plans least recently taken make room for this one in the budget. */
  while (fftPlanBytes + workspace > getFFTPlanBudget()) {
    int64_t oldest = -1;
    for (int64_t i = 0; i != numFFTPlans; ++i)
      if (!fftPlans[i].graphUsers &&
          (oldest < 0 || fftPlans[i].lastUse < fftPlans[oldest].lastUse))
        oldest = i;
    if (oldest < 0) break;
    destroyFFTPlan(oldest);
  }
  struct FFTPlan *entry = &fftPlans[numFFTPlans++];
  entry->k1 = k1;
  entry->k2 = k2;
  entry->k3 = k3;
  entry->forward = forward;
  entry->single = single;
  entry->tables = tables;
  entry->callbackInfo = info;
  entry->graphUsers = 0;
  entry->plan = made;
  entry->bytes = workspace;
  entry->lastUse = ++fftUses;
  fftPlanBytes += workspace;
  *plan = made;
  return 1;
}

/* A captured transform keeps its plan and workspace across graph launches.
   An aligned restart can reach capture before its first ordinary FFT. */
int mdrtRetainFFTPlan(int64_t k1, int64_t k2, int64_t k3, int forward, int single, const float *tables) {
  cufftHandle plan;
  enter();
  if (!getFFTPlan(k1, k2, k3, forward, single, tables, &plan)) return -1;
  for (int64_t i = 0; i != numFFTPlans; ++i)
    if (fftPlans[i].plan == plan) {
      ++fftPlans[i].graphUsers;
      return plan;
    }
  abort();
}

void mdrtReleaseFFTPlan(int handle) {
  for (int64_t i = 0; i != numFFTPlans; ++i)
    if (fftPlans[i].plan == handle) {
      if (--fftPlans[i].graphUsers == 0 && fftPlanBytes > getFFTPlanBudget())
        destroyFFTPlan(i);
      return;
    }
  abort();
}

static int isDriverGone(void);
/* No unretained plan is in use when a trim destroys the plans: each transform
   (mdrtCudaFFT*3D*) looks its plan up in getFFTPlan and executes it at
   once, with no allocation of the runtime between the two, and no handle
   is kept across calls. The trims within getFFTPlan run before the plan it
   returns is made (a failed cufftPlan3d) or entered in the cache (the
   eviction of others). The work already issued with a plan is run first
   (finish), so that destroying it cannot cut that work short. */
static uint64_t trimIdle(void) {
  if (isDriverGone())
    return 0;
  finish();
  /* The graphs of spans hold device memory of the driver; they are made
     again as the spans repeat. */
  releaseAllSpans();
  mergeIdle();
  /* Only an arena that is idle as a whole goes back; the idle pieces of an
     arena that holds a block in use stay. */
  uint64_t released = fftPlanBytes;
  for (int i = numFree; i-- != 0;) {
    struct Block piece = freeBlocks[i];
    if (piece.pointer != piece.base || piece.extent != piece.baseSize)
      continue;
    released += piece.extent;
    poolBytes -= piece.extent;
    freeBlocks[i] = freeBlocks[--numFree];
    releaseArena(piece, 0, 1);
  }
  for (int64_t i = numFFTPlans; i-- > 0;)
    if (!fftPlans[i].graphUsers) destroyFFTPlan(i);
  released -= fftPlanBytes;
  return released;
}

/* Gives the idle blocks of the pool and the plans of cuFFT back to the
   driver (#622), for an embedding program between simulations; the bytes
   given back. The blocks in use and those of open activations stay. */
uint64_t mdrtDeviceTrim(void) {
  if (!context)
    return 0;
  enter();
  return trimIdle();
}

/* The accounts of the memory of the device (#622), for tests: the blocks
   in use and their bytes, the idle blocks of the pool and their bytes, the
   blocks of the pool's allocator that the driver holds, the plans of
   cuFFT and their workspaces, and the memory of the device that the driver
   reports free (-1 without a context). */
void mdrtDeviceMemoryCounts(int64_t counts[8]) {
  counts[0] = numLive;
  counts[1] = (int64_t)liveBytes;
  counts[2] = numFree;
  counts[3] = (int64_t)poolBytes;
  counts[4] = numDriverBlocks;
  counts[5] = numFFTPlans;
  counts[6] = (int64_t)fftPlanBytes;
  size_t free = 0, total = 0;
  counts[7] = context && cuCtxSetCurrent(context) == CUDA_SUCCESS &&
                      cuMemGetInfo(&free, &total) == CUDA_SUCCESS
                  ? (int64_t)free
                  : -1;
}

/* Records a transform while spans record (SPAN_FFT); 1 if recorded. */
static int recordFFT(int fft, cufftHandle plan, const void *in, void *out) {
  if (!isRecording())
    return 0;
  struct SpanOp *op = appendOp(SPAN_FFT);
  op->fft = fft;
  op->plan = plan;
  op->a = (uint64_t)(uintptr_t)in;
  op->b = (uint64_t)(uintptr_t)out;
  isPending = 1;
  return 1;
}

/* Issues a recorded transform; 0 if cuFFT failed. */
static int issueFFT(int fft, int plan, uint64_t in, uint64_t out) {
  cufftResult result = CUFFT_SUCCESS;
  switch (fft) {
  case 0:
    result = cufftExecD2Z(plan, (cufftDoubleReal *)(uintptr_t)in,
                          (cufftDoubleComplex *)(uintptr_t)out);
    break;
  case 1:
    result = cufftExecZ2D(plan, (cufftDoubleComplex *)(uintptr_t)in,
                          (cufftDoubleReal *)(uintptr_t)out);
    break;
  case 2:
    result = cufftExecR2C(plan, (cufftReal *)(uintptr_t)in,
                          (cufftComplex *)(uintptr_t)out);
    break;
  case 3:
    result = cufftExecC2R(plan, (cufftComplex *)(uintptr_t)in,
                          (cufftReal *)(uintptr_t)out);
    break;
  }
  return result == CUFFT_SUCCESS;
}

/* The graph of a span holds the plans of its transforms (graphUsers), so
   that a trim or the budget does not destroy them under it. */
static void retainFFTPlanHandle(int plan, int delta) {
  for (int64_t i = 0; i != numFFTPlans; ++i)
    if (fftPlans[i].plan == plan) {
      fftPlans[i].graphUsers += delta;
      return;
    }
}

/* The forward transform of the real grid of k1 x k2 x k3 points into the
   half-complex grid of k1 x k2 x (k3 / 2 + 1) numbers, interleaved, as
   mdrtFFTForward3D of the host computes it. */
void _mlir_ciface_mdrtCudaFFTForward3D(DeviceBuffer1D *real,
                                       DeviceBuffer1D *complex, int64_t k1,
                                       int64_t k2, int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  double *in = real->aligned + real->offset;
  double *out = complex->aligned + complex->offset;
  if (getFFTPlan(k1, k2, k3, 1, 0, NULL, &plan) &&
      !recordFFT(0, plan, in, out)) {
    mdrtSpansFlush();
    checkFFT(cufftExecD2Z(plan, (cufftDoubleReal *)in,
                          (cufftDoubleComplex *)out),
             "cufftExecD2Z", k1, k2, k3);
  }
  isPending = 1;
  end(LAUNCH, start);
}

/* The backward transform, not normalized, which overwrites the
   half-complex grid. */
void _mlir_ciface_mdrtCudaFFTBackward3D(DeviceBuffer1D *complex,
                                        DeviceBuffer1D *real, int64_t k1,
                                        int64_t k2, int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  double *in = complex->aligned + complex->offset;
  double *out = real->aligned + real->offset;
  if (getFFTPlan(k1, k2, k3, 0, 0, NULL, &plan) &&
      !recordFFT(1, plan, in, out)) {
    mdrtSpansFlush();
    checkFFT(cufftExecZ2D(plan, (cufftDoubleComplex *)in,
                          (cufftDoubleReal *)out),
             "cufftExecZ2D", k1, k2, k3);
  }
  isPending = 1;
  end(LAUNCH, start);
}

/* The transforms of a grid in f32, for the mixed and single modes. */
void _mlir_ciface_mdrtCudaFFTForward3DF32(DeviceBuffer1DF32 *real,
                                          DeviceBuffer1DF32 *complex,
                                          int64_t k1, int64_t k2,
                                          int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  float *in = real->aligned + real->offset;
  float *out = complex->aligned + complex->offset;
  if (getFFTPlan(k1, k2, k3, 1, 1, NULL, &plan) &&
      !recordFFT(2, plan, in, out)) {
    mdrtSpansFlush();
    checkFFT(cufftExecR2C(plan, (cufftReal *)in, (cufftComplex *)out),
             "cufftExecR2C", k1, k2, k3);
  }
  isPending = 1;
  end(LAUNCH, start);
}

static void fftBackwardF32(DeviceBuffer1DF32 *complex, DeviceBuffer1DF32 *real,
                           const float *tables, int64_t k1, int64_t k2,
                           int64_t k3) {
  enter();
  double start = begin();
  cufftHandle plan;
  float *in = complex->aligned + complex->offset;
  float *out = real->aligned + real->offset;
  if (getFFTPlan(k1, k2, k3, 0, 1, tables, &plan) &&
      !recordFFT(3, plan, in, out)) {
    mdrtSpansFlush();
    checkFFT(cufftExecC2R(plan, (cufftComplex *)in, (cufftReal *)out),
             "cufftExecC2R", k1, k2, k3);
  }
  isPending = 1;
  end(LAUNCH, start);
}

void _mlir_ciface_mdrtCudaFFTBackward3DF32(DeviceBuffer1DF32 *complex,
                                           DeviceBuffer1DF32 *real,
                                           int64_t k1, int64_t k2, int64_t k3) {
  fftBackwardF32(complex, real, NULL, k1, k2, k3);
}

#ifdef MDIR_PME_CALLBACK
void _mlir_ciface_mdrtCudaFFTGaussianBackwardF32(
    DeviceBuffer1DF32 *complex, DeviceBuffer1DF32 *real, DeviceBuffer1DF32 *tables,
    int64_t k1, int64_t k2, int64_t k3) {
  fftBackwardF32(complex, real, tables->aligned + tables->offset, k1, k2, k3);
}
#endif

/*===----------------------------------------------------------------------===
 * The buffers of a structure of groups of neighbors (D89)
 *===----------------------------------------------------------------------===*/

/* The buffers that a build of a structure of groups fills, which grow when
   a build finds them too small: the particle at each place, the number of
   entries of each group (a group is 16 places), the entries and the masks
   in blocks of 64, the group and the number within its list of each
   block, and the shift of each place to the frame of its group (D95).
   A dual list (D114) adds the entries and the masks of the inner list,
   each block of the outer list pruned into the same block, and the number
   of entries that each block keeps; they are allocated where compiled code
   first asks for them. The pruning of a dual list adds, also where first
   asked for (D268): the block of each group's list by its
   number within the list (`blockOf`, a group's row as wide as the longest
   list takes blocks; its length is set by `mdrtGroupsReserveBlockOf` at
   each build), the units of work that hold entries of the inner list and
   their number (the active units, in the order of the blocks), their
   counts in runs of 32 and of 1024 blocks (on the way to the active
   units), and the positions of the places in the frames of their
   groups in f32 (`mdrtGroupsPlaced`, three numbers a place). Compiled code
   takes a buffer where it uses it (`mdrtGroupsBuffer`), so a buffer that
   grew is the one it takes. */

struct Groups {
  void *data[GROUPS_BUFFERS];
  int64_t places;
  int64_t blocks;
  /* The length of `blockOf`. */
  int64_t blockOf;
  /* Whether a build of the structure has completed, and the growths since
     that build that `mdrtGroupsBuilt` has not yet handed over. The first
     build of a structure sizes its blocks (D331); a growth after
     it is noted in the log. */
  bool built;
  int64_t lateGrowths;
  /* The segments of a list of units, 1 for groups
     (mdrtGroupsSetSegments). */
  int64_t segments;
  /* The frames of a list of units and their rows of 32-bit values a place
     (mdrtGroupsFrame). */
  void *frames[MDRT_GROUPS_FRAMES];
  int64_t frameValues[MDRT_GROUPS_FRAMES];
};

/* A memref of rank 1 as the C interface of MLIR passes it. */
static int64_t getGroupsLength(const struct Groups *groups, int which) {
  switch (mdrtGroupsDescriptor(which)->shape) {
  case MDRT_GROUPS_PLACES: return groups->places;
  case MDRT_GROUPS_GROUPS: return groups->places / 16;
  case MDRT_GROUPS_ENTRIES: return groups->blocks * 64;
  case MDRT_GROUPS_BLOCK_OF: return groups->blockOf > 0 ? groups->blockOf : 1;
  case MDRT_GROUPS_SCALAR: return 1;
  case MDRT_GROUPS_RUNS: return groups->blocks / 32 + 1;
  case MDRT_GROUPS_WIDE_RUNS: return groups->blocks / 1024 + 1;
  case MDRT_GROUPS_PLACED: return groups->places * 3;
  case MDRT_GROUPS_BLOCKS: return groups->blocks;
  case MDRT_GROUPS_SEGMENT_LISTS: return groups->places / 16 * groups->segments;
  case MDRT_GROUPS_SEGMENTS: return groups->segments + 1;
  }
  abort();
}

static void allocateGroup(struct Groups *groups, int which) {
  allocationTag = mdrtGroupsDescriptor(which)->role;
  groups->data[which] = mgpuMemAlloc(
      (uint64_t)getGroupsLength(groups, which) * sizeof(int32_t), NULL,
      false);
  allocationTag = NULL;
}

/* The buffers of the outer list; those of the inner one wait for their
   first use. */
static void allocateGroups(struct Groups *groups) {
  for (int which = 0; which != GROUPS_INNER_ENTRIES; ++which)
    allocateGroup(groups, which);
}

/* After a wait for the device: the buffers go back to the driver. */
static void releaseGroups(struct Groups *groups) {
  for (int which = 0; which != GROUPS_BUFFERS; ++which) {
    releaseReplaced(groups->data[which]);
    groups->data[which] = NULL;
  }
  for (int which = 0; which != MDRT_GROUPS_FRAMES; ++which) {
    releaseReplaced(groups->frames[which]);
    groups->frames[which] = NULL;
  }
}

/* A structure with room for `places` places (a multiple of 16) and
   `blocks` blocks. The lowering asks for one block a group: the first build
   counts the blocks it needs, and its growth sizes the pool
   (D331). */
int64_t mdrtGroupsCreate(int64_t places, int64_t blocks) {
  struct Groups *groups = calloc(1, sizeof(struct Groups));
  if (!groups) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  groups->places = (places + 15) / 16 * 16;
  groups->blocks = blocks > 0 ? blocks : 1;
  groups->segments = 1;
  allocateGroups(groups);
  recordCallHandle(groups);
  return (int64_t)(intptr_t)groups;
}

/* A list of units in `segments` segments (D417): the buffers of its lists
   by group and segment, and of its offsets, take that many. Called before
   the first use of those buffers. */
void mdrtGroupsSetSegments(int64_t handle, int64_t segments) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  groups->segments = segments > 0 ? segments : 1;
}

void _mlir_ciface_mdrtGroupsBuffer(struct Buffer1 *result, int64_t handle,
                                   int64_t which) {
  if (!mdrtGroupsDescriptor(which)) {
    fprintf(stderr, "mdrt: invalid groups buffer slot %lld (expected 0..%d)\n",
            (long long)which, GROUPS_BUFFERS - 1);
    abort();
  }
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (!groups->data[which])
    allocateGroup(groups, (int)which);
  result->allocated = groups->data[which];
  result->aligned = groups->data[which];
  result->offset = 0;
  result->size = getGroupsLength(groups, (int)which);
  result->stride = 1;
}

/* For the tests of the lists of units (convert-md-exec-to-gpu with
   dump-segments, D419): writes the list of `handle`, of
   `places` places and `records` records, after `builds` builds of it and
   `prunes` prunings of every list, to dump_<n>.bin in the directory
   that MDRT_SEGMENTS_DUMP names, n counting the dumps from 0. The file holds
   six i64 (n, places, records, segments, the builds and the prunings so
   far), then for each buffer of SEGMENT_ORDER, SHIFTS, RECORD_ENTRIES,
   RECORD_MASKS, RECORD_GROUPS, RECORD_COUNTS, SEGMENT_OFFSETS,
   INNER_ENTRIES, INNER_MASKS, INNER_COUNTS, ACTIVE_UNITS, ACTIVE_COUNT and
   ORDER (after a build, the reference particle of each place) its length in
   i32 (0 where the list has not allocated it) and its values. */
void mdrtSegmentsDump(int64_t handle, int64_t places, int64_t records,
                      int64_t builds, int64_t prunes) {
  const char *dir = getenv("MDRT_SEGMENTS_DUMP");
  if (!dir)
    return;
  static int64_t serial = 0;
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  enter();
  mdrtSpansFlush();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  char path[4096];
  snprintf(path, sizeof path, "%s/dump_%06lld.bin", dir, (long long)serial);
  FILE *file = fopen(path, "wb");
  if (!file) {
    fprintf(stderr, "mdrt: cannot write %s\n", path);
    abort();
  }
  int64_t header[6] = {serial, places, records, groups->segments,
                       builds, prunes};
  fwrite(header, sizeof header, 1, file);
  struct { int slot; int64_t length; } parts[] = {
      {GROUPS_SEGMENT_ORDER, places},     {GROUPS_SHIFTS, places},
      {GROUPS_RECORD_ENTRIES, records * 64}, {GROUPS_RECORD_MASKS, records * 64},
      {GROUPS_RECORD_GROUPS, records},    {GROUPS_RECORD_COUNTS, records},
      {GROUPS_SEGMENT_OFFSETS, groups->segments + 1},
      {GROUPS_INNER_ENTRIES, records * 64}, {GROUPS_INNER_MASKS, records * 64},
      {GROUPS_INNER_COUNTS, records},     {GROUPS_ACTIVE_UNITS, records},
      {GROUPS_ACTIVE_COUNT, 1},           {GROUPS_ORDER, places}};
  for (size_t k = 0; k != sizeof parts / sizeof parts[0]; ++k) {
    int64_t length = groups->data[parts[k].slot] ? parts[k].length : 0;
    fwrite(&length, sizeof length, 1, file);
    if (!length)
      continue;
    int32_t *host = malloc((size_t)length * sizeof(int32_t));
    mdrtSpansFlush();
    check(cuMemcpyDtoH(host, (CUdeviceptr)groups->data[parts[k].slot],
                       (size_t)length * sizeof(int32_t)),
          "cuMemcpyDtoH");
    fwrite(host, sizeof(int32_t), (size_t)length, file);
    free(host);
  }
  fclose(file);
  ++serial;
}

/* Makes room for `places` places and `blocks` blocks, a quarter more than
   asked, if the structure has less. What the buffers held is lost: the
   caller builds again. */
void mdrtGroupsGrow(int64_t handle, int64_t places, int64_t blocks) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (places <= groups->places && blocks <= groups->blocks)
    return;
  /* No kernel may still use the buffers that are freed. */
  enter();
  mdrtSpansFlush();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  releaseGroups(groups);
  if (places > groups->places)
    groups->places = (places + places / 4 + 15) / 16 * 16;
  bool lists = blocks > groups->blocks;
  if (lists)
    groups->blocks = blocks + blocks / 4;
  allocateGroups(groups);
  /* The log counts the growths of the lists, not those of the places. */
  if (groups->built && lists)
    ++groups->lateGrowths;
}

/* After a complete build: the structure has been built, and the growths
   since its first build that this build made (0 or 1, in general) go to the
   log through `mdrtNoteGroups`. */
int64_t mdrtGroupsBuilt(int64_t handle) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  int64_t late = groups->lateGrowths;
  groups->built = true;
  groups->lateGrowths = 0;
  return late;
}

/* Makes room for `length` entries of `blockOf`, a quarter more than asked,
   if it has less. What it held is lost: the caller fills it again. */
void mdrtGroupsReserveBlockOf(int64_t handle, int64_t length) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (length <= groups->blockOf && groups->data[GROUPS_BLOCK_OF])
    return;
  if (length > groups->blockOf)
    groups->blockOf = length + length / 4;
  if (groups->data[GROUPS_BLOCK_OF]) {
    /* No kernel may still use the buffer that is freed. */
    enter();
    mdrtSpansFlush();
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    releaseReplaced(groups->data[GROUPS_BLOCK_OF]);
    groups->data[GROUPS_BLOCK_OF] = NULL;
  }
  allocateGroup(groups, GROUPS_BLOCK_OF);
}

/* Frame `which` of a list of units (D422),
   `values` 32-bit values a place, allocated at its first use after a
   growth. Its rows are of `width` values. */
static void *getGroupsFrame(struct Groups *groups, int64_t which,
                            int64_t values) {
  if (which < 0 || which >= MDRT_GROUPS_FRAMES || values <= 0) {
    fprintf(stderr, "mdrt: invalid frame %lld of %lld values a place\n",
            (long long)which, (long long)values);
    abort();
  }
  if (groups->frames[which] && groups->frameValues[which] != values) {
    fprintf(stderr, "mdrt: frame %lld taken with %lld and %lld values a "
                    "place\n",
            (long long)which, (long long)groups->frameValues[which],
            (long long)values);
    abort();
  }
  if (!groups->frames[which]) {
    allocationTag = "units: frames";
    groups->frames[which] = mgpuMemAlloc(
        (uint64_t)(groups->places * values) * sizeof(int32_t), NULL, false);
    allocationTag = NULL;
    groups->frameValues[which] = values;
  }
  return groups->frames[which];
}

/* A frame of `rows` rows of three f32 a place, as a memref of rank 2. */
void _mlir_ciface_mdrtGroupsFrame(struct Buffer2 *result, int64_t handle,
                                  int64_t which, int64_t rows) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  void *data = getGroupsFrame(groups, which, 3 * rows);
  result->allocated = data;
  result->aligned = data;
  result->offset = 0;
  result->sizes[0] = groups->places * rows;
  result->sizes[1] = 3;
  result->strides[0] = 3;
  result->strides[1] = 1;
}

/* A frame of `rows` values a place, f32 or i32, as a memref of rank 1. */
static void getGroupsFrameField(struct Buffer1 *result, int64_t handle,
                                int64_t which, int64_t rows) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  void *data = getGroupsFrame(groups, which, rows);
  result->allocated = data;
  result->aligned = data;
  result->offset = 0;
  result->size = groups->places * rows;
  result->stride = 1;
}
void _mlir_ciface_mdrtGroupsFrameFieldF32(struct Buffer1 *result,
                                          int64_t handle, int64_t which,
                                          int64_t rows) {
  getGroupsFrameField(result, handle, which, rows);
}
void _mlir_ciface_mdrtGroupsFrameFieldI32(struct Buffer1 *result,
                                          int64_t handle, int64_t which,
                                          int64_t rows) {
  getGroupsFrameField(result, handle, which, rows);
}

/* The positions of the places in the frames of their groups, f32 and three
   a place, as the C interface of MLIR passes a memref of rank 2. */
void _mlir_ciface_mdrtGroupsPlaced(struct Buffer2 *result, int64_t handle) {
  struct Groups *groups = (struct Groups *)(intptr_t)handle;
  if (!groups->data[GROUPS_PLACED])
    allocateGroup(groups, GROUPS_PLACED);
  result->allocated = groups->data[GROUPS_PLACED];
  result->aligned = groups->data[GROUPS_PLACED];
  result->offset = 0;
  result->sizes[0] = groups->places;
  result->sizes[1] = 3;
  result->strides[0] = 3;
  result->strides[1] = 1;
}

/*===----------------------------------------------------------------------===
 * The rows of a neighbor matrix
 *===----------------------------------------------------------------------===*/

/* The entries of a neighbor matrix, a row for each particle, which grow
   wider when a build finds a particle with more neighbors than a row
   holds. Compiled code takes the buffer where it uses it
   (`mdrtMatrixEntries`). */
struct Matrix {
  void *data;
  int64_t rows;
  int64_t width;
};

static void allocateMatrix(struct Matrix *matrix) {
  allocationTag = "matrix: rows";
  matrix->data = mgpuMemAlloc(
      (uint64_t)(matrix->rows * matrix->width) * sizeof(int32_t), NULL,
      false);
  allocationTag = NULL;
}

int64_t mdrtMatrixCreate(int64_t rows, int64_t width) {
  struct Matrix *matrix = calloc(1, sizeof(struct Matrix));
  if (!matrix) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  matrix->rows = rows > 0 ? rows : 1;
  matrix->width = width > 0 ? width : 1;
  allocateMatrix(matrix);
  recordCallHandle(matrix);
  return (int64_t)(intptr_t)matrix;
}

void _mlir_ciface_mdrtMatrixEntries(struct Buffer2 *result, int64_t handle) {
  struct Matrix *matrix = (struct Matrix *)(intptr_t)handle;
  result->allocated = matrix->data;
  result->aligned = matrix->data;
  result->offset = 0;
  result->sizes[0] = matrix->rows;
  result->sizes[1] = matrix->width;
  result->strides[0] = matrix->width;
  result->strides[1] = 1;
}

/* Makes the rows hold `width` entries, a quarter more than asked, if they
   hold fewer. What they held is lost: the caller builds again. */
void mdrtMatrixGrow(int64_t handle, int64_t width) {
  struct Matrix *matrix = (struct Matrix *)(intptr_t)handle;
  if (width <= matrix->width)
    return;
  enter();
  mdrtSpansFlush();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  releaseReplaced(matrix->data);
  matrix->width = width + width / 4;
  allocateMatrix(matrix);
}

/*===----------------------------------------------------------------------===
 * The slots of the pairs of images of a learned potential
 *===----------------------------------------------------------------------===*/

/* The candidates of the pairs of images that each build of a neighbor
   matrix gives its particles, a row (j, s_a, s_b, s_c) for each, and the
   slots that each step writes them into, a row (i, j, s_a, s_b, s_c) for
   each (D277, docs/metatomic-ml1.md, Section 12).
   A build sets their number, `length`; the buffers grow, a quarter more
   than asked, when a build needs more. What they held is lost: the build
   writes the candidates anew. Compiled code takes a buffer where it uses it
   (`mdrtEdgesBuffer`). */
struct Edges {
  void *candidates;
  void *slots;
  int64_t capacity;
  int64_t length;
};

static void allocateEdges(struct Edges *edges) {
  allocationTag = "edges: candidates";
  edges->candidates = mgpuMemAlloc(
      (uint64_t)edges->capacity * 4 * sizeof(int32_t), NULL, false);
  allocationTag = "edges: slots";
  edges->slots = mgpuMemAlloc(
      (uint64_t)edges->capacity * 5 * sizeof(int32_t), NULL, false);
  allocationTag = NULL;
}

int64_t mdrtEdgesCreate(void) {
  struct Edges *edges = calloc(1, sizeof(struct Edges));
  if (!edges) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  edges->capacity = 1;
  allocateEdges(edges);
  recordCallHandle(edges);
  return (int64_t)(intptr_t)edges;
}

/* The builds of slots in the process, which numbers each. */
static int64_t edgeBuilds = 0;

/* Makes the buffers hold `length` rows, and sets their number. Returns the
   number of this build in the process, which no other build has, so that a
   reader can tell that the slots changed. */
int64_t mdrtEdgesReserve(int64_t handle, int64_t length) {
  struct Edges *edges = (struct Edges *)(intptr_t)handle;
  if (length < 0) {
    fprintf(stderr, "mdrt: a build counted %lld pairs of images\n",
            (long long)length);
    abort();
  }
  if (length > edges->capacity) {
    enter();
    mdrtSpansFlush();
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    releaseReplaced(edges->candidates);
    releaseReplaced(edges->slots);
    edges->capacity = length + length / 4;
    allocateEdges(edges);
  }
  edges->length = length;
  return ++edgeBuilds;
}

/* The candidates (`which` 0, four numbers a row) or the slots (1, five),
   as many rows as the last build set. */
void _mlir_ciface_mdrtEdgesBuffer(struct Buffer2 *result, int64_t handle,
                                  int64_t which) {
  struct Edges *edges = (struct Edges *)(intptr_t)handle;
  int64_t columns = which == 0 ? 4 : 5;
  void *data = which == 0 ? edges->candidates : edges->slots;
  result->allocated = data;
  result->aligned = data;
  result->offset = 0;
  result->sizes[0] = edges->length;
  result->sizes[1] = columns;
  result->strides[0] = columns;
  result->strides[1] = 1;
}

/*===----------------------------------------------------------------------===
 * Calls of an entry by an embedding program
 *===----------------------------------------------------------------------===*/

/* An embedding program opens an activation before it first calls an entry
   (D215), enters it whenever the code of the activation
   runs, leaves it when the code waits between parts, and closes it when
   the activation ends: its blocks return to the pool, where later
   activations take the blocks of the same sizes. */
void *mdrtDeviceActivationOpen(void) {
  struct Activation *a = calloc(1, sizeof(struct Activation));
  if (!a) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  return a;
}

void mdrtDeviceActivationEnter(void *activation) {
  currentActivation = activation;
}

void mdrtDeviceActivationLeave(void) { currentActivation = NULL; }

/* Whether the driver has shut down, as it may have when a simulation ends
   after the program's own end (mgpuModuleUnload): the memory of the device
   is then gone with it. */
static int isDriverGone(void) {
  if (!context)
    return 1;
  CUresult result = cuCtxSetCurrent(context);
  return result == CUDA_ERROR_DEINITIALIZED ||
         result == CUDA_ERROR_INVALID_CONTEXT ||
         result == CUDA_ERROR_CONTEXT_IS_DESTROYED;
}

void mdrtDeviceActivationClose(void *activation) {
  struct Activation *a = activation;
  if (!a)
    return;
  if (currentActivation == a)
    currentActivation = NULL;
  int gone = isDriverGone();
  /* No work of the activation may still use a block that is reused or
     freed. */
  if (!gone)
    finish();
  for (int i = numLive; i-- != 0;) {
    if (liveBlocks[i].owner != a)
      continue;
    struct Block block = liveBlocks[i];
    liveBlocks[i] = liveBlocks[--numLive];
    liveBytes -= block.size;
    poolBlock(block, gone, 0);
  }
  /* The blocks of the activation merge with their idle neighbors, so that
     the arenas idle as a whole follow the budget. */
  if (!gone)
    mergeIdle();
  for (size_t i = 0; i != a->numManaged && !gone; ++i)
    check(cuMemFree(a->managed[i]), "cuMemFree");
  /* The copies of the constants of the bodies that the activation made. */
  releaseDeviceConstants(a, 0, gone);
  for (size_t i = 0; i != a->numHandles; ++i)
    free(a->handles[i]);
  free(a->managed);
  free(a->handles);
  free(a);
}

/* Memory of the device that an embedding program holds outside any
   activation, and copies to and from it on the stream of the kernels: the
   state of a simulation at the end of its last part, which a part that
   fails returns to (D196, D215). */
void *mdrtDeviceAllocateKept(uint64_t size) {
  enter();
  CUdeviceptr pointer = 0;
  if (size) {
    int64_t before = beginAccount();
    check(allocateDevice(&pointer, size), "cuMemAlloc");
    endAccount(ACCOUNT_KEPT, before, (int64_t)size);
  }
  return (void *)pointer;
}

void mdrtDeviceFreeKept(void *pointer) {
  if (!pointer || isDriverGone())
    return;
  finish();
  check(cuMemFree((CUdeviceptr)pointer), "cuMemFree");
}

/* Copies `size` bytes from the device to the device, after the work issued
   before it and before the work issued after it; the host does not wait. */
void mdrtDeviceCopyWithin(void *destination, const void *source,
                          uint64_t size) {
  if (!size)
    return;
  mgpuStreamCreate();
  mdrtSpansFlush();
  check(cuMemcpyAsync((CUdeviceptr)destination, (CUdeviceptr)source, size,
                      sharedStream),
        "cuMemcpyAsync");
  isPending = 1;
}

/* Copies `size` bytes from the device to the host once the work issued
   before it is done, and returns when they are there. */
void mdrtDeviceCopyToHost(void *destination, const void *source,
                          uint64_t size) {
  if (!size)
    return;
  if (copyAudit) copyAudit->metadataDownloads += size;
  mgpuStreamCreate();
  mdrtSpansFlush();
  check(cuMemcpyDtoHAsync(destination, (CUdeviceptr)source, size,
                          sharedStream),
        "cuMemcpyDtoHAsync");
  isPending = 1;
  finish();
}

/*===----------------------------------------------------------------------===
 * Copies of the outputs while the steps go on (D400, #113)
 *===----------------------------------------------------------------------===

  A staging slot holds the copy of a few buffers of the device (a frame: the
  positions and the numbers of the particles) for a writer of the host that
  reads them later, on a thread of its own. mdrtStagingCopy copies each
  buffer into a buffer of the device that the slot owns, on the stream of
  the kernels: after the work that produced it and before any later work
  overwrites it, a copy within the device of a few microseconds.
  mdrtStagingEnd then copies those buffers into pinned memory of the host on
  a stream of copies of its own, which runs beside the kernels of the next
  steps, records an event, and has the device write the slot's mark, the
  number of the end, after them. mdrtStagingWait, from any thread, returns
  at once for a slot whose mark is that of its last end and otherwise waits
  for the event; until it returns, and until the writer is done with the
  memory of the host, the slot is not given to another frame (the caller
  keeps a ring of slots).

  The mark is what tells a frame copied before a kernel failed from one
  that was not. Once a kernel has failed, the error is the context's, and
  cuEventQuery and cuEventSynchronize return it for every event, those
  that had completed before as well; and the driver still runs a function
  of the host that was enqueued behind copies that never ran. The device
  writes no mark after the failure. (Measured with the driver API 13.2, on
  an RTX 4090 with the driver 595.45.04 and on an RTX 5090 with 595.104.02:
  CUDA_ERROR_LAUNCH_FAILED for an event whose stream had been synchronized
  before the failing launch; a function of the host enqueued behind a copy
  that waited for the failed stream ran; a value of cuStreamWriteValue32
  there was not written. The probes queue the copy behind the failed
  stream; a copy already running on the copy engine at the trap was not
  measured.) */

#define MDRT_STAGING_PARTS 4

struct Staging {
  CUevent produced, copied;
  /* The ends of the slot, and the mark that the stream of copies writes
     after the copies of each end (cuStreamWriteValue32, into pinned memory
     of the host mapped for the device): the copies of the last end have
     arrived when the mark equals the number of the end. The device writes
     the mark as stream work, after the copies and a fence, so a mark never
     claims copies that did not run: once a kernel has failed, the device
     runs no more work, while the driver would still run a function of the
     host enqueued before (measured, see mdrtStagingEnd), and the query of
     the event returns the error of the context. A number, not a flag: the
     mark of an end may land after the wait for its event has returned and
     the slot has been staged again. */
  uint32_t ends;
  volatile uint32_t *mark;
  CUdeviceptr markDevice;
  struct {
    CUdeviceptr device;
    void *host;
    uint64_t room, size;
    int pending;
  } parts[MDRT_STAGING_PARTS];
};

static CUstream copyStream = NULL;

void *mdrtStagingCreate(void) {
  enter();
  struct Staging *slot = calloc(1, sizeof(struct Staging));
  if (!slot) {
    fprintf(stderr, "mdrt: out of memory of the host for a staging slot\n");
    exit(1);
  }
  if (!copyStream)
    check(cuStreamCreate(&copyStream, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate");
  check(cuEventCreate(&slot->produced, CU_EVENT_DISABLE_TIMING),
        "cuEventCreate");
  check(cuEventCreate(&slot->copied, CU_EVENT_DISABLE_TIMING),
        "cuEventCreate");
  check(cuMemHostAlloc((void **)&slot->mark, sizeof(uint32_t),
                       CU_MEMHOSTALLOC_DEVICEMAP),
        "cuMemHostAlloc");
  check(cuMemHostGetDevicePointer(&slot->markDevice, (void *)slot->mark, 0),
        "cuMemHostGetDevicePointer");
  *slot->mark = 0;
  return slot;
}

/* Copies `size` bytes of the buffer of the device at `source` into part
   `part` of the slot, on the stream of the kernels; returns the memory of
   the host where they will be once mdrtStagingWait returns. The caller
   knows that `source` is memory of the device (the program runs there);
   it may be memory that the library does not record (a kept buffer). */
void *mdrtStagingCopy(void *opaque, int32_t part, const void *source,
                      uint64_t size) {
  struct Staging *slot = opaque;
  if (part < 0 || part >= MDRT_STAGING_PARTS)
    return NULL;
  CUstream stream = mgpuStreamCreate();
  if (slot->parts[part].room < size) {
    if (slot->parts[part].device)
      mdrtDeviceFreeKept((void *)slot->parts[part].device);
    if (slot->parts[part].host)
      check(cuMemFreeHost(slot->parts[part].host), "cuMemFreeHost");
    slot->parts[part].device = (CUdeviceptr)mdrtDeviceAllocateKept(size);
    check(cuMemHostAlloc(&slot->parts[part].host, size, 0),
          "cuMemHostAlloc");
    slot->parts[part].room = size;
  }
  if (size) {
    mdrtSpansFlush();
    check(cuMemcpyAsync(slot->parts[part].device, (CUdeviceptr)source, size,
                        stream),
          "cuMemcpyAsync");
    isPending = 1;
  }
  slot->parts[part].size = size;
  slot->parts[part].pending = 1;
  return slot->parts[part].host;
}

/* MDRT_STAGING_MARK_DELAY=<ms>, for the tests: the stream of copies runs a
   function of the host that waits that long between the copies of an end
   and its mark, so that the mark lands after the wait for the end's event
   has returned and the slot has been staged again. */
static void CUDA_CB delayMark(void *opaque) {
  long delay = (long)(intptr_t)opaque;
  struct timespec pause = {delay / 1000, (delay % 1000) * 1000000L};
  nanosleep(&pause, NULL);
}

static long getMarkDelay(void) {
  static long delay = -1;
  if (delay < 0) {
    const char *value = getenv("MDRT_STAGING_MARK_DELAY");
    delay = value && *value ? strtol(value, NULL, 10) : 0;
  }
  return delay;
}

/* A kernel that fails (a trap) on the stream of the kernels, as a defect of
   a kernel would fail a run, for the tests. */
static void launchTrap(void) {
  static const char trap[] = ".version 7.5\n.target sm_70\n"
                             ".address_size 64\n"
                             ".visible .entry mdrtFault() { trap; }\n";
  CUmodule module;
  CUfunction function;
  check(cuModuleLoadData(&module, trap), "cuModuleLoadData");
  check(cuModuleGetFunction(&function, module, "mdrtFault"),
        "cuModuleGetFunction");
  mdrtSpansFlush();
  check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, mgpuStreamCreate(),
                       NULL, NULL),
        "cuLaunchKernel");
  isPending = 1;
}

/* The number of the end of a slot in the process, and whether the tests
   fail a kernel there: MDRT_FAULT_AFTER_FRAMES=N after the N-th end, once
   its copies and those before have arrived (`after`), and
   MDRT_FAULT_BEFORE_COPY=N at the N-th end, once the copies before it have
   arrived and before those of the end itself, which never run. */
static int isFaultedEnd(int64_t end, int after) {
  static int64_t afterEnd = -1, beforeEnd = -1;
  if (afterEnd < 0) {
    const char *value = getenv("MDRT_FAULT_AFTER_FRAMES");
    afterEnd = value && *value ? strtoll(value, NULL, 10) : 0;
    value = getenv("MDRT_FAULT_BEFORE_COPY");
    beforeEnd = value && *value ? strtoll(value, NULL, 10) : 0;
  }
  return end == (after ? afterEnd : beforeEnd);
}

/* Copies the parts copied into the slot since its last end to the host, on
   the stream of copies, once the stream of the kernels has copied them; then
   writes the slot's mark. The host does not wait. */
void mdrtStagingEnd(void *opaque) {
  struct Staging *slot = opaque;
  static int64_t ends = 0;
  ++ends;
  CUstream stream = mgpuStreamCreate();
  ++slot->ends;
  if (isFaultedEnd(ends, 0)) {
    check(cuStreamSynchronize(copyStream), "cuStreamSynchronize");
    launchTrap();
  }
  mdrtSpansFlush();
  check(cuEventRecord(slot->produced, stream), "cuEventRecord");
  check(cuStreamWaitEvent(copyStream, slot->produced, 0),
        "cuStreamWaitEvent");
  for (int i = 0; i != MDRT_STAGING_PARTS; ++i) {
    if (!slot->parts[i].pending)
      continue;
    slot->parts[i].pending = 0;
    if (!slot->parts[i].size)
      continue;
    if (copyAudit)
      copyAudit->downloads += slot->parts[i].size;
    check(cuMemcpyDtoHAsync(slot->parts[i].host, slot->parts[i].device,
                            slot->parts[i].size, copyStream),
          "cuMemcpyDtoHAsync");
  }
  check(cuEventRecord(slot->copied, copyStream), "cuEventRecord");
  long delay = getMarkDelay();
  if (delay > 0)
    check(cuLaunchHostFunc(copyStream, delayMark, (void *)(intptr_t)delay),
          "cuLaunchHostFunc");
  check(cuStreamWriteValue32(copyStream, slot->markDevice, slot->ends,
                             CU_STREAM_WRITE_VALUE_DEFAULT),
        "cuStreamWriteValue32");
  if (isFaultedEnd(ends, 1)) {
    check(cuStreamSynchronize(copyStream), "cuStreamSynchronize");
    launchTrap();
  }
}

/* Waits, on any thread, until the copies of the last end of the slot are in
   the memory of the host. Returns 0, or the error of the driver if a kernel
   failed before they arrived, with the call that returned it in `*call`;
   the caller hands both to mdrtStagingFail on the thread of the steps: this
   may be the thread of a writer, which does not end the process itself. */
int32_t mdrtStagingWait(void *opaque, const char **call) {
  struct Staging *slot = opaque;
  if (__atomic_load_n(slot->mark, __ATOMIC_ACQUIRE) == slot->ends)
    return 0;
  *call = "cuCtxSetCurrent";
  CUresult result = cuCtxSetCurrent(context);
  if (result == CUDA_SUCCESS) {
    *call = "cuEventSynchronize";
    result = cuEventSynchronize(slot->copied);
  }
  /* The mark may have landed while the event was waited for. */
  if (result != CUDA_SUCCESS &&
      __atomic_load_n(slot->mark, __ATOMIC_ACQUIRE) == slot->ends)
    return 0;
  return (int32_t)result;
}

/* Has the host run `hook` once before a failure of the driver ends the
   process (check): the writer of the frames, which waits with
   mdrtStagingWait and never calls check, writes there the frames whose
   copies had arrived. */
void mdrtSetFailureHook(void (*hook)(void)) { failureHook = hook; }

/* Reports the error `result` of the call `call` that mdrtStagingWait
   returned on the thread of a writer, as any failure of the driver is
   reported, and ends the process: from the thread of the steps, which the
   writer has told. */
void mdrtStagingFail(int32_t result, const char *call) {
  char what[96];
  snprintf(what, sizeof what, "%s of a staged frame", call);
  check((CUresult)result, what);
}

void mdrtStagingDestroy(void *opaque) {
  struct Staging *slot = opaque;
  if (!slot)
    return;
  if (!isDriverGone()) {
    for (int i = 0; i != MDRT_STAGING_PARTS; ++i) {
      if (slot->parts[i].device)
        mdrtDeviceFreeKept((void *)slot->parts[i].device);
      if (slot->parts[i].host)
        cuMemFreeHost(slot->parts[i].host);
    }
    cuMemFreeHost((void *)slot->mark);
    cuEventDestroy(slot->produced);
    cuEventDestroy(slot->copied);
  }
  free(slot);
}

/*===----------------------------------------------------------------------===
 * Work of a plugin of learned potentials (D270)
 *===----------------------------------------------------------------------===*/

/* The stream of the kernels, for a plugin that issues work there itself (a
   learned potential, docs/metatomic-ml1.md): that work runs in order with
   the kernels, and the library counts it as issued, so that the host waits
   for it before it reads what the device computed. */
void *mdrtDeviceExternalStream(void) {
  mgpuStreamCreate();
  mdrtSpansFlush();
  isPending = 1;
  return (void *)sharedStream;
}

/* Memory of the device that the library records as its own, outside any
   activation: a copy from or to it does not wait for the device, and an
   activation that closes does not take it back. A plugin's outputs, which
   live as long as the plugin's model. */
void *mdrtDeviceAllocateRecorded(uint64_t size) {
  enter();
  CUdeviceptr pointer = 0;
  if (!size)
    return NULL;
  int64_t before = beginAccount();
  check(allocateDevice(&pointer, size), "cuMemAlloc");
  endAccount(ACCOUNT_KEPT, before, (int64_t)size);
  recordLiveBlock((struct Block){.pointer = pointer,
                                 .size = size,
                                 .tag = "a plugin's outputs",
                                 .extent = size,
                                 .base = pointer,
                                 .baseSize = size});
  return (void *)pointer;
}

void mdrtDeviceFreeRecorded(void *pointer) {
  if (!pointer || isDriverGone())
    return;
  CUdeviceptr address = (CUdeviceptr)pointer;
  for (int i = numLive; i-- != 0;)
    if (liveBlocks[i].pointer == address) {
      liveBlocks[i] = liveBlocks[--numLive];
      break;
    }
  finish();
  check(cuMemFree(address), "cuMemFree");
}

/*===----------------------------------------------------------------------===
 * Views of the buffers of an embedding program (D220)
 *===----------------------------------------------------------------------===*/

/* The ordinal of the device of the context, among the devices that
   CUDA_VISIBLE_DEVICES leaves visible: the index under which a consumer of
   a view (the DLPack specification's device_id) names it. */
int32_t mdrtDeviceOrdinal(void) {
  enter();
  CUdevice device;
  check(cuCtxGetDevice(&device), "cuCtxGetDevice");
  return (int32_t)device;
}

/* Makes the stream of a consumer wait for the work issued on the stream of
   the kernels so far, without a wait of the host, as the DLPack Python
   specification asks of a producer (`__dlpack__(stream=...)`): 1 is the
   legacy default stream, 2 the per-thread default stream, any other value a
   CUstream of the context. */
void mdrtDeviceHandOff(uint64_t consumer) {
  static CUevent handoffEvent = NULL;
  mgpuStreamCreate();
  if (!handoffEvent)
    check(cuEventCreate(&handoffEvent, CU_EVENT_DISABLE_TIMING),
          "cuEventCreate");
  CUstream stream = consumer == 1   ? CU_STREAM_LEGACY
                    : consumer == 2 ? CU_STREAM_PER_THREAD
                                    : (CUstream)(uintptr_t)consumer;
  mdrtSpansFlush();
  check(cuEventRecord(handoffEvent, sharedStream), "cuEventRecord");
  check(cuStreamWaitEvent(stream, handoffEvent, 0), "cuStreamWaitEvent");
}

/* Waits for all the work issued in the context, on any stream: that of a
   consumer that read the buffers of a view, before they are written or
   freed. */
void mdrtDeviceWaitAll(void) {
  if (isDriverGone())
    return;
  mdrtSpansFlush();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  isPending = 0;
}

/*===----------------------------------------------------------------------===
  The structures of the tuples, kept from one activation to the next
  (D253, mdrt_tuples.h)
  ===----------------------------------------------------------------------===

  As in the runtime of the host (mdrt.c), with the incidence structure and
  the copy of the members on the device. What the store keeps on the device
  is memory of its own, outside the activations and the pool of blocks: an
  activation that ends leaves it, and it is freed when the store closes. The
  renumbered members are on the host; the identities are read from the
  device where the members are renumbered. */

#include "mdrt_tuples.h"

static struct MdrtTupleStore *currentTuples = NULL;

void *mdrtDeviceTuplesOpen(void) { return mdrtTupleStoreOpen(); }

void mdrtDeviceTuplesBegin(void *store) { mdrtTupleStoreBegin(store); }

void mdrtDeviceTuplesKeep(void *store, int32_t keeps) {
  ((struct MdrtTupleStore *)store)->keeps = keeps != 0;
}

void mdrtDeviceTuplesEnter(void *store) { currentTuples = store; }

void mdrtDeviceTuplesLeave(void) { currentTuples = NULL; }

static void freeBondedDirectories(struct MdrtTuples *entry, int gone);

void mdrtDeviceTuplesClose(void *opaque) {
  struct MdrtTupleStore *store = opaque;
  if (!store)
    return;
  if (currentTuples == store)
    currentTuples = NULL;
  if (begin() >= 0.0 && (store->deviceBytes[0] || store->deviceBytes[3]))
    fprintf(stderr,
            "mdrt: the store of the tuples kept on the device %lld bytes of "
            "incidence structures, %lld of renumbered members, %lld of the "
            "members and indexes of the tuple sets, and %lld of the order\n",
            (long long)store->deviceBytes[0], (long long)store->deviceBytes[1],
            (long long)store->deviceBytes[2], (long long)store->deviceBytes[3]);
  int gone = isDriverGone();
  /* No work may still read what is freed. */
  if (!gone)
    finish();
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    free(e->file);
    free(e->ids);
    free(e->members);
    if (!gone && e->deviceIncidence)
      check(cuMemFree((CUdeviceptr)e->deviceIncidence), "cuMemFree");
    if (!gone && e->deviceMembers)
      check(cuMemFree((CUdeviceptr)e->deviceMembers), "cuMemFree");
    freeBondedDirectories(e, gone);
    if (!gone && e->deviceRenumbered)
      check(cuMemFree((CUdeviceptr)e->deviceRenumbered), "cuMemFree");
    if (!gone && e->ownsSet && e->deviceFile)
      check(cuMemFree((CUdeviceptr)e->deviceFile), "cuMemFree");
    free(e);
  }
  if (!gone) {
    void *own[3] = {store->deviceIds, store->deviceFlag, store->devicePlace};
    for (int k = 0; k != 3; ++k)
      if (own[k])
        check(cuMemFree((CUdeviceptr)own[k]), "cuMemFree");
  }
  free(store->staging);
  free(store->entries);
  free(store);
}

void mdrtDeviceTuplesCounts(void *opaque, int64_t *counts) {
  struct MdrtTupleStore *store = opaque;
  counts[0] = store ? store->kept : 0;
  counts[1] = store ? store->built : 0;
}

/* Copies between the host and the device, done when they return. */
static void tuplesToDevice(void *device, const void *host, size_t bytes) {
  if (!bytes)
    return;
  if (copyAudit) copyAudit->metadataUploads += bytes;
  mgpuStreamCreate();
  double start = begin();
  mdrtSpansFlush();
  check(cuMemcpyHtoDAsync((CUdeviceptr)device, host, bytes, sharedStream),
        "cuMemcpyHtoDAsync");
  isPending = 1;
  end(COPY, start);
  finish();
}

static void tuplesToHost(void *host, const void *device, size_t bytes) {
  if (!bytes)
    return;
  if (copyAudit) copyAudit->metadataDownloads += bytes;
  mgpuStreamCreate();
  double start = begin();
  mdrtSpansFlush();
  check(cuMemcpyDtoHAsync(host, (CUdeviceptr)device, bytes, sharedStream),
        "cuMemcpyDtoHAsync");
  isPending = 1;
  end(COPY, start);
  finish();
}

/* Memory of the device that the store keeps. */
static void *tuplesDeviceKept(size_t bytes) {
  CUdeviceptr pointer = 0;
  if (bytes) {
    int64_t before = beginAccount();
    check(allocateDevice(&pointer, bytes), "cuMemAlloc");
    endAccount(ACCOUNT_TUPLES, before, (int64_t)bytes);
  }
  return (void *)pointer;
}

/* Builds the incidence structure of `members` at `device`, in ragged rows
   (mdrtTuplesRange): the structure is written on the host a range of
   elements at a time, in memory of a few megabytes (16 MiB at most), and
   each range is uploaded before the next is written. The host never holds
   the whole structure, whose memory would be mapped and faulted in at
   every build (#294). */
static void tuplesBuildOnDevice(void *device, const int32_t *members,
                                int64_t tuples, int64_t arity,
                                int64_t particles) {
  /* The memory of the ranges is the store's, taken once and cleared once:
     taking and clearing it for every structure cost more than the build of
     the structures of JAC (0.5 ms with 4 MiB, 1.2 ms with 16). */
  enum { STAGING_BYTES = 16 << 20 };
  struct MdrtTupleStore *store = currentTuples;
  struct MdrtTupleIndex index;
  mdrtTupleIndexBuild(&index, members, tuples, arity, particles);
  int64_t elements = mdrtTuplesSize(tuples, arity, particles);
  int64_t range = (int64_t)(STAGING_BYTES / sizeof(int32_t));
  size_t needed = (size_t)range * sizeof(int32_t);
  int32_t *staging = store ? store->staging : NULL;
  if (!staging || (store && store->stagingBytes < needed)) {
    if (store)
      free(store->staging);
    staging = calloc(1, needed);
    if (!staging) {
      fprintf(stderr, "mdrt: out of memory of the host\n");
      abort();
    }
    if (store) {
      store->staging = staging;
      store->stagingBytes = needed;
    }
  }
  for (int64_t begin = 0; begin < elements; begin += range) {
    int64_t end = begin + range < elements ? begin + range : elements;
    mdrtTuplesRange(&index, members, tuples, arity, particles, begin, end,
                    staging);
    tuplesToDevice((int32_t *)device + begin, staging,
                   (size_t)(end - begin) * sizeof(int32_t));
  }
  if (!store)
    free(staging);
  mdrtTupleIndexFree(&index);
}

double mdrtDeviceTuplesSeconds(void *opaque) {
  struct MdrtTupleStore *store = opaque;
  return store ? store->seconds : 0.0;
}

static void tuplesRenumber(struct MdrtMemref2 *result,
                           struct MdrtMemref2 *members,
                           struct MdrtMemref1 *ids) {
  enter();
  const int32_t *file = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  int64_t particles = ids->size;
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  int32_t *id = mdrtTuplesAllocate((size_t)particles * sizeof(int32_t));
  tuplesToHost(id, (const int32_t *)ids->aligned + ids->offset,
               (size_t)particles * sizeof(int32_t));
  if (!currentTuples) {
    int32_t *fresh = mdrtTuplesAllocate(bytes);
    mdrtTuplesRenumber(file, tuples, arity, id, particles, fresh);
    mdrtTuplesSetMemref2(result, fresh, tuples, arity);
    free(id);
    return;
  }
  int found = 0;
  struct MdrtTuples *e = mdrtTupleStoreTake(currentTuples, file, tuples, arity,
                                            id, particles, &found);
  if (!found) {
    mdrtTuplesRenumber(file, tuples, arity, id, particles, e->members);
  } else if (mdrtTuplesChecks()) {
    int32_t *anew = mdrtTuplesAllocate(bytes);
    mdrtTuplesRenumber(file, tuples, arity, id, particles, anew);
    if (memcmp(anew, e->members, bytes) != 0)
      mdrtTuplesDiffer("the renumbered members of a tuple set");
    free(anew);
  }
  free(id);
  ++e->held;
  mdrtTuplesSetMemref2(result, e->members, tuples, arity);
}

void _mlir_ciface_mdrtDeviceTuplesRenumber(struct MdrtMemref2 *result,
                                           struct MdrtMemref2 *members,
                                           struct MdrtMemref1 *ids) {
  double start = mdrtTuplesNow();
  tuplesRenumber(result, members, ids);
  mdrtTuplesSpent(currentTuples, start);
}

static void tuplesIncidence(struct MdrtMemref1 *result,
                            struct MdrtMemref2 *members, int64_t particles) {
  enter();
  const int32_t *given = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  struct MdrtTuples *e = mdrtTupleStoreFind(currentTuples, given);
  if (!e && currentTuples) {
    /* The members of the files, in their order. */
    int found = 0;
    e = mdrtTupleStoreTake(currentTuples, given, tuples, arity, NULL,
                           particles, &found);
    e->given = given;
  }
  if (e && e->particles != particles)
    e = NULL;
  /* The size does not depend on the order. */
  int64_t elements = mdrtTuplesSize(tuples, arity, particles);
  size_t bytes = (size_t)elements * sizeof(int32_t);
  if (!e) {
    allocationTag = "tuples: incidence";
    void *device = mgpuMemAlloc(bytes, NULL, false);
    allocationTag = NULL;
    tuplesBuildOnDevice(device, given, tuples, arity, particles);
    mdrtTuplesSetMemref1(result, device, elements);
    return;
  }
  if (!e->hasIncidence) {
    if (!e->deviceIncidence || e->elements != elements) {
      if (e->deviceIncidence) {
        finish();
        check(cuMemFree((CUdeviceptr)e->deviceIncidence), "cuMemFree");
      }
      e->deviceIncidence = tuplesDeviceKept(bytes);
    }
    e->elements = elements;
    tuplesBuildOnDevice(e->deviceIncidence, given, tuples, arity, particles);
    e->hasIncidence = 1;
  } else if (mdrtTuplesChecks()) {
    int32_t *anew = mdrtTuplesAllocate(bytes);
    mdrtTuplesFill(given, tuples, arity, particles, anew);
    int32_t *kept = mdrtTuplesAllocate(bytes);
    tuplesToHost(kept, e->deviceIncidence, bytes);
    if (memcmp(anew, kept, bytes) != 0)
      mdrtTuplesDiffer("the incidence structure of a tuple set");
    free(kept);
    free(anew);
  }
  ++e->held;
  mdrtTuplesSetMemref1(result, e->deviceIncidence, elements);
}

void _mlir_ciface_mdrtDeviceTuplesIncidence(struct MdrtMemref1 *result,
                                            struct MdrtMemref2 *members,
                                            int64_t particles) {
  double start = mdrtTuplesNow();
  tuplesIncidence(result, members, particles);
  mdrtTuplesSpent(currentTuples, start);
}

/* The members on the device, for the loops that evaluate each tuple once:
   those that the incidence structure was built from. */
static void tuplesMembers(struct MdrtMemref2 *result,
                          struct MdrtMemref2 *members) {
  enter();
  const int32_t *given = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  struct MdrtTuples *e = mdrtTupleStoreFind(currentTuples, given);
  if (!e) {
    allocationTag = "tuples: members";
    void *device = mgpuMemAlloc(bytes, NULL, false);
    allocationTag = NULL;
    tuplesToDevice(device, given, bytes);
    mdrtTuplesSetMemref2(result, device, tuples, arity);
    return;
  }
  if (!e->hasDeviceMembers) {
    if (!e->deviceMembers)
      e->deviceMembers = tuplesDeviceKept(bytes);
    tuplesToDevice(e->deviceMembers, given, bytes);
    e->hasDeviceMembers = 1;
  } else if (mdrtTuplesChecks()) {
    int32_t *kept = mdrtTuplesAllocate(bytes);
    tuplesToHost(kept, e->deviceMembers, bytes);
    if (memcmp(kept, given, bytes) != 0)
      mdrtTuplesDiffer("the copy on the device of the members of a tuple "
                       "set");
    free(kept);
  }
  ++e->held;
  mdrtTuplesSetMemref2(result, e->deviceMembers, tuples, arity);
}

void _mlir_ciface_mdrtDeviceTuplesMembers(struct MdrtMemref2 *result,
                                          struct MdrtMemref2 *members) {
  double start = mdrtTuplesNow();
  tuplesMembers(result, members);
  mdrtTuplesSpent(currentTuples, start);
}

/* Where the code would free what it took: the renumbered members, on the
   host, or a structure on the device. */
static void tuplesRelease(void *pointer) {
  if (!pointer || mdrtTupleStoreGiveBack(currentTuples, pointer))
    return;
  if (isOnDevice(pointer))
    mgpuMemFree(pointer, NULL);
  else
    free(pointer);
}

void _mlir_ciface_mdrtDeviceTuplesRelease(struct MdrtMemref2 *buffer) {
  tuplesRelease(buffer->allocated);
}

/* The same for an incidence structure, a buffer of ragged rows. */
void _mlir_ciface_mdrtDeviceTuplesReleaseRows(struct MdrtMemref1 *buffer) {
  tuplesRelease(buffer->allocated);
}

/*===----------------------------------------------------------------------===
  The structures of the tuples built on the device (D257, #295)
  ===----------------------------------------------------------------------===

  The code compares the identities that it holds with the store's copy of
  the last order (a kernel that writes a flag, which mdrtDeviceTuplesOrderEnd
  reads), takes an entry for each tuple set (mdrtDeviceTuplesTake), and,
  where the entry is to be built, launches the kernels of the inverse of
  the order, the renumbered members, and the incidence structure in ragged
  rows from them (D314: a count of the memberships of each
  place, a scan, a scatter, a sort of each row, and a thread for each
  entry; MDExecToGPU.cpp, buildOnDevice). The host uploads the members in
  the order of the files once for each tuple set. The entry
  becomes valid when the code says that it is built
  (mdrtDeviceTuplesBuilt), never before: an activation that ends before
  leaves it to be built again. A failed launch ends the process (check). */

static struct MdrtTupleStore *deviceTupleStore(void) {
  if (!currentTuples) {
    fprintf(stderr, "mdrt: the structures of the tuples are built on the "
                    "device without a store; this is a defect of mdir\n");
    abort();
  }
  return currentTuples;
}

static void setMemref1(struct MdrtMemref1 *result, void *memory,
                       int64_t size) {
  result->allocated = memory;
  result->aligned = memory;
  result->offset = 0;
  result->size = size;
  result->stride = 1;
}

/* The store's copy of the identities of the last order, with the flag of
   the comparison cleared; a copy that is new holds no identity (-1), so
   that the comparison finds a new order. */
void _mlir_ciface_mdrtDeviceTuplesOrderKept(struct MdrtMemref1 *result,
                                            struct MdrtMemref1 *ids) {
  enter();
  double start = mdrtTuplesNow();
  struct MdrtTupleStore *store = deviceTupleStore();
  int64_t n = ids->size;
  mgpuStreamCreate();
  if (store->deviceIdsCount != n) {
    if (store->deviceIds) {
      finish();
      check(cuMemFree((CUdeviceptr)store->deviceIds), "cuMemFree");
    }
    store->deviceIds = tuplesDeviceKept((size_t)n * sizeof(int32_t));
    store->deviceIdsCount = n;
    store->deviceBytes[3] += n * (int64_t)sizeof(int32_t) + 4;
    mdrtSpansFlush();
    if (n)
      check(cuMemsetD32Async((CUdeviceptr)store->deviceIds, 0xFFFFFFFFu,
                             (size_t)n, sharedStream),
            "cuMemsetD32Async");
    ++store->order;
  }
  if (!store->deviceFlag)
    store->deviceFlag = tuplesDeviceKept(sizeof(int32_t));
  mdrtSpansFlush();
  check(cuMemsetD32Async((CUdeviceptr)store->deviceFlag, 0, 1, sharedStream),
        "cuMemsetD32Async");
  isPending = 1;
  setMemref1(result, store->deviceIds, n);
  mdrtTuplesSpent(store, start);
}

void _mlir_ciface_mdrtDeviceTuplesOrderFlag(struct MdrtMemref1 *result) {
  setMemref1(result, deviceTupleStore()->deviceFlag, 1);
}

/* The flag is read (4 bytes); where an identity differed, the identities
   are copied on the device and the number of the order advances. */
void _mlir_ciface_mdrtDeviceTuplesOrderEnd(struct MdrtMemref1 *ids) {
  enter();
  double start = mdrtTuplesNow();
  struct MdrtTupleStore *store = deviceTupleStore();
  int32_t differs = 0;
  tuplesToHost(&differs, store->deviceFlag, sizeof(int32_t));
  if (differs) {
    mdrtSpansFlush();
    if (ids->size)
      check(cuMemcpyDtoDAsync(
                (CUdeviceptr)store->deviceIds,
                (CUdeviceptr)((const int32_t *)ids->aligned + ids->offset),
                (size_t)ids->size * sizeof(int32_t), sharedStream),
            "cuMemcpyDtoDAsync");
    isPending = 1;
    ++store->order;
  }
  mdrtTuplesSpent(store, start);
}

/* Compares a structure built or kept on the device with the one that the
   host builds from the same inputs (MDRT_CHECK_KEPT_TUPLES). */
static void tuplesCheckOnDevice(struct MdrtTupleStore *store,
                                struct MdrtTuples *e) {
  int64_t tuples = e->tuples, arity = e->arity, particles = e->particles;
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  int32_t *renumbered = mdrtTuplesAllocate(bytes);
  if (e->ordered) {
    int32_t *ids = mdrtTuplesAllocate((size_t)particles * sizeof(int32_t));
    tuplesToHost(ids, store->deviceIds, (size_t)particles * sizeof(int32_t));
    mdrtTuplesRenumber(e->file, tuples, arity, ids, particles, renumbered);
    free(ids);
    int32_t *kept = mdrtTuplesAllocate(bytes);
    tuplesToHost(kept, e->deviceRenumbered, bytes);
    if (memcmp(kept, renumbered, bytes) != 0)
      mdrtTuplesDiffer("the renumbered members of a tuple set on the device");
    free(kept);
  } else {
    memcpy(renumbered, e->file, bytes);
  }
  /* The build on the device sorts each row (emitDeviceRaggedRows), so its
     structure is the host's element for element. */
  int64_t elements = mdrtTuplesSize(tuples, arity, particles);
  size_t structure = (size_t)elements * sizeof(int32_t);
  int32_t *anew = mdrtTuplesAllocate(structure);
  mdrtTuplesFill(renumbered, tuples, arity, particles, anew);
  int32_t *kept = mdrtTuplesAllocate(structure);
  if (elements == e->elements)
    tuplesToHost(kept, e->deviceIncidence, structure);
  if (elements != e->elements || memcmp(anew, kept, structure) != 0)
    mdrtTuplesDiffer("the incidence structure of a tuple set on the device");
  free(kept);
  free(anew);
  free(renumbered);
}

static int sameSet(const struct MdrtTuples *e, const int32_t *file,
                   int64_t tuples, int64_t arity, int64_t particles) {
  return e->device && e->tuples == tuples && e->arity == arity &&
         e->particles == particles &&
         memcmp(e->file, file, (size_t)(tuples * arity) * sizeof(int32_t)) ==
             0;
}

/* The members in the order of the files, on the device, for a tuple set
   that has none yet. */
static void uploadSet(struct MdrtTuples *e) {
  size_t bytes = (size_t)(e->tuples * e->arity) * sizeof(int32_t);
  e->deviceFile = tuplesDeviceKept(bytes);
  tuplesToDevice(e->deviceFile, e->file, bytes);
  e->ownsSet = 1;
}

/* An entry for the members `members` of `particles` particles, in the
   current order (`ordered`) or in the order of the files: one taken as it
   is, or one to build, which mdrtDeviceTuplesBuild then says. */
int64_t _mlir_ciface_mdrtDeviceTuplesTake(struct MdrtMemref2 *members,
                                          int64_t particles, int32_t ordered) {
  enter();
  double start = mdrtTuplesNow();
  struct MdrtTupleStore *store = deviceTupleStore();
  const int32_t *file = mdrtTuplesMembersOf(members);
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    if (!sameSet(e, file, tuples, arity, particles) ||
        e->ordered != (ordered != 0) || !e->deviceValid ||
        (ordered && e->order != store->order) ||
        (!store->keeps && e->held == 0))
      continue;
    ++store->kept;
    e->building = 0;
    ++e->held;
    if (mdrtTuplesChecks())
      tuplesCheckOnDevice(store, e);
    mdrtTuplesSpent(store, start);
    return (int64_t)(intptr_t)e;
  }
  ++store->built;
  struct MdrtTuples *e = NULL;
  for (size_t i = 0; i != store->count && !e; ++i) {
    struct MdrtTuples *other = store->entries[i];
    if (sameSet(other, file, tuples, arity, particles) &&
        other->ordered == (ordered != 0) && other->held == 0)
      e = other;
  }
  if (!e) {
    e = calloc(1, sizeof(struct MdrtTuples));
    if (!e) {
      fprintf(stderr, "mdrt: out of memory of the host\n");
      abort();
    }
    e->device = 1;
    e->ordered = ordered != 0;
    e->tuples = tuples;
    e->arity = arity;
    e->particles = particles;
    e->file = mdrtTuplesAllocate(bytes);
    memcpy(e->file, file, bytes);
    struct MdrtTuples *owner = NULL;
    for (size_t i = 0; i != store->count && !owner; ++i)
      if (sameSet(store->entries[i], file, tuples, arity, particles) &&
          store->entries[i]->ownsSet)
        owner = store->entries[i];
    if (owner) {
      e->deviceFile = owner->deviceFile;
    } else {
      uploadSet(e);
      store->deviceBytes[2] += (int64_t)bytes;
    }
    e->elements = mdrtTuplesSize(tuples, arity, particles);
    int64_t structure = e->elements * (int64_t)sizeof(int32_t);
    e->deviceIncidence = tuplesDeviceKept((size_t)structure);
    store->deviceBytes[0] += structure;
    if (isMemoryDetailed())
      fprintf(stderr,
              "mdrt: device memory: the incidence structure of %lld tuples "
              "of %lld: %lld particles, ragged rows of %lld entries of %lld "
              "values, %lld bytes\n",
              (long long)tuples, (long long)arity, (long long)particles,
              (long long)(tuples * arity), (long long)(arity + 2),
              (long long)structure);
    if (e->ordered) {
      e->deviceRenumbered = tuplesDeviceKept(bytes);
      store->deviceBytes[1] += (int64_t)bytes;
    }
    if (store->count == store->room) {
      store->room = store->room ? 2 * store->room : 8;
      store->entries =
          realloc(store->entries, store->room * sizeof(*store->entries));
      if (!store->entries) {
        fprintf(stderr, "mdrt: out of memory of the host\n");
        abort();
      }
    }
    store->entries[store->count++] = e;
  }
  if (ordered && store->devicePlaceCount < particles) {
    if (store->devicePlace) {
      finish();
      check(cuMemFree((CUdeviceptr)store->devicePlace), "cuMemFree");
    }
    store->devicePlace = tuplesDeviceKept((size_t)particles * sizeof(int32_t));
    store->devicePlaceCount = particles;
    store->deviceBytes[3] += particles * (int64_t)sizeof(int32_t);
  }
  e->deviceValid = 0;
  e->building = 1;
  e->order = store->order;
  ++e->held;
  mdrtTuplesSpent(store, start);
  return (int64_t)(intptr_t)e;
}

#include "mdrt_bonded_units.h"
#include <math.h>

int32_t mdrtDeviceTuplesBuild(int64_t handle) {
  return ((struct MdrtTuples *)(intptr_t)handle)->building;
}

void _mlir_ciface_mdrtDeviceTuplesFile(struct MdrtMemref2 *result,
                                       int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  mdrtTuplesSetMemref2(result, e->deviceFile, e->tuples, e->arity);
}

void _mlir_ciface_mdrtDeviceTuplesRenumbered(struct MdrtMemref2 *result,
                                             int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  mdrtTuplesSetMemref2(result,
                       e->ordered ? e->deviceRenumbered : e->deviceFile,
                       e->tuples, e->arity);
}

/* The members that the structure is built from (those of
   mdrtDeviceTuplesRenumbered), as one row: the particle of each
   membership. */
void _mlir_ciface_mdrtDeviceTuplesMemberships(struct MdrtMemref1 *result,
                                              int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  setMemref1(result, e->ordered ? e->deviceRenumbered : e->deviceFile,
             e->tuples * e->arity);
}

void _mlir_ciface_mdrtDeviceTuplesPlace(struct MdrtMemref1 *result,
                                        int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  struct MdrtTupleStore *store = deviceTupleStore();
  setMemref1(result, e->ordered ? store->devicePlace : NULL,
             e->ordered ? e->particles : 0);
}

void _mlir_ciface_mdrtDeviceTuplesStructure(struct MdrtMemref1 *result,
                                            int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  setMemref1(result, e->deviceIncidence, e->elements);
}

/* The code has launched the kernels of the build, or taken the entry: it
   is valid from here on. */
void mdrtDeviceTuplesBuilt(int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  int built = e->building;
  e->building = 0;
  e->deviceValid = 1;
  if (built && mdrtTuplesChecks()) {
    enter();
    tuplesCheckOnDevice(deviceTupleStore(), e);
  }
}

/* Where the code would free what it took. */
void mdrtDeviceTuplesDone(int64_t handle) {
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)handle;
  if (e->held > 0)
    --e->held;
}

/*===----------------------------------------------------------------------===
  Matrix products of the bodies of neural stages (D302)
  ===----------------------------------------------------------------------===*/

#include <cublas_v2.h>

/* One handle, on the stream of the kernels, in the order of their launches.
   The default math mode: f32 products are computed in f32, without TF32. */
static cublasHandle_t blasHandle = NULL;

static int checkBlas(cublasStatus_t status, const char *what) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  char message[160];
  snprintf(message, sizeof message,
           "%s failed with the cuBLAS status %d; the run has failed", what,
           (int)status);
  stopWith(message);
  return 0;
}

/* C = A B + beta C in f32, for row-major matrices: C is m x n with rows ldc
   apart, A is m x k (k x m with transA) with rows lda apart, and B is k x n
   (n x k with transB) with rows ldb apart; the pointers are the device's.
   With batch > 1, `batch` such products, the matrices of each the previous
   ones' moved by strideA, strideB, and strideC elements: cuBLAS's
   strided-batched GEMM. cuBLAS takes column-major matrices, as which a
   row-major one is its transpose: C^T = B^T A^T. */
void mdrtGemmF32(int64_t m, int64_t n, int64_t k, int64_t a, int64_t lda,
                 int64_t strideA, int32_t transA, int64_t b, int64_t ldb,
                 int64_t strideB, int32_t transB, int64_t c, int64_t ldc,
                 int64_t strideC, float beta, int64_t batch) {
  enter();
  double start = begin();
  if (!blasHandle) {
    int64_t before = beginAccount();
    if (!checkBlas(cublasCreate(&blasHandle), "cublasCreate"))
      return;
    endAccount(ACCOUNT_BLAS, before, 0);
    if (!checkBlas(cublasSetMathMode(blasHandle, CUBLAS_DEFAULT_MATH),
                   "cublasSetMathMode"))
      return;
  }
  if (!checkBlas(cublasSetStream(blasHandle, (cudaStream_t)mgpuStreamCreate()),
                 "cublasSetStream"))
    return;
  float one = 1.0f;
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  mdrtSpansFlush();
  if (batch == 1)
    checkBlas(cublasSgemm(blasHandle, opB, opA, (int)n, (int)m, (int)k, &one,
                          (const float *)(intptr_t)b, (int)ldb,
                          (const float *)(intptr_t)a, (int)lda, &beta,
                          (float *)(intptr_t)c, (int)ldc),
              "cublasSgemm");
  else
    checkBlas(cublasSgemmStridedBatched(
                  blasHandle, opB, opA, (int)n, (int)m, (int)k, &one,
                  (const float *)(intptr_t)b, (int)ldb, (long long)strideB,
                  (const float *)(intptr_t)a, (int)lda, (long long)strideA,
                  &beta, (float *)(intptr_t)c, (int)ldc, (long long)strideC,
                  (int)batch),
              "cublasSgemmStridedBatched");
  isPending = 1;
  end(LAUNCH, start);
}

/* The constants that the products read (the weights of the bodies), copied
   to the device when first read, and found again by the address and size
   of their host copy and by their owner: the activation of the embedding
   program that made the copy (D215), or none outside an activation. An
   address of the host identifies a constant only while its program is
   loaded: after the program is released, another one may put its own
   constants there. So the copies go with their owner: an activation's when
   it closes, those made outside one when a program unloads its modules
   (mgpuModuleUnload), and all of them with the context
   (mgpuSetDefaultDevice). */
struct DeviceConstant {
  void *owner;
  int64_t host, bytes;
  CUdeviceptr device;
};
static struct DeviceConstant *deviceConstants = NULL;
static int64_t numDeviceConstants = 0, deviceConstantsCapacity = 0;

static void releaseDeviceConstants(void *owner, int all, int gone) {
  for (int64_t i = numDeviceConstants; i-- != 0;) {
    if (!all && deviceConstants[i].owner != owner)
      continue;
    if (!gone)
      check(cuMemFree(deviceConstants[i].device), "cuMemFree");
    deviceConstants[i] = deviceConstants[--numDeviceConstants];
  }
}

static void releaseBlas(int gone) {
  if (blasHandle && !gone)
    cublasDestroy(blasHandle);
  blasHandle = NULL;
}

int64_t mdrtDeviceConstant(int64_t host, int64_t bytes) {
  enter();
  void *owner = currentActivation;
  for (int64_t i = 0; i != numDeviceConstants; ++i)
    if (deviceConstants[i].owner == owner && deviceConstants[i].host == host &&
        deviceConstants[i].bytes == bytes)
      return (int64_t)deviceConstants[i].device;
  if (numDeviceConstants == deviceConstantsCapacity) {
    int64_t capacity =
        deviceConstantsCapacity ? 2 * deviceConstantsCapacity : 64;
    struct DeviceConstant *grown = realloc(
        deviceConstants, (size_t)capacity * sizeof(struct DeviceConstant));
    if (!grown) {
      stopWith("no memory of the host for the constants of a body's "
               "products; the run has failed");
      return 0;
    }
    deviceConstants = grown;
    deviceConstantsCapacity = capacity;
  }
  CUdeviceptr device = 0;
  int64_t before = beginAccount();
  check(allocateDevice(&device, (size_t)bytes), "cuMemAlloc");
  endAccount(ACCOUNT_CONSTANTS, before, bytes);
  if (copyAudit) copyAudit->metadataUploads += bytes;
  mdrtSpansFlush();
  check(cuMemcpyHtoD(device, (const void *)(intptr_t)host, (size_t)bytes),
        "cuMemcpyHtoD");
  /* From pageable memory the copy may still be under way when it returns,
     and the stream of the kernels does not wait for the null stream: the
     products that read the constant would race with its copy. */
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  struct DeviceConstant *entry = &deviceConstants[numDeviceConstants++];
  entry->owner = owner;
  entry->host = host;
  entry->bytes = bytes;
  entry->device = device;
  return (int64_t)device;
}

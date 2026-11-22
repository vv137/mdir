/* The driver builds this force-only transform policy on the first
   reciprocal evaluation. The supplied metadata is rewritten by the PME
   table kernel. A cached plan reads it only during its own execution. */
#ifdef MDIR_PME_LTO
#include "pme_callback_code.h"
#include <cufftXt.h>
#endif

enum { PME_MODES = 4, PME_ROUNDS = 4 };
struct PMEPolicy {
  int k[3], single, tilted, device, version, available, winner;
  struct PMEPolicy *next;
};
struct PMETrial {
  struct PMEPolicy *policy;
  CUstream stream;
  CUevent start, stop, lastUse;
  CUdeviceptr metadata;
  cufftHandle callback[2];
  size_t workspace;
  int modes[PME_MODES], count, measured, pinned;
  float samples[PME_MODES][PME_ROUNDS];
};
static struct PMETrial pmeTrials[2];
static struct PMEPolicy *pmePolicies;

static void releasePMETrial(struct PMETrial *s) {
  if (!s->metadata || s->pinned)
    return;
  if (s->lastUse) {
    check(cuEventSynchronize(s->lastUse), "cuEventSynchronize PME");
    cuEventDestroy(s->lastUse);
  }
  for (int i = 0; i != 2; ++i)
    if (s->callback[i])
      cufftDestroy(s->callback[i]);
  if (s->start)
    cuEventDestroy(s->start);
  if (s->stop)
    cuEventDestroy(s->stop);
  fftPlanBytes -= s->workspace;
  memset(s, 0, sizeof *s);
}

static void trimPMETrials(void) {
  for (int i = 0; i < 2; ++i)
    releasePMETrial(&pmeTrials[i]);
  struct PMEPolicy **link = &pmePolicies;
  while (*link) {
    struct PMEPolicy *policy = *link;
    if (policy == pmeTrials[0].policy || policy == pmeTrials[1].policy) {
      link = &policy->next;
    } else {
      *link = policy->next;
      free(policy);
    }
  }
}

#ifdef MDIR_PME_LTO
static int makePMECallback(struct PMETrial *s, int dir, int single, int tilted,
                           int k1, int k2, int k3) {
  if (dir && (uint64_t)k1 * k2 > UINT32_MAX / (uint64_t)(k3 / 2 + 1))
    return 0;
  cufftHandle h;
  if (cufftCreate(&h) != CUFFT_SUCCESS)
    return 0;
  const char *symbol = dir ? (single ? "pmeScaleF32" : "pmeScaleF64")
                           : (single ? "pmeLoadF32" : "pmeLoadF64");
  if (dir && tilted)
    symbol = single ? "pmeScaleF32Tilted" : "pmeScaleF64Tilted";
  cufftXtCallbackType cb =
      dir ? (single ? CUFFT_CB_LD_COMPLEX : CUFFT_CB_LD_COMPLEX_DOUBLE)
          : (single ? CUFFT_CB_LD_REAL : CUFFT_CB_LD_REAL_DOUBLE);
  cufftType type =
      dir ? (single ? CUFFT_C2R : CUFFT_Z2D) : (single ? CUFFT_R2C : CUFFT_D2Z);
  void *info = (void *)(uintptr_t)s->metadata;
  cufftResult result = cufftXtSetJITCallback(h, symbol, pmeCallbackCode,
                                             sizeof pmeCallbackCode, cb, &info);
  size_t bytes = 0;
  if (result == CUFFT_SUCCESS)
    result = cufftMakePlan3d(h, k1, k2, k3, type, &bytes);
  if (result == CUFFT_SUCCESS)
    result = cufftSetStream(h, (cudaStream_t)s->stream);
  if (result != CUFFT_SUCCESS || bytes > getFFTPlanBudget() ||
      fftPlanBytes > getFFTPlanBudget() - bytes) {
    if (getenv("MDRT_PME_REPORT"))
      fprintf(stderr,
              "mdrt: PME callback %s unavailable (cuFFT %d, workspace %zu)\n",
              symbol, result, bytes);
    cufftDestroy(h);
    return 0;
  }
  s->workspace += bytes;
  fftPlanBytes += bytes;
  s->callback[dir] = h;
  return 1;
}
#endif

static struct PMEPolicy *findPMEPolicy(int k1, int k2, int k3, int single,
                                       int tilted, int version, int available) {
  for (struct PMEPolicy *p = pmePolicies; p; p = p->next)
    if (p->k[0] == k1 && p->k[1] == k2 && p->k[2] == k3 &&
        p->single == single && p->tilted == tilted &&
        p->device == defaultDevice && p->version == version &&
        p->available == available)
      return p;
  struct PMEPolicy *p = calloc(1, sizeof *p);
  if (!p)
    return NULL;
  p->k[0] = k1;
  p->k[1] = k2;
  p->k[2] = k3;
  p->single = single;
  p->tilted = tilted;
  p->device = defaultDevice;
  p->version = version;
  p->available = available;
  p->winner = available ? -1 : 0;
  p->next = pmePolicies;
  return pmePolicies = p;
}

intptr_t mdrtCudaPMEPrepare(int64_t k1, int64_t k2, int64_t k3, int64_t single,
                            int64_t tilted, uintptr_t metadata) {
#ifndef MDIR_PME_LTO
  return 0;
#else
  CUstream stream = mgpuStreamCreate();
  struct PMETrial *s = &pmeTrials[onSide != 0];
  struct PMEPolicy *p = s->policy;
  if (!p || s->stream != stream || s->metadata != metadata || p->k[0] != k1 ||
      p->k[1] != k2 || p->k[2] != k3 || p->single != single ||
      p->tilted != tilted) {
    releasePMETrial(s);
    /* Make room for the two ordinary plans first. Under a tight workspace
       budget the explicit implementation remains available. */
    cufftHandle plain;
    if (!getFFTPlan(k1, k2, k3, 1, single, &plain) ||
        !getFFTPlan(k1, k2, k3, 0, single, &plain))
      return 0;
    s->metadata = metadata;
    s->stream = stream;
    int available = 0, version = 0;
    cufftGetVersion(&version);
    if (fftPlanBytes < getFFTPlanBudget()) {
      for (int dir = 0; dir < 2; ++dir)
        available |= makePMECallback(s, dir, single, tilted, k1, k2, k3) << dir;
    }
    p = findPMEPolicy(k1, k2, k3, single, tilted, version, available);
    if (!p) {
      releasePMETrial(s);
      return 0;
    }
    s->policy = p;
    for (int mode = 0; mode < PME_MODES; ++mode)
      if (!(mode & ~available))
        s->modes[s->count++] = mode;
    check(cuEventCreate(&s->lastUse, CU_EVENT_DISABLE_TIMING),
          "cuEventCreate PME");
    if (p->winner < 0) {
      check(cuEventCreate(&s->start, CU_EVENT_DEFAULT), "cuEventCreate PME");
      check(cuEventCreate(&s->stop, CU_EVENT_DEFAULT), "cuEventCreate PME");
    }
  }
  s->pinned = 1;
  s->measured = p->winner < 0 ? s->count * PME_ROUNDS : 0;
  return (intptr_t)s;
#endif
}

int64_t mdrtCudaPMETrials(intptr_t handle) {
  return handle ? ((struct PMETrial *)handle)->measured : 0;
}

int64_t mdrtCudaPMEBegin(intptr_t handle, int64_t iteration) {
  if (!handle)
    return 0;
  struct PMETrial *s = (struct PMETrial *)handle;
  if (iteration >= s->measured)
    return s->policy->winner;
  check(cuEventRecord(s->start, s->stream), "cuEventRecord PME");
  return s->modes[iteration % s->count];
}

void mdrtCudaPMEFFT(intptr_t handle, int64_t dir, uintptr_t input,
                    uintptr_t output) {
  struct PMETrial *s = (struct PMETrial *)handle;
  cufftHandle plan = s->callback[dir];
  cufftResult result;
  if (s->policy->single)
    result = dir ? cufftExecC2R(plan, (cufftComplex *)input, (float *)output)
                 : cufftExecR2C(plan, (float *)input, (cufftComplex *)output);
  else
    result =
        dir ? cufftExecZ2D(plan, (cufftDoubleComplex *)input, (double *)output)
            : cufftExecD2Z(plan, (double *)input, (cufftDoubleComplex *)output);
  isPending = 1;
  checkFFT(result, "cuFFT PME callback", s->policy->k[0], s->policy->k[1],
           s->policy->k[2]);
}

static float medianPME(float a, float b, float c) {
  if (a > b) {
    float t = a;
    a = b;
    b = t;
  }
  if (b > c) {
    float t = b;
    b = c;
    c = t;
  }
  return a > b ? a : b;
}

void mdrtCudaPMEEnd(intptr_t handle, int64_t iteration) {
  if (!handle)
    return;
  struct PMETrial *s = (struct PMETrial *)handle;
  if (iteration >= s->measured)
    return;
  int mode = s->modes[iteration % s->count], round = iteration / s->count;
  check(cuEventRecord(s->stop, s->stream), "cuEventRecord PME");
  check(cuEventSynchronize(s->stop), "cuEventSynchronize PME");
  check(cuEventElapsedTime(&s->samples[mode][round], s->start, s->stop),
        "cuEventElapsedTime PME");
  if (iteration + 1 != s->measured)
    return;
  float best = 0;
  for (int i = 0; i < s->count; ++i) {
    mode = s->modes[i];
    float *t = s->samples[mode];
    float median = medianPME(t[1], t[2], t[3]);
    if (!i || median < best) {
      best = median;
      s->policy->winner = mode;
    }
    if (getenv("MDRT_PME_REPORT"))
      fprintf(stderr, "mdrt: PME transform trial %d %.3f us\n", mode,
              1000 * median);
  }
  if (getenv("MDRT_PME_REPORT"))
    fprintf(stderr, "mdrt: PME transform mode %d for %d x %d x %d\n",
            s->policy->winner, s->policy->k[0], s->policy->k[1],
            s->policy->k[2]);
}

void mdrtCudaPMERelease(intptr_t handle) {
  if (!handle)
    return;
  struct PMETrial *s = (struct PMETrial *)handle;
  check(cuEventRecord(s->lastUse, s->stream), "cuEventRecord PME");
  s->pinned = 0;
}

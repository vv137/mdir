// Graph replay of a fixed-cell TOWERS step. The canonical banks
// contain the last accepted state even if the force or constraint check fails.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <map>

extern "C" cudaStream_t mgpuStreamCreate();
extern "C" void mdrtCountBuild();
extern "C" int mdrtRetainFFTPlan(int64_t, int64_t, int64_t, int, int, const float *);
extern "C" void mdrtReleaseFFTPlan(int);
extern "C" void mdrtStopConstraintFailure(int64_t, int64_t);
extern "C" void mdrtSpansFlush();
extern "C" void mdrtSpansSuspend(int32_t);

namespace {
void ck(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "mdrt: TOWERS step graph %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}
struct Progress {
  int64_t accepted, origin;
  double dt, time, originTime;
  int fatal, badPosition;
};

__global__ void positions(const double *xyz, size_t size, Progress *state) {
  if (state->fatal) return;
  for (size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < size;
       i += size_t(gridDim.x) * blockDim.x)
    if (!isfinite(xyz[i]) || fabs(xyz[i]) > 1.e100)
      atomicExch(&state->badPosition, 1);
}

struct TrialTimer {
  bool constructed = false;
  int64_t cold = 0;
  double instantiateSeconds = 0;
  cudaEvent_t start{}, stop{};
  std::vector<std::pair<float, int64_t>> samples;
};
thread_local TrialTimer trial;

struct Flags { int *f[4]; };
__global__ void acceptDrift(Flags flags, Progress *state) {
  int rejected = 0;
  for (int j = 3; j >= 0; --j) {
    if (!flags.f[j]) continue;
    if (*flags.f[j]) rejected = j + 1;
    *flags.f[j] = 0;
  }
  if (!state->fatal) state->fatal = rejected ? rejected : state->badPosition ? 5 : 0;
  state->badPosition = 0;
}
__global__ void acceptForce(Progress *state, const int *status) {
  if (state->fatal) return;
  if (*status) { state->fatal = 6; return; }
  ++state->accepted;
  state->time = state->originTime + double(state->accepted - state->origin) * state->dt;
}
constexpr int maxUnroll = 16;
struct Graph {
  int64_t site;
  std::vector<void *> banks;
  std::vector<int64_t> bindings;
  std::vector<size_t> bytes, permutation;
  struct Phase {
    std::vector<void *> inputs, outputs;
    cudaGraph_t drift{}, force{};
    double *xyz{};
  };
  std::vector<Phase> phases;
  void **rows{};
  size_t capturing{}, published{};
  std::map<std::pair<size_t, int>, cudaGraphExec_t> ringExecs;
  Flags flags{};
  int *status{};
  size_t atoms{};
};
struct Cache {
  cudaStream_t stream;
  Progress *state{};
  std::vector<Graph> graphs;
  Graph *current{};
  std::unordered_map<void *, void *> partners;
  std::vector<void *> owned;
  std::vector<int> fftPlans;
  // The last bytes written to each mirror of host memory.
  std::unordered_map<void *, std::vector<char>> mirrored;
  int64_t mirrorWrites = 0, chunks = 0;
  size_t ownedBytes = 0;
  int failure = 0;
};

struct Builder {
  cudaGraph_t graph;
  cudaGraphNode_t tail{};
  void child(cudaGraph_t childGraph) {
    cudaGraphNode_t next;
    ck(cudaGraphAddChildGraphNode(&next, graph, tail ? &tail : nullptr,
                                  tail ? 1 : 0, childGraph), "add captured phase");
    tail = next;
  }
  template<class Kernel, class... Args>
  void kernel(dim3 grid, int block, Kernel f, Args... args) {
    cudaKernelNodeParams p{};
    p.func = reinterpret_cast<void *>(f);
    p.gridDim = dim3(grid);
    p.blockDim = dim3(block);
    void *params[] = {&args...};
    p.kernelParams = params;
    cudaGraphNode_t next;
    ck(cudaGraphAddKernelNode(&next, graph, tail ? &tail : nullptr, tail ? 1 : 0, &p),
       "add check kernel");
    tail = next;
  }
};

cudaGraphExec_t unroll(Cache &cache, Graph &g, int width, size_t first) {
  auto &exec = g.ringExecs[{first, width}];
  if (exec) return exec;
  if (trial.start) trial.constructed = true;
  cudaGraph_t graph;
  ck(cudaGraphCreate(&graph, 0), "create unrolled graph");
  Builder body{graph};
  for (int i = 0; i < width; ++i) {
    auto &phase = g.phases[(first + i) % g.phases.size()];
    body.child(phase.drift);
    body.kernel((g.atoms*3+255)/256, 256, positions, phase.xyz, g.atoms*3, cache.state);
    body.kernel(1, 1, acceptDrift, g.flags, cache.state);
    body.child(phase.force);
    body.kernel(1, 1, acceptForce, cache.state, g.status);
  }
  std::chrono::steady_clock::time_point began;
  if (trial.start) began = std::chrono::steady_clock::now();
  ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate unrolled graph");
  if (trial.start)
    trial.instantiateSeconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - began).count();
  ck(cudaGraphDestroy(graph), "release unrolled graph");
  return exec;
}

}

extern "C" intptr_t mdrtStepCreate() {
  auto *c = new Cache{};
  c->stream = mgpuStreamCreate();
  ck(cudaMalloc(&c->state, sizeof(Progress)), "allocate progress");
  return reinterpret_cast<intptr_t>(c);
}

extern "C" void mdrtStepPlan(intptr_t handle, int64_t k1, int64_t k2, int64_t k3,
                             int64_t forward, int64_t single, intptr_t tables) {
  auto &c = *reinterpret_cast<Cache *>(handle);
  int plan = mdrtRetainFFTPlan(k1, k2, k3, forward, single, reinterpret_cast<const float *>(tables));
  if (plan < 0) std::exit(1);
  c.fftPlans.push_back(plan);
}

extern "C" intptr_t mdrtStepFatalAddress(intptr_t handle) {
  return reinterpret_cast<intptr_t>(&reinterpret_cast<Cache *>(handle)->state->fatal);
}

// Pair actual allocations across enclosing coupling/output permutations.
static void *partner(Cache &c, void *src, size_t bytes) {
  auto it = c.partners.find(src);
  if (it != c.partners.end()) return it->second;
  void *dst;
  ck(cudaMalloc(&dst, bytes), "allocate proposal bank");
  c.partners.emplace(src, dst);
  c.partners.emplace(dst, src);
  c.owned.push_back(dst);
  c.ownedBytes += bytes;
  return dst;
}

// Follow each yield cycle by address. A read-only yield forwards its
// input; a fully defined cycle can write directly in logical result order.
static void planBanks(Cache &c, Graph &g, const std::vector<bool> &writes) {
  auto inputs = g.banks;
  std::vector<bool> direct = writes;
  for (size_t i = 0; i < inputs.size(); ++i)
    for (size_t j = g.permutation[i]; j != i; j = g.permutation[j])
      if (!writes[j]) direct[i] = false;
  do {
    auto &phase = g.phases.emplace_back();
    phase.inputs = inputs;
    phase.outputs = inputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
      size_t j = g.permutation[i];
      if (writes[j])
        phase.outputs[j] = partner(c, inputs[direct[j] ? i : j], g.bytes[j]);
    }
    for (size_t i = 0; i < inputs.size(); ++i)
      inputs[i] = phase.outputs[g.permutation[i]];
  } while (inputs != g.banks);
  std::vector<void *> rows;
  for (auto &p : g.phases) {
    rows.insert(rows.end(), p.inputs.begin(), p.inputs.end());
    rows.insert(rows.end(), p.outputs.begin(), p.outputs.end());
  }
  ck(cudaMalloc(&g.rows, rows.size() * sizeof(void *)), "allocate bank pointer table");
  ck(cudaMemcpyAsync(g.rows, rows.data(), rows.size() * sizeof(void *), cudaMemcpyHostToDevice,
                     c.stream), "initialize bank pointer table");
  ck(cudaStreamSynchronize(c.stream), "wait for bank pointer table");
}

extern "C" intptr_t mdrtStepBankRow(intptr_t handle) {
  auto &g = *reinterpret_cast<Cache *>(handle)->current;
  return reinterpret_cast<intptr_t>(g.rows + 2 * g.capturing * g.banks.size());
}
extern "C" int64_t mdrtStepPhases(intptr_t handle) {
  return reinterpret_cast<Cache *>(handle)->current->phases.size();
}
extern "C" intptr_t mdrtStepBank(intptr_t handle, int64_t i, int64_t role) {
  auto &g = *reinterpret_cast<Cache *>(handle)->current;
  const auto &p = g.phases[role == 2 ? g.published : g.capturing];
  return reinterpret_cast<intptr_t>(role == 1 ? p.outputs[i] : p.inputs[i]);
}

extern "C" int32_t mdrtStepBegin(intptr_t handle, int64_t site, int64_t count,
                                 int64_t bindingCount, intptr_t data) {
  mdrtSpansFlush();
  auto &c = *reinterpret_cast<Cache *>(handle);
  auto *table = reinterpret_cast<const int64_t *>(data);
  std::vector<void *> banks;
  for (int64_t i = 0; i < count; ++i) banks.push_back(reinterpret_cast<void *>(table[3*i]));
  const int64_t *extra = table + 3*count;
  std::vector<int64_t> bindings(extra, extra + bindingCount);
  for (auto &g : c.graphs)
    if (g.site == site && g.banks == banks && g.bindings == bindings) {
      c.current = &g;
      return 0;
    }
  if (trial.start) trial.constructed = true;
  auto &g = c.graphs.emplace_back();
  c.current = &g;
  g.site = site;
  g.banks = banks;
  g.bindings = std::move(bindings);
  std::vector<bool> writes;
  for (int64_t i = 0; i < count; ++i) {
    g.bytes.push_back(table[3*i+1]);
    auto field = table[3*i+2];
    writes.push_back(field >= count);
    g.permutation.push_back(field % count);
  }
  planBanks(c, g, writes);
  return 1;
}

extern "C" void mdrtStepCapture(intptr_t handle) {
  auto &c = *reinterpret_cast<Cache *>(handle);
  // The launches of the capture reach the stream, not the record.
  mdrtSpansSuspend(1);
  ck(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal), "begin drift");
}

extern "C" void mdrtStepDrift(intptr_t handle, intptr_t xyz, int64_t atoms,
                              intptr_t a, intptr_t b, intptr_t d, intptr_t e) {
  auto &c = *reinterpret_cast<Cache *>(handle);
  auto &g = *c.current;
  auto &phase = g.phases[g.capturing];
  phase.xyz = reinterpret_cast<double *>(xyz);
  g.atoms = atoms;
  g.flags = {{reinterpret_cast<int *>(a), reinterpret_cast<int *>(b),
              reinterpret_cast<int *>(d), reinterpret_cast<int *>(e)}};
  ck(cudaStreamEndCapture(c.stream, &phase.drift), "end drift");
  ck(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal), "begin force");
}
extern "C" void mdrtStepForce(intptr_t handle, intptr_t status) {
  auto &c = *reinterpret_cast<Cache *>(handle);
  auto &g = *c.current;
  g.status = reinterpret_cast<int *>(status);
}

extern "C" void mdrtStepEnd(intptr_t handle) {
  auto &c = *reinterpret_cast<Cache *>(handle);
  auto &g = *c.current;
  ck(cudaStreamEndCapture(c.stream, &g.phases[g.capturing].force), "end guarded force and kick");
  if (++g.capturing < g.phases.size())
    ck(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal), "capture reverse bank roles");
  else
    mdrtSpansSuspend(0);
}

extern "C" void mdrtStepSeed(intptr_t handle, int64_t start, double initialTime, double dt) {
  mdrtSpansFlush();
  auto &c = *reinterpret_cast<Cache *>(handle);
  Progress p{start, start, dt, initialTime, initialTime, 0, 0};
  ck(cudaMemcpyAsync(c.state, &p, sizeof(p), cudaMemcpyHostToDevice, c.stream), "restore checkpoint clock");
  ck(cudaStreamSynchronize(c.stream), "wait for clock initialization");
}

__global__ void acceptedBoundary(Progress *p) {
  ++p->accepted;
  p->time = p->originTime + double(p->accepted - p->origin) * p->dt;
}
extern "C" void mdrtStepBoundary(intptr_t handle) {
  mdrtSpansFlush();
  auto &c = *reinterpret_cast<Cache *>(handle);
  acceptedBoundary<<<1,1,0,c.stream>>>(c.state);
  ck(cudaGetLastError(), "advance boundary step");
}

extern "C" int64_t mdrtStepRead(intptr_t handle, int64_t expected) {
  mdrtSpansFlush();
  auto &c = *reinterpret_cast<Cache *>(handle);
  int64_t accepted = -1;
  ck(cudaMemcpyAsync(&accepted, &c.state->accepted, sizeof(accepted), cudaMemcpyDeviceToHost, c.stream),
     "read absolute step for schedule boundary");
  ck(cudaStreamSynchronize(c.stream), "wait for absolute step");
  if (accepted != expected) {
    std::fprintf(stderr, "mdrt: graph absolute step %lld, schedule expects %lld\n",
                 (long long)accepted, (long long)expected);
    std::exit(1);
  }
  return accepted;
}

extern "C" int64_t mdrtStepRun(intptr_t handle, int64_t start, int64_t k) {
  mdrtSpansFlush();
  auto &c = *reinterpret_cast<Cache *>(handle);
  auto &g = *c.current;
  g.published = 0;
  if (!k) return start;
  ++c.chunks;
  Progress p{};
  int64_t cursor = start;
  bool replay = false;
  while (cursor < start + k) {
    const int width = replay ? 1 : std::min<int64_t>(start + k - cursor, maxUnroll);
    auto exec = unroll(c, g, width, (cursor - start) % g.phases.size());
    ck(cudaGraphLaunch(exec, c.stream), "launch");
    ck(cudaMemcpyAsync(&p, c.state, sizeof(p), cudaMemcpyDeviceToHost, c.stream), "read progress");
    ck(cudaStreamSynchronize(c.stream), "wait for progress");
    if (p.accepted < cursor || p.accepted > cursor + width ||
        (!p.fatal && p.accepted != cursor + width)) {
      std::fprintf(stderr, "mdrt: inconsistent TOWERS graph progress\n");
      std::exit(1);
    }
    cursor = p.accepted;
    if (p.fatal && replay) break;
    replay = p.fatal != 0;
    if (replay) {
      // The input phase at cursor still owns the last accepted state. Retry
      // exactly the failed proposal before allowing any later step to run.
      std::fprintf(stderr, "mdrt: replaying step %lld after graph check %d\n",
                   (long long)(cursor + 1), p.fatal);
      ck(cudaMemsetAsync(&c.state->fatal, 0, sizeof(int), c.stream), "retry rejected proposal");
    }
  }
  g.published = (p.accepted - start) % g.phases.size();
  for (int64_t n = start; n < p.accepted; ++n) mdrtCountBuild();
  c.failure = p.fatal;
  if (!c.failure && (p.accepted != start+k ||
      fabs(p.time - (p.originTime + double(p.accepted-p.origin)*p.dt)) > 1.e-12))
    c.failure = 7;
  if (c.failure) ck(cudaStreamSynchronize(c.stream), "restore accepted state");
  return p.accepted;
}

// The kernels of a chunk read the arguments that the cell of a barostat sets
// from a device mirror (D[towers-graph-npt]): the host computes them before
// each chunk, and the words go by value in a launch on the graph's stream,
// ahead of the chunk; unchanged words issue nothing.
namespace {
struct Words { uint32_t w[512]; };
__global__ void mirrorWords(uint32_t *dst, Words words, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = words.w[i];
}
}
extern "C" void mdrtStepMirror(intptr_t handle, intptr_t device, intptr_t host, int64_t bytes) {
  auto &c = *reinterpret_cast<Cache *>(handle);
  if (bytes <= 0 || bytes > int64_t(sizeof(Words)) || bytes % 4) {
    std::fprintf(stderr, "mdrt: TOWERS graph mirror of %lld bytes\n", (long long)bytes);
    std::exit(1);
  }
  auto *dst = reinterpret_cast<void *>(device);
  const char *src = reinterpret_cast<const char *>(host);
  auto &last = c.mirrored[dst];
  if (last.size() == size_t(bytes) && std::equal(last.begin(), last.end(), src)) return;
  // A fault for the tests: keep the first cell, as a graph that baked it in
  // would.
  static const bool fault = [] {
    const char *v = std::getenv("MDRT_STEP_MIRROR_FAULT");
    return v && *v && *v != '0';
  }();
  if (fault && !last.empty()) return;
  ++c.mirrorWrites;
  last.assign(src, src + bytes);
  mdrtSpansFlush();
  Words words{};
  std::memcpy(words.w, src, bytes);
  mirrorWords<<<1, 128, 0, c.stream>>>(static_cast<uint32_t *>(dst), words, int(bytes / 4));
  ck(cudaGetLastError(), "write the cell mirror");
}

extern "C" void mdrtStepCheck(intptr_t handle, int64_t accepted) {
  auto &c = *reinterpret_cast<Cache *>(handle);
  if (!c.failure) return;
  if (c.failure <= 4) mdrtStopConstraintFailure(accepted+1, c.failure-1);
  std::fprintf(stderr, "mdrt: TOWERS graph failed at step %lld, code %d; last accepted %lld\n",
               (long long)(accepted+1), c.failure, (long long)accepted);
  std::exit(1);
}
extern "C" void mdrtStepDestroy(intptr_t handle) {
  mdrtSpansFlush();
  auto *c = reinterpret_cast<Cache *>(handle);
  if (c->chunks)
    std::fprintf(stderr, "mdrt: TOWERS step graph: %zu chunk graphs, %lld chunks, %lld cell writes, "
                 "%zu proposal banks of %.1f MiB\n",
                 c->graphs.size(), (long long)c->chunks, (long long)c->mirrorWrites, c->owned.size(),
                 c->ownedBytes / 1048576.0);
  for (auto &g : c->graphs) {
    for (auto &entry : g.ringExecs) ck(cudaGraphExecDestroy(entry.second), "release replay");
    for (auto &phase : g.phases) {
      ck(cudaGraphDestroy(phase.drift), "release drift");
      ck(cudaGraphDestroy(phase.force), "release force and kick");
    }
    ck(cudaFree(g.rows), "release bank pointer table");
  }
  for (int plan : c->fftPlans) mdrtReleaseFFTPlan(plan);
  for (void *p : c->owned) ck(cudaFree(p), "release proposal bank");
  ck(cudaFree(c->state), "release progress");
  delete c;
}

// Timers exist only during setup. Discard the first half of the chunks
// after excluding graph construction, then pool elapsed time per step.
extern "C" void mdrtStepTrialBegin() {
  mdrtSpansFlush();
  if (!trial.start) {
    ck(cudaEventCreate(&trial.start), "create trial start");
    ck(cudaEventCreate(&trial.stop), "create trial end");
  }
  trial.constructed = false;
  ck(cudaEventRecord(trial.start, mgpuStreamCreate()), "begin ordinary chunk trial");
}
extern "C" void mdrtStepTrialEnd(int64_t steps) {
  mdrtSpansFlush();
  ck(cudaEventRecord(trial.stop, mgpuStreamCreate()), "end ordinary chunk trial");
  ck(cudaEventSynchronize(trial.stop), "wait for ordinary chunk trial");
  float ms;
  ck(cudaEventElapsedTime(&ms, trial.start, trial.stop), "time ordinary chunk");
  if (trial.constructed) ++trial.cold;
  else if (steps) trial.samples.emplace_back(ms, steps);
}
extern "C" int mdrtStepTrialResult(double *mean, int64_t *measured, int64_t *cold,
                                   double *instantiate, double *warm, double *timed) {
  *instantiate += trial.instantiateSeconds;
  *cold = trial.cold;
  *measured = trial.samples.size() / 2;
  double us = 0;
  int64_t steps = 0;
  for (size_t i = trial.samples.size() - *measured; i < trial.samples.size(); ++i) {
    us += 1000.0 * trial.samples[i].first;
    steps += trial.samples[i].second;
  }
  *timed += us * 1.e-6;
  for (size_t i = 0; i < trial.samples.size() - *measured; ++i)
    *warm += trial.samples[i].first * 1.e-3;
  *mean = steps ? us / steps : 0;
  if (trial.start) ck(cudaEventDestroy(trial.start), "release trial start");
  if (trial.stop) ck(cudaEventDestroy(trial.stop), "release trial end");
  trial = {};
  return *measured >= 5 && *mean > 0;
}

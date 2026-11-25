/* Atom-local valence work units, Cerutti et al. (2024), Sec. IV B. */
struct BondedWords {
  int32_t *data;
  size_t size, capacity;
};

static void bondedWord(struct BondedWords *words, int32_t value) {
  if (words->size == INT32_MAX) {
    fprintf(stderr, "mdrt: bonded instruction offsets exceed i32\n");
    abort();
  }
  if (words->size == words->capacity) {
    words->capacity = words->capacity ? 2 * words->capacity : 256;
    int32_t *data = realloc(words->data, words->capacity * sizeof(int32_t));
    if (!data)
      abort();
    words->data = data;
  }
  words->data[words->size++] = value;
}

struct BondedDirectory {
  struct MdrtTuples *sets[5];
  int unpacked, count;
  int64_t blocks, size;
  void *words;
  struct BondedDirectory *next;
};

static void freeBondedDirectories(struct MdrtTuples *e, int gone) {
  struct BondedDirectory *plan = e->bondedPlans;
  while (plan) {
    struct BondedDirectory *next = plan->next;
    if (!gone)
      check(cuMemFree((CUdeviceptr)plan->words), "cuMemFree");
    free(plan);
    plan = next;
  }
}

/* Directory offset -> number of imports, owner interval, blocks in that
   interval, instruction count per set, imports
   (file atom, complemented for a halo), then each set's instructions.
   Split pathological intervals at an instruction boundary before shared
   memory fills. Each such block still flushes only its owned components. */
static void buildBondedDirectory(struct BondedDirectory *plan) {
  int64_t particles = plan->sets[0]->particles;
  size_t groups = (particles + 127) / 128;
  struct BondedWords data = {0}, directory = {0};
  struct BondedWords *lists = calloc(groups ? groups : 1, sizeof(*lists));
  if (!lists)
    abort();
  for (int s = 0; s < plan->count; ++s) {
    struct MdrtTuples *e = plan->sets[s];
    for (int t = 0; t < e->tuples; ++t) {
      const int32_t *atoms = e->file + (size_t)t * e->arity;
      for (int j = 0; j < e->arity; ++j) {
        int group = atoms[j] / 128, k = 0;
        while (k < j && atoms[k] / 128 != group)
          ++k;
        if (k == j) {
          bondedWord(lists + group, s);
          bondedWord(lists + group, t);
        }
      }
    }
  }
  /* Shorter lists expose more independent blocks when even one block per
     occupied owner group would leave multiprocessors idle. */
  size_t termLimit = SIZE_MAX;
  if (plan->count == 4 && !plan->unpacked) {
    size_t occupied = 0;
    for (size_t g = 0; g < groups; ++g)
      occupied += lists[g].size != 0;
    CUdevice device;
    int multiprocessors;
    check(cuCtxGetDevice(&device), "cuCtxGetDevice");
    check(cuDeviceGetAttribute(&multiprocessors,
                               CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
          "cuDeviceGetAttribute");
    if (occupied < (size_t)multiprocessors)
      termLimit = 64;
  }
  int32_t *local = mdrtTuplesAllocate(particles * sizeof(int32_t));
  memset(local, -1, particles * sizeof(int32_t));
  for (size_t g = 0; g < groups; ++g) {
    struct BondedWords list = lists[g];
    size_t cursor = 0, firstBlock = directory.size;
    while (cursor < list.size) {
      size_t start = cursor;
      int count = 0, atoms[256], lengths[5] = {0};
      while (cursor < list.size && (cursor - start) / 2 < termLimit) {
        int s = list.data[cursor], t = list.data[cursor + 1];
        struct MdrtTuples *e = plan->sets[s];
        const int32_t *members = e->file + (size_t)t * e->arity;
        int needed = 0;
        for (int j = 0; j < e->arity; ++j)
          needed += local[members[j]] < 0;
        if (count + needed > 256)
          break;
        for (int j = 0; j < e->arity; ++j) {
          int atom = members[j];
          if (local[atom] >= 0)
            continue;
          local[atom] = count;
          atoms[count++] = atom;
        }
        ++lengths[s];
        cursor += 2;
      }
      bondedWord(&directory, data.size);
      bondedWord(&data, count);
      bondedWord(&data, g);
      bondedWord(&data, 0);
      for (int s = 0; s < plan->count; ++s)
        bondedWord(&data, lengths[s]);
      for (int i = 0; i < count; ++i)
        bondedWord(&data, atoms[i] / 128 == (int)g ? atoms[i] : ~atoms[i]);
      for (size_t k = start; k < cursor; k += 2) {
        struct MdrtTuples *e = plan->sets[list.data[k]];
        int t = list.data[k + 1];
        bondedWord(&data, t);
        uint64_t packed = 0;
        for (int j = 0; j < e->arity; ++j) {
          int slot = local[e->file[t * e->arity + j]];
          if (plan->unpacked)
            bondedWord(&data, slot);
          else
            packed |= (uint64_t)slot << (10 * j);
        }
        if (!plan->unpacked) {
          bondedWord(&data, (int32_t)packed);
          if (e->arity > 3)
            bondedWord(&data, (int32_t)(packed >> 32));
        }
      }
      for (int i = 0; i < count; ++i)
        local[atoms[i]] = -1;
    }
    for (size_t b = firstBlock; b < directory.size; ++b)
      data.data[directory.data[b] + 2] = directory.size - firstBlock;
    free(list.data);
  }
  free(lists);
  free(local);
  if (!directory.size) {
    bondedWord(&directory, 0);
    bondedWord(&data, 0);
    bondedWord(&data, 0);
    bondedWord(&data, 1);
    for (int i = 0; i < plan->count; ++i)
      bondedWord(&data, 0);
  }
  if (data.size + directory.size > INT32_MAX)
    abort();
  plan->blocks = directory.size;
  for (size_t k = 0; k < directory.size; ++k)
    directory.data[k] += plan->blocks;
  for (size_t k = 0; k < data.size; ++k)
    bondedWord(&directory, data.data[k]);
  free(data.data);
  plan->size = directory.size;
  size_t bytes = directory.size * sizeof(int32_t);
  plan->words = tuplesDeviceKept(bytes);
  tuplesToDevice(plan->words, directory.data, bytes);
  deviceTupleStore()->deviceBytes[2] += bytes;
  free(directory.data);
}

int64_t mdrtBondedDirectory(int32_t unpacked, int64_t a, int64_t b, int64_t c,
                            int64_t d, int64_t e) {
  int64_t handles[] = {a, b, c, d, e};
  struct MdrtTuples *sets[5] = {0};
  int count = 0;
  while (count < 5 && handles[count]) {
    sets[count] = (struct MdrtTuples *)(intptr_t)handles[count];
    ++count;
  }
  if (!count)
    abort();
  struct BondedDirectory **link = &sets[0]->bondedPlans;
  while (*link) {
    if ((*link)->unpacked == unpacked &&
        memcmp(sets, (*link)->sets, sizeof(sets)) == 0)
      return (int64_t)(intptr_t)*link;
    link = &(*link)->next;
  }
  struct BondedDirectory *plan = calloc(1, sizeof(*plan));
  if (!plan)
    abort();
  memcpy(plan->sets, sets, sizeof(sets));
  plan->unpacked = unpacked;
  plan->count = count;
  buildBondedDirectory(plan);
  *link = plan;
  return (int64_t)(intptr_t)plan;
}
void _mlir_ciface_mdrtBondedWords(struct MdrtMemref1 *result, int64_t handle) {
  struct BondedDirectory *plan = (struct BondedDirectory *)(intptr_t)handle;
  setMemref1(result, plan->words, plan->size);
}
int64_t mdrtBondedBlockCount(int64_t handle) {
  return ((struct BondedDirectory *)(intptr_t)handle)->blocks;
}

/* The optional fixed-frame oracle consumes floating-point slot values from
   the direct global-index evaluator. It never decodes work-unit instructions.
 */
struct BondedCheck {
  size_t n;
  __int128 *sum;
  long double *magnitude, *termDifference, *continuous;
  uint64_t *degree;
};
int64_t mdrtBondedCheckBegin(int64_t particles) {
  struct BondedCheck *c = calloc(1, sizeof(*c));
  if (!c)
    abort();
  c->n = particles;
  c->sum = calloc(3 * particles, sizeof(*c->sum));
  c->magnitude = calloc(3 * particles, sizeof(*c->magnitude));
  c->termDifference = calloc(3 * particles, sizeof(*c->termDifference));
  c->continuous = calloc(3 * particles, sizeof(*c->continuous));
  c->degree = calloc(particles, sizeof(*c->degree));
  if (particles && (!c->sum || !c->magnitude || !c->termDifference ||
                    !c->continuous || !c->degree))
    abort();
  return (int64_t)(intptr_t)c;
}
void _mlir_ciface_mdrtBondedCheckTerms(int64_t checkHandle, int64_t setHandle,
                                       struct MdrtMemref2 *slots,
                                       struct MdrtMemref2 *original,
                                       double scale) {
  struct BondedCheck *c = (struct BondedCheck *)(intptr_t)checkHandle;
  struct MdrtTuples *e = (struct MdrtTuples *)(intptr_t)setHandle;
  size_t n = e->tuples * e->arity;
  int32_t *renumbered = NULL;
  if (e->ordered) {
    renumbered = mdrtTuplesAllocate(n * sizeof(int32_t));
    tuplesToHost(renumbered, e->deviceRenumbered, n * sizeof(int32_t));
  }
  const int32_t *indices = renumbered ? renumbered : e->file;
  double *values = mdrtTuplesAllocate(n * 3 * sizeof(double));
  tuplesToHost(values, (double *)slots->aligned + slots->offset,
               n * 3 * sizeof(double));
  double *reference = values;
  if (slots->aligned != original->aligned ||
      slots->offset != original->offset) {
    reference = mdrtTuplesAllocate(n * 3 * sizeof(double));
    tuplesToHost(reference, (double *)original->aligned + original->offset,
                 n * 3 * sizeof(double));
  }
  for (size_t k = 0; k < n; ++k) {
    size_t atom = indices[k];
    ++c->degree[atom];
    for (int a = 0; a < 3; ++a) {
      double f = values[3 * k + a];
      c->continuous[3 * atom + a] += f;
      double scaled = nearbyint(f * scale);
      if (!(scaled >= -0x1p63 && scaled < 0x1p63))
        abort();
      c->sum[3 * atom + a] += (int64_t)scaled;
      double originalForce = reference[3 * k + a];
      c->termDifference[3 * atom + a] += fabsl((long double)originalForce - f);
      c->magnitude[3 * atom + a] += fabs(originalForce);
    }
  }
  if (reference != values)
    free(reference);
  free(renumbered);
  free(values);
}
static void checkBondedResult(int64_t handle, struct MdrtMemref2 *units,
                              struct MdrtMemref2 *floating, double scale,
                              int single) {
  struct BondedCheck *c = (struct BondedCheck *)(intptr_t)handle;
  size_t n = 3 * c->n;
  int64_t *words = mdrtTuplesAllocate(n * sizeof(*words));
  size_t bytes = single ? sizeof(float) : sizeof(double);
  void *reference = mdrtTuplesAllocate(n * bytes);
  tuplesToHost(words, (int64_t *)units->aligned + units->offset,
               n * sizeof(*words));
  tuplesToHost(reference, (char *)floating->aligned + floating->offset * bytes,
               n * bytes);
  long double maxError = 0, maxRatio = 0;
  size_t mismatches = 0, outside = 0, active = 0;
  long double squares = 0;
  uint64_t digest = 14695981039346656037ULL;
  const long double u = single ? 0x1p-24L : 0x1p-53L;
  for (size_t i = 0; i < n; ++i) {
    mismatches += c->sum[i] != words[i];
    digest = (digest ^ (uint64_t)words[i]) * 1099511628211ULL;
    uint64_t degree = c->degree[i / 3];
    long double decoded = (long double)words[i] / scale;
    if (degree) {
      long double delta = decoded - c->continuous[i];
      squares += delta * delta;
      ++active;
    }
    double published = single ? (double)(float)decoded : (double)decoded;
    double main = single ? ((float *)reference)[i] : ((double *)reference)[i];
    long double difference = fabsl((long double)main - published);
    long double ku = degree * u;
    long double bound = degree / (2 * scale) + fabsl(decoded - published) +
                        c->termDifference[i];
    if (ku < 1)
      bound += ku / (1 - ku) * c->magnitude[i];
    else
      bound = INFINITY;
    outside += !(difference <= bound);
    if (difference > maxError)
      maxError = difference;
    if (bound > 0 && difference / bound > maxRatio)
      maxRatio = difference / bound;
  }
  fprintf(stderr,
          "bonded check: %zu components, %zu word mismatches, "
          "%zu outside bound, max error %.9Lg, max ratio %.9Lg, hash %016llx\n",
          n, mismatches, outside, maxError, maxRatio,
          (unsigned long long)digest);
  fprintf(stderr, "bonded rounding: %zu incident components, Q18 RMS %.9Lg\n",
          active, active ? sqrtl(squares / active) : 0.L);
  free(c->continuous);
  free(reference);
  free(words);
  free(c->termDifference);
  free(c->magnitude);
  free(c->degree);
  free(c->sum);
  free(c);
  if (mismatches || outside)
    abort();
}
#define BONDED_FINISH(NAME, SINGLE)                                            \
  void _mlir_ciface_##NAME(int64_t handle, struct MdrtMemref2 *units,          \
                           struct MdrtMemref2 *floating, double scale) {       \
    checkBondedResult(handle, units, floating, scale, SINGLE);                 \
  }
BONDED_FINISH(mdrtBondedCheckF32, 1)
BONDED_FINISH(mdrtBondedCheckF64, 0)
#undef BONDED_FINISH

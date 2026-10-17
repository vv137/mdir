/* The MDIR runtime: functions that compiled code calls. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* D184: a diagnostic central-difference comparison failed.
   The first five arguments are the lowered memref<?xi8> descriptor. */
void mdrtDerivativeFailure(void *allocated, const char *aligned, int64_t offset,
                           int64_t size, int64_t stride, double analytic,
                           double numerical, double error, double tolerance) {
  (void)allocated;
  for (int64_t i = 0; i < size; ++i)
    fputc(aligned[offset + i * stride], stderr);
  fprintf(stderr, ": analytic %.17g, central difference %.17g, "
                  "error %.6g exceeds tolerance %.6g\n",
          analytic, numerical, error, tolerance);
  fflush(stderr);
  abort();
}

/* The number of times that compiled code has built a neighbor structure
   since the library was loaded. */
static int64_t numBuilds = 0;

void mdrtCountBuild(void) { ++numBuilds; }

int64_t mdrtGetBuildCount(void) { return numBuilds; }

/* The number of times that compiled code has pruned the inner list of a
   dual list from its outer one (D114). */
static int64_t numPrunes = 0;

void mdrtCountPrune(void) { ++numPrunes; }

int64_t mdrtGetPruneCount(void) { return numPrunes; }

/* The check of the pruning (MDIR_CHECK_PRUNING, D268): at each
   step, compiled code counts the pairs within the cutoff in the outer list
   and in the inner one, through its active units, at that step's
   positions. The inner list is a part of the outer one, so the two counts
   differ only if it misses a pair, and the run stops. */
static int64_t numPruneChecks = 0, numPruneCheckedPairs = 0;
static int pruneCheckFailed = 0;

static void reportPruneChecks(void) {
  if (pruneCheckFailed)
    return;
  fprintf(stderr,
          "mdrt: the inner lists were checked at %lld steps: %lld pairs "
          "within the cutoff, none missed\n",
          (long long)numPruneChecks, (long long)numPruneCheckedPairs);
}

void mdrtCheckPruned(int64_t outer, int64_t inner) {
  if (outer != inner) {
    fprintf(stderr,
            "mdrt: the inner list holds %lld pairs within the cutoff, and "
            "the outer list %lld, at the check %lld (MDIR_CHECK_PRUNING)\n",
            (long long)inner, (long long)outer,
            (long long)numPruneChecks + 1);
    fflush(stderr);
    pruneCheckFailed = 1;
    exit(1);
  }
  if (numPruneChecks++ == 0)
    atexit(reportPruneChecks);
  numPruneCheckedPairs += outer;
}

/* The number of builds at a fixed interval that found the structure no
   longer valid: between the build before and this one, pairs within the
   cutoff may have been left out (the opt-in policy `interval`, D88). */
static int64_t numLateBuilds = 0;

void mdrtCountLateBuild(void) { ++numLateBuilds; }

int64_t mdrtGetLateBuildCount(void) { return numLateBuilds; }

/* The largest number of blocks of entries that a build of a structure of
   groups of neighbors (D89) took, the blocks it had, and the longest list,
   for the log of the run. */
static int64_t groupsBlocks = 0, groupsCapacity = 0, groupsLongest = 0;
/* The groups, over all builds, whose partners of excluded pairs did not
   fit the memory of a warp and took their excluded pairs from the rows
   (D106), for the log of the run. */
static int64_t groupsOverflow = 0;

void mdrtNoteGroups(int64_t blocks, int64_t capacity, int64_t longest,
                    int64_t overflow) {
  if (blocks > groupsBlocks) {
    groupsBlocks = blocks;
    groupsCapacity = capacity;
  }
  if (longest > groupsLongest)
    groupsLongest = longest;
  groupsOverflow += overflow;
}

/* Sets the counts of the builds and prunings and the notes of the groups
   back to none, so that the log of a run counts its own alone after the
   trial runs of a search of the skins (D281). */
void mdrtResetCounts(void) {
  numBuilds = numPrunes = numLateBuilds = 0;
  groupsBlocks = groupsCapacity = groupsLongest = groupsOverflow = 0;
}

/* A program that embeds a run may take its failures instead of the process
   exiting (D196). The handler records the failure; the run
   goes on to the end of its segment, whose state the program discards. */
static void (*stopHandler)(const char *) = NULL;
void mdrtSetStopHandler(void (*handler)(const char *)) {
  stopHandler = handler;
}

/* Delivers a failure: to the handler of an embedding program, or to the
   standard error, ending the process. */
static void stopWith(const char *message) {
  if (stopHandler) {
    stopHandler(message);
    return;
  }
  fprintf(stderr, "mdir: %s\n", message);
  exit(1);
}

/* In a step of a run with constraints, the stop of a build whose positions
   are not numbers is recorded and the step runs to its end, which D225
   makes safe: the tests of the constraints are read after the evaluation,
   and a failed set stops the run with its own message, naming the step and
   the set; otherwise mdrtStopPending delivers the recorded stop after the
   tests (D278, the maintainer's ruling on PR #381). */
static int deferringStops = 0, hasPendingStop = 0;
static char pendingStop[256];
void mdrtDeferStops(void) { deferringStops = 1; }
void mdrtStopPending(void) {
  deferringStops = 0;
  if (!hasPendingStop)
    return;
  hasPendingStop = 0;
  stopWith(pendingStop);
}

/* A build of a neighbor structure found positions that are not numbers,
   or are numbers far outside the cell (beyond 1e100, or whose wrap does not
   land in the cell, D176): the run has failed, whatever produced them
   (D107, D225). */
void mdrtStopNotNumbers(int64_t count) {
  char message[256];
  if (count == 1)
    snprintf(message, sizeof message,
             "a position is not a number or is far outside the cell at a "
             "build of the neighbor structure; the run has failed (a time "
             "step too long, a bad contact, or a defect of mdir)");
  else
    snprintf(message, sizeof message,
             "%lld positions are not numbers or are far outside the cell at "
             "a build of the neighbor structure; the run has failed (a time "
             "step too long, a bad contact, or a defect of mdir)",
             (long long)count);
  if (deferringStops) {
    if (!hasPendingStop) {
      snprintf(pendingStop, sizeof pendingStop, "%s", message);
      hasPendingStop = 1;
    }
    return;
  }
  stopWith(message);
}

/* The constraints of a set, SETTLE (`set` 0) or the groups of SHAKE of `set`
   hydrogens, left a length off its target or not a number in the step
   `step`: the run has failed (D278). */
void mdrtStopConstraintFailure(int64_t step, int64_t set) {
  char name[32];
  if (set == 0)
    snprintf(name, sizeof name, "settles");
  else
    snprintf(name, sizeof name, "shake%lld", (long long)set);
  char message[320];
  snprintf(message, sizeof message,
           "the constraints of the set '%s' failed at step %lld: a "
           "constrained length is off its target or is not a number; the run "
           "has failed (a time step too long, a bad contact, or a defect of "
           "mdir)",
           name, (long long)step);
  /* The failure of the constraints, not the positions that are not numbers
     that it left, is what the run reports. */
  hasPendingStop = 0;
  stopWith(message);
}

int64_t mdrtGetGroupsBlocks(void) { return groupsBlocks; }
int64_t mdrtGetGroupsCapacity(void) { return groupsCapacity; }
int64_t mdrtGetGroupsLongest(void) { return groupsLongest; }
int64_t mdrtGetGroupsOverflow(void) { return groupsOverflow; }

/*===----------------------------------------------------------------------===
 * Random numbers
 *===----------------------------------------------------------------------===*/

/* Philox 4x32 with 10 rounds (Salmon et al. 2011): the four words of
   `counter` under `key`, into `result`. */
void mdrtPhilox4x32(const uint32_t counter[4], const uint32_t key[2],
                    uint32_t result[4]) {
  uint32_t c0 = counter[0], c1 = counter[1], c2 = counter[2],
           c3 = counter[3];
  uint32_t k0 = key[0], k1 = key[1];
  for (int round = 0; round != 10; ++round) {
    if (round != 0) {
      k0 += 0x9E3779B9u;
      k1 += 0xBB67AE85u;
    }
    uint64_t p0 = (uint64_t)0xD2511F53u * c0;
    uint64_t p1 = (uint64_t)0xCD9E8D57u * c2;
    uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
    uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
    c0 = n0;
    c1 = (uint32_t)p1;
    c2 = n2;
    c3 = (uint32_t)p0;
  }
  result[0] = c0;
  result[1] = c1;
  result[2] = c2;
  result[3] = c3;
}

/* The numbers that one entity draws from one stream in one step, from the
   key of A13 in docs/decisions.md: the seed is the key of Philox; the
   counter is the step, the entity, and the stream with the draw index.
   `block` counts the calls of Philox, each of which gives four words. */
typedef struct {
  uint32_t key[2];
  uint32_t counter[4];
  uint32_t words[4];
  int used;
} Draws;

static void initDraws(Draws *draws, uint64_t seed, int64_t step,
                      uint32_t entity, uint32_t stream) {
  draws->key[0] = (uint32_t)seed;
  draws->key[1] = (uint32_t)(seed >> 32);
  draws->counter[0] = (uint32_t)(uint64_t)step;
  draws->counter[1] = (uint32_t)((uint64_t)step >> 32);
  draws->counter[2] = entity;
  /* The stream in the high 8 bits, the index of the block in the low 24. */
  draws->counter[3] = stream << 24;
  draws->used = 4;
}

static uint32_t drawWord(Draws *draws) {
  if (draws->used == 4) {
    mdrtPhilox4x32(draws->counter, draws->key, draws->words);
    ++draws->counter[3];
    draws->used = 0;
  }
  return draws->words[draws->used++];
}

/* Uniform in (0, 1), with 53 random bits. */
static double drawUniform(Draws *draws) {
  uint64_t high = drawWord(draws), low = drawWord(draws);
  uint64_t bits = ((high << 32) | low) >> 11;
  return ((double)bits + 0.5) * 0x1.0p-53;
}

/* Standard normal, by the method of Box and Muller; the second number of
   each pair is not used, so a number depends only on its draw index. */
static double drawNormal(Draws *draws) {
  double u1 = drawUniform(draws), u2 = drawUniform(draws);
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* Gamma with shape `a` and scale 1 (Marsaglia and Tsang 2000). */
static double drawGamma(Draws *draws, double a) {
  if (a < 1.0) {
    double u = drawUniform(draws);
    return drawGamma(draws, a + 1.0) * pow(u, 1.0 / a);
  }
  double d = a - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
  for (;;) {
    double x = drawNormal(draws), v = 1.0 + c * x;
    if (v <= 0.0)
      continue;
    v = v * v * v;
    double u = drawUniform(draws);
    if (log(u) < 0.5 * x * x + d - d * v + d * log(v))
      return d * v;
  }
}

/* The factor that stochastic velocity rescaling (Bussi, Donadio, and
   Parrinello 2007, the appendix) scales the velocities by: `kinetic` is the
   kinetic energy, `target` its mean at the temperature of the bath,
   `freedom` the number of degrees of freedom, and `decay` exp(-t / tau)
   for the time t since the thermostat acted last. The numbers are those
   of stream 0 at `step`. */
double mdrtBussiFactor(int64_t seed, int64_t step, double kinetic,
                       double target, double freedom, double decay) {
  if (!(kinetic > 0.0) || !(freedom > 0.0))
    return 1.0;
  Draws draws;
  initDraws(&draws, (uint64_t)seed, step, /*entity=*/0, /*stream=*/0);
  double r1 = drawNormal(&draws);
  /* The sum of the squares of freedom - 1 standard normals: chi-squared,
     or twice a gamma of half the shape. */
  double rest = freedom > 1.0 ? 2.0 * drawGamma(&draws, 0.5 * (freedom - 1.0))
                              : 0.0;
  double next = kinetic +
                (1.0 - decay) * (target * (r1 * r1 + rest) / freedom -
                                 kinetic) +
                2.0 * r1 *
                    sqrt(kinetic * target / freedom * (1.0 - decay) * decay);
  return sqrt(next / kinetic);
}

/* The change of the logarithm of the volume that stochastic cell rescaling
   (Bernetti and Bussi 2020) makes over one period of the barostat. The
   square root of the volume, λ = √V, takes one step of Euler and Maruyama
   of Eq. (7), Eq. (S7) of the supplementary material:

     λ' = λ − (f λ / 2)(P0 − P − k_B T c / (2V)) + √(k_B T f c / 2) R,
     f = β_T Δt_p / τ_p,

   whose noise, unlike that of the step in ε = ln V, does not depend on the
   volume, which makes a move and its reverse as likely as the paper's
   reversible integrators need. The two agree to first order in Δt_p; the
   term k_B T c / (2V) is what the Itô chain rule adds for λ. Returns
   Δε = 2 ln(λ'/λ). The pressures are in bar, `compressibility` β_T in
   1/bar, `volume` V in nm³ before the scaling, `kT` k_B T at the
   temperature of the bath in kJ/mol, `rate` Δt_p / τ_p, and
   c = 16.6053906717 bar nm³ mol/kJ. The positions and the cell are then
   scaled by exp(Δε / 3), the velocities by exp(−Δε / 3). The number R is
   that of stream 1 at `step`. */
double mdrtBarostatStrain(int64_t seed, int64_t step, double pressure,
                          double target, double volume, double kT,
                          double compressibility, double rate) {
  const double conversion = 16.6053906717;
  Draws draws;
  initDraws(&draws, (uint64_t)seed, step, /*entity=*/0, /*stream=*/1);
  double r = drawNormal(&draws);
  double f = compressibility * rate;
  double thermal = kT * conversion;
  double lambda = sqrt(volume);
  double next = lambda -
                0.5 * f * lambda * (target - pressure - thermal / (2.0 * volume)) +
                sqrt(0.5 * thermal * f) * r;
  return 2.0 * log(next / lambda);
}

/* Semi-isotropic stochastic cell rescaling (D119): one Euler-Maruyama step
   of eqs. (9a) and (9b) of Bernetti and Bussi, J. Chem. Phys. 153, 114107
   (2020), for e_xy = ln A and e_z = ln L, with their own normal numbers of
   stream 1 of the step: the first for the area, which the isotropic
   coupling takes as well, and the second for the height. The pressures and
   the target in bar; the tension, of a surface times their number, in
   bar nm; the volume and the height in nm^3 and nm; kT in kJ/mol; the
   compressibility in 1/bar; rate, the period over the time constant. With
   no tension the sum of the two is the step of e = ln V of eq. (5). */
/* The pressures that semi-isotropic coupling took over the run, that of x
   and y, that of z, and their difference, summed over the periods and over
   blocks of 100 periods, whose spread gives the errors of the means; the
   driver reports them at the end (mdrtGetSemiPressures). */
enum { SEMI_BLOCK = 100 };
static struct {
  double lateral, sum[3], block[3], blockSum[3], blockSquares[3];
  int64_t count, inBlock, blocks;
} semiPressures;

static void addSemiPressures(double normal) {
  double values[3] = {semiPressures.lateral, normal,
                      semiPressures.lateral - normal};
  for (int k = 0; k != 3; ++k) {
    semiPressures.sum[k] += values[k];
    semiPressures.block[k] += values[k];
  }
  ++semiPressures.count;
  if (++semiPressures.inBlock == SEMI_BLOCK) {
    for (int k = 0; k != 3; ++k) {
      double mean = semiPressures.block[k] / SEMI_BLOCK;
      semiPressures.blockSum[k] += mean;
      semiPressures.blockSquares[k] += mean * mean;
      semiPressures.block[k] = 0.0;
    }
    semiPressures.inBlock = 0;
    ++semiPressures.blocks;
  }
}

/* The means of the pressures of x and y, of z, and of their difference, in
   bar, in out[0..2], and the errors of the means from the blocks in
   out[3..5] (0 with fewer than two blocks). Returns the number of periods. */
int64_t mdrtGetSemiPressures(double *out) {
  int64_t n = semiPressures.count, b = semiPressures.blocks;
  for (int k = 0; k != 3; ++k) {
    out[k] = n > 0 ? semiPressures.sum[k] / (double)n : 0.0;
    double error = 0.0;
    if (b > 1) {
      double mean = semiPressures.blockSum[k] / (double)b;
      double var = (semiPressures.blockSquares[k] - b * mean * mean) /
                   (double)(b - 1);
      error = var > 0.0 ? sqrt(var / (double)b) : 0.0;
    }
    out[3 + k] = error;
  }
  return n;
}

static void drawSemi(int64_t seed, int64_t step, double *r) {
  Draws draws;
  initDraws(&draws, (uint64_t)seed, step, /*entity=*/0, /*stream=*/1);
  r[0] = drawNormal(&draws);
  r[1] = drawNormal(&draws);
}

double mdrtBarostatStrainArea(int64_t seed, int64_t step, double pressure,
                              double target, double volume, double height,
                              double kT, double compressibility, double rate,
                              double tension) {
  const double conversion = 16.6053906717;
  double r[2];
  drawSemi(seed, step, r);
  double f = compressibility * rate;
  double thermal = kT * conversion;
  semiPressures.lateral = pressure;
  return -(2.0 * f / 3.0) * (target - tension / height - pressure) +
         sqrt(4.0 * thermal * f / (3.0 * volume)) * r[0];
}

double mdrtBarostatStrainHeight(int64_t seed, int64_t step, double pressure,
                                double target, double volume, double kT,
                                double compressibility, double rate) {
  const double conversion = 16.6053906717;
  double r[2];
  drawSemi(seed, step, r);
  double f = compressibility * rate;
  double thermal = kT * conversion;
  addSemiPressures(pressure);
  return -(f / 3.0) * (target - pressure) +
         sqrt(2.0 * thermal * f / (3.0 * volume)) * r[1];
}

/* Anisotropic stochastic cell rescaling (D163c): one Euler-Maruyama step
   of the strain of axis `axis`, e_a = ln L_a, with its own normal number,
   the number `axis` of stream 1 of the step (the first two are those of
   semi-isotropic coupling):

     de_a = -(f_a / 3)(P0 - P_aa) + sqrt(2 kT c f_a / (3 V)) R_a,

   f_a = beta_a * rate. For each axis this is the step of z of eq. (9b) of
   Bernetti and Bussi, J. Chem. Phys. 153, 114107 (2020); with the density
   V exp(-(P0 V + F) / kT) in the three strains, whose derivative along e_a
   is 1 + V (P_aa - P0) / kT, the drift of the zero flux is the one above,
   since the diffusion, proportional to 1 / V, falls along e_a by as much
   as the factor V of the density adds. Units as for semi-isotropic
   coupling. The pressures of the axes are summed for the means at the
   end, as those of semi-isotropic coupling are. */
static struct {
  double sum[3], block[3], blockSum[3], blockSquares[3];
  int64_t count[3], inBlock[3], blocks[3];
} axisPressures;

double mdrtBarostatStrainAxis(int64_t seed, int64_t step, int64_t axis,
                              double pressure, double target, double volume,
                              double kT, double compressibility,
                              double rate) {
  const double conversion = 16.6053906717;
  Draws draws;
  initDraws(&draws, (uint64_t)seed, step, /*entity=*/0, /*stream=*/1);
  double r = 0.0;
  for (int64_t k = 0; k <= axis; ++k)
    r = drawNormal(&draws);
  int a = (int)axis;
  axisPressures.sum[a] += pressure;
  axisPressures.block[a] += pressure;
  ++axisPressures.count[a];
  if (++axisPressures.inBlock[a] == SEMI_BLOCK) {
    double mean = axisPressures.block[a] / SEMI_BLOCK;
    axisPressures.blockSum[a] += mean;
    axisPressures.blockSquares[a] += mean * mean;
    axisPressures.block[a] = 0.0;
    axisPressures.inBlock[a] = 0;
    ++axisPressures.blocks[a];
  }
  double f = compressibility * rate;
  double thermal = kT * conversion;
  return -(f / 3.0) * (target - pressure) +
         sqrt(2.0 * thermal * f / (3.0 * volume)) * r;
}

/* The means of the pressures of x, y, and z that anisotropic coupling took,
   in bar, in out[0..2], and the errors of the means from blocks of 100
   periods in out[3..5]. Returns the number of periods. */
int64_t mdrtGetAxisPressures(double *out) {
  for (int k = 0; k != 3; ++k) {
    int64_t n = axisPressures.count[k], b = axisPressures.blocks[k];
    out[k] = n > 0 ? axisPressures.sum[k] / (double)n : 0.0;
    double error = 0.0;
    if (b > 1) {
      double mean = axisPressures.blockSum[k] / (double)b;
      double var = (axisPressures.blockSquares[k] - b * mean * mean) /
                   (double)(b - 1);
      error = var > 0.0 ? sqrt(var / (double)b) : 0.0;
    }
    out[3 + k] = error;
  }
  return axisPressures.count[0];
}

/*===----------------------------------------------------------------------===
 * The rows of a neighbor matrix on the host
 *===----------------------------------------------------------------------===*/

/* The entries of a neighbor matrix on the host, a row for each particle,
   which grow wider when a build finds a particle with more neighbors than
   a row holds. Compiled code takes the buffer where it uses it
   (`mdrtHostMatrixEntries`). */
struct HostMatrix {
  int32_t *data;
  int64_t rows;
  int64_t width;
};

/* A memref of rank 2 as the C interface of MLIR passes it. */
struct HostBuffer2 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

static void allocateHostMatrix(struct HostMatrix *matrix) {
  matrix->data = malloc((size_t)(matrix->rows * matrix->width) *
                        sizeof(int32_t));
  if (!matrix->data) {
    fprintf(stderr,
            "mdrt: out of memory of the host for a neighbor matrix of "
            "%lld rows of %lld\n",
            (long long)matrix->rows, (long long)matrix->width);
    abort();
  }
}

/* An activation of an entry that an embedding program (a Python
   simulation) keeps across its calls (D215): the matrices
   that its code makes belong to it and are destroyed when the program
   closes it (#110). Several may be open, one for each live simulation; the
   one whose code runs is current. `mdir run` opens none. */
struct HostActivation {
  struct HostMatrix **matrices;
  size_t numMatrices, roomMatrices;
};
static struct HostActivation *currentActivation = NULL;

int64_t mdrtHostMatrixCreate(int64_t rows, int64_t width) {
  struct HostMatrix *matrix = calloc(1, sizeof(struct HostMatrix));
  if (!matrix) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  matrix->rows = rows > 0 ? rows : 1;
  matrix->width = width > 0 ? width : 1;
  allocateHostMatrix(matrix);
  struct HostActivation *a = currentActivation;
  if (a) {
    if (a->numMatrices == a->roomMatrices) {
      a->roomMatrices = a->roomMatrices ? 2 * a->roomMatrices : 8;
      a->matrices =
          realloc(a->matrices, a->roomMatrices * sizeof(*a->matrices));
      if (!a->matrices) {
        fprintf(stderr, "mdrt: out of memory of the host\n");
        abort();
      }
    }
    a->matrices[a->numMatrices++] = matrix;
  }
  return (int64_t)(intptr_t)matrix;
}

void *mdrtActivationOpen(void) {
  struct HostActivation *a = calloc(1, sizeof(struct HostActivation));
  if (!a) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  return a;
}

void mdrtActivationEnter(void *activation) { currentActivation = activation; }

void mdrtActivationLeave(void) { currentActivation = NULL; }

void mdrtActivationClose(void *activation) {
  struct HostActivation *a = activation;
  if (!a)
    return;
  if (currentActivation == a)
    currentActivation = NULL;
  for (size_t i = 0; i != a->numMatrices; ++i) {
    free(a->matrices[i]->data);
    free(a->matrices[i]);
  }
  free(a->matrices);
  free(a);
}

void _mlir_ciface_mdrtHostMatrixEntries(struct HostBuffer2 *result,
                                        int64_t handle) {
  struct HostMatrix *matrix = (struct HostMatrix *)(intptr_t)handle;
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
void mdrtHostMatrixGrow(int64_t handle, int64_t width) {
  struct HostMatrix *matrix = (struct HostMatrix *)(intptr_t)handle;
  if (width <= matrix->width)
    return;
  free(matrix->data);
  matrix->width = width + width / 4;
  allocateHostMatrix(matrix);
}

/*===----------------------------------------------------------------------===
  The structures of the tuples, kept from one activation to the next
  (D253, mdrt_tuples.h)
  ===----------------------------------------------------------------------===

  An embedding program opens a store, tells it where an activation begins,
  enters it while the code of the activation runs, and closes it when the
  program ends. The code takes the renumbered members of a tuple set and
  its incidence structure from these functions and gives them back where
  it would have freed them; without a store each is built for the caller,
  which frees it. */

#include "mdrt_tuples.h"

static struct MdrtTupleStore *currentTuples = NULL;

void *mdrtTuplesOpen(void) { return mdrtTupleStoreOpen(); }

void mdrtTuplesBegin(void *store) { mdrtTupleStoreBegin(store); }

/* Whether the store gives a structure that it has, or builds every one
   anew: for the tests, which compare the two. */
void mdrtTuplesKeep(void *store, int32_t keeps) {
  ((struct MdrtTupleStore *)store)->keeps = keeps != 0;
}

void mdrtTuplesEnter(void *store) { currentTuples = store; }

void mdrtTuplesLeave(void) { currentTuples = NULL; }

void mdrtTuplesClose(void *opaque) {
  struct MdrtTupleStore *store = opaque;
  if (!store)
    return;
  if (currentTuples == store)
    currentTuples = NULL;
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    free(e->file);
    free(e->ids);
    free(e->members);
    free(e->incidence);
    free(e);
  }
  free(store->entries);
  free(store);
}

/* The numbers of structures that were taken from the store and that were
   built, since it was opened. */
void mdrtTuplesCounts(void *opaque, int64_t *counts) {
  struct MdrtTupleStore *store = opaque;
  counts[0] = store ? store->kept : 0;
  counts[1] = store ? store->built : 0;
}

/* The time that the code has spent in the functions below with the store,
   in seconds. */
double mdrtTuplesSeconds(void *opaque) {
  struct MdrtTupleStore *store = opaque;
  return store ? store->seconds : 0.0;
}

static void tuplesRenumber(struct MdrtMemref2 *result,
                           struct MdrtMemref2 *members,
                           struct MdrtMemref1 *ids) {
  const int32_t *file = mdrtTuplesMembersOf(members);
  const int32_t *id = (const int32_t *)ids->aligned + ids->offset;
  int64_t tuples = members->sizes[0], arity = members->sizes[1];
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  if (!currentTuples) {
    int32_t *fresh = mdrtTuplesAllocate(bytes);
    mdrtTuplesRenumber(file, tuples, arity, id, ids->size, fresh);
    mdrtTuplesSetMemref2(result, fresh, tuples, arity);
    return;
  }
  int found = 0;
  struct MdrtTuples *e = mdrtTupleStoreTake(currentTuples, file, tuples, arity,
                                            id, ids->size, &found);
  if (!found) {
    mdrtTuplesRenumber(file, tuples, arity, id, ids->size, e->members);
  } else if (mdrtTuplesChecks()) {
    int32_t *anew = mdrtTuplesAllocate(bytes);
    mdrtTuplesRenumber(file, tuples, arity, id, ids->size, anew);
    if (memcmp(anew, e->members, bytes) != 0)
      mdrtTuplesDiffer("the renumbered members of a tuple set");
    free(anew);
  }
  ++e->held;
  mdrtTuplesSetMemref2(result, e->members, tuples, arity);
}

void _mlir_ciface_mdrtTuplesRenumber(struct MdrtMemref2 *result,
                                     struct MdrtMemref2 *members,
                                     struct MdrtMemref1 *ids) {
  double start = mdrtTuplesNow();
  tuplesRenumber(result, members, ids);
  mdrtTuplesSpent(currentTuples, start);
}

static void tuplesIncidence(struct MdrtMemref2 *result,
                            struct MdrtMemref2 *members, int64_t particles) {
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
  /* The width does not depend on the order: an entry that has had a
     structure has the width of the next. */
  int64_t width = e && e->width
                      ? e->width
                      : mdrtTuplesWidth(given, tuples, arity, particles);
  size_t bytes = (size_t)(particles * width) * sizeof(int32_t);
  if (!e) {
    int32_t *fresh = mdrtTuplesAllocate(bytes);
    mdrtTuplesFill(given, tuples, arity, particles, width, fresh);
    mdrtTuplesSetMemref2(result, fresh, particles, width);
    return;
  }
  if (!e->hasIncidence) {
    if (!e->incidence || e->width != width) {
      free(e->incidence);
      e->incidence = mdrtTuplesAllocate(bytes);
    }
    e->width = width;
    mdrtTuplesFill(given, tuples, arity, particles, width, e->incidence);
    e->hasIncidence = 1;
  } else if (mdrtTuplesChecks()) {
    int64_t anewWidth = mdrtTuplesWidth(given, tuples, arity, particles);
    int32_t *anew =
        mdrtTuplesAllocate((size_t)(particles * anewWidth) * sizeof(int32_t));
    mdrtTuplesFill(given, tuples, arity, particles, anewWidth, anew);
    if (anewWidth != width ||
        !mdrtTuplesSameRows(anew, e->incidence, arity, particles, width))
      mdrtTuplesDiffer("the incidence structure of a tuple set");
    free(anew);
  }
  ++e->held;
  mdrtTuplesSetMemref2(result, e->incidence, particles, width);
}

void _mlir_ciface_mdrtTuplesIncidence(struct MdrtMemref2 *result,
                                      struct MdrtMemref2 *members,
                                      int64_t particles) {
  double start = mdrtTuplesNow();
  tuplesIncidence(result, members, particles);
  mdrtTuplesSpent(currentTuples, start);
}

/* Where the code would free the members or the structure that it took. */
void _mlir_ciface_mdrtTuplesRelease(struct MdrtMemref2 *buffer) {
  if (!mdrtTupleStoreGiveBack(currentTuples, buffer->allocated))
    free(buffer->allocated);
}

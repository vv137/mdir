/* The MDIR runtime: functions that compiled code calls. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

/* A build of a neighbor structure found positions that are not numbers:
   the run has failed, whatever produced them (D107). */
void mdrtStopNotNumbers(int64_t count) {
  if (count == 1)
    fprintf(stderr, "mdir: a position is not a number at a build of the "
                    "neighbor structure; the run has failed (a time step too "
                    "long, a bad contact, or a defect of mdir)\n");
  else
    fprintf(stderr,
            "mdir: %lld positions are not numbers at a build of the neighbor "
            "structure; the run has failed (a time step too long, a bad "
            "contact, or a defect of mdir)\n",
            (long long)count);
  exit(1);
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

/*===----------------------------------------------------------------------===
 * FFT of particle mesh Ewald on the host
 *===----------------------------------------------------------------------===*/

#include "pocketfft.h"

/* The descriptor of a buffer of one dimension of f64, as the C interface
   of MLIR passes it. */
typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} Buffer1D;

/* The plans of the lengths that a run uses, kept for the run. */
enum { numPlans = 32 };
static struct {
  size_t length;
  rfft_plan real;
  cfft_plan complex;
} plans[numPlans];

static rfft_plan getRealPlan(size_t length) {
  for (int i = 0; i != numPlans; ++i) {
    if (plans[i].length == length && plans[i].real)
      return plans[i].real;
    if (plans[i].length == 0 || (plans[i].length == length)) {
      plans[i].length = length;
      plans[i].real = make_rfft_plan(length);
      return plans[i].real;
    }
  }
  fprintf(stderr, "mdrt: too many lengths of FFT\n");
  abort();
}

static cfft_plan getComplexPlan(size_t length) {
  for (int i = 0; i != numPlans; ++i) {
    if (plans[i].length == length && plans[i].complex)
      return plans[i].complex;
    if (plans[i].length == 0 || (plans[i].length == length)) {
      plans[i].length = length;
      plans[i].complex = make_cfft_plan(length);
      return plans[i].complex;
    }
  }
  fprintf(stderr, "mdrt: too many lengths of FFT\n");
  abort();
}

/* Transforms of the complex lines along the first two dimensions of the
   half-complex grid `c` of k1 x k2 x h complex numbers, interleaved. */
static void transformColumns(double *c, int64_t k1, int64_t k2, int64_t h,
                             int forward) {
  int64_t longest = k1 > k2 ? k1 : k2;
  double *line = (double *)malloc(2 * (size_t)longest * sizeof(double));
  cfft_plan second = getComplexPlan((size_t)k2);
  for (int64_t a = 0; a != k1; ++a)
    for (int64_t z = 0; z != h; ++z) {
      for (int64_t b = 0; b != k2; ++b) {
        line[2 * b] = c[2 * ((a * k2 + b) * h + z)];
        line[2 * b + 1] = c[2 * ((a * k2 + b) * h + z) + 1];
      }
      if (forward)
        cfft_forward(second, line, 1.0);
      else
        cfft_backward(second, line, 1.0);
      for (int64_t b = 0; b != k2; ++b) {
        c[2 * ((a * k2 + b) * h + z)] = line[2 * b];
        c[2 * ((a * k2 + b) * h + z) + 1] = line[2 * b + 1];
      }
    }
  cfft_plan first = getComplexPlan((size_t)k1);
  for (int64_t b = 0; b != k2; ++b)
    for (int64_t z = 0; z != h; ++z) {
      for (int64_t a = 0; a != k1; ++a) {
        line[2 * a] = c[2 * ((a * k2 + b) * h + z)];
        line[2 * a + 1] = c[2 * ((a * k2 + b) * h + z) + 1];
      }
      if (forward)
        cfft_forward(first, line, 1.0);
      else
        cfft_backward(first, line, 1.0);
      for (int64_t a = 0; a != k1; ++a) {
        c[2 * ((a * k2 + b) * h + z)] = line[2 * a];
        c[2 * ((a * k2 + b) * h + z) + 1] = line[2 * a + 1];
      }
    }
  free(line);
}

/* The forward transform, exp(−2π i k·m / K), of the real grid `real` of
   k1 x k2 x k3 points into the half-complex grid `complex` of
   k1 x k2 x (k3 / 2 + 1) numbers, interleaved. */
void _mlir_ciface_mdrtFFTForward3D(Buffer1D *real, Buffer1D *complex,
                                   int64_t k1, int64_t k2, int64_t k3) {
  const double *r = real->aligned + real->offset;
  double *c = complex->aligned + complex->offset;
  int64_t h = k3 / 2 + 1;
  rfft_plan plan = getRealPlan((size_t)k3);
  double *line = (double *)malloc((size_t)k3 * sizeof(double));
  for (int64_t row = 0; row != k1 * k2; ++row) {
    for (int64_t z = 0; z != k3; ++z)
      line[z] = r[row * k3 + z];
    rfft_forward(plan, line, 1.0);
    /* From r0, r1, i1, r2, i2, ... to pairs. */
    double *out = c + 2 * row * h;
    out[0] = line[0];
    out[1] = 0.0;
    for (int64_t z = 1; z != h; ++z) {
      out[2 * z] = line[2 * z - 1];
      out[2 * z + 1] = 2 * z < k3 ? line[2 * z] : 0.0;
    }
  }
  free(line);
  transformColumns(c, k1, k2, h, /*forward=*/1);
}

/* The backward transform, exp(+2π i k·m / K), not normalized, of the
   half-complex grid `complex`, which it overwrites, into `real`. */
void _mlir_ciface_mdrtFFTBackward3D(Buffer1D *complex, Buffer1D *real,
                                    int64_t k1, int64_t k2, int64_t k3) {
  double *c = complex->aligned + complex->offset;
  double *r = real->aligned + real->offset;
  int64_t h = k3 / 2 + 1;
  transformColumns(c, k1, k2, h, /*forward=*/0);
  rfft_plan plan = getRealPlan((size_t)k3);
  double *line = (double *)malloc((size_t)k3 * sizeof(double));
  for (int64_t row = 0; row != k1 * k2; ++row) {
    const double *in = c + 2 * row * h;
    line[0] = in[0];
    for (int64_t z = 1; z != h; ++z) {
      line[2 * z - 1] = in[2 * z];
      if (2 * z < k3)
        line[2 * z] = in[2 * z + 1];
    }
    rfft_backward(plan, line, 1.0);
    for (int64_t z = 0; z != k3; ++z)
      r[row * k3 + z] = line[z];
  }
  free(line);
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

int64_t mdrtHostMatrixCreate(int64_t rows, int64_t width) {
  struct HostMatrix *matrix = calloc(1, sizeof(struct HostMatrix));
  if (!matrix) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  matrix->rows = rows > 0 ? rows : 1;
  matrix->width = width > 0 ? width : 1;
  allocateHostMatrix(matrix);
  return (int64_t)(intptr_t)matrix;
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

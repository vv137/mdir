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

/* Called when a particle has more neighbors than a row of the neighbor
   matrix holds. The run cannot continue: pairs would be missed. */
void mdrtReportNeighborOverflow(int64_t needed, int64_t width) {
  fprintf(stderr,
          "mdrt: a particle has %lld neighbors, but the neighbor structure "
          "holds %lld per particle\n",
          (long long)needed, (long long)width);
  fflush(stderr);
  abort();
}

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

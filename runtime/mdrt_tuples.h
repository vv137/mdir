/* The structures of the tuples of a topology that an embedding program keeps
   from one activation of its entry to the next (D[activation-start], #294).

   When the particles are put in an order, the members of every tuple set
   are renumbered to their places and the incidence structure of the set is
   built from them: for every particle a row with the number of its tuples
   and, for each, the number of the tuple, the place of the particle in it,
   and its members (md_exec.build_incidence). Both are functions of two
   inputs alone: the members in the order of the files, and the particle at
   each place (the identities). A start of an activation that orders the
   particles as the last one did (an update of the tunables, a commit of a
   borrow, an evaluation, a checkpoint, where the particles have not moved)
   built the same structures again and, on a device, uploaded them again.

   A store keeps them for the program that opens it. A kept structure is
   used only where both of its inputs are equal to those given, element for
   element (kept copies are compared, nothing is inferred from what the host
   did since), so that a kept structure that is used is the one that would
   be built. With MDRT_CHECK_KEPT_TUPLES set, every structure that is taken
   from the store is also built anew and compared, and a difference ends the
   program.

   A structure that is not found replaces one of the same tuple set that
   the current activation has not taken, in its memory, which has the same
   size: a start in a new order allocates nothing.

   This header holds the store and the builds on the host; each runtime
   library has its own copy (the host's in mdrt.c, the device's in
   mdrt_cuda.c, where the incidence structure and a copy of the members are
   on the device) and exports its own functions. A program that opens no
   store (`mdir run`) gets structures of its own, which it frees. */
#ifndef MDRT_TUPLES_H
#define MDRT_TUPLES_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Memrefs of ranks 1 and 2 as the C interface of MLIR passes them. */
struct MdrtMemref1 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

struct MdrtMemref2 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

/* The first element of a buffer of members, which is contiguous. */
static const int32_t *mdrtTuplesMembersOf(const struct MdrtMemref2 *buffer) {
  if (buffer->sizes[0] > 0 &&
      (buffer->strides[1] != 1 || buffer->strides[0] != buffer->sizes[1])) {
    fprintf(stderr, "mdrt: the members of a tuple set are not contiguous; "
                    "this is a defect of mdir\n");
    abort();
  }
  return (const int32_t *)buffer->aligned + buffer->offset;
}

static void mdrtTuplesSetMemref2(struct MdrtMemref2 *result, void *memory,
                                 int64_t rows, int64_t columns) {
  result->allocated = memory;
  result->aligned = memory;
  result->offset = 0;
  result->sizes[0] = rows;
  result->sizes[1] = columns;
  result->strides[0] = columns;
  result->strides[1] = 1;
}

struct MdrtTuples {
  /* What the structures are a function of. */
  int64_t tuples, arity, particles;
  int32_t *file; /* the members in the order of the files, a copy */
  int32_t *ids;  /* the particle at each place, a copy; null: that order */
  /* The members at their places; null for the order of the files, where
     they are the members given. */
  int32_t *members;
  /* The incidence structure: its row width, 0 before it is built, and where
     it is, on the host or on the device. */
  int64_t width;
  int32_t *incidence;
  void *deviceIncidence;
  /* The copy of the members on the device, for the loops that evaluate
     each tuple once. */
  void *deviceMembers;
  int hasIncidence, hasDeviceMembers;
  /* How many of its buffers the code of the activation holds: taken and
     not given back. An entry that is held is not built anew. */
  int64_t held;
  /* The buffer of the members that the activation gave for the order of
     the files. */
  const int32_t *given;
};

struct MdrtTupleStore {
  struct MdrtTuples **entries;
  size_t count, room;
  /* Whether a structure whose inputs are those given is taken as it is. A
     store that does not (for the tests, which compare with it) builds
     every structure anew, in the memory of the last. */
  int keeps;
  /* Structures taken from the store, and built, and the time that the
     code has spent in the functions that give them. */
  int64_t kept, built;
  double seconds;
  /* On a device: the memory of the host in which the rows of a structure
     are written, a range of particles at a time, before their upload; of a
     bounded size (mdrt_cuda.c). */
  int32_t *staging;
  size_t stagingBytes;
};

static double mdrtTuplesNow(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (double)time.tv_sec + 1.0e-9 * (double)time.tv_nsec;
}

/* Adds the time since `start` to that of `store`, if there is one. */
static void mdrtTuplesSpent(struct MdrtTupleStore *store, double start) {
  if (store)
    store->seconds += mdrtTuplesNow() - start;
}

static void *mdrtTuplesAllocate(size_t bytes) {
  void *memory = malloc(bytes ? bytes : 1);
  if (!memory) {
    fprintf(stderr, "mdrt: out of memory of the host for %zu bytes of the "
                    "tuples\n",
            bytes);
    abort();
  }
  return memory;
}

static int mdrtTuplesChecks(void) {
  static int checks = -1;
  if (checks < 0)
    checks = getenv("MDRT_CHECK_KEPT_TUPLES") != NULL;
  return checks;
}

static void mdrtTuplesDiffer(const char *what) {
  fprintf(stderr,
          "mdrt: %s that was kept from an earlier activation differs from "
          "the one built anew; this is a defect of mdir\n",
          what);
  fflush(stderr);
  abort();
}

/* The members at their places: member' = the place p with ids[p] = member
   (md_exec.renumber). */
static void mdrtTuplesRenumber(const int32_t *file, int64_t tuples,
                               int64_t arity, const int32_t *ids,
                               int64_t particles, int32_t *members) {
  int32_t *place = mdrtTuplesAllocate((size_t)particles * sizeof(int32_t));
  for (int64_t p = 0; p != particles; ++p)
    place[ids[p]] = (int32_t)p;
  for (int64_t i = 0, e = tuples * arity; i != e; ++i)
    members[i] = place[file[i]];
  free(place);
}

/* The width of a row of the incidence structure: one number, and for each
   tuple of the particle with the most tuples its number, the place of the
   particle in it, and its members. It does not depend on the order of the
   particles. */
static int64_t mdrtTuplesWidth(const int32_t *members, int64_t tuples,
                               int64_t arity, int64_t particles) {
  int32_t *counts = calloc(particles ? (size_t)particles : 1, sizeof(int32_t));
  if (!counts) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  for (int64_t i = 0, e = tuples * arity; i != e; ++i)
    ++counts[members[i]];
  int64_t widest = 0;
  for (int64_t p = 0; p != particles; ++p)
    if (counts[p] > widest)
      widest = counts[p];
  free(counts);
  return widest * (arity + 2) + 1;
}

/* The tuples of each particle, in the order of the tuples: for the particle
   p, `entries` holds from 2 first[p] to 2 first[p + 1] the number of each of
   its tuples and its place in it. The rows of an incidence structure are
   written from it one after another, a range of particles at a time, where
   writing them tuple by tuple would visit the rows of a structure of
   hundreds of megabytes in the order of the members (#294). */
struct MdrtTupleIndex {
  int64_t *first;
  int32_t *entries;
};

static void mdrtTupleIndexBuild(struct MdrtTupleIndex *index,
                                const int32_t *members, int64_t tuples,
                                int64_t arity, int64_t particles) {
  index->first = calloc((size_t)particles + 2, sizeof(int64_t));
  if (!index->first) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  index->entries =
      mdrtTuplesAllocate((size_t)(2 * tuples * arity) * sizeof(int32_t));
  /* The count of the particle p at first[p + 2], so that the sums that
     follow leave at first[p + 1] where the next entry of p goes, and, once
     every entry is placed, where those of p + 1 begin. */
  int64_t *first = index->first;
  for (int64_t i = 0, e = tuples * arity; i != e; ++i)
    ++first[members[i] + 2];
  for (int64_t p = 0; p != particles; ++p)
    first[p + 2] += first[p + 1];
  for (int64_t t = 0; t != tuples; ++t)
    for (int64_t s = 0; s != arity; ++s) {
      int64_t at = first[members[t * arity + s] + 1]++;
      index->entries[2 * at] = (int32_t)t;
      index->entries[2 * at + 1] = (int32_t)s;
    }
}

static void mdrtTupleIndexFree(struct MdrtTupleIndex *index) {
  free(index->first);
  free(index->entries);
}

/* The rows of the particles from `begin` to `end`, at `rows`: the number of
   tuples of the particle and, for each in the order of the tuples, its
   number, the place of the particle in it, and its members. Of a row, the
   entries after the tuples of its particle are not written. */
static void mdrtTuplesRows(const struct MdrtTupleIndex *index,
                           const int32_t *members, int64_t arity,
                           int64_t width, int64_t begin, int64_t end,
                           int32_t *rows) {
  for (int64_t p = begin; p != end; ++p) {
    int32_t *row = rows + (p - begin) * width;
    int64_t from = index->first[p], count = index->first[p + 1] - from;
    row[0] = (int32_t)count;
    int32_t *at = row + 1;
    for (int64_t k = 0; k != count; ++k, at += arity + 2) {
      int32_t tuple = index->entries[2 * (from + k)];
      at[0] = tuple;
      at[1] = index->entries[2 * (from + k) + 1];
      memcpy(at + 2, members + (int64_t)tuple * arity,
             (size_t)arity * sizeof(int32_t));
    }
  }
}

/* The whole structure on the host. */
static void mdrtTuplesFill(const int32_t *members, int64_t tuples,
                           int64_t arity, int64_t particles, int64_t width,
                           int32_t *incidence) {
  struct MdrtTupleIndex index;
  mdrtTupleIndexBuild(&index, members, tuples, arity, particles);
  mdrtTuplesRows(&index, members, arity, width, 0, particles, incidence);
  mdrtTupleIndexFree(&index);
}

/* Whether two incidence structures hold the same rows, in what a loop over
   tuples reads of them. */
static int mdrtTuplesSameRows(const int32_t *a, const int32_t *b,
                              int64_t arity, int64_t particles,
                              int64_t width) {
  for (int64_t p = 0; p != particles; ++p) {
    const int32_t *x = a + p * width, *y = b + p * width;
    if (x[0] != y[0] || x[0] < 0 || 1 + (int64_t)x[0] * (arity + 2) > width)
      return 0;
    if (memcmp(x, y, (size_t)(1 + (int64_t)x[0] * (arity + 2)) *
                         sizeof(int32_t)) != 0)
      return 0;
  }
  return 1;
}

static struct MdrtTupleStore *mdrtTupleStoreOpen(void) {
  struct MdrtTupleStore *store = calloc(1, sizeof(struct MdrtTupleStore));
  if (!store) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  store->keeps = 1;
  return store;
}

static int mdrtTuplesSameInputs(const struct MdrtTuples *e,
                                const int32_t *file, int64_t tuples,
                                int64_t arity, const int32_t *ids,
                                int64_t particles) {
  if (e->tuples != tuples || e->arity != arity || e->particles != particles ||
      (e->ids != NULL) != (ids != NULL))
    return 0;
  if (ids && memcmp(e->ids, ids, (size_t)particles * sizeof(int32_t)) != 0)
    return 0;
  return memcmp(e->file, file,
                (size_t)(tuples * arity) * sizeof(int32_t)) == 0;
}

/* An activation begins: the one before it has ended, without giving back
   what it held, and its buffers of the host are gone. */
static void mdrtTupleStoreBegin(struct MdrtTupleStore *store) {
  for (size_t i = 0; i != store->count; ++i) {
    store->entries[i]->held = 0;
    store->entries[i]->given = NULL;
  }
}

/* The structures of the members `file` under the order `ids` (null: the
   order of the files), kept or, if `*found` is set to 0, to be built: then
   the entry is one whose structures are not valid, in the memory of an
   entry of the same tuple set that the activation does not hold, or new.
   `particles` is the number of particles. The caller counts what it hands
   out of the entry in `held`. */
static struct MdrtTuples *mdrtTupleStoreTake(struct MdrtTupleStore *store,
                                             const int32_t *file,
                                             int64_t tuples, int64_t arity,
                                             const int32_t *ids,
                                             int64_t particles, int *found) {
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    /* What the code holds is taken as it is in any store: it was built
       for the inputs that are compared here. */
    if ((!store->keeps && e->held == 0) ||
        !mdrtTuplesSameInputs(e, file, tuples, arity, ids, particles))
      continue;
    ++store->kept;
    *found = 1;
    return e;
  }
  *found = 0;
  ++store->built;
  size_t bytes = (size_t)(tuples * arity) * sizeof(int32_t);
  /* An entry of the same members, in the same kind of order, that no code
     that runs can hold: its memory has the sizes that are needed. */
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    if (e->held != 0 || e->tuples != tuples ||
        e->arity != arity || e->particles != particles ||
        (e->ids != NULL) != (ids != NULL) ||
        memcmp(e->file, file, bytes) != 0)
      continue;
    if (ids)
      memcpy(e->ids, ids, (size_t)particles * sizeof(int32_t));
    e->hasIncidence = e->hasDeviceMembers = 0;
    e->given = NULL;
    return e;
  }
  struct MdrtTuples *e = calloc(1, sizeof(struct MdrtTuples));
  if (!e) {
    fprintf(stderr, "mdrt: out of memory of the host\n");
    abort();
  }
  e->tuples = tuples;
  e->arity = arity;
  e->particles = particles;
  e->file = mdrtTuplesAllocate(bytes);
  memcpy(e->file, file, bytes);
  if (ids) {
    e->ids = mdrtTuplesAllocate((size_t)particles * sizeof(int32_t));
    memcpy(e->ids, ids, (size_t)particles * sizeof(int32_t));
    e->members = mdrtTuplesAllocate(bytes);
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
  return e;
}

/* The entry whose renumbered members, or whose members given for the order
   of the files by the activation that runs, are at `members`; or null. */
static struct MdrtTuples *mdrtTupleStoreFind(struct MdrtTupleStore *store,
                                             const int32_t *members) {
  if (!store || !members)
    return NULL;
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    if (e->members == members || e->given == members)
      return e;
  }
  return NULL;
}

/* Takes back a buffer that the code held: returns 1 if `pointer` is memory
   of the store, which stays, and 0 if it is the caller's to free. */
static int mdrtTupleStoreGiveBack(struct MdrtTupleStore *store,
                                  const void *pointer) {
  if (!store || !pointer)
    return 0;
  for (size_t i = 0; i != store->count; ++i) {
    struct MdrtTuples *e = store->entries[i];
    if (e->members != pointer && e->incidence != pointer &&
        e->deviceIncidence != pointer && e->deviceMembers != pointer)
      continue;
    if (e->held > 0)
      --e->held;
    return 1;
  }
  return 0;
}

#endif /* MDRT_TUPLES_H */

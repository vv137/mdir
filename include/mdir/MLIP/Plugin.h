/*===- Plugin.h - The C boundary of a plugin of learned potentials -------===*
 *
 * The functions that MDIR's core calls in an optional plugin that evaluates a
 * learned potential (D270, docs/metatomic-ml1.md, Section 6.1).
 * The core is C++17 without RTTI and exceptions and never links an external
 * framework; the plugin (C++20, libtorch) is loaded with dlopen and reached
 * through these functions alone. No exception crosses them: a function
 * returns MDIR_MLIP_OK or MDIR_MLIP_ERROR, and mdir_mlip_last_error() says
 * why.
 *
 * Units are MDIR's: nm, kJ/mol, kJ/mol/nm. Device buffers are addresses of the
 * device in the CUDA context that the core uses (the primary context); the
 * core owns them and the stream, the plugin owns its allocator's memory and
 * the model's intermediates. A call returns with its work enqueued on the
 * stream that it is given, without waiting for it, except where it says so.
 *
 *===----------------------------------------------------------------------===*/

#ifndef MDIR_MLIP_PLUGIN_H
#define MDIR_MLIP_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this boundary; a plugin of another version is refused. */
#define MDIR_MLIP_ABI_VERSION 1

#define MDIR_MLIP_OK 0
#define MDIR_MLIP_ERROR 1

typedef struct mdir_mlip_model mdir_mlip_model;

/* What a model is loaded with. */
typedef struct mdir_mlip_load_options {
  /* The file of the model. */
  const char *path;
  /* The loader: "METATOMIC", a TorchScript artifact of metatomic. */
  const char *backend;
  /* The CUDA device, as the CUDA runtime numbers the visible devices. */
  int32_t device;
  /* The centers per call that the engine asks for, 0 for the whole system;
     honored only by a model that declares that it is local. */
  int64_t centers_per_call;
} mdir_mlip_load_options;

/* What a loaded model is. The strings live as long as the model. */
typedef struct mdir_mlip_description {
  /* The cutoff of the pair list that the model asks for, in nm, and whether
     the list holds each pair in both directions and only pairs within it. */
  double cutoff;
  int32_t full_list;
  int32_t strict;
  /* The type of the model's inputs: 32 or 64 bits. */
  int32_t dtype_bits;
  /* Whether the model declares that the energy of a selection of centers
     takes only their own pairs (extra metadata `mdir::local_centers`), and
     the centers per call that the plugin then uses (0: all in one call). */
  int32_t local_centers;
  int64_t centers_per_call;
  /* The range of the interactions in nm, negative if infinite. */
  double interaction_range;
  /* The atomic numbers that the model takes. */
  int64_t num_types;
  const int64_t *types;
  /* The model's own units and the factors to MDIR's (value in MDIR's unit =
     factor * value in the model's). */
  const char *length_unit;
  const char *energy_unit;
  double length_factor;
  double energy_factor;
  /* The name of the model, as its metadata gives it, and the versions of
     the framework that the plugin was built with. */
  const char *name;
  const char *framework;
} mdir_mlip_description;

/* One evaluation. */
typedef struct mdir_mlip_inputs {
  /* The number of particles, and on the device: their positions (n x 3, nm,
     64 bits if `position_bits` is 64, else 32), and their atomic numbers
     (n, int32), in the order that the engine keeps the particles in. */
  int64_t count;
  const void *positions;
  int32_t position_bits;
  const int32_t *numbers;
  /* The cell on the host: 9 numbers, row by row the three edge vectors, in
     nm, and whether each direction is periodic. */
  double cell[9];
  int32_t periodic[3];
  /* The CUstream that the work goes to. */
  void *stream;
} mdir_mlip_inputs;

typedef struct mdir_mlip_outputs {
  /* On the device: the forces, n x 3, of `force_bits` bits (64 or 32), the
     engine's own buffer of the forces of its other terms, which the model's
     forces are added to; the energy (1) and the virial W = -dE/d(strain)
     (9, row by row), 64 bits, which are written. */
  void *forces;
  int32_t force_bits;
  double *energy;
  double *virial;
  /* On the host, written: the pairs of the list and the calls of the
     model that this evaluation made. */
  int64_t pairs;
  int64_t calls;
} mdir_mlip_outputs;

/* The memory of the device that the plugin's allocator holds, in bytes. */
typedef struct mdir_mlip_memory {
  int64_t allocated;
  int64_t peak_allocated;
  int64_t reserved;
  int64_t peak_reserved;
} mdir_mlip_memory;

int32_t mdir_mlip_abi_version(void);
/* The version of the framework the plugin was built against, as
   "torch 2.14.1, metatomic-torch 0.1.19", and that of libtorch alone. */
const char *mdir_mlip_build_versions(void);
const char *mdir_mlip_torch_version(void);
const char *mdir_mlip_last_error(void);

int mdir_mlip_load(const mdir_mlip_load_options *options,
                   mdir_mlip_model **model);
int mdir_mlip_describe(const mdir_mlip_model *model,
                       mdir_mlip_description *description);
/* Builds the pair list on the device, calls the model (in chunks of centers
   where it is local and the engine asks for them), differentiates, and adds
   the forces to the engine's. Waits for the device only for the number of
   pairs. */
int mdir_mlip_evaluate(mdir_mlip_model *model, const mdir_mlip_inputs *inputs,
                       mdir_mlip_outputs *outputs);
/* The memory of the plugin's allocator; `reset` starts the peaks anew. */
int mdir_mlip_memory_stats(mdir_mlip_model *model, int32_t reset,
                           mdir_mlip_memory *memory);
void mdir_mlip_free(mdir_mlip_model *model);

#ifdef __cplusplus
}
#endif

#endif /* MDIR_MLIP_PLUGIN_H */

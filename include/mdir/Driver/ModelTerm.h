//===- ModelTerm.h - Learned potentials through the plugin ------*- C++ -*-===//
//
// The terms of [[energy.model]] (D270, docs/metatomic-ml1.md):
// the core's side of the plugin of include/mdir/MLIP/Plugin.h, which it loads
// with dlopen when a run has such a term. The core neither includes the
// headers of the framework nor links it.
//
//===----------------------------------------------------------------------===//

#ifndef MDIR_DRIVER_MODELTERM_H
#define MDIR_DRIVER_MODELTERM_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"
#include "llvm/Support/Error.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

struct mdir_mlip_model;

namespace mdir {
namespace driver {

/// The path of the plugin: `MDIR_MLIP_PLUGIN`, or libmdir_metatomic.so in
/// `directory`, the directory of the runtime libraries.
std::string getPluginPath(llvm::StringRef directory);

/// The version of libtorch that the plugin at `path` was built against,
/// which loads it.
llvm::Expected<std::string> getPluginTorchVersion(llvm::StringRef path);

/// The directory of the runtime libraries, where the plugin is unless
/// `MDIR_MLIP_PLUGIN` names it, for getModelCutoff: set by `mdir run` and
/// by the compiler of a Python simulation before they build a program.
void setPluginDirectory(llvm::StringRef directory);

/// The cutoff of the pair list of the model of `term`, in nm, which the
/// plugin reads from the model's file without a device (D[mlip-pairlist]):
/// the program of a run with the term is built with it.
llvm::Expected<double> getModelCutoff(const Control::ModelTerm &term);

/// How the plugin takes the pair list (`MDIR_MLIP_PAIRS`, D[mlip-pairlist],
/// docs/environment.md): MDIR's list of the pairs within the cutoff
/// ("exact", the default), every slot of MDIR's list ("padded"), or the
/// plugin's own search ("search"), as the constants MDIR_MLIP_PAIRS_* of
/// include/mdir/MLIP/Plugin.h. An unknown value is an error.
llvm::Expected<int> getPairsMode();

/// The models of one simulation, loaded on its device. A step evaluates
/// each with evaluate(), which adds its forces to those of the program; the
/// energy and the virial of the last evaluation stay on the device until the
/// program or the log reads them (readLast(), read()).
class ModelTerms {
public:
  /// Loads the plugin (getPluginPath(`directory`)) and the models of
  /// `control` on the device of MDIR's runtime for NVIDIA GPUs, which the
  /// program of the run has loaded, for the particles of `system`, whose
  /// elements the models must take.
  static llvm::Expected<std::unique_ptr<ModelTerms>>
  load(const Control &control, const System &system,
       llvm::StringRef directory);
  ~ModelTerms();

  /// Evaluates model `index` on the positions (device, n x 3, of `bits`
  /// bits) and atomic numbers (device, n) of the particles in the order of
  /// the program, in the cell `cell` (9 numbers, the edge vectors row by
  /// row), on the stream of its kernels, and adds its forces to `forces`
  /// (device, n x 3, of `forceBits` bits). Returns an empty string or what
  /// failed.
  /// `slots`, `pairCounts`, and `pairOffsets` (device) are MDIR's pair
  /// list (D[mlip-pairlist]) of `numSlots` slots from the build that
  /// `builds` counts, or null where the plugin searches its own.
  std::string evaluate(int64_t index, const void *positions, int bits,
                       const int32_t *numbers, int64_t count,
                       const double cell[9], void *forces, int forceBits,
                       const int32_t *slots, int64_t numSlots,
                       const int32_t *pairCounts, const int32_t *pairOffsets,
                       int64_t builds);
  /// The energy and the virial (row by row) of the last evaluation of
  /// `index`, read from the device, which the host waits for.
  void readLast(int64_t index, double &energy, double virial[9]) const;
  int64_t getCount() const { return count; }

  /// The energy and the virial tensor (row by row) of the last evaluations
  /// of all the models, in kJ/mol, read from the device.
  void read(double &energy, std::array<double, 9> &virial) const;
  /// What the log says of the models at the start.
  std::string describe() const;
  /// The peak memory of the plugin's allocator and the pairs of the last
  /// evaluation, for the log at the end.
  std::string describeUse() const;

  /// The names of the models, for the terms at the start.
  std::vector<std::string> getNames() const;

private:
  ModelTerms() = default;
  struct Model {
    std::string name;
    mdir_mlip_model *handle = nullptr;
    std::string description;
    /// Device: the energy and the virial (10).
    double *results = nullptr;
    int64_t pairs = 0, calls = 0, evaluations = 0;
  };
  std::vector<Model> models;
  int64_t count = 0;
  /// How the plugin takes the pair list (getPairsMode).
  int pairsMode = 0;
};

} // namespace driver
} // namespace mdir

/// The functions that the program of a run with a model term calls
/// (D270): model `index` at the positions, the atomic
/// numbers, and the cell (its diagonal and tilts b_x, c_x, c_y), its forces
/// added to `forces`; with `Energy`, its energy and virial written to
/// `energy` (1) and `virial` (9). With `Edges`, it takes MDIR's pair list
/// (D[mlip-pairlist]): the slots, the counts, the offsets, and the
/// number of the build of the slots of `md_exec.build_edges`.
extern "C" {
void _mlir_ciface_mdrtModelEvaluateEdges(
    int64_t index, void *positions, void *numbers, void *slots, void *counts,
    void *offsets, int64_t builds, double lx, double ly, double lz,
    double bx, double cx, double cy, int64_t bits, int64_t forceBits,
    void *forces);
void _mlir_ciface_mdrtModelEvaluateEdgesEnergy(
    int64_t index, void *positions, void *numbers, void *slots, void *counts,
    void *offsets, int64_t builds, double lx, double ly, double lz,
    double bx, double cx, double cy, int64_t bits, int64_t forceBits,
    void *forces, void *energy, void *virial);
void _mlir_ciface_mdrtModelEvaluate(int64_t index, void *positions,
                                    void *numbers, double lx, double ly,
                                    double lz, double bx, double cx,
                                    double cy, int64_t bits,
                                    int64_t forceBits, void *forces);
void _mlir_ciface_mdrtModelEvaluateEnergy(
    int64_t index, void *positions, void *numbers, double lx, double ly,
    double lz, double bx, double cx, double cy, int64_t bits,
    int64_t forceBits, void *forces, void *energy, void *virial);
}

#endif // MDIR_DRIVER_MODELTERM_H

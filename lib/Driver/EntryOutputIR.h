// Owned output configuration and actual values at native entry rows.
#ifndef MDIR_LIB_DRIVER_ENTRYOUTPUTIR_H
#define MDIR_LIB_DRIVER_ENTRYOUTPUTIR_H
#include "OutputIR.h"
namespace mdir::driver::builder {
struct EntryGradientProductIR {
  // Ordinals in the frame boundary's request buffer and returned buffers.
  unsigned energyCotangentRequest, positionCotangentResult;
  // Whether the product entry also gives the derivative of the product in
  // the positions (#771, D383).
  bool positions = false;
  // Whether the product entry gives instead forward mode in the tunables:
  // the derivative of @tunable along the directions of its slots and
  // fields, and its forces (#726, D[force-matching-forward]).
  bool direction = false;
};
struct EntryGradientOutputIR {
  TunableGradientOutputConfig output;
  unsigned request = 0, seeds = 0;
  std::vector<unsigned> constants;
  std::optional<unsigned> seedField;
  std::optional<EntryGradientProductIR> product;
};
struct EntryOutputIRConfig {
  std::optional<FreeEnergyOutputConfig> freeEnergy;
  std::vector<ObservableOutputTermIR> observables;
  size_t observableCount = 0;
  unsigned observableBaseArgument = 0;
  // Ordinals in Program.startValues, resolved from semantic control metadata.
  std::vector<unsigned> observableTunables;
  std::optional<EntryGradientOutputIR> gradient;
};
struct EntryOutputPointIR {
  const PotentialStateIR &physical;
  mlir::Value step, identities, interval, zero, one;
  mlir::ValueRange startValues;
  bool initial, timed;
};
void buildEntryOutputs(mlir::OpBuilder &, const EntryOutputPointIR &, const EntryOutputIRConfig &);
}
#endif

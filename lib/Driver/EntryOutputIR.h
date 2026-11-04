// Owned output configuration and actual values at native entry rows.
#ifndef MDIR_LIB_DRIVER_ENTRYOUTPUTIR_H
#define MDIR_LIB_DRIVER_ENTRYOUTPUTIR_H
#include "OutputIR.h"
namespace mdir::driver::builder {
struct EntryGradientOutputIR {
  TunableGradientOutputConfig output;
  unsigned request = 0, seeds = 0;
  std::vector<unsigned> constants;
  std::optional<unsigned> seedField;
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

// Preflight of the periodic TOWERS launch on the selected device.
#pragma once
#include "mdir/Driver/System.h"
#include "llvm/Support/JSON.h"

namespace mdir::tool {
/// Resolve the choice available before a timing trial. AUTO as the value
/// means an eligible input still needs the paired measurements.
/// Records in `choice` whether spans are recorded, with the overrides of
/// the environment (D448).
void addSpansChoice(llvm::json::Object &choice, bool onDevice, bool trial);
llvm::json::Object initialStepGraphChoice(llvm::StringRef request,
                                         llvm::StringRef refusal, size_t particles,
                                          bool given = true,
                                          llvm::StringRef spansRefusal = {});
/// The record of `[execution] term_streams` (D[concurrent-terms]): the
/// request, the schedule the compiler gives the force terms, and why the
/// two differ where they do.
llvm::json::Object termStreamsChoice(const driver::Control &control);
llvm::Expected<llvm::json::Object>
checkTowersShape(const driver::Control &control, const driver::System &system,
                 bool setup, llvm::StringRef checkpointHeight = {},
                 driver::TowersSetup *resolved = nullptr);
}

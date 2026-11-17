// The entries of the tuning policies that an architecture selects
// (D[tuning-policies]), printed for device-policy.test:
//
//   mdir-device-policy-test ARCH [CLASS]
//
// prints each table's values and whether its entry is a fallback (not
// measured on ARCH).
#include "mdir/Dialect/MDExec/DevicePolicy.h"

#include "llvm/Support/raw_ostream.h"

using namespace mdir::md_exec::policy;

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    llvm::errs() << "usage: mdir-device-policy-test ARCH [CLASS]\n";
    return 2;
  }
  DevicePolicy policy = getDevicePolicy(argv[1], argc == 3 ? argv[2] : "");
  llvm::outs() << "arch '" << policy.arch << "'\n"
               << "pruning fused-groups=" << policy.pruning.fusedGroups
               << " fallback=" << policy.pruningFallback << " ("
               << policy.pruningBasis.decision << ")\n"
               << "scatter tile-min-points=" << policy.scatter.tileMinPoints
               << " tile-edge=" << policy.scatter.tileEdge
               << " hash-least=" << policy.scatter.hashLeast
               << " hash-probes=" << policy.scatter.hashProbes
               << " pair-hash-least=" << policy.scatter.pairHashLeast
               << " fallback=" << policy.scatterFallback << "\n"
               << "launch block-size=" << policy.launch.blockSize
               << " row-lanes=" << policy.launch.rowLanes
               << " stage-block-size=" << policy.launch.stageBlockSize
               << " block-memory=" << policy.launch.blockMemoryBytes
               << " gather-lanes=" << policy.launch.getGatherLanes(4) << ","
               << policy.launch.getGatherLanes(6) << ","
               << policy.launch.getGatherLanes(8)
               << " fallback=" << policy.launchFallback << "\n";
  return 0;
}

#ifndef MDIR_EXPERIMENT_PHYSICAL_COVERAGE_H
#define MDIR_EXPERIMENT_PHYSICAL_COVERAGE_H
#include "PrototypeSession.h"
namespace prototype_io {
struct PhysicalCoverageResult {
  uint64_t required[3]{}, missing[3]{}, extra[3]{};
  int firstKind=-1, firstI=-1, firstJ=-1, firstImage[3]{};
};
// Diagnostic only: f64 physical site cutoff against retained INNER image bits.
// Counts are unique interacting unit-pair/image bits, not site-pair counts.
PhysicalCoverageResult verifySessionPhysical(Session*,const double *wholeCoordinatesAngstrom);
struct SessionPhysicalData {
 int waters,atoms,sites,types;
 double cell[3],cutoff,dOE,qWater[4],aOO,bOO;
 std::vector<double> charges,aOS,bOS,aSS,bSS;
 std::vector<int> atomTypes;
 const int *exRows,*exNeighbors;
};
SessionPhysicalData sessionPhysicalData(const Session*);
}
#endif

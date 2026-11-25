#ifndef MDRT_NCCL_H
#define MDRT_NCCL_H

#include <mpi.h>
#include <stdint.h>

void mdrtNCCLStart(MPI_Comm world);
void mdrtNCCLFinish(void);
int mdrtNCCLActive(void);
struct NCCLPlan;
struct NCCLPlan *mdrtNCCLPlan(int64_t domains, const int64_t *ghosts,
    int64_t capacity, const int64_t *sourceDomain, const int64_t *sourceRow,
    const int64_t *image, const double *edges);
void mdrtNCCLFreePlan(struct NCCLPlan *p);
void mdrtNCCLHalo(const struct NCCLPlan *p, double *owned, int64_t nOwned,
    double *ghosts, int k, int image, int reverse);
double *mdrtNCCLPositions(const struct NCCLPlan *plan, double *owned, int64_t rows);
void mdrtNCCLFreePositions(double *positions);
double *mdrtNCCLAllocate(size_t bytes);
void mdrtNCCLUpload(double *device, const double *host, size_t bytes);
void mdrtNCCLDownload(double *host, const double *device, size_t bytes);
void mdrtNCCLImport(double *dest, const double *device, size_t bytes);
void mdrtNCCLAdvance(const double *, const double *, const double *, double *, long long, double);
int mdrtNCCLCoverage(const double *, const double *, long long, double);
int mdrtNCCLNode(int rank);

#endif

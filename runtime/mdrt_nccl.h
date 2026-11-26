#ifndef MDRT_NCCL_H
#define MDRT_NCCL_H

#include <mpi.h>
#include <stdint.h>

void mdrtNCCLStart(MPI_Comm world);
void mdrtNCCLFinish(void);
void mdrtNCCLRowsReady(void);
int mdrtNCCLActive(void);
struct NCCLPlan;
struct NCCLPlan *mdrtNCCLSelfPlan(double *x, int64_t rows, const double *edges,
                                  double reach, int64_t *ghosts);
struct NCCLPlan *mdrtNCCLPlan(int64_t ownedRows, int64_t domains, const int64_t *ghosts,
    int64_t capacity, const int64_t *sourceDomain, const int64_t *sourceRow,
    const int64_t *image, const double *edges);
double *mdrtNCCLBeginPositions(struct NCCLPlan *, double *, int64_t);
int *mdrtNCCLPairPlaces(struct NCCLPlan *, const int *, int64_t, int, int,
                        void *, int64_t *);
int *mdrtNCCLCenterMask(struct NCCLPlan *, int64_t, int);
double *mdrtNCCLPairPositions(struct NCCLPlan *, double *, double *, int64_t, int, void *);
void mdrtNCCLWriteCenters(double *, const double *, const int *, int64_t, void *);
void mdrtNCCLFreePlan(struct NCCLPlan *p);
void mdrtNCCLResidentHalo(const struct NCCLPlan *, double *, int64_t, double *, int, int, int);
void mdrtNCCLHalo(const struct NCCLPlan *p, double *owned, int64_t nOwned,
    double *ghosts, int k, int image, int reverse);
double *mdrtNCCLPackedRows(const struct NCCLPlan *plan, double *owned, int64_t rows, int k, int image);
int64_t *mdrtNCCLFaceRecords(const double *positions, long long rows,
    int domain, const int64_t *grid, const double *length, double reach, int64_t *count);
void mdrtNCCLWrapRows(const double *source, double *target, long long rows,
                      const double *length);
void mdrtNCCLWaitForRows(void *compute);
void mdrtNCCLImportOnStream(double *, const double *, size_t, void *);
void mdrtNCCLGatherCoordinates(double *all, const double *owned, int64_t rows, int64_t stride);
void mdrtNCCLJoin(double *out, const double *owned, size_t n, const double *ghost, size_t g);
void mdrtNCCLBuildTupleRows(const int *, int, int, const int *, const int *, const double *,
    const int64_t *, const double *, int, int, const int *, const int *, int, int *, int *, int *);
int *mdrtNCCLUploadIndices(const int *rows, int64_t n);
void mdrtNCCLDownloadRows(const double *source, const int *indices, int64_t rows,
                          int width, double *host);
void mdrtNCCLReorder(const double *owned, const double *incoming, int64_t received,
                     const int *map, double *result, int64_t rows, int width);
void mdrtNCCLNarrow(const double *, void *, long long);
void mdrtNCCLFreePositions(double *positions);
double *mdrtNCCLAllocate(size_t bytes);
void mdrtNCCLUpload(double *device, const double *host, size_t bytes);
void mdrtNCCLDownload(double *host, const double *device, size_t bytes);
void mdrtNCCLImport(double *dest, const double *device, size_t bytes);
void mdrtNCCLAdvance(const double *, const double *, const double *, double *, long long, double);
int mdrtNCCLCoverage(const double *, const double *, long long, double);
int mdrtNCCLNode(int rank);

#endif

#ifndef MDRT_PENCILS_H
#define MDRT_PENCILS_H

#include <stdint.h>

/* D[distribute-pme]. Bounds use global (x,y,z) indices, with the upper
   end excluded. Each buffer is compact in z-fastest order. The final
   x pencils store pairs of real and imaginary components. */
struct PmeBox {
  int lo[3], hi[3];
};
struct PmePencils;
struct PmePencils *mdrtPmePencils(int nx, int ny, int nz, int px, int py,
                                  int order, int mpi, int mixed);
struct PmePencils *mdrtPmeSpatialPencils(int nx, int ny, int nz, int px, int py,
                                         int order, int mpi, int mixed,
                                         int64_t gx, int64_t gy, int64_t gz);
struct PmeBox mdrtPmeRealBoxAt(const struct PmePencils *mesh, int rank);
struct PmeBox mdrtPmeRealBox(const struct PmePencils *mesh);
struct PmeBox mdrtPmeSpectrumBox(const struct PmePencils *mesh);
void mdrtPmeForward(struct PmePencils *mesh, const double *real,
                    double *spectrum);
void mdrtPmeBackward(struct PmePencils *mesh, const double *spectrum,
                     double *real);
void mdrtPmeFreePencils(struct PmePencils *mesh);

#endif

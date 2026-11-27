// Device row maps of the halo incidence operator and its transpose.
#include "mdrt_mpi.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr char source[] = R"(
extern "C" __global__ void indexPlaces(const int *order, const int *slots, int *places,
                                       long long rows, long long owned) {
  long long place = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (place >= rows) return;
  int particle = order[place];
  if (particle >= 0 && particle < owned) places[slots[particle]] = place;
}
extern "C" __global__ void writeCenters(double *out, const double *forces,
                                        const int *mask, long long n) {
  long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n && mask[i/3]) out[i] = forces[i];
}
extern "C" __global__ void pack(const double *field, const int *row,
                                double *wire, long long n, int k) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) wire[i] = field[(long long)row[i/k]*k + i%k];
}
extern "C" __global__ void unpack(const double *wire, const int *row,
                                  double *field, long long n, int k,
                                  const double *shift, int transpose) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long dest = (long long)row[i/k]*k + i%k;
  if (transpose) atomicAdd(field + dest, wire[i]);
  else field[dest] = wire[i] + (shift ? shift[(i/k)*3+i%k] : 0.);
}
extern "C" __global__ void expandMigration(const int2 *spans, int n, int *map, int rows) {
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if (r>=rows) return;
  int lo=0, hi=n;
  while (lo+1<hi) {
    int mid=lo+(hi-lo)/2;
    if (spans[mid].x<=r) lo=mid; else hi=mid;
  }
  int2 s=spans[lo];
  map[r]=s.y<0 ? s.y : s.y+(r-s.x);
}
extern "C" __global__ void reorder(const double *owned, const double *incoming,
    const int *map, double *result, long long rows, int width) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows*width) return;
  int row = map[i/width];
  result[i] = row >= 0 ? owned[(long long)row*width+i%width]
                       : incoming[(-1LL-row)*width+i%width];
}
struct Box { double length[3]; int grid[3]; };
struct HaloBox { double lo[3], hi[3]; };

__device__ void extend(double *target, double value, bool lower) {
  auto *bits = reinterpret_cast<unsigned long long *>(target);
  auto old = atomicCAS(bits, 0ULL, 0ULL);
  while (lower ? value < __longlong_as_double(old) : value > __longlong_as_double(old)) {
    auto prior = atomicCAS(bits, old, __double_as_longlong(value));
    if (prior == old) return;
    old = prior;
  }
}

__device__ void routePoint(const double *source, Box cell, double *point) {
  for (int k = 0; k < 3; ++k) {
    double value = source[k], length = cell.length[k];
    point[k] = __dsub_rn(value, __dmul_rn(length, floor(__ddiv_rn(value, length))));
  }
}

extern "C" __global__ void epochBoxes(const double *x, int rows, Box cell,
    const HaloBox *grid, HaloBox *boxes, int physicalRows) {
  __shared__ HaloBox tile[27];
  const double infinity = __longlong_as_double(0x7ff0000000000000ULL);
  int t = threadIdx.x;
  if (t < 27)
    for (int k = 0; k < 3; ++k) {
      tile[t].lo[k] = infinity;
      tile[t].hi[k] = -infinity;
    }
  __syncthreads();
  int row = blockIdx.x * blockDim.x + t;
  if (row < rows) {
    const double *p = x + 3LL * row;
    double geometry[3];
    if (physicalRows) {
      routePoint(p, cell, geometry);
      p = geometry;
    }
    int key = 0;
    for (int k = 0; k < 3; ++k) {
      double delta = p[k] - 0.5 * (grid->lo[k] + grid->hi[k]);
      double half = 0.5 * cell.length[k];
      key = 3 * key + (delta < -half ? 0 : delta > half ? 2 : 1);
    }
    for (int k = 0; k < 3; ++k) {
      extend(&tile[key].lo[k], p[k], true);
      extend(&tile[key].hi[k], p[k], false);
    }
  }
  __syncthreads();
  if (t < 27 && tile[t].lo[0] <= tile[t].hi[0])
    for (int k = 0; k < 3; ++k) {
      extend(&boxes[t].lo[k], tile[t].lo[k], true);
      extend(&boxes[t].hi[k], tile[t].hi[k], false);
    }
}

__device__ unsigned haloImages(const double *x, Box cell, const HaloBox *boxes,
                               double reach, int local) {
  unsigned mask = 0;
  double reach2 = reach * reach;
  for (int b = 0; b < 27; ++b) {
    const HaloBox &box = boxes[b];
    if (box.lo[0] > box.hi[0]) continue;
    int first[3], last[3];
    for (int k = 0; k < 3; ++k) {
      first[k] = x[k] - cell.length[k] >= box.lo[k] - reach ? -1
                 : x[k] >= box.lo[k] - reach ? 0 : 1;
      last[k] = x[k] + cell.length[k] <= box.hi[k] + reach ? 1
                : x[k] <= box.hi[k] + reach ? 0 : -1;
    }
    for (int a = first[0]; a <= last[0]; ++a)
      for (int c = first[1]; c <= last[1]; ++c)
        for (int d = first[2]; d <= last[2]; ++d) {
          if (local && !a && !c && !d) continue;
          int shifts[] = {a, c, d};
          double distance[3];
          for (int k = 0; k < 3; ++k) {
            double p = __dadd_rn(x[k], __dmul_rn(shifts[k], cell.length[k]));
            distance[k] = p < box.lo[k] ? box.lo[k] - p
                          : p > box.hi[k] ? p - box.hi[k] : 0.0;
          }
          double xy = __dadd_rn(__dmul_rn(distance[0], distance[0]),
                               __dmul_rn(distance[1], distance[1]));
          if (__dadd_rn(xy, __dmul_rn(distance[2], distance[2])) < reach2)
            mask |= 1U << ((a + 1) * 9 + (c + 1) * 3 + d + 1);
        }
  }
  return mask;
}

extern "C" __global__ void epochOwners(const double *x, int rows, Box box,
    const int *members, int *owners, int *bad) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int *group = members + 4LL * row;
  int center = group[0] < 0 ? row : group[0];
  int owner = 0;
  for (int k = 2; k >= 0; --k) {
    double length = box.length[k], origin = x[3LL * center + k], sum = 0;
    if (!isfinite(x[3LL * row + k] / length)) {
      atomicExch(bad, 1);
      return;
    }
    int count = 1;
    for (int atom = 1; atom < 4 && group[atom] >= 0; ++atom) {
      double delta = x[3LL * group[atom] + k] - origin;
      double image = floor(__dadd_rn(__ddiv_rn(delta, length), 0.5));
      sum += __dsub_rn(delta, __dmul_rn(length, image));
      ++count;
    }
    double fraction = __ddiv_rn(origin + __ddiv_rn(sum, count), length);
    if (!isfinite(fraction)) {
      atomicExch(bad, 1);
      return;
    }
    int divisions = box.grid[k];
    int part = (int)floor((fraction - floor(fraction)) * divisions);
    owner = owner * divisions + max(0, min(divisions - 1, part));
  }
  owners[row] = owner;
}
__device__ unsigned faceMask(const double lower[3], const double upper[3], double reach) {
  unsigned mask=0;
  // With a cell wider than two reaches only immediate neighbors qualify.
  for (int a=(lower[0]<=reach ? -1 : 0); a<=(upper[0]<=reach ? 1 : 0); ++a)
    for (int b=(lower[1]<=reach ? -1 : 0); b<=(upper[1]<=reach ? 1 : 0); ++b)
      for (int c=(lower[2]<=reach ? -1 : 0); c<=(upper[2]<=reach ? 1 : 0); ++c) {
        if (!a && !b && !c) continue;
        double dx=a<0?lower[0]:a>0?upper[0]:0.;
        double dy=b<0?lower[1]:b>0?upper[1]:0.;
        double dz=c<0?lower[2]:c>0?upper[2]:0.;
        if (dx*dx+dy*dy+dz*dz<=reach*reach)
          mask |= 1u<<((a+1)*9+(b+1)*3+c+1);
      }
  return mask;
}

// Exclusive prefix within a block of 256 contiguous rows. Every thread
// participates, including the padding after the final particle.
__device__ int prefix(int count, int *warps) {
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  int total=count;
  for (int step=1; step<32; step*=2) {
    int earlier=__shfl_up_sync(0xffffffff,total,step);
    if (lane>=step) total+=earlier;
  }
  if (lane==31) warps[warp]=total;
  __syncthreads();
  int first=total-count;
  for (int w=0; w<warp; ++w) first+=warps[w];
  return first;
}

// One radix digit is the destination. The same stable scatter handles
// migrating rows and the periodic image masks of halo rows.
extern "C" __global__ void epochRoutes(const double *x, int rows, Box box,
    const HaloBox *boxes, double reach, const int *owners, int local, int physicalRows,
    unsigned *masks, int *tiles, const int *offsets, long long *records) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int dest = blockIdx.y;
  long long slot = (long long)dest * rows + row;
  unsigned mask = 0;
  if (row < rows) {
    if (records) mask = masks[slot];
    else if (owners) mask = dest != local && owners[row] == dest;
    else if (physicalRows && dest != local) {
      double point[3];
      routePoint(x + 3LL * row, box, point);
      // Any qualifying periodic image selects the one unshifted foreign row.
      mask = haloImages(point, box, boxes + 27LL * dest, reach, 0) ? 1U << 13 : 0;
    } else if (!physicalRows) {
      mask = haloImages(x + 3LL * row, box, boxes + 27LL * dest, reach, dest == local);
    }
  }
  __shared__ int warps[8];
  int position = prefix(mask != 0, warps);
  long long tile = (long long)dest * gridDim.x + blockIdx.x;
  if (!records) {
    if (row < rows) masks[slot] = mask;
    if (threadIdx.x == 255) tiles[tile] = position + (mask != 0);
  } else if (mask) {
    long long out = offsets[dest] + tiles[tile] + position;
    records[2 * out] = row;
    records[2 * out + 1] = ((long long)dest << 32) | mask;
  }
}

extern "C" __global__ void epochOffsets(int *tiles, int n, int *counts) {
  int dest = blockIdx.x;
  __shared__ int warps[8], chunk;
  int total = 0;
  for (int first = 0; first < n; first += 256) {
    int t = first + threadIdx.x;
    int *column = tiles + (long long)dest * n;
    int count = t < n ? column[t] : 0;
    int offset = prefix(count, warps);
    if (t < n) column[t] = total + offset;
    if (threadIdx.x == 255) chunk = offset + count;
    __syncthreads();
    total += chunk;
    __syncthreads();
  }
  if (threadIdx.x == 0) counts[dest] = total;
}

extern "C" __global__ void epochBuckets(const int *counts, int n, int *offsets) {
  __shared__ int warps[8];
  int d = threadIdx.x;
  int count = d < n ? counts[d] : 0;
  int first = prefix(count, warps);
  if (d <= n) offsets[d] = first;
}
extern "C" __global__ void applyFaces(long long *faces, const long long *records, int n) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < n) faces[records[2LL*i]] = records[2LL*i+1];
}
extern "C" __global__ void initialFaces(long long *owners, int n) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < n) owners[i] <<= 32;
}
// Two columns number local owned and ghost rows. Additional columns
// number each nonempty peer message in identity and descending bit order.
struct Column { int kind, domain, offset; unsigned mask; };
extern "C" __global__ void faceColumns(const long long *faces, int n, int local,
    const unsigned *incoming, const Column *columns, int *blocks, int stage,
    int *owned, int *ghost, unsigned *previous, int *source, int *dest,
    double *shifts, Box box) {
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  long long face = j < n ? faces[j] : -1;
  int owner = face >> 32;
  unsigned mask = (unsigned)face;
  int c = blockIdx.y + (stage == 2 ? 2 : 0);
  Column col = columns[c];
  unsigned bits = 0;
  if (col.kind == 1 && owner >= 0) bits = mask & incoming[owner];
  if (col.kind >= 2 && owner == col.domain) bits = mask & col.mask;
  int count = col.kind == 0 ? owner == local : __popc(bits);
  __shared__ int warps[8];
  int place = prefix(count,warps);
  long long block = (long long)c*gridDim.x+blockIdx.x;
  if (stage == 0) {
    if (threadIdx.x == 255) blocks[block] = place+count;
    return;
  }
  if (!count) return;
  place += blocks[block];
  if (col.kind == 0) {
    owned[j] = place;
    previous[place] = mask;
  } else if (col.kind == 1) {
    ghost[j] = place;
  } else if (col.kind == 2) {
    for (int k=0; k<count; ++k) source[place+k] = owned[j];
  } else {
    int row = ghost[j];
    while (bits) {
      int bit = 31-__clz(bits);
      bits ^= 1u<<bit;
      dest[place] = row++;
      int code = local;
      for (int k=0; k<3; ++k) {
        int rel = (k==0 ? bit/9 : k==1 ? bit/3%3 : bit%3)-1;
        int cell = code%box.grid[k]-rel;
        code /= box.grid[k];
        shifts[3LL*place+k] = ((cell >= box.grid[k])-(cell < 0))*box.length[k];
      }
      ++place;
    }
  }
}
extern "C" __global__ void prefixFaces(int *blocks, int n, const Column *columns) {
  int col = blockIdx.x, sum = columns[col].offset;
  __shared__ int warps[8], chunk;
  for (int i=0; i<n; i+=256) {
    int b = i+threadIdx.x;
    long long index = (long long)col*n+b;
    int count = b<n ? blocks[index] : 0;
    int offset = prefix(count,warps);
    if (b<n) blocks[index] = sum+offset;
    if (threadIdx.x == 255) chunk = offset+count;
    __syncthreads();
    sum += chunk;
    __syncthreads();
  }
}
// The rank-indexed coordinates are unshifted. Resolve the midpoint and
// the canonical image of each member before building particle incidences.
extern "C" __global__ void scheduleTuples(const int *members, int count, int arity,
    const int *locations, const int *particles, const double *x, Box box,
    int domain, int group, int physicalRows, int *mapped) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  int base = t * arity;
  int anchor = members[base];
  for (int a=1; a<arity; ++a) anchor=min(anchor,members[base+a]);
  int owner = locations[3*anchor+2];
  int translation[3] = {};
  if (!group) {
    owner = 0;
    for (int k=2; k>=0; --k) {
      double origin = x[3LL*locations[3*anchor]+k];
      double lo=0., hi=0., length=box.length[k];
      for (int a=0; a<arity; ++a) {
        double delta=x[3LL*locations[3*members[base+a]]+k]-origin;
        delta=__dsub_rn(delta,__dmul_rn(floor(delta/length+.5),length));
        lo=fmin(lo,delta); hi=fmax(hi,delta);
      }
      double u=__dadd_rn(origin,.5*(lo+hi))/length;
      translation[k]=(int)floor(u);
      int cell=max(0,min(box.grid[k]-1,(int)floor((u-floor(u))*box.grid[k])));
      owner=owner*box.grid[k]+cell;
    }
  }
  for (int a=0; a<arity; ++a) {
    int j=members[base+a], row=-1;
    if (group && locations[3*j+2]!=owner) {
      printf("constraint group split across domains at tuple %d\n",t+1);
      asm("trap;");
    }
    if (owner==domain && physicalRows && !group) {
      row = locations[3*j+1];
    } else if (owner==domain) {
      int image[3]={};
      if (!group)
        for (int k=0; k<3; ++k) {
          double delta=x[3LL*locations[3*j]+k]-x[3LL*locations[3*anchor]+k];
          image[k]=-(int)floor(delta/box.length[k]+.5)-translation[k];
        }
      for (row=locations[3*j+1]; row>=0; row=particles[5*row+1])
        if (particles[5*row+2]==image[0] && particles[5*row+3]==image[1] &&
            particles[5*row+4]==image[2]) break;
    }
    if (owner==domain && row<0) {
      printf("missing tuple owner coverage: tuple %d domain %d lacks atom %d\n",t+1,domain,j+1);
      asm("trap;");
    }
    mapped[base+a]=row;
  }
}

extern "C" __global__ void tupleRows(const int *particles, const int *starts,
    const int *entries, const int *mapped, int rows, int arity, int *blocks,
    int *csr, int fill) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  int begin=0,end=0,count=0;
  if (row<rows) {
    int id=particles[5*row];
    begin=starts[id]; end=starts[id+1];
    for (int i=begin;i<end;++i) count+=(mapped[entries[i]]==row);
  }
  __shared__ int warps[8];
  int offset=prefix(count,warps);
  if (!fill) {
    if (threadIdx.x==255) blocks[blockIdx.x]=offset+count;
    return;
  }
  if (row>=rows) return;
  offset+=blocks[blockIdx.x];
  csr[1+row]=offset;
  for (int i=begin;i<end;++i) {
    int m=entries[i];
    if (mapped[m]!=row) continue;
    int t=m/arity;
    int *dest=csr+rows+2+(long long)(arity+2)*offset++;
    dest[0]=t; dest[1]=m%arity;
    for (int a=0;a<arity;++a) dest[a+2]=mapped[t*arity+a];
  }
}
extern "C" __global__ void prefixTupleRows(int *blocks,int n,int *csr,int rows,int arity) {
  int sum=0;
  for (int i=0;i<n;++i) { int k=blocks[i]; blocks[i]=sum; sum+=k; }
  csr[0]=rows; csr[1+rows]=sum;
  for (int i=0;i<arity+2;++i) csr[rows+2+(long long)sum*(arity+2)+i]=0;
}
extern "C" __global__ void selfFaceCounts(const double *x, long long rows, Box box,
    double reach, unsigned *masks, int *blocks) {
  long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned mask=0;
  if (r<rows) {
    double lower[3],upper[3];
    for (int k=0; k<3; ++k) {
      lower[k]=fmax(0.,x[3*r+k]);
      upper[k]=fmax(0.,box.length[k]-x[3*r+k]);
    }
    masks[r]=mask=faceMask(lower,upper,reach);
  }
  __shared__ int warps[8];
  int count=__popc(mask);
  int first=prefix(count,warps);
  if (threadIdx.x==255) blocks[blockIdx.x]=first+count;
}
extern "C" __global__ void selfFaceFill(long long rows, Box box, const unsigned *masks,
    const int *blocks, int *source, int *dest, double *shifts) {
  long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned mask=r<rows ? masks[r] : 0;
  __shared__ int warps[8];
  int place=blocks[blockIdx.x]+prefix(__popc(mask),warps);
  while (mask) {
    int bit=31-__clz(mask);
    mask ^= 1u<<bit;
    source[place]=(int)r;
    dest[place]=place;
    shifts[3LL*place]=(1-bit/9)*box.length[0];
    shifts[3LL*place+1]=(1-(bit/3)%3)*box.length[1];
    shifts[3LL*place+2]=(1-bit%3)*box.length[2];
    ++place;
  }
}
extern "C" __global__ void selfGather(const double *owned, const int *source,
    const double *shifts, double *ghosts, long long n, int width) {
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) ghosts[i]=owned[(long long)source[i/width]*width+i%width]+(shifts ? shifts[i] : 0.);
}
extern "C" __global__ void faceRecords(const double *x, long long rows, Box box,
    int oldDomain, double reach, const unsigned *previous, int *count, long long *output) {
  long long r = (long long)blockIdx.x*blockDim.x+threadIdx.x;
  if (r >= rows) return;
  int cell[3];
  double lower[3], upper[3];
  for (int k=0; k<3; ++k) {
    double length=box.length[k], u=x[3*r+k]/length;
    if (!isfinite(u)) {
      int i=atomicAdd(count,1);
      output[2LL*i]=-1;
      output[2LL*i+1]=0;
      return;
    }
    double whole=floor(u), wrapped=__dsub_rn(x[3*r+k],__dmul_rn(length,whole));
    int grid=box.grid[k];
    cell[k]=max(0,min(grid-1,(int)floor((u-whole)*grid)));
    lower[k]=fmax(0.,wrapped-length*cell[k]/grid);
    upper[k]=fmax(0.,length*(cell[k]+1)/grid-wrapped);
  }
  int domain=cell[0]+box.grid[0]*(cell[1]+box.grid[1]*cell[2]);
  unsigned mask=faceMask(lower,upper,reach);
  if (mask != (previous ? previous[r] : 0u) || domain != oldDomain) {
    int i=atomicAdd(count,1);
    output[2LL*i]=r;
    output[2LL*i+1]=((long long)domain<<32)|mask;
  }
}
extern "C" __global__ void wrapRows(const double *source, double *target,
                                     long long rows, Box box) {
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=3*rows) return;
  double x=source[i], length=box.length[i%3];
  target[i]=__dsub_rn(x,__dmul_rn(length,floor(x/length)));
}
extern "C" __global__ void narrow(const double *source, int *target, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) target[i] = (int)source[i];
}
extern "C" __global__ void advance(const double *field, const double *slope,
    const double *mass, double *result, long long n, double dt) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3*n) return;
  double w = !mass ? dt : mass[i/3] == 0. ? 0. : dt/mass[i/3];
  result[i] = __dadd_rn(field[i], __dmul_rn(w, slope[i]));
}
extern "C" __global__ void coverage(const double *x, const double *ref,
    long long n, double limit, int *flag) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double r2 = 0.;
  for (int c = 0; c < 3; ++c) {
    double v = x[3*i+c]-ref[3*i+c];
    if (!isfinite(v)) { atomicMax(flag,2); return; }
    r2 = __dadd_rn(r2,__dmul_rn(v,v));
  }
  if (!(r2 < limit)) atomicMax(flag,1);
})";
CUmodule module;
CUfunction pack, unpack, advance, coverage, narrow, reorder, faceRecords, wrapRows;
CUfunction expandMigration, initialFaces, applyFaces, faceColumns, prefixFaces;
CUfunction selfFaceCounts, selfFaceFill, selfGather;

void check(CUresult status) {
  if (status == CUDA_SUCCESS) return;
  const char *message = "CUDA halo kernel error";
  cuGetErrorString(status, &message);
  mdrtMPIAbort(message);
}

CUfunction indexPlaces, writeCenters, scheduleTuples, tupleRows, prefixTupleRows;
CUfunction epochBoxes, epochOwners, epochRoutes, epochOffsets, epochBuckets;

void compile() {
  CUdevice device;
  int major, minor;
  check(cuCtxGetDevice(&device));
  check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  const char *options[] = {arch.c_str()};
  nvrtcProgram program;
  auto verify = [](nvrtcResult result) {
    if (result != NVRTC_SUCCESS) mdrtMPIAbort(nvrtcGetErrorString(result));
  };
  verify(nvrtcCreateProgram(&program, source, "halo.cu", 0, nullptr, nullptr));
  if (nvrtcCompileProgram(program, 1, options) != NVRTC_SUCCESS) {
    size_t size;
    verify(nvrtcGetProgramLogSize(program, &size));
    std::vector<char> log(size);
    verify(nvrtcGetProgramLog(program, log.data()));
    mdrtMPIAbort(log.data());
  }
  size_t size;
  verify(nvrtcGetPTXSize(program, &size));
  std::vector<char> ptx(size);
  verify(nvrtcGetPTX(program, ptx.data()));
  verify(nvrtcDestroyProgram(&program));
  check(cuModuleLoadData(&module, ptx.data()));
  check(cuModuleGetFunction(&scheduleTuples, module, "scheduleTuples"));
  check(cuModuleGetFunction(&tupleRows, module, "tupleRows"));
  check(cuModuleGetFunction(&prefixTupleRows, module, "prefixTupleRows"));
  check(cuModuleGetFunction(&indexPlaces, module, "indexPlaces"));
  check(cuModuleGetFunction(&writeCenters, module, "writeCenters"));
  check(cuModuleGetFunction(&epochBoxes, module, "epochBoxes"));
  check(cuModuleGetFunction(&epochOwners, module, "epochOwners"));
  check(cuModuleGetFunction(&epochRoutes, module, "epochRoutes"));
  check(cuModuleGetFunction(&epochOffsets, module, "epochOffsets"));
  check(cuModuleGetFunction(&epochBuckets, module, "epochBuckets"));
  check(cuModuleGetFunction(&pack, module, "pack"));
  check(cuModuleGetFunction(&unpack, module, "unpack"));
  check(cuModuleGetFunction(&expandMigration,module,"expandMigration"));
  check(cuModuleGetFunction(&initialFaces,module,"initialFaces"));
  check(cuModuleGetFunction(&applyFaces,module,"applyFaces"));
  check(cuModuleGetFunction(&faceColumns,module,"faceColumns"));
  check(cuModuleGetFunction(&prefixFaces,module,"prefixFaces"));
  check(cuModuleGetFunction(&selfFaceCounts,module,"selfFaceCounts"));
  check(cuModuleGetFunction(&selfFaceFill,module,"selfFaceFill"));
  check(cuModuleGetFunction(&selfGather,module,"selfGather"));
  check(cuModuleGetFunction(&faceRecords,module,"faceRecords"));
  check(cuModuleGetFunction(&wrapRows,module,"wrapRows"));
  check(cuModuleGetFunction(&reorder, module, "reorder"));
  check(cuModuleGetFunction(&narrow, module, "narrow"));
  check(cuModuleGetFunction(&advance, module, "advance"));
  check(cuModuleGetFunction(&coverage, module, "coverage"));
}
}

extern "C" void mdrtNCCLRows(int gather, const double *input, const int *rows,
                             double *output, long long n, int k,
                             const double *shifts, int transpose, CUstream stream) {
  if (!n) return;
  if (!module) compile();
  void *args[] = {&input, &rows, &output, &n, &k, &shifts, &transpose};
  check(cuLaunchKernel(gather ? pack : unpack, (n + 255)/256, 1, 1, 256, 1, 1,
                       0, stream, args, nullptr));
}

extern "C" void mdrtNCCLAdvance(const double *x, const double *v,
    const double *mass, double *result, long long rows, double dt) {
  if (!rows) return;
  if (!module) compile();
  void *args[] = {&x,&v,&mass,&result,&rows,&dt};
  check(cuLaunchKernel(advance, (3*rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
}

static CUdeviceptr coverageFlag, faceRecordBuffer;
static size_t faceRecordCapacity;

extern "C" void mdrtNCCLReleaseScratch() {
  if (faceRecordBuffer) {
    check(cuMemFree(faceRecordBuffer));
    faceRecordBuffer = 0;
    faceRecordCapacity = 0;
  }
  if (coverageFlag) {
    check(cuMemFree(coverageFlag));
    coverageFlag = 0;
  }
}

extern "C" int mdrtNCCLCoverage(const double *x, const double *ref,
                                long long rows, double limit) {
  if (!rows) return 0;
  if (!module) compile();
  if (!coverageFlag) check(cuMemAlloc(&coverageFlag,sizeof(int)));
  check(cuMemsetD32(coverageFlag,0,1));
  void *args[] = {&x,&ref,&rows,&limit,&coverageFlag};
  check(cuLaunchKernel(coverage,(rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
  int value;
  check(cuMemcpyDtoH(&value,coverageFlag,sizeof(value)));
  return value;
}

extern "C" void mdrtNCCLNarrow(const double *source, void *target, long long rows) {
  if (!rows) return;
  if (!module) compile();
  void *args[] = {&source,&target,&rows};
  check(cuLaunchKernel(narrow,(rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
  check(cuStreamSynchronize(nullptr));
}

extern "C" void mdrtNCCLExpandMigration(const void *spans, int n, int *map, int rows) {
  if (!rows) return;
  if (!module) compile();
  void *args[]={&spans,&n,&map,&rows};
  check(cuLaunchKernel(expandMigration,(rows+255LL)/256,1,1,256,1,1,0,nullptr,args,nullptr));
}

extern "C" void mdrtNCCLReorderKernel(const double *owned, const double *incoming,
    const int *map, double *result, long long rows, int width) {
  if (!rows) return;
  if (!module) compile();
  void *args[] = {&owned, &incoming, &map, &result, &rows, &width};
  check(cuLaunchKernel(reorder,(rows*width+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
}

namespace {
struct Box { double length[3]; int grid[3]; };

Box makeBox(const int64_t *grid, const double *edges) {
  Box box;
  for (int k = 0; k < 3; ++k) {
    box.grid[k] = grid[k];
    box.length[k] = edges[k];
  }
  return box;
}
}

extern "C" void mdrtNCCLEpochBoxes(const double *x, int rows, const int64_t *grid,
    const double *edges, int domain, int physicalRows, double *host) {
  constexpr int central = 13;
  const double inf = std::numeric_limits<double>::infinity();
  Box box = makeBox(grid, edges);
  for (int b = 0; b < 27; ++b) {
    std::fill_n(host + 6 * b, 3, inf);
    std::fill_n(host + 6 * b + 3, 3, -inf);
  }
  double bounds[6];
  for (int k = 0; k < 3; ++k) {
    int c = domain % box.grid[k];
    domain /= box.grid[k];
    bounds[k] = edges[k] * c / box.grid[k];
    bounds[k + 3] = edges[k] * (c + 1) / box.grid[k];
  }
  std::copy_n(bounds, 6, host + 6 * central);
  if (!rows) return;
  if (!module) compile();
  CUdeviceptr storage;
  constexpr size_t bytes = 27 * 6 * sizeof(double);
  check(cuMemAlloc(&storage, bytes + sizeof bounds));
  CUdeviceptr boundsDevice = storage + bytes;
  check(cuMemcpyHtoD(storage, host, bytes));
  check(cuMemcpyHtoD(boundsDevice, bounds, sizeof bounds));
  void *args[] = {&x, &rows, &box, &boundsDevice, &storage, &physicalRows};
  check(cuLaunchKernel(epochBoxes, (rows + 255LL) / 256, 1, 1, 256, 1, 1,
                       0, nullptr, args, nullptr));
  check(cuMemcpyDtoH(host, storage, bytes));
  check(cuMemFree(storage));
}

extern "C" int64_t *mdrtNCCLEpochRoutes(const double *x, int rows, int local,
    int domains, const int64_t *grid, const double *edges, const int *members,
    const double *boxes, double reach, int physicalRows, int64_t *size) {
  *size = 0;
  if (!rows) return nullptr;
  if (!module) compile();
  Box box = makeBox(grid, edges);
  int tilesPerRank = (rows + 255LL) / 256;
  size_t maskBytes = size_t(rows) * domains * sizeof(unsigned);
  size_t tileBytes = size_t(tilesPerRank) * domains * sizeof(int);
  size_t countBytes = size_t(domains) * sizeof(int);
  size_t auxBytes = members ? (size_t(rows) + 1) * sizeof(int)
                            : size_t(domains) * 27 * 6 * sizeof(double);
  // The box storage requires a double-aligned base after the integer columns.
  size_t listsBytes = maskBytes + tileBytes + 2 * countBytes + sizeof(int);
  size_t auxOffset = (listsBytes + 7) & ~size_t(7);
  CUdeviceptr storage;
  check(cuMemAlloc(&storage, auxOffset + auxBytes));
  CUdeviceptr masks = storage, tiles = masks + maskBytes;
  CUdeviceptr counts = tiles + tileBytes, offsets = counts + countBytes;
  CUdeviceptr deviceOwners = 0, deviceBoxes = 0;
  if (members) {
    deviceOwners = storage + auxOffset;
    CUdeviceptr bad = deviceOwners + size_t(rows) * sizeof(int);
    check(cuMemsetD32(bad, 0, 1));
    void *args[] = {&x, &rows, &box, &members, &deviceOwners, &bad};
    check(cuLaunchKernel(epochOwners, tilesPerRank, 1, 1, 256, 1, 1,
                         0, nullptr, args, nullptr));
    int invalid;
    check(cuMemcpyDtoH(&invalid, bad, sizeof invalid));
    if (invalid) mdrtMPIAbort("migration requires finite positions and images");
  } else {
    deviceBoxes = storage + auxOffset;
    check(cuMemcpyHtoD(deviceBoxes, boxes, auxBytes));
  }
  CUdeviceptr records = 0;
  void *args[] = {&x, &rows, &box, &deviceBoxes, &reach, &deviceOwners, &local,
                  &physicalRows, &masks, &tiles, &offsets, &records};
  check(cuLaunchKernel(epochRoutes, tilesPerRank, domains, 1, 256, 1, 1,
                       0, nullptr, args, nullptr));
  void *tileArgs[] = {&tiles, &tilesPerRank, &counts};
  check(cuLaunchKernel(epochOffsets, domains, 1, 1, 256, 1, 1, 0, nullptr,
                       tileArgs, nullptr));
  void *bucketArgs[] = {&counts, &domains, &offsets};
  check(cuLaunchKernel(epochBuckets, 1, 1, 1, 256, 1, 1, 0, nullptr,
                       bucketArgs, nullptr));
  int count;
  check(cuMemcpyDtoH(&count, offsets + countBytes, sizeof count));
  int64_t *host = nullptr;
  if (count) {
    size_t bytes = 2 * size_t(count) * sizeof(int64_t);
    host = static_cast<int64_t *>(std::malloc(bytes));
    if (!host) mdrtMPIAbort("out of memory collecting device epoch routes");
    check(cuMemAlloc(&records, bytes));
    check(cuLaunchKernel(epochRoutes, tilesPerRank, domains, 1, 256, 1, 1,
                         0, nullptr, args, nullptr));
    check(cuMemcpyDtoH(host, records, bytes));
    check(cuMemFree(records));
  }
  check(cuMemFree(storage));
  *size = count;
  return host;
}

extern "C" void mdrtNCCLUpdateFaceState(int64_t *faces, const int64_t *records,
                                        int n, int initial) {
  if (!n) return;
  if (!module) compile();
  void *initArgs[] = {&faces,&n};
  void *deltaArgs[] = {&faces,&records,&n};
  check(cuLaunchKernel(initial ? initialFaces : applyFaces, (n+255LL)/256,1,1,
      256,1,1,0,nullptr,initial ? initArgs : deltaArgs,nullptr));
}

extern "C" void mdrtNCCLFaceColumns(const int64_t *faces, int n, int local,
    const unsigned *incoming, const void *columns, int columnCount, int *blocks,
    int *owned, int *ghost, unsigned *previous, int *source, int *dest,
    double *shifts, const int64_t *grid, const double *edges) {
  if (!n) return;
  if (!module) compile();
  Box box;
  for (int k=0; k<3; ++k) { box.grid[k]=grid[k]; box.length[k]=edges[k]; }
  int numBlocks=(n+255LL)/256;
  for (int stage=0; stage<3; ++stage) {
    int height = stage==0 ? columnCount : stage==1 ? 2 : columnCount-2;
    if (!height) continue;
    void *args[] = {&faces,&n,&local,&incoming,&columns,&blocks,&stage,
                    &owned,&ghost,&previous,&source,&dest,&shifts,&box};
    check(cuLaunchKernel(faceColumns,numBlocks,height,1,256,1,1,0,nullptr,args,nullptr));
    if (stage==0) {
      void *scan[] = {&blocks,&numBlocks,&columns};
      check(cuLaunchKernel(prefixFaces,columnCount,1,1,256,1,1,0,nullptr,scan,nullptr));
    }
  }
}
extern "C" int64_t *mdrtNCCLFaceRecords(const double *positions, long long rows,
    int domain, const int64_t *grid, const double *length, double reach, const unsigned *previous, int64_t *size) {
  *size=0;
  if (!rows) return nullptr;
  if (!module) compile();
  Box box;
  for (int k=0; k<3; ++k) {
    box.length[k]=length[k];
    box.grid[k]=grid[k];
  }
  // The count and its records belong to one synchronous classification.
  // Reserve both together, rounding capacity up to a whole launch tile.
  if ((size_t)rows > faceRecordCapacity) {
    if (faceRecordBuffer) check(cuMemFree(faceRecordBuffer));
    faceRecordCapacity = ((size_t)rows + 255) / 256 * 256;
    check(cuMemAlloc(&faceRecordBuffer,
                     (2 * faceRecordCapacity + 1) * sizeof(int64_t)));
  }
  CUdeviceptr deviceCount = faceRecordBuffer;
  CUdeviceptr records = faceRecordBuffer + sizeof(int64_t);
  check(cuMemsetD32(deviceCount,0,1));
  void *args[]={&positions,&rows,&box,&domain,&reach,&previous,&deviceCount,&records};
  check(cuLaunchKernel(faceRecords,(rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
  int count;
  check(cuMemcpyDtoH(&count,deviceCount,sizeof count));
  size_t bytes=2*(size_t)count*sizeof(int64_t);
  auto *host=static_cast<int64_t *>(std::malloc(bytes ? bytes : sizeof(int64_t)));
  if (!host) mdrtMPIAbort("out of memory collecting face records");
  if (bytes) check(cuMemcpyDtoH(host,records,bytes));
  *size=count;
  return host;
}

extern "C" void mdrtNCCLWrapRows(const double *source, double *target, long long rows,
                                 const double *length) {
  if (!rows) return;
  if (!module) compile();
  Box box{};
  for (int k=0; k<3; ++k) box.length[k]=length[k];
  void *args[]={&source,&target,&rows,&box};
  check(cuLaunchKernel(wrapRows,(3*rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
}

extern "C" void mdrtNCCLSelfFaces(const double *x, long long rows, const double *length,
    double reach, unsigned *masks, int *blocks, int *source, int *dest, double *shifts) {
  if (!rows) return;
  if (!module) compile();
  Box box{};
  for (int k=0; k<3; ++k) box.length[k]=length[k];
  void *countArgs[]={&x,&rows,&box,&reach,&masks,&blocks};
  void *fillArgs[]={&rows,&box,&masks,&blocks,&source,&dest,&shifts};
  check(cuLaunchKernel(source ? selfFaceFill : selfFaceCounts,(rows+255)/256,1,1,
                       256,1,1,0,nullptr,source ? fillArgs : countArgs,nullptr));
}

extern "C" void mdrtNCCLSelfGather(const double *owned, const int *source, const double *shifts,
    double *ghosts, long long elements, int width, CUstream stream) {
  if (!elements) return;
  if (!module) compile();
  void *args[]={&owned,&source,&shifts,&ghosts,&elements,&width};
  check(cuLaunchKernel(selfGather,(elements+255)/256,1,1,256,1,1,0,stream,args,nullptr));
}

extern "C" void mdrtNCCLIndexPlaces(const int *order, const int *slots, int *places,
                                    int64_t rows, int64_t owned, void *stream) {
  if (!rows) return;
  if (!module) compile();
  void *args[] = {&order, &slots, &places, &rows, &owned};
  check(cuLaunchKernel(indexPlaces, (rows+255)/256, 1, 1, 256, 1, 1, 0,
                       (CUstream)stream, args, nullptr));
}

extern "C" void mdrtNCCLWriteCenters(double *out, const double *forces,
    const int *mask, int64_t rows, void *stream) {
  long long n = rows*3;
  if (!n) return;
  if (!module) compile();
  // The destination allocation and subsequent integrator work belong
  // to the default stream. Order both edges without waiting on the CPU.
  static CUevent handoff;
  if (!handoff) check(cuEventCreate(&handoff, CU_EVENT_DISABLE_TIMING));
  check(cuEventRecord(handoff, nullptr));
  check(cuStreamWaitEvent((CUstream)stream, handoff, 0));
  void *args[] = {&out, &forces, &mask, &n};
  check(cuLaunchKernel(writeCenters, (n+255)/256,1,1,256,1,1,0,
                       (CUstream)stream,args,nullptr));
  check(cuEventRecord(handoff, (CUstream)stream));
  check(cuStreamWaitEvent(nullptr, handoff, 0));
}

extern "C" void mdrtNCCLBuildTupleRows(const int *members, int count, int arity,
    const int *locations, const int *particles, const double *x,
    const int64_t *grid, const double *edges, int domain, int group, int physicalRows,
    const int *starts, const int *entries, int rows, int *mapped,
    int *blocks, int *csr) {
  if (!module) compile();
  Box box;
  for (int k=0; k<3; ++k) { box.grid[k]=grid[k]; box.length[k]=edges[k]; }
  void *args[]={&members,&count,&arity,&locations,&particles,&x,&box,&domain,&group,&physicalRows,&mapped};
  if (count) check(cuLaunchKernel(scheduleTuples,(count+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
  int n=(rows+255)/256,fill=0;
  void *rowArgs[]={&particles,&starts,&entries,&mapped,&rows,&arity,&blocks,&csr,&fill};
  if (n) check(cuLaunchKernel(tupleRows,n,1,1,256,1,1,0,nullptr,rowArgs,nullptr));
  void *prefixArgs[]={&blocks,&n,&csr,&rows,&arity};
  check(cuLaunchKernel(prefixTupleRows,1,1,1,1,1,1,0,nullptr,prefixArgs,nullptr));
  fill=1;
  if (n) check(cuLaunchKernel(tupleRows,n,1,1,256,1,1,0,nullptr,rowArgs,nullptr));
}

#ifndef MDIR_EXPERIMENT_PROTOTYPE_GATE_INPUT_H
#define MDIR_EXPERIMENT_PROTOTYPE_GATE_INPUT_H
#include "EntryFormat.h"
#include <fstream>
#include <cmath>
#include <limits>
namespace prototype_io {
inline constexpr uint64_t GATE_MAGIC = 0x31544147544f5250ULL;
struct GateInput {
  uint64_t waters, atoms, waterGroups, atomGroups, capturedChunks, unitCapacity;
  std::array<double,3> cell, reaches;
  std::vector<double> fractions;
  std::vector<int32_t> graphRows, graphNeighbors;
  NativeBasis native;
  std::vector<NativeEntry> expected;
};
inline GateInput readGateInput(const char *path) {
  std::ifstream f(path,std::ios::binary);
  if (!f) throw std::runtime_error("cannot open native snapshot gate input");
  auto read=[&](void *p,size_t bytes) {
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) ||
        !f.read(static_cast<char *>(p),static_cast<std::streamsize>(bytes)))
      throw std::runtime_error("truncated gate input");
  };
  uint64_t h[10]; read(h,sizeof h);
  if (h[0]!=GATE_MAGIC || h[1]!=1 || h[2]>INT32_MAX || h[3]>INT32_MAX || h[2]+h[3]>INT32_MAX || h[6]>INT32_MAX || h[7]>INT32_MAX)
    throw std::runtime_error("gate input header/domain mismatch");
  GateInput in{};
  in.waters=h[2];in.atoms=h[3];in.waterGroups=h[4];in.atomGroups=h[5];in.capturedChunks=h[8];in.unitCapacity=h[9];
  if (in.waterGroups>in.waters || in.atomGroups>in.atoms)
    throw std::runtime_error("group count exceeds unit count");
  read(in.cell.data(),24);read(in.reaches.data(),24);
  size_t n=in.waters+in.atoms;
  in.fractions.resize(3*n);read(in.fractions.data(),in.fractions.size()*8);
  std::vector<int32_t> order(n),starts(in.waterGroups+in.atomGroups),lengths(starts.size());
  read(order.data(),order.size()*4);read(starts.data(),starts.size()*4);read(lengths.data(),lengths.size()*4);
  in.native.waters=static_cast<int32_t>(in.waters);in.native.place.assign(n,-1);
  for (size_t p=0;p<n;++p) {
    int32_t unit=order[p];
    if (unit<0 || static_cast<size_t>(unit)>=n || in.native.place[unit]!=-1)
      throw std::runtime_error("invalid snapshot order");
    in.native.place[unit]=static_cast<int32_t>(p);
  }
  for (size_t g=0;g<starts.size();++g) {
    int32_t first=starts[g],length=lengths[g];
    if (first<0 || length<1 || length>16 || static_cast<size_t>(first)+length>n)
      throw std::runtime_error("invalid snapshot group bounds");
    NativeGroup group{first,std::vector<int32_t>(order.begin()+first,order.begin()+first+length)};
    (g<in.waterGroups?in.native.waterGroups:in.native.atomGroups).push_back(group);
  }
  in.native.frames.resize(n);
  for(size_t unit=0;unit<n;++unit)for(unsigned a=0;a<3;++a) {
    double value=std::floor(in.fractions[3*unit+a]);
    if(!std::isfinite(value) || value<INT32_MIN || value>INT32_MAX)
      throw std::runtime_error("native reference frame outside admitted domain");
    in.native.frames[unit][a]=static_cast<int64_t>(value);
  }
  in.native.validate();
  in.graphRows.resize(in.atoms+1);in.graphNeighbors.resize(h[7]);
  read(in.graphRows.data(),in.graphRows.size()*4);read(in.graphNeighbors.data(),in.graphNeighbors.size()*4);
  if (in.graphRows.front()!=0 || in.graphRows.back()!=static_cast<int64_t>(in.graphNeighbors.size()) ||
      !std::is_sorted(in.graphRows.begin(),in.graphRows.end()))
    throw std::runtime_error("invalid SS exclusion graph");
  for (int32_t v:in.graphNeighbors)if(v<0 || static_cast<uint64_t>(v)>=in.atoms)
    throw std::runtime_error("SS graph neighbor outside atom domain");
  for (uint64_t e=0;e<h[6];++e) {
    int32_t row[6];read(row,sizeof row);
    in.expected.push_back({row[0],row[1],row[2],static_cast<uint32_t>(row[3]),static_cast<uint32_t>(row[4]),row[5]});
  }
  if (f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing gate input data");
  std::sort(in.expected.begin(),in.expected.end());
  (void)expandNative(in.expected,in.native);
  return in;
}
}
#endif

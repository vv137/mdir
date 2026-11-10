#include "GateInput.h"
#include <iostream>
#include <map>
using namespace prototype_io;
int main(int argc,char **argv) try {
  if(argc<2)throw std::runtime_error("supply native .gate inputs");
  for(int i=1;i<argc;++i) {
    auto in=readGateInput(argv[i]);
    auto pairs=expandNative(in.expected,in.native);
    auto roundtrip=regroup(pairs,in.native);
    if(roundtrip!=in.expected)throw std::runtime_error("native exact-mask/frame regroup roundtrip differs");
    std::map<int32_t,uint64_t> counts;
    for(auto &entry:in.expected)++counts[entry.row];
    uint64_t chunks=0;for(auto [row,count]:counts)chunks+=(count+63)/64;
    if(chunks!=in.capturedChunks || chunks>in.unitCapacity)
      throw std::runtime_error("native chunk/capacity evidence differs");
    std::cout<<argv[i]<<" PASS native exact entry/mask/frame roundtrip: "<<in.expected.size()
        <<" records, "<<pairs.size()<<" physical bits, "<<chunks<<" chunks\n";
  }
  return 0;
} catch(const std::exception &e){std::cerr<<"STOP: "<<e.what()<<'\n';return 1;}

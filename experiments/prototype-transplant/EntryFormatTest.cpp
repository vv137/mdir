#include "EntryFormat.h"
#include <iostream>
#include <functional>
using namespace prototype_io;
static void require(bool good, const char *message) {
  if (!good) throw std::runtime_error(message);
}
static void rejects(const std::function<void()> &f, const char *message) {
  try { f(); } catch (const std::runtime_error &) { return; }
  throw std::runtime_error(message);
}
static uint32_t pack(uint32_t mask, Image image) {
  uint32_t value = mask;
  for (unsigned a = 0; a < 3; ++a) {
    if (image[a] < -4 || image[a] >= 12) throw std::runtime_error("test image outside prototype domain");
    value |= uint32_t(image[a] + 4) << (16 + 4 * a);
  }
  return value;
}
int main() {
  NativeBasis native{4, {1,3,0,2,5,6,4},
      {{0,{2,0}}, {2,{3,1}}}, {{4,{6}}, {5,{4,5}}}, {}, {}, {}};
  native.validate();
  PrototypeBasis proto{4, {1,2,0,3}, {5,6,4},
      {{1,0,0},{-1,2,0},{1,0,0},{0,-1,0},{0,1,0},{0,1,0},{1,1,-1}}};
  proto.waterOrder.resize(16,-1); proto.atomOrder.resize(16,-1);
  std::vector<PrototypeEntry> raw{{0,0,3,pack(7,{1,0,-1})},
      {1,0,0,pack(5,{-1,1,0})},{2,0,2,pack(3,{0,0,0})}};
  std::vector<PhysicalPair> expected{
      {0,0,3,{0,-1,-1}},{0,2,3,{0,-1,-1}},{0,3,1,{-2,3,1}},
      {1,4,1,{-2,2,0}},{1,5,1,{-2,2,0}},
      {2,4,5,{0,0,0}},{2,6,4,{-1,0,1}}};
  std::sort(expected.begin(),expected.end());
  auto physical=expandPrototype(raw,proto,native);
  require(physical==expected,"nonidentity shift/ownership expansion differs");
  auto entries=regroup(physical,native);
  std::vector<NativeEntry> expectedEntries{
      {0,0,3,nativeImage({0,-1,-1}),3,0},
      {0,1,1,nativeImage({-2,3,1}),1,0},
      {1,3,1,nativeImage({-2,2,0}),3,0},
      {2,4,4,nativeImage({-1,0,1}),1,0},
      {2,5,5,nativeImage({0,0,0}),1,0}};
  std::sort(expectedEntries.begin(),expectedEntries.end());
  require(entries==expectedEntries,"native regroup/image/mask differs");
  require(expandNative(entries,native)==expected,"native roundtrip differs");
  auto framed=native;
  framed.frames[0]={1,0,0};
  auto frameEntries=regroup(physical,framed);
  require(frameEntries.size()==expectedEntries.size()+1,
      "distinct native I-frame masks incorrectly merged");
  require(expandNative(frameEntries,framed)==expected,
      "I-frame partition changed physical entry semantics");
  std::reverse(raw.begin(),raw.end());
  require(regroup(expandPrototype(raw,proto,native),native)==expectedEntries,
      "input order incorrectly affects converted entry multiset");
  for (Image image : {Image{-512,0,511},Image{511,-512,0}})
    require(decodeNativeImage(nativeImage(image))==image,"native image boundary roundtrip");
  require(prototypeImage(pack(1,{-4,11,0}))==Image{-4,11,0},"prototype image boundary roundtrip");
  rejects([&]{ nativeImage({512,0,0}); },"native upper-domain error not refused");
  rejects([&]{ nativeImage({-513,0,0}); },"native lower-domain error not refused");
  rejects([&]{ auto bad=raw;bad.push_back(raw[0]);expandPrototype(bad,proto,native); },"duplicate not refused");
  rejects([&]{ expandPrototype({{0,0,3,pack(1u<<7,{0,0,0})}},proto,native); },"padding mask not refused");
  rejects([&]{ auto bad=entries;bad[0].reserved=1;expandNative(bad,native); },"reserved field not refused");
  rejects([&]{ auto bad=native;bad.place[0]=bad.place[1];bad.validate(); },"nonpermutation not refused");
  require(regroup({},native).empty(),"empty conversion failed");
  NativeBasis identity{2,{0,1,2},{{0,{0,1}}},{{2,{2}}},{},{},{}};
  identity.validate();
  PrototypeBasis identityProto{2,{0,1},{2},{{0,0,0},{0,0,0},{0,0,0}}};
  identityProto.waterOrder.resize(16,-1);identityProto.atomOrder.resize(16,-1);
  auto identityPairs=expandPrototype({{0,0,1,pack(1,{0,0,0})},
      {1,0,0,pack(1,{1,-1,0})}},identityProto,identity);
  std::vector<PhysicalPair> identityExpected{{0,0,1,{0,0,0}},{1,2,0,{1,-1,0}}};
  require(identityPairs==identityExpected,"identity mapping differs");
  std::cout<<"PASS: identity-independent groups, half ownership, SW masks, image shifts/domains, exact fields, multiplicity refusals and empty format\n";
}

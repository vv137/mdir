#include "Bonded.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

static void fail(const char *message) {
  std::cerr << "mdir-cpu-lj: " << message << '\n';
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort();
}
void Bonded::load(const std::string &path,
                  const std::vector<long long> &rootIDs, bool mixed) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int selected = !path.empty();
  MPI_Bcast(&selected, 1, MPI_INT, 0, MPI_COMM_WORLD);
  active = selected;
  if (!active)
    return;
  if (!rank) {
    std::ifstream file(path);
    if (!file)
      fail("cannot open bond file");
    std::set<long long> known(rootIDs.begin(), rootIDs.end());
    std::set<std::pair<long long, long long>> seen;
    std::string line;
    while (std::getline(file, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos)
        continue;
      std::istringstream row(line);
      std::string extra;
      Bond b;
      if (!(row >> b.a >> b.b >> b.k >> b.r0) || (row >> extra))
        fail("invalid bond row");
      if (b.b < b.a)
        std::swap(b.a, b.b);
      if (b.a == b.b || !known.count(b.a) || !known.count(b.b) ||
          !seen.emplace(b.a, b.b).second)
        fail("bond requires distinct known IDs and a unique unordered pair");
      for (double p : {b.k, b.r0})
        if (!std::isfinite(p) || p <= 0 ||
            (mixed && (!std::isfinite(float(p)) || float(p) <= 0)))
          fail("bond parameters are outside selected precision range");
      if (bonds.size() >= size_t(std::numeric_limits<int>::max() / 10))
        fail("too many bonds");
      bonds.push_back(b);
    }
    if (file.bad())
      fail("bond file read error");
  }
  int count = bonds.size();
  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<long long> ends(2 * count);
  std::vector<double> params(2 * count);
  if (!rank)
    for (int i = 0; i < count; ++i) {
      ends[2 * i] = bonds[i].a;
      ends[2 * i + 1] = bonds[i].b;
      params[2 * i] = bonds[i].k;
      params[2 * i + 1] = bonds[i].r0;
    }
  MPI_Bcast(ends.data(), ends.size(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(params.data(), params.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  bonds.resize(count);
  for (int i = 0; i < count; ++i)
    bonds[i] = {ends[2 * i], ends[2 * i + 1], params[2 * i], params[2 * i + 1]};
}
void Bonded::rebuild(uint64_t nextEpoch, const std::vector<long long> &ids,
                     int n) {
  if (!active)
    return;
  if (phase != Idle || nextEpoch <= epoch)
    fail("invalid topology epoch transition");
  epoch = nextEpoch;
  owned = n;
  std::vector<int> counts(ranks), offsets(ranks);
  MPI_Allgather(&owned, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  long long total = 0;
  for (int r = 0; r < ranks; ++r) {
    offsets[r] = total;
    total += counts[r];
    if (total > std::numeric_limits<int>::max() / 3)
      fail("topology directory capacity overflow");
  }
  std::vector<long long> directory(total);
  MPI_Allgatherv(ids.data(), owned, MPI_LONG_LONG, directory.data(),
                 counts.data(), offsets.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
  std::map<long long, int> owners, localIndex;
  for (int r = 0; r < ranks; ++r)
    for (int i = offsets[r]; i < offsets[r] + counts[r]; ++i)
      if (!owners.emplace(directory[i], r).second)
        fail("duplicate topology owner");
  for (int i = 0; i < owned; ++i)
    localIndex[ids[i]] = i;
  std::vector<std::set<long long>> needed(ranks);
  std::vector<Bond> work;
  for (auto b : bonds) {
    if (!owners.count(b.a) || !owners.count(b.b))
      fail("missing topology owner");
    if (owners[b.a] != rank)
      continue;
    work.push_back(b);
    if (owners[b.b] != rank)
      needed[owners[b.b]].insert(b.b);
  }
  int localWork = work.size(), globalWork;
  MPI_Allreduce(&localWork, &globalWork, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (globalWork != int(bonds.size()))
    fail("incomplete bond work assignment");
  askCounts.assign(ranks, 0);
  askOffsets.resize(ranks);
  serveCounts.resize(ranks);
  serveOffsets.resize(ranks);
  std::vector<long long> asks;
  for (int r = 0; r < ranks; ++r) {
    askOffsets[r] = asks.size();
    askCounts[r] = needed[r].size();
    for (auto id : needed[r]) {
      localIndex[id] = owned + asks.size();
      asks.push_back(id);
    }
  }
  if (asks.size() + owned > size_t(std::numeric_limits<int>::max() / 3))
    fail("topology local capacity overflow");
  MPI_Alltoall(askCounts.data(), 1, MPI_INT, serveCounts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);
  long long serve = 0;
  for (int r = 0; r < ranks; ++r) {
    serveOffsets[r] = serve;
    serve += serveCounts[r];
    if (serve > std::numeric_limits<int>::max() / 3)
      fail("topology send capacity overflow");
  }
  std::vector<long long> requested(serve);
  MPI_Alltoallv(asks.data(), askCounts.data(), askOffsets.data(), MPI_LONG_LONG,
                requested.data(), serveCounts.data(), serveOffsets.data(),
                MPI_LONG_LONG, MPI_COMM_WORLD);
  serveIndex.clear();
  for (auto id : requested) {
    auto it = localIndex.find(id);
    if (it == localIndex.end() || it->second >= owned)
      fail("topology request reached wrong owner");
    serveIndex.push_back(it->second);
  }
  for (int r = 0; r < ranks; ++r) {
    askCounts[r] *= 3;
    askOffsets[r] *= 3;
    serveCounts[r] *= 3;
    serveOffsets[r] *= 3;
  }
  positions.resize(3 * (owned + asks.size()));
  send.resize(3 * serve);
  receive.resize(3 * asks.size());
  partial.resize(3 * owned);
  endpoints.clear();
  parameters.clear();
  for (auto b : work) {
    endpoints.push_back(localIndex.at(b.a));
    endpoints.push_back(localIndex.at(b.b));
    parameters.push_back(b.k);
    parameters.push_back(b.r0);
  }
  contributions.resize(6 * work.size());
  results.resize(10 * work.size());
}
void Bonded::exchange(bool reverse, bool asynchronous) {
  requests.clear();
  auto &sc = reverse ? askCounts : serveCounts,
       &so = reverse ? askOffsets : serveOffsets;
  auto &rc = reverse ? serveCounts : askCounts,
       &ro = reverse ? serveOffsets : askOffsets;
  auto &s = reverse ? receive : send, &d = reverse ? send : receive;
  if (!asynchronous) {
    MPI_Alltoallv(s.data(), sc.data(), so.data(), MPI_DOUBLE, d.data(),
                  rc.data(), ro.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return;
  }
  int tag = reverse ? 82 : 81;
  for (int r = 0; r < ranks; ++r)
    if (rc[r]) {
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(d.data() + ro[r], rc[r], MPI_DOUBLE, r, tag, MPI_COMM_WORLD,
                &requests.back());
    }
  for (int r = 0; r < ranks; ++r)
    if (sc[r]) {
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Isend(s.data() + so[r], sc[r], MPI_DOUBLE, r, tag, MPI_COMM_WORLD,
                &requests.back());
    }
}
void Bonded::start(const std::vector<double> &x, bool asynchronous) {
  if (phase != Idle || !epoch)
    fail("topology start requires idle current map");
  std::copy_n(x.data(), 3 * owned, positions.data());
  std::fill(positions.begin() + 3 * owned, positions.end(),
            std::numeric_limits<double>::quiet_NaN());
  for (size_t i = 0; i < serveIndex.size(); ++i)
    std::copy_n(x.data() + 3 * serveIndex[i], 3, send.data() + 3 * i);
  phase = Forward;
  exchange(false, asynchronous);
}
void Bonded::wait(const double *box) {
  if (phase != Forward)
    fail("topology wait without start");
  if (!requests.empty())
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  std::copy(receive.begin(), receive.end(), positions.begin() + 3 * owned);
  for (int b = 0; b < size(); ++b) {
    double r2 = 0;
    for (int k = 0; k < 3; ++k) {
      double d = positions[3 * endpoints[2 * b] + k] -
                 positions[3 * endpoints[2 * b + 1] + k];
      d -= box[k] * std::nearbyint(d / box[k]);
      r2 += d * d;
    }
    if (!std::isfinite(r2) || r2 == 0)
      fail("bond has coincident or nonfinite endpoints");
  }
  phase = Readable;
}
void Bonded::reverseStart(bool asynchronous) {
  if (phase != Readable)
    fail("bond reverse requires readable topology values");
  std::fill(partial.begin(), partial.end(), 0);
  std::fill(receive.begin(), receive.end(), 0);
  for (int b = 0; b < size(); ++b)
    for (int end = 0; end < 2; ++end) {
      int slot = endpoints[2 * b + end];
      for (int k = 0; k < 3; ++k) {
        double value = contributions[6 * b + 3 * end + k];
        if (!std::isfinite(value))
          fail("nonfinite bonded contribution");
        if (slot < owned)
          partial[3 * slot + k] += value;
        else
          receive[3 * (slot - owned) + k] += value;
      }
    }
  phase = Reverse;
  exchange(true, asynchronous);
}
void Bonded::reverseWait(std::vector<double> &force, double *totals) {
  if (phase != Reverse)
    fail("bond completion without reverse start");
  if (!requests.empty())
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  for (size_t i = 0; i < serveIndex.size(); ++i)
    for (int k = 0; k < 3; ++k)
      partial[3 * serveIndex[i] + k] += send[3 * i + k];
  for (int i = 0; i < 3 * owned; ++i)
    force[i] += partial[i];
  for (int b = 0; b < size(); ++b)
    for (int k = 0; k < 10; ++k) {
      if (!std::isfinite(results[10 * b + k]))
        fail("nonfinite bonded result");
      totals[k] += results[10 * b + k];
    }
  phase = Idle;
}

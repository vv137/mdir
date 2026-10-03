#ifndef MDIR_TOOLS_CPU_LJ_BONDED_H
#define MDIR_TOOLS_CPU_LJ_BONDED_H
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

// A restricted ID-indexed topology route. It is independent of spatial support.
class Bonded {
public:
  void load(const std::string &path, const std::vector<long long> &rootIDs,
            bool mixed);
  void rebuild(uint64_t epoch, const std::vector<long long> &ids, int owned);
  void start(const std::vector<double> &positions, bool asynchronous);
  void wait(const double *box);
  void reverseStart(bool asynchronous);
  void reverseWait(std::vector<double> &force, double *totals);
  bool enabled() const { return active; }
  int size() const { return parameters.size() / 2; }
  int localSize() const { return positions.size() / 3; }
  std::vector<int32_t> endpoints;
  std::vector<double> parameters, positions, contributions, results;

private:
  struct Bond {
    long long a, b;
    double k, r0;
  };
  std::vector<Bond> bonds;
  bool active = false;
  int rank = 0, ranks = 1, owned = 0;
  uint64_t epoch = 0;
  enum Phase { Idle, Forward, Readable, Reverse } phase = Idle;
  std::vector<int> askCounts, askOffsets, serveCounts, serveOffsets, serveIndex;
  std::vector<double> send, receive, partial;
  std::vector<MPI_Request> requests;
  void exchange(bool reverse, bool asynchronous);
};
std::string bondedKernel(const double *box, bool mixed);
#endif

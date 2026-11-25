// Distributed trajectory accounting. Topology and logical transfer edges are
// replicated. Only phase clocks need to travel to the writer.
#pragma once
#include "../../runtime/mdrt_dist_stats.h"
#include "Manifest.h"
#include "mdir/Driver/Output.h"
#include <array>
#include <chrono>
#include <map>

namespace mdir::tool {
class DistributionLog {
  using Clock = std::chrono::steady_clock;
  using Edge = std::pair<int64_t, int64_t>;
  using Times = std::array<double, DistPhaseCount>;
  struct Counts {
    int64_t in = 0, out = 0;
  };
  struct Sample {
    Times seconds{};
    std::vector<Counts> moves;
    std::map<Edge, int64_t> bytes;
    explicit Sample(size_t n) : moves(n) {}
  };

  driver::Log &log;
  Manifest &manifest;
  int rank, ranks;
  int64_t domains, first, step, period;
  bool setup = true;
  bool enteredStep = false;
  double reach;
  int enclosing = 0;
  FILE *trace = nullptr;
  std::vector<int64_t> owned, ghosts;
  llvm::json::Array boxes;
  Sample current, interval, total;
  Clock::time_point started = Clock::now(), began = started;
  std::array<Clock::time_point, DistPhaseCount> starts;
  static constexpr const char *names[] = {"wall", "pairs", "other_terms",
      "halo_copy", "halo_sum", "all_reduce", "migration", "wait"};
  int owner(int64_t d) const {
    auto q = domains / ranks, r = domains % ranks;
    return d < r * (q + 1) ? d / (q + 1) : r + (d - r * (q + 1)) / q;
  }
  void domain(int64_t d, int64_t n, int64_t g, const double *lo, const double *hi) {
    owned[d] = n;
    ghosts[d] = g;
    if (!setup || rank) return;
    llvm::json::Array lower, upper;
    for (int i = 0; i < 3; ++i) {
      lower.push_back(lo[i] * 10);
      upper.push_back(hi[i] * 10);
    }
    double a = 10*(hi[0]-lo[0]), b = 10*(hi[1]-lo[1]), c = 10*(hi[2]-lo[2]);
    // The Euclidean halo rounds each edge and corner of the domain box.
    const double pi = std::acos(-1.);
    double volume = 2*reach*(a*b+b*c+c*a) + pi*reach*reach*(a+b+c) +
                    4*pi*reach*reach*reach/3;
    log.print("MDIR: domain %lld rank %d box Ang [%g,%g] [%g,%g] [%g,%g] owned %lld ghosts %lld position import %lld bytes, halo volume %.9g Ang^3\n",
        (long long)d, owner(d), lo[0]*10, hi[0]*10, lo[1]*10, hi[1]*10,
        lo[2]*10, hi[2]*10, (long long)n, (long long)g, (long long)g*24, volume);
    boxes.push_back(llvm::json::Object{{"domain", d}, {"rank", owner(d)},
        {"lower_angstrom", std::move(lower)}, {"upper_angstrom", std::move(upper)},
        {"import_volume_angstrom3", volume}, {"owned", n}, {"ghosts", g}, {"position_import_bytes", g * 24}});
    if (d + 1 == domains) {
      setup = false;
      if (auto e = manifest.domainSetup(std::move(boxes)))
        error = llvm::toString(std::move(e));
    }
  }

  static void add(Sample &to, const Sample &from) {
    for (unsigned i = 0; i < to.seconds.size(); ++i)
      to.seconds[i] += from.seconds[i];
    for (size_t i = 0; i < to.moves.size(); ++i) {
      to.moves[i].in += from.moves[i].in;
      to.moves[i].out += from.moves[i].out;
    }
    for (auto [edge, bytes] : from.bytes) to.bytes[edge] += bytes;
  }

  void rows(Sample &s, const char *label, bool detailed) {
    auto print = [&](const char *format, auto... args) {
      if (detailed) std::fprintf(trace, format, args...);
      else log.print(format, args...);
    };
    for (int64_t d = 0; d < domains; ++d)
      if (!detailed || owner(d) == rank)
        print("DIST: %s step %lld rank %d domain %lld owned %lld ghost %lld migrated_in %lld migrated_out %lld\n",
              label, (long long)step, owner(d), (long long)d, (long long)owned[d],
              (long long)ghosts[d], (long long)s.moves[d].in, (long long)s.moves[d].out);
    for (auto [edge, bytes] : s.bytes)
      if (!detailed || owner(edge.first) == rank || owner(edge.second) == rank)
        print("DIST: %s step %lld domain %lld sent %lld bytes to domain %lld received %lld bytes\n",
              label, (long long)step, (long long)edge.first, (long long)bytes,
              (long long)edge.second, (long long)bytes);
    // A logical halo edge includes self-images and in-process copies.
    // Network traffic includes only edges crossing the rank boundary.
    enum { Owned, Ghost, In, Out, Sent, Received, NetworkSent, NetworkReceived, Count };
    std::vector<std::array<int64_t, Count>> counts(ranks);
    for (int64_t d = 0; d < domains; ++d) {
      auto &row = counts[owner(d)];
      row[Owned] += owned[d];
      row[Ghost] += ghosts[d];
      row[In] += s.moves[d].in;
      row[Out] += s.moves[d].out;
    }
    for (auto [edge, bytes] : s.bytes) {
      int from = owner(edge.first), to = owner(edge.second);
      counts[from][Sent] += bytes;
      counts[to][Received] += bytes;
      if (from != to) {
        counts[from][NetworkSent] += bytes;
        counts[to][NetworkReceived] += bytes;
      }
    }
    const char *columns[] = {"owned", "ghost", "migrated_in", "migrated_out",
        "sent_bytes", "received_bytes", "network_sent_bytes", "network_received_bytes"};
    for (int i = 0; i < Count; ++i) {
      int64_t low = counts.front()[i], high = low, sum = 0;
      for (auto &row : counts) {
        low = std::min(low, row[i]);
        high = std::max(high, row[i]);
        sum += row[i];
      }
      print("DIST: %s step %lld %s min/mean/max %lld %.8g %lld\n", label,
            (long long)step, columns[i], (long long)low, double(sum)/ranks, (long long)high);
      for (int r = 0; r < ranks; ++r)
        if (!detailed || r == rank)
          print("DIST: %s step %lld rank %d %s %lld share %.6f\n", label,
                (long long)step, r, columns[i], (long long)counts[r][i],
                sum ? double(counts[r][i])/sum : 0.);
    }
    std::vector<double> times(size_t(ranks) * DistPhaseCount);
    if (gather) gather(s.seconds.data(), times.data(), DistPhaseCount);
    else std::copy(s.seconds.begin(), s.seconds.end(), times.begin());
    if (rank && !detailed) return;
    for (int i = 0; i < DistPhaseCount; ++i) {
      double sum = 0, low = times[i], high = times[i];
      for (int r = 0; r < ranks; ++r) {
        double value = times[r * DistPhaseCount + i];
        sum += value;
        low = std::min(low, value);
        high = std::max(high, value);
      }
      print("DIST: %s step %lld %s min/mean/max %.9g %.9g %.9g s\n",
            label, (long long)step, names[i], low, sum/ranks, high);
      for (int r = 0; r < ranks; ++r) {
        if (detailed && r != rank) continue;
        double v = times[r * DistPhaseCount + i];
        print("DIST: %s step %lld rank %d %s %.9g s share %.6f\n",
              label, (long long)step, r, names[i], v, sum ? v/sum : 0.);
      }
    }
    if (detailed && std::fflush(trace)) error = "cannot flush distributed step trace";
  }
  void phase(int64_t code) {
    auto now = Clock::now();
    if (code == DistStep) {
      // Finish the preceding step after its energy, frame and checkpoint
      // callbacks. The first marker opens the first step of this invocation.
      if (!enteredStep) {
        enteredStep = true;
        return;
      }
      ++step;
      current.seconds[0] = std::chrono::duration<double>(now-began).count();
      if (trace) rows(current, "trace", true);
      add(interval, current);
      current = Sample(domains);
      if (period > 0 && (step-first) % period == 0) {
        rows(interval, "interval", false);
        add(total, interval);
        interval = Sample(domains);
      }
      began = Clock::now();
    } else if (code > 0) {
      starts[code] = now;
      if (code != DistWait) enclosing = code;
    } else {
      double duration = std::chrono::duration<double>(now-starts[-code]).count();
      current.seconds[-code] += duration;
      if (code == -DistWait && enclosing) current.seconds[enclosing] -= duration;
      if (code != -DistWait) enclosing = 0;
    }
  }
public:
  std::string error;
  void (*gather)(const double *, double *, int) = nullptr;
  DistributionLog(driver::Log &log, Manifest &manifest, int rank, int ranks,
                  int64_t n, int64_t first, int64_t period, double reach)
      : log(log), manifest(manifest), rank(rank), ranks(ranks), domains(n),
        first(first), step(first), period(period), reach(reach), owned(n), ghosts(n),
        current(n), interval(n), total(n) {}
  bool open(const char *prefix, const std::string &checkpoint) {
    if (!prefix || !*prefix) return true;
    auto path = std::string(prefix) + ".rank" + std::to_string(rank) + ".log";
    if (!checkpoint.empty()) {
      std::vector<std::string> reserved{checkpoint, checkpoint + ".partial",
                                        driver::getPreviousCheckpointPath(checkpoint)};
      if (auto e = checkManifestPath(path, {}, reserved)) {
        error = "distributed trace path conflicts with checkpoint: " + llvm::toString(std::move(e));
        return false;
      }
    }
    trace = std::fopen(path.c_str(), "wx");
    if (!trace) error = "cannot create distributed trace '" + path + "': " + std::strerror(errno);
    return trace != nullptr;
  }
  void finish() {
    if (enteredStep) phase(DistStep);
    add(interval, current);
    add(total, interval);
    total.seconds[0] = std::chrono::duration<double>(Clock::now()-started).count();
    rows(total, "total", false);
  }
  ~DistributionLog() { if (trace) std::fclose(trace); }
  DistObserver observer{this,
    [](void *p, int64_t d, int64_t n, int64_t g, const double *lo, const double *hi) {
      static_cast<DistributionLog *>(p)->domain(d,n,g,lo,hi);
    }, [](void *p, int64_t from, int64_t to) {
      auto &s = static_cast<DistributionLog *>(p)->current;
      ++s.moves[from].out;
      ++s.moves[to].in;
    }, [](void *p, int64_t from, int64_t to, int64_t bytes) {
      static_cast<DistributionLog *>(p)->current.bytes[{from,to}] += bytes;
    }, [](void *p, int64_t code) { static_cast<DistributionLog *>(p)->phase(code); }};
};
} // namespace mdir::tool

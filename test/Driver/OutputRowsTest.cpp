// What a run keeps of the rows of its energies (#300): nothing for a row,
// and the sums of the summary at the end of `mdir run`, which are those of
// a pass over all the rows to the bit. Its log keeps nothing of them
// either once no file will open (#304).
#include "mdir/Driver/Output.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdlib>
#include <malloc.h>
#include <random>
#include <vector>
using namespace mdir;
using namespace mdir::driver;

static void require(bool ok, const char *what) {
  if (!ok) {
    llvm::errs() << what << "\n";
    std::exit(1);
  }
}

// The mean and the error of ten blocks of all the values, as `mdir run`
// took them from the rows it kept, before #300.
static std::pair<double, double> summarize(const std::vector<double> &values) {
  size_t n = values.size();
  double sum = 0.0;
  for (double value : values)
    sum += value;
  double mean = sum / static_cast<double>(n);
  size_t blocks = std::min<size_t>(10, n), size = n / blocks;
  double error = 0.0;
  if (blocks > 1) {
    double total = 0.0, squares = 0.0;
    for (size_t b = 0; b != blocks; ++b) {
      double block = 0.0;
      for (size_t i = 0; i != size; ++i)
        block += values[b * size + i];
      block /= static_cast<double>(size);
      total += block;
      squares += block * block;
    }
    double average = total / static_cast<double>(blocks);
    double variance =
        (squares - static_cast<double>(blocks) * average * average) /
        static_cast<double>(blocks - 1);
    error = std::sqrt(std::max(variance, 0.0) / static_cast<double>(blocks));
  }
  return {mean, error};
}

static size_t getBytesInUse() {
  struct mallinfo2 info = mallinfo2();
  return info.uordblks + info.hblkhd;
}

int main() {
  // The sums in constant memory are those of the pass, to the bit, for
  // numbers of rows that leave a rest after ten blocks and that leave
  // none, and for fewer than ten rows.
  std::mt19937_64 engine(300);
  std::normal_distribution<double> draw(300.0, 25.0);
  int64_t compared = 0;
  for (int64_t rows :
       {1, 2, 3, 7, 9, 10, 11, 19, 20, 21, 99, 100, 101, 1000, 1009, 65537}) {
    std::vector<double> values(rows);
    for (double &value : values)
      value = draw(engine);
    BlockMean mean;
    mean.plan(rows);
    for (double value : values)
      mean.add(value);
    auto [expected, error] = summarize(values);
    require(mean.isComplete() && mean.count == rows, "the count of the rows");
    require(mean.getMean() == expected, "the mean differs from the pass");
    require(mean.getError() == error, "the error differs from the pass");
    ++compared;
  }
  llvm::outs() << "the means and the errors of ten blocks are those of a "
                  "pass over the rows for "
               << compared << " numbers of rows\n";

  // A run of other rows than planned says so.
  BlockMean fewer;
  fewer.plan(100);
  for (int k = 0; k != 50; ++k)
    fewer.add(1.0);
  require(!fewer.isComplete() && fewer.blocksDone == 5, "a run cut short");

  // A run keeps nothing for a row of its energies: the bytes in use after
  // 1024 rows and after 2048, with the temperatures of the solvent and of
  // the solute, as `mdir run` plans them. The log is quiet: what it keeps
  // for a file that may open is not of the rows of the summary.
  const int64_t rows = 1024;
  Output output;
  output.log.quiet = true;
  output.energyPeriod = 1;
  output.degreesOfFreedom = 3000.0;
  output.solventFreedom = 2000.0;
  output.volume = 27.0;
  output.rateStep = rows;
  for (BlockMean *mean : {&output.soluteOptimal, &output.solventOptimal,
                          &output.soluteFull, &output.solventFull})
    mean->plan(2 * rows);
  setOutput(&output);
  auto write = [&](int64_t from, int64_t to) {
    for (int64_t step = from; step != to; ++step) {
      if (step > 0)
        mdrtWriteSolvent(2500.0 + step % 7, 2510.0 + step % 5);
      _mlir_ciface_mdrtWriteEnergies(step, -1.0e4, 3750.0 + step % 3, 12.0,
                                     -500.0, 3762.0 + step % 3);
    }
  };
  write(0, rows + 1);
  size_t before = getBytesInUse();
  write(rows + 1, 2 * rows + 1);
  size_t after = getBytesInUse();
  setOutput(nullptr);
  require(output.solventOptimal.count == 2 * rows &&
              output.soluteFull.count == 2 * rows &&
              output.solventOptimal.isComplete(),
          "the rows of the temperatures");
  require(output.numEnergyTimes == 2 * rows + 1 && output.rateFrom &&
              output.rateFrom->first == rows &&
              output.rateLast.first == 2 * rows,
          "the rows of the rate");
  long long growth =
      static_cast<long long>(after) - static_cast<long long>(before);
  llvm::outs() << "bytes in use from row " << rows << " to row " << 2 * rows
               << ": " << growth << "\n";
  // The rows that a run kept took 16 bytes each for their times alone, in
  // a vector that doubles: 16384 bytes at least here, and 65536 with the
  // four temperatures when it was measured. The bound lets through what
  // grows by less than a byte a row.
  require(growth <= 1024, "the run keeps memory for each row");

  // A log that prints and has no file keeps nothing of its rows either
  // (#304): `mdir run` without `log` in [output] says after its set-up
  // that no file will open. Until then the log keeps what it prints, for
  // the file that may open.
  Output printed;
  printed.energyPeriod = 1;
  printed.degreesOfFreedom = 3000.0;
  printed.solventFreedom = 2000.0;
  printed.volume = 27.0;
  printed.rateStep = rows;
  for (BlockMean *mean : {&printed.soluteOptimal, &printed.solventOptimal,
                          &printed.soluteFull, &printed.solventFull})
    mean->plan(2 * rows);
  llvm::outs().flush();
  printed.log.print("MDIR: a line of the set-up of a run\n");
  require(printed.log.getNumPendingBytes() != 0,
          "the log keeps no text for the file that may open");
  printed.log.keepNothing();
  require(printed.log.getNumPendingBytes() == 0,
          "the log keeps text for a file that will not open");
  setOutput(&printed);
  write(0, rows + 1);
  before = getBytesInUse();
  write(rows + 1, 2 * rows + 1);
  after = getBytesInUse();
  setOutput(nullptr);
  printed.log.flush();
  require(printed.numEnergyTimes == 2 * rows + 1, "the rows of the log");
  growth = static_cast<long long>(after) - static_cast<long long>(before);
  llvm::outs() << "bytes in use of a log without a file from row " << rows
               << " to row " << 2 * rows << ": " << growth << ", and "
               << printed.log.getNumPendingBytes() << " bytes of text kept\n";
  // A row of the log is 121 bytes of text here, kept in a string that
  // doubles: 1024 rows are 123904 bytes at least, and 245760 when it was
  // measured, with 247929 bytes of text kept. The bound lets through what
  // grows by less than a byte a row.
  require(growth <= 1024, "the log keeps memory for each row");
  require(printed.log.getNumPendingBytes() == 0,
          "the log keeps the text of its rows");
  llvm::outs() << "rows passed\n";
  return 0;
}

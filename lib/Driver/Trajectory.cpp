// Trajectories: the positions of the particles at intervals, in DCD or in
// XTC (D141).

#include "mdir/Driver/Trajectory.h"

#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <unistd.h>

using namespace mdir::driver;

TrajectoryWriter::~TrajectoryWriter() { close(); }

void TrajectoryWriter::close() {
  if (file)
    std::fclose(file);
  file = nullptr;
}

std::unique_ptr<TrajectoryWriter>
mdir::driver::createTrajectoryWriter(TrajectoryFormat format) {
  if (format == TrajectoryFormat::XTC)
    return std::make_unique<XTCWriter>();
  return std::make_unique<DCDWriter>();
}

//===----------------------------------------------------------------------===//
// DCD
//===----------------------------------------------------------------------===//

/// Writes a record: its length in bytes, the bytes, and the length again.
static void writeRecord(std::FILE *file, const void *data, int32_t size) {
  std::fwrite(&size, sizeof(size), 1, file);
  std::fwrite(data, 1, size, file);
  std::fwrite(&size, sizeof(size), 1, file);
}

DCDWriter::~DCDWriter() { close(); }

void DCDWriter::writeHeader() {
  // The first record: the tag, and 20 numbers that describe the file.
  struct {
    char tag[4];
    int32_t numbers[20];
  } head;
  std::memcpy(head.tag, "CORD", 4);
  std::memset(head.numbers, 0, sizeof(head.numbers));
  head.numbers[0] = numFrames;
  head.numbers[1] = static_cast<int32_t>(first);
  head.numbers[2] = static_cast<int32_t>(period);
  head.numbers[3] = numFrames * static_cast<int32_t>(period);
  // The time step, in units of 48.88821 fs.
  float delta = static_cast<float>(timestep / 0.04888821);
  std::memcpy(&head.numbers[9], &delta, sizeof(delta));
  // The frames hold the cell.
  head.numbers[10] = 1;
  // The version of the format.
  head.numbers[19] = 24;
  writeRecord(file, &head, sizeof(head));

  struct {
    int32_t count;
    char line[80];
  } title;
  title.count = 1;
  std::memset(title.line, ' ', sizeof(title.line));
  const char *text = "Written by MDIR";
  std::memcpy(title.line, text, std::strlen(text));
  writeRecord(file, &title, sizeof(title));

  int32_t count = static_cast<int32_t>(numParticles);
  writeRecord(file, &count, sizeof(count));
}

llvm::Error DCDWriter::open(const std::string &path, size_t numParticles,
                            int64_t first, int64_t period, double timestep,
                            const double box[3]) {
  file = std::fopen(path.c_str(), "wb");
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write '%s'", path.c_str());
  this->numParticles = numParticles;
  this->first = first;
  this->period = period;
  this->timestep = timestep;
  for (int i = 0; i != 3; ++i)
    this->box[i] = box[i];
  writeHeader();
  return llvm::Error::success();
}

llvm::Expected<int64_t> DCDWriter::append(const std::string &path,
                                          size_t numParticles,
                                          int64_t frames, int64_t period,
                                          double timestep,
                                          const double box[3]) {
  auto fail = [&](const llvm::Twine &message) -> llvm::Error {
    close();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "'" + path + "' " + message);
  };
  file = std::fopen(path.c_str(), "r+b");
  if (!file)
    return fail("cannot be opened to continue it");
  // The records that writeHeader writes: the description, the title, and
  // the number of particles.
  auto readInt = [&](int32_t &value) {
    return std::fread(&value, sizeof(value), 1, file) == 1;
  };
  int32_t size, end, count;
  struct {
    char tag[4];
    int32_t numbers[20];
  } head;
  bool read = readInt(size) && size == sizeof(head) &&
              std::fread(&head, sizeof(head), 1, file) == 1 && readInt(end) &&
              end == size && std::memcmp(head.tag, "CORD", 4) == 0;
  if (!read || head.numbers[10] != 1 || head.numbers[19] != 24)
    return fail("is not a trajectory that MDIR wrote");
  read = readInt(size) && size == 84 &&
         std::fseek(file, size, SEEK_CUR) == 0 && readInt(end) &&
         end == size && readInt(size) && size == 4 && readInt(count) &&
         readInt(end) && end == size;
  if (!read)
    return fail("is not a trajectory that MDIR wrote");
  if (count != static_cast<int32_t>(numParticles))
    return fail("holds " + llvm::Twine(count) + " particles, and the run " +
                llvm::Twine(numParticles));
  if (head.numbers[2] != static_cast<int32_t>(period))
    return fail("has a frame every " + llvm::Twine(head.numbers[2]) +
                " steps, and the run writes one every " +
                llvm::Twine(period));

  // The frames that the file holds in full: the cell, then x, y, and z.
  long header = std::ftell(file);
  long frameSize =
      (8 + 6 * sizeof(double)) + 3 * (8 + numParticles * sizeof(float));
  std::fseek(file, 0, SEEK_END);
  long length = std::ftell(file);
  int64_t held = (length - header) / frameSize;
  if (held < frames)
    return fail("holds " + llvm::Twine(held) + " frames, fewer than the " +
                llvm::Twine(frames) + " that the checkpoint counts");
  long kept = header + frames * frameSize;
  std::fflush(file);
  if (kept != length && ::ftruncate(::fileno(file), kept) != 0)
    return fail("cannot be cut to the frames that the checkpoint counts");

  this->numParticles = numParticles;
  this->first = head.numbers[1];
  this->period = period;
  this->timestep = timestep;
  for (int i = 0; i != 3; ++i)
    this->box[i] = box[i];
  numFrames = static_cast<int32_t>(frames);
  std::rewind(file);
  writeHeader();
  std::fseek(file, kept, SEEK_SET);
  std::fflush(file);
  return held - frames;
}

void DCDWriter::writeFrame(const float *positions, int64_t, double) {
  // The cell: a, cos γ, b, cos β, cos α, c, as NAMD and OpenMM write it
  // (docs/triclinic-m2.md, Section 1); right angles for an orthorhombic
  // cell.
  double b = std::sqrt(tilt[0] * tilt[0] + box[1] * box[1]);
  double c = std::sqrt(tilt[1] * tilt[1] + tilt[2] * tilt[2] + box[2] * box[2]);
  double cell[6] = {box[0],
                    tilt[0] / b,
                    b,
                    tilt[1] / c,
                    0.0,
                    c};
  cell[4] = (tilt[0] * tilt[1] + box[1] * tilt[2]) / (b * c);
  writeRecord(file, cell, sizeof(cell));

  std::vector<float> component(numParticles);
  for (int c = 0; c != 3; ++c) {
    for (size_t i = 0; i != numParticles; ++i)
      component[i] = positions[3 * i + c];
    writeRecord(file, component.data(),
                static_cast<int32_t>(numParticles * sizeof(float)));
  }

  // Keep the number of frames in the header up to date, so that the file
  // can be read if the run ends early.
  ++numFrames;
  long end = std::ftell(file);
  std::rewind(file);
  writeHeader();
  std::fseek(file, end, SEEK_SET);
  std::fflush(file);
}


//===----------------------------------------------------------------------===//
// XTC
//===----------------------------------------------------------------------===//
//
// The format of GROMACS, compatible with its readers: XDR (big-endian
// numbers of 4 bytes, opaque data padded to 4 bytes), a frame of
//
//   1995, the number of particles, the step, the time (float, ps),
//   the vectors of the cell (9 floats, nm), the number of particles again,
//
// then the positions: for at most 9 particles 3 floats each; else the
// precision p (float), the least and the largest of the integers round(p x)
// along each axis, the index of the first size of small differences, and
// the compressed bits, counted in bytes. The positions are integers
// within the box of those bounds. Each is written with the bits of the
// product of the three ranges, or, where a range passes 2²⁴, with the bits
// of each range; a bit then tells whether the number of the following
// particles that lie close to it and are written as small differences
// changed, in 5 bits if so, along with whether the size of a small
// difference grows or shrinks by a step of a table of sizes, each about
// 2^(1/3) times the one before. The first close particle is written
// before the one it follows, which suits water, whose oxygen is farther
// from the hydrogens than they are from each other.

namespace {

/// The sizes of small differences: about 2^(1/3) apart, from index 9 on.
const int magicInts[] = {
    0,       0,       0,       0,       0,       0,       0,       0,
    0,       8,       10,      12,      16,      20,      25,      32,
    40,      50,      64,      80,      101,     128,     161,     203,
    256,     322,     406,     512,     645,     812,     1024,    1290,
    1625,    2048,    2580,    3250,    4096,    5060,    6501,    8192,
    10321,   13003,   16384,   20642,   26007,   32768,   41285,   52015,
    65536,   82570,   104031,  131072,  165140,  208063,  262144,  330280,
    416127,  524287,  660561,  832255,  1048576, 1321122, 1664510, 2097152,
    2642245, 3329021, 4194304, 5284491, 6658042, 8388607, 10568983,
    13316085, 16777216};
const int firstIndex = 9;
const int lastIndex =
    static_cast<int>(sizeof(magicInts) / sizeof(magicInts[0]));

/// The number of bits that the numbers below `size` take.
int bitsOf(unsigned size) {
  unsigned number = 1;
  int bits = 0;
  while (size >= number && bits < 32) {
    ++bits;
    number <<= 1;
  }
  return bits;
}

/// The number of bits that three numbers below `sizes` take together, as
/// one number of mixed radix.
int bitsOf(const unsigned sizes[3]) {
  unsigned bytes[32];
  int count = 1;
  bytes[0] = 1;
  for (int i = 0; i != 3; ++i) {
    unsigned carry = 0;
    int b = 0;
    for (; b < count; ++b) {
      carry = bytes[b] * sizes[i] + carry;
      bytes[b] = carry & 0xff;
      carry >>= 8;
    }
    while (carry != 0) {
      bytes[b++] = carry & 0xff;
      carry >>= 8;
    }
    count = b;
  }
  int bits = 0;
  unsigned number = 1;
  --count;
  while (bytes[count] >= number) {
    ++bits;
    number *= 2;
  }
  return bits + count * 8;
}

/// Bits written from the most significant one.
class BitWriter {
public:
  void put(int bits, unsigned value) {
    while (bits >= 8) {
      last = (last << 8) | ((value >> (bits - 8)) & 0xff);
      bytes.push_back(static_cast<uint8_t>(last >> pending));
      bits -= 8;
    }
    if (bits > 0) {
      last = (last << bits) | (value & ((1u << bits) - 1));
      pending += bits;
      if (pending >= 8) {
        pending -= 8;
        bytes.push_back(static_cast<uint8_t>(last >> pending));
      }
    }
  }
  /// Three numbers below `sizes` as one number of mixed radix, in `bits`
  /// bits: the bytes from the least significant one.
  void putInts(int bits, const unsigned sizes[3], const unsigned values[3]) {
    unsigned digits[32];
    int count = 0;
    unsigned value = values[0];
    do {
      digits[count++] = value & 0xff;
      value >>= 8;
    } while (value != 0);
    for (int i = 1; i != 3; ++i) {
      unsigned carry = values[i];
      int b = 0;
      for (; b < count; ++b) {
        carry = digits[b] * sizes[i] + carry;
        digits[b] = carry & 0xff;
        carry >>= 8;
      }
      while (carry != 0) {
        digits[b++] = carry & 0xff;
        carry >>= 8;
      }
      count = b;
    }
    if (bits >= count * 8) {
      for (int b = 0; b != count; ++b)
        put(8, digits[b]);
      put(bits - count * 8, 0);
    } else {
      for (int b = 0; b != count - 1; ++b)
        put(8, digits[b]);
      put(bits - (count - 1) * 8, digits[count - 1]);
    }
  }
  /// The bytes, the last one padded with zeros.
  std::vector<uint8_t> finish() {
    if (pending > 0)
      bytes.push_back(static_cast<uint8_t>(last << (8 - pending)));
    pending = 0;
    return bytes;
  }

private:
  std::vector<uint8_t> bytes;
  unsigned last = 0;
  int pending = 0;
};

/// The reader of what BitWriter writes.
class BitReader {
public:
  BitReader(const uint8_t *data, size_t size) : data(data), size(size) {}
  bool overrun() const { return failed; }
  unsigned get(int bits) {
    unsigned mask = bits >= 32 ? ~0u : (1u << bits) - 1;
    unsigned value = 0;
    while (bits >= 8) {
      last = (last << 8) | next();
      value |= ((last >> pending) & 0xff) << (bits - 8);
      bits -= 8;
    }
    if (bits > 0) {
      if (pending < bits) {
        pending += 8;
        last = (last << 8) | next();
      }
      pending -= bits;
      value |= (last >> pending) & ((1u << bits) - 1);
    }
    return value & mask;
  }
  void getInts(int bits, const unsigned sizes[3], unsigned values[3]) {
    unsigned digits[32] = {0};
    int count = 0;
    while (bits > 8) {
      digits[count++] = get(8);
      bits -= 8;
    }
    if (bits > 0)
      digits[count++] = get(bits);
    for (int i = 2; i > 0; --i) {
      unsigned remainder = 0;
      for (int b = count - 1; b >= 0; --b) {
        remainder = (remainder << 8) | digits[b];
        digits[b] = remainder / sizes[i];
        remainder -= digits[b] * sizes[i];
      }
      values[i] = remainder;
    }
    values[0] = digits[0] | (digits[1] << 8) | (digits[2] << 16) |
                (digits[3] << 24);
  }

private:
  unsigned next() {
    if (position >= size) {
      failed = true;
      return 0;
    }
    return data[position++];
  }
  const uint8_t *data;
  size_t size;
  size_t position = 0;
  unsigned last = 0;
  int pending = 0;
  bool failed = false;
};

/// XDR: numbers big-endian, opaque data padded to 4 bytes.
void putInt(std::vector<uint8_t> &out, int32_t value) {
  uint32_t bits = static_cast<uint32_t>(value);
  for (int shift = 24; shift >= 0; shift -= 8)
    out.push_back(static_cast<uint8_t>(bits >> shift));
}
void putFloat(std::vector<uint8_t> &out, float value) {
  int32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  putInt(out, bits);
}
int32_t getInt(const uint8_t *p) {
  return static_cast<int32_t>((uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
                              (uint32_t(p[2]) << 8) | uint32_t(p[3]));
}
float getFloat(const uint8_t *p) {
  int32_t bits = getInt(p);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

} // namespace

std::vector<uint8_t> mdir::driver::compressXTC(const float *positions,
                                               size_t count,
                                               float precision) {
  std::vector<uint8_t> out;
  putInt(out, static_cast<int32_t>(count));
  if (count <= 9) {
    for (size_t i = 0; i != 3 * count; ++i)
      putFloat(out, positions[i]);
    return out;
  }
  putFloat(out, precision);

  // The integers, their bounds, and the least sum of the differences
  // between neighbors along the three axes.
  std::vector<int> ints(3 * count);
  int least[3] = {INT_MAX, INT_MAX, INT_MAX};
  int most[3] = {INT_MIN, INT_MIN, INT_MIN};
  long smallestDifference = LONG_MAX;
  for (size_t i = 0; i != count; ++i) {
    for (int k = 0; k != 3; ++k) {
      float scaled = positions[3 * i + k] * precision;
      int value = static_cast<int>(scaled >= 0.0f ? scaled + 0.5f
                                                  : scaled - 0.5f);
      ints[3 * i + k] = value;
      least[k] = std::min(least[k], value);
      most[k] = std::max(most[k], value);
    }
    if (i > 0) {
      long difference = 0;
      for (int k = 0; k != 3; ++k)
        difference += std::labs(static_cast<long>(ints[3 * i + k]) -
                                ints[3 * (i - 1) + k]);
      smallestDifference = std::min(smallestDifference, difference);
    }
  }
  for (int k = 0; k != 3; ++k)
    putInt(out, least[k]);
  for (int k = 0; k != 3; ++k)
    putInt(out, most[k]);

  unsigned sizes[3], bitsEach[3] = {0, 0, 0};
  bool large = false;
  for (int k = 0; k != 3; ++k) {
    sizes[k] = static_cast<unsigned>(most[k] - least[k]) + 1;
    large |= sizes[k] > 0xffffff;
  }
  int bits = 0;
  if (large)
    for (int k = 0; k != 3; ++k)
      bitsEach[k] = bitsOf(sizes[k]);
  else
    bits = bitsOf(sizes);

  int small = firstIndex;
  while (small < lastIndex - 1 && magicInts[small] < smallestDifference)
    ++small;
  putInt(out, small);
  int largest = std::min(lastIndex - 1, small + 8);
  int lowest = largest - 8;
  int smaller = magicInts[std::max(firstIndex, small - 1)] / 2;
  int smallNumber = magicInts[small] / 2;
  unsigned smallSizes[3];
  for (unsigned &size : smallSizes)
    size = magicInts[small];
  int larger = magicInts[largest] / 2;

  BitWriter writer;
  int previousRun = -1;
  int previous[3] = {0, 0, 0};
  size_t i = 0;
  auto close = [&](const int *a, const int *b, int bound) {
    for (int k = 0; k != 3; ++k)
      if (std::abs(a[k] - b[k]) >= bound)
        return false;
    return true;
  };
  while (i < count) {
    int *current = &ints[3 * i];
    // Whether the size of the small differences grows or shrinks.
    int change = 0;
    if (small < largest && i >= 1 && close(current, previous, larger))
      change = 1;
    else if (small > lowest)
      change = -1;
    bool isSmall = false;
    if (i + 1 < count && close(current, current + 3, smallNumber)) {
      // The first close particle goes first.
      for (int k = 0; k != 3; ++k)
        std::swap(current[k], current[3 + k]);
      isSmall = true;
    }
    unsigned shifted[3];
    for (int k = 0; k != 3; ++k)
      shifted[k] = static_cast<unsigned>(current[k] - least[k]);
    if (large)
      for (int k = 0; k != 3; ++k)
        writer.put(bitsEach[k], shifted[k]);
    else
      writer.putInts(bits, sizes, shifted);
    std::copy(current, current + 3, previous);
    ++i;

    unsigned run[24];
    int length = 0;
    if (!isSmall && change == -1)
      change = 0;
    while (isSmall && length < 24) {
      current = &ints[3 * i];
      if (change == -1) {
        long square = 0;
        for (int k = 0; k != 3; ++k)
          square += static_cast<long>(current[k] - previous[k]) *
                    (current[k] - previous[k]);
        if (square >= static_cast<long>(smaller) * smaller)
          change = 0;
      }
      for (int k = 0; k != 3; ++k)
        run[length++] =
            static_cast<unsigned>(current[k] - previous[k] + smallNumber);
      std::copy(current, current + 3, previous);
      ++i;
      isSmall = i < count && close(&ints[3 * i], previous, smallNumber);
    }
    if (length != previousRun || change != 0) {
      previousRun = length;
      writer.put(1, 1);
      writer.put(5, static_cast<unsigned>(length + change + 1));
    } else {
      writer.put(1, 0);
    }
    for (int k = 0; k < length; k += 3)
      writer.putInts(small, smallSizes, run + k);
    if (change != 0) {
      small += change;
      if (change < 0) {
        smallNumber = smaller;
        smaller = magicInts[small - 1] / 2;
      } else {
        smaller = smallNumber;
        smallNumber = magicInts[small] / 2;
      }
      for (unsigned &size : smallSizes)
        size = magicInts[small];
    }
  }
  std::vector<uint8_t> bytes = writer.finish();
  putInt(out, static_cast<int32_t>(bytes.size()));
  out.insert(out.end(), bytes.begin(), bytes.end());
  while (out.size() % 4 != 0)
    out.push_back(0);
  return out;
}

llvm::Error mdir::driver::decompressXTC(const uint8_t *data, size_t size,
                                        size_t count,
                                        std::vector<float> &positions) {
  auto fail = [](const char *message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "a frame of XTC %s", message);
  };
  if (size < 4 || static_cast<size_t>(getInt(data)) != count)
    return fail("holds another number of particles");
  positions.assign(3 * count, 0.0f);
  if (count <= 9) {
    if (size < 4 + 12 * count)
      return fail("is cut short");
    for (size_t i = 0; i != 3 * count; ++i)
      positions[i] = getFloat(data + 4 + 4 * i);
    return llvm::Error::success();
  }
  if (size < 40)
    return fail("is cut short");
  float precision = getFloat(data + 4);
  int least[3], most[3];
  for (int k = 0; k != 3; ++k) {
    least[k] = getInt(data + 8 + 4 * k);
    most[k] = getInt(data + 20 + 4 * k);
  }
  int small = getInt(data + 32);
  size_t length = static_cast<size_t>(getInt(data + 36));
  if (small < firstIndex || small >= lastIndex || 40 + length > size)
    return fail("is not valid");
  unsigned sizes[3], bitsEach[3] = {0, 0, 0};
  bool large = false;
  for (int k = 0; k != 3; ++k) {
    sizes[k] = static_cast<unsigned>(most[k] - least[k]) + 1;
    large |= sizes[k] > 0xffffff;
  }
  int bits = 0;
  if (large)
    for (int k = 0; k != 3; ++k)
      bitsEach[k] = bitsOf(sizes[k]);
  else
    bits = bitsOf(sizes);
  int smaller = magicInts[std::max(firstIndex, small - 1)] / 2;
  int smallNumber = magicInts[small] / 2;
  unsigned smallSizes[3];
  for (unsigned &s : smallSizes)
    s = magicInts[small];

  BitReader reader(data + 40, length);
  float inverse = 1.0f / precision;
  size_t i = 0, written = 0;
  int run = 0;
  auto store = [&](const int *value) {
    for (int k = 0; k != 3; ++k)
      positions[3 * written + k] = static_cast<float>(value[k]) * inverse;
    ++written;
  };
  while (i < count) {
    int current[3];
    unsigned shifted[3];
    if (large)
      for (int k = 0; k != 3; ++k)
        shifted[k] = reader.get(bitsEach[k]);
    else
      reader.getInts(bits, sizes, shifted);
    for (int k = 0; k != 3; ++k)
      current[k] = static_cast<int>(shifted[k]) + least[k];
    ++i;
    int previous[3] = {current[0], current[1], current[2]};
    int change = 0;
    if (reader.get(1) == 1) {
      run = static_cast<int>(reader.get(5));
      change = run % 3;
      run -= change;
      --change;
    }
    if (run > 0) {
      if (written + run / 3 + 1 > count)
        return fail("is not valid");
      for (int k = 0; k < run; k += 3) {
        unsigned value[3];
        reader.getInts(small, smallSizes, value);
        ++i;
        int next[3];
        for (int c = 0; c != 3; ++c)
          next[c] = static_cast<int>(value[c]) + previous[c] - smallNumber;
        if (k == 0) {
          // The first close particle was written first.
          store(next);
          store(previous);
        } else {
          store(next);
        }
        std::copy(next, next + 3, previous);
      }
    } else {
      store(current);
    }
    small += change;
    if (change < 0) {
      smallNumber = smaller;
      smaller = small > firstIndex ? magicInts[small - 1] / 2 : 0;
    } else if (change > 0) {
      smaller = smallNumber;
      smallNumber = magicInts[small] / 2;
    }
    for (unsigned &s : smallSizes)
      s = magicInts[small];
    if (reader.overrun() || small < firstIndex || small >= lastIndex)
      return fail("is not valid");
  }
  if (written != count)
    return fail("is not valid");
  return llvm::Error::success();
}

XTCWriter::~XTCWriter() { close(); }

llvm::Error XTCWriter::open(const std::string &path, size_t numParticles,
                            int64_t first, int64_t period, double timestep,
                            const double box[3]) {
  file = std::fopen(path.c_str(), "wb");
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write '%s'", path.c_str());
  this->numParticles = numParticles;
  this->first = first;
  this->period = period;
  this->timestep = timestep;
  for (int i = 0; i != 3; ++i)
    this->box[i] = box[i];
  return llvm::Error::success();
}

llvm::Expected<int64_t> XTCWriter::append(const std::string &path,
                                          size_t numParticles,
                                          int64_t frames, int64_t period,
                                          double timestep,
                                          const double box[3]) {
  auto fail = [&](const llvm::Twine &message) -> llvm::Error {
    close();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "'" + path + "' " + message);
  };
  file = std::fopen(path.c_str(), "r+b");
  if (!file)
    return fail("cannot be opened to continue it");
  // The frames, by their headers: 56 bytes up to the second count of the
  // particles, then the positions, whose size the frame gives.
  long kept = 0;
  int64_t held = 0;
  int64_t lastStep = 0;
  std::fseek(file, 0, SEEK_END);
  long length = std::ftell(file);
  long start = 0;
  while (start + 56 <= length) {
    uint8_t head[56];
    std::fseek(file, start, SEEK_SET);
    if (std::fread(head, 1, sizeof(head), file) != sizeof(head))
      break;
    if (getInt(head) != 1995)
      return fail("is not a trajectory in XTC");
    if (static_cast<size_t>(getInt(head + 4)) != numParticles ||
        static_cast<size_t>(getInt(head + 52)) != numParticles)
      return fail("holds " + llvm::Twine(getInt(head + 4)) +
                  " particles, and the run " + llvm::Twine(numParticles));
    int64_t step = getInt(head + 8);
    if (held > 0 && step - lastStep != period)
      return fail("has a frame every " + llvm::Twine(step - lastStep) +
                  " steps, and the run writes one every " +
                  llvm::Twine(period));
    lastStep = step;
    long body = static_cast<long>(12 * numParticles);
    if (numParticles > 9) {
      uint8_t rest[36];
      if (std::fread(rest, 1, sizeof(rest), file) != sizeof(rest))
        break;
      long bytes = getInt(rest + 32);
      body = 36 + (bytes + 3) / 4 * 4;
    }
    long end = start + 56 + body;
    if (end > length)
      break;
    start = end;
    ++held;
    if (held == frames)
      kept = end;
  }
  if (held < frames)
    return fail("holds " + llvm::Twine(held) + " frames, fewer than the " +
                llvm::Twine(frames) + " that the checkpoint counts");
  std::fflush(file);
  if (kept != length && ::ftruncate(::fileno(file), kept) != 0)
    return fail("cannot be cut to the frames that the checkpoint counts");
  std::fseek(file, kept, SEEK_SET);
  this->numParticles = numParticles;
  this->period = period;
  this->timestep = timestep;
  for (int i = 0; i != 3; ++i)
    this->box[i] = box[i];
  numFrames = static_cast<int32_t>(frames);
  return held - frames;
}

void XTCWriter::writeFrame(const float *positions, int64_t step,
                           double time) {
  std::vector<uint8_t> out;
  putInt(out, 1995);
  putInt(out, static_cast<int32_t>(numParticles));
  putInt(out, static_cast<int32_t>(step));
  putFloat(out, static_cast<float>(time));
  // The vectors of the cell in nm: a = (a_x, 0, 0), b = (b_x, b_y, 0),
  // c = (c_x, c_y, c_z).
  double vectors[9] = {box[0],  0.0,     0.0,    tilt[0], box[1],
                       0.0,     tilt[1], tilt[2], box[2]};
  for (double value : vectors)
    putFloat(out, static_cast<float>(value * 0.1));
  std::vector<float> nm(3 * numParticles);
  for (size_t i = 0; i != nm.size(); ++i)
    nm[i] = positions[i] * 0.1f;
  std::vector<uint8_t> body = compressXTC(nm.data(), numParticles, 1000.0f);
  out.insert(out.end(), body.begin(), body.end());
  std::fwrite(out.data(), 1, out.size(), file);
  std::fflush(file);
  ++numFrames;
}

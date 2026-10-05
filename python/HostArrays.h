// Host interchange always validates before copying into native ownership.
#ifndef MDIR_PYTHON_HOST_ARRAYS_H
#define MDIR_PYTHON_HOST_ARRAYS_H
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
namespace host {
inline std::string shape(const py::array &a) {
  std::string result = "(";
  for (py::ssize_t i = 0; i < a.ndim(); ++i) {
    if (i) result += ", ";
    result += std::to_string(a.shape(i));
  }
  if (a.ndim() == 1) result += ",";
  return result + ")";
}
inline py::array array(py::handle source, const std::string &name) {
  py::array a;
  try {
    if (PyObject_CheckBuffer(source.ptr())) {
      a = py::array::ensure(source, 0);
    } else if (py::hasattr(source, "__dlpack__") &&
               py::hasattr(source, "__dlpack_device__")) {
      auto device = py::cast<std::pair<int, int>>(source.attr("__dlpack_device__")());
      if (device.first != 1)
        throw InputError(name + ": host input requires a CPU DLPack device; found " +
                         std::to_string(device.first));
      a = py::cast<py::array>(py::module_::import("numpy").attr("from_dlpack")(source));
    } else {
      throw InputError(name + ": expected a buffer-protocol or CPU DLPack array; lists are not accepted");
    }
  } catch (const py::error_already_set &e) {
    throw InputError(name + ": cannot import host array: " + e.what());
  } catch (const py::cast_error &e) {
    throw InputError(name + ": invalid array protocol: " + e.what());
  }
  if (!a) throw InputError(name + ": cannot import buffer as a NumPy array");
  return a;
}
template <class T> bool dtype(const py::array &a) {
  return py::cast<bool>(a.dtype().attr("__eq__")(py::dtype::of<T>()));
}
inline void dimensions(const py::array &a, const std::string &name,
                       std::optional<size_t> rows, std::optional<size_t> columns) {
  if (a.ndim() != (columns ? 2 : 1) ||
      (columns && a.shape(1) != static_cast<py::ssize_t>(*columns)) ||
      (rows && a.shape(0) != static_cast<py::ssize_t>(*rows))) {
    std::string expected = "(" + (rows ? std::to_string(*rows) : "N");
    expected += columns ? ", " + std::to_string(*columns) + ")" : ",)";
    throw InputError(name + ": expected shape " + expected + "; found " + shape(a));
  }
}
inline void contiguous(const py::array &a, const std::string &name) {
  if (!(a.flags() & py::array::c_style))
    throw InputError(name + ": expected C-contiguous storage; found shape " + shape(a));
}
inline std::vector<double> doubles(py::handle source, const std::string &name,
                                    std::optional<size_t> rows = {},
                                    std::optional<size_t> columns = {}) {
  auto a = array(source, name);
  dimensions(a, name, rows, columns);
  if (!dtype<double>(a))
    throw InputError(name + ": expected native float64; found " +
                     py::str(a.dtype()).cast<std::string>() + " with shape " + shape(a));
  contiguous(a, name);
  std::vector<double> values(a.size());
  if (!values.empty()) std::memcpy(values.data(), a.data(), values.size() * sizeof(double));
  for (double v : values)
    if (!std::isfinite(v))
      throw InputError(name + ": nonfinite value in shape " + shape(a));
  return values;
}
template <class T>
py::array_t<T> copy(const T *values, size_t count, std::vector<py::ssize_t> dimensions) {
  py::array_t<T> result(dimensions);
  if (count) std::memcpy(result.mutable_data(), values, count * sizeof(T));
  result.attr("setflags")(py::arg("write") = false);
  return result;
}
inline size_t tupleCount(const driver::TupleTerm &term) {
  if (!term.arity || term.particles.size() % term.arity)
    throw InputError("TupleTerm.particles: arity must be positive and divide the native particle count");
  return term.particles.size() / term.arity;
}
inline py::array_t<int64_t> particles(const driver::TupleTerm &term) {
  size_t rows = tupleCount(term);
  std::vector<int64_t> values(term.particles.begin(), term.particles.end());
  return copy(values.data(), values.size(), {static_cast<py::ssize_t>(rows), term.arity});
}
inline void particles(driver::TupleTerm &term, py::handle source) {
  if (!term.arity) throw InputError("TupleTerm.particles: arity must be positive");
  const std::string name = "TupleTerm.particles";
  auto a = array(source, name);
  dimensions(a, name, {}, term.arity);
  bool i32 = dtype<int32_t>(a), i64 = dtype<int64_t>(a);
  if (!i32 && !i64)
    throw InputError(name + ": expected native int32 or int64; found " +
                     py::str(a.dtype()).cast<std::string>() + " with shape " + shape(a));
  contiguous(a, name);
  std::vector<unsigned> values(a.size());
  for (size_t i = 0; i < values.size(); ++i) {
    int64_t value;
    // Buffers need not be aligned; copy scalars before reading them.
    if (i32) {
      int32_t v; std::memcpy(&v, static_cast<const char *>(a.data()) + i * sizeof(v), sizeof(v)); value = v;
    } else std::memcpy(&value, static_cast<const char *>(a.data()) + i * sizeof(value), sizeof(value));
    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<unsigned>::max())
      throw InputError(name + ": particle ID is negative or exceeds the native range in shape " + shape(a));
    values[i] = static_cast<unsigned>(value);
  }
  for (const auto &parameter : term.parameters)
    if (parameter.second.size() != static_cast<size_t>(a.shape(0)))
      throw InputError(name + ": row count disagrees with parameter '" + parameter.first + "'");
  term.particles = std::move(values);
}
inline py::list parameters(const driver::TupleTerm &term) {
  py::list result;
  for (const auto &p : term.parameters)
    result.append(py::make_tuple(p.first, copy(p.second.data(), p.second.size(),
                          {static_cast<py::ssize_t>(p.second.size())})));
  return result;
}
inline void parameters(driver::TupleTerm &term, py::sequence source) {
  std::vector<std::pair<std::string, std::vector<double>>> values;
  for (py::handle entry : source) {
    if (!py::isinstance<py::sequence>(entry) || py::len(entry) != 2)
      throw InputError("TupleTerm.parameters: expected ordered name/array pairs");
    auto pair = py::reinterpret_borrow<py::sequence>(entry);
    if (!py::isinstance<py::str>(pair[0]))
      throw InputError("TupleTerm.parameters: parameter names must be strings");
    auto name = py::cast<std::string>(pair[0]);
    auto count = term.particles.empty() ? std::optional<size_t>{} : tupleCount(term);
    values.emplace_back(name, doubles(pair[1], "TupleTerm.parameters['" + name + "']", count));
  }
  term.parameters = std::move(values);
}
} // namespace host
#endif

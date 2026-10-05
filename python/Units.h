// Unit quantities at the Python boundary (D200).
//
// A value with `value_in_unit_system` is taken as an `openmm.unit.Quantity`
// (duck typing: OpenMM is not a dependency). It is converted to the unit
// that its setter declares, by `value_in_unit`; `openmm.unit` is imported
// only then. A plain value keeps its meaning in the public units (D191).
// The declared units are those of `md_unit_system` but for pressure, which
// is bar here and a unit derived from amu, nm, ps, and the mole there.
#ifndef MDIR_PYTHON_UNITS_H
#define MDIR_PYTHON_UNITS_H
#include <pybind11/eval.h>
namespace units {
/// The unit a setter takes: an expression in the names of `openmm.unit`,
/// and the text that errors name. No expression: plain values only.
struct Unit { const char *expression = nullptr, *text = nullptr; };
inline constexpr Unit none{};
inline constexpr Unit nm{"nanometer", "nm"};
inline constexpr Unit inverseNm{"nanometer**-1", "1/nm"};
inline constexpr Unit nmPerPs{"nanometer/picosecond", "nm/ps"};
inline constexpr Unit ps{"picosecond", "ps"};
inline constexpr Unit kelvin{"kelvin", "K"};
inline constexpr Unit bar{"bar", "bar"};
inline constexpr Unit inverseBar{"bar**-1", "1/bar"};
inline constexpr Unit springConstant{"kilojoule_per_mole/nanometer**2", "kJ/mol/nm^2"};
inline constexpr Unit second{"second", "s"};

inline bool isQuantity(py::handle value) {
  return py::hasattr(value, "value_in_unit_system");
}
/// The magnitude of a quantity in `unit`, or an InputError naming `name`.
inline py::object strip(py::handle value, const std::string &name, Unit unit) {
  if (!unit.expression)
    throw InputError(name + ": takes plain values only, since it has no declared unit; "
                     "found a unit quantity");
  py::object target;
  try {
    auto module = py::module_::import("openmm.unit");
    target = py::eval(unit.expression, module.attr("__dict__"));
  } catch (const py::error_already_set &e) {
    throw InputError(name + ": a unit quantity needs openmm.unit: " +
                     py::str(e.value()).cast<std::string>());
  }
  try {
    return value.attr("value_in_unit")(target);
  } catch (const py::error_already_set &e) {
    throw InputError(name + ": expected a quantity in " + unit.text + "; " +
                     py::str(e.value()).cast<std::string>());
  }
}
/// A real number: a plain one as it is, a quantity in `unit`.
inline double scalar(py::handle value, const std::string &name, Unit unit) {
  if (!isQuantity(value)) {
    try {
      return py::cast<double>(value);
    } catch (const py::cast_error &) {
      throw py::type_error(name + ": expected a real number; found " +
                           py::str(py::type::of(value)).cast<std::string>());
    }
  }
  py::object magnitude = strip(value, name, unit);
  auto numpy = py::module_::import("numpy");
  bool real = !py::isinstance<py::bool_>(magnitude) &&
              (PyFloat_Check(magnitude.ptr()) || PyLong_Check(magnitude.ptr()) ||
               py::isinstance(magnitude, numpy.attr("floating")) ||
               py::isinstance(magnitude, numpy.attr("integer")));
  if (!real)
    throw InputError(name + ": expected a real number in " + unit.text +
                     "; found a quantity of " +
                     py::str(py::type::of(magnitude)).cast<std::string>());
  return py::cast<double>(magnitude);
}
} // namespace units
#endif

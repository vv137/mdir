// Tunable parameters at the Python boundary (D213,
// docs/python-tunable.md): their declarations on a System, what maps are
// built from, and the values of a simulation's tunables as a mapping of
// names to arrays.
#ifndef MDIR_PYTHON_TUNABLES_H
#define MDIR_PYTHON_TUNABLES_H
#include <algorithm>
namespace tunables {
/// The unit of the values of a tunable of `parameter`: a constant or a
/// parameter of an expression has those of its expression, plain numbers.
inline units::Unit unitOf(const std::string &parameter, const std::string &term) {
  if (!term.empty()) return units::none;
  if (parameter == "charge") return units::charge;
  if (parameter == "sigma") return units::nm;
  if (parameter == "epsilon") return units::energy;
  return units::none;
}
inline py::array_t<int64_t> indices(const std::vector<int64_t> &values) {
  return host::copy(values.data(), values.size(), {static_cast<py::ssize_t>(values.size())});
}
/// A one-dimensional array of integers, int32 or int64.
inline std::vector<int64_t> integers(py::handle source, const std::string &name) {
  auto a = host::array(source, name);
  host::dimensions(a, name, {}, {});
  bool i32 = host::dtype<int32_t>(a), i64 = host::dtype<int64_t>(a);
  if (!i32 && !i64)
    throw InputError(name + ": expected native int32 or int64; found " +
                     py::str(a.dtype()).cast<std::string>() + " with shape " + host::shape(a));
  host::contiguous(a, name);
  std::vector<int64_t> values(a.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (i32) {
      int32_t v;
      std::memcpy(&v, static_cast<const char *>(a.data()) + i * sizeof(v), sizeof(v));
      values[i] = v;
    } else {
      std::memcpy(&values[i], static_cast<const char *>(a.data()) + i * sizeof(int64_t),
                  sizeof(int64_t));
    }
  }
  return values;
}
inline void bindTunable(py::module_ &m) {
  py::class_<model::Tunable>(m, "Tunable")
    .def(py::init([](std::string name, std::string parameter, std::optional<std::string> term,
                     py::object map, py::object values, std::optional<std::string> mixing) {
      model::Tunable t;
      t.name = std::move(name);
      t.parameter = std::move(parameter);
      t.term = term.value_or("");
      std::string where = "Tunable('" + t.name + "')";
      if (!map.is_none()) t.map = integers(map, where + ".map");
      if (!values.is_none())
        t.values = host::doubles(values, where + ".values", {}, {}, unitOf(t.parameter, t.term));
      if (mixing) {
        if (*mixing == "arithmetic") t.mixing = driver::Mixing::Arithmetic;
        else if (*mixing == "geometric") t.mixing = driver::Mixing::Geometric;
        else throw InputError(where + ": mixing is \"arithmetic\" or \"geometric\"");
        if (t.parameter != "sigma" || !t.term.empty())
          throw InputError(where + ": a mixing rule is for \"sigma\" only");
      }
      return t;
    }), py::arg("name"), py::arg("parameter"), py::arg("term") = py::none(),
        py::arg("map") = py::none(), py::arg("values") = py::none(),
        py::arg("mixing") = py::none())
    .def_property_readonly("name", [](const model::Tunable &t) { return t.name; })
    .def_property_readonly("parameter", [](const model::Tunable &t) { return t.parameter; })
    .def_property_readonly("term", [](const model::Tunable &t) -> py::object {
      if (t.term.empty()) return py::none();
      return py::str(t.term);
    })
    .def_property_readonly("map", [](const model::Tunable &t) -> py::object {
      if (t.map.empty()) return py::none();
      return indices(t.map);
    })
    .def_property_readonly("values", [](const model::Tunable &t) -> py::object {
      if (t.values.empty()) return py::none();
      return host::copy(t.values.data(), t.values.size(),
                        {static_cast<py::ssize_t>(t.values.size())});
    })
    .def_property_readonly("mixing", [](const model::Tunable &t) {
      return t.mixing == driver::Mixing::Geometric ? "geometric" : "arithmetic";
    })
    .def("__repr__", [](const model::Tunable &t) {
      return "Tunable('" + t.name + "', '" + t.parameter + "'" +
             (t.term.empty() ? "" : ", term='" + t.term + "'") + ")";
    });
}
/// What maps are built from: read-only copies of the topology's values.
template <class SystemClass> void bindModelArrays(SystemClass &system) {
  using Owner = typename SystemClass::type;
  system.def_property_readonly("charges", [](const Owner &o) {
    const auto &v = o.value.topology.charges;
    return host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(v.size())});
  });
  system.def_property_readonly("particle_types", [](const Owner &o) {
    std::vector<int64_t> v(o.value.topology.types.begin(), o.value.topology.types.end());
    return indices(v);
  });
  system.def_property_readonly("type_names", [](const Owner &o) { return o.value.topology.typeNames; });
  system.def_property_readonly("atom_names", [](const Owner &o) { return o.value.topology.atomNames; });
  system.def_property_readonly("residue_indices", [](const Owner &o) {
    const auto &t = o.value.topology;
    std::vector<int64_t> v(t.getNumParticles(), 0);
    for (size_t i = 0; i < v.size(); ++i) {
      auto next = std::upper_bound(t.residueStarts.begin(), t.residueStarts.end(), i);
      v[i] = static_cast<int64_t>(next - t.residueStarts.begin()) - 1;
    }
    return indices(v);
  });
  system.def_property_readonly("residue_names", [](const Owner &o) {
    const auto &t = o.value.topology;
    std::vector<std::string> names(t.getNumParticles());
    for (size_t i = 0; i < names.size(); ++i) {
      auto next = std::upper_bound(t.residueStarts.begin(), t.residueStarts.end(), i);
      size_t r = static_cast<size_t>(next - t.residueStarts.begin());
      names[i] = r > 0 && r - 1 < t.residueNames.size() ? t.residueNames[r - 1] : "";
    }
    return names;
  });
}
/// The declarations of a prepared model, for `Program.plan`.
inline py::list describe(const model::TunableSet &set) {
  py::list result;
  for (const auto &entry : set.tunables) {
    py::dict d;
    d["name"] = entry.name;
    d["parameter"] = entry.parameter;
    d["term"] = entry.term.empty() ? py::object(py::none()) : py::object(py::str(entry.term));
    d["sites"] = entry.map.size();
    d["entries"] = entry.entries;
    d["map"] = indices(entry.map);
    d["unit"] = entry.unit;
    if (entry.kind == model::TunableSet::Entry::Sigma)
      d["mixing"] = entry.mixing == driver::Mixing::Geometric ? "geometric" : "arithmetic";
    result.append(d);
  }
  return result;
}
} // namespace tunables
#endif

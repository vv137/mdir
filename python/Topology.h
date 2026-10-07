// A read-only view of a topology at the Python boundary
// (D[python-topology], docs/python-topology.md): copies of its atoms,
// residues, and bonded tuples in input order, the constraints that
// preparation resolved, and the Amber masks of the control file.
#ifndef MDIR_PYTHON_TOPOLOGY_H
#define MDIR_PYTHON_TOPOLOGY_H
#include "mdir/Driver/Selection.h"
namespace topology {
/// A copy of a topology taken when the view was made, so that later changes
/// to its owner do not reach it. Positions and velocities belong to the
/// initial state and are left out.
struct View {
  std::shared_ptr<const driver::Topology> topology;
  /// Whether the view is of a prepared program, whose constraints are
  /// resolved.
  bool prepared = false;
};
inline View of(const driver::Topology &source, bool prepared) {
  auto copy = std::make_shared<driver::Topology>(source);
  copy->positions.clear(); copy->positions.shrink_to_fit();
  copy->velocities.clear(); copy->velocities.shrink_to_fit();
  return View{std::move(copy), prepared};
}
template <class T> py::array_t<int64_t> column(const std::vector<T> &values) {
  std::vector<int64_t> v(values.begin(), values.end());
  return host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(v.size())});
}
/// An (n, arity) int64 array of the particles of each row of `rows`.
template <class Row, class Get>
py::array_t<int64_t> tuples(const std::vector<Row> &rows, py::ssize_t arity, Get get) {
  std::vector<int64_t> v;
  v.reserve(rows.size() * arity);
  for (const Row &row : rows) get(row, v);
  return host::copy(v.data(), v.size(), {static_cast<py::ssize_t>(rows.size()), arity});
}
inline void bind(py::module_ &m) {
  using driver::Topology;
  auto t = [](const View &v) -> const Topology & { return *v.topology; };
  py::class_<View>(m, "Topology")
    .def_property_readonly("particle_count", [t](const View &v) { return t(v).getNumParticles(); })
    .def_property_readonly("residue_count", [t](const View &v) { return t(v).residueStarts.size(); })
    .def_property_readonly("prepared", [](const View &v) { return v.prepared; })
    // Atoms.
    .def_property_readonly("atom_names", [t](const View &v) { return t(v).atomNames; })
    .def_property_readonly("atomic_numbers", [t](const View &v) { return column(t(v).atomicNumbers); })
    .def_property_readonly("masses", [t](const View &v) {
      const auto &m = t(v).masses;
      return host::copy(m.data(), m.size(), {static_cast<py::ssize_t>(m.size())});
    })
    .def_property_readonly("charges", [t](const View &v) {
      const auto &q = t(v).charges;
      return host::copy(q.data(), q.size(), {static_cast<py::ssize_t>(q.size())});
    })
    .def_property_readonly("particle_types", [t](const View &v) { return column(t(v).types); })
    .def_property_readonly("type_names", [t](const View &v) { return t(v).typeNames; })
    .def_property_readonly("residue_indices", [t](const View &v) { return column(t(v).residueOf); })
    // Residues.
    .def_property_readonly("residue_names", [t](const View &v) { return t(v).residueNames; })
    .def_property_readonly("residue_starts", [t](const View &v) { return column(t(v).residueStarts); })
    // Bonded tuples, one row for each term of the topology.
    .def_property_readonly("bonds", [t](const View &v) {
      return tuples(t(v).bonds, 2, [](const Topology::Bond &b, auto &out) {
        out.push_back(b.i); out.push_back(b.j);
      });
    })
    .def_property_readonly("angles", [t](const View &v) {
      return tuples(t(v).angles, 3, [](const Topology::Angle &a, auto &out) {
        out.push_back(a.i); out.push_back(a.j); out.push_back(a.k);
      });
    })
    .def_property_readonly("dihedrals", [t](const View &v) {
      return tuples(t(v).dihedrals, 4, [](const Topology::Dihedral &d, auto &out) {
        out.push_back(d.i); out.push_back(d.j); out.push_back(d.k); out.push_back(d.l);
      });
    })
    .def_property_readonly("improper_dihedrals", [t](const View &v) {
      std::vector<bool> flags;
      for (const auto &d : t(v).dihedrals) flags.push_back(d.improper);
      std::vector<uint8_t> bytes(flags.begin(), flags.end());
      py::array_t<bool> result({static_cast<py::ssize_t>(bytes.size())});
      if (!bytes.empty()) std::memcpy(result.mutable_data(), bytes.data(), bytes.size());
      result.attr("setflags")(py::arg("write") = false);
      return result;
    })
    .def_property_readonly("harmonic_impropers", [t](const View &v) {
      return tuples(t(v).harmonicImpropers, 4, [](const Topology::HarmonicImproper &d, auto &out) {
        out.push_back(d.i); out.push_back(d.j); out.push_back(d.k); out.push_back(d.l);
      });
    })
    // The constraints that preparation resolved: none (None) on a view that
    // was not prepared, whose constraints depend on the System's settings.
    .def_property_readonly("constraints", [t](const View &v) -> py::object {
      if (!v.prepared) return py::none();
      std::vector<int64_t> pairs;
      for (const auto &shake : t(v).shakes)
        for (unsigned h : shake.hydrogens) { pairs.push_back(shake.center); pairs.push_back(h); }
      return host::copy(pairs.data(), pairs.size(), {static_cast<py::ssize_t>(pairs.size() / 2), 2});
    })
    .def_property_readonly("rigid_waters", [t](const View &v) -> py::object {
      if (!v.prepared) return py::none();
      return tuples(t(v).settles, 3, [](const Topology::Settle &s, auto &out) {
        out.push_back(s.oxygen); out.push_back(s.oxygen + 1); out.push_back(s.oxygen + 2);
      });
    })
    .def_property_readonly("virtual_sites", [t](const View &v) {
      return tuples(t(v).virtualSites, 4, [](const Topology::VirtualSite &s, auto &out) {
        out.push_back(s.site); out.push_back(s.i); out.push_back(s.j); out.push_back(s.k);
      });
    })
    // The particles that an Amber mask selects, by the parser of the control
    // file and mdir.Restraint (lib/Driver/Selection.cpp).
    .def("select", [t](const View &v, const std::string &mask) {
      auto flags = unwrap(driver::selectParticles(mask, t(v)));
      std::vector<int64_t> chosen;
      for (size_t i = 0; i != flags.size(); ++i)
        if (flags[i]) chosen.push_back(static_cast<int64_t>(i));
      return host::copy(chosen.data(), chosen.size(), {static_cast<py::ssize_t>(chosen.size())});
    }, py::arg("mask"))
    .def("__repr__", [t](const View &v) {
      return "Topology(" + std::to_string(t(v).getNumParticles()) + " particles, " +
             std::to_string(t(v).residueStarts.size()) + " residues, " +
             std::to_string(t(v).bonds.size()) + " bonds" + (v.prepared ? ", prepared)" : ")");
    });
}
} // namespace topology
#endif

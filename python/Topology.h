// A read-only view of a topology at the Python boundary
// (D221, docs/python-topology.md): copies of its atoms,
// residues, and bonded tuples in input order, the constraints that
// preparation resolved, and the Amber masks of the control file.
#ifndef MDIR_PYTHON_TOPOLOGY_H
#define MDIR_PYTHON_TOPOLOGY_H
#include "mdir/Driver/Selection.h"
#include <set>
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
/// The bonds of an OpenMM topology: those of the potential without the H–H
/// "bond" that Amber lists in a three-site water, the O–H bonds of the
/// waters that a GROMACS topology gives to SETTLE, and a bond from each
/// virtual site to the particle that places it first, as OpenMM's readers
/// give them.
inline std::vector<std::pair<unsigned, unsigned>> chemicalBonds(const driver::Topology &t) {
  std::set<std::pair<unsigned, unsigned>> seen;
  std::vector<std::pair<unsigned, unsigned>> result;
  auto add = [&](unsigned i, unsigned j) {
    if (seen.insert({std::min(i, j), std::max(i, j)}).second) result.push_back({i, j});
  };
  // Hydrogens bonded to an oxygen in their residue, for the water test.
  std::vector<int> oxygenOf(t.getNumParticles(), -1);
  for (const auto &b : t.bonds)
    for (auto [h, o] : {std::pair{b.i, b.j}, std::pair{b.j, b.i}})
      if (t.atomicNumbers[h] == 1 && t.atomicNumbers[o] == 8 && t.residueOf[h] == t.residueOf[o])
        oxygenOf[h] = static_cast<int>(o);
  for (const auto &b : t.bonds) {
    bool water = t.atomicNumbers[b.i] == 1 && t.atomicNumbers[b.j] == 1 &&
                 oxygenOf[b.i] >= 0 && oxygenOf[b.i] == oxygenOf[b.j];
    if (!water) add(b.i, b.j);
  }
  for (const auto &s : t.settles) { add(s.oxygen, s.oxygen + 1); add(s.oxygen, s.oxygen + 2); }
  for (const auto &v : t.virtualSites) add(v.i, v.site);
  return result;
}
/// An `openmm.app.Topology` of the view, OpenMM imported only here (no
/// dependency, as D200): names as in the file, elements from the atomic
/// numbers (none for 0 or less), the residues, one chain for each run of
/// residues that no bond joins to the residues around it, and the cell.
inline py::object toOpenMM(const driver::Topology &t, std::optional<driver::Cell> cell) {
  py::module_ app, unit, openmm;
  try {
    openmm = py::module_::import("openmm");
    app = py::module_::import("openmm.app");
    unit = py::module_::import("openmm.unit");
  } catch (py::error_already_set &e) {
    throw py::import_error(std::string("Topology.to_openmm needs OpenMM: ") + e.what());
  }
  auto bonds = chemicalBonds(t);
  size_t residues = t.residueStarts.size();
  // reach[r]: the last residue that a bond joins to residue r or before.
  std::vector<unsigned> reach(residues);
  for (size_t r = 0; r != residues; ++r) reach[r] = static_cast<unsigned>(r);
  for (auto [i, j] : bonds) {
    unsigned a = std::min(t.residueOf[i], t.residueOf[j]), b = std::max(t.residueOf[i], t.residueOf[j]);
    reach[a] = std::max(reach[a], b);
  }
  for (size_t r = 1; r < residues; ++r) reach[r] = std::max(reach[r], reach[r - 1]);
  py::object top = app.attr("Topology")();
  py::object Element = app.attr("element").attr("Element");
  py::list atoms;
  py::object chain;
  for (size_t r = 0; r != residues; ++r) {
    if (r == 0 || reach[r - 1] < r) chain = top.attr("addChain")();
    py::object residue = top.attr("addResidue")(t.residueNames[r], chain);
    size_t end = r + 1 < residues ? t.residueStarts[r + 1] : t.getNumParticles();
    for (size_t i = t.residueStarts[r]; i != end; ++i) {
      py::object element = py::none();
      if (t.atomicNumbers[i] > 0) element = Element.attr("getByAtomicNumber")(t.atomicNumbers[i]);
      atoms.append(top.attr("addAtom")(t.atomNames[i], element, residue));
    }
  }
  for (auto [i, j] : bonds) top.attr("addBond")(atoms[i], atoms[j]);
  if (cell) {
    auto vectors = cell->getVectors();
    py::list rows;
    for (const auto &row : vectors) rows.append(openmm.attr("Vec3")(row[0], row[1], row[2]));
    top.attr("setPeriodicBoxVectors")(unit.attr("Quantity")(rows, unit.attr("nanometer")));
  }
  return top;
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
    .def("to_openmm", [t](const View &v, std::optional<driver::Cell> cell) {
      return toOpenMM(t(v), cell);
    }, py::arg("cell") = py::none())
    .def("__repr__", [t](const View &v) {
      return "Topology(" + std::to_string(t(v).getNumParticles()) + " particles, " +
             std::to_string(t(v).residueStarts.size()) + " residues, " +
             std::to_string(t(v).bonds.size()) + " bonds" + (v.prepared ? ", prepared)" : ")");
    });
}
} // namespace topology
#endif

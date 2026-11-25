### Changed

- D[distribute-steady] expands distributed CPU phase timing, sizes owned fields by domain occupancy, reuses ordered halo schedules, and preserves displacement precision in mixed midpoint evaluation.
- Constrained CPU force evaluations reuse MATRIX candidates within the pairlist skin. Distributed midpoint diagnostics report the interaction cell and its required ghost rows.

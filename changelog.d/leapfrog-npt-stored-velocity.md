### Fixed

- Leapfrog with a barostat projects its stored backward half-kick through
  the constraints, preventing constraint-normal force components from
  changing written H5MD and checkpoint velocities (D393).
